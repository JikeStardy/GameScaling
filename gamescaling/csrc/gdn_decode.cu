// Optional inference-only SM70 kernel for HF Qwen3.5's recurrent delta update.
// q/k normalization is intentionally performed by the exact Torch reference
// operations in Python. Every state/update/reduction here uses FP32. Build
// without fast math or FMA contraction; never mutate the incoming state.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_runtime.h>
#include <cmath>
#include <optional>
#include <tuple>

namespace {
constexpr int kHeads = 16;
constexpr int kDim = 128;
constexpr int kColumns = 32;
constexpr int kWarps = 4;

template <typename G, typename Beta>
__global__ void recurrent_kernel(
    const at::Half* __restrict__ query,
    const at::Half* __restrict__ key,
    const at::Half* __restrict__ value,
    const G* __restrict__ g,
    const Beta* __restrict__ beta,
    const float* __restrict__ incoming,
    at::Half* __restrict__ output,
    float* __restrict__ outgoing) {
  // Each warp owns one quarter of K; its lanes own 32 adjacent V columns.
  // Thus all state memory accesses are coalesced across the warp.
  const int head = blockIdx.y;
  const int column = blockIdx.x * kColumns + threadIdx.x % kColumns;
  const int lane = threadIdx.x % kColumns;
  const int warp = threadIdx.x / kColumns;
  const int vector_base = head * kDim;
  const int state_base = head * kDim * kDim;
  const float gate = expf(static_cast<float>(g[head]));
  const float strength = static_cast<float>(beta[head]);
  const float scale = 1.0f / sqrtf(static_cast<float>(kDim));
  float state[kDim / kWarps];
  float memory_sum = 0.0f;
  __shared__ float partial[kWarps][kColumns];

#pragma unroll
  for (int j = 0; j < kDim / kWarps; ++j) {
    const int k = warp + j * kWarps;
    const int index = state_base + k * kDim + column;
    const float old_value = incoming == nullptr ? 0.0f : incoming[index];
    state[j] = old_value * gate;
    memory_sum += state[j] * static_cast<float>(key[vector_base + k]);
  }
  partial[warp][lane] = memory_sum;
  __syncthreads();
  const float memory = (partial[0][lane] + partial[1][lane])
                     + (partial[2][lane] + partial[3][lane]);
  const float delta = (static_cast<float>(value[vector_base + column]) - memory) * strength;
  // All warps must finish reading the first reduction before reusing storage.
  __syncthreads();

  float result_sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kDim / kWarps; ++j) {
    const int k = warp + j * kWarps;
    const float updated = state[j] + static_cast<float>(key[vector_base + k]) * delta;
    const float scaled_query = static_cast<float>(query[vector_base + k]) * scale;
    result_sum += updated * scaled_query;
    if (outgoing != nullptr) {
      outgoing[state_base + k * kDim + column] = updated;
    }
  }
  partial[warp][lane] = result_sum;
  __syncthreads();
  if (warp == 0) {
    const float result = (partial[0][lane] + partial[1][lane])
                       + (partial[2][lane] + partial[3][lane]);
    output[vector_base + column] = at::Half(result);
  }
}

template <typename G, typename Beta>
void launch(const at::Tensor& query, const at::Tensor& key,
            const at::Tensor& value, const at::Tensor& g,
            const at::Tensor& beta, const std::optional<at::Tensor>& incoming,
            at::Tensor& output, std::optional<at::Tensor>& outgoing) {
  const dim3 grid(kDim / kColumns, kHeads);
  const dim3 block(kColumns * kWarps);
  const auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
  recurrent_kernel<G, Beta><<<grid, block, 0, stream>>>(
      query.data_ptr<at::Half>(), key.data_ptr<at::Half>(), value.data_ptr<at::Half>(),
      g.data_ptr<G>(), beta.data_ptr<Beta>(),
      incoming.has_value() ? incoming->data_ptr<float>() : nullptr,
      output.data_ptr<at::Half>(), outgoing.has_value() ? outgoing->data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, std::optional<at::Tensor>> recurrent(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& g, const at::Tensor& beta,
    const std::optional<at::Tensor>& incoming, bool output_final_state) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  const c10::cuda::CUDAGuard guard(query.device());
  const auto* properties = at::cuda::getDeviceProperties(query.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0, "kernel requires SM70");
  TORCH_CHECK(query.sizes() == at::IntArrayRef({1, 1, kHeads, kDim}), "unsupported query shape");
  for (const auto* tensor : {&query, &key, &value}) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous(),
                "q/k/v must be contiguous on the same CUDA device");
    TORCH_CHECK(tensor->sizes() == query.sizes() && tensor->scalar_type() == at::kHalf,
                "q/k/v must have the supported shape and FP16 dtype");
  }
  for (const auto* tensor : {&g, &beta}) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous(),
                "g/beta must be contiguous on the same CUDA device");
    TORCH_CHECK(tensor->sizes() == at::IntArrayRef({1, 1, kHeads}), "unsupported g/beta shape");
    TORCH_CHECK(tensor->scalar_type() == at::kFloat || tensor->scalar_type() == at::kHalf,
                "g/beta must be FP32 or FP16");
  }
  if (incoming.has_value()) {
    TORCH_CHECK(incoming->device() == query.device() && incoming->is_contiguous()
                && incoming->scalar_type() == at::kFloat
                && incoming->sizes() == at::IntArrayRef({1, kHeads, kDim, kDim}),
                "incoming state must have supported shape, FP32 dtype and CUDA device");
  }
  auto output = at::empty_like(query);
  std::optional<at::Tensor> outgoing = std::nullopt;
  if (output_final_state) {
    outgoing = at::empty({1, kHeads, kDim, kDim}, query.options().dtype(at::kFloat));
  }
  if (g.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat) {
    launch<float, float>(query, key, value, g, beta, incoming, output, outgoing);
  } else if (g.scalar_type() == at::kFloat) {
    launch<float, at::Half>(query, key, value, g, beta, incoming, output, outgoing);
  } else if (beta.scalar_type() == at::kFloat) {
    launch<at::Half, float>(query, key, value, g, beta, incoming, output, outgoing);
  } else {
    launch<at::Half, at::Half>(query, key, value, g, beta, incoming, output, outgoing);
  }
  return {output, outgoing};
}
}  // namespace

TORCH_LIBRARY(gamescaling_v100_gdn_v1, m) {
  m.def("recurrent(Tensor query, Tensor key, Tensor value, Tensor g, Tensor beta, "
        "Tensor? initial_state, bool output_final_state) -> (Tensor, Tensor?)");
  m.impl("recurrent", TORCH_FN(recurrent));
}
