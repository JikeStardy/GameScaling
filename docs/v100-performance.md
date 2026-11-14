# V100 inference speed verification

Measured in the ai-devel container on 2026-11-14, using the same
GameScaling-0.8B-S3 checkpoint, PyTorch 2.8.0+cu128 and Transformers 5.14.1.
The V100 is SM70. Model parameters stay FP16 on cuda:0, with no CPU/disk
weight offload. Image size, history and output budget are unchanged.

## Controlled comparison

`tools/profile_speed.py` runs each variant serially in one process, with two
warmups and 15 measured requests for each **full** history depth. The public
Doom image is 854×480; generation is greedy with a 512-token output budget.
Every request has one new JPEG COM identity and the preceding history IDs.
All decoded pixels remain exactly identical. This controls cache hit/miss
load; it is not a sequence of new gameplay frames or a game ability test.
The measured timer includes server preprocessing and generation, excluding
client encoding, network and request construction. Profiling is separate
from these uninstrumented trials.

| Full historical frames | Eager/reference p50 | Optimized p50 | Latency reduction |
| --- | ---: | ---: | ---: |
| 0 | 606.53 ms | 448.39 ms | 26.1% |
| 2 | 768.26 ms | 541.57 ms | 29.5% |
| 5 | 1002.94 ms | 716.21 ms | 28.6% |

Optimized means vision-only SDPA, SM70 recurrent GDN CUDA kernel, triangular
solve prefill, bounded 64 MiB GPU vision-feature cache and single-token decode
CUDA Graph. Text attention remains eager. These are opt-in server options;
the default path remains the reference implementation. No quantization,
image/history reduction, shorter output limit or changed sampling policy is
required to enable them.

The three variants (reference, optimized without graph, optimized with graph)
produced 153 raw requests including warmups; all passed strict raw-action
validation. Each formal history group contains 15 results. Completion lengths
were 31/31/32 tokens for histories 0/2/5. Graph capture happened once per
history capacity (three captures total), with 1547 real replay steps and zero
capture/replay errors. Only two graph entries are retained. Kernel Python
call counters count capture execution, not every graph replay.

Greedy output matched the reference at histories 2 and 5. At history 0,
SDPA/fused floating-point ordering changed mouse Y from +1 to -1. The graph
variant matched the optimized eager variant in every history group. Numerical
kernel tests and valid action format do **not** establish unchanged game task
accuracy. Actual Windows/gameplay performance remains to be tested.

The earlier HTTP acceptance numbers (about 667/847/948 ms) used temperature 1,
a repeated image and history limits that build up/reset over six requests.
They cannot be directly divided by this fixed-full-history greedy benchmark.
HTTP acceptance must be repeated separately after deployment, with its raw
responses and runtime metadata retained outside Git.

## Reproduce

Stop only the identity-verified inference process before a GPU experiment;
do not run a second full model alongside it. Use the serving environment:

```bash
export MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=7.0
export TORCH_EXTENSIONS_DIR=/path/to/private/torch-extensions
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python tools/profile_speed.py \
  --model-path /path/to/GameScaling-0.8B-S3 \
  --output /path/to/new-evidence-directory \
  --variants eager gdn-vision-cache-solve gdn-vision-cache-solve-graph \
  --trials 15 --rolling
```

A unique output directory preserves each raw run. CUDA 12.8 nvcc, a C++
compiler and ninja must already be available for the recurrent extension.
`--profile` adds separate diagnostic probes; use clean runs for speed claims.
The first graph capture and first CUDA/model warmup cost extra time, so retain
warmup rows and report cold latency separately when it matters to clients.
