"""Optional SM70 recurrent GatedDeltaNet experiment.

Only the single-token state update is fused. The checkpoint's exact Torch
FP16 q/k normalization is retained: changing it to FP32 changes rounding.
Compilation is explicit, not an import side effect. Unsupported calls use the
original Transformers function, including training/autograd calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable


def load_recurrent_kernel(build_directory: str | None = None, *, verbose: bool = False):
    """Build/load one process-local operator without Python bindings or new deps."""
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("The optional fused recurrent operator requires a CUDA SM70 device")
    namespace = "gamescaling_v100_gdn_v1"
    if hasattr(getattr(torch.ops, namespace), "recurrent"):
        return getattr(torch.ops, namespace).recurrent
    source = Path(__file__).with_name("csrc") / "gdn_decode.cu"
    identity = hashlib.sha256(source.read_bytes() + torch.__version__.encode()).hexdigest()[:12]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    load(
        name="gamescaling_v100_gdn_" + identity,
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false", "-gencode=arch=compute_70,code=sm_70"],
        build_directory=build_directory,
        with_cuda=True,
        is_python_module=False,
        verbose=verbose,
    )
    return getattr(torch.ops, namespace).recurrent


class V100RecurrentGatedDeltaRule:
    """Original-call-compatible adapter with a deliberately small fast domain."""

    def __init__(self, original: Callable, kernel: Callable):
        import torch

        self.original, self.kernel, self.torch = original, kernel, torch
        self.kernel_calls = 0
        self.fallback_calls = 0
        self._device_support: dict[int, bool] = {}

    def _supported(self, query, key, value, g, beta, initial_state) -> bool:
        torch = self.torch
        tensors = (query, key, value, g, beta)
        if torch.is_grad_enabled() or not all(isinstance(x, torch.Tensor) for x in tensors):
            return False
        if query.device.type != "cuda" or any(x.device != query.device for x in tensors):
            return False
        if query.shape != (1, 1, 16, 128) or key.shape != query.shape or value.shape != query.shape:
            return False
        if any(x.dtype != torch.float16 for x in (query, key, value)):
            return False
        if any(x.shape != (1, 1, 16) or x.dtype not in (torch.float16, torch.float32) for x in (g, beta)):
            return False
        if initial_state is not None and (
            not isinstance(initial_state, torch.Tensor)
            or initial_state.shape != (1, 16, 128, 128)
            or initial_state.device != query.device
            or initial_state.dtype != torch.float32
        ):
            return False
        index = query.device.index
        if index not in self._device_support:
            self._device_support[index] = torch.cuda.get_device_capability(query.device) == (7, 0)
        return self._device_support[index]

    def __call__(self, query, key, value, g, beta, initial_state, output_final_state,
                 use_qk_l2norm_in_kernel=False):
        if not self._supported(query, key, value, g, beta, initial_state):
            self.fallback_calls += 1
            return self.original(
                query, key, value, g, beta, initial_state, output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        if use_qk_l2norm_in_kernel:
            # Exactly the HF 5.14.1 l2norm order/dtypes. In particular, do NOT
            # upcast before these operations; the source normalizes in FP16.
            query = query * self.torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
            key = key * self.torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
        self.kernel_calls += 1
        return self.kernel(
            query.contiguous(), key.contiguous(), value.contiguous(),
            g.contiguous(), beta.contiguous(),
            initial_state.contiguous() if initial_state is not None else None,
            output_final_state,
        )


@dataclass
class RecurrentKernelInstallation:
    """Keeps original callables so callers can restore the reference backend."""

    modules: list[tuple[str, Any, Callable, V100RecurrentGatedDeltaRule]]

    def restore(self) -> None:
        for _, module, original, adapter in self.modules:
            if module.recurrent_gated_delta_rule is adapter:
                module.recurrent_gated_delta_rule = original

    def statistics(self) -> dict[str, Any]:
        return {"backend": "cuda_sm70_recurrent_v1", "layers": len(self.modules),
                "kernel_calls": sum(adapter.kernel_calls for _, _, _, adapter in self.modules),
                "fallback_calls": sum(adapter.fallback_calls for _, _, _, adapter in self.modules)}


def install_recurrent_kernel(model, build_directory: str | None = None, *, verbose: bool = False):
    """Install only Qwen3.5's known 16-head/128-dimension recurrent modules."""
    selected = []
    for name, module in model.named_modules():
        if (module.__class__.__name__ == "Qwen3_5GatedDeltaNet"
                and getattr(module, "num_v_heads", None) == 16
                and getattr(module, "head_k_dim", None) == 128
                and getattr(module, "head_v_dim", None) == 128
                and callable(getattr(module, "recurrent_gated_delta_rule", None))):
            if isinstance(module.recurrent_gated_delta_rule, V100RecurrentGatedDeltaRule):
                raise RuntimeError("The recurrent kernel is already installed")
            selected.append((name, module, module.recurrent_gated_delta_rule))
    if not selected:
        raise RuntimeError("No supported Qwen3.5 recurrent layers found; nothing was changed")
    kernel = load_recurrent_kernel(build_directory, verbose=verbose)
    installed = []
    for name, module, original in selected:
        adapter = V100RecurrentGatedDeltaRule(original, kernel)
        module.recurrent_gated_delta_rule = adapter
        installed.append((name, module, original, adapter))
    return RecurrentKernelInstallation(installed)
