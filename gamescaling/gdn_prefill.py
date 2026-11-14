# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 The GameScaling fork contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional Qwen3.5 GDN prefill experiment using a unit-triangular solve.

Adapted from Transformers v5.14.1 modeling_qwen3_5.py, lines 248--326:
https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py

The only algorithmic change is the strictly-lower triangular row recurrence.
If that matrix is A, the recurrence plus identity constructs (I-A)^-1.
This module instead solves (I-A) X = I in FP32. Floating-point operation
ordering differs; tests and full-model A/B acceptance are required before use.
It is explicit/optional and neither imports Torch nor installs itself on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def torch_chunk_gated_delta_rule_solve(
    query, key, value, g, beta, chunk_size=64, initial_state=None,
    output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs,
):
    """Reference GDN chunk arithmetic with only its row solve replaced.

    This standalone function also accepts small CPU tensors for mathematical
    tests. Production installation is guarded by the narrower adapter below.
    Like the exact HF reference, q/k normalization happens BEFORE the FP32 cast.
    """
    import torch
    import torch.nn.functional as F

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    # For strictly-lower A, upstream updates row_i to A_i (I-A_<i)^-1.
    # Adding I therefore yields X satisfying (I-A) X = I. Do not use a
    # generic inverse or change the FP32 dtype of the source computation.
    identity = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    attn = torch.linalg.solve_triangular(
        identity - attn, identity.expand_as(attn), upper=False, unitriangular=True,
    )
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    # Retain the original reference allocation to isolate the solve change.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class TorchChunkGatedDeltaRuleSolve:
    """Guarded inference adapter; unsupported calls use the original function."""

    def __init__(self, original: Callable):
        import torch

        self.original = original
        self.torch = torch
        self.solve_calls = 0
        self.fallback_calls = 0
        self._device_support: dict[int, bool] = {}

    def _supported(self, query, key, value, g, beta, chunk_size, initial_state, kwargs) -> bool:
        torch = self.torch
        if torch.is_grad_enabled() or chunk_size != 64 or type(chunk_size) is not int:
            return False
        if set(kwargs) - {"cu_seqlens"} or kwargs.get("cu_seqlens") is not None:
            return False
        tensors = (query, key, value, g, beta)
        if not all(isinstance(x, torch.Tensor) for x in tensors):
            return False
        if query.device.type != "cuda" or any(x.device != query.device for x in tensors):
            return False
        if query.ndim != 4 or query.shape[0] != 1 or not 1 <= query.shape[1] <= 8192:
            return False
        if query.shape[2:] != (16, 128) or key.shape != query.shape or value.shape != query.shape:
            return False
        if any(x.dtype != torch.float16 for x in (query, key, value)):
            return False
        if any(x.shape != query.shape[:3] or x.dtype not in (torch.float16, torch.float32) for x in (g, beta)):
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

    def __call__(self, query, key, value, g, beta, chunk_size=64, initial_state=None,
                 output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        args = (query, key, value, g, beta)
        options = {"chunk_size": chunk_size, "initial_state": initial_state,
                   "output_final_state": output_final_state,
                   "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel, **kwargs}
        if not self._supported(query, key, value, g, beta, chunk_size, initial_state, kwargs):
            self.fallback_calls += 1
            return self.original(*args, **options)
        self.solve_calls += 1
        return torch_chunk_gated_delta_rule_solve(*args, **options)


@dataclass
class PrefillSolveInstallation:
    modules: list[tuple[str, Any, Callable, TorchChunkGatedDeltaRuleSolve]]

    def restore(self) -> None:
        for _, module, original, adapter in self.modules:
            if module.chunk_gated_delta_rule is adapter:
                module.chunk_gated_delta_rule = original

    def statistics(self) -> dict[str, Any]:
        return {"backend": "torch_triangular_solve_v1", "layers": len(self.modules),
                "solve_calls": sum(adapter.solve_calls for _, _, _, adapter in self.modules),
                "fallback_calls": sum(adapter.fallback_calls for _, _, _, adapter in self.modules)}


def install_prefill_solve(model) -> PrefillSolveInstallation:
    """Explicitly install on compatible Qwen3.5 modules, with rollback handles."""
    selected = []
    for name, module in model.named_modules():
        if (module.__class__.__name__ == "Qwen3_5GatedDeltaNet"
                and getattr(module, "num_v_heads", None) == 16
                and getattr(module, "head_k_dim", None) == 128
                and getattr(module, "head_v_dim", None) == 128
                and callable(getattr(module, "chunk_gated_delta_rule", None))):
            if isinstance(module.chunk_gated_delta_rule, TorchChunkGatedDeltaRuleSolve):
                raise RuntimeError("The prefill triangular solve is already installed")
            selected.append((name, module, module.chunk_gated_delta_rule))
    if not selected:
        raise RuntimeError("No supported Qwen3.5 prefill layers found; nothing was changed")
    installed = []
    for name, module, original in selected:
        adapter = TorchChunkGatedDeltaRuleSolve(original)
        module.chunk_gated_delta_rule = adapter
        installed.append((name, module, original, adapter))
    return PrefillSolveInstallation(installed)
