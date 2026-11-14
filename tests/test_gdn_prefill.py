"""Algebra, CPU diagnostics and optional SM70 numerical acceptance."""
from pathlib import Path
import random
import subprocess
import sys

import pytest


def test_import_has_no_torch_side_effect():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import gamescaling.gdn_prefill; assert 'torch' not in sys.modules"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_strict_lower_recurrence_is_unit_triangular_inverse():
    """Independent Python arithmetic proves the orientation, without Torch."""
    rng = random.Random(817)
    for n in (1, 2, 8, 16):
        a = [[rng.uniform(-.1, .1) if j < i else 0.0 for j in range(n)] for i in range(n)]
        row_recurrence = [row[:] for row in a]
        for i in range(1, n):
            original_row = row_recurrence[i][:i]
            sub = [row[:i] for row in row_recurrence[:i]]
            for j in range(i):
                row_recurrence[i][j] = original_row[j] + sum(original_row[k] * sub[k][j] for k in range(i))
        inverse = [[row_recurrence[i][j] + (i == j) for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                residual = sum(((i == k) - a[i][k]) * inverse[k][j] for k in range(n))
                assert abs(residual - (i == j)) < 1e-12


def _torch_and_reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    return torch, torch_chunk_gated_delta_rule


@pytest.mark.parametrize("length", [1, 63, 64, 65, 1361, 3636])
@pytest.mark.parametrize("normalized", [False, True])
def test_cpu_small_shape_matches_reference(length, normalized):
    torch, reference = _torch_and_reference()
    from gamescaling.gdn_prefill import torch_chunk_gated_delta_rule_solve
    torch.manual_seed(615)
    q, k, v = [torch.randn((1, length, 1, 8)) * .1 for _ in range(3)]
    g = -torch.rand((1, length, 1)) * .05
    beta = torch.rand_like(g)
    initial = torch.randn((1, 1, 8, 8)) * .1
    saved = initial.clone()
    with torch.inference_mode():
        expected, expected_state = reference(q, k, v, g, beta, initial_state=initial,
                                             output_final_state=True, use_qk_l2norm_in_kernel=normalized)
        actual, actual_state = torch_chunk_gated_delta_rule_solve(
            q, k, v, g, beta, initial_state=initial, output_final_state=True,
            use_qk_l2norm_in_kernel=normalized)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual_state, expected_state, rtol=3e-4, atol=2e-5)
    assert actual.dtype == q.dtype and actual_state.dtype == torch.float32
    assert torch.equal(initial, saved)


def test_cpu_adapter_fallback_preserves_options():
    torch = pytest.importorskip("torch")
    from gamescaling.gdn_prefill import TorchChunkGatedDeltaRuleSolve
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs)); return "original"
    adapter = TorchChunkGatedDeltaRuleSolve(original)
    q = torch.zeros((1, 1, 16, 128), dtype=torch.float16)
    g = torch.zeros((1, 1, 16))
    with torch.inference_mode():
        result = adapter(q, q, q, g, g, chunk_size=64, initial_state=None,
                         output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
    assert result == "original" and adapter.fallback_calls == 1 and adapter.solve_calls == 0
    assert calls[0][1] == {"chunk_size": 64, "initial_state": None, "output_final_state": True,
                           "use_qk_l2norm_in_kernel": True, "cu_seqlens": None}


@pytest.fixture(scope="module")
def cuda_context():
    torch, reference = _torch_and_reference()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("real SM70 CUDA device required")
    from gamescaling.gdn_prefill import TorchChunkGatedDeltaRuleSolve
    return torch, reference, TorchChunkGatedDeltaRuleSolve(reference)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 1361, 3636])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("has_initial", [False, True])
def test_cuda_model_shape_matches_reference(cuda_context, length, normalized, has_initial):
    torch, reference, adapter = cuda_context
    torch.manual_seed(319)
    q, k, v = [torch.randn((1, length, 16, 128), device="cuda", dtype=torch.float16) * .1
               for _ in range(3)]
    g = -torch.rand((1, length, 16), device="cuda", dtype=torch.float32) * .05
    beta = torch.rand_like(g)
    initial = torch.randn((1, 16, 128, 128), device="cuda") * .1 if has_initial else None
    saved = initial.clone() if initial is not None else None
    before = adapter.solve_calls
    with torch.inference_mode():
        expected, expected_state = reference(q, k, v, g, beta, initial_state=initial,
                                             output_final_state=True, use_qk_l2norm_in_kernel=normalized)
        actual, actual_state = adapter(q, k, v, g, beta, initial_state=initial,
                                       output_final_state=True, use_qk_l2norm_in_kernel=normalized,
                                       cu_seqlens=None)
        output_only, no_state = adapter(q, k, v, g, beta, initial_state=initial,
                                        output_final_state=False, use_qk_l2norm_in_kernel=normalized)
    assert adapter.solve_calls == before + 2
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=3e-4, atol=2e-5)
    torch.testing.assert_close(output_only, actual, rtol=0, atol=0)
    assert no_state is None
    assert actual.dtype == torch.float16 and actual_state.dtype == torch.float32
    if initial is not None:
        assert torch.equal(initial, saved)
        assert actual_state.data_ptr() != initial.data_ptr()


def test_cuda_packed_or_wrong_chunk_uses_original(cuda_context):
    torch, _, _ = cuda_context
    from gamescaling.gdn_prefill import TorchChunkGatedDeltaRuleSolve
    calls = []
    def original(*args, **kwargs):
        calls.append(kwargs); return "original"
    adapter = TorchChunkGatedDeltaRuleSolve(original)
    q = torch.zeros((1, 64, 16, 128), device="cuda", dtype=torch.float16)
    g = torch.zeros((1, 64, 16), device="cuda")
    with torch.inference_mode():
        assert adapter(q, q, q, g, g, chunk_size=32) == "original"
        packed = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
        assert adapter(q, q, q, g, g, cu_seqlens=packed) == "original"
    assert adapter.fallback_calls == 2 and adapter.solve_calls == 0
    assert calls[-1]["cu_seqlens"] is packed
