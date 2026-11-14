"""The optional kernel's CPU fallbacks and real-CUDA numerical contract."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def test_import_is_lazy():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import gamescaling.v100_gdn; assert 'torch' not in sys.modules"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def torch_module():
    return pytest.importorskip("torch")


def test_cpu_and_autograd_use_original(torch_module):
    from gamescaling.v100_gdn import V100RecurrentGatedDeltaRule

    torch = torch_module
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "reference"
    def forbidden(*args):
        raise AssertionError("unsupported call reached the CUDA kernel")
    adapter = V100RecurrentGatedDeltaRule(original, forbidden)
    q = torch.zeros((1, 1, 16, 128), dtype=torch.float16)
    gate = torch.zeros((1, 1, 16))
    with torch.inference_mode():
        assert adapter(q, q, q, gate, gate, None, True) == "reference"
    assert adapter(q, q, q, gate, gate, None, False, use_qk_l2norm_in_kernel=True) == "reference"
    assert adapter.kernel_calls == 0 and adapter.fallback_calls == 2
    assert calls[-1][1] == {"use_qk_l2norm_in_kernel": True}


@pytest.fixture(scope="module")
def cuda_adapter():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("real SM70 CUDA device required")
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    from gamescaling.v100_gdn import V100RecurrentGatedDeltaRule, load_recurrent_kernel
    kernel = load_recurrent_kernel(verbose=True)
    return torch, torch_recurrent_gated_delta_rule, V100RecurrentGatedDeltaRule(torch_recurrent_gated_delta_rule, kernel)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("gate_half,beta_half", [(False, False), (False, True), (True, False), (True, True)])
def test_single_step_matches_reference(cuda_adapter, normalized, initial, gate_half, beta_half):
    torch, original, adapter = cuda_adapter
    torch.manual_seed(2731)
    q, k, v = [torch.randn((1, 1, 16, 128), device="cuda", dtype=torch.float16) * .1 for _ in range(3)]
    g = -torch.rand((1, 1, 16), device="cuda", dtype=torch.float16 if gate_half else torch.float32)
    beta = torch.rand((1, 1, 16), device="cuda", dtype=torch.float16 if beta_half else torch.float32)
    state = torch.randn((1, 16, 128, 128), device="cuda", dtype=torch.float32) * .1 if initial else None
    before = state.clone() if state is not None else None
    with torch.inference_mode():
        expected, expected_state = original(q, k, v, g, beta, state, True, normalized)
        actual, actual_state = adapter(q, k, v, g, beta, state, True, normalized)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)
        torch.testing.assert_close(actual_state, expected_state, rtol=1e-4, atol=2e-6)
        output_only, no_state = adapter(q, k, v, g, beta, state, False, normalized)
    assert no_state is None
    torch.testing.assert_close(output_only, actual, rtol=0, atol=0)
    if state is not None:
        assert torch.equal(state, before), "incoming state was modified"
        assert actual_state.data_ptr() != state.data_ptr()
    assert actual.device.type == "cuda" and actual_state.dtype == torch.float32


@pytest.mark.parametrize("amplitude", [0.0, 1e-4, .1, 1.0])
def test_normalization_edges_and_long_state(cuda_adapter, amplitude):
    torch, original, adapter = cuda_adapter
    torch.manual_seed(431)
    reference_state = actual_state = None
    with torch.inference_mode():
        for _ in range(128):
            q, k = [torch.randn((1, 1, 16, 128), device="cuda", dtype=torch.float16) * amplitude for _ in range(2)]
            v = torch.randn_like(q) * .1
            g = -torch.rand((1, 1, 16), device="cuda", dtype=torch.float32) * .25
            beta = torch.rand_like(g)
            expected, reference_state = original(q, k, v, g, beta, reference_state, True, True)
            actual, actual_state = adapter(q, k, v, g, beta, actual_state, True, True)
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)
            torch.testing.assert_close(actual_state, reference_state, rtol=1e-3, atol=1e-5)


def test_unsupported_sequence_uses_reference(cuda_adapter):
    torch, original, adapter = cuda_adapter
    q = torch.zeros((1, 2, 16, 128), device="cuda", dtype=torch.float16)
    gates = torch.zeros((1, 2, 16), device="cuda")
    count = adapter.fallback_calls
    with torch.inference_mode():
        actual, state = adapter(q, q, q, gates, gates, None, True, True)
    assert adapter.fallback_calls == count + 1
    assert actual.shape == q.shape and state.shape == (1, 16, 128, 128)
