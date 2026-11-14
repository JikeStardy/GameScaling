"""Decode graph state safety, plus optional real tensor/CUDA checks."""

from types import SimpleNamespace

import pytest

from gamescaling.cuda_graph import _CacheSnapshot, _clone_output


def test_snapshot_restores_metadata_and_original_container_identities():
    flags = {0: True}
    layer = SimpleNamespace(has_previous_state=flags, counter=7, nested=[{"v": 2}])
    layers = [layer]
    cache = SimpleNamespace(layers=layers, offloading=False)
    snapshot = _CacheSnapshot(cache, SimpleNamespace(Tensor=()))
    layer.has_previous_state = {0: False}
    flags[0] = False
    layer.counter = 19
    layer.new_field = "probe"
    layer.nested[0]["v"] = 100
    cache.layers = []
    cache.offloading = True
    snapshot.restore()
    assert cache.layers is layers and cache.layers == [layer]
    assert layer.has_previous_state is flags and flags == {0: True}
    assert layer.counter == 7 and layer.nested == [{"v": 2}]
    assert not hasattr(layer, "new_field") and cache.offloading is False


def test_snapshot_handles_shared_objects_and_cycles():
    shared = {"active": True}
    cache = SimpleNamespace(a=shared, b=shared)
    cache.self = cache
    snapshot = _CacheSnapshot(cache, SimpleNamespace(Tensor=()))
    shared.clear()
    cache.a = None
    snapshot.restore()
    assert cache.a is cache.b is shared
    assert shared == {"active": True} and cache.self is cache


def test_real_cpu_snapshot_restores_tensor_values_addresses_and_rebindings():
    torch = pytest.importorskip("torch")
    old = torch.arange(4.0)
    counter = torch.tensor(8)
    flags = {0: True}
    cache = SimpleNamespace(layers=[SimpleNamespace(
        keys=old, cumulative_length=counter, conv_states={0: old},
        recurrent_states={0: torch.ones(2, 2)}, has_previous_state=flags,
    )])
    snapshot = _CacheSnapshot(cache, torch)
    address = old.data_ptr()
    old.add_(99)
    counter.add_(4)
    cache.layers[0].keys = torch.zeros(4)
    cache.layers[0].recurrent_states[0].zero_()
    flags[0] = False
    snapshot.restore()
    layer = cache.layers[0]
    assert layer.keys is old and old.data_ptr() == address
    assert torch.equal(old, torch.arange(4.0))
    assert counter.item() == 8 and torch.equal(layer.recurrent_states[0], torch.ones(2, 2))
    assert flags[0] is True


def test_returned_logits_do_not_alias_graph_replay_storage():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast

    del transformers
    cache = SimpleNamespace()
    logits = torch.ones(1, 1, 3)
    output = Qwen3_5CausalLMOutputWithPast(logits=logits, past_key_values=cache,
                                            rope_deltas=torch.tensor([[2]]))
    detached = _clone_output(output, torch, cache)
    logits.zero_()
    output.rope_deltas.zero_()
    assert isinstance(detached, type(output))
    assert torch.equal(detached.logits, torch.ones(1, 1, 3))
    assert detached.rope_deltas.item() == 2 and detached.past_key_values is cache


def test_cuda_probe_does_not_consume_additional_cache_steps():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    from gamescaling.cuda_graph import _capture_forward

    class ToyModel:
        def __init__(self):
            self.cache = SimpleNamespace(counter=torch.tensor(5., device="cuda"))

        def forward(self, input_ids):
            self.cache.counter.add_(1)
            return {"logits": input_ids.float() + self.cache.counter}

    model = ToyModel()
    token = torch.ones(1, 1, dtype=torch.long, device="cuda")
    graph, output = _capture_forward(model.forward, {"input_ids": token}, model.cache, torch, 2)
    assert model.cache.counter.item() == 5
    graph.replay()
    assert model.cache.counter.item() == 6 and output["logits"].item() == 7
    graph.replay()
    assert model.cache.counter.item() == 7 and output["logits"].item() == 8


def test_capture_failure_restores_cache_and_propagates():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    from gamescaling.cuda_graph import _capture_forward

    cache = SimpleNamespace(counter=torch.tensor(9., device="cuda"), flags={0: True})

    def fail(input_ids):
        cache.counter.add_(1)
        cache.flags[0] = False
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("intentional probe failure")
        return {"logits": input_ids + cache.counter}

    with pytest.raises(RuntimeError, match="intentional probe failure"):
        _capture_forward(fail, {"input_ids": torch.ones(1, 1, device="cuda")}, cache, torch, 1)
    assert cache.counter.item() == 9 and cache.flags == {0: True}
