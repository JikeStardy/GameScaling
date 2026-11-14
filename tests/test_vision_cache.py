"""Real CPU tensor tests; the lightweight HTTP-test environment can skip them."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers.modeling_outputs import BaseModelOutputWithPooling

from gamescaling.vision_cache import VisionFeatureCache


class FakeVisionModel:
    training = False
    visual = SimpleNamespace(spatial_merge_size=2)

    def __init__(self):
        self.calls = []
        self.fail = False
        self.invalid_output = False

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        self.calls.append((pixel_values.clone(), image_grid_thw.clone(), dict(kwargs)))
        if self.fail:
            raise RuntimeError("encoder failed")
        sizes = [int(row.prod()) for row in image_grid_thw]
        features = [chunk.reshape(-1, 4, pixel_values.shape[1]).mean(1)
                    for chunk in pixel_values.split(sizes)]
        if self.invalid_output:
            return BaseModelOutputWithPooling(pooler_output=torch.cat(features))
        # Return split views just like Qwen3.5, to exercise independent storage.
        joined = torch.cat(features)
        return BaseModelOutputWithPooling(
            last_hidden_state=pixel_values,
            pooler_output=joined.split([len(f) for f in features]),
        )


def request(model, cache, keys, values, grids=None):
    grids = grids or [[1, 2, 2] for _ in values]
    grid = torch.tensor(grids)
    pixels = torch.cat([torch.full((t * h * w, 2), float(v))
                        for v, (t, h, w) in zip(values, grids)])
    with torch.inference_mode(), cache.context(keys):
        return model.get_image_features(pixels, image_grid_thw=grid, return_dict=True)


def test_hits_misses_reorder_and_request_duplicates():
    model = FakeVisionModel()
    cache = VisionFeatureCache(model)
    a = request(model, cache, ["a", "b", "a"], [1, 2, 1])
    assert len(model.calls) == 1 and len(model.calls[0][1]) == 2
    assert [f[0, 0].item() for f in a.pooler_output] == [1, 2, 1]
    b = request(model, cache, ["b", "c", "a"], [2, 3, 1])
    assert len(model.calls) == 2 and len(model.calls[1][1]) == 1
    assert [f[0, 0].item() for f in b.pooler_output] == [2, 3, 1]
    request(model, cache, ["a"], [1])
    assert len(model.calls) == 2
    assert cache.stats()["hits"] == 3 and cache.stats()["misses"] == 3


def test_grid_and_fingerprint_are_part_of_identity():
    model = FakeVisionModel()
    cache = VisionFeatureCache(model)
    request(model, cache, [("a", "eager")], [1])
    larger = request(model, cache, [("a", "eager")], [1], [[1, 2, 4]])
    request(model, cache, [("a", "sdpa")], [1])
    assert len(model.calls) == 3
    assert larger.pooler_output[0].shape == (2, 2)


def test_storage_cloned_and_byte_limit_bounded():
    model = FakeVisionModel()
    cache = VisionFeatureCache(model, max_bytes=16, max_entries=4)
    original = request(model, cache, ["a", "b"], [1, 2])
    assert cache.stats()["entries"] == 2 and cache.stats()["bytes"] == 16
    cached = request(model, cache, ["a", "b"], [1, 2])
    assert cached.pooler_output[0].untyped_storage().nbytes() == 8
    assert cached.pooler_output[0].data_ptr() != original.pooler_output[0].data_ptr()
    request(model, cache, ["c"], [3])
    assert cache.stats()["entries"] == 2 and cache.stats()["bytes"] == 16
    assert cache.stats()["evictions"] == 1


def test_eviction_does_not_lose_hits_needed_by_current_request():
    model = FakeVisionModel()
    cache = VisionFeatureCache(model, max_bytes=8, max_entries=1)
    request(model, cache, ["a"], [1])
    out = request(model, cache, ["new", "a"], [2, 1])
    assert [f[0, 0].item() for f in out.pooler_output] == [2, 1]
    assert cache.stats()["entries"] == 1 and cache.stats()["bytes"] == 8


def test_oversized_feature_is_used_but_not_retained():
    model = FakeVisionModel()
    cache = VisionFeatureCache(model, max_bytes=4)
    request(model, cache, ["a"], [1])
    request(model, cache, ["a"], [1])
    assert len(model.calls) == 2 and cache.stats()["bytes"] == 0


def test_entry_limit_clear_and_close():
    model = FakeVisionModel()
    original = model.get_image_features
    cache = VisionFeatureCache(model, max_entries=1)
    request(model, cache, ["a", "b"], [1, 2])
    assert cache.stats()["entries"] == 1
    cache.clear()
    assert cache.stats()["entries"] == 0 and cache.stats()["misses"] == 2
    cache.clear(reset_stats=True)
    assert cache.stats()["misses"] == 0
    cache.close()
    assert model.get_image_features == original


def test_exception_resets_context_and_does_not_poison_cache():
    model = FakeVisionModel()
    cache = VisionFeatureCache(model)
    model.fail = True
    with pytest.raises(RuntimeError, match="encoder failed"):
        request(model, cache, ["a"], [1])
    assert cache.stats()["entries"] == 0
    model.fail = False
    request(model, cache, ["b"], [2])
    assert cache.stats()["entries"] == 1
    # No context must bypass, despite an existing entry for these pixels.
    with torch.inference_mode():
        model.get_image_features(torch.full((4, 2), 2.), torch.tensor([[1, 2, 2]]), return_dict=True)
    assert len(model.calls) == 3 and cache.stats()["bypasses"] == 1


@pytest.mark.parametrize("extra", [{"output_attentions": True}, {"output_hidden_states": True},
                                   {"unknown_option": 1}, {"return_dict": False}])
def test_unsupported_options_bypass_without_retaining(extra):
    model = FakeVisionModel()
    cache = VisionFeatureCache(model)
    kwargs = {"return_dict": True, **extra}
    with torch.inference_mode(), cache.context(["a"]):
        model.get_image_features(torch.ones((4, 2)), torch.tensor([[1, 2, 2]]), **kwargs)
    assert len(model.calls) == 1 and cache.stats()["entries"] == 0
    assert model.calls[0][2] == kwargs


def test_mismatched_image_count_and_invalid_output_fall_back():
    model = FakeVisionModel()
    cache = VisionFeatureCache(model)
    with torch.inference_mode(), cache.context(["a", "b"]):
        model.get_image_features(torch.ones((4, 2)), torch.tensor([[1, 2, 2]]), return_dict=True)
    assert cache.stats()["entries"] == 0
    model.invalid_output = True
    request(model, cache, ["a", "a"], [1, 1])
    # One attempt on the deduplicated miss, then original full-input fallback.
    assert [len(c[1]) for c in model.calls] == [1, 1, 2]
    assert cache.stats()["entries"] == 0


def test_training_grad_and_disabled_cache_bypass():
    model = FakeVisionModel()
    cache = VisionFeatureCache(model)
    with cache.context(["a"]):
        model.get_image_features(torch.ones((4, 2)), torch.tensor([[1, 2, 2]]), return_dict=True)
    model.training = True
    request(model, cache, ["a"], [1])
    assert cache.stats()["entries"] == 0
    cache.close()
    disabled = VisionFeatureCache(model, max_bytes=0)
    model.training = False
    request(model, disabled, ["a"], [1])
    assert disabled.stats()["bypasses"] == 1


def test_context_validation_and_nested_context_cleanup():
    model = FakeVisionModel()
    cache = VisionFeatureCache(model)
    with pytest.raises(TypeError):
        with cache.context("one-key"):
            pass
    with cache.context(["a"]):
        with pytest.raises(RuntimeError, match="already active"):
            with cache.context(["b"]):
                pass
    request(model, cache, ["a"], [1])
    assert cache.stats()["entries"] == 1
