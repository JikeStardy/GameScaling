"""Bounded, per-image feature reuse for the frozen Qwen3.5 inference path.

The server must hold its generation lock around ``context`` and ``generate``.
Keys must identify the original image plus its authentication/configuration
namespace. No text state, generated action, KV cache or GDN state is retained.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import threading
from typing import Any, Hashable, Iterable


class VisionFeatureCache:
    """Attach to ``Qwen3_5ForConditionalGeneration.model``.

    Only the standard inference ``pooler_output`` path is cached. Calls outside
    an explicit request context, or with unsupported options/shapes, retain the
    original method. Lifecycle operations belong under the generation lock.
    """

    def __init__(self, model, max_bytes: int = 64 * 1024**2, max_entries: int = 64):
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("cache limits must be nonnegative")
        import torch
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        self._torch = torch
        self._output_type = BaseModelOutputWithPooling
        self._model = model
        self._original = model.get_image_features
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries: OrderedDict[Hashable, Any] = OrderedDict()
        self._bytes = 0
        self._counts = dict(hits=0, misses=0, evictions=0, bypasses=0)
        self._lock = threading.RLock()
        self._keys: ContextVar[tuple[Hashable, ...] | None] = ContextVar(
            f"vision_cache_keys_{id(self)}", default=None,
        )

        @wraps(self._original)
        def wrapped(pixel_values, image_grid_thw=None, **kwargs):
            return self._get_features(pixel_values, image_grid_thw, kwargs)

        self._wrapped = wrapped
        model.get_image_features = wrapped

    @contextmanager
    def context(self, keys: Iterable[Hashable]):
        """Scope ordered image identities to one inference, including failures."""
        if isinstance(keys, (str, bytes)):
            raise TypeError("keys must be an ordered collection of image identities")
        ordered = tuple(keys)
        for key in ordered:
            if key is None:
                raise ValueError("image identities must not be None")
            hash(key)
        if self._keys.get() is not None:
            raise RuntimeError("a vision cache context is already active")
        token = self._keys.set(ordered)
        try:
            yield self
        finally:
            self._keys.reset(token)

    def clear(self, reset_stats: bool = False):
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            if reset_stats:
                self._counts = dict(hits=0, misses=0, evictions=0, bypasses=0)

    def stats(self) -> dict[str, int]:
        """Hit/miss counts describe unique image identities per invocation."""
        with self._lock:
            return dict(self._counts, entries=len(self._entries), bytes=self._bytes,
                        max_bytes=self.max_bytes, max_entries=self.max_entries)

    def close(self):
        """Remove this hook without overwriting a subsequently installed hook."""
        with self._lock:
            if self._model.get_image_features is self._wrapped:
                self._model.get_image_features = self._original
            self.clear()

    def _bypass(self, pixels, grid, kwargs):
        self._counts["bypasses"] += 1
        return self._original(pixels, image_grid_thw=grid, **kwargs)

    def _layout(self, pixels, grid, keys, kwargs):
        torch = self._torch
        allowed = {"return_dict", "output_attentions", "output_hidden_states", "use_cache"}
        if (not self.max_bytes or not self.max_entries or not keys
                or torch.is_grad_enabled() or getattr(self._model, "training", True)
                or set(kwargs) - allowed or kwargs.get("return_dict") is not True
                or kwargs.get("output_attentions") or kwargs.get("output_hidden_states")):
            return None
        if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 2
                or not isinstance(grid, torch.Tensor) or grid.ndim != 2
                or grid.shape[1] != 3 or grid.shape[0] != len(keys)
                or grid.dtype not in (torch.int32, torch.int64)):
            return None
        merge = getattr(getattr(self._model, "visual", None), "spatial_merge_size", None)
        if not isinstance(merge, int) or merge <= 0:
            return None
        rows = [tuple(row) for row in grid.tolist()]
        counts = [t * h * w for t, h, w in rows]
        if (any(min(row) <= 0 for row in rows)
                or any(count % (merge * merge) for count in counts)
                or sum(counts) != pixels.shape[0]):
            return None
        identities = [(key, row, str(pixels.device), pixels.dtype)
                      for key, row in zip(keys, rows)]
        return identities, counts, [n // (merge * merge) for n in counts]

    def _valid_output(self, output, sizes, pixels):
        features = getattr(output, "pooler_output", None)
        return (isinstance(output, self._output_type)
                and isinstance(features, (tuple, list)) and len(features) == len(sizes)
                and all(isinstance(f, self._torch.Tensor) and f.ndim == 2
                        and f.shape[0] == size and f.shape[1] > 0 and f.device == pixels.device
                        and self._torch.is_floating_point(f)
                        for f, size in zip(features, sizes)))

    def _insert(self, key, feature):
        size = feature.numel() * feature.element_size()
        if size > self.max_bytes:
            return
        # torch.split results share the entire original batch storage. An
        # explicit clone makes the byte budget reflect the retained allocation.
        retained = feature.detach().clone()
        while self._entries and (self._bytes + size > self.max_bytes
                                 or len(self._entries) >= self.max_entries):
            _, removed = self._entries.popitem(last=False)
            self._bytes -= removed.numel() * removed.element_size()
            self._counts["evictions"] += 1
        self._entries[key] = retained
        self._bytes += size

    def _get_features(self, pixels, grid, kwargs):
        # The outer service lock serializes whole generations. This lock also
        # makes concurrent health/stats reads and cache eviction consistent.
        with self._lock:
            layout = self._layout(pixels, grid, self._keys.get(), kwargs)
            if layout is None:
                return self._bypass(pixels, grid, kwargs)
            keys, counts, sizes = layout
            found, missing = {}, []
            for i, key in enumerate(keys):
                if key in found or any(keys[j] == key for j in missing):
                    continue
                if key in self._entries:
                    found[key] = self._entries[key]
                    self._entries.move_to_end(key)
                    self._counts["hits"] += 1
                else:
                    missing.append(i)
                    self._counts["misses"] += 1
            output = None
            if missing:
                whole_batch = len(missing) == len(keys)
                if whole_batch:
                    miss_pixels, miss_grid = pixels, grid
                else:
                    chunks = pixels.split(counts, dim=0)
                    miss_pixels = self._torch.cat([chunks[i] for i in missing], dim=0)
                    miss_grid = self._torch.cat([grid[i:i + 1] for i in missing], dim=0)
                output = self._original(miss_pixels, image_grid_thw=miss_grid, **kwargs)
                if not self._valid_output(output, [sizes[i] for i in missing], pixels):
                    self._counts["bypasses"] += 1
                    return output if whole_batch else self._original(pixels, image_grid_thw=grid, **kwargs)
                # Preserve all request hits locally before insertion can evict
                # entries needed by a later image slot in this same request.
                for i, feature in zip(missing, output.pooler_output):
                    found[keys[i]] = feature
                for i in missing:
                    self._insert(keys[i], found[keys[i]])
                if whole_batch:
                    return output
            return self._output_type(pooler_output=tuple(found[key] for key in keys))
