"""Opt-in Qwen3.5/SM70 single-token CUDA Graph prototype.

Keep image prefill, generation bookkeeping, sampling and stopping eager. Use
``cache_implementation='static', disable_compile=True`` with ordinary generate,
under the service generation lock and ``adapter.context()``. This module never
changes precision, image size, token budget or the sampling policy.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import threading
from typing import Any


class _CacheSnapshot:
    """Restore both tensor contents and original metadata/container identities."""

    def __init__(self, cache, torch_module):
        self._tensors, self._objects, self._containers = [], [], []
        seen = set()

        def visit(value):
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            if isinstance(value, torch_module.Tensor):
                self._tensors.append((value, value.clone()))
            elif isinstance(value, dict):
                self._containers.append((value, dict(value)))
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                self._containers.append((value, list(value)))
                for item in value:
                    visit(item)
            elif isinstance(value, tuple):
                for item in value:
                    visit(item)
            elif isinstance(value, set):
                self._containers.append((value, set(value)))
            elif not callable(value) and hasattr(value, "__dict__"):
                attributes = dict(vars(value))
                self._objects.append((value, attributes))
                for item in attributes.values():
                    visit(item)

        visit(cache)

    def restore(self):
        for value, attributes in self._objects:
            vars(value).clear()
            vars(value).update(attributes)
        for value, contents in self._containers:
            if isinstance(value, list):
                value[:] = contents
            else:
                value.clear()
                value.update(contents)
        for tensor, contents in self._tensors:
            tensor.copy_(contents)


def _capture_forward(forward, inputs, cache, torch, warmup_steps):
    """Probe without consuming state; callers must replay once for the real step.

    Exceptions propagate after restoration. A failed capture must never be
    silently retried through eager forward against partially advanced state.
    """
    device = inputs["input_ids"].device
    with torch.cuda.device(device):
        current = torch.cuda.current_stream(device)
        snapshot = _CacheSnapshot(cache, torch)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                for _ in range(warmup_steps):
                    forward(**inputs)
                    # Warmup must not walk beyond the request's capacity when
                    # only one decode slot remains in the static allocation.
                    snapshot.restore()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = forward(**inputs)
        finally:
            # Capture executes the forward once too. Restore on the same stream
            # and order the caller's subsequent replay after that restoration.
            with torch.cuda.stream(stream):
                snapshot.restore()
            current.wait_stream(stream)
        return graph, output


def _clone_output(output, torch, cache):
    """ModelOutput tensors belong to this call, rather than the graph's pool."""
    if output is cache:
        return cache
    if isinstance(output, torch.Tensor):
        return output.clone()
    if isinstance(output, dict):
        values = {key: _clone_output(value, torch, cache) for key, value in output.items()}
        return type(output)(**values) if type(output) is not dict else values
    if isinstance(output, tuple):
        return tuple(_clone_output(value, torch, cache) for value in output)
    if isinstance(output, list):
        return [_clone_output(value, torch, cache) for value in output]
    return output


def _tensor_layout(tensor):
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device


def _structure_key(value, torch):
    if isinstance(value, torch.Tensor):
        return "tensor", _tensor_layout(value)
    if isinstance(value, dict):
        return tuple((key, _structure_key(item, torch)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_structure_key(item, torch) for item in value)
    return value


def _copy_structure(destination, source, torch):
    if isinstance(destination, torch.Tensor):
        destination.copy_(source)
    elif isinstance(destination, dict):
        for key in destination:
            _copy_structure(destination[key], source[key], torch)


def _clone_structure(value, torch):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_structure(item, torch) for key, item in value.items()}
    return value


def _cache_addresses(cache, torch):
    addresses, seen = [], set()

    def visit(value, path):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            addresses.append((path, value.data_ptr(), _tensor_layout(value)))
        elif isinstance(value, dict):
            for key, item in sorted(value.items()):
                visit(item, (*path, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))
        elif not callable(value) and hasattr(value, "__dict__"):
            for key, item in sorted(vars(value).items()):
                visit(item, (*path, key))

    visit(cache, ())
    return tuple(addresses)


def _callable_identity(value):
    return id(getattr(value, "__func__", value))


class DecodeGraphInstallation:
    """Install a bounded decode-only forward hook; inactive outside context.

    Backend/model configuration may change between contexts, never during one
    generation. Capture/replay failures are surfaced; unsupported calls bypass
    before mutating inputs. The caller must reset a failed generation's cache.
    """

    def __init__(self, model, warmup_steps: int = 2, max_graphs: int = 2):
        if warmup_steps < 1 or max_graphs < 1:
            raise ValueError("warmup_steps and max_graphs must be positive")
        import torch
        from transformers.cache_utils import StaticCache

        self._torch, self._cache_type = torch, StaticCache
        self._model, self._original = model, model.forward
        self._warmup_steps, self._max_graphs = warmup_steps, max_graphs
        self._graphs: OrderedDict[Any, Any] = OrderedDict()
        self._active = ContextVar(f"decode_graph_{id(self)}", default=None)
        self._lock = threading.RLock()
        self._closed = False
        self._counts = dict(captures=0, replays=0, bypasses=0, evictions=0,
                            capture_errors=0, replay_errors=0)

        @wraps(self._original)
        def wrapped(*args, **kwargs):
            with self._lock:
                fingerprint = self._active.get()
                inputs = None if args or fingerprint is None else self._supported_inputs(kwargs)
                if inputs is None:
                    self._counts["bypasses"] += 1
                    return self._original(*args, **kwargs)
                return self._run(inputs, fingerprint)

        self._wrapped = wrapped
        model.forward = wrapped

    def _fingerprint(self):
        torch = self._torch
        modules = []
        for module in self._model.modules():
            config = getattr(module, "config", None)
            implementations = tuple((name, _callable_identity(getattr(module, name)))
                                    for name in ("forward", "causal_conv1d_update",
                                                 "recurrent_gated_delta_rule", "chunk_gated_delta_rule")
                                    if hasattr(module, name))
            modules.append((id(module), getattr(config, "_attn_implementation", None), implementations))
        weights = tuple((tensor.data_ptr(), _tensor_layout(tensor))
                        for tensor in (*self._model.parameters(), *self._model.buffers()))
        backends = tuple(getattr(torch.backends.cuda, name)() for name in
                         ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"))
        return tuple(modules), weights, backends, torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")

    @contextmanager
    def context(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("decode graph installation has been restored")
            if self._active.get() is not None:
                raise RuntimeError("decode graph context is already active")
            token = self._active.set(self._fingerprint())
            try:
                yield self
            finally:
                self._active.reset(token)

    def statistics(self):
        with self._lock:
            return dict(self._counts, graphs=len(self._graphs), max_graphs=self._max_graphs)

    def restore(self):
        with self._lock:
            if self._model.forward is self._wrapped:
                self._model.forward = self._original
            self._graphs.clear()
            self._closed = True

    def _supported_inputs(self, kwargs):
        torch = self._torch
        allowed = {"input_ids", "attention_mask", "position_ids", "past_key_values", "inputs_embeds",
                   "labels", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw",
                   "mm_token_type_ids", "logits_to_keep", "use_cache", "return_dict",
                   "output_attentions", "output_hidden_states"}
        if (set(kwargs) - allowed or torch.is_grad_enabled() or self._model.training
                or getattr(self._model.config, "model_type", None) != "qwen3_5"
                or any(kwargs.get(key) is not None for key in
                       ("inputs_embeds", "labels", "pixel_values", "pixel_values_videos"))
                or kwargs.get("use_cache") is not True or kwargs.get("return_dict") is not True
                or kwargs.get("output_attentions") or kwargs.get("output_hidden_states")
                or not isinstance(kwargs.get("logits_to_keep"), int) or kwargs["logits_to_keep"] != 1):
            return None
        ids, positions = kwargs.get("input_ids"), kwargs.get("position_ids")
        if (not isinstance(ids, torch.Tensor) or ids.shape != (1, 1) or ids.device.type != "cuda"
                or ids.dtype != torch.int64 or torch.cuda.get_device_capability(ids.device) != (7, 0)
                or not isinstance(positions, torch.Tensor) or positions.shape != (4, 1, 1)
                or positions.device != ids.device or positions.dtype != torch.int64):
            return None
        cache, mask = kwargs.get("past_key_values"), kwargs.get("attention_mask")
        if (not isinstance(cache, self._cache_type) or getattr(cache, "offloading", False)
                or not isinstance(mask, dict) or set(mask) != {"full_attention", "linear_attention"}
                or mask["linear_attention"] is not None):
            return None
        full_mask = mask["full_attention"]
        if (not isinstance(full_mask, torch.Tensor) or full_mask.ndim != 4
                or full_mask.shape[:3] != (1, 1, 1) or full_mask.device != ids.device):
            return None
        for layer in cache.layers:
            if type(layer).__name__ == "StaticLayer":
                if (not layer.is_initialized or layer.keys.device != ids.device
                        or layer.keys.shape[0] != 1 or layer.keys.shape[-2] != full_mask.shape[-1]
                        or layer.cumulative_length.device != ids.device):
                    return None
            elif type(layer).__name__ == "LinearAttentionLayer":
                if (layer.record_past or not all(layer.has_previous_state.values())
                        or not all(layer.is_conv_states_initialized.values())
                        or not all(layer.is_recurrent_states_initialized.values())):
                    return None
            else:
                return None
        for module in self._model.modules():
            rope_type = getattr(module, "rope_type", "default")
            if isinstance(rope_type, str) and ("dynamic" in rope_type or rope_type == "longrope"):
                return None
        # These metadata tensors affect image prefill/position-id construction,
        # both already completed outside the captured single-token forward.
        return {"input_ids": ids, "position_ids": positions, "attention_mask": mask,
                "past_key_values": cache, "use_cache": True, "return_dict": True,
                "logits_to_keep": 1, "output_attentions": False, "output_hidden_states": False}

    def _run(self, inputs, fingerprint):
        torch, cache = self._torch, inputs["past_key_values"]
        structure = {key: value for key, value in inputs.items() if key != "past_key_values"}
        key = id(cache), _cache_addresses(cache, torch), _structure_key(structure, torch), fingerprint
        if key not in self._graphs:
            fixed = {key: _clone_structure(value, torch) for key, value in inputs.items()}
            try:
                graph, output = _capture_forward(self._original, fixed, cache, torch, self._warmup_steps)
            except BaseException:
                self._counts["capture_errors"] += 1
                raise
            if len(self._graphs) >= self._max_graphs:
                self._graphs.popitem(last=False)
                self._counts["evictions"] += 1
            self._graphs[key] = graph, output, fixed
            self._counts["captures"] += 1
        graph, output, fixed = self._graphs[key]
        self._graphs.move_to_end(key)
        _copy_structure(fixed, inputs, torch)
        try:
            graph.replay()
        except BaseException:
            self._counts["replay_errors"] += 1
            raise
        self._counts["replays"] += 1
        # rope_deltas may change after another request's eager image prefill.
        # It is output metadata, not a graph input when explicit positions exist.
        result = _clone_output(output, torch, cache)
        if hasattr(result, "rope_deltas"):
            current = getattr(self._model.model, "rope_deltas", None)
            result.rope_deltas = current.clone() if isinstance(current, torch.Tensor) else current
        return result
