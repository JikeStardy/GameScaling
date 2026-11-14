"""HTTP contract tests using an injected engine; no torch, CUDA or model load."""

import copy
from contextlib import contextmanager
import os
from pathlib import Path
import string
import subprocess
import sys
import threading
import types

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from gamescaling import ACTION_END, ACTION_START, PolicyConfig, build_request, finalize, parse_action
from gamescaling.policy import encode_image, user_turn
from gamescaling.server import (
    CompletionRequest, GenerationResult, TransformersEngine, create_app, decode_completion, decode_image, processor_messages,
    strip_matching_stop, strip_stop,
)

KEY = "test-api-key-never-returned"
HEADERS = {"Authorization": f"Bearer {KEY}"}
ACTION_BODY = "0 0 0 ; W ; W LB ; ; ; ; "


class FakeEngine:
    def __init__(self):
        self.requests = []
        self.error = False
        self.length = False

    def health(self):
        return {"status": "ready", "parameter_devices": ["cuda:0"], "parameter_dtypes": ["torch.float16"]}

    def generate(self, request):
        self.requests.append(request)
        if self.error:
            raise RuntimeError("sensitive model path /private/example")
        prefix = "\n\n</think>\n\n" if request.messages[-1].content == "<think>" else "\n\n"
        return GenerationResult(prefix + ACTION_START + ACTION_BODY,
                                "length" if self.length else "stop", 25, 12, {"generate": 5.0})


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine):
    with TestClient(create_app(engine, KEY)) as value:
        yield value


def body(reasoning=False):
    cfg = PolicyConfig(reasoning=reasoning, max_tokens=512)
    image_url = encode_image(Image.new("RGB", (32, 18)), cfg)
    return build_request([{"role": "system", "content": "Observe the game."}, user_turn(image_url)], cfg)


def test_auth_and_model_listing(client):
    assert client.get("/health").json() == {"status": "ready", "busy": False}
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/v1/models", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "gamescaling"
    assert response.json()["metadata"]["parameter_devices"] == ["cuda:0"]
    assert KEY not in response.text


@pytest.mark.parametrize("reasoning", [False, True])
def test_official_request_and_finalize_round_trip(client, engine, reasoning):
    response = client.post("/v1/chat/completions", headers=HEADERS, json=body(reasoning))
    assert response.status_code == 200, response.text
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert ACTION_END not in choice["message"]["content"]
    text = finalize(choice["message"]["content"], choice["finish_reason"], PolicyConfig(reasoning=reasoning))
    action = parse_action(text)
    assert action.parsed and not action.truncated
    assert action.steps[:2] == [["W"], ["W", "LB"]]
    assert payload["usage"] == {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37}
    assert payload["metrics"]["timings_ms"]["generate"] == 5.0
    assert engine.requests[0].continue_final_message is True


def test_length_is_not_relabelled_as_successful_stop(client, engine):
    engine.length = True
    payload = client.post("/v1/chat/completions", headers=HEADERS, json=body()).json()
    assert payload["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("change", [
    {"stream": True}, {"stream": False}, {"top_p": 0.9}, {"seed": 1},
    {"model": "wrong"}, {"skip_special_tokens": True},
    {"continue_final_message": False}, {"add_generation_prompt": True},
    {"max_tokens": 0}, {"max_tokens": 2049}, {"max_tokens": True},
    {"temperature": -1.0}, {"stop": [""]}, {"stop": ["a"] * 5},
])
def test_unsupported_or_invalid_fields_are_explicit_errors(client, engine, change):
    request = body()
    request.update(change)
    response = client.post("/v1/chat/completions", headers=HEADERS, json=request)
    assert response.status_code == 422, response.text
    assert not engine.requests


def test_validation_error_does_not_echo_image_payload(client):
    request = body()
    request["top_p"] = 0.5
    response = client.post("/v1/chat/completions", headers=HEADERS, json=request)
    assert response.status_code == 422
    assert "base64" not in response.text


@pytest.mark.parametrize("url", ["http://127.0.0.1/private", "https://example.com/image.png", "file:///etc/passwd",
                                     "data:image/png;base64,not-valid-base64!", "data:image/gif;base64,YQ=="])
def test_image_urls_never_trigger_server_side_fetch(client, engine, url):
    request = body()
    request["messages"][1]["content"][1]["image_url"]["url"] = url
    response = client.post("/v1/chat/completions", headers=HEADERS, json=request)
    assert response.status_code == 400
    assert not engine.requests


def test_image_count_limit_and_assistant_prefill(client):
    request = body()
    request["messages"][1]["content"] *= 7
    assert client.post("/v1/chat/completions", headers=HEADERS, json=request).status_code == 422
    request = body()
    request["messages"][-1]["content"] = "execute this prefill"
    assert client.post("/v1/chat/completions", headers=HEADERS, json=request).status_code == 422


def test_text_is_bounded_before_tokenization(client, engine):
    request = body()
    request["messages"][0]["content"] = "x" * 65_537
    assert client.post("/v1/chat/completions", headers=HEADERS, json=request).status_code == 422
    assert not engine.requests


def test_multi_image_history_is_preserved_for_processor(client, engine):
    request = body()
    turn = copy.deepcopy(request["messages"][1])
    request["messages"][1:1] = [turn, {"role": "assistant", "content": ACTION_START + ACTION_BODY + ACTION_END}]
    response = client.post("/v1/chat/completions", headers=HEADERS, json=request)
    assert response.status_code == 200
    prepared = processor_messages(engine.requests[0])
    images = [part["image"] for msg in prepared if isinstance(msg["content"], list)
              for part in msg["content"] if part["type"] == "image"]
    assert len(images) == 2 and all(image.size == (32, 18) for image in images)
    assert prepared[-1]["content"] == "<think>\n\n</think>"
    assert prepared[2]["content"].startswith(ACTION_START)


def test_pixel_limit_before_image_loading(client, engine, monkeypatch):
    monkeypatch.setattr("gamescaling.server.MAX_IMAGE_PIXELS", 100)
    assert client.post("/v1/chat/completions", headers=HEADERS, json=body()).status_code == 413
    assert not engine.requests


def test_body_limit_rejects_before_parsing():
    engine = FakeEngine()
    with TestClient(create_app(engine, KEY, max_body_bytes=100)) as client:
        response = client.post("/v1/chat/completions", headers=HEADERS, json=body())
        assert response.status_code == 413
        assert not engine.requests


def test_failed_generation_does_not_leak_internal_error_and_releases_lock(client, engine):
    engine.error = True
    response = client.post("/v1/chat/completions", headers=HEADERS, json=body())
    assert response.status_code == 500
    assert "private" not in response.text
    engine.error = False
    assert client.post("/v1/chat/completions", headers=HEADERS, json=body()).status_code == 200


def test_concurrent_request_is_rejected_without_queueing():
    entered, release = threading.Event(), threading.Event()

    class BlockingEngine(FakeEngine):
        def generate(self, request):
            entered.set()
            assert release.wait(5)
            return super().generate(request)

    with TestClient(create_app(BlockingEngine(), KEY)) as client:
        responses = []
        worker = threading.Thread(target=lambda: responses.append(
            client.post("/v1/chat/completions", headers=HEADERS, json=body())))
        worker.start()
        try:
            assert entered.wait(5)
            response = client.post("/v1/chat/completions", headers=HEADERS, json=body())
            assert response.status_code == 503 and response.json()["error"]["code"] == "model_busy"
            assert client.get("/health").json()["busy"] is True
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive() and responses[0].status_code == 200


def test_stop_strings_use_first_occurrence_and_preserve_other_special_tokens():
    assert strip_stop("<|action_start|>0 0 0 STOP ignored END", ["END", "STOP"]) == ("<|action_start|>0 0 0 ", True)
    assert strip_stop("special <think> is preserved", ["END"]) == ("special <think> is preserved", False)


def test_matching_stop_records_the_actual_first_marker():
    text = ACTION_START + ACTION_BODY + ACTION_END + " trailing STOP"
    assert strip_matching_stop(text, ["STOP", ACTION_END]) == (ACTION_START + ACTION_BODY, ACTION_END)
    assert strip_matching_stop("first STOP and END", ["END", "STOP"]) == ("first ", "STOP")
    assert strip_matching_stop("no marker", [ACTION_END]) == ("no marker", None)


class StubTokenizer:
    tokens = {1: ACTION_START + ACTION_BODY, 2: ACTION_END, 3: "<|im_end|>"}

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(self.tokens[token] for token in ids)


def test_eos_without_action_end_does_not_report_a_matched_action_stop():
    text, matched, ended_eos = decode_completion(StubTokenizer(), [1, 3], {3}, [ACTION_END])
    assert text == ACTION_START + ACTION_BODY
    assert matched is None and ended_eos is True


def test_action_stop_is_observed_before_an_independent_eos_is_removed():
    text, matched, ended_eos = decode_completion(StubTokenizer(), [1, 2, 3], {3}, [ACTION_END])
    assert text == ACTION_START + ACTION_BODY
    assert matched == ACTION_END and ended_eos is True
    # Explicit stop strings may also name an EOS token.
    text, matched, ended_eos = decode_completion(StubTokenizer(), [1, 3], {3}, ["<|im_end|>"])
    assert text == ACTION_START + ACTION_BODY
    assert matched == "<|im_end|>" and ended_eos is True


@pytest.mark.parametrize("matched,ended_eos", [(ACTION_END, False), (None, True), (None, False)])
def test_completion_response_exposes_actual_stop_metadata(matched, ended_eos):
    class StopEngine(FakeEngine):
        def generate(self, request):
            result = super().generate(request)
            result.matched_stop = matched
            result.ended_eos = ended_eos
            return result

    with TestClient(create_app(StopEngine(), KEY)) as client:
        response = client.post("/v1/chat/completions", headers=HEADERS, json=body())
        assert response.status_code == 200
        metrics = response.json()["metrics"]
        assert metrics["matched_stop"] == matched
        assert metrics["ended_eos"] is ended_eos


def test_empty_placeholder_api_keys_are_rejected():
    for key in ("", "EMPTY"):
        with pytest.raises(ValueError):
            create_app(FakeEngine(), key)


@pytest.mark.parametrize("options", [
    {"attention_implementation": "flash_attention_2"},
    {"stop_implementation": "eos_token_only"},
    {"gdn_implementation": "unsupported"},
    {"prefill_implementation": "unsupported"},
    {"decode_implementation": "unsupported"},
    {"vision_cache_mib": -1}, {"vision_cache_mib": 513},
    {"vision_cache_mib": True}, {"vision_cache_mib": 64.0},
])
def test_invalid_runtime_options_fail_before_importing_torch(options):
    with pytest.raises(ValueError):
        TransformersEngine("/model-not-needed", **options)


def stop_engine(tokenizer, implementation="legacy"):
    engine = object.__new__(TransformersEngine)
    engine.processor = types.SimpleNamespace(tokenizer=tokenizer)
    engine.stop_implementation = implementation
    engine._tensor_stop_criterion = None
    engine._tensor_stop_tokenizer = None
    return engine


class TokenRows:
    """Only the two indexing operations needed to inspect wrapper routing."""
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        row, column = index
        if isinstance(row, int):
            return self.rows[row][column]
        return TokenRows([r[column] for r in self.rows[row]])


@pytest.fixture
def fake_stop_transformers(monkeypatch):
    module = types.ModuleType("transformers")

    class SpyTensorCriterion:
        instances = []

        def __init__(self, tokenizer, stops):
            self.tokenizer, self.stops = tokenizer, stops
            self.seen = []
            self.result = object()
            self.instances.append(self)

        def __call__(self, ids, scores, **kwargs):
            self.seen.append(ids.rows)
            return self.result

    module.StoppingCriteria = type("StoppingCriteria", (), {})
    module.StoppingCriteriaList = list
    module.StopStringCriteria = SpyTensorCriterion
    monkeypatch.setitem(sys.modules, "transformers", module)
    return SpyTensorCriterion


def test_tensor_stop_is_cached_and_only_sees_generated_continuation(fake_stop_transformers):
    tokenizer = StubTokenizer()
    engine = stop_engine(tokenizer, "tensor")
    first = engine._stopping_criteria(2, [ACTION_END])[0]
    spy = fake_stop_transformers.instances[0]
    assert first(TokenRows([[2, 3, 1]]), None) is spy.result
    assert spy.seen == [[[1]]]
    second = engine._stopping_criteria(1, [ACTION_END])[0]
    assert second(TokenRows([[3, 2]]), None) is spy.result
    assert spy.seen[-1] == [[2]]
    assert len(fake_stop_transformers.instances) == 1
    engine.processor.tokenizer = StubTokenizer()
    engine._stopping_criteria(1, [ACTION_END])
    assert len(fake_stop_transformers.instances) == 2


@pytest.mark.parametrize("stops", [["STOP"], [ACTION_END, "STOP"]])
def test_tensor_mode_keeps_arbitrary_stops_on_original_string_path(fake_stop_transformers, stops):
    class Tokenizer(StubTokenizer):
        tokens = {**StubTokenizer.tokens, 4: "ST", 5: "OP"}

    engine = stop_engine(Tokenizer(), "tensor")
    criterion = engine._stopping_criteria(1, stops)[0]
    assert criterion(TokenRows([[3, 4, 5]]), None) is True
    assert not fake_stop_transformers.instances
    assert engine._stopping_criteria(1, []) is None


def test_runtime_stop_switch_rejects_unknown_value(fake_stop_transformers):
    engine = stop_engine(StubTokenizer(), "invalid")
    with pytest.raises(ValueError):
        engine._stopping_criteria(1, [ACTION_END])


def test_visual_cache_keys_preserve_order_ignore_task_and_clear_on_backend_change():
    class Cache:
        def __init__(self):
            self.requests, self.clears = [], 0

        @contextmanager
        def context(self, keys):
            self.requests.append(keys)
            yield

        def clear(self):
            self.clears += 1

    engine = object.__new__(TransformersEngine)
    engine._vision_cache = Cache()
    engine._model_path = "/one-model"
    engine.transformers = types.SimpleNamespace(__version__="same-transformers")
    engine.torch = types.SimpleNamespace(__version__="same-torch")
    engine.dtypes = ["torch.float16"]
    vision = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="eager"))
    text = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="eager"))
    engine.model = types.SimpleNamespace(model=types.SimpleNamespace(visual=vision, language_model=text))
    engine._vision_cache_backend = engine._cache_fingerprint()
    first = body()
    other_url = encode_image(Image.new("RGB", (32, 18), "red"), PolicyConfig())
    first["messages"][1]["content"].append({"type": "image_url", "image_url": {"url": other_url}})
    with engine._vision_context(CompletionRequest.model_validate(first)):
        pass
    next_task = copy.deepcopy(first)
    next_task["messages"][0]["content"] = "A different task must reuse images, never an action reply."
    with engine._vision_context(CompletionRequest.model_validate(next_task)):
        pass
    assert engine._vision_cache.requests[0] == engine._vision_cache.requests[1]
    assert engine._vision_cache.requests[0][0] != engine._vision_cache.requests[0][1]
    assert engine._vision_cache.clears == 0
    vision.config._attn_implementation = "sdpa"
    with engine._vision_context(CompletionRequest.model_validate(first)):
        pass
    assert engine._vision_cache.clears == 1
    assert engine._vision_cache.requests[-1] != engine._vision_cache.requests[0]


def test_health_reports_actual_mixed_attention_and_optional_backend_stats():
    engine = object.__new__(TransformersEngine)
    engine.devices, engine.dtypes = ["cuda:0"], ["torch.float16"]
    engine.stop_implementation = "tensor"
    engine.max_context_tokens = 8192
    engine.transformers = types.SimpleNamespace(__version__="tested")
    engine.torch = types.SimpleNamespace(
        __version__="tested", version=types.SimpleNamespace(cuda="12.8"),
        cuda=types.SimpleNamespace(get_device_name=lambda: "V100", get_device_capability=lambda: (7, 0),
                                   memory_allocated=lambda: 10, max_memory_allocated=lambda: 20),
    )
    engine.model = types.SimpleNamespace(model=types.SimpleNamespace(
        visual=types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="sdpa")),
        language_model=types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="eager")),
    ))
    engine._vision_cache = types.SimpleNamespace(stats=lambda: {"entries": 1, "hits": 5})
    engine._recurrent_kernel = types.SimpleNamespace(statistics=lambda: {"kernel_calls": 18, "fallback_calls": 0})
    engine._prefill_solve = None
    engine._decode_graph = None
    health = engine.health()
    assert health["attention_implementation"] == "mixed"
    assert health["attention_implementations"] == {"vision": "sdpa", "text": "eager"}
    assert health["stop_implementation"] == "tensor"
    assert health["recurrent_kernel"]["kernel_calls"] == 18
    assert health["linear_attention_backend"] == "pytorch_reference_prefill_cuda_sm70_recurrent"
    assert health["prefill_kernel"] == {"backend": "pytorch_reference", "solve_calls": 0}
    assert health["decode_implementation"] == "eager"
    assert health["decode_graph"] == {"enabled": False}
    assert health["vision_cache"] == {"entries": 1, "hits": 5}
    assert "image" not in repr(health["vision_cache"])
    engine._prefill_solve = types.SimpleNamespace(statistics=lambda: {
        "backend": "torch_triangular_solve_v1", "solve_calls": 2, "fallback_calls": 1,
    })
    engine._decode_graph = types.SimpleNamespace(statistics=lambda: {"captures": 1, "replays": 12})
    health = engine.health()
    assert health["linear_attention_backend"] == "torch_triangular_solve_prefill_cuda_sm70_recurrent"
    assert health["prefill_kernel"]["solve_calls"] == 2
    assert health["prefill_kernel"]["fallback_calls"] == 1
    assert health["decode_implementation"] == "cuda-graph"
    assert health["decode_graph"] == {"captures": 1, "replays": 12}


@pytest.mark.parametrize("graph_enabled", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_generate_graph_context_is_inside_inference_and_uses_static_cache_only_when_enabled(graph_enabled, raises):
    state = {"inference": False, "graph": False}

    class Tensor:
        shape = (1, 1)

        def to(self, **kwargs):
            assert kwargs == {"device": "cuda:0"}
            return self

        def __getitem__(self, key):
            assert key == (0, slice(1, None))
            return self

        def tolist(self):
            return [1, 3]

    @contextmanager
    def inference():
        state["inference"] = True
        try:
            yield
        finally:
            state["inference"] = False

    @contextmanager
    def graph():
        assert state["inference"]
        state["graph"] = True
        try:
            yield
        finally:
            state["graph"] = False

    def generate(**options):
        assert state == {"inference": True, "graph": graph_enabled}
        assert options["do_sample"] is False
        assert "temperature" not in options
        if graph_enabled:
            assert options["cache_implementation"] == "static"
            assert options["disable_compile"] is True
        else:
            assert "cache_implementation" not in options
            assert "disable_compile" not in options
        if raises:
            raise RuntimeError("forward failed")
        return Tensor()

    engine = object.__new__(TransformersEngine)
    engine.torch = types.SimpleNamespace(
        float16="float16", is_floating_point=lambda value: False,
        cuda=types.SimpleNamespace(synchronize=lambda: None), inference_mode=inference,
    )
    engine.processor = types.SimpleNamespace(
        tokenizer=StubTokenizer(), apply_chat_template=lambda *a, **kw: {"input_ids": Tensor()},
    )
    engine.model = types.SimpleNamespace(generate=generate, generation_config=types.SimpleNamespace(eos_token_id=3))
    engine.max_context_tokens = 8192
    engine._vision_cache = None
    engine._decode_graph = types.SimpleNamespace(context=graph) if graph_enabled else None
    payload = body()
    payload.update(temperature=0, stop=None)
    request = CompletionRequest.model_validate(payload)
    if raises:
        with pytest.raises(RuntimeError, match="forward failed"):
            engine.generate(request)
    else:
        result = engine.generate(request)
        assert result.ended_eos is True
        assert result.finish_reason == "stop"
        assert result.content == ACTION_START + ACTION_BODY
    assert state == {"inference": False, "graph": False}


@pytest.fixture
def tensor_stop_tokenizer():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    symbols = ["[UNK]", *string.printable, ACTION_END, "x" + ACTION_END + "tail"]
    vocabulary = {token: i for i, token in enumerate(symbols)}
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocabulary, merges=[], unk_token="[UNK]"))
    backend.decoder = tokenizers.decoders.Fuse()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", additional_special_tokens=[ACTION_END],
        clean_up_tokenization_spaces=False,
    )
    return torch, tokenizer, vocabulary


@pytest.mark.parametrize("tokens", [
    [ACTION_END], list(ACTION_END), ["x" + ACTION_END + "tail"],
    ["h", "i", *ACTION_END], ["n", "o", "p", "e"],
])
def test_tensor_and_legacy_action_stop_agree_incrementally(tensor_stop_tokenizer, tokens):
    torch, tokenizer, vocab = tensor_stop_tokenizer
    engine = stop_engine(tokenizer, "legacy")
    legacy = engine._stopping_criteria(1, [ACTION_END])[0]
    engine.stop_implementation = "tensor"
    tensor = engine._stopping_criteria(1, [ACTION_END])[0]
    for end in range(1, len(tokens)+1):
        ids = torch.tensor([[vocab["p"], *(vocab[t] for t in tokens[:end])]])
        original = legacy(ids, None)
        optimized = bool(tensor(ids, None).item())
        assert original == optimized
        if original:
            break


def test_tensor_stop_cannot_match_across_prompt_boundary(tensor_stop_tokenizer):
    torch, tokenizer, vocab = tensor_stop_tokenizer
    prompt, continuation = list("<|action_"), list("end|>")
    engine = stop_engine(tokenizer, "tensor")
    criterion = engine._stopping_criteria(len(prompt), [ACTION_END])[0]
    ids = torch.tensor([[vocab[t] for t in prompt + continuation]])
    assert not bool(criterion(ids, None).item())


@pytest.mark.parametrize("build_directory", [None, "directory with spaces"])
def test_launch_script_passes_optional_build_directory_without_word_splitting(tmp_path, build_directory):
    launcher = tmp_path / "argv-only"
    launcher.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    launcher.chmod(0o700)
    env = {"PATH": os.environ.get("PATH", os.defpath)}
    env.update(GAMESCALING_PYTHON=str(launcher), GAMESCALING_API_KEY_FILE="/private/key")
    if build_directory is not None:
        env["GAMESCALING_KERNEL_BUILD_DIRECTORY"] = build_directory
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "scripts/serve_transformers.sh"), "/model/path"],
                            env=env, capture_output=True, text=True, check=True)
    args = result.stdout.splitlines()
    assert args[:4] == ["-m", "gamescaling.server", "--model-path", "/model/path"]
    assert args[args.index("--attention-implementation")+1] == "eager"
    assert args[args.index("--gdn-implementation")+1] == "reference"
    assert args[args.index("--prefill-implementation")+1] == "reference"
    assert args[args.index("--decode-implementation")+1] == "eager"
    if build_directory is None:
        assert "--kernel-build-directory" not in args
    else:
        assert args[-2:] == ["--kernel-build-directory", build_directory]


def test_launch_script_passes_explicit_prefill_and_decode_options(tmp_path):
    launcher = tmp_path / "argv-only"
    launcher.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    launcher.chmod(0o700)
    env = {"PATH": os.environ.get("PATH", os.defpath), "GAMESCALING_PYTHON": str(launcher),
           "GAMESCALING_API_KEY_FILE": "/private/key", "GAMESCALING_PREFILL_IMPLEMENTATION": "solve",
           "GAMESCALING_DECODE_IMPLEMENTATION": "cuda-graph"}
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "scripts/serve_transformers.sh"), "/model/path"],
                            env=env, capture_output=True, text=True, check=True)
    args = result.stdout.splitlines()
    assert args[args.index("--prefill-implementation")+1] == "solve"
    assert args[args.index("--decode-implementation")+1] == "cuda-graph"
