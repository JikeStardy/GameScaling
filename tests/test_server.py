"""HTTP contract tests using an injected engine; no torch, CUDA or model load."""

import copy
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from gamescaling import ACTION_END, ACTION_START, PolicyConfig, build_request, finalize, parse_action
from gamescaling.policy import encode_image, user_turn
from gamescaling.server import (
    GenerationResult, create_app, decode_completion, decode_image, processor_messages,
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
