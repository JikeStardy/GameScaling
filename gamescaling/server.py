"""Single-GPU Transformers server for the unmodified GameScaling client.

This implements the client's non-streaming request subset, not the full OpenAI
API. Torch and Transformers are imported only when loading the real engine.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import io
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "gamescaling"
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
MAX_IMAGES = 6
MAX_TEXT_CHARACTERS = 65_536


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        self.message, self.status, self.code = message, status, code


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ImageURL(StrictModel):
    url: str


class ContentPart(StrictModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None

    @model_validator(mode="after")
    def check_part(self):
        if self.type == "text" and (self.text is None or self.image_url is not None):
            raise ValueError("text parts require only text")
        if self.type == "image_url" and (self.image_url is None or self.text is not None):
            raise ValueError("image_url parts require only image_url")
        return self


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class CompletionRequest(StrictModel):
    model: Literal["gamescaling"]
    messages: list[Message] = Field(min_length=2, max_length=32)
    temperature: float = Field(default=1.0, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    stop: list[str] | str | None = None
    skip_special_tokens: Literal[False] = False
    continue_final_message: Literal[True] = True
    add_generation_prompt: Literal[False] = False
    _prepared_messages: list[dict[str, Any]] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def check_messages(self):
        turns = self.messages
        if turns[0].role == "system":
            if not isinstance(turns[0].content, str):
                raise ValueError("system content must be text")
            turns = turns[1:]
        if not turns or len(turns) % 2:
            raise ValueError("messages must contain user/assistant pairs ending in assistant prefill")
        for i, message in enumerate(turns):
            expected = "user" if i % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError("messages must alternate user and assistant after optional system")
            if message.role == "assistant" and not isinstance(message.content, str):
                raise ValueError("assistant content must be text")
        if turns[-1].content not in ("<think>", "<think>\n\n</think>"):
            raise ValueError("unsupported assistant prefill")
        stops = self.stop_strings
        if len(stops) > 4 or any(not s or len(s) > 128 for s in stops):
            raise ValueError("stop must contain at most four nonempty strings of at most 128 characters")
        image_count = sum(
            part.type == "image_url"
            for message in self.messages if isinstance(message.content, list)
            for part in message.content
        )
        if not 1 <= image_count <= MAX_IMAGES:
            raise ValueError(f"requests require 1..{MAX_IMAGES} images")
        text_characters = sum(
            len(message.content) if isinstance(message.content, str)
            else sum(len(part.text or "") for part in message.content)
            for message in self.messages
        )
        if text_characters > MAX_TEXT_CHARACTERS:
            raise ValueError("text exceeds the preprocessing size limit")
        return self

    @property
    def stop_strings(self) -> list[str]:
        return [self.stop] if isinstance(self.stop, str) else (self.stop or [])


def decode_image(url: str) -> Image.Image:
    """Decode bounded, inline images only. No server-side URL fetching."""
    mime, separator, encoded = url.partition(",")
    if not separator or mime not in ("data:image/jpeg;base64", "data:image/png;base64"):
        raise RequestError("images must be inline JPEG or PNG base64 data URLs")
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise RequestError("encoded image exceeds the byte limit", 413, "image_too_large")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise RequestError("image exceeds the byte limit", 413, "image_too_large")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise RequestError("image payload is not JPEG or PNG")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise RequestError("image exceeds the pixel limit", 413, "image_too_large")
            image.load()
            return image.convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RequestError("invalid image data") from exc


def processor_messages(request: CompletionRequest) -> list[dict[str, Any]]:
    if request._prepared_messages is not None:
        return request._prepared_messages
    messages = []
    for message in request.messages:
        content: Any = message.content
        if isinstance(content, list):
            content = [
                {"type": "text", "text": part.text} if part.type == "text"
                else {"type": "image", "image": decode_image(part.image_url.url)}
                for part in content
            ]
        messages.append({"role": message.role, "content": content})
    request._prepared_messages = messages
    return messages


def strip_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    text, matched_stop = strip_matching_stop(text, stops)
    return text, matched_stop is not None


def strip_matching_stop(text: str, stops: list[str]) -> tuple[str, str | None]:
    matches = [(index, stop) for stop in stops if (index := text.find(stop)) >= 0]
    if not matches:
        return text, None
    index, matched_stop = min(matches, key=lambda match: match[0])
    return text[:index], matched_stop


def decode_completion(tokenizer, token_ids: list[int], eos_ids: set[int], stops: list[str]):
    """Record a real stop match before stripping an independent EOS token.

    A generic ``finish_reason=stop`` can mean EOS, so it does not establish that
    an action_end marker was generated. Match against the complete decoded
    continuation first, including EOS in case an explicit stop is itself EOS.
    """
    ended_eos = bool(token_ids and token_ids[-1] in eos_ids)
    text, matched_stop = strip_matching_stop(
        tokenizer.decode(token_ids, skip_special_tokens=False), stops,
    )
    if matched_stop is None and ended_eos:
        text = tokenizer.decode(token_ids[:-1], skip_special_tokens=False)
    return text, matched_stop, ended_eos


@dataclass
class GenerationResult:
    content: str
    finish_reason: Literal["stop", "length"]
    prompt_tokens: int
    completion_tokens: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    matched_stop: str | None = None
    ended_eos: bool = False


class Engine(Protocol):
    def generate(self, request: CompletionRequest) -> GenerationResult: ...
    def health(self) -> dict[str, Any]: ...


class TransformersEngine:
    """FP16 model weights on one CUDA GPU; PyTorch reference linear attention."""

    def __init__(self, model_path: str, max_context_tokens: int = 8192):
        import importlib.util

        for package in ("fla", "causal_conv1d", "flash_attn"):
            if importlib.util.find_spec(package) is not None:
                raise RuntimeError(f"Remove optional fused package {package} from this environment")
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required; CPU fallback is disabled")
        if "sm_70" not in torch.cuda.get_arch_list() and torch.cuda.get_device_capability()[0] == 7:
            raise RuntimeError("This PyTorch build does not include SM70 support")
        self.torch, self.transformers = torch, transformers
        self.max_context_tokens = max_context_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float16,
            attn_implementation="eager", device_map={"": "cuda:0"}, use_kernels=False,
        ).eval()
        devices = {str(p.device) for p in self.model.parameters()}
        if devices != {"cuda:0"}:
            raise RuntimeError(f"All model parameters must be on cuda:0; observed {devices}")
        self.devices = sorted(devices)
        self.dtypes = sorted({str(p.dtype) for p in self.model.parameters()})
        torch.cuda.synchronize()

    def health(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "status": "ready", "backend": "transformers", "model": MODEL_NAME,
            "parameter_devices": self.devices, "parameter_dtypes": self.dtypes,
            "attention_implementation": "eager", "cpu_offload": False, "disk_offload": False,
            "linear_attention_backend": "pytorch_reference_optional_fused_packages_absent",
            "torch_version": torch.__version__, "transformers_version": self.transformers.__version__,
            "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(),
            "gpu_capability": list(torch.cuda.get_device_capability()),
            "max_context_tokens": self.max_context_tokens,
            "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "compute_note": "FP16 weight storage; reference kernels include FP32 intermediate operations",
        }

    def generate(self, request: CompletionRequest) -> GenerationResult:
        from transformers import StoppingCriteria, StoppingCriteriaList

        torch, processor = self.torch, self.processor
        started = time.perf_counter()
        messages = processor_messages(request)
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            continue_final_message=True, add_generation_prompt=False,
        )
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if prompt_tokens + request.max_tokens > self.max_context_tokens:
            raise RequestError("prompt plus max_tokens exceeds the context budget", 400, "context_length_exceeded")
        inputs = {
            key: value.to(device="cuda:0", dtype=torch.float16) if torch.is_floating_point(value)
            else value.to(device="cuda:0")
            for key, value in inputs.items()
        }
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        tokenizer = processor.tokenizer
        stops = request.stop_strings

        class StringStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0, prompt_tokens:], skip_special_tokens=False)
                return any(stop in text for stop in stops)

        options: dict[str, Any] = {
            "max_new_tokens": request.max_tokens, "do_sample": request.temperature > 0,
            "use_cache": True,
        }
        if request.temperature > 0:
            options["temperature"] = request.temperature
        if stops:
            options["stopping_criteria"] = StoppingCriteriaList([StringStop()])
        with torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        torch.cuda.synchronize()
        generated = time.perf_counter()
        new_ids = output[0, prompt_tokens:].tolist()
        completion_tokens = len(new_ids)
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        # EOS is a generation terminator, not assistant content. Action tags are
        # preserved; a real matched stop is separately reported for validation.
        text, matched_stop, ended_eos = decode_completion(tokenizer, new_ids, eos_ids, stops)
        finish = "stop" if matched_stop is not None or ended_eos else "length"
        return GenerationResult(
            content=text, finish_reason=finish, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timings_ms={"preprocess": (prepared-started)*1000,
                        "generate": (generated-prepared)*1000,
                        "total": (time.perf_counter()-started)*1000},
            matched_stop=matched_stop, ended_eos=ended_eos,
        )


class BodyLimitMiddleware:
    """Bound requests before JSON/base64 parsing, including chunked bodies."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            size += len(event.get("body", b""))
            if size > self.limit:
                response = JSONResponse({"error": {"message": "request body too large", "code": "body_too_large"}}, 413)
                return await response(scope, receive, send)
            chunks.append(event)
            if not event.get("more_body", False):
                break
        position = 0

        async def replay():
            nonlocal position
            if position < len(chunks):
                event = chunks[position]
                position += 1
                return event
            return await receive()

        await self.app(scope, replay, send)


def create_app(engine: Engine, api_key: str, max_body_bytes: int = MAX_BODY_BYTES) -> FastAPI:
    if not api_key or api_key == "EMPTY":
        raise ValueError("A nonempty API key other than EMPTY is required")
    app = FastAPI(title="GameScaling Transformers server", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimitMiddleware, limit=max_body_bytes)
    generation_lock = threading.Lock()

    def authenticate(authorization: str | None = Header(default=None)):
        supplied = authorization or ""
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + api_key).encode()):
            raise RequestError("invalid API key", 401, "invalid_api_key")

    @app.exception_handler(RequestError)
    async def request_error(_request: Request, exc: RequestError):
        return JSONResponse({"error": {"message": exc.message, "code": exc.code}}, exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        # Pydantic's default errors include the original base64 image/request.
        errors = [{"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse({"error": {"message": "unsupported or invalid request", "details": errors}}, 422)

    @app.get("/health")
    def health():
        return {"status": "ready", "busy": generation_lock.locked()}

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    def models():
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
                "metadata": engine.health()}

    @app.post("/v1/chat/completions", dependencies=[Depends(authenticate)])
    def completion(body: CompletionRequest):
        if not generation_lock.acquire(blocking=False):
            raise RequestError("model is busy; retry after the active request finishes", 503, "model_busy")
        try:
            # Validate images even for an injected engine; no input URL is ever
            # fetched by this service or passed unvalidated to a backend.
            processor_messages(body)
            result = engine.generate(body)
        except RequestError:
            raise
        except Exception as exc:
            LOGGER.exception("model generation failed")
            raise RequestError("model generation failed; consult server logs", 500, "generation_failed") from exc
        finally:
            generation_lock.release()
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
            "created": int(time.time()), "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.content},
                         "finish_reason": result.finish_reason}],
            "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                      "total_tokens": result.prompt_tokens + result.completion_tokens},
            "metrics": {"timings_ms": result.timings_ms, "matched_stop": result.matched_stop,
                        "ended_eos": result.ended_eos},
        }

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.max_context_tokens < 512 or args.max_context_tokens > 32768:
        parser.error("max-context-tokens must be between 512 and 32768")
    key_file = Path(args.api_key_file)
    if key_file.stat().st_mode & 0o077:
        parser.error("api-key-file must not be readable by group/other users (chmod 600)")
    key = key_file.read_text().strip()
    if not key or key == "EMPTY":
        parser.error("api-key-file must contain a nonempty key other than EMPTY")
    engine = TransformersEngine(args.model_path, args.max_context_tokens)
    app = create_app(engine, key)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False,
                limit_concurrency=8, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
