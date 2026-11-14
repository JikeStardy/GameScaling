# V100: GameScaling-0.8B-S3 with Transformers

This fork starts from upstream commit
`843a32078bd1cc7316388fb64433556f61075d26`. Its first deployment scope is a Linux
inference service compatible with the unmodified `GameScalingPolicy`. Windows
capture, input execution, and actual Zenless Zone Zero gameplay are a later
phase. The upstream Doom image is a protocol fixture, not a gameplay score.

## Runtime and model

Use a dedicated Python 3.12 environment. The deployment candidate uses PyTorch
2.8.0 (CUDA 12.8), torchvision 0.23.0 and Transformers 5.14.1. This differs from
the initial plan's candidate PyTorch 2.10/CUDA 12.6: the existing 2.8/CUDA 12.8
build was directly verified to include `sm_70` and execute FP16 CUDA matmul on
the target V100. The complete model request must still pass acceptance.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-server.txt
```

Download on the inference machine using ModelScope (it may use a separate
environment from serving):

```bash
MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=8 modelscope download \
  --model THUSI-Lab/GameScaling-0.8B-S3 \
  --local_dir /workspace/models/gamescaling/GameScaling-0.8B-S3
```

The parallel variable applies to the `modelscope_hub` backend used by ModelScope
1.38.1. Archive the download manifest and SHA256 values with deployment records;
the mutable `master` name alone is not a reproducible model revision.

The server explicitly loads FP16 model weights onto `cuda:0` and rejects CPU/disk
parameter placement. By default it uses eager full attention and the PyTorch
reference linear-attention path without compilation. Optional `fla`,
`causal-conv1d` and `flash-attn` packages must be absent. Reference kernels
perform some intermediate arithmetic in FP32; this is not an all-FP16
computation claim. The acceleration options below are explicit opt-ins.

## Start and use

Generate a private key outside the repository once:

```bash
mkdir -p /workspace/services/gamescaling-secrets
chmod 700 /workspace/services/gamescaling-secrets
python -c 'import secrets; print(secrets.token_urlsafe(32))' \
  > /workspace/services/gamescaling-secrets/api-key
chmod 600 /workspace/services/gamescaling-secrets/api-key
```

Do not overwrite the key on a routine restart. Activate the serving environment:

```bash
export GAMESCALING_API_KEY_FILE=/workspace/services/gamescaling-secrets/api-key
export GAMESCALING_HOST=127.0.0.1
export GAMESCALING_PORT=8000
bash scripts/serve_transformers.sh /workspace/models/gamescaling/GameScaling-0.8B-S3
```

The launcher exposes independent options for controlled comparisons. CLI options
are also available when invoking `python -m gamescaling.server` directly:

| CLI option | Launcher environment variable | Values / default |
| --- | --- | --- |
| `--attention-implementation` | `GAMESCALING_ATTENTION` | `eager` (default), `sdpa`, `vision-sdpa` |
| `--stop-implementation` | `GAMESCALING_STOP_IMPLEMENTATION` | `legacy` (default), `tensor` |
| `--vision-cache-mib` | `GAMESCALING_VISION_CACHE_MIB` | 0–512 MiB; 0 disables caching (default) |
| `--gdn-implementation` | `GAMESCALING_GDN_IMPLEMENTATION` | `reference` (default), `cuda` |
| `--prefill-implementation` | `GAMESCALING_PREFILL_IMPLEMENTATION` | `reference` (default), `solve` |
| `--decode-implementation` | `GAMESCALING_DECODE_IMPLEMENTATION` | `eager` (default), `cuda-graph` |
| `--kernel-build-directory` | `GAMESCALING_KERNEL_BUILD_DIRECTORY` | Optional CUDA extension build directory |

`vision-sdpa` selects SDPA only for vision attention; text full attention remains
eager and Gated DeltaNet (GDN) is selected separately. The visual cache keeps
bounded image features, keyed by the exact submitted image data and runtime
fingerprint. It preserves image order and still processes the complete prompt
and history on every request. It does not cache actions or cross-request KV.
`tensor` uses Transformers stop-string matching only for the official single
`<|action_end|>` stop; arbitrary strings and multiple stops retain the original
string matcher, with unchanged response and finish-reason semantics.

`gdn-implementation=cuda` compiles the supplied SM70 CUDA extension and fuses
only supported single-token recurrent steps; prefill remains a separate choice.
`prefill-implementation=solve` uses FP32 triangular solves for supported prefill
shapes. Unsupported shapes use the original reference path. Arithmetic order
can differ, so compare raw actions as well as latency, especially near token
ties. These flags do not establish gameplay quality.

`decode-implementation=cuda-graph` captures supported single-token forwards
while prefill and generation bookkeeping remain eager. It requests a static KV
cache with `disable_compile=True` and enters the graph context inside inference
mode. Its bounded graph cache reports captures, replays, bypasses and errors;
initial warmup/capture costs remain part of request timing. Unsupported inputs
bypass to eager before changing decode state. Capture failures restore the probe
state. Capture and replay errors propagate rather than silently retrying a
partially consumed step; the next generation must reset the failed cache.

For an explicit acceleration experiment, keep the same images, history, output
budget and FP16 model while selecting:

```bash
export GAMESCALING_ATTENTION=vision-sdpa
export GAMESCALING_VISION_CACHE_MIB=64
export GAMESCALING_GDN_IMPLEMENTATION=cuda
export GAMESCALING_PREFILL_IMPLEMENTATION=solve
export GAMESCALING_DECODE_IMPLEMENTATION=cuda-graph
bash scripts/serve_transformers.sh /workspace/models/gamescaling/GameScaling-0.8B-S3
```

This is a validation configuration, not an assumed faster default. Authenticated
`GET /v1/models` metadata reports actual attention backends, prefill/recurrent
kernel calls and fallbacks, image-cache counters, and graph statistics. Inspect
the counters to verify that the intended paths really execute; parameter
placement remains entirely on the GPU for every option.

One process owns the model. Keep one Uvicorn worker; a second worker would load
a second copy. The process can be run using `nohup` in the container with its
stdout/stderr redirected to a deployment log. Before stopping or restarting,
verify a saved PID against its command, start time and working directory.

For an existing LAN mapping, use its actual container port and bind address.
The target container's existing LAN `12388 -> 8888` mapping was confirmed with
a temporary nonce HTTP probe. Reusing that mapping requires
`GAMESCALING_HOST=0.0.0.0` and `GAMESCALING_PORT=8888`; the LAN API address then
becomes `http://<server-lan-ip>:12388/v1`. This needs no Docker or host changes.
No Windows client connectivity has been measured yet. For access over another
network, use the existing container SSH connection and forward port 8888.

Use the same private key in the client:

```python
from pathlib import Path
from PIL import Image
from gamescaling import GameScalingPolicy, PolicyConfig

config = PolicyConfig(
    base_url="http://127.0.0.1:8000/v1",
    api_key=Path("/path/to/private/api-key").read_text().strip(),
    reasoning=False,
    history_len=2,
    image_size=(854, 480),
    max_tokens=512,
)
prompt = Path("prompts/example_doom_battle1.txt").read_text()
policy = GameScalingPolicy(prompt, config)
decision = policy.act(Image.open("examples/doom_battle1.jpg"))
print(decision.text, decision.action.parsed, decision.latency_ms)
```

## Supported HTTP contract

- `GET /health`: readiness only, without credentials.
- `GET /v1/models`: Bearer authentication, model `gamescaling`, runtime/GPU
  placement metadata and allocator memory counters.
- `POST /v1/chat/completions`: the upstream client's non-streaming message
  format with inline JPEG/PNG data URIs, assistant prefill, temperature,
  `max_tokens`, stop strings, `skip_special_tokens=false`,
  `continue_final_message=true`, `add_generation_prompt=false`.

Unsupported fields return an explicit error. Remote image URLs are rejected.
The checkpoint's own processor/template renders the full history. Returned
content contains new output only, with the requested stop string removed;
the upstream client restores its prefill and action-end marker. Token budget
exhaustion returns `finish_reason=length`. The service reports token counts
and preprocessing/generation timing. It serializes inference and returns 503
for a concurrent generation request. There is no streaming or arbitrary queue.

Default limits: context plus reserved output 8192 tokens, output up to 2048
tokens, at most six images, 32 MiB request body, 8 MiB encoded-image payload
per decoded image, 4 million pixels per image. Limits produce explicit errors;
they do not silently trim history or downscale images. The example client's
explicit image resizing happens before submission.

## Acceptance and evidence

Offline contract tests require no torch or model:

```bash
python -m pytest tests -q
python examples/run_example.py --offline
```

Real GPU verification uses the unchanged official policy against the public
Doom screenshot, with 100 measured no-reasoning requests split across history
lengths 0/2/5, three reasoning requests and one warmup per configuration:

```bash
python tools/verify_transformers.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key-file "$GAMESCALING_API_KEY_FILE" \
  --output deployment-results/acceptance-001
```

Each raw HTTP reply and parsed action is retained in `requests.jsonl`;
`summary.json` reports errors, strict raw-action validity and p50/p95/max HTTP
latency by history/reasoning configuration. The validator checks original
mouse ranges, six key groups and key vocabulary before parser clamping or
dropped keys can hide malformed output. Never present a repaired or truncated
reply as a strictly valid action. Raw latency data includes rejected/error
requests. A single repeated image cannot establish game-playing ability.

Keep model hashes, code commit, full installed-package freeze, all-GPU
parameter metadata, load log, measured CPU/GPU memory, and the acceptance
output alongside each deployment. Generated evidence, keys, environments,
weights, logs and PID files are excluded from Git. Actual measurements are
added to a separate deployment report after acceptance, not assumed here.

## Device and service boundaries

The target ai-devel container has a 10 GiB CPU-memory limit and a 32 GiB V100.
The service uses GPU residency throughout inference; there is no weight
offloading, container-limit adjustment, driver change or host operation.
Previously stopped GLiNER, Laya, Kev and ASR services must remain stopped.
The Windows gameplay client and its performance targets are not part of this
Linux-only delivery.
