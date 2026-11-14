"""Serial, fixed-input GPU experiments; separate profiling from latency trials."""
from __future__ import annotations

import argparse
import base64
import dataclasses
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from gamescaling import PolicyConfig, build_request, parse_action
from gamescaling.policy import assistant_turn, encode_image, user_turn
from gamescaling.server import CompletionRequest, TransformersEngine
from tools.verify_transformers import strict_action


def make_request(root: Path, history: int, sequence: int | None = None) -> CompletionRequest:
    cfg = PolicyConfig(reasoning=False, image_size=(854, 480), temperature=0.0,
                       max_tokens=512)
    image = encode_image(Image.open(root / "examples/doom_battle1.jpg"), cfg)
    def frame_url(index):
        if sequence is None:
            return image
        # Unique encoded frame identities with EXACTLY the same decoded pixels.
        # A JPEG comment simulates a new-frame miss, while h prior IDs are hits.
        # This is a controlled cache-workload simulation, not new gameplay data.
        header, encoded = image.split(",", 1)
        jpeg = base64.b64decode(encoded)
        assert jpeg[:2] == b"\xff\xd8"
        comment = f"gamescaling-frame-{index}".encode()
        data = jpeg[:2] + b"\xff\xfe" + (len(comment)+2).to_bytes(2, "big") + comment + jpeg[2:]
        return header + "," + base64.b64encode(data).decode()
    action = parse_action((root / "examples/doom_battle1_action.txt").read_text()).history_text
    messages = [{"role": "system", "content": (root / "prompts/example_doom_battle1.txt").read_text().rstrip("\n")}]
    for j in range(history):
        messages.extend([user_turn(frame_url((sequence or 0)-history+j)), assistant_turn(action)])
    messages.append(user_turn(frame_url(sequence or 0)))
    return CompletionRequest.model_validate(build_request(messages, cfg))


def instrument(engine, request):
    """CUDA events are diagnostic only; uninstrumented runs decide speed."""
    torch = engine.torch
    records = {"model": [], "vision": []}
    handles = []
    for name, module in (("model", engine.model), ("vision", engine.model.model.visual)):
        starts = []

        def pre(_module, args, kwargs, name=name, starts=starts):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            starts.append((event, time.perf_counter()))

        def post(_module, args, kwargs, result, name=name, starts=starts):
            start, cpu = starts.pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            records[name].append((start, end, (time.perf_counter()-cpu)*1000))

        handles.extend([module.register_forward_pre_hook(pre, with_kwargs=True),
                        module.register_forward_hook(post, with_kwargs=True)])
    try:
        result = engine.generate(request)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    model = records["model"]
    return {
        "result": dataclasses.asdict(result),
        "forward_calls": len(model),
        "prefill_gpu_event_ms": model[0][0].elapsed_time(model[0][1]),
        "prefill_host_dispatch_ms": model[0][2],
        "vision_gpu_event_ms": sum(a.elapsed_time(b) for a, b, _ in records["vision"]),
        "decode_gpu_event_sum_ms": sum(a.elapsed_time(b) for a, b, _ in model[1:]),
        "decode_host_dispatch_sum_ms": sum(c for _, _, c in model[1:]),
        "note": "vision is nested in prefill; events include GPU stream idle gaps; diagnostic run only",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--variants", nargs="+", default=["eager", "sdpa", "vision-sdpa", "text-sdpa"])
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--rolling", action="store_true",
                    help="Unique encoded frame IDs, h historical hits plus one fresh miss")
    ap.add_argument("--target-p50-ms", type=float,
                    help="Exit nonzero unless every measured case meets this target")
    args = ap.parse_args()
    if args.trials < 3:
        ap.error("at least three latency trials")
    if not args.variants or args.variants[0] != "eager":
        ap.error("eager must be the first variant for the paired reference")
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    engine = TransformersEngine(args.model_path)
    baselines = {}
    summary = {}
    prefill_solve = None
    decode_graph = None
    original_generate = engine.model.generate
    for variant in args.variants:
        if decode_graph is not None:
            decode_graph.restore()
            decode_graph = None
        engine.model.generate = original_generate
        if prefill_solve is not None:
            prefill_solve.restore()
            prefill_solve = None
        if engine._recurrent_kernel is not None:
            engine._recurrent_kernel.restore()
            engine._recurrent_kernel = None
        if engine._vision_cache is not None:
            engine._vision_cache.close()
            engine._vision_cache = None
        engine.stop_implementation = "legacy"
        engine.model.set_attn_implementation("eager")
        if variant == "sdpa":
            engine.model.set_attn_implementation("sdpa")
        elif variant == "vision-sdpa":
            engine.model.model.visual.set_attn_implementation("sdpa")
        elif variant == "text-sdpa":
            engine.model.model.language_model.set_attn_implementation("sdpa")
        elif variant.startswith("gdn"):
            from gamescaling.v100_gdn import install_recurrent_kernel
            engine._recurrent_kernel = install_recurrent_kernel(engine.model)
            if "vision" in variant:
                engine.model.model.visual.set_attn_implementation("sdpa")
            if "tensor" in variant:
                engine.stop_implementation = "tensor"
            if "cache" in variant:
                from gamescaling.vision_cache import VisionFeatureCache
                engine._vision_cache = VisionFeatureCache(engine.model.model)
                engine._vision_cache_backend = engine._cache_fingerprint()
            if "solve" in variant:
                from gamescaling.gdn_prefill import install_prefill_solve
                prefill_solve = install_prefill_solve(engine.model)
            if "graph" in variant:
                from gamescaling.cuda_graph import DecodeGraphInstallation
                decode_graph = DecodeGraphInstallation(engine.model)
                def graph_generate(*positional, **keywords):
                    with decode_graph.context():
                        return original_generate(*positional, cache_implementation="static",
                                                 disable_compile=True, **keywords)
                engine.model.generate = graph_generate
        elif variant != "eager":
            raise ValueError(variant)
        print(json.dumps({"variant": variant, "event": "started"}), flush=True)
        with (args.output / (variant + ".jsonl")).open("x") as out:
            for history in (0, 2, 5):
                # Independent request objects keep preprocessing work identical.
                for i in range(args.trials + 2):
                    request = make_request(root, history, 1000*(history+1)+i if args.rolling else None)
                    engine.torch.manual_seed(1729)
                    start = time.perf_counter()
                    result = engine.generate(request)
                    wall = (time.perf_counter()-start)*1000
                    text = PolicyConfig(reasoning=False).prefill + result.content
                    if result.matched_stop:
                        text += result.matched_stop
                    action = parse_action(text)
                    valid, reason = strict_action(text, result.finish_reason)
                    assert valid and action.parsed and not action.truncated and not action.dropped, reason
                    row = {"variant": variant, "history": history, "warmup": i < 2,
                           "wall_ms": wall, "result": dataclasses.asdict(result),
                           "vision_cache": engine._vision_cache.stats() if engine._vision_cache else None}
                    out.write(json.dumps(row) + "\n"); out.flush()
                rows = [json.loads(l) for l in (args.output / (variant + ".jsonl")).read_text().splitlines()]
                measured = [r for r in rows if r["history"] == history and not r["warmup"]]
                result_text = measured[0]["result"]["content"]
                if variant == "eager":
                    baselines[history] = result_text
                data = {"variant": variant, "history": history,
                        "p50_ms": statistics.median(r["wall_ms"] for r in measured),
                        "max_ms": max(r["wall_ms"] for r in measured),
                        "completion_tokens": [r["result"]["completion_tokens"] for r in measured],
                        "greedy_matches_eager": all(r["result"]["content"] == baselines.get(history) for r in measured),
                        "backend_config": {"vision": engine.model.model.visual.config._attn_implementation,
                                           "text": engine.model.model.language_model.config._attn_implementation},
                        "rolling_frame_ids": args.rolling,
                        "kernel_stats": engine._recurrent_kernel.statistics() if engine._recurrent_kernel else None,
                        "vision_cache": engine._vision_cache.stats() if engine._vision_cache else None}
                data["prefill_solve"] = prefill_solve.statistics() if prefill_solve else None
                data["decode_graph"] = decode_graph.statistics() if decode_graph else None
                if args.profile:
                    # Keep diagnostic probes on the same one-new-frame workload;
                    # they are separate from the uninstrumented latency trials.
                    sequence = 1000*(history+1)+args.trials+2 if args.rolling else None
                    data["profile"] = instrument(engine, make_request(root, history, sequence))
                    with engine.torch.profiler.profile(activities=[engine.torch.profiler.ProfilerActivity.CPU]) as p:
                        engine.generate(make_request(root, history, sequence+1 if sequence is not None else None))
                    data["sdpa_dispatch"] = [event.key for event in p.key_averages() if "scaled_dot_product" in event.key]
                summary[f"{variant}/h{history}"] = data
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(data), flush=True)
    if args.target_p50_ms is not None:
        return 0 if all(r["p50_ms"] <= args.target_p50_ms for r in summary.values()) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
