"""Exercise the unmodified upstream client against a real GPU service.

This replays a single public Doom frame. It measures protocol, placement and
latency, not game-playing ability or a gameplay benchmark score.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import requests

from gamescaling import GameScalingPolicy, PolicyConfig
from gamescaling.action import ACTION_END, strict_action


def summarize(rows: list[dict]) -> dict:
    latencies = sorted(r["decision"]["latency_ms"] for r in rows)
    return {
        "requests": len(rows),
        "http_errors": sum(bool(r["decision"]["error"]) for r in rows),
        "parsed": sum(r["decision"]["action"]["parsed"] for r in rows),
        "strict_valid": sum(r["strict_valid"] for r in rows),
        "latency_ms": {
            "p50": statistics.median(latencies),
            "p95_nearest_rank": latencies[math.ceil(len(latencies) * .95) - 1],
            "max": max(latencies),
        },
    }


def validate_raw_reply(reply: dict, cfg: PolicyConfig) -> tuple[bool, str]:
    """Restore only documented protocol omissions, never parser repairs."""
    if reply.get("http_status") != 200:
        return False, "http_error"
    payload = reply.get("body", {})
    choices = payload.get("choices") or []
    if not choices:
        return False, "missing_choice"
    choice = choices[0]
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str):
        return False, "missing_content"
    text = cfg.prefill + content
    # Unlike upstream finalize(), do not insert a missing ACTION_START or
    # presume that every EOS stop removed an action-end tag.
    matched = payload.get("metrics", {}).get("matched_stop")
    if matched == ACTION_END and ACTION_END in cfg.stop:
        text += ACTION_END
    return strict_action(text, choice.get("finish_reason"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=100,
                    help="Measured no-reasoning requests, distributed over histories 0/2/5")
    ap.add_argument("--reasoning-trials", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()
    if args.trials < 3 or args.reasoning_trials < 1:
        ap.error("trials must be >=3 and reasoning-trials >=1")
    root = Path(__file__).resolve().parents[1]
    image_path = root / "examples/doom_battle1.jpg"
    prompt = (root / "prompts/example_doom_battle1.txt").read_text().rstrip("\n")
    key = args.api_key_file.read_text().strip()
    if not key:
        ap.error("empty API key file")
    headers = {"Authorization": f"Bearer {key}"}
    args.output.mkdir(parents=True, exist_ok=True)
    frame = Image.open(image_path).convert("RGB")
    model_response = requests.get(args.base_url.rstrip("/") + "/models",
                                  headers=headers, timeout=30)
    model_response.raise_for_status()
    environment = {
        "started_unix": time.time(), "models_before": model_response.json(),
        "fixture": "upstream examples/doom_battle1.jpg",
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "image_size_sent": [854, 480], "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "purpose": "single-fixture protocol/latency test; no game effect score",
    }
    rows: list[dict] = []
    raw_path = args.output / "requests.jsonl"
    # Never overwrite a previous run's raw evidence.
    with raw_path.open("x") as out:
        configs = [(False, h, args.trials // 3 + (i < args.trials % 3))
                   for i, h in enumerate((0, 2, 5))]
        configs.append((True, 0, args.reasoning_trials))
        for reasoning, history, count in configs:
            cfg = PolicyConfig(base_url=args.base_url, api_key=key,
                               history_len=history, reasoning=reasoning,
                               image_size=(854, 480), max_tokens=args.max_tokens,
                               temperature=args.temperature, timeout=args.timeout)
            policy = GameScalingPolicy(prompt, cfg)
            captured: list[dict] = []
            accepted_history = 0

            def capture(response, *unused, **kwargs):
                try:
                    captured.append({"http_status": response.status_code,
                                     "body": response.json()})
                except ValueError:
                    captured.append({"http_status": response.status_code,
                                     "body_text": response.text[:20000]})

            policy._session.hooks["response"].append(capture)
            try:
                for i in range(count + 1):
                    # One distinct warmup per configuration, then a new history.
                    warmup = i == 0
                    if i <= 1 or (i - 1) % 6 == 0:
                        policy.reset()
                        accepted_history = 0
                    history_before = accepted_history
                    captured.clear()
                    d = policy.act(frame)
                    valid, reason = (validate_raw_reply(captured[-1], cfg)
                                     if captured else (False, "no_http_response"))
                    row = {"id": len(rows), "reasoning": reasoning,
                           "history_len": history, "history_before": history_before,
                           "warmup": warmup, "decision": dataclasses.asdict(d),
                           "raw_http": list(captured), "strict_valid": valid and not d.error,
                           "validation_reason": reason}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    rows.append(row)
                    if not valid or d.error:
                        # A failed action is not real game history. Start afresh.
                        policy.reset()
                        accepted_history = 0
                    else:
                        accepted_history = min(history, accepted_history + 1)
                    print(json.dumps({"request": row["id"], "warmup": warmup,
                                      "history": history, "reasoning": reasoning,
                                      "strict_valid": row["strict_valid"],
                                      "latency_ms": d.latency_ms}), flush=True)
            finally:
                policy._session.close()
    measured = [r for r in rows if not r["warmup"]]
    groups = {}
    for reasoning, history, _ in configs:
        selected = [r for r in measured if r["reasoning"] == reasoning
                    and r["history_len"] == history]
        groups[f"reasoning={reasoning},history={history}"] = summarize(selected)
    final = requests.get(args.base_url.rstrip("/") + "/models", headers=headers, timeout=30)
    final.raise_for_status()
    environment.update({"completed_unix": time.time(), "models_after": final.json(),
                        "groups": groups, "summary": summarize(measured)})
    (args.output / "summary.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False))
    print(json.dumps(environment["summary"], indent=2))
    return 0 if all(r["strict_valid"] for r in measured) else 1


if __name__ == "__main__":
    raise SystemExit(main())
