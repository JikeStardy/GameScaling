"""Run a serial GameScaling control loop against a visible Windows game window."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import sys
from threading import Thread
import time
from urllib.parse import urlsplit

from PIL import Image

from .action import ACTION_END, KEY_VOCAB, STEP_MS, parse_action, strict_action
from .policy import GameScalingPolicy, PolicyConfig, Proposal
from .windows_io import WindowsDesktop


class StopRun(RuntimeError):
    """The current session cannot safely execute another action."""


def strict_proposal(proposal: Proposal, config: PolicyConfig, allowed_keys: set[str]):
    result = proposal.decision
    if result.error:
        raise StopRun("Inference request failed. Check API address, credentials and service status.")
    text = config.prefill + proposal.raw_content
    if proposal.matched_stop == ACTION_END and ACTION_END in config.stop:
        text += ACTION_END
    valid, reason = strict_action(text, result.finish_reason)
    if not valid:
        raise StopRun(f"Rejected model action ({reason}).")
    action = parse_action(text)
    denied = action.pressed - (allowed_keys - {"Twelve"})
    if denied:
        raise StopRun(f"Model requested keys outside the enabled set: {', '.join(sorted(denied))}")
    return action


class ActionExecutor:
    def __init__(self, desktop: WindowsDesktop):
        self.desktop = desktop
        self.held: set[str] = set()

    def release(self):
        errors = []
        for token in sorted(self.held):
            try:
                self.desktop.key_up(token)
            except OSError as exc:
                errors.append(exc)
            else:
                self.held.remove(token)
        if errors:
            raise StopRun(f"Windows failed to release {len(errors)} input(s).") from errors[0]

    def execute(self, action, dimensions, check, wait):
        width, height = dimensions
        dx, dy, scroll = action.mouse
        try:
            check()
            if dx or dy:
                self.desktop.mouse_move(round(dx * width / 1000), round(dy * height / 1000))
            if round(scroll):
                check()
                self.desktop.scroll(round(scroll))
            for group in action.steps:
                check()
                desired = set(group)
                for token in sorted(self.held - desired):
                    self.desktop.key_up(token)
                    self.held.remove(token)
                for token in sorted(desired - self.held):
                    check()
                    self.desktop.key_down(token)
                    self.held.add(token)
                wait(STEP_MS / 1000)
        finally:
            self.release()


def request_proposal(policy, image, check, deadline=None):
    """Poll stop/focus while waiting. The worker can only generate a suggestion."""
    replies = SimpleQueue()

    def request():
        try:
            replies.put(policy.propose(image))
        except Exception as exc:
            replies.put(exc)

    check()
    Thread(target=request, daemon=True).start()
    while True:
        check()
        if deadline is not None and time.monotonic() >= deadline:
            raise StopRun("The screenshot is too old to use its response.")
        try:
            result = replies.get(timeout=0.01)
        except Empty:
            continue
        check()
        if deadline is not None and time.monotonic() >= deadline:
            raise StopRun("The screenshot is too old to use its response.")
        if isinstance(result, Exception):
            raise StopRun("Inference failed while reading the API response.") from result
        return result


def run_window(args, policy, allowed_keys):
    desktop = WindowsDesktop(args.window, process_name=args.process)
    executor = ActionExecutor(desktop)
    print(f"Target: {desktop.title} ({desktop.process_name or 'unknown process'})", flush=True)
    print("Switch to the game window. Press F12 to stop.", flush=True)
    focus_deadline = time.monotonic() + args.focus_timeout
    while not desktop.is_foreground():
        if desktop.stop_pressed() or time.monotonic() >= focus_deadline:
            raise StopRun("Stopped before the target window acquired focus.")
        time.sleep(0.01)
    deadline = time.monotonic() + args.seconds

    def check():
        if desktop.stop_pressed():
            raise StopRun("Stopped by F12.")
        if not desktop.is_foreground():
            raise StopRun("Stopped because the target window lost focus.")
        if time.monotonic() >= deadline:
            raise StopRun("Time limit reached.")

    def wait(delay):
        until = time.monotonic() + delay
        while True:
            check()
            remaining = until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.01, remaining))

    try:
        # Give the manually activated window a moment to finish switching.
        wait(0.2)
        for index in range(args.turns):
            check()
            captured_at = time.monotonic()
            image = desktop.capture()
            fresh_until = captured_at + args.max_age_ms / 1000 if args.execute else None
            proposal = request_proposal(policy, image, check, fresh_until)
            action = strict_proposal(proposal, policy.cfg, allowed_keys)
            age = (time.monotonic() - captured_at) * 1000
            if args.execute:
                if age >= args.max_age_ms or desktop.dimensions() != image.size:
                    raise StopRun("Screenshot expired or the game window was resized.")
                if any(action.mouse[:2]) and not args.mouse_look:
                    raise StopRun("Mouse movement is disabled. Enable it with --mouse-look.")
                if action.mouse[2] and not args.wheel:
                    raise StopRun("Scrolling is disabled. Enable it with --wheel.")
                executor.execute(action, image.size, check, wait)
                policy.remember(proposal)
            print(f"Turn {index + 1} {'executed' if args.execute else 'observed'} "
                  f"({age:.0f} ms): {action.to_text()}", flush=True)
    finally:
        executor.release()
        policy.reset()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--window", help="Unique visible window title or title substring")
    source.add_argument("--image", type=Path, help="Send one image file and print its action")
    parser.add_argument("--process", help="Optional executable basename, e.g. ZenlessZoneZero.exe")
    parser.add_argument("--prompt", required=True, type=Path, help="Completed GameScaling system prompt")
    parser.add_argument("--base-url", default=os.environ.get("GAMESCALING_BASE_URL", ""))
    parser.add_argument("--execute", action="store_true", help="Send validated inputs to the bound window")
    parser.add_argument("--keys", help="Comma-separated GameScaling key tokens enabled for execution")
    parser.add_argument("--mouse-look", action="store_true", help="Allow relative mouse movement")
    parser.add_argument("--wheel", action="store_true", help="Allow mouse wheel actions")
    parser.add_argument("--history", type=int, choices=range(6), default=2)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=15, help="Session time limit after focus")
    parser.add_argument("--focus-timeout", type=int, default=20)
    parser.add_argument("--max-age-ms", type=int, default=1500, help="Maximum age of a frame at execution start")
    args = parser.parse_args(argv)
    if any(value < 1 for value in (args.turns, args.seconds, args.focus_timeout, args.max_age_ms)):
        parser.error("turns, seconds, focus-timeout and max-age-ms must be positive")
    if args.execute and not args.window:
        parser.error("--execute requires --window")
    if args.execute and not args.keys:
        parser.error("--execute requires --keys")
    supported_keys = KEY_VOCAB - {"Twelve"}
    allowed_keys = {item.strip() for item in args.keys.split(",")} if args.keys else set(supported_keys)
    if not allowed_keys <= supported_keys:
        parser.error("--keys must use GameScaling tokens, and F12 (Twelve) is reserved")
    try:
        url = urlsplit(args.base_url)
    except ValueError:
        parser.error("Invalid API Base URL")
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment or url.path.rstrip("/") != "/v1":
        parser.error("Set GAMESCALING_BASE_URL to the GameScaling API URL ending in /v1")
    key = os.environ.get("GAMESCALING_API_KEY", "").strip()
    if not key or key == "EMPTY":
        parser.error("Set GAMESCALING_API_KEY for the current PowerShell session")
    try:
        prompt = args.prompt.read_text(encoding="utf-8-sig").strip()
        if not prompt or any(mark in prompt for mark in ("{GAME_NAME}", "{GAME_RULES}", "{KEY_BINDINGS}")):
            parser.error("Fill the game name, rules and key bindings in the prompt")
        config = PolicyConfig(base_url=args.base_url, api_key=key, image_size=(854, 480),
                              history_len=args.history, reasoning=False, max_tokens=512, timeout=10)
        policy = GameScalingPolicy(prompt, config)
        if args.image:
            with Image.open(args.image) as source_image:
                proposal = policy.propose(source_image.convert("RGB"))
            action = strict_proposal(proposal, policy.cfg, allowed_keys)
            print(f"Observed ({proposal.decision.latency_ms:.0f} ms): {action.to_text()}")
        else:
            run_window(args, policy, allowed_keys)
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
