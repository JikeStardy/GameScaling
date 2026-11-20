from PIL import Image
import pytest

from gamescaling.action import ACTION_END, ACTION_START, Action
from gamescaling.policy import GameScalingPolicy, PolicyConfig
from gamescaling import windows_client
from gamescaling.windows_client import ActionExecutor, StopRun, strict_proposal
from gamescaling.windows_io import _keyboard_input


class Desktop:
    def __init__(self):
        self.events = []
        self.focused = True

    def is_foreground(self):
        return self.focused

    def key_down(self, key):
        self.events.append(("down", key))

    def key_up(self, key):
        self.events.append(("up", key))

    def mouse_move(self, dx, dy):
        self.events.append(("move", dx, dy))

    def scroll(self, notches):
        self.events.append(("scroll", notches))


def test_interrupted_chunk_releases_held_keys_after_focus_loss():
    desktop = Desktop()
    executor = ActionExecutor(desktop)
    from gamescaling.action import parse_action

    def wait(_delay):
        desktop.focused = False
        raise StopRun("window lost focus")

    with pytest.raises(StopRun):
        executor.execute(parse_action(f"{ACTION_START}0 0 0 ; W ; W ; ; ; ; {ACTION_END}"),
                         (1920, 1080), lambda: None, wait)
    assert desktop.events == [("down", "W"), ("up", "W")]


@pytest.mark.parametrize("content,matched,finish", [
    (f"{ACTION_START}nan 0 0 ; W ; ; ; ; ; ", ACTION_END, "stop"),
    (f"{ACTION_START}0 0 0 ; W ; ; ; ; ; ", None, "stop"),
    (f"{ACTION_START}0 0 0 ; W ; ; ; ; ; ", ACTION_END, "length"),
    (f"{ACTION_START}0 0 0 ; Twelve ; ; ; ; ; ", ACTION_END, "stop"),
])
def test_malformed_or_reserved_output_is_never_eligible_for_execution(monkeypatch, content, matched, finish):
    policy = GameScalingPolicy("prompt", PolicyConfig(reasoning=False))

    class Reply:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
                    "metrics": {"matched_stop": matched}}

    monkeypatch.setattr(policy._session, "post", lambda *a, **k: Reply())
    proposal = policy.propose(Image.new("RGB", (16, 9)))
    with pytest.raises(StopRun):
        strict_proposal(proposal, policy.cfg, {"W"})
    assert len(policy.messages("new")) == 2


def test_only_committed_proposal_enters_history_and_keys_are_held_between_steps(monkeypatch):
    policy = GameScalingPolicy("prompt", PolicyConfig(reasoning=False))

    class Reply:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": f"{ACTION_START}5 0 0 ; W ; W ; ; ; ; "},
                                 "finish_reason": "stop"}],
                    "metrics": {"matched_stop": ACTION_END}}

    monkeypatch.setattr(policy._session, "post", lambda *a, **k: Reply())
    proposal = policy.propose(Image.new("RGB", (16, 9)))
    assert len(policy.messages("new")) == 2
    desktop = Desktop()
    action = strict_proposal(proposal, policy.cfg, {"W"})
    ActionExecutor(desktop).execute(action, (1920, 1080), lambda: None, lambda _: None)
    assert desktop.events == [("move", 10, 0), ("down", "W"), ("up", "W")]
    policy.remember(proposal)
    assert len(policy.messages("new")) == 4


def test_probe_sends_one_200ms_chunk_without_server_credentials(monkeypatch, capsys):
    class ProbeDesktop(Desktop):
        title = "Training window"
        process_name = "game.exe"

        def stop_pressed(self):
            return False

        def dimensions(self):
            return (1920, 1080)

    desktop = ProbeDesktop()
    modes = []

    def bind(_title, *, process_name, keyboard_mode):
        modes.append((process_name, keyboard_mode))
        return desktop

    monkeypatch.setattr(windows_client, "WindowsDesktop", bind)
    monkeypatch.delenv("GAMESCALING_API_KEY", raising=False)
    monkeypatch.delenv("GAMESCALING_BASE_URL", raising=False)
    assert windows_client.main([
        "--window", "Training", "--process", "game.exe", "--execute",
        "--probe-key", "S", "--keyboard-mode", "virtual-key",
    ]) == 0
    assert modes == [("game.exe", "virtual-key")]
    assert desktop.events == [("down", "S"), ("up", "S")]
    assert "Probe sent: S held for 200 ms" in capsys.readouterr().out


@pytest.mark.parametrize("mode,expected", [
    ("scan-code", (0, 0x1f, 0x8, 0xA)),
    ("virtual-key", (0x53, 0, 0, 0x2)),
])
def test_probe_key_encodes_standard_win32_down_and_up_records(mode, expected):
    down = _keyboard_input("S", False, mode).ki
    up = _keyboard_input("S", True, mode).ki
    assert (down.wVk, down.wScan, down.dwFlags, up.dwFlags) == expected
    assert (up.wVk, up.wScan) == (down.wVk, down.wScan)


def test_subpixel_motion_and_fractional_scroll_do_not_report_sent():
    desktop = Desktop()
    action = Action(mouse=(0.01, -0.01, 0.1), parsed=True)
    assert ActionExecutor(desktop).execute(action, (1920, 1080), lambda: None, lambda _: None) is False
    assert desktop.events == []
