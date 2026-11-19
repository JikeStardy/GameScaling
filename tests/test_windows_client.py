from PIL import Image
import pytest

from gamescaling.action import ACTION_END, ACTION_START
from gamescaling.policy import GameScalingPolicy, PolicyConfig
from gamescaling.windows_client import ActionExecutor, StopRun, strict_proposal


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
