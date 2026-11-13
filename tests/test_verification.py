from gamescaling import PolicyConfig
from tools.verify_transformers import strict_action, validate_raw_reply


def test_strict_action_rejects_what_upstream_parser_can_repair():
    assert strict_action('<think></think><|action_start|>0 0 0 ; W ; ; ; ; ; <|action_end|>', 'stop')[0]
    for body in ('nan 0 0 ; ; ; ; ; ;', '1001 0 0 ; ; ; ; ; ;',
                 '0 0 0 ; W BAD ; ; ; ; ;', '0 0 0 ; A B C D E ; ; ; ; ;',
                 '0 0 0 ; ; ; ; ;'):
        assert not strict_action('<|action_start|>' + body + '<|action_end|>', 'stop')[0]


def test_truncated_or_unclosed_action_is_invalid():
    assert not strict_action('<think>unfinished', 'length')[0]
    assert not strict_action('<|action_start|>0 0 0 ; ; ; ; ; ;', 'stop')[0]
    assert not strict_action('<|action_start|>0 0 0 ; ; ; ; ; ; <|action_end|>junk', 'stop')[0]


def test_raw_validation_does_not_accept_upstream_bare_body_repair():
    cfg = PolicyConfig(reasoning=False)
    def reply(content, matched):
        return {'http_status': 200, 'body': {
            'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
            'metrics': {'matched_stop': matched}}}
    assert not validate_raw_reply(reply('0 0 0 ; W ; ; ; ; ; ', '<|action_end|>'), cfg)[0]
    action = '<|action_start|>0 0 0 ; W ; ; ; ; ; '
    assert not validate_raw_reply(reply(action, None), cfg)[0]
    assert validate_raw_reply(reply(action, '<|action_end|>'), cfg)[0]
    assert validate_raw_reply(reply(action + '<|action_end|>', None), cfg)[0]
