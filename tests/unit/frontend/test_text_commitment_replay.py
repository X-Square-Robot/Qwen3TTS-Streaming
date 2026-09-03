import random

from tools.text_commitment_replay import _chunks, _run


def test_token_replay_matches_single_shot_oracle():
    text = "是99%的概率，3*2=6，**正文**"
    streamed, commits, error = _run(text, _chunks(text, "token", random.Random(1)))
    full, _, full_error = _run(text, [text])
    assert error == full_error == ""
    assert streamed == full
    assert commits
