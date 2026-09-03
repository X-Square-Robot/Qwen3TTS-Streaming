from engine.frontend.text_commitment.committer import IncrementalTextCommitter
from engine.frontend.text_commitment.types import CommitKind, SpanKind
from engine.core.session import Session
from engine.core.types import SessionConfig
from engine.frontend.interface import FrontendInterface
import json


def test_percent_waits_for_suffix_and_commits_append_only():
    c = IncrementalTextCommitter()
    assert [x.tts_text for x in c.feed("是").commits] == ["是"]
    assert c.feed("99").commits == ()
    assert c.feed("%").commits == ()
    out = c.feed("的概率")
    assert [x.tts_text for x in out.commits] == ["百分之九十九", "的概率"]
    assert c.pending_raw == ""


def test_ordinal_and_word_wait_until_final():
    c = IncrementalTextCommitter()
    assert c.feed("21").commits == ()
    assert c.feed("st").commits == ()
    out = c.feed(" ")
    assert out.commits[0].span_kind in (SpanKind.ORDINAL, SpanKind.NUMBER)
    assert c.feed("inter").commits == ()
    assert c.feed("national", final=True).commits[0].tts_text == "international"


def test_timeout_creates_fence_and_late_suffix_is_new_literal():
    c = IncrementalTextCommitter()
    c.feed("99", now=0.0)
    timed = c.poll(now=1.0)
    assert timed.fallback and timed.events == ("text.fallback",)
    late = c.feed("%", now=1.1)
    assert "text.span.late_extension" in late.events


def test_json_final_projection_reads_scalar_values():
    c = IncrementalTextCommitter()
    out = c.feed('{"answer": 42, "ok": true}', final=True)
    assert out.commits[0].tts_text == "42 True"


def test_markdown_and_math_projection_do_not_leak_structure():
    c = IncrementalTextCommitter()
    out = c.feed("**标题** [正文](https://x.test)", final=True)
    assert "标题 正文" == "".join(x.tts_text for x in out.commits)
    c = IncrementalTextCommitter()
    out = c.feed("3*2=6", final=True)
    assert "三乘二等于六" == "".join(x.tts_text for x in out.commits)


def test_emoji_sequences_are_filtered_without_leaking_components():
    c = IncrementalTextCommitter()
    out = []
    for ch in "😀👩‍💻1️⃣":
        out.extend(x.tts_text for x in c.feed(ch).commits)
    out.extend(x.tts_text for x in c.feed("", final=True).commits)
    assert "".join(out) == ""


def test_code_fence_keeps_comments_and_string_literals_only():
    c = IncrementalTextCommitter()
    out = c.feed('```python\nx=1\n# hello\nprint("world")\n```', final=True)
    text = "".join(x.tts_text for x in out.commits)
    assert "# hello" in text and '"world"' in text
    assert "x=1" not in text and "print" not in text


def test_tn_commit_log_contains_raw_and_spoken_text(caplog):
    interface = FrontendInterface.__new__(FrontendInterface)
    session = Session("log", SessionConfig())
    commit = IncrementalTextCommitter().feed("是99%的", final=True).commits[0]
    with caplog.at_level("INFO", logger="engine.lifecycle"):
        interface._log_tn_commits(session, (commit,))
    payloads = [json.loads(record.message) for record in caplog.records if record.name == "engine.lifecycle"]
    assert payloads
    payload = payloads[-1]
    assert payload["phase"] == "text.tn_commit"
    assert payload["raw_text"]
    assert payload["spoken_text"]
