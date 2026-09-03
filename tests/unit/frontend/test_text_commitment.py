from engine.frontend.text_commitment.committer import IncrementalTextCommitter
from engine.frontend.text_commitment.types import LanguageKind, SpanKind
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


def test_structured_numeric_fallbacks_cover_decimal_units_currency_and_inequality():
    cases = {
        "1/2=0.5": {"一除以二等于零点五"},
        # wetext uses the natural Chinese measure ordering; the fallback uses
        # the equally readable number-first form when wetext is unavailable.
        "7μg/m³": {"每立方米七微克", "七微克每立方米"},
        "¥10.09": {"十点零九元", "十点零九人民币"},
        "5 > 3": {"五大于三"},
        "-5°C": {"负五摄氏度"},
        "PM2.5": {"PM二点五", "PM two.five"},
        "B-0109": {"B杠零一零九", "B-oh one oh nine"},
    }
    for raw, expected in cases.items():
        output = "".join(item.tts_text for item in IncrementalTextCommitter().feed(raw, final=True).commits)
        assert output in expected, (raw, output)


def test_numeric_spans_follow_english_context():
    c = IncrementalTextCommitter()
    output = "".join(item.tts_text for item in c.feed("20% of 100 is 20", final=True).commits)
    assert output == "twenty percent of one hundred is twenty"


def test_numeric_spans_follow_chinese_script_context():
    cases = {
        "概率是20%": "概率是百分之二十",
        "20%概率": "百分之二十概率",
        "the chance is 20%": "the chance is twenty percent",
    }
    for raw, expected in cases.items():
        output = "".join(item.tts_text for item in IncrementalTextCommitter().feed(raw, final=True).commits)
        assert output == expected, (raw, output)


def test_commit_records_selected_language_route():
    english = IncrementalTextCommitter().feed("20% of 100 is 20", final=True).commits
    chinese = IncrementalTextCommitter().feed("概率是20%", final=True).commits
    assert any(item.language == LanguageKind.EN for item in english)
    assert any(item.language == LanguageKind.ZH for item in chinese)


def test_version_and_build_identifiers_are_not_sent_to_generic_wetext_rules():
    raw = "引擎版本号：0.2.1a1，模型版本号：zehan@20260818，引擎编译版本号：rime@20260902_580_5090_v1"
    expected = "引擎版本号：零点二点一a一，模型版本号：zehan艾特二零二六零八一八，引擎编译版本号：rime艾特二零二六零九零二下划线五八零下划线五零九零下划线v一"
    full = IncrementalTextCommitter().feed(raw, final=True)
    assert "".join(item.tts_text for item in full.commits) == expected
    assert [item.span_kind for item in full.commits if item.raw_text.startswith("0.")] == [SpanKind.VERSION]
    assert [item.span_kind for item in full.commits if item.raw_text.startswith("zehan")] == [SpanKind.IDENTIFIER]
    assert [item.span_kind for item in full.commits if item.raw_text.startswith("rime")] == [SpanKind.IDENTIFIER]

    streamed = IncrementalTextCommitter()
    output = []
    for chunk in (raw[:16], raw[16:31], raw[31:49], raw[49:]):
        output.extend(item.tts_text for item in streamed.feed(chunk).commits)
    output.extend(item.tts_text for item in streamed.feed("", final=True).commits)
    assert "".join(output) == expected


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
    assert payload["language"] == "zh"
