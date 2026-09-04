from engine.frontend.text_commitment.committer import IncrementalTextCommitter
from engine.frontend.text_commitment.types import (
    LanguageKind,
    SpanKind,
    TextNormalizationConfig,
)
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


def test_ascii_and_unicode_multiplication_stay_in_one_formula_span():
    expected = "四乘六等于二十四"
    for raw in ("4x6=24", "4×6=24", "4 x 6 = 24"):
        result = IncrementalTextCommitter().feed(raw, final=True)
        assert len(result.commits) == 1
        assert result.commits[0].span_kind is SpanKind.MATH
        assert result.commits[0].tts_text == expected


def test_formula_parentheses_are_kept_until_balanced():
    raw = "3 * (2 + 1) = 9"
    result = IncrementalTextCommitter().feed(raw, final=True)
    assert len(result.commits) == 1
    assert result.commits[0].span_kind is SpanKind.MATH
    assert result.commits[0].tts_text == "三乘左括号二加一右括号等于九"


def test_leading_grouped_formula_and_comparison_operators_are_readable():
    grouped = IncrementalTextCommitter().feed("(3+2)*4=20", final=True)
    assert "".join(item.tts_text for item in grouped.commits) == (
        "左括号三加二右括号乘四等于二十"
    )
    for raw, expected in (
        ("2>=1", "二大于等于一"),
        ("2<=1", "二小于等于一"),
        ("2!=1", "二不等于一"),
        ("2^3=8", "二的幂三等于八"),
    ):
        result = IncrementalTextCommitter().feed(raw, final=True)
        assert "".join(item.tts_text for item in result.commits) == expected

    markdown = IncrementalTextCommitter().feed("*bold*", final=True)
    assert "".join(item.tts_text for item in markdown.commits) == "bold"


def test_inline_markdown_delimiters_and_images_are_split_from_adjacent_words():
    cases = {
        "before**bold**after": "beforeboldafter",
        "before*italic*after": "beforeitalicafter",
        "before__bold__after": "beforeboldafter",
        "before~~strike~~after": "beforestrikeafter",
        "before![alt](https://example.test/image.png)after": "beforeafter",
        "* item": "item",
    }
    for raw, expected in cases.items():
        one_shot = IncrementalTextCommitter().feed(raw, final=True)
        assert "".join(item.tts_text for item in one_shot.commits) == expected

        # Every delimiter and body character may arrive in its own transport
        # packet.  The image/link URL must never be emitted as spoken text.
        streamed = IncrementalTextCommitter()
        output: list[str] = []
        for char in raw:
            output.extend(item.tts_text for item in streamed.feed(char).commits)
        output.extend(item.tts_text for item in streamed.feed("", final=True).commits)
        assert "".join(output) == expected


def test_math_star_is_not_consumed_as_markdown_formatting():
    for raw, expected in (
        ("3*2=6", "三乘二等于六"),
        ("3 * 2 = 6", "三乘二等于六"),
    ):
        result = IncrementalTextCommitter().feed(raw, final=True)
        assert len(result.commits) == 1
        assert result.commits[0].span_kind is SpanKind.MATH
        assert result.commits[0].tts_text == expected


def test_markdown_code_fence_closes_as_one_span():
    raw = '```python\n# comment\nprint("read me")\n```'
    result = IncrementalTextCommitter().feed(raw, final=True)
    assert len(result.commits) == 1
    assert result.commits[0].span_kind is SpanKind.MARKDOWN
    assert result.commits[0].tts_text == '# comment "read me"'


def test_hyphenated_numeric_range_is_not_arithmetic_without_context():
    for raw in ("3-2", "3 - 2"):
        result = IncrementalTextCommitter().feed(raw, final=True)
        assert len(result.commits) == 1
        assert result.commits[0].span_kind is SpanKind.NUMBER
        assert result.commits[0].tts_text == "三至二"

    arithmetic = IncrementalTextCommitter().feed("3-2=1", final=True)
    assert arithmetic.commits[0].span_kind is SpanKind.MATH
    assert arithmetic.commits[0].tts_text == "三减二等于一"


def test_markdown_line_prefix_markers_are_not_spoken():
    for raw, body in (("- item", "item"), ("+ item", "item"), ("> quote", "quote"), (">quote", "quote")):
        result = IncrementalTextCommitter().feed(raw, final=True)
        assert "".join(x.tts_text for x in result.commits) == body


def test_json_array_projects_scalar_values_and_survives_packet_splits():
    raw = '["answer", 42, true, {"nested": "ok"}]'
    one_shot = "".join(x.tts_text for x in IncrementalTextCommitter().feed(raw, final=True).commits)
    streamed = IncrementalTextCommitter()
    output: list[str] = []
    for ch in raw:
        output.extend(x.tts_text for x in streamed.feed(ch).commits)
    output.extend(x.tts_text for x in streamed.feed("", final=True).commits)
    assert one_shot == "answer 42 True ok"
    assert "".join(output) == one_shot
    assert all("[" not in value and "]" not in value for value in output)


def test_right_closing_delimiters_do_not_extend_numeric_or_formula_spans():
    for raw, expected in (
        ("99%)", "99%)"),
        ("99%）", "99%）"),
        ("3*2=6)", "三乘二等于六)"),
        ("3*2=6）", "三乘二等于六）"),
    ):
        result = IncrementalTextCommitter().feed(raw, final=True)
        assert "".join(item.tts_text for item in result.commits) == expected
        assert result.commits[-1].raw_text in (")", "）")


def test_ordered_markdown_list_marker_is_suppressed_but_decimal_survives():
    listed = IncrementalTextCommitter().feed("1. item", final=True)
    assert "".join(item.tts_text for item in listed.commits) == "item"

    listed_many_digits = IncrementalTextCommitter().feed("12. item", final=True)
    assert "".join(item.tts_text for item in listed_many_digits.commits) == "item"

    decimal = IncrementalTextCommitter().feed("1.5", final=True)
    assert "".join(item.tts_text for item in decimal.commits) == "1.5"


def test_structured_span_max_length_falls_back_and_does_not_pin_session():
    config = TextNormalizationConfig(max_pending_chars=4)
    committer = IncrementalTextCommitter(config)
    decision = committer.feed('["abcdef')
    assert decision.fallback is True
    assert "text.fallback" in decision.events
    assert committer.pending_raw == ""


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
