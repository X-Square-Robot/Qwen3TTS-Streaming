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


def test_closed_span_reuses_spoken_result_mapping_without_another_backend_pass():
    from engine.frontend.text_commitment.normalizer_backend import (
        MappedNormalization,
        NormalizationMapping,
    )
    from engine.frontend.text_commitment.wetext_backend import WetextAdapter

    class Backend:
        closed_calls = 0

        def normalize_closed(self, text, *, language, domain):
            self.closed_calls += 1
            return MappedNormalization(
                input_text=text,
                output_text="ABC",
                mappings=(
                    NormalizationMapping("replace", "word", 0, 1, 0, 1, "a", "A"),
                    NormalizationMapping("replace", "word", 1, 3, 1, 3, "bc", "BC"),
                ),
            )

        def normalize_with_mapping(self, *args, **kwargs):
            raise AssertionError("closed result already contains the mapping")

    backend = Backend()
    c = IncrementalTextCommitter(adapter=WetextAdapter(backend=backend))
    c.feed("甲a")
    c.feed("bc")
    assert backend.closed_calls == 0
    result = c.feed("", final=True)
    assert backend.closed_calls == 1
    word = next(commit for commit in result.commits if commit.raw_text == "abc")
    assert word.tts_text == "ABC"
    assert word.mapping == ((1, 2), (2, 4))
    assert word.output_mapping == ((1, 2, 0, 1), (2, 4, 1, 3))


def test_percent_waits_for_suffix_and_commits_append_only():
    c = IncrementalTextCommitter()
    assert [x.tts_text for x in c.feed("是").commits] == ["是"]
    assert c.feed("99").commits == ()
    assert c.feed("%").commits == ()
    out = c.feed("的概率")
    assert [x.tts_text for x in out.commits] == ["百分之九十九", "的概率"]
    assert c.pending_raw == ""


def test_single_digit_tail_waits_through_transport_delay_before_extension():
    """Emoji carry must not let a deadline split ``25`` into ``2`` + ``5``."""

    c = IncrementalTextCommitter()
    first = c.feed("今天温度2", now=0.0)
    assert [item.tts_text for item in first.commits] == ["今天温度"]
    assert c.pending_raw == "2"

    delayed = c.poll(now=1.0)
    assert delayed.commits == ()
    assert c.pending_raw == "2"

    c.feed("5", now=1.1)
    final = c.feed("", final=True, now=1.2)
    assert "".join(item.tts_text for item in final.commits) == "二十五"


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


def test_markdown_projection_carries_output_alignment():
    from engine.frontend.text_commitment.projector import project_readable_with_mapping

    value, mapping = project_readable_with_mapping("**标题**")
    assert value == "标题"
    assert mapping[-1][2:] == (1, 2)
    assert mapping[0][0:2] == (2, 3)


def test_ascii_and_unicode_multiplication_stay_in_one_formula_span():
    expected = "四乘六等于二十四"
    for raw in ("4x6=24", "4×6=24", "4 x 6 = 24"):
        result = IncrementalTextCommitter().feed(raw, final=True)
        assert len(result.commits) == 1
        assert result.commits[0].span_kind is SpanKind.MATH
        assert result.commits[0].tts_text == expected


def test_vulgar_fraction_is_normalized_as_a_number():
    result = IncrementalTextCommitter().feed("½", final=True)

    assert len(result.commits) == 1
    assert result.commits[0].span_kind is SpanKind.NUMBER
    assert result.commits[0].tts_text == "二分之一"


def test_standalone_math_and_temperature_symbols_are_spoken():
    assert "".join(item.tts_text for item in IncrementalTextCommitter().feed("≤ ≥ ℃", final=True).commits) == "小于等于 大于等于 摄氏度"


def test_chinese_calendar_date_uses_digit_by_digit_year():
    result = IncrementalTextCommitter().feed("1999年10月10日\u00a0  ", final=True)

    assert "".join(item.tts_text for item in result.commits).strip() == "一九九九年十月十日"


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


def test_inline_hash_is_literal_punctuation_and_does_not_open_pending_markdown():
    raw = "比如# 一级标题，后面继续"
    result = IncrementalTextCommitter().feed(raw, final=True)

    assert all(item.raw_text != "#" for item in result.commits)
    assert result.commits[-1].raw_end == len(raw)

    streamed = IncrementalTextCommitter()
    streamed.feed("比如#")
    split = streamed.feed(" 一级标题，后面继续", final=True)
    assert "".join(item.tts_text for item in split.commits) == "一级标题，后面继续"


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


def test_leading_decimal_is_atomic_across_packet_boundary():
    one_shot = IncrementalTextCommitter().feed("值为：.5", final=True)
    streamed = IncrementalTextCommitter()
    commits = list(streamed.feed("值为：.").commits)
    commits.extend(streamed.feed("5", final=True).commits)
    expected = "".join(item.tts_text for item in one_shot.commits)
    assert "".join(item.tts_text for item in commits) == expected
    assert any(item.raw_text == ".5" for item in commits)

    full_width = IncrementalTextCommitter()
    full_width_commits = list(full_width.feed("．").commits)
    full_width_commits.extend(full_width.feed("５", final=True).commits)
    assert any(item.raw_text == "．５" for item in full_width_commits)


def test_terminator_run_is_coalesced_across_packet_boundaries():
    streamed = IncrementalTextCommitter()
    commits = list(streamed.feed("你好!").commits)
    commits.extend(streamed.feed("?!", final=True).commits)
    assert "".join(item.tts_text for item in commits) == "你好!?!"
    assert any(item.raw_text == "?!" for item in commits)

    streamed = IncrementalTextCommitter()
    dots = list(streamed.feed(".").commits)
    dots.extend(streamed.feed("..", final=True).commits)
    assert any(item.raw_text == "..." for item in dots)


def test_newline_blocks_spaced_numeric_unit_lookahead():
    streamed = IncrementalTextCommitter()
    commits = list(streamed.feed("5\n").commits)
    commits.extend(streamed.feed("kg", final=True).commits)
    assert "".join(item.raw_text for item in commits) == "5\nkg"
    assert not any(item.raw_text == "5\nkg" for item in commits)

    crlf = IncrementalTextCommitter()
    crlf_commits = list(crlf.feed("5\r").commits)
    crlf_commits.extend(crlf.feed("\nkg", final=True).commits)
    assert not any(item.raw_text == "5\r\nkg" for item in crlf_commits)

    ordinary = IncrementalTextCommitter().feed("5 kg", final=True)
    assert any(item.raw_text == "5 kg" for item in ordinary.commits)


def test_inline_ordered_markers_are_suppressed_across_streaming_packets():
    """Compact model enumerations are lists even when they have no newlines.

    The common ``：1. ...。2. ...`` form used in generated prose must have the
    same result for one-shot and split transport input.  The dot is list
    formatting and must not be sent to the synthesizer; a decimal remains
    numeric because its next character is a digit.
    """

    raw = "以下几类：1. **日常家务**。2. **宠物照顾**。3. **信息查询**"
    expected = "以下几类：一、日常家务。二、宠物照顾。三、信息查询"

    one_shot = IncrementalTextCommitter().feed(raw, final=True)
    assert "".join(item.tts_text for item in one_shot.commits) == expected

    streamed = IncrementalTextCommitter()
    output: list[str] = []
    for chunk in ("以下几类：1.", " **日常家务**。2.", " **宠物照顾**。3.", " **信息查询**"):
        output.extend(item.tts_text for item in streamed.feed(chunk).commits)
    output.extend(item.tts_text for item in streamed.feed("", final=True).commits)
    assert "".join(output) == expected

    decimal = IncrementalTextCommitter().feed("版本 1.5", final=True)
    assert "".join(item.tts_text for item in decimal.commits) == "版本 一点五"


def test_inline_markdown_markers_in_chinese_prose_do_not_reach_cursor_labels():
    raw = "标题的话，比如# 一级标题，## 二级标题。比如- 项目一，1. 项目二。"

    result = IncrementalTextCommitter().feed(raw, final=True)

    assert "".join(item.tts_text for item in result.commits) == (
        "标题的话，比如一级标题，二级标题。比如项目一，项目二。"
    )


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


def test_chinese_decimal_currency_keeps_amount_together_and_uses_decimal_reading():
    """A dotted amount must not be reinterpreted as a month/day pair."""

    raw = "今天花了12.5元。"
    one_shot = IncrementalTextCommitter().feed(raw, final=True)
    expected = "今天花了十二点五元。"
    assert "".join(item.tts_text for item in one_shot.commits) == expected
    amount = next(item for item in one_shot.commits if item.raw_text == "12.5元")
    assert amount.span_kind is SpanKind.NUMBER

    streamed = IncrementalTextCommitter()
    commits = list(streamed.feed("今天花了12.").commits)
    commits.extend(streamed.feed("5元。", final=True).commits)
    assert "".join(item.tts_text for item in commits) == expected
    assert any(item.raw_text == "12.5元" for item in commits)


def test_phone_numbers_are_not_parsed_as_math_and_split_composite_calls():
    cases = {
        "010-1234-9876": "零一零一二三四九八七六",
        "+86-19023459876": "加八六一九零二三四五九八七六",
        "(+86)19123459876": "加八六一九一二三四五九八七六",
        "+86-19023459876/(+86)19123459876":
        "加八六一九零二三四五九八七六，加八六一九一二三四五九八七六",
    }
    for raw, expected in cases.items():
        result = IncrementalTextCommitter().feed(raw, final=True)
        assert "".join(item.tts_text for item in result.commits) == expected
        assert all(item.span_kind is SpanKind.PHONE for item in result.commits)

    streamed = IncrementalTextCommitter()
    output: list[str] = []
    raw = "+86-19023459876/(+86)19123459876"
    for chunk in ("+86-190", "23459876/", "(+86)", "19123459876"):
        output.extend(item.tts_text for item in streamed.feed(chunk).commits)
    output.extend(item.tts_text for item in streamed.feed("", final=True).commits)
    assert "".join(output) == cases[raw]


def test_order_id_uses_digit_sequence_and_html_entities_are_decoded():
    raw = "金额是188888元，订单号为188888号 &#x20;"
    result = IncrementalTextCommitter().feed(raw, final=True)
    output = "".join(item.tts_text for item in result.commits)
    assert output == "金额是十八万八千八百八十八元，订单号为一八八八八八号  "
    entity = next(item for item in result.commits if item.raw_text == "&#x20;")
    assert entity.tts_text == " "
    assert entity.mapping == ((raw.index("&#x20;"), raw.index("&#x20;") + 6),)


def test_sentence_head_number_waits_past_separator_for_script_context():
    """A leading ``99%`` must wait for later Chinese evidence."""

    streamed = IncrementalTextCommitter()
    first = streamed.feed("99%")
    assert first.commits == ()
    assert streamed.feed("，").commits == ()
    result = streamed.feed("我目前", final=True)
    assert "".join(item.tts_text for item in result.commits) == "百分之九十九，我目前"

    one_shot = IncrementalTextCommitter().feed("99%，我目前", final=True)
    assert "".join(item.tts_text for item in one_shot.commits) == "百分之九十九，我目前"


def test_url_projection_decodes_embedded_html_space_entities():
    for entity in ("&#x20;", "&#x20"):
        raw = f"访问http://example.com/{entity}然后继续"
        result = IncrementalTextCommitter().feed(raw, final=True)
        output = "".join(item.tts_text for item in result.commits)
        assert "and hash x twenty" not in output
        assert "HTTP colon slash slash example dot com slash" in output
        assert output.endswith("然后继续")


def test_badcase_long_mixed_text_is_packet_cadence_invariant():
    """The shipped TN badcase must not timeout spans between slow packets.

    LLM-PK sends short deltas; at the 90 ms preset a URL, phone number, or
    date can legitimately span several packets.  Compare the complete input
    with 4-character packets and explicit idle polls at all supported cadences
    so deadline handling cannot turn a suffix into literal ``#x20`` markup.
    """

    from pathlib import Path

    raw = (Path(__file__).parents[3] / "resources/dataset/badcase/tn_streaming_cases.txt").read_text().splitlines()[-1]

    def render(chunks, delay):
        committer = IncrementalTextCommitter()
        rendered: list[str] = []
        decisions = []
        now = 0.0
        for chunk in chunks:
            decision = committer.feed(chunk, now=now)
            decisions.append(decision)
            rendered.extend(item.tts_text for item in decision.commits)
            now += delay
            decision = committer.poll(now=now)
            decisions.append(decision)
            rendered.extend(item.tts_text for item in decision.commits)
        decision = committer.feed("", final=True, now=now)
        decisions.append(decision)
        rendered.extend(item.tts_text for item in decision.commits)
        return "".join(rendered), decisions

    expected, full_decisions = render([raw], 0.0)
    assert not any(decision.fallback for decision in full_decisions)
    for delay in (0.025, 0.045, 0.09):
        actual, decisions = render([raw[i : i + 4] for i in range(0, len(raw), 4)], delay)
        assert actual == expected
        assert not any(decision.fallback for decision in decisions)
        assert not any("text.span.late_extension" in decision.events for decision in decisions)
        assert "http://example.com/" not in actual
        assert "&#x20" not in actual


def test_id_card_numbers_are_spoken_digit_by_digit():
    cases = (
        "身份证43092220000315301X",
        "身份证430922200003152010",
    )
    expected = (
        "身份证四三零九二二二零零零零三一五三零一X",
        "身份证四三零九二二二零零零零三一五二零一零",
    )
    for raw, spoken in zip(cases, expected):
        result = IncrementalTextCommitter().feed(raw, final=True)
        assert "".join(item.tts_text for item in result.commits) == spoken
        assert any(item.span_kind is SpanKind.ID_CARD for item in result.commits)


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
    raw = "引擎版本号：0.2.1a1，模型版本号：researcher@20260818，引擎编译版本号：builder@20260902_580_5090_v1"
    expected = "引擎版本号：零点二点一a一，模型版本号：researcher艾特二零二六零八一八，引擎编译版本号：builder艾特二零二六零九零二下划线五八零下划线五零九零下划线v一"
    full = IncrementalTextCommitter().feed(raw, final=True)
    assert "".join(item.tts_text for item in full.commits) == expected
    assert [item.span_kind for item in full.commits if item.raw_text.startswith("0.")] == [SpanKind.VERSION]
    assert [item.span_kind for item in full.commits if item.raw_text.startswith("researcher")] == [SpanKind.IDENTIFIER]
    assert [item.span_kind for item in full.commits if item.raw_text.startswith("builder")] == [SpanKind.IDENTIFIER]

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
