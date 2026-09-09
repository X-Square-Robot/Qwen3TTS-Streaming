"""Contract-level invariants for the incremental text commitment boundary.

These tests deliberately exercise the committer directly, including input
that arrives with ``final=True`` after an earlier streaming call.  The
committer's raw offsets are offsets into the original Unicode ``str``; they
must not silently become offsets into an emoji-filtered or projected view.
"""

from engine.frontend.text_commitment.committer import IncrementalTextCommitter
from engine.frontend.text_commitment.types import (
    CommitKind,
    LanguageKind,
    SpanKind,
    TextNormalizationConfig,
    TextInputMetadata,
)


def test_final_structured_delta_never_replays_a_committed_prefix():
    """A final structured suffix is still an append-only delta.

    ``final=True`` closes the current stream; it is not permission to project
    and emit the complete history a second time.  This is the regression case
    for a prefix such as ``hello `` followed by a Markdown span.
    """

    committer = IncrementalTextCommitter()
    prefix = committer.feed("hello ")
    prefix_text = "".join(commit.tts_text for commit in prefix.commits)
    prefix_end = max((commit.raw_end for commit in prefix.commits), default=0)
    assert prefix_text == "hello "

    suffix = committer.feed("**world**", final=True)
    suffix_text = "".join(commit.tts_text for commit in suffix.commits)

    assert suffix_text == "world"
    assert prefix_text + suffix_text == "hello world"
    assert all(commit.raw_start >= prefix_end for commit in suffix.commits)
    assert all(commit.raw_end > commit.raw_start for commit in suffix.commits)


def test_emoji_filter_keeps_original_raw_offsets_and_mapping():
    """Filtering an emoji must not shift the following raw span left.

    The emoji occupies one Unicode code-point position between the Chinese
    prefix and ``99%``.  The normalized number therefore starts at offset 2,
    even though its spoken text has no representation for the emoji.
    """

    raw = "是😀99%"
    decision = IncrementalTextCommitter().feed(raw, final=True)
    commits = decision.commits

    number = next(commit for commit in commits if commit.raw_text == "99%")
    assert number.span_kind is SpanKind.NUMBER
    assert number.raw_start == raw.index("99%") == 2
    assert number.raw_end == raw.index("99%") + len("99%") == 5
    assert number.mapping == ((2, 5),)

    # Every emitted interval remains in the original raw coordinate domain;
    # no commit may overlap or move backwards around the deleted grapheme.
    previous_end = 0
    for commit in commits:
        assert 0 <= commit.raw_start <= commit.raw_end <= len(raw)
        assert commit.raw_start >= previous_end
        previous_end = commit.raw_end


def test_compatibility_expansion_keeps_mapping_inside_original_raw_span():
    """Backend spellings such as ``℃`` must not escape source coordinates."""

    raw = "今天温度25℃"
    commits = IncrementalTextCommitter().feed(raw, final=True).commits
    number = next(commit for commit in commits if commit.raw_text == "25℃")

    assert number.raw_end == raw.index("25℃") + len("25℃")
    assert number.mapping == ((number.raw_start, number.raw_end),)


def test_unknown_language_waits_and_uses_literal_final_fallback_for_percent():
    """A bare percent span cannot be routed to a language-specific graph.

    With no surrounding script evidence, the mixed-language resolver keeps the
    span pending.  At finalization the deterministic safety fallback preserves
    the raw spelling and records UNKNOWN rather than guessing Chinese or
    English.
    """

    committer = IncrementalTextCommitter()
    pending = committer.feed("20%")
    assert pending.commits == ()
    assert pending.pending_raw == "20%"
    assert pending.pending_kind is SpanKind.NUMBER

    final = committer.feed("", final=True)
    assert len(final.commits) == 1
    commit = final.commits[0]
    assert commit.raw_text == "20%"
    assert commit.span_kind is SpanKind.NUMBER
    assert commit.language is LanguageKind.UNKNOWN
    assert commit.commit_kind is CommitKind.FALLBACK
    assert commit.tts_text == "20%"
    assert final.pending_raw == ""


def test_ordinal_suffix_routes_to_english_without_zh_misparse():
    """The complete ``st`` suffix is local English evidence.

    It must not enter the Chinese graph (which produces ``二十一秒t``), even
    when the ordinal is the first token in a mixed-language session.
    """

    committer = IncrementalTextCommitter()
    pending = committer.feed("21st")
    assert pending.commits == ()
    assert pending.pending_raw == "21st"
    assert pending.pending_kind is SpanKind.ORDINAL

    final = committer.feed("", final=True)
    assert len(final.commits) == 1
    commit = final.commits[0]
    assert commit.raw_text == "21st"
    assert commit.span_kind is SpanKind.ORDINAL
    assert commit.language is LanguageKind.EN
    assert commit.commit_kind is CommitKind.NORMALIZED
    assert commit.tts_text == "twenty first"
    assert commit.tts_text != "二十一秒t"
    assert final.pending_raw == ""


def test_compatibility_digits_and_operators_are_one_semantic_span():
    """Full-width model output must not fragment into per-digit literals."""

    raw = "３＊２＝６"
    final = IncrementalTextCommitter().feed(raw, final=True)
    assert len(final.commits) == 1
    commit = final.commits[0]
    assert commit.raw_text == raw
    assert commit.span_kind is SpanKind.MATH
    assert commit.raw_start == 0
    assert commit.raw_end == len(raw)
    assert commit.tts_text == "三乘二等于六"


def test_compatibility_date_is_not_misclassified_as_subtraction():
    raw = "２０２６－０７－２８"
    final = IncrementalTextCommitter().feed(raw, final=True)
    assert len(final.commits) == 1
    commit = final.commits[0]
    assert commit.span_kind is SpanKind.NUMBER
    assert commit.tts_text == "二零二六年七月二十八日"


def test_combining_mark_stays_with_latin_grapheme_across_packets():
    committer = IncrementalTextCommitter()
    output = []
    commits = []
    for chunk in ("e", "\u0301", " cafe"):
        update = committer.feed(chunk)
        commits.extend(update.commits)
        output.extend(item.tts_text for item in update.commits)
    update = committer.feed("", final=True)
    commits.extend(update.commits)
    output.extend(item.tts_text for item in update.commits)
    assert "".join(output) == "e\u0301 cafe"
    assert all(item.raw_text != "\u0301" for item in commits)


def test_latin_span_before_han_boundary_keeps_language_metadata():
    final = IncrementalTextCommitter().feed("é中", final=True)
    assert [item.language for item in final.commits] == [LanguageKind.EN, LanguageKind.ZH]


def test_final_flush_is_an_append_fence_for_late_transport_data():
    committer = IncrementalTextCommitter()
    first = committer.feed("99", final=True)
    assert first.state.value == "done"
    raw_before = committer.raw_text
    fence_before = committer.commit_fence

    late = committer.feed("%")
    assert late.commits == ()
    assert late.reason == "after_final"
    assert late.state.value == "done"
    assert "text.input.after_final" in late.events
    assert committer.raw_text == raw_before
    assert committer.commit_fence == fence_before


def test_language_hint_is_bound_to_an_open_span_across_packets():
    committer = IncrementalTextCommitter()
    first = committer.feed(
        "20",
        metadata=TextInputMetadata(language_hint=LanguageKind.EN),
    )
    assert first.pending_raw == "20"
    committer.feed("%")
    final = committer.feed("", final=True)
    assert final.commits[0].language is LanguageKind.EN
    assert final.commits[0].tts_text == "twenty percent"


def test_late_extension_literal_fence_is_scoped_to_one_new_span():
    """A timeout must not disable TN for unrelated later spans."""

    # Keep the timeout deterministic without sleeping in the test.
    committer = IncrementalTextCommitter(
        TextNormalizationConfig(semantic_max_wait_ms=10, semantic_idle_wait_ms=10)
    )
    committer.feed("99", now=0.0)
    timed = committer.poll(now=1.0)
    assert timed.fallback

    # A separating space means the next word is not a late suffix.  It should
    # retain ordinary structured/normalization behavior.
    decision = committer.feed(" **bold** 20%", final=True, now=1.1)
    assert "**bold**" not in "".join(item.tts_text for item in decision.commits)
    assert any(item.tts_text == "bold" for item in decision.commits)
    number = next(item for item in decision.commits if item.raw_text == "20%")
    assert number.commit_kind is not CommitKind.FALLBACK


def test_unconfirmed_markdown_is_literal_and_not_pronounced_as_punctuation():
    for raw in ("**bold", "~foo~", "#tag"):
        result = IncrementalTextCommitter().feed(raw, final=True)
        assert "".join(item.tts_text for item in result.commits) == raw
        assert all(item.commit_kind is CommitKind.FALLBACK for item in result.commits)


def test_late_structured_suffix_still_projects_readable_payload():
    """A contiguous late suffix gets a new fence without leaking syntax."""

    committer = IncrementalTextCommitter()
    committer.feed("99", now=0.0)
    committer.poll(now=1.0)
    result = committer.feed("**bold**", final=True, now=1.1)
    assert "".join(item.tts_text for item in result.commits) == "bold"
    assert result.commits[0].commit_kind is CommitKind.FALLBACK
