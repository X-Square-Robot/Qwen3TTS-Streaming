"""Regression contracts for local language routing in mixed-mode TN."""

from engine.frontend.text_commitment import IncrementalTextCommitter
from engine.frontend.text_commitment.types import LanguageKind, SpanKind


def test_embedded_percent_after_english_word_uses_local_english_route():
    """A glued ``English20%`` suffix must not inherit an earlier Han span.

    The lexer is allowed to keep ordinary model identifiers (``PM2.5``,
    ``B2B``) intact.  A trailing percent marker is the narrow semantic
    boundary that lets the committer split the word and numeric suffix before
    either crosses the append-only fence.
    """

    result = IncrementalTextCommitter().feed("中文99% English20%", final=True)

    assert [(item.raw_text, item.language, item.span_kind) for item in result.commits] == [
        ("中文", LanguageKind.ZH, SpanKind.PLAIN),
        ("99%", LanguageKind.ZH, SpanKind.NUMBER),
        (" ", LanguageKind.ZH, SpanKind.PLAIN),
        ("English", LanguageKind.EN, SpanKind.ENGLISH_WORD),
        ("20%", LanguageKind.EN, SpanKind.NUMBER),
    ]
    assert "".join(item.tts_text for item in result.commits) == (
        "中文百分之九十九 Englishtwenty percent"
    )
