from __future__ import annotations

import builtins

import pytest

from engine.core.native_cursor_labelizer import (
    NativeCursorLabelizer,
    NativeCursorLabelizerError,
    load_native_cursor_labelizer,
)


def _vocab() -> dict[str, int]:
    return {
        "bai": 1,
        "fen": 2,
        "zhi": 3,
        "er": 4,
        "shi": 5,
        "san": 6,
        "en:a": 7,
        "en:b": 8,
    }


def test_labelizer_converts_chinese_and_english_with_model_ids() -> None:
    labelizer = NativeCursorLabelizer(_vocab())

    assert labelizer("百分之二十三 Ab") == (1, 2, 3, 4, 5, 6, 7, 8)


def test_labelizer_encodes_spans_for_chinese_and_english() -> None:
    labelizer = NativeCursorLabelizer(_vocab())

    ids, spans = labelizer.encode_with_spans("百分之二十三 Ab")

    assert ids == (1, 2, 3, 4, 5, 6, 7, 8)
    assert spans == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (7, 8),
        (8, 9),
    )
    assert len(ids) == len(spans)


def test_labelizer_ignores_whitespace_and_punctuation() -> None:
    labelizer = NativeCursorLabelizer({"ni": 1, "hao": 2})

    assert labelizer("你，好！ ") == (1, 2)


def test_labelizer_uses_phrase_level_polyphone_conversion() -> None:
    labelizer = NativeCursorLabelizer({"yin": 1, "hang": 2})

    assert labelizer("银行") == (1, 2)

    assert labelizer.encode_with_spans("银行") == ((1, 2), ((0, 1), (1, 2)))


def test_labelizer_encodes_empty_text_with_empty_spans() -> None:
    assert NativeCursorLabelizer({"en:a": 1}).encode_with_spans("") == ((), ())


def test_labelizer_spans_keep_punctuation_and_whitespace_gaps() -> None:
    labelizer = NativeCursorLabelizer({"en:a": 1, "en:b": 2})

    assert labelizer.encode_with_spans("a,  b!") == (
        (1, 2),
        ((0, 1), (4, 5)),
    )


@pytest.mark.parametrize("text", ["23", "🙂", "é"])
def test_labelizer_rejects_unspoken_or_unmapped_characters(text: str) -> None:
    labelizer = NativeCursorLabelizer({"er": 1})

    with pytest.raises(NativeCursorLabelizerError):
        labelizer(text)

    with pytest.raises(NativeCursorLabelizerError):
        labelizer.encode_with_spans(text)


def test_labelizer_ignores_math_and_unit_symbols():
    labelizer = NativeCursorLabelizer({"en:a": 1, "en:b": 2})

    assert labelizer.encode_with_spans("A ≥ B ℃") == ((1, 2), ((0, 1), (4, 5)))


@pytest.mark.parametrize(
    "vocab",
    [
        {"a": 0},
        {"a": 1, "b": 3},
        {"a": 1, "b": 1},
        {"a": True},
    ],
)
def test_labelizer_rejects_invalid_vocab(vocab) -> None:
    with pytest.raises(NativeCursorLabelizerError):
        NativeCursorLabelizer(vocab)


def test_checkpoint_mapping_and_fingerprint_are_stable() -> None:
    first = load_native_cursor_labelizer({"vocab": _vocab()})
    second = load_native_cursor_labelizer(
        {"vocab": dict(reversed(list(_vocab().items())))}
    )

    assert first.vocab_sha256 == second.vocab_sha256
    assert load_native_cursor_labelizer(
        {"vocab": _vocab()}, expected_vocab_sha256=first.vocab_sha256
    ).vocab_size == 8
    with pytest.raises(NativeCursorLabelizerError, match="fingerprint"):
        load_native_cursor_labelizer({"vocab": _vocab()}, expected_vocab_sha256="0" * 64)


def test_labelizer_reports_missing_pypinyin(monkeypatch) -> None:
    real_import = builtins.__import__

    def fail_pypinyin(name, *args, **kwargs):
        if name == "pypinyin":
            raise ImportError("missing test dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pypinyin)
    with pytest.raises(NativeCursorLabelizerError, match="pypinyin"):
        NativeCursorLabelizer({"ni": 1})("你")
