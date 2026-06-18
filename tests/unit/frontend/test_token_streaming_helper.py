from __future__ import annotations

import pytest

from tests.support.token_streaming import (
    _build_reencodable_text_chunks,
    validate_token_text_chunks,
)


def test_validate_token_text_chunks_accepts_exact_roundtrip():
    tokenized = validate_token_text_chunks(
        "你好。",
        [10, 11, 12],
        ["你", "好", "。"],
    )

    assert tokenized.text == "你好。"
    assert tokenized.chunks == ["你", "好", "。"]
    assert tokenized.token_ids == [10, 11, 12]
    assert tokenized.chunk_token_ids == [[10], [11], [12]]
    assert tokenized.chunk_count == 3


def test_validate_token_text_chunks_accepts_grouped_byte_level_tokens():
    tokenized = validate_token_text_chunks(
        "你好。",
        [10, 11, 12, 13],
        ["你", "好", "。"],
        chunk_token_ids=[[10, 11], [12], [13]],
    )

    assert tokenized.chunks == ["你", "好", "。"]
    assert tokenized.chunk_token_ids == [[10, 11], [12], [13]]
    assert tokenized.max_chunk_tokens == 2


def test_validate_token_text_chunks_rejects_roundtrip_mismatch():
    with pytest.raises(ValueError, match="do not reconstruct"):
        validate_token_text_chunks(
            "你好。",
            [10, 11],
            ["你", "好"],
        )


def test_validate_token_text_chunks_rejects_empty_chunks():
    with pytest.raises(ValueError, match="empty text chunks"):
        validate_token_text_chunks(
            "你好",
            [10, 11, 12],
            ["你", "", "好"],
        )


def test_validate_token_text_chunks_rejects_bad_grouped_token_ids():
    with pytest.raises(ValueError, match="chunk token ids do not reconstruct"):
        validate_token_text_chunks(
            "你好",
            [10, 11, 12],
            ["你", "好"],
            chunk_token_ids=[[10], [12]],
        )


def test_build_reencodable_text_chunks_groups_visible_char_continuations():
    class _FakeTokenizer:
        def encode_ids(self, text, add_special_tokens=False):
            mapping = {
                "你": [10, 11],
                "好": [12],
                "。": [13],
                "你好": [10, 11, 12],
                "好。": [12, 13],
                "你好。": [10, 11, 12, 13],
            }
            return mapping[text]

    chunks, chunk_token_ids = _build_reencodable_text_chunks(
        "你好。",
        [10, 11, 12, 13],
        _FakeTokenizer(),
    )

    assert chunks == ["你", "好", "。"]
    assert chunk_token_ids == [[10, 11], [12], [13]]
