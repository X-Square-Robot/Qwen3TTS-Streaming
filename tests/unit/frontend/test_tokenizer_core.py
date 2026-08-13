"""
Unit tests for LightQwen3TTSTokenizer.
Run: pytest tests/unit/test_tokenizer_core.py -v
"""

import pytest

from tests.conftest import TOKENIZER_DIR, REPO_ROOT
from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer


@pytest.fixture(scope="module")
def tok():
    if not TOKENIZER_DIR.is_dir():
        pytest.skip(f"Tokenizer dir not found: {TOKENIZER_DIR}")
    return LightQwen3TTSTokenizer(str(TOKENIZER_DIR))


# ── construction ────────────────────────────────────────────────


class TestConstruction:
    def test_invalid_dir_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            LightQwen3TTSTokenizer("/nonexistent/path")

    def test_missing_vocab_raises(self, tmp_path):
        (tmp_path / "merges.txt").write_text("#version: 0.2\n")
        with pytest.raises(ValueError, match="Vocab file"):
            LightQwen3TTSTokenizer(str(tmp_path))

    def test_missing_merges_raises(self, tmp_path):
        (tmp_path / "vocab.json").write_text("{}")
        with pytest.raises(ValueError, match="Merges file"):
            LightQwen3TTSTokenizer(str(tmp_path))


# ── encode_ids ──────────────────────────────────────────────────


class TestEncodeIds:
    def test_returns_list_of_int(self, tok):
        ids = tok.encode_ids("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) >= 1

    def test_empty_string(self, tok):
        ids = tok.encode_ids("")
        assert isinstance(ids, list)
        assert len(ids) == 0


# ── encode_with_offsets ─────────────────────────────────────────


class TestEncodeWithOffsets:
    def test_offsets_cover_full_text(self, tok):
        text = "Hello, world!"
        ids, offsets = tok.encode_with_offsets(text)
        assert len(ids) == len(offsets)
        assert offsets[0][0] == 0
        assert offsets[-1][1] == len(text)

    def test_offsets_are_monotonic(self, tok):
        text = "The quick brown fox jumps over the lazy dog."
        _, offsets = tok.encode_with_offsets(text)
        for i in range(1, len(offsets)):
            assert offsets[i][0] >= offsets[i - 1][0]

    def test_offsets_no_gap_no_overlap(self, tok):
        text = "Hello, world!"
        _, offsets = tok.encode_with_offsets(text)
        for i in range(1, len(offsets)):
            assert offsets[i][0] == offsets[i - 1][1], (
                f"Gap/overlap between token {i - 1} and {i}: {offsets[i - 1]} vs {offsets[i]}"
            )

    def test_offset_slices_partition_mixed_text(self, tok):
        text = "你好，这是 token player + audio 并行流式测试。"
        _, offsets = tok.encode_with_offsets(text, add_special_tokens=False)
        assert "".join(text[start:end] for start, end in offsets) == text


# ── encode_with_tokens (BPE internal) ──────────────────────────


class TestEncodeWithTokens:
    def test_token_count_matches_ids(self, tok):
        ids, tokens = tok.encode_with_tokens("Hello, world!")
        assert len(ids) == len(tokens)


# ── encode_with_text (original spans) ──────────────────────────


class TestEncodeWithText:
    def test_spans_concatenate_to_original(self, tok):
        text = "Hello, world!"
        _, spans = tok.encode_with_text(text)
        assert "".join(spans) == text

    def test_no_bpe_artifacts(self, tok):
        """Spans must be plain text slices, no byte-level BPE chars like Ġ."""
        text = "Hello, world!"
        _, spans = tok.encode_with_text(text)
        for s in spans:
            assert "\u0120" not in s, f"BPE artifact Ġ found in span: {s!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "你好世界",
            "中英混合 test 123",
            "你好，这是 token player + audio 并行流式测试。",
            "Hello, how are you?",
            "  leading spaces",
            "trailing spaces  ",
            "multiple   spaces",
            "line\nbreak",
            "tab\there",
            "特殊符号：！@#￥%",
        ],
    )
    def test_spans_roundtrip_various(self, tok, text):
        _, spans = tok.encode_with_text(text)
        assert "".join(spans) == text


# ── decode roundtrip ────────────────────────────────────────────


class TestDecodeRoundtrip:
    @pytest.mark.parametrize(
        "text",
        [
            "Hello, world!",
            "你好世界",
            "中英混合 test 123",
            "  leading spaces",
            "trailing spaces  ",
            "line\nbreak",
        ],
    )
    def test_encode_decode_roundtrip(self, tok, text):
        ids = tok.encode_ids(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_decode_empty(self, tok):
        assert tok.decode([]) == ""


# ── special tokens ──────────────────────────────────────────────


class TestSpecialTokens:
    SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]

    def test_special_token_single_id(self, tok):
        """Each special token should encode as a single token id."""
        for st in self.SPECIAL_TOKENS:
            ids = tok.encode_ids(st)
            assert len(ids) == 1, f"{st!r} encoded as {len(ids)} tokens, expected 1"

    def test_chat_template_roundtrip(self, tok):
        text = "<|im_start|>assistant\n你好<|im_end|>"
        ids = tok.encode_ids(text)
        decoded = tok.decode(ids, skip_special_tokens=False)
        assert decoded == text

    def test_decode_skip_special_tokens(self, tok):
        text = "<|im_start|>assistant\n你好<|im_end|>"
        ids = tok.encode_ids(text)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert "<|im_start|>" not in decoded
        assert "<|im_end|>" not in decoded
        assert "你好" in decoded


# ── HF consistency (optional) ──────────────────────────────────


class TestHFConsistency:
    @pytest.fixture(scope="class")
    def hf_tok(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")
        hf_dir = (
            REPO_ROOT
            / "workspace"
            / "exported"
            / "tokenizer"
            / "Qwen3-TTS-Tokenizer-12Hz"
        )
        if not hf_dir.is_dir():
            pytest.skip(f"HF tokenizer dir not found: {hf_dir}")
        return AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)

    @pytest.mark.parametrize(
        "text",
        [
            "<|im_start|>assistant\n你好世界<|im_end|>",
            "<|im_start|>assistant\nHello, how are you?<|im_end|>",
            "中英混合 test 123",
        ],
    )
    def test_ids_match_hf(self, tok, hf_tok, text):
        light_ids = tok.encode_ids(text)
        hf_ids = hf_tok(text)["input_ids"]
        assert light_ids == hf_ids, f"Mismatch for: {text!r}"
