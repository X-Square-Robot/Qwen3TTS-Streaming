"""
L1 unit tests: lightweight_tokenizer (T1.1, T4.1 tokenizer consistency).
Run from repo root: pytest tests/unit/test_lightweight_tokenizer.py -v
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOKENIZER_DIR = REPO_ROOT / "workspace" / "exported" / "tokenizer" / "Qwen3-TTS-Tokenizer-12Hz"


@pytest.fixture(scope="module")
def tokenizer_dir():
    return str(TOKENIZER_DIR)


@pytest.fixture(scope="module")
def lightweight_tok(tokenizer_dir):
    if not Path(tokenizer_dir).is_dir():
        pytest.skip(f"Tokenizer dir not found: {tokenizer_dir} (run export/download first)")
    from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
    tok = load_lightweight_tokenizer(tokenizer_dir)
    if tok is None:
        pytest.skip("load_lightweight_tokenizer returned None (tokenizer.json or vocab+merges missing)")
    return tok


def test_load_lightweight_tokenizer_output_type(lightweight_tok):
    """T1.1: output is numpy [1, S] int64 when return_tensors='pt'."""
    import numpy as np
    result = lightweight_tok("<|im_start|>assistant\n你好<|im_end|>", return_tensors="pt")
    ids = result["input_ids"]
    assert isinstance(ids, np.ndarray), f"Expected numpy, got {type(ids)}"
    assert ids.ndim == 2, f"Expected 2D, got {ids.ndim}D"
    assert ids.dtype == np.int64
    assert ids.shape[0] == 1
    assert ids.shape[1] > 3  # at least <|im_start|> assistant \n


def test_lightweight_tokenizer_without_return_tensors(lightweight_tok):
    """Without return_tensors='pt', input_ids can be list (API compatibility)."""
    result = lightweight_tok("hello")
    assert "input_ids" in result
    ids = result["input_ids"]
    assert ids is not None
    if hasattr(ids, "__len__"):
        assert len(ids) >= 1


def test_lightweight_tokenizer_multiple_texts(lightweight_tok):
    """Multiple prompts tokenize without error."""
    import numpy as np
    texts = [
        "<|im_start|>assistant\n你好世界<|im_end|>",
        "<|im_start|>assistant\nHello, how are you?<|im_end|>",
        "<|im_start|>assistant\n中英混合 test 123<|im_end|>",
    ]
    for text in texts:
        result = lightweight_tok(text, return_tensors="pt")
        ids = result["input_ids"]
        assert isinstance(ids, np.ndarray)
        assert ids.ndim == 2 and ids.shape[0] == 1 and ids.dtype == np.int64


@pytest.mark.skipif(
    not TOKENIZER_DIR.is_dir(),
    reason="Tokenizer dir and transformers needed for HF comparison",
)
def test_lightweight_vs_hf_tokenizer_consistency(tokenizer_dir):
    """T4.1: token ids match HuggingFace AutoTokenizer when both available."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
    import numpy as np

    hf_tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    lt_tok = load_lightweight_tokenizer(tokenizer_dir)
    if lt_tok is None:
        pytest.skip("Lightweight tokenizer failed to load")

    test_texts = [
        "<|im_start|>assistant\n你好世界<|im_end|>",
        "<|im_start|>assistant\nHello, how are you?<|im_end|>",
        "<|im_start|>assistant\n中英混合 test 123<|im_end|>",
    ]
    for text in test_texts:
        hf_ids = hf_tok(text)["input_ids"]
        lt_result = lt_tok(text, return_tensors="pt")["input_ids"]
        lt_ids = np.asarray(lt_result).flatten().tolist()
        assert hf_ids == lt_ids, f"Mismatch for: {text!r}"
