"""
Unit test: ConvTranspose1d replacement equivalence.

1. Baseline: PyTorch ConvTranspose1d == insert-zeros + Conv1d (kernel flip) - verifies
   our understanding of ConvTranspose math.
2. Conv1dInsertZeros vs ConvTranspose1d - verifies the patch used for TRT BF16.

Run: pytest tests/unit/test_conv1d_subpixel.py -v
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "export"))
from utils import Conv1dInsertZeros


def _conv_transpose_via_insert_zeros(x: torch.Tensor, tc: nn.ConvTranspose1d) -> torch.Tensor:
    """Reference: ConvTranspose1d via insert-zeros then Conv1d. Same weight, no reshape."""
    B, C, L = x.shape
    K = tc.kernel_size[0] if isinstance(tc.kernel_size, (tuple, list)) else tc.kernel_size
    S = tc.stride[0] if isinstance(tc.stride, (tuple, list)) else tc.stride
    # Insert S-1 zeros: x[i] -> positions i*S. Length = (L-1)*S + 1
    L_up = (L - 1) * S + 1
    up = torch.zeros(B, C, L_up, dtype=x.dtype, device=x.device)
    up[..., 0::S] = x
    # Pad both sides by (K-1) for ConvTranspose default
    up = F.pad(up, (K - 1, K - 1), mode="constant", value=0)
    # Conv1d: (in,out,K) -> (out,in,K). PyTorch ConvTranspose flips kernel along K.
    W = tc.weight  # (in_ch, out_ch, K)
    W_conv = W.permute(1, 0, 2).flip(-1)  # (out_ch, in_ch, K), flipped on kernel dim
    out = F.conv1d(up, W_conv, tc.bias, stride=1, padding=0)
    return out


def test_conv_transpose_equals_insert_zeros():
    """Baseline: PyTorch ConvTranspose1d MUST equal insert-zeros + Conv1d (same weight).

    If this fails, PyTorch ConvTranspose has different semantics than we assume.
    If this passes, the bug is in Conv1dInsertZeros weight mapping.
    """
    configs = [(2, 2), (16, 8), (10, 5), (6, 3)]
    for K, S in configs:
        tc = nn.ConvTranspose1d(64, 32, K, stride=S)
        x = torch.randn(2, 64, 8)
        with torch.no_grad():
            y_tc = tc(x)
            y_iz = _conv_transpose_via_insert_zeros(x, tc)
        assert y_tc.shape == y_iz.shape, f"K={K} S={S}: shape {y_tc.shape} vs {y_iz.shape}"
        mad = (y_tc.float() - y_iz.float()).abs().max().item()
        assert mad < 1e-5, (
            f"K={K} S={S}: insert-zeros must equal ConvTranspose (baseline), mad={mad:.6e}"
        )


def _compare(tc: nn.ConvTranspose1d, sub: Conv1dInsertZeros, x: torch.Tensor, tol: float = 1e-5):
    """Run both on x, return (match, max_abs_diff, cosine_sim)."""
    with torch.no_grad():
        y_tc = tc(x)
        y_sub = sub(x)
    assert y_tc.shape == y_sub.shape, f"Shape mismatch: {y_tc.shape} vs {y_sub.shape}"
    diff = (y_tc.float() - y_sub.float()).abs()
    mad = diff.max().item()
    a = y_tc.float().flatten()
    b = y_sub.float().flatten()
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
    match = mad < tol and cos > 0.9999
    return match, mad, cos


def test_conv1d_subpixel_simple():
    """K=S=2 (upsample block), right_pad=0."""
    in_ch, out_ch, K, S = 1024, 1024, 2, 2
    tc = nn.ConvTranspose1d(in_ch, out_ch, K, stride=S)
    sub = Conv1dInsertZeros(tc)
    x = torch.randn(1, in_ch, 4)
    match, mad, cos = _compare(tc, sub, x)
    assert match, f"K={K} S={S}: mad={mad:.6f}, cos={cos:.6f}"


def test_conv1d_subpixel_decoder_block():
    """K=16 S=8 (decoder block), right_pad=8."""
    in_ch, out_ch, K, S = 768, 384, 16, 8
    tc = nn.ConvTranspose1d(in_ch, out_ch, K, stride=S)
    sub = Conv1dInsertZeros(tc)
    x = torch.randn(1, in_ch, 4)
    match, mad, cos = _compare(tc, sub, x)
    assert match, f"K={K} S={S}: mad={mad:.6f}, cos={cos:.6f}"


def test_conv1d_subpixel_multiple_configs():
    """All (K,S) from code2wav."""
    configs = [
        (1024, 1024, 2, 2),
        (1536, 768, 16, 8),
        (768, 384, 10, 5),
        (384, 192, 8, 4),
        (192, 96, 6, 3),
    ]
    for in_ch, out_ch, K, S in configs:
        tc = nn.ConvTranspose1d(in_ch, out_ch, K, stride=S)
        sub = Conv1dInsertZeros(tc)
        x = torch.randn(1, in_ch, 8)
        match, mad, cos = _compare(tc, sub, x)
        assert match, f"in={in_ch} out={out_ch} K={K} S={S}: mad={mad:.6f}, cos={cos:.6f}"


def test_conv1d_subpixel_from_real_decoder():
    """Use actual ConvTranspose weights from code2wav decoder."""
    tok_path = REPO_ROOT / "workspace" / "models" / "Qwen3-TTS-Tokenizer-12Hz"
    if not tok_path.exists():
        import pytest
        pytest.skip("Tokenizer not found (run download first)")

    sys.path.insert(0, str(REPO_ROOT / "third_party" / "Qwen3-TTS"))
    from utils import load_speech_tokenizer

    tokenizer = load_speech_tokenizer(tok_path, device="cpu", dtype=torch.float32)
    decoder = tokenizer.decoder

    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2CausalTransConvNet,
    )
    for mod in decoder.modules():
        if isinstance(mod, Qwen3TTSTokenizerV2CausalTransConvNet):
            tc = mod.conv
            if isinstance(tc, nn.ConvTranspose1d):
                sub = Conv1dInsertZeros(tc)
                x = torch.randn(1, tc.in_channels, 4)
                match, mad, cos = _compare(tc, sub, x)
                assert match, f"Real decoder layer: mad={mad:.6f}, cos={cos:.6f}"
                return
    import pytest
    pytest.skip("No ConvTranspose found in decoder")
