"""Tests for engine/backend/batch_helper.py (packed KV format)."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO_ROOT / "engine"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.backend.batch_helper import (
    pad_packed_kv,
    padded_attention_bias,
    split_packed_kv,
)


def test_pad_packed_kv_uniform_lengths():
    """Two sessions with equal past_len — no padding needed."""
    kv0 = torch.ones(1, 4, 2, 3, 8, dtype=torch.float32)
    kv1 = torch.full((1, 4, 2, 3, 8), 2.0, dtype=torch.float32)

    batched, past_seq_lens = pad_packed_kv(
        [kv0, kv1], device=torch.device("cpu"), dtype=torch.float32,
    )

    assert past_seq_lens.tolist() == [3, 3]
    assert batched.shape == (2, 4, 2, 3, 8)
    assert torch.allclose(batched[0], kv0[0])
    assert torch.allclose(batched[1], kv1[0])


def test_pad_packed_kv_heterogeneous_lengths():
    """Two sessions with different past_len — shorter one gets padded."""
    kv_short = torch.ones(1, 4, 2, 2, 8, dtype=torch.float32)
    kv_long = torch.full((1, 4, 2, 5, 8), 2.0, dtype=torch.float32)

    batched, past_seq_lens = pad_packed_kv(
        [kv_short, kv_long], device=torch.device("cpu"), dtype=torch.float32,
    )

    assert past_seq_lens.tolist() == [2, 5]
    assert batched.shape == (2, 4, 2, 5, 8)
    # Short session: original data in [0:2], zeros in [2:5]
    assert torch.allclose(batched[0, :, :, :2, :], kv_short[0])
    assert torch.count_nonzero(batched[0, :, :, 2:, :]) == 0
    # Long session: all original data
    assert torch.allclose(batched[1], kv_long[0])


def test_split_packed_kv_uniform():
    """Uniform past_len — split without depadding."""
    present = torch.randn(2, 4, 2, 6, 8)
    splits = split_packed_kv(present, [5, 5], padded_past_len=5, seq=1)

    assert len(splits) == 2
    assert splits[0].shape == (1, 4, 2, 6, 8)
    assert torch.equal(splits[0], present[0:1])


def test_split_packed_kv_heterogeneous():
    """Heterogeneous past_len — depad correctly."""
    padded_past = 5
    seq = 1
    present = torch.randn(2, 4, 2, padded_past + seq, 8)

    # Fill identifiable data
    present[0, :, :, :3, :] = 1.0   # row 0 real past (len=3)
    present[0, :, :, 3:5, :] = -1.0  # row 0 padding
    present[0, :, :, 5:6, :] = 2.0   # row 0 new token
    present[1, :, :, :5, :] = 3.0    # row 1 real past (len=5)
    present[1, :, :, 5:6, :] = 4.0   # row 1 new token

    splits = split_packed_kv(present, [3, 5], padded_past_len=padded_past, seq=seq)

    assert len(splits) == 2
    # Row 0: [0:3] real + [5:6] new = total 4
    assert splits[0].shape == (1, 4, 2, 4, 8)
    assert torch.allclose(splits[0][:, :, :, :3, :], torch.tensor(1.0))
    assert torch.allclose(splits[0][:, :, :, 3:4, :], torch.tensor(2.0))
    # Row 1: [0:6] as-is = total 6
    assert splits[1].shape == (1, 4, 2, 6, 8)
    assert torch.allclose(splits[1][:, :, :, :5, :], torch.tensor(3.0))
    assert torch.allclose(splits[1][:, :, :, 5:6, :], torch.tensor(4.0))


def test_padded_attention_bias_masks_padding():
    """Attention bias should mask padded positions with -inf."""
    past_seq_lens = torch.tensor([2, 4], dtype=torch.long)

    bias = padded_attention_bias(
        past_seq_lens,
        seq=1,
        padded_past_len=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert bias.shape == (2, 1, 1, 5)
    assert torch.isneginf(bias[0, 0, 0, 2:4]).all()
    assert torch.equal(bias[0, 0, 0, :2], torch.zeros(2))
    assert torch.equal(bias[1], torch.zeros_like(bias[1]))


def test_padded_attention_bias_vectorized():
    """Verify the vectorized implementation matches expected masking."""
    past_seq_lens = torch.tensor([1, 3, 5], dtype=torch.long)
    bias = padded_attention_bias(
        past_seq_lens, seq=1, padded_past_len=5,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert bias.shape == (3, 1, 1, 6)
    # Row 0: past_len=1, mask [1:5]
    assert torch.isneginf(bias[0, 0, 0, 1:5]).all()
    assert bias[0, 0, 0, 0].item() == 0.0
    # Row 1: past_len=3, mask [3:5]
    assert torch.isneginf(bias[1, 0, 0, 3:5]).all()
    assert (bias[1, 0, 0, :3] == 0).all()
    # Row 2: past_len=5, no mask
    assert (bias[2, 0, 0, :5] == 0).all()
