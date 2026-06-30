"""Batched decode helpers for packed KV tensors.

Packed format: single [B, L*2, H, S, D] tensor per cache type instead of
2*L individual [B, H, S, D] tensors.  Reduces TRT I/O binding count from
~201 to ~61 and eliminates Python-loop overhead in batch assembly.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention bias construction
# ---------------------------------------------------------------------------


def _apply_causal_mask(bias: torch.Tensor, past_len: int, seq: int) -> torch.Tensor:
    """Lower-triangular causal mask over the new-token region.

    For seq == 1 (decode) this is a no-op.
    """
    if seq <= 1:
        return bias
    causal = torch.triu(
        torch.full((seq, seq), float("-inf"), device=bias.device, dtype=bias.dtype),
        diagonal=1,
    )
    bias[:, :, :, past_len : past_len + seq] += causal.unsqueeze(0).unsqueeze(0)
    return bias


def zeros_attention_bias(
    batch: int,
    seq: int,
    past_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Uniform-past-length attention bias: all positions visible."""
    total = past_len + seq
    bias = torch.zeros(batch, 1, seq, total, device=device, dtype=dtype)
    return _apply_causal_mask(bias, past_len, seq)


def padded_attention_bias(
    past_seq_lens: torch.Tensor,
    seq: int,
    padded_past_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Attention bias for heterogeneous past lengths.

    Masks out padding positions in the KV cache for each row.
    """
    batch = int(past_seq_lens.shape[0])
    total = int(padded_past_len) + int(seq)
    bias = torch.zeros(batch, 1, seq, total, device=device, dtype=dtype)
    if padded_past_len > 0:
        positions = torch.arange(padded_past_len, device=device).unsqueeze(0)
        mask = positions >= past_seq_lens.unsqueeze(1)
        bias[:, 0, :, :padded_past_len].masked_fill_(mask.unsqueeze(1), float("-inf"))
    return _apply_causal_mask(bias, padded_past_len, seq)


def uniform_past_seq_lens(
    batch: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.full((batch,), past_len, device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# Packed KV cache padding and splitting
# ---------------------------------------------------------------------------


def pad_packed_kv(
    session_kv: Sequence[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad heterogeneous packed KV caches to max length and batch them.

    Args:
        session_kv: list of per-session packed KV tensors.
            Each tensor shape: [1, L*2, H, seq_len_i, D].

    Returns:
        (batched_kv, past_seq_lens) where batched_kv has shape
        [B, L*2, H, padded_past_len, D].
    """
    if not session_kv:
        return torch.empty(0, device=device, dtype=dtype), torch.empty(
            (0,), device=device, dtype=torch.long
        )

    past_seq_lens = torch.tensor(
        [int(kv.shape[3]) for kv in session_kv],
        device=device,
        dtype=torch.long,
    )
    padded_past_len = int(past_seq_lens.max().item())

    if len(set(int(kv.shape[3]) for kv in session_kv)) == 1:
        batched = torch.cat(
            [kv.to(device=device, dtype=dtype) for kv in session_kv],
            dim=0,
        ).contiguous()
        return batched, past_seq_lens

    padded = []
    for kv in session_kv:
        kv = kv.to(device=device, dtype=dtype)
        cur_len = int(kv.shape[3])
        if cur_len < padded_past_len:
            # F.pad pads from last dim backwards: (D_right, D_left, S_right, S_left, ...)
            # We pad dim=3 (S) on the right only.
            kv = F.pad(kv, (0, 0, 0, padded_past_len - cur_len))
        padded.append(kv)

    return torch.cat(padded, dim=0).contiguous(), past_seq_lens


def split_packed_kv(
    present_kv: torch.Tensor,
    original_past_lens: list[int],
    padded_past_len: int,
    seq: int,
) -> list[torch.Tensor]:
    """Extract per-session packed KV from padded batch output.

    Args:
        present_kv: [B, L*2, H, padded_past_len + seq, D]
        original_past_lens: actual past length per session before this step
        padded_past_len: the padded dimension used for batching
        seq: number of new tokens (typically 1 for decode)

    Returns:
        list of [1, L*2, H, actual_past_len + seq, D] tensors per session.
    """
    uniform = len(set(original_past_lens)) <= 1

    if uniform:
        return [present_kv[i : i + 1] for i in range(present_kv.shape[0])]

    results = []
    for i, orig_pl in enumerate(original_past_lens):
        if orig_pl >= padded_past_len:
            results.append(present_kv[i : i + 1, :, :, : orig_pl + seq, :].contiguous())
        else:
            real_past = present_kv[i : i + 1, :, :, :orig_pl, :]
            new_part = present_kv[
                i : i + 1, :, :, padded_past_len : padded_past_len + seq, :
            ]
            results.append(torch.cat([real_past, new_part], dim=3).contiguous())
    return results
