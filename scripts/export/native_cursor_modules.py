"""TensorRT-exportable native cursor modules.

The cursor observes the codebook-0 token sampled by the Talker.  This module
keeps the neural part deliberately separate from TN and raw-text projection:
the graph receives integer label ids and fixed-size recurrent state, and
returns numeric cursor state/estimates.  Unicode, owner spans and public
high-water marks remain outside the graph.

The streaming trunk is equivalent to the released ``la=1`` cursor head:
three causal residual convolution blocks run on the current frame, while the
last dilated block consumes the current frame as one-frame lookahead and emits
the previous frame's feature.  Histories are activation tensors, not codec
token ids, so the initial zero padding is represented exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_OFFSETS: tuple[int, ...] = (-2, -1, 0, 1, 2, 3, 4)
DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)
DEFAULT_RATE_LABELS_PER_FRAME = 0.352
DEFAULT_CURSOR_D = 256
DEFAULT_CURSOR_CODEC_VOCAB = 3072
DEFAULT_CURSOR_HISTORY = sum(2 * d for d in DEFAULT_DILATIONS)


class CausalConvBlock(nn.Module):
    """One residual dilated convolution from the released cursor head."""

    def __init__(self, d: int, dilation: int, lookahead: int = 0) -> None:
        super().__init__()
        self.d = int(d)
        self.dilation = int(dilation)
        self.lookahead = int(lookahead)
        self.norm = nn.LayerNorm(self.d)
        self.conv = nn.Conv1d(self.d, self.d, 3, dilation=self.dilation)

    @property
    def history(self) -> int:
        return 2 * self.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Offline reference path; ``x`` is ``[B,T,D]``."""
        y = self.norm(x).transpose(1, 2)
        pad_total = 2 * self.dilation
        right = min(self.lookahead, pad_total)
        y = F.pad(y, (pad_total - right, right))
        y = F.gelu(self.conv(y)).transpose(1, 2)
        return x + y


class CursorHead(nn.Module):
    """The 2M-parameter observer, with an export-friendly reference forward."""

    def __init__(
        self,
        n_labels_plus_blank: int,
        *,
        d: int = DEFAULT_CURSOR_D,
        text_kernel: int = 3,
        offsets: Sequence[int] = DEFAULT_OFFSETS,
        codec_vocab_size: int = DEFAULT_CURSOR_CODEC_VOCAB,
        rate_labels_per_frame: float = DEFAULT_RATE_LABELS_PER_FRAME,
    ) -> None:
        super().__init__()
        offsets = tuple(int(v) for v in offsets)
        if not offsets or tuple(sorted(set(offsets))) != offsets:
            raise ValueError(f"offsets must be sorted and unique: {offsets}")
        if rate_labels_per_frame <= 0:
            raise ValueError("rate_labels_per_frame must be positive")

        self.d = int(d)
        self.offsets = offsets
        self.rate_labels_per_frame = float(rate_labels_per_frame)
        self.emb = nn.Embedding(int(codec_vocab_size), self.d)
        blocks = [CausalConvBlock(self.d, dil) for dil in DEFAULT_DILATIONS]
        blocks[-1].lookahead = 1
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(self.d, int(n_labels_plus_blank))

        self.text_emb = nn.Embedding(int(n_labels_plus_blank), self.d)
        # The released checkpoint ties text_emb to the content classifier.
        self.text_emb.weight = self.out.weight
        self.bos = nn.Parameter(torch.zeros(self.d))
        self.eos = nn.Parameter(torch.zeros(self.d))
        self.text_conv = nn.Conv1d(self.d, self.d, int(text_kernel))

        self.wq = nn.Linear(self.d, self.d)
        self.wk = nn.Linear(self.d, self.d, bias=False)
        self.off_emb = nn.Embedding(len(offsets), self.d)
        self.wl = nn.Linear(3, self.d, bias=False)
        self.v = nn.Linear(self.d, 1, bias=False)
        self.off_bias = nn.Parameter(torch.zeros(len(offsets)))
        self.register_buffer(
            "offset_values",
            torch.tensor(offsets, dtype=torch.float32),
            persistent=False,
        )

    @property
    def n_labels_plus_blank(self) -> int:
        return int(self.out.out_features)

    @property
    def left_context(self) -> int:
        return int(sum(block.history for block in self.blocks) - 1)

    @property
    def right_context(self) -> int:
        return 1

    @property
    def history_width(self) -> int:
        return int(sum(block.history for block in self.blocks))

    def encode_text(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode a fixed-size ``[B,M]`` label table with BOS/EOS rows."""
        e = self.text_emb(labels.long())
        k = int(self.text_conv.kernel_size[0])
        y = F.pad(e.transpose(1, 2), (k - 1, 0))
        e = e + F.gelu(self.text_conv(y)).transpose(1, 2)
        b = labels.shape[0]
        return torch.cat(
            [self.bos.expand(b, 1, -1), e, self.eos.expand(b, 1, -1)], dim=1
        )

    def forward_trunk(self, codec0: torch.Tensor) -> torch.Tensor:
        """Offline trunk reference; ``codec0`` is ``[B,T]``."""
        x = self.emb(codec0.long())
        for block in self.blocks:
            x = block(x)
        return x

    def location_features(
        self,
        mu_prev: torch.Tensor,
        frames_since_advance: torch.Tensor,
        mean_delta: torch.Tensor,
    ) -> torch.Tensor:
        phi = mu_prev - torch.floor(mu_prev)
        return torch.stack(
            [
                phi,
                frames_since_advance * self.rate_labels_per_frame,
                mean_delta.clamp(max=1.0),
            ],
            dim=-1,
        )

    def window_logits(
        self,
        h: torch.Tensor,
        etab: torch.Tensor,
        base: torch.Tensor,
        loc: torch.Tensor,
        label_count: torch.Tensor,
    ) -> torch.Tensor:
        """Score local labels around ``floor(mu)``.

        ``base``/``label_count`` use one-based label coordinates where zero is
        the BOS row.  The fixed-size text table is masked by label_count.
        """
        b, t, _ = h.shape
        offsets = self.offset_values.to(device=h.device)
        candidate = base[:, :, None] + offsets[None, None, :]
        valid = (candidate >= 0) & (candidate <= label_count[:, :, None])
        idx = candidate.long().clamp(0, etab.shape[1] - 1)
        gathered = torch.gather(
            etab,
            1,
            idx.reshape(b, -1, 1).expand(-1, -1, self.d),
        ).reshape(b, t, len(self.offsets), self.d)
        score = self.wq(h)[:, :, None, :] + self.off_emb.weight[None, None, :, :]
        score = score + self.wk(gathered) + self.wl(loc)[:, :, None, :]
        logits = self.v(torch.tanh(score)).squeeze(-1)
        logits = logits + self.off_bias[None, None, :]
        return logits.masked_fill(~valid, -1.0e4)


def _stream_block(
    block: CausalConvBlock,
    x_t: torch.Tensor,
    history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one no-lookahead block using activation history."""
    norm_t = block.norm(x_t)
    window = torch.cat([history, norm_t.unsqueeze(1)], dim=1)
    # Input length is 2*dilation+1, so this produces exactly one output.
    y = block.conv(window.transpose(1, 2)).squeeze(-1)
    y = F.gelu(y)
    out_t = x_t + y
    new_history = torch.cat([history[:, 1:, :], norm_t.unsqueeze(1)], dim=1)
    return out_t, new_history


class CursorStreamingStep(nn.Module):
    """One codec-frame cursor update suitable for fusion into the TTS graph."""

    def __init__(self, head: CursorHead) -> None:
        super().__init__()
        self.head = head
        if len(head.blocks) != 4 or head.blocks[-1].lookahead != 1:
            raise ValueError("streaming export currently requires four blocks and lookahead=1")

    def forward(
        self,
        codec0: torch.Tensor,
        label_ids: torch.Tensor,
        label_count: torch.Tensor,
        cursor_active: torch.Tensor,
        mu: torch.Tensor,
        frames_since_advance: torch.Tensor,
        delta_history: torch.Tensor,
        conv_history: torch.Tensor,
        last_trunk_input: torch.Tensor,
        seen_frames: torch.Tensor,
        text_start_frame: torch.Tensor,
        override_valid: torch.Tensor,
        override_mu: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Advance the cursor by one codebook-0 token.

        State shapes:

        - ``conv_history``: ``[B,30,D]`` (2, 4, 8 and 16 values per block);
        - ``last_trunk_input``: ``[B,D]`` (the residual input for the
          lookahead block's previous frame);
        - all scalar state tensors are batch vectors.
        """
        b = codec0.shape[0]
        d = self.head.d
        if conv_history.shape[1] != self.head.history_width:
            raise ValueError(
                f"conv_history width {conv_history.shape[1]} != {self.head.history_width}"
            )

        # The first three blocks consume the current frame causally.
        x_t = self.head.emb(codec0.long())
        offset = 0
        next_histories: list[torch.Tensor] = []
        for block in self.head.blocks[:-1]:
            width = block.history
            hist = conv_history[:, offset : offset + width, :]
            x_t, hist_new = _stream_block(block, x_t, hist)
            next_histories.append(hist_new)
            offset += width

        # The final block has one-frame lookahead.  Its convolution sees the
        # current normalized activation but its residual is the previous frame.
        last_block = self.head.blocks[-1]
        hist = conv_history[:, offset : offset + last_block.history, :]
        norm_t = last_block.norm(x_t)
        window = torch.cat([hist, norm_t.unsqueeze(1)], dim=1)
        y_prev = last_block.conv(window.transpose(1, 2)).squeeze(-1)
        h_prev = last_trunk_input + F.gelu(y_prev)
        hist_new = torch.cat([hist[:, 1:, :], norm_t.unsqueeze(1)], dim=1)
        next_histories.append(hist_new)
        next_conv_history = torch.cat(next_histories, dim=1)

        # ``seen_frames`` is the count before this token.  Thus the emitted
        # feature describes frame seen_frames-1 and is valid from the second
        # input token onward.
        target_frame = seen_frames - 1
        has_previous = seen_frames > 0
        labels_visible = label_count > 0
        text_visible = target_frame >= text_start_frame
        valid = (
            cursor_active.bool()
            & has_previous
            & labels_visible
            & text_visible
        )

        etab = self.head.encode_text(label_ids.long())
        base_mu = torch.where(override_valid.bool(), override_mu, mu)
        safe_count = label_count.to(base_mu.dtype).clamp(min=0.0)
        base_mu = base_mu.clamp(min=0.0)
        base_mu = torch.minimum(base_mu, safe_count)
        base = torch.floor(base_mu).long().view(b, 1)
        loc = self.head.location_features(
            base_mu,
            frames_since_advance,
            delta_history.mean(dim=1),
        ).view(b, 1, 3)
        known = label_count.long().view(b, 1)
        logits = self.head.window_logits(
            h_prev.view(b, 1, d),
            etab,
            base,
            loc,
            known,
        )
        probability = logits.softmax(dim=-1).view(b, -1)
        delta_candidate = (probability * self.head.offset_values).sum(dim=-1)
        delta = torch.where(valid, delta_candidate, torch.zeros_like(delta_candidate))
        candidate_mu = (base_mu + delta).clamp(min=0.0)
        candidate_mu = torch.minimum(candidate_mu, safe_count)
        # ``cursor_active`` is a lifecycle gate, not merely a visibility gate.
        # Prefill and other non-decoder calls still traverse the fused graph,
        # but must not consume a codec frame or mutate recurrent cursor state.
        active = cursor_active.bool()
        new_mu = torch.where(valid, candidate_mu, base_mu)
        new_mu = torch.where(active, new_mu, mu)

        crossed = torch.floor(new_mu) > torch.floor(base_mu)
        fs_candidate = torch.where(
            crossed,
            torch.zeros_like(frames_since_advance),
            frames_since_advance + 1.0,
        )
        new_frames_since = torch.where(valid, fs_candidate, frames_since_advance)
        new_frames_since = torch.where(active, new_frames_since, frames_since_advance)
        shifted_delta = torch.cat(
            [delta_history[:, 1:], delta.unsqueeze(1)], dim=1
        )
        new_delta_history = torch.where(
            valid.unsqueeze(1), shifted_delta, delta_history
        )
        confidence = torch.where(
            valid,
            probability.max(dim=-1).values,
            torch.zeros_like(delta_candidate),
        )
        candidate_label = torch.floor(new_mu).long()
        # Use INT64 for the binding rather than a BOOL output.  TensorRT's
        # explicit I/O format contract in this repository already handles
        # integer state tensors, while the CPU adapter can still treat zero/
        # non-zero as a boolean.
        valid_out = valid.to(dtype=torch.int64)
        next_conv_history = torch.where(
            active.view(b, 1, 1), next_conv_history, conv_history
        )
        next_last_trunk_input = torch.where(active.view(b, 1), x_t, last_trunk_input)
        new_seen_frames = torch.where(active, seen_frames + 1, seen_frames)

        return (
            valid_out,
            new_mu,
            delta,
            confidence,
            candidate_label,
            new_frames_since,
            new_delta_history,
            next_conv_history,
            next_last_trunk_input,
            new_seen_frames,
        )


def build_cursor_head_from_checkpoint(
    checkpoint_or_path: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[CursorHead, dict[str, Any]]:
    """Load a released native cursor checkpoint for export.

    Returns the model and a small metadata dictionary used by the manifest.
    The function accepts an already loaded mapping to keep exporter tests
    independent from filesystem assets.
    """
    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint = torch.load(
            str(checkpoint_or_path),
            map_location=map_location,
            weights_only=False,
        )
    else:
        checkpoint = dict(checkpoint_or_path)
    if "state_dict" not in checkpoint or "vocab" not in checkpoint:
        raise ValueError("cursor checkpoint must contain state_dict and vocab")

    state_dict = checkpoint["state_dict"]
    out_weight = state_dict.get("trunk.out.weight", state_dict.get("out.weight"))
    if out_weight is None:
        raise ValueError("cursor checkpoint is missing trunk.out.weight/out.weight")
    d = int(out_weight.shape[1])
    args = checkpoint.get("args") or {}
    offsets = checkpoint.get("offsets") or DEFAULT_OFFSETS
    right_context = int(
        checkpoint.get("right_context", args.get("lookahead_frames", 1))
    )
    if right_context != 1:
        raise ValueError("the fused streaming graph currently supports right_context=1 only")
    model = CursorHead(
        len(checkpoint["vocab"]) + 1,
        d=d,
        text_kernel=int(args.get("text_kernel", 3)),
        offsets=offsets,
        codec_vocab_size=int(
            checkpoint.get(
                "codec_vocab_size",
                args.get("codec_vocab_size", DEFAULT_CURSOR_CODEC_VOCAB),
            )
        ),
        rate_labels_per_frame=float(
            checkpoint.get(
                "rate_labels_per_frame",
                args.get("rate_labels_per_frame", DEFAULT_RATE_LABELS_PER_FRAME),
            )
        ),
    )

    # Research checkpoints use trunk.* names; normalize only the optional
    # unwrapped names so strict loading remains useful for released heads.
    normalized = dict(state_dict)
    for key in list(normalized):
        if key.startswith("trunk.emb.0."):
            normalized["emb." + key[len("trunk.emb.0.") :]] = normalized.pop(key)
        elif key.startswith("trunk.blocks."):
            normalized["blocks." + key[len("trunk.blocks.") :]] = normalized.pop(key)
        elif key.startswith("trunk.out."):
            normalized["out." + key[len("trunk.out.") :]] = normalized.pop(key)
    missing, unexpected = model.load_state_dict(normalized, strict=False)
    missing = [key for key in missing if key != "text_emb.weight"]
    if missing or unexpected:
        raise RuntimeError(
            f"cursor checkpoint does not match export model: missing={missing} unexpected={unexpected}"
        )
    return model.eval(), {
        "vocab_size": len(checkpoint["vocab"]),
        "offsets": [int(v) for v in model.offsets],
        "left_context": model.left_context,
        "right_context": model.right_context,
        "codec_vocab_size": int(model.emb.num_embeddings),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "labels_version": str(checkpoint.get("labels_version", "") or ""),
        "head_sha256": str(checkpoint.get("head_sha256", "") or ""),
        "vocab_sha256": str(checkpoint.get("vocab_sha256", "") or ""),
        "rules_sha256": str(checkpoint.get("rules_sha256", "") or ""),
        "model_fingerprint": str(checkpoint.get("model_fingerprint", "") or ""),
        "training_scope": str(checkpoint.get("training_scope", "") or ""),
        "native_frame_us": int(checkpoint.get("native_frame_us", 80_000) or 80_000),
    }


__all__ = [
    "CursorHead",
    "CursorStreamingStep",
    "CausalConvBlock",
    "DEFAULT_CURSOR_D",
    "DEFAULT_CURSOR_CODEC_VOCAB",
    "DEFAULT_CURSOR_HISTORY",
    "build_cursor_head_from_checkpoint",
]
