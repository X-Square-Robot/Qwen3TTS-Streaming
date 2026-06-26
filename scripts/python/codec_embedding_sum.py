#!/usr/bin/env python3
"""
Codec Embedding Sum: 3D gather + sum optimization (§2.1 architecture).

Replaces 16 separate Embedding.forward() + Python loop with a single
pre-stacked [16, vocab, hidden] tensor and advanced indexing + sum,
reducing decode-step control overhead from ~0.15ms to ~0.05ms.

Usage:
  from codec_embedding_sum import CodecEmbeddingSum
  module = CodecEmbeddingSum.from_exported_weights(weights_dir, device, dtype)
  codec_sum = module(codec_ids)  # [B, 16] -> [B, H]
"""

from pathlib import Path
import time

import torch
import torch.nn as nn

# Default number of codec groups (Talker 1 + Code Predictor 15)
NUM_CODE_GROUPS = 16


def _stack_weights_impl(
    talker_weight: torch.Tensor,
    cp_weights: list,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build [16, V_max, H] stacked tensor from Talker (1) + CP (15) embedding weights.
    Talker has vocab 3072, CP has vocab 2048; CP rows are zero-padded to 3072.
    """
    if dtype is None:
        dtype = talker_weight.dtype
    if device is None:
        device = talker_weight.device
    talker_weight = talker_weight.to(device=device, dtype=dtype)
    V_talker = talker_weight.shape[0]
    H = talker_weight.shape[1]
    parts = [talker_weight.unsqueeze(0)]
    for w in cp_weights:
        w = w.to(device=device, dtype=dtype)
        V_cp = w.shape[0]
        if V_cp < V_talker:
            padded = torch.zeros(V_talker, H, dtype=dtype, device=device)
            padded[:V_cp, :] = w
            parts.append(padded.unsqueeze(0))
        else:
            parts.append(w.unsqueeze(0))
    stacked = torch.cat(parts, dim=0)
    return stacked


class CodecEmbeddingSum(nn.Module):
    """
    Single 3D gather + sum over 16 codec embedding tables.
    stacked_weight: [G, V, H] with G=16, V=3072, H=talker_hidden_size.
    forward(codec_ids [B, G]) -> [B, H].
    """

    def __init__(self, stacked_weight: torch.Tensor):
        super().__init__()
        G, V, H = stacked_weight.shape
        self.num_groups = G
        self.vocab_size = V
        self.hidden_size = H
        self.register_buffer("stacked_weight", stacked_weight)

    def forward(self, codec_ids: torch.Tensor) -> torch.Tensor:
        """
        codec_ids: [B, G] long, G <= num_groups.
        Returns: [B, H] sum of looked-up embeddings.
        """
        B, G = codec_ids.shape
        G = min(G, self.num_groups)
        codec_ids = codec_ids[:, :G]
        stacked = self.stacked_weight[:G]
        group_idx = torch.arange(G, device=codec_ids.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        looked_up = stacked[group_idx, codec_ids, :]
        return looked_up.sum(dim=1)

    @staticmethod
    def stack_weights(
        talker_weight: torch.Tensor,
        cp_weights: list,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Build [16, V_max, H] stacked tensor (vocab-aligned, CP zero-padded)."""
        return _stack_weights_impl(talker_weight, cp_weights, dtype=dtype, device=device)

    @classmethod
    def from_exported_weights(
        cls,
        weights_dir: Path,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> "CodecEmbeddingSum":
        """Load from codec_embeddings.pt and build stacked 3D tensor."""
        weights_dir = Path(weights_dir)
        data = torch.load(weights_dir / "codec_embeddings.pt", map_location=device, weights_only=True)
        talker_sd = data["talker_codec_embedding"]
        cp_sds = data["code_predictor_codec_embeddings"]
        talker_weight = talker_sd["weight"]
        cp_weights = [sd["weight"] for sd in cp_sds]
        stacked = cls.stack_weights(talker_weight, cp_weights, dtype=dtype, device=device)
        return cls(stacked)

    @classmethod
    def from_model(cls, model, dtype: torch.dtype = None) -> "CodecEmbeddingSum":
        """Build from live PyTorch Qwen3 TTS model (for verification)."""
        talker = model.talker
        talker_weight = talker.model.codec_embedding.weight
        cp_weights = [emb.weight for emb in talker.code_predictor.model.codec_embedding]
        if dtype is None:
            dtype = talker_weight.dtype
        stacked = cls.stack_weights(talker_weight, cp_weights, dtype=dtype, device=talker_weight.device)
        return cls(stacked)


def codec_sum_naive(
    talker_embedding: nn.Embedding,
    cp_embeddings: list,
    codec_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference: 16 Embedding.forward() + loop. codec_ids [B, 16] -> [B, H]."""
    B, G = codec_ids.shape
    H = talker_embedding.embedding_dim
    device = codec_ids.device
    dtype = next(talker_embedding.parameters()).dtype
    out = torch.zeros(B, H, device=device, dtype=dtype)
    for i in range(min(16, G)):
        if i == 0:
            out += talker_embedding(codec_ids[:, i])
        elif i - 1 < len(cp_embeddings):
            out += cp_embeddings[i - 1](codec_ids[:, i])
    return out


def benchmark(
    module: CodecEmbeddingSum,
    naive_talker_embedding: nn.Embedding = None,
    naive_cp_embeddings: list = None,
    batch_size: int = 1,
    num_warmup: int = 100,
    num_repeat: int = 1000,
    device: torch.device = None,
) -> dict:
    """
    Compare latency: 3D gather vs naive loop.
    Returns dict with keys: opt_ms, naive_ms, speedup.
    """
    if device is None:
        # CodecEmbeddingSum holds its weights in a buffer, not a Parameter, so
        # next(module.parameters()) would raise StopIteration.
        device = module.stacked_weight.device
    if isinstance(device, str):
        device = torch.device(device)
    B = batch_size
    G = module.num_groups
    V = module.vocab_size
    codec_ids = torch.randint(0, min(2048, V), (B, G), device=device, dtype=torch.long)

    module.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = module(codec_ids)
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_repeat):
            _ = module(codec_ids)
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize()
        opt_ms = (time.perf_counter() - t0) / num_repeat * 1000
    out = {"opt_ms": opt_ms, "naive_ms": None, "speedup": None}

    if naive_talker_embedding is not None and naive_cp_embeddings is not None:
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = codec_sum_naive(naive_talker_embedding, naive_cp_embeddings, codec_ids)
            if getattr(device, "type", None) == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_repeat):
                _ = codec_sum_naive(naive_talker_embedding, naive_cp_embeddings, codec_ids)
            if getattr(device, "type", None) == "cuda":
                torch.cuda.synchronize()
            naive_ms = (time.perf_counter() - t0) / num_repeat * 1000
        out["naive_ms"] = naive_ms
        out["speedup"] = naive_ms / opt_ms if opt_ms > 0 else 0
    return out
