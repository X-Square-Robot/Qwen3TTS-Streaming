from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/unit/operators/<file> -> repo root
EXPORT_DIR = REPO_ROOT / "scripts" / "export"
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

from utils import CodePredictorUnrolled


def _zero_linear(in_features: int, out_features: int) -> nn.Linear:
    """A Linear whose weight and bias are zero (so it outputs a zero tensor)."""
    layer = nn.Linear(in_features, out_features)
    with torch.no_grad():
        layer.weight.zero_()
        if layer.bias is not None:
            layer.bias.zero_()
    return layer


class _IdentityAttention(nn.Module):
    """Self-attention stub whose ``o_proj`` is zero, so the attention block
    contributes nothing and the surrounding residual makes it an identity.

    ``CodePredictorUnrolled`` no longer calls ``layer(...)`` directly; it
    decomposes the block and runs attention via ``_run_attention``, which reads
    these sub-module attributes.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.head_dim = hidden_size
        self.num_key_value_groups = 1
        self.scaling = 1.0
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.o_proj = _zero_linear(hidden_size, hidden_size)


class _ZeroMLP(nn.Module):
    """MLP stub returning zeros, so the residual makes the block an identity."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden)


class _IdentityLayer(nn.Module):
    """A decoder layer that is a no-op identity under the unrolled contract:
    layernorms are identities and both the attention and MLP branches
    contribute zero, so ``residual + branch == residual``.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = _IdentityAttention(hidden_size)
        self.post_attention_layernorm = nn.Identity()
        self.mlp = _ZeroMLP()


class _DummyRotary(nn.Module):
    """Returns a non-rotating (cos=1, sin=0) embedding so attention is unrotated."""

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        batch, seq_len = x.shape[0], x.shape[1]
        head_dim = x.shape[-1]
        cos = torch.ones(batch, seq_len, head_dim, dtype=x.dtype, device=x.device)
        sin = torch.zeros(batch, seq_len, head_dim, dtype=x.dtype, device=x.device)
        return cos, sin


class _SequenceLengthProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_lens: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seq_lens.append(int(x.shape[1]))
        offsets = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1)
        return x + offsets


class _ThresholdHead(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        score = hidden[..., 0]
        return torch.stack((-score, score), dim=-1)


def _zero_embedding(num_embeddings: int, hidden_size: int) -> nn.Embedding:
    emb = nn.Embedding(num_embeddings, hidden_size)
    with torch.no_grad():
        emb.weight.zero_()
    return emb


def test_code_predictor_projects_prefix_once_then_only_new_tokens():
    hidden_size = 4
    projection = _SequenceLengthProjection()
    codec_embeddings = nn.ModuleList(
        [
            _zero_embedding(2, hidden_size),
            _zero_embedding(2, hidden_size),
        ]
    )
    lm_heads = nn.ModuleList([_ThresholdHead(), _ThresholdHead(), _ThresholdHead()])

    code_predictor = SimpleNamespace(
        model=SimpleNamespace(
            layers=nn.ModuleList([_IdentityLayer(hidden_size)]),
            norm=nn.Identity(),
            rotary_emb=_DummyRotary(),
            codec_embedding=codec_embeddings,
        ),
        small_to_mtp_projection=projection,
        lm_head=lm_heads,
        config=SimpleNamespace(hidden_size=hidden_size),
    )
    wrapper = CodePredictorUnrolled(
        code_predictor=code_predictor,
        talker_codec_embedding=_zero_embedding(2, hidden_size),
    ).eval()

    past_hidden = torch.zeros(1, 1, hidden_size)
    codec_token_0 = torch.tensor([0], dtype=torch.long)
    out = wrapper(past_hidden, codec_token_0)

    # The initial prefix is projected once; later stages only project the new codec embedding.
    assert projection.seq_lens == [2, 1, 1]
    torch.testing.assert_close(out, torch.tensor([[1, 0, 0]], dtype=torch.long))
