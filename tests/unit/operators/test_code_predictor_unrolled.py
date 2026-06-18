from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = REPO_ROOT / "scripts" / "export"
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

from utils import CodePredictorUnrolled


class _IdentityLayer(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        return (hidden,)


class _DummyRotary(nn.Module):
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        return None


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
            layers=nn.ModuleList([_IdentityLayer()]),
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
