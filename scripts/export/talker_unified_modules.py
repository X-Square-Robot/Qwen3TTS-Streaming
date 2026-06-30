"""
Shared Talker ONNX building blocks: backbone (layers + codec_head) and fused (+ CP + codec sum).

Used by export_07_talker_backbone, export_08_talker_unified, export_09_talker_code2wav_fused.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

from utils import CodePredictorUnrolled

LOGITS_TOPK = 50


class UnifiedKVCache:
    """Single cache that concats past + new KV per layer. S_past=0: cat([], new) -> new."""

    def __init__(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
        self._past = list(past_key_values)
        self._delta: list[Tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(
            self._past
        )

    def get_seq_length(self) -> int:
        if self._past[0] is None or self._past[0][0] is None:
            return 0
        return self._past[0][0].shape[2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        past_k, past_v = self._past[layer_idx]
        self._delta[layer_idx] = (key_states, value_states)
        full_k = torch.cat([past_k, key_states], dim=2)
        full_v = torch.cat([past_v, value_states], dim=2)
        self._past[layer_idx] = (full_k, full_v)
        return full_k, full_v

    def get_present(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._past[layer_idx]

    def get_delta(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = self._delta[layer_idx]
        if delta is None:
            raise RuntimeError(f"Delta KV not populated for layer {layer_idx}")
        return delta


def build_codec_embedding_sum_from_model(model, device, dtype=torch.float32):
    """Build CodecEmbeddingSum from live model (stacked [16, 3072, H])."""
    try:
        from codec_embedding_sum import CodecEmbeddingSum
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "scripts" / "python"))
        from codec_embedding_sum import CodecEmbeddingSum
    return CodecEmbeddingSum.from_model(model, dtype=dtype).to(device).eval()


def _rotate_half_export(x: torch.Tensor, half_dim: int) -> torch.Tensor:
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv_export(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states.reshape(batch, num_kv_heads, 1, seq_len, head_dim)
    hidden_states = hidden_states.expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _build_talker_rotary_embeddings_export(
    rotary_emb: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = rotary_emb.inv_freq.to(
        device=hidden_states.device, dtype=torch.float32
    ).reshape(1, 1, 1, -1)
    if position_ids.dim() == 4:
        pos = position_ids
    else:
        pos = position_ids.unsqueeze(-1)

    device_type = (
        hidden_states.device.type
        if isinstance(hidden_states.device.type, str)
        and hidden_states.device.type != "mps"
        else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = pos * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * rotary_emb.attention_scaling
        sin = emb.sin() * rotary_emb.attention_scaling
    return cos, sin


def _mix_talker_multimodal_rope_export(
    rope_tensor: torch.Tensor,
    mrope_section: List[int],
    interleaved: bool,
) -> torch.Tensor:
    if interleaved:
        half_dim = sum(mrope_section)
        prefix = []
        modality_num = len(mrope_section)
        half = rope_tensor[..., :half_dim]
        for idx in range(half_dim):
            src_modality = 0
            for modality in range(1, modality_num):
                end_idx = mrope_section[modality] * modality_num
                if (
                    idx >= modality
                    and idx < end_idx
                    and ((idx - modality) % modality_num == 0)
                ):
                    src_modality = modality
                    break
            prefix.append(half[:, src_modality, :, idx : idx + 1])
        mixed_half = torch.cat(prefix, dim=-1)
        return torch.cat((mixed_half, mixed_half), dim=-1)

    sections = list(mrope_section) * 2
    offset = 0
    chunks = []
    for idx, section in enumerate(sections):
        chunk = rope_tensor.narrow(-1, offset, section)
        chunks.append(chunk[:, idx % len(mrope_section), :, :])
        offset += section
    return torch.cat(chunks, dim=-1)


def _apply_talker_multimodal_rotary_pos_emb_export(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_scaling: dict,
    half_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mrope_section = list(rope_scaling["mrope_section"])
    interleaved = bool(rope_scaling["interleaved"])
    cos_mix = _mix_talker_multimodal_rope_export(
        cos, mrope_section, interleaved
    ).unsqueeze(1)
    sin_mix = _mix_talker_multimodal_rope_export(
        sin, mrope_section, interleaved
    ).unsqueeze(1)
    query_states = (query_states * cos_mix) + (
        _rotate_half_export(query_states, half_dim) * sin_mix
    )
    key_states = (key_states * cos_mix) + (
        _rotate_half_export(key_states, half_dim) * sin_mix
    )
    return query_states, key_states


def _run_talker_attention_export(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: UnifiedKVCache,
    cache_position: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    del cache_position
    batch = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    num_heads = attn_module.q_proj.out_features // attn_module.head_dim
    num_kv_heads = attn_module.k_proj.out_features // attn_module.head_dim

    query_states = attn_module.q_proj(hidden_states).reshape(
        batch, seq_len, num_heads, attn_module.head_dim
    )
    key_states = attn_module.k_proj(hidden_states).reshape(
        batch, seq_len, num_kv_heads, attn_module.head_dim
    )
    value_states = attn_module.v_proj(hidden_states).reshape(
        batch, seq_len, num_kv_heads, attn_module.head_dim
    )

    query_states = attn_module.q_norm(query_states).transpose(1, 2)
    key_states = attn_module.k_norm(key_states).transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _apply_talker_multimodal_rotary_pos_emb_export(
        query_states,
        key_states,
        cos,
        sin,
        attn_module.rope_scaling,
        attn_module.head_dim // 2,
    )
    key_states, value_states = cache.update(
        key_states, value_states, attn_module.layer_idx, cache_kwargs=None
    )

    key_states = _repeat_kv_export(key_states, attn_module.num_key_value_groups)
    value_states = _repeat_kv_export(value_states, attn_module.num_key_value_groups)

    attn_weights = (
        torch.matmul(query_states, key_states.transpose(2, 3)) * attn_module.scaling
    )
    attn_weights = attn_weights + attention_mask
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        value_states.dtype
    )

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).reshape(
        batch, seq_len, attn_module.o_proj.in_features
    )
    return attn_module.o_proj(attn_output)


def _run_talker_layer_export(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: UnifiedKVCache,
    cache_position: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = _run_talker_attention_export(
        layer.self_attn,
        hidden_states,
        attention_mask,
        cache,
        cache_position,
        position_embeddings,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


class TalkerUnifiedONNX(nn.Module):
    """Talker backbone for both prefill and decode: single path with dynamic S and S_past."""

    def __init__(self, talker_model, codec_head, *, emit_delta_kv: bool = False):
        super().__init__()
        self.layers = talker_model.layers
        self.norm = talker_model.norm
        self.rotary_emb = talker_model.rotary_emb
        self.codec_head = codec_head
        self.num_layers = len(self.layers)
        self.emit_delta_kv = emit_delta_kv

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        B, S, H = input_embeds.shape
        device = input_embeds.device
        dtype = input_embeds.dtype
        n = self.num_layers
        attention_bias = None
        past_seq_lens = None
        if len(inputs) == 2 * n:
            past_key_values = inputs
        elif len(inputs) == 1 + 2 * n:
            attention_bias = inputs[0]
            past_key_values = inputs[1:]
        elif len(inputs) == 2 + 2 * n:
            attention_bias = inputs[0]
            past_seq_lens = inputs[1]
            past_key_values = inputs[2:]
        else:
            raise ValueError(
                f"Expected {2 * n}, {1 + 2 * n}, or {2 + 2 * n} extra tensors, got {len(inputs)}"
            )
        past_list = [
            (past_key_values[2 * i], past_key_values[2 * i + 1]) for i in range(n)
        ]
        cache = UnifiedKVCache(past_list)
        S_past = cache.get_seq_length()
        S_total = S_past + S

        position_embeddings = _build_talker_rotary_embeddings_export(
            self.rotary_emb,
            input_embeds,
            position_ids,
        )

        row_idx = torch.arange(S, device=device, dtype=torch.long).unsqueeze(1) + S_past
        col_idx = torch.arange(S_total, device=device, dtype=torch.long).unsqueeze(0)
        neg_val = -1.0e4
        causal_mask = (col_idx > row_idx).to(dtype=dtype) * neg_val
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S_total)
        if attention_bias is not None:
            causal_mask = causal_mask + attention_bias
        if past_seq_lens is not None:
            # Keep past_seq_lens as a live ONNX input for BLS bookkeeping while
            # letting attention_bias carry the actual padded-key masking semantics.
            causal_mask = causal_mask + (
                past_seq_lens.narrow(0, 0, 1)
                .to(device=device, dtype=dtype)
                .reshape(1, 1, 1, 1)
                * 0.0
            )

        cache_position = torch.arange(
            S_past, S_past + S, device=device, dtype=torch.long
        )

        hidden = input_embeds
        for layer in self.layers:
            hidden = _run_talker_layer_export(
                layer,
                hidden,
                causal_mask,
                cache,
                cache_position,
                position_embeddings,
            )

        hidden = self.norm(hidden)
        logits = self.codec_head(hidden)

        outputs = [hidden, logits]
        for i in range(n):
            if self.emit_delta_kv:
                k, v = cache.get_delta(i)
            else:
                k, v = cache.get_present(i)
            outputs.append(k)
            outputs.append(v)
        return tuple(outputs)


class TalkerUnifiedFusedONNX(nn.Module):
    """Talker (prefill or decode) + logits processing + CP 15-step + Codec Embedding Sum.

    Logits processing pipeline (compiled into ONNX/TRT):
      1. suppress_tokens — static const mask, zero runtime cost
      2. repetition_penalty — element-wise via token_counts vector
      3. top-K=50 — TRT-optimized TopK kernel
      4. temperature + Gumbel-Max — deterministic inside graph, noise from outside
      5. one-hot counts update — auto-regressive state
    """

    def __init__(
        self,
        talker_unified: TalkerUnifiedONNX,
        code_predictor_unrolled: CodePredictorUnrolled,
        codec_embedding_sum: nn.Module,
        vocab_size: int = 3072,
        codec_eos_token_id: int = 2150,
    ):
        super().__init__()
        self.talker_unified = talker_unified
        self.cp = code_predictor_unrolled
        self.codec_sum = codec_embedding_sum
        self.num_layers = talker_unified.num_layers
        self.vocab_size = vocab_size

        suppress_ids = [
            i for i in range(vocab_size - 1024, vocab_size) if i != codec_eos_token_id
        ]
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        mask[suppress_ids] = True
        self.register_buffer("suppress_mask", mask)

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        token_counts: torch.Tensor,
        gumbel_noise: torch.Tensor,
        cp_gumbel_noise: torch.Tensor,
        temperature: torch.Tensor,
        penalty: torch.Tensor,
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        talker_out = self.talker_unified(input_embeds, position_ids, *inputs)
        hidden = talker_out[0]
        logits = talker_out[1]
        present_kv = list(talker_out[2:])

        g0 = logits[:, -1, :].clone()

        g0 = g0.masked_fill(self.suppress_mask, -1e9)

        has_appeared = token_counts > 0
        penalized = torch.where(g0 > 0, g0 / penalty, g0 * penalty)
        g0 = torch.where(has_appeared, penalized, g0)

        topk_vals, topk_idx = g0.topk(LOGITS_TOPK, dim=-1)

        topk_vals = topk_vals / temperature
        selected = (topk_vals + gumbel_noise).argmax(dim=-1)
        codec_token_0 = topk_idx.gather(1, selected.unsqueeze(1)).squeeze(1)

        one_hot = F.one_hot(codec_token_0, self.vocab_size).to(token_counts.dtype)
        updated_token_counts = token_counts + one_hot

        cp_tokens = self.cp(
            hidden[:, -1:, :],
            codec_token_0,
            cp_gumbel_noise=cp_gumbel_noise,
            temperature=temperature,
        )
        full_codec = torch.cat([codec_token_0.unsqueeze(1), cp_tokens.long()], dim=1)
        codec_sum = self.codec_sum(full_codec).unsqueeze(1)

        return (
            codec_sum,
            full_codec,
            hidden,
            logits,
            updated_token_counts,
            *present_kv,
        )


def build_talker_backbone_module(
    model, device: str = "cpu"
) -> Tuple[TalkerUnifiedONNX, int, int, int, int]:
    """Build TalkerUnifiedONNX only (no CP / codec sum)."""
    talker = model.talker
    talker_config = talker.model.config
    setattr(talker_config, "_attn_implementation", "eager")

    talker_model = talker.model.to(device).eval()
    codec_head = talker.codec_head.to(device).eval()
    backbone = (
        TalkerUnifiedONNX(
            talker_model,
            codec_head,
            emit_delta_kv=False,
        )
        .to(device)
        .eval()
    )

    num_layers = talker_config.num_hidden_layers
    hidden_size = talker_config.hidden_size
    num_kv_heads = talker_config.num_key_value_heads
    head_dim = getattr(
        talker_config,
        "head_dim",
        talker_config.hidden_size // talker_config.num_attention_heads,
    )
    return backbone, num_layers, hidden_size, num_kv_heads, head_dim


def build_talker_unified_fused_module(
    model, device: str = "cpu"
) -> Tuple[TalkerUnifiedFusedONNX, int, int, int, int]:
    """Build TalkerUnifiedFusedONNX (prefill+decode+CP+codec_sum)."""
    talker = model.talker
    talker_config = talker.model.config
    setattr(talker_config, "_attn_implementation", "eager")
    if hasattr(talker.code_predictor, "model") and hasattr(
        talker.code_predictor.model, "config"
    ):
        setattr(talker.code_predictor.model.config, "_attn_implementation", "eager")

    talker_model = talker.model.to(device).eval()
    codec_head = talker.codec_head.to(device).eval()
    backbone = (
        TalkerUnifiedONNX(
            talker_model,
            codec_head,
            emit_delta_kv=True,
        )
        .to(device)
        .eval()
    )

    code_predictor = talker.code_predictor.to(device).eval()
    talker_codec_emb = talker.model.codec_embedding.to(device).eval()
    cp_unrolled = (
        CodePredictorUnrolled(
            code_predictor,
            talker_codec_emb,
            logits_topk=LOGITS_TOPK,
        )
        .to(device)
        .eval()
    )

    codec_sum_module = build_codec_embedding_sum_from_model(model, device)

    vocab_size = talker_config.vocab_size
    codec_eos_token_id = getattr(talker_config, "codec_eos_token_id", 2150)

    fused = (
        TalkerUnifiedFusedONNX(
            backbone,
            cp_unrolled,
            codec_sum_module,
            vocab_size=vocab_size,
            codec_eos_token_id=codec_eos_token_id,
        )
        .to(device)
        .eval()
    )

    num_layers = talker_config.num_hidden_layers
    hidden_size = talker_config.hidden_size
    num_kv_heads = talker_config.num_key_value_heads
    head_dim = getattr(
        talker_config,
        "head_dim",
        talker_config.hidden_size // talker_config.num_attention_heads,
    )
    return fused, num_layers, hidden_size, num_kv_heads, head_dim
