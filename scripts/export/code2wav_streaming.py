"""
Stateful code2wav streaming wrapper for ONNX/TRT export.

Unified chunk_T=4: input codes [B, 16, 4], output wav [B, 7680] + 37 updated state tensors.
Uses SlidingWindowKVCache (window=72), streaming CausalConvNet (left-context buffer),
and streaming CausalTransConvNet (overlap-add) for incremental decode.
"""

from __future__ import annotations

import types
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from qwen_tts.core.tokenizer_12hz import (
        modeling_qwen3_tts_tokenizer_v2 as qwen_tokenizer_v2,
    )
except Exception:
    qwen_tokenizer_v2 = None


# -----------------------------------------------------------------------------
# Sliding-window KV cache (concat past + new, then truncate to last 72)
# -----------------------------------------------------------------------------


class SlidingWindowKVCache:
    """KV cache that truncates to last window_size positions after each update."""

    def __init__(
        self,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        window_size: int = 72,
    ):
        self._past = list(past_key_values)
        self._delta: list[Tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(
            self._past
        )
        self.window_size = window_size

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
        if self.window_size is not None and self.window_size > 0:
            full_k = full_k[:, :, -self.window_size :, :]
            full_v = full_v[:, :, -self.window_size :, :]
        self._past[layer_idx] = (full_k, full_v)
        return full_k, full_v

    def get_present(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._past[layer_idx]

    def get_delta(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = self._delta[layer_idx]
        if delta is None:
            raise RuntimeError(f"Delta KV not populated for layer {layer_idx}")
        return delta


# -----------------------------------------------------------------------------
# Streaming causal conv: left-context buffer instead of zero-pad
# -----------------------------------------------------------------------------


def streaming_causal_conv(
    conv_module: nn.Module,
    x: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run causal conv with left context from state. state shape [B, C_in, padding].
    Returns (out [B, C_out, T], new_state [B, C_in, padding]).
    """
    if state.shape[0] != x.shape[0]:
        state = state[: x.shape[0]]
    padding = conv_module.padding  # left padding length
    full = torch.cat([state, x], dim=-1)
    out = conv_module.conv(full)
    new_state = full[..., -padding:]
    return out, new_state


# -----------------------------------------------------------------------------
# Streaming causal transconv: overlap-add at chunk boundary
# -----------------------------------------------------------------------------


def streaming_causal_transconv(
    transconv_module: nn.Module,
    x: torch.Tensor,
    overlap_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run causal transconv and overlap-add with previous chunk tail.
    overlap_state shape [B, C_out, right_pad]. right_pad = kernel_size - stride.
    overlap_state is bias-free to avoid double-counting bias in the overlap region.
    Returns (out [B, C_out, T*stride], new_overlap [B, C_out, right_pad]).
    Note: this implementation avoids https://github.com/NVIDIA/TensorRT-Incubator/issues/565 @ 2026.03.26
    """
    if overlap_state.shape[0] != x.shape[0]:
        overlap_state = overlap_state[: x.shape[0]]
    raw = transconv_module.conv(x)  # length (M-1)*S + K, includes bias
    rp = transconv_module.right_pad
    if rp > 0:
        # Functional form (no in-place slice write) is more stable for ONNX/TRT.
        head = raw[..., :rp] + overlap_state
        body = raw[..., rp:-rp]
        output = torch.cat([head, body], dim=-1)
    else:
        output = raw
    if rp > 0:
        new_overlap = raw[..., -rp:].clone()
        bias = getattr(transconv_module.conv, "bias", None)
        if bias is not None:
            new_overlap = new_overlap - bias.view(1, -1, 1)
    else:
        new_overlap = overlap_state
    return output, new_overlap


# -----------------------------------------------------------------------------
# Code2Wav streaming wrapper (chunk_T=4 fixed)
# -----------------------------------------------------------------------------

# State layout: 2 * num_hidden_layers KV + NUM_CONV + NUM_TRANSCONV (e.g. 8 layers → 37 tensors).
NUM_CONV = 17
NUM_TRANSCONV = 4
CHUNK_T = 4
SAMPLES_PER_CHUNK = CHUNK_T * 1920  # 7680
COLD_START_DUMMY_PAST_LEN = 1


def resolve_code2wav_state_batch_size(batch_size: int) -> int:
    # Static-state-batch workaround removed: keep state batch fully dynamic with request batch.
    return batch_size


def num_code2wav_hidden_layers(decoder: nn.Module) -> int:
    """Match the actual pre_transformer depth (checkpoint may disagree with config)."""
    pt = getattr(decoder, "pre_transformer", None)
    layers = getattr(pt, "layers", None) if pt is not None else None
    if layers is not None and len(layers) > 0:
        return int(len(layers))
    return int(getattr(decoder.config, "num_hidden_layers", 8))


def count_code2wav_state_tensors(decoder: nn.Module) -> int:
    """Total streaming state tensors (KV + conv + transconv)."""
    return 2 * num_code2wav_hidden_layers(decoder) + NUM_CONV + NUM_TRANSCONV


def get_initial_state_shapes(
    decoder: nn.Module,
    batch_size: int = 1,
    past_kv_len: int = COLD_START_DUMMY_PAST_LEN,
    conv_state_batch_size: int | None = None,
):
    """Return list of (name, shape) for code2wav streaming state tensors. Used for ONNX export."""
    cfg = decoder.config
    codebook_dim = getattr(cfg, "codebook_dim", 512)
    latent_dim = cfg.latent_dim
    decoder_dim = cfg.decoder_dim
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(
        cfg, "head_dim", getattr(cfg, "hidden_size", 512) // cfg.num_attention_heads
    )
    B = batch_size
    state_B = batch_size if conv_state_batch_size is None else conv_state_batch_size
    n_layers = num_code2wav_hidden_layers(decoder)
    shapes = []
    for i in range(n_layers):
        shapes.append((f"past_kv_{i}_k", (B, num_kv_heads, past_kv_len, head_dim)))
        shapes.append((f"past_kv_{i}_v", (B, num_kv_heads, past_kv_len, head_dim)))
    shapes.append(("conv_state_0", (state_B, codebook_dim, 2)))
    shapes.append(("conv_state_1", (state_B, latent_dim, 6)))
    shapes.append(("conv_state_2", (state_B, latent_dim, 6)))
    shapes.append(("conv_state_3", (state_B, latent_dim, 6)))
    for block_idx in range(4):
        out_dim = decoder_dim // (2 ** (block_idx + 1))
        shapes.append((f"conv_state_{4 + block_idx * 3 + 0}", (state_B, out_dim, 6)))
        shapes.append((f"conv_state_{4 + block_idx * 3 + 1}", (state_B, out_dim, 18)))
        shapes.append((f"conv_state_{4 + block_idx * 3 + 2}", (state_B, out_dim, 54)))
    shapes.append(("conv_state_16", (state_B, decoder_dim // (2**4), 6)))
    for block_idx in range(4):
        out_dim = decoder_dim // (2 ** (block_idx + 1))
        rp = [8, 5, 4, 3][block_idx]
        shapes.append((f"transconv_overlap_{block_idx}", (state_B, out_dim, rp)))
    return shapes


def create_initial_states(
    decoder: nn.Module, device: torch.device, dtype: torch.dtype, batch_size: int = 1
):
    """Create zero-initialized state tensors for the streaming wrapper."""
    shapes = get_initial_state_shapes(
        decoder,
        batch_size=batch_size,
        past_kv_len=COLD_START_DUMMY_PAST_LEN,
    )
    return [torch.zeros(s, device=device, dtype=dtype) for _, s in shapes]


def freeze_quantizer_codebooks_for_export(decoder: nn.Module) -> None:
    """
    Materialize codebook embedding tables for export so ONNX graph does not carry
    per-call clamp/divide subgraphs from cluster_usage + embedding_sum.
    """
    quantizer = getattr(decoder, "quantizer", None)
    if quantizer is None:
        return

    rvq_modules = []
    for name in ("rvq_first", "rvq_rest"):
        mod = getattr(quantizer, name, None)
        if mod is not None:
            rvq_modules.append(mod)

    for rvq in rvq_modules:
        vq = getattr(rvq, "vq", None)
        layers = getattr(vq, "layers", None)
        if layers is None:
            continue
        for layer in layers:
            codebook = getattr(layer, "_codebook", None)
            if codebook is None:
                continue
            with torch.no_grad():
                embedding = codebook.embedding_sum / codebook.cluster_usage.clamp(
                    min=codebook.epsilon
                ).unsqueeze(1)
            if hasattr(codebook, "inference_embedding"):
                codebook.inference_embedding = embedding
            else:
                codebook.register_buffer("inference_embedding", embedding)
            if not hasattr(codebook, "_orig_decode_for_export"):
                codebook._orig_decode_for_export = codebook.decode

            def _decode_with_frozen_embedding(
                self, codes: torch.Tensor
            ) -> torch.Tensor:
                return F.embedding(codes, self.inference_embedding)

            codebook.decode = types.MethodType(_decode_with_frozen_embedding, codebook)


def _build_rvq_embedding_table(rvq: nn.Module) -> torch.Tensor | None:
    """
    Build stacked embedding table [n_q, vocab, dim] from RVQ codebooks.
    Returns None when RVQ/codebooks are unavailable.
    """
    if rvq is None:
        return None
    n_q = int(getattr(rvq, "n_q", 0))
    vq = getattr(rvq, "vq", None)
    layers = getattr(vq, "layers", None) if vq is not None else None
    if layers is None or n_q <= 0:
        return None

    emb_list = []
    for idx in range(n_q):
        codebook = getattr(layers[idx], "_codebook", None)
        if codebook is None:
            return None
        emb = getattr(codebook, "inference_embedding", None)
        if emb is None:
            with torch.no_grad():
                emb = codebook.embedding_sum / codebook.cluster_usage.clamp(
                    min=codebook.epsilon
                ).unsqueeze(1)
        emb_list.append(emb.detach())
    if not emb_list:
        return None
    return torch.stack(emb_list, dim=0).contiguous()


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


def _build_decoder_rope_embeddings_export(
    rotary_emb: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = rotary_emb.inv_freq.to(
        device=hidden_states.device, dtype=torch.float32
    ).reshape(1, 1, -1)
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


def _apply_rotary_pos_emb_export(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_states = (query_states * cos) + (
        _rotate_half_export(query_states, half_dim) * sin
    )
    key_states = (key_states * cos) + (_rotate_half_export(key_states, half_dim) * sin)
    return query_states, key_states


def _run_decoder_attention_export(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    cache: SlidingWindowKVCache,
    cache_position: torch.Tensor,
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
    query_states, key_states = _apply_rotary_pos_emb_export(
        query_states,
        key_states,
        cos,
        sin,
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
    attn_weights = F.softmax(attn_weights, dim=-1)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).reshape(
        batch, seq_len, attn_module.o_proj.in_features
    )
    return attn_module.o_proj(attn_output)


def _run_decoder_transformer_layer_export(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: SlidingWindowKVCache,
    cache_position: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = _run_decoder_attention_export(
        layer.self_attn,
        hidden_states,
        position_embeddings,
        attention_mask,
        cache,
        cache_position,
    )
    hidden_states = residual + layer.self_attn_layer_scale(hidden_states)

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + layer.mlp_layer_scale(hidden_states)
    return hidden_states


_ATTN_MASK_SLICE_PATCHED = False
_QUANTIZER_SPLIT_PATCHED = False
_ROPE_RESHAPE_PATCHED = False
_SNAKEBETA_PATCHED = False


def patch_quantizer_decode_no_split_for_export(decoder: nn.Module) -> None:
    """
    Avoid Split-heavy quantizer decode graph by using narrow/index based slicing.
    """
    quantizer = getattr(decoder, "quantizer", None)
    if quantizer is None:
        return

    n_sem = int(getattr(quantizer, "n_q_semantic", 1))
    n_aco = int(getattr(quantizer, "n_q_acoustic", 0))

    def _quantizer_decode_no_split(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [B, K, T]
        codes_sem = codes.narrow(1, 0, n_sem)
        quantized = self.rvq_first.decode(codes_sem)
        if n_aco > 0:
            codes_aco = codes.narrow(1, n_sem, n_aco)
            quantized = quantized + self.rvq_rest.decode(codes_aco)
        return quantized

    quantizer.decode = types.MethodType(_quantizer_decode_no_split, quantizer)

    for rvq_name in ("rvq_first", "rvq_rest"):
        rvq = getattr(quantizer, rvq_name, None)
        if rvq is None:
            continue
        n_q = int(getattr(rvq, "n_q", 0))
        vq = getattr(rvq, "vq", None)
        layers = getattr(vq, "layers", None)
        if vq is None or layers is None or n_q <= 0:
            continue

        def _rvq_decode_no_split(self, codes: torch.Tensor, _n_q=n_q) -> torch.Tensor:
            # codes: [B, n_q, T]
            codes_t = codes.transpose(0, 1)  # [n_q, B, T]
            quantized = None
            for idx in range(_n_q):
                layer = self.vq.layers[idx]
                cur = layer.decode(codes_t[idx])
                quantized = cur if quantized is None else (quantized + cur)
            return self.output_proj(quantized)

        rvq.decode = types.MethodType(_rvq_decode_no_split, rvq)


def patch_quantizer_split_decode_for_export(decoder: nn.Module) -> None:
    """
    Replace `codes[:, :n]` / `codes[:, n:]` slicing with a fixed-size split.
    This keeps semantics identical while avoiding dynamic Slice front-nodes.
    """
    global _QUANTIZER_SPLIT_PATCHED
    if _QUANTIZER_SPLIT_PATCHED:
        return

    quantizer = getattr(decoder, "quantizer", None)
    if quantizer is None:
        return

    n_sem = int(getattr(quantizer, "n_q_semantic", 1))
    n_aco = int(getattr(quantizer, "n_q_acoustic", 0))
    rvq_first = getattr(quantizer, "rvq_first", None)
    rvq_rest = getattr(quantizer, "rvq_rest", None)
    if rvq_first is None or rvq_rest is None:
        return

    def _decode_with_fixed_split(self, codes: torch.Tensor) -> torch.Tensor:
        codes_first, codes_rest = torch.split(codes, [n_sem, n_aco], dim=1)
        quantized = self.rvq_first.decode(codes_first)
        if n_aco > 0:
            quantized = quantized + self.rvq_rest.decode(codes_rest)
        return quantized

    quantizer.decode = types.MethodType(_decode_with_fixed_split, quantizer)
    _QUANTIZER_SPLIT_PATCHED = True


def patch_decoder_attention_mask_slice_for_export() -> None:
    """
    Patch eager attention to avoid `attention_mask[..., :key_len]` slicing in graph.
    We feed externally prepared mask with exact key length in export/runtime path.
    """
    global _ATTN_MASK_SLICE_PATCHED
    if _ATTN_MASK_SLICE_PATCHED or qwen_tokenizer_v2 is None:
        return

    base_impl = getattr(qwen_tokenizer_v2, "eager_attention_forward", None)
    if base_impl is None:
        return

    def _eager_attention_forward_no_mask_slice(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = qwen_tokenizer_v2.repeat_kv(key, module.num_key_value_groups)
        value_states = qwen_tokenizer_v2.repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    qwen_tokenizer_v2.eager_attention_forward = _eager_attention_forward_no_mask_slice
    _ATTN_MASK_SLICE_PATCHED = True


def patch_decoder_rope_reshape_for_export() -> None:
    """
    Rewrite RoPE helpers with reshape+broadcast style to avoid unsqueeze-heavy subgraphs.
    """
    global _ROPE_RESHAPE_PATCHED
    if _ROPE_RESHAPE_PATCHED or qwen_tokenizer_v2 is None:
        return

    rope_cls = getattr(
        qwen_tokenizer_v2, "Qwen3TTSTokenizerV2DecoderRotatoryEmbedding", None
    )
    apply_rope = getattr(qwen_tokenizer_v2, "apply_rotary_pos_emb", None)
    rotate_half = getattr(qwen_tokenizer_v2, "rotate_half", None)
    if rope_cls is None or apply_rope is None or rotate_half is None:
        return

    @torch.no_grad()
    def _rope_forward_reshape(self, x, position_ids):
        bsz = int(position_ids.shape[0])
        seq = int(position_ids.shape[1])
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32).reshape(
            1, 1, -1
        )  # [1,1,D]
        pos = position_ids.to(device=x.device, dtype=torch.float32).reshape(
            bsz, seq, 1
        )  # [B,T,1]

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = pos * inv_freq  # [B,T,D]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _apply_rotary_pos_emb_reshape(
        q, k, cos, sin, position_ids=None, unsqueeze_dim=1
    ):
        del position_ids, unsqueeze_dim
        # q/k: [B, H, T, D], cos/sin: [B, T, D]
        bsz = int(cos.shape[0])
        seq = int(cos.shape[1])
        dim = int(cos.shape[2])
        cos_r = cos.reshape(bsz, 1, seq, dim)
        sin_r = sin.reshape(bsz, 1, seq, dim)
        q_embed = (q * cos_r) + (rotate_half(q) * sin_r)
        k_embed = (k * cos_r) + (rotate_half(k) * sin_r)
        return q_embed, k_embed

    rope_cls.forward = _rope_forward_reshape
    qwen_tokenizer_v2.apply_rotary_pos_emb = _apply_rotary_pos_emb_reshape
    _ROPE_RESHAPE_PATCHED = True


def patch_decoder_snakebeta_for_export(decoder: nn.Module) -> None:
    """
    Rewrite SnakeBeta broadcast into static reshapes so ONNX does not emit
    long Unsqueeze/Add/Reciprocal chains for every decoder activation.
    """
    global _SNAKEBETA_PATCHED
    if _SNAKEBETA_PATCHED:
        return

    patched = False
    for module in decoder.modules():
        if module.__class__.__name__ != "SnakeBeta":
            continue
        if not hasattr(module, "alpha") or not hasattr(module, "beta"):
            continue

        channels = int(module.alpha.numel())

        def _snakebeta_forward(
            self, hidden_states: torch.Tensor, _channels: int = channels
        ) -> torch.Tensor:
            alpha = torch.exp(self.alpha.reshape(1, _channels, 1))
            # exp(beta) is strictly positive, so exp(-beta) is the same scaling
            # as 1 / exp(beta) without the extra broadcast + reciprocal chain.
            beta_inv = torch.exp(-self.beta.reshape(1, _channels, 1))
            sin_term = torch.sin(hidden_states * alpha)
            return hidden_states + (sin_term * sin_term) * beta_inv

        module.forward = types.MethodType(_snakebeta_forward, module)
        patched = True

    if patched:
        _SNAKEBETA_PATCHED = True


class Code2WavStreamingWrapper(nn.Module):
    """
    Stateful code2wav for ONNX/TRT: codes [B,16,T] + cache_position + c2w_attention_bias + state tensors,
    output wav [B, 7680] + updated states (layout matches get_initial_state_shapes).
    """

    def __init__(
        self,
        decoder: nn.Module,
        window_size: int = 72,
        *,
        emit_delta_kv: bool = False,
    ):
        super().__init__()
        self.decoder = decoder
        patch_decoder_snakebeta_for_export(self.decoder)
        freeze_quantizer_codebooks_for_export(self.decoder)
        patch_quantizer_decode_no_split_for_export(self.decoder)
        self.window_size = window_size
        cfg = decoder.config
        self.num_layers = num_code2wav_hidden_layers(decoder)
        self.emit_delta_kv = emit_delta_kv
        # KV states are 2 * num_layers; conv/transconv follow immediately (not a fixed 16-slot pad).
        self._num_kv_state_tensors = 2 * self.num_layers
        self.hidden_size = getattr(cfg, "hidden_size", 512)
        self.latent_dim = cfg.latent_dim
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(
            cfg, "head_dim", self.hidden_size // cfg.num_attention_heads
        )
        self._use_direct_quantizer_decode = True
        self._rvq_first_proj = None
        self._rvq_rest_proj = None
        self._setup_direct_quantizer_decode()
        # Export-only identity bridge. A real Conv op is harder for ONNX/TRT to
        # optimize away than clone/reduce identity chains, while keeping batch dynamic.
        self.decoder_bridge = nn.Conv1d(
            self.latent_dim,
            self.latent_dim,
            kernel_size=1,
            groups=self.latent_dim,
            bias=False,
        )
        with torch.no_grad():
            self.decoder_bridge.weight.fill_(1.0)
        self.decoder_bridge.weight.requires_grad_(False)
        # Force eager attention for ONNX
        if hasattr(cfg, "_attn_implementation"):
            self._saved_attn_impl = getattr(cfg, "_attn_implementation", None)
        setattr(cfg, "_attn_implementation", "eager")

    def _setup_direct_quantizer_decode(self) -> None:
        def _upsert_buffer(name: str, value: torch.Tensor) -> None:
            if name in self._buffers:
                self._buffers[name] = value
                return
            if hasattr(self, name):
                delattr(self, name)
            self.register_buffer(name, value, persistent=False)

        quantizer = getattr(self.decoder, "quantizer", None)
        if quantizer is None:
            self._use_direct_quantizer_decode = False
            return

        n_sem = int(getattr(quantizer, "n_q_semantic", 1))
        n_aco = int(getattr(quantizer, "n_q_acoustic", 0))
        rvq_first = getattr(quantizer, "rvq_first", None)
        rvq_rest = getattr(quantizer, "rvq_rest", None)
        first_table = _build_rvq_embedding_table(rvq_first)
        rest_table = _build_rvq_embedding_table(rvq_rest) if n_aco > 0 else None
        first_proj = (
            getattr(rvq_first, "output_proj", None) if rvq_first is not None else None
        )
        rest_proj = (
            getattr(rvq_rest, "output_proj", None) if rvq_rest is not None else None
        )

        if (
            first_table is None
            or first_proj is None
            or (n_aco > 0 and (rest_table is None or rest_proj is None))
        ):
            self._use_direct_quantizer_decode = False
            return

        _upsert_buffer("_rvq_first_table", first_table)
        if rest_table is not None:
            _upsert_buffer("_rvq_rest_table", rest_table)
        self._rvq_first_proj = first_proj
        self._rvq_rest_proj = rest_proj
        _upsert_buffer("_idx_sem", torch.arange(0, n_sem, dtype=torch.long))
        if n_aco > 0:
            _upsert_buffer(
                "_idx_aco", torch.arange(n_sem, n_sem + n_aco, dtype=torch.long)
            )

    @staticmethod
    def _decode_rvq_from_table(
        table: torch.Tensor,
        proj: nn.Module,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        # table: [n_q, vocab, d], codes: [B, n_q, T]
        codes_t = codes.transpose(0, 1).long()  # [n_q, B, T]
        n_q, _, d = table.shape
        bsz = codes_t.shape[1]
        table_expand = table.unsqueeze(1).expand(n_q, bsz, -1, -1)
        gather_idx = codes_t.unsqueeze(-1).expand(-1, -1, -1, d)
        quantized = torch.gather(table_expand, 2, gather_idx).sum(dim=0)  # [B, T, d]
        quantized = quantized.permute(0, 2, 1).contiguous()  # [B, d, T]
        return proj(quantized)

    def _decode_quantizer_direct(self, codes: torch.Tensor) -> torch.Tensor:
        codes_sem = torch.index_select(codes, 1, self._idx_sem)
        quantized = self._decode_rvq_from_table(
            self._rvq_first_table, self._rvq_first_proj, codes_sem
        )
        if (
            self._idx_aco is not None
            and self._rvq_rest_table is not None
            and self._rvq_rest_proj is not None
        ):
            codes_aco = torch.index_select(codes, 1, self._idx_aco)
            quantized = quantized + self._decode_rvq_from_table(
                self._rvq_rest_table,
                self._rvq_rest_proj,
                codes_aco,
            )
        return quantized

    def forward(
        self,
        codes: torch.Tensor,
        cache_position: torch.Tensor,
        c2w_attention_bias: torch.Tensor,
        *states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        codes: [B, 16, T], cache_position: [T] or [B, T] (absolute positions for this chunk).
        c2w_attention_bias: [B, 1, T, T_pad+T] additive mask built outside graph.
        states: 2*num_hidden_layers KV + 17 conv + 4 transconv.
        Returns: (wav [B, 7680], *new_states).
        """
        B = codes.shape[0]
        device = codes.device
        dtype = (
            codes.dtype if codes.dtype.is_floating_point else torch.get_default_dtype()
        )

        # Unpack states
        kv_list = [(states[2 * i], states[2 * i + 1]) for i in range(self.num_layers)]
        off = self._num_kv_state_tensors
        conv_state_templates = list(states[off : off + NUM_CONV])
        transconv_state_templates = list(states[off + NUM_CONV :])
        conv_states = list(conv_state_templates)
        transconv_states = list(transconv_state_templates)

        # 1. Quantizer (stateless)
        if self._use_direct_quantizer_decode:
            hidden = self._decode_quantizer_direct(codes)
        else:
            hidden = self.decoder.quantizer.decode(codes)  # [B, codebook_dim, 4]

        # 2. Pre-conv streaming
        hidden, conv_states[0] = streaming_causal_conv(
            self.decoder.pre_conv, hidden, conv_states[0]
        )
        hidden = hidden.transpose(1, 2)  # [B, 4, latent_dim]

        # 3. Transformer with sliding-window KV cache
        cache = SlidingWindowKVCache(kv_list, self.window_size)

        hidden = self.decoder.pre_transformer.input_proj(hidden)  # [B, 4, hidden_size]
        # position_ids [1, 4] to match transformer's expectation (same as cache_position.unsqueeze(0))
        position_ids = (
            cache_position.unsqueeze(0) if cache_position.dim() == 1 else cache_position
        )
        if position_ids.shape[0] != B:
            position_ids = position_ids.expand(B, -1)
        position_embeddings = _build_decoder_rope_embeddings_export(
            self.decoder.pre_transformer.rotary_emb,
            hidden,
            position_ids,
        )

        causal_mask = c2w_attention_bias.contiguous()

        for layer in self.decoder.pre_transformer.layers:
            hidden = _run_decoder_transformer_layer_export(
                layer,
                hidden,
                causal_mask,
                cache,
                cache_position,
                position_embeddings,
            )

        hidden = self.decoder.pre_transformer.norm(hidden)
        hidden = self.decoder.pre_transformer.output_proj(hidden)  # [B, 4, latent_dim]
        hidden = hidden.permute(0, 2, 1)  # [B, latent_dim, 4]

        # Create an explicit tensor boundary between transformer and decoder path.
        hidden = hidden.contiguous()
        # A channel-wise 1x1 Conv is exact identity and survives export/simplify
        # better than clone/reduce no-ops.
        hidden = self.decoder_bridge(hidden)

        # 4. Upsample blocks (2x): transconv (no overlap) + ConvNeXt with streaming
        hidden = self._upsample_streaming(hidden, conv_states)

        # 5. Decoder: initial conv + 4 blocks + SnakeBeta + final conv
        hidden, conv_states[3] = streaming_causal_conv(
            self.decoder.decoder[0], hidden, conv_states[3]
        )
        for block_idx in range(4):
            hidden, conv_states, transconv_states = self._decoder_block_streaming(
                block_idx, hidden, conv_states, transconv_states
            )
        hidden = self.decoder.decoder[5](hidden)  # SnakeBeta
        wav, conv_states[16] = streaming_causal_conv(
            self.decoder.decoder[6], hidden, conv_states[16]
        )
        # fix myelin issue by clamp
        wav = wav.reshape(B, -1)
        # now we can clamp without myelin issue
        wav = wav.clamp(min=-1, max=1)

        # Pack new states
        new_kv = []
        for i in range(self.num_layers):
            if self.emit_delta_kv:
                k, v = cache.get_delta(i)
            else:
                k, v = cache.get_present(i)
            new_kv.append(k)
            new_kv.append(v)
        return (wav,) + tuple(new_kv) + tuple(conv_states) + tuple(transconv_states)

    def _upsample_streaming(
        self,
        hidden: torch.Tensor,
        conv_states: List[torch.Tensor],
    ) -> torch.Tensor:
        for i, blocks in enumerate(self.decoder.upsample):
            transconv, convnext = blocks[0], blocks[1]
            hidden = transconv(hidden)  # right_pad=0 for (2,2)
            inp = hidden
            hidden, conv_states[1 + i] = streaming_causal_conv(
                convnext.dwconv, hidden, conv_states[1 + i]
            )
            hidden = hidden.permute(0, 2, 1)
            hidden = convnext.norm(hidden)
            hidden = convnext.pwconv1(hidden)
            hidden = convnext.act(hidden)
            hidden = convnext.pwconv2(hidden)
            hidden = convnext.gamma * hidden
            hidden = hidden.permute(0, 2, 1)
            hidden = inp + hidden
        return hidden

    def _decoder_block_streaming(
        self,
        block_idx: int,
        hidden: torch.Tensor,
        conv_states: List[torch.Tensor],
        transconv_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        block = self.decoder.decoder[1 + block_idx].block
        hidden = block[0](hidden)  # SnakeBeta
        hidden, transconv_states[block_idx] = streaming_causal_transconv(
            block[1], hidden, transconv_states[block_idx]
        )
        for ru_idx in range(3):
            res_unit = block[2 + ru_idx]
            residual = hidden
            hidden = res_unit.act1(hidden)
            hidden, conv_states[4 + block_idx * 3 + ru_idx] = streaming_causal_conv(
                res_unit.conv1, hidden, conv_states[4 + block_idx * 3 + ru_idx]
            )
            hidden = res_unit.act2(hidden)
            hidden = res_unit.conv2(hidden)  # kernel 1, no state
            hidden = residual + hidden
        return hidden, conv_states, transconv_states
