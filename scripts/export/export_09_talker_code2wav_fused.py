#!/usr/bin/env python3
"""
[Step 09] Export Talker Unified + Code2Wav (chunk_T=1) fused ONNX.

Single engine per decode step: talker (prefill+decode+CP+codec_sum) -> full_codec -> code2wav(1 frame) -> wav.

Packed KV format: Talker KV and C2W KV are each packed into a single 5-D tensor,
reducing TRT I/O binding count from ~201 to ~61.  Conv/transconv states remain
individual I/O due to heterogeneous shapes.

Inputs:
    input_embeds, position_ids, attention_bias, cache_position,
    c2w_attention_bias,
    talker_past_kv     — [B, num_layers*2, kv_heads, S_past, head_dim]
    c2w_past_kv        — [B, n_c2w*2, c2w_heads, S_c2w, c2w_head_dim]
    c2w_conv_state_*   — 17 heterogeneous conv state tensors
    c2w_transconv_overlap_* — 4 heterogeneous transconv overlap tensors

Outputs:
    wav, codec_sum, full_codec, hidden, logits, updated_token_counts,
    talker_new_kv      — [B, num_layers*2, kv_heads, S_step, head_dim]
    c2w_new_kv         — [B, n_c2w*2, c2w_heads, chunk_t, c2w_head_dim]
    c2w_new_conv_state_*, c2w_new_transconv_overlap_*

Depends on: tokenizer (code2wav decoder) + TTS variant (talker).

Also writes ``triton_manifest.json``; Phase C requires this file.

**Precision policy** (see ``docs/architecture.md``):

- **ONNX graph float I/O** is exported as **FP32** (``utils.ONNX_EXPORT_DTYPE``).
- ``triton_manifest.json`` field ``engine_dtype`` refers to **TRT builder compute**.
- ``triton_io_float_dtype`` must match the ONNX/TRT engine I/O.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

_scripts_export = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_export))
sys.path.insert(0, str(_scripts_export.parent / "python"))

from code2wav_streaming import (
    COLD_START_DUMMY_PAST_LEN,
    Code2WavStreamingWrapper,
    get_initial_state_shapes,
    NUM_CONV,
    NUM_TRANSCONV,
    num_code2wav_hidden_layers,
)
from talker_unified_modules import build_talker_unified_fused_module, LOGITS_TOPK
from triton_manifest_io import build_manifest_for_export

from utils import (
    setup_logging,
    resolve_tokenizer_path,
    resolve_model_path,
    ensure_output_dir,
    load_speech_tokenizer,
    load_tts_model,
    patch_decoder_transconv_for_trt,
    export_onnx,
    verify_onnx,
    to_numpy,
    resolve_device,
    add_common_args,
    MODEL_VARIANTS,
    ONNX_EXPORT_DTYPE,
)

logger = logging.getLogger("onnx_export")

FUSED_CHUNK_T = 1


class TalkerCode2WavFusedONNX(nn.Module):
    """TalkerUnifiedFusedONNX + Code2WavStreamingWrapper (T=1 per forward).

    Packed KV interface: Talker and C2W KV caches are each passed as a
    single 5-D tensor [B, L*2, H, S, D] instead of 2*L individual tensors.
    Inside forward(), they are unbind'd to feed the sub-modules (these ops
    become Slice nodes in ONNX, essentially free in TRT).
    """

    def __init__(
        self,
        talker_fused: nn.Module,
        code2wav: Code2WavStreamingWrapper,
        n_c2w_layers: int,
    ):
        super().__init__()
        self.talker_fused = talker_fused
        self.code2wav = code2wav
        self.num_layers = talker_fused.num_layers
        self.n_c2w_layers = n_c2w_layers

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_bias: torch.Tensor,
        token_counts: torch.Tensor,
        gumbel_noise: torch.Tensor,
        cp_gumbel_noise: torch.Tensor,
        temperature: torch.Tensor,
        penalty: torch.Tensor,
        cache_position: torch.Tensor,
        c2w_attention_bias: torch.Tensor,
        talker_past_kv: torch.Tensor,
        c2w_past_kv: torch.Tensor,
        *conv_transconv_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        talker_past_kv: [B, num_layers*2, kv_heads, S_past, head_dim]
        c2w_past_kv:    [B, n_c2w*2, c2w_heads, S_c2w, c2w_head_dim]
        conv_transconv_states: 17 conv + 4 transconv (heterogeneous shapes)
        """
        n = self.num_layers
        n_c2w = self.n_c2w_layers

        # Unpack talker KV: [B, L*2, H, S, D] -> list of [B, H, S, D]
        past_kv = [talker_past_kv[:, i, :, :, :] for i in range(n * 2)]

        codec_sum, full_codec, hidden, logits, updated_token_counts, *present_kv = (
            self.talker_fused(
                input_embeds,
                position_ids,
                token_counts,
                gumbel_noise,
                cp_gumbel_noise,
                temperature,
                penalty,
                attention_bias,
                *past_kv,
            )
        )

        # Pack talker delta KV: list of [B, H, S_step, D] -> [B, L*2, H, S_step, D]
        talker_new_kv = torch.stack(present_kv, dim=1)

        # Unpack C2W KV: [B, N*2, H, S, D] -> list of [B, H, S, D]
        c2w_kv_list = [c2w_past_kv[:, i, :, :, :] for i in range(n_c2w * 2)]

        _cb = 2048
        fc = full_codec.long().clamp(0, _cb - 1)
        codes = fc.unsqueeze(-1)

        c2w_out = self.code2wav(
            codes,
            cache_position,
            c2w_attention_bias,
            *c2w_kv_list,
            *conv_transconv_states,
        )
        wav = c2w_out[0]

        # Pack C2W delta KV
        c2w_new_kv = list(c2w_out[1 : 1 + 2 * n_c2w])
        c2w_new_kv = torch.stack(c2w_new_kv, dim=1)

        new_conv_transconv = c2w_out[1 + 2 * n_c2w :]

        return (
            wav,
            codec_sum,
            full_codec,
            hidden,
            logits,
            updated_token_counts,
            talker_new_kv,
            c2w_new_kv,
            *new_conv_transconv,
        )


def _export_talker_code2wav_fused_onnx(
    model,
    variant: str,
    output_dir: Path,
    device: str = "cpu",
    opset_version: int = 18,
    engine_dtype: str = "bf16",
    triton_io_float_dtype: str = "bf16",
) -> str:
    tokenizer_path = resolve_tokenizer_path(None)
    tokenizer_model = load_speech_tokenizer(
        tokenizer_path, device=device, dtype=torch.float32
    )
    decoder = tokenizer_model.decoder.to(device).eval()
    patch_decoder_transconv_for_trt(decoder)
    code2wav = (
        Code2WavStreamingWrapper(
            decoder,
            emit_delta_kv=True,
        )
        .to(device)
        .eval()
    )

    talker_fused, num_layers, hidden_size, num_kv_heads, head_dim = (
        build_talker_unified_fused_module(model, device=device)
    )
    vocab_size = model.talker.model.config.vocab_size
    cp_num_stages = talker_fused.cp.num_stages
    n_c2w = num_code2wav_hidden_layers(decoder)

    c2w_cfg = decoder.config
    c2w_kv_heads = c2w_cfg.num_key_value_heads
    c2w_head_dim = getattr(
        c2w_cfg,
        "head_dim",
        getattr(c2w_cfg, "hidden_size", 512) // c2w_cfg.num_attention_heads,
    )

    fused = TalkerCode2WavFusedONNX(talker_fused, code2wav, n_c2w).to(device).eval()

    # ---- Dummy inputs ----
    B, one, S_past = 1, 1, 0
    dummy_embeds = torch.randn(
        B, one, hidden_size, device=device, dtype=ONNX_EXPORT_DTYPE
    )
    position_ids = torch.full((B, 3, one, 1), S_past, device=device, dtype=torch.long)
    attention_bias = torch.zeros(
        B, 1, one, S_past + one, device=device, dtype=ONNX_EXPORT_DTYPE
    )

    dummy_token_counts = torch.zeros(B, vocab_size, device=device, dtype=torch.int64)
    dummy_gumbel_noise = torch.zeros(B, LOGITS_TOPK, device=device, dtype=torch.float32)
    dummy_cp_gumbel_noise = torch.zeros(
        B,
        cp_num_stages,
        LOGITS_TOPK,
        device=device,
        dtype=torch.float32,
    )
    dummy_temperature = torch.ones(B, 1, device=device, dtype=torch.float32)
    dummy_penalty = torch.full((B, 1), 1.05, device=device, dtype=torch.float32)

    # Packed talker KV: [B, num_layers*2, kv_heads, S_past, head_dim]
    # Use S_past=0 not allowed by ONNX (zero-size dim), use 0 dynamic
    talker_past_kv_dummy = torch.zeros(
        B,
        num_layers * 2,
        num_kv_heads,
        max(S_past, 0),
        head_dim,
        device=device,
        dtype=ONNX_EXPORT_DTYPE,
    )

    cache_position = torch.zeros(B, FUSED_CHUNK_T, device=device, dtype=torch.float32)
    TRACE_C2W_PAST_LEN = 1
    c2w_attention_bias = torch.zeros(
        B,
        1,
        FUSED_CHUNK_T,
        TRACE_C2W_PAST_LEN + FUSED_CHUNK_T,
        device=device,
        dtype=ONNX_EXPORT_DTYPE,
    )

    # Packed C2W KV: [B, n_c2w*2, c2w_heads, c2w_past_len, c2w_head_dim]
    c2w_past_kv_dummy = torch.randn(
        B,
        n_c2w * 2,
        c2w_kv_heads,
        TRACE_C2W_PAST_LEN,
        c2w_head_dim,
        device=device,
        dtype=ONNX_EXPORT_DTYPE,
    )

    # Conv/transconv states (heterogeneous shapes, skip KV entries)
    c2w_state_batch = B
    state_shapes = get_initial_state_shapes(
        decoder,
        batch_size=B,
        past_kv_len=TRACE_C2W_PAST_LEN,
        conv_state_batch_size=c2w_state_batch,
    )
    state_shapes_cold = get_initial_state_shapes(
        decoder,
        batch_size=B,
        past_kv_len=COLD_START_DUMMY_PAST_LEN,
        conv_state_batch_size=c2w_state_batch,
    )
    conv_transconv_names = []
    conv_transconv_tensors = []
    conv_transconv_names_cold = []
    for name, shape in state_shapes:
        if "past_kv" not in name:
            conv_transconv_names.append(name)
            conv_transconv_tensors.append(
                torch.randn(shape, device=device, dtype=ONNX_EXPORT_DTYPE)
            )
    for name, _ in state_shapes_cold:
        if "past_kv" not in name:
            conv_transconv_names_cold.append(name)

    dummy_inputs = (
        dummy_embeds,
        position_ids,
        attention_bias,
        dummy_token_counts,
        dummy_gumbel_noise,
        dummy_cp_gumbel_noise,
        dummy_temperature,
        dummy_penalty,
        cache_position,
        c2w_attention_bias,
        talker_past_kv_dummy,
        c2w_past_kv_dummy,
        *conv_transconv_tensors,
    )

    with torch.no_grad():
        out = fused(*dummy_inputs)

    # ---- I/O names ----
    input_names = [
        "input_embeds",
        "position_ids",
        "attention_bias",
        "token_counts",
        "gumbel_noise",
        "cp_gumbel_noise",
        "temperature",
        "penalty",
        "cache_position",
        "c2w_attention_bias",
        "talker_past_kv",
        "c2w_past_kv",
    ]
    for name in conv_transconv_names_cold:
        input_names.append(f"c2w_{name}")

    output_names = [
        "wav",
        "codec_sum",
        "full_codec",
        "hidden",
        "logits",
        "updated_token_counts",
        "talker_new_kv",
        "c2w_new_kv",
    ]
    for i in range(NUM_CONV):
        output_names.append(f"c2w_new_conv_state_{i}")
    for i in range(NUM_TRANSCONV):
        output_names.append(f"c2w_new_transconv_overlap_{i}")

    # ---- Dynamic axes ----
    dynamic_axes = {
        "input_embeds": {0: "batch", 1: "seq"},
        "position_ids": {0: "batch", 1: "three", 2: "seq"},
        "attention_bias": {0: "batch", 2: "seq", 3: "key_total"},
        "token_counts": {0: "batch"},
        "gumbel_noise": {0: "batch"},
        "cp_gumbel_noise": {0: "batch"},
        "temperature": {0: "batch"},
        "penalty": {0: "batch"},
        "cache_position": {0: "batch", 1: "chunk_t"},
        "c2w_attention_bias": {0: "batch", 2: "chunk_t", 3: "c2w_key_total"},
        "talker_past_kv": {0: "batch", 3: "S_past"},
        "c2w_past_kv": {0: "batch", 3: "c2w_past_len"},
        "wav": {0: "batch"},
        "codec_sum": {0: "batch"},
        "full_codec": {0: "batch"},
        "hidden": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
        "updated_token_counts": {0: "batch"},
        "talker_new_kv": {0: "batch", 3: "S_step"},
        "c2w_new_kv": {0: "batch", 3: "chunk_t"},
    }
    for name in conv_transconv_names_cold:
        key = f"c2w_{name}"
        if key in input_names:
            dynamic_axes[key] = {0: "batch"}
    for i in range(NUM_CONV):
        dynamic_axes[f"c2w_new_conv_state_{i}"] = {0: "batch"}
    for i in range(NUM_TRANSCONV):
        dynamic_axes[f"c2w_new_transconv_overlap_{i}"] = {0: "batch"}

    onnx_path = str(output_dir / "talker_code2wav_fused.onnx")
    export_onnx(
        model=fused,
        dummy_inputs=dummy_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        onnx_path=onnx_path,
        opset_version=opset_version,
        simplify=False,
        do_constant_folding=False,
    )

    # ---- Manifest ----
    c2w_out_names = []
    for i in range(NUM_CONV):
        c2w_out_names.append(f"c2w_new_conv_state_{i}")
    for i in range(NUM_TRANSCONV):
        c2w_out_names.append(f"c2w_new_transconv_overlap_{i}")
    c2w_sliding_window = getattr(c2w_cfg, "sliding_window", None) or 72

    layout = {
        "num_code2wav_hidden_layers": n_c2w,
        "c2w_state_input_names": [f"c2w_{name}" for name in conv_transconv_names_cold],
        "c2w_state_output_names": c2w_out_names,
        "initial_state_shapes": [
            list(shape) for name, shape in state_shapes_cold if "past_kv" not in name
        ],
        "packed_kv": True,
        "c2w_kv_heads": c2w_kv_heads,
        "c2w_head_dim": c2w_head_dim,
        "c2w_sliding_window": c2w_sliding_window,
        "logits_topk": LOGITS_TOPK,
        "cp_num_stages": cp_num_stages,
    }

    weights_cfg_path = output_dir / "weights" / "config.json"
    weights_cfg: dict = {}
    if weights_cfg_path.is_file():
        with open(weights_cfg_path, encoding="utf-8") as f:
            weights_cfg = json.load(f)
    else:
        logger.warning(
            f"No {weights_cfg_path} — triton_manifest.json talker section uses export defaults"
        )
    manifest = build_manifest_for_export(
        variant,
        weights_cfg,
        layout,
        engine_mode="trt",
        engine_dtype=engine_dtype,
        triton_io_float_dtype=triton_io_float_dtype,
    )
    manifest_path = output_dir / "triton_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote {manifest_path}")

    # ---- Verification case 1: zero past ----
    cpu_fused = fused.cpu().eval()
    cpu_inputs_1 = tuple(t.cpu() for t in dummy_inputs)
    with torch.no_grad():
        cpu_out_1 = cpu_fused(*cpu_inputs_1)

    test_inputs_1 = {}
    for name, tensor in zip(input_names, cpu_inputs_1):
        test_inputs_1[name] = (
            to_numpy(tensor) if tensor.is_floating_point() else tensor.numpy()
        )
    torch_outputs_1 = {
        output_names[i]: to_numpy(cpu_out_1[i]) for i in range(len(output_names))
    }
    ok1 = verify_onnx(onnx_path, test_inputs_1, torch_outputs_1, atol=2e-3, rtol=1e-2)

    # ---- Verification case 2: non-zero past ----
    B2, S_past2 = 1, 4
    c2w_past2 = 5
    cpu_embeds2 = torch.randn(B2, one, hidden_size, dtype=ONNX_EXPORT_DTYPE)
    cpu_pos2 = torch.full((B2, 3, one, 1), S_past2, dtype=torch.long)
    cpu_bias2 = torch.zeros(B2, 1, one, S_past2 + one, dtype=ONNX_EXPORT_DTYPE)
    cpu_bias2[0, :, :, :2] = -1.0e4
    cpu_tc2 = torch.zeros(B2, vocab_size, dtype=torch.int64)
    cpu_gn2 = torch.zeros(B2, LOGITS_TOPK, dtype=torch.float32)
    cpu_cp_gn2 = torch.zeros(B2, cp_num_stages, LOGITS_TOPK, dtype=torch.float32)
    cpu_temp2 = torch.ones(B2, 1, dtype=torch.float32)
    cpu_pen2 = torch.full((B2, 1), 1.05, dtype=torch.float32)
    cpu_cache2 = torch.zeros(B2, FUSED_CHUNK_T, dtype=torch.float32)
    cpu_c2w_bias2 = torch.zeros(
        B2,
        1,
        FUSED_CHUNK_T,
        c2w_past2 + FUSED_CHUNK_T,
        dtype=ONNX_EXPORT_DTYPE,
    )
    cpu_c2w_bias2[0, :, :, :3] = -1.0e4

    talker_kv2 = torch.randn(
        B2,
        num_layers * 2,
        num_kv_heads,
        S_past2,
        head_dim,
        dtype=ONNX_EXPORT_DTYPE,
    )
    c2w_kv2 = torch.randn(
        B2,
        n_c2w * 2,
        c2w_kv_heads,
        c2w_past2,
        c2w_head_dim,
        dtype=ONNX_EXPORT_DTYPE,
    )

    # Conv/transconv states at alternate sizes
    state_shapes2 = get_initial_state_shapes(
        decoder,
        batch_size=B2,
        past_kv_len=c2w_past2,
        conv_state_batch_size=B2,
    )
    conv_transconv_tensors2 = []
    for name, shape in state_shapes2:
        if "past_kv" not in name:
            conv_transconv_tensors2.append(torch.randn(shape, dtype=ONNX_EXPORT_DTYPE))

    cpu_inputs_2 = (
        cpu_embeds2,
        cpu_pos2,
        cpu_bias2,
        cpu_tc2,
        cpu_gn2,
        cpu_cp_gn2,
        cpu_temp2,
        cpu_pen2,
        cpu_cache2,
        cpu_c2w_bias2,
        talker_kv2,
        c2w_kv2,
        *conv_transconv_tensors2,
    )
    with torch.no_grad():
        cpu_out_2 = cpu_fused(*cpu_inputs_2)

    test_inputs_2 = {}
    for name, tensor in zip(input_names, cpu_inputs_2):
        test_inputs_2[name] = (
            to_numpy(tensor) if tensor.is_floating_point() else tensor.numpy()
        )
    torch_outputs_2 = {
        output_names[i]: to_numpy(cpu_out_2[i]) for i in range(len(output_names))
    }
    ok2 = verify_onnx(onnx_path, test_inputs_2, torch_outputs_2, atol=2e-3, rtol=1e-2)

    if ok1 and ok2:
        logger.info(
            "Talker+Code2Wav fused ONNX verification PASSED (base + alternate-prefill cases)"
        )
    else:
        logger.warning("Talker+Code2Wav fused ONNX verification had differences")
    fused.to(device)

    del tokenizer_model
    if device != "cpu":
        torch.cuda.empty_cache()
    return onnx_path


def export_talker_code2wav_fused(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    engine_dtype: str = "bf16",
    triton_io_float_dtype: str = "bf16",
) -> dict:
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant)
    logger.info(f"Loading model: {variant} from {model_path}")
    model = load_tts_model(model_path, device="cpu", dtype=torch.float32)
    onnx_path = _export_talker_code2wav_fused_onnx(
        model,
        variant,
        out_dir,
        device=device,
        engine_dtype=engine_dtype,
        triton_io_float_dtype=triton_io_float_dtype,
    )
    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return {"onnx": onnx_path}


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Export Talker + Code2Wav fused ONNX")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument(
        "--engine-dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp16", "fp32", "fp8"),
        help="TensorRT builder precision written to triton_manifest.json (trtexec --bf16/--fp16/--fp8)",
    )
    parser.add_argument(
        "--triton-io-float-dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp16", "fp32"),
        help="Float I/O binding + Triton TYPE_* (must match Phase B trtexec --inputIOFormats/--outputFormats)",
    )
    add_common_args(parser)
    args = parser.parse_args()
    device = resolve_device(args.device)
    variants = [args.variant] if args.variant else list(MODEL_VARIANTS.keys())
    for variant in variants:
        if variant not in MODEL_VARIANTS:
            logger.error(f"Unknown variant: {variant}")
            continue
        try:
            results = export_talker_code2wav_fused(
                variant,
                args.models_dir,
                args.output_dir,
                device,
                engine_dtype=args.engine_dtype,
                triton_io_float_dtype=args.triton_io_float_dtype,
            )
            for k, v in results.items():
                logger.info(f"[{variant}] {k}: {v}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
