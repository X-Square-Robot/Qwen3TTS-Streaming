#!/usr/bin/env python3
"""Replay a dumped executor call against official PyTorch modules.

Typical usage:

  mamba run -n qwen3-tts python scripts/python/analyze_engine_dump.py \
      --dump /path/to/000123_decode_story_0.pt --compare-c2w

The dump is produced by setting ``ENGINE_DUMP_DIR`` before starting the engine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports("repo", "scripts_export", "scripts_python", "third_party_qwen")

from code2wav_streaming import Code2WavStreamingWrapper, num_code2wav_hidden_layers
from talker_unified_modules import build_talker_unified_fused_module
from utils import (
    load_speech_tokenizer,
    load_tts_model,
    patch_decoder_transconv_for_trt,
    resolve_tokenizer_path,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True, help="Path to one .pt dump payload")
    p.add_argument("--model-path", default=None, help="Official model directory; defaults to dump metadata weights_dir")
    p.add_argument("--tokenizer-path", default=None, help="Speech tokenizer directory for --compare-c2w")
    p.add_argument("--device", default=None, help="cuda/cpu; default: cuda if available else cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="Reference module dtype")
    p.add_argument("--cos-threshold", type=float, default=0.9999, help="Primary pass/fail threshold for float tensors")
    p.add_argument("--compare-c2w", action="store_true", help="Also replay Code2WavStreamingWrapper")
    p.add_argument(
        "--c2w-codes-source",
        default="dumped",
        choices=["dumped", "reference"],
        help="Whether Code2Wav consumes dumped full_codec or reference talker full_codec",
    )
    p.add_argument("--report-json", default=None, help="Optional path to write a JSON report")
    return p.parse_args()


def _resolve_device(raw: str | None) -> str:
    if raw:
        return raw
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(raw: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[raw]


def _move_tensor(t: torch.Tensor, device: str, float_dtype: torch.dtype) -> torch.Tensor:
    if t.is_floating_point():
        return t.to(device=device, dtype=float_dtype)
    return t.to(device=device)


def _cosine_score(ref: torch.Tensor, got: torch.Tensor) -> float:
    if ref.numel() == 0 and got.numel() == 0:
        return 1.0
    ref_flat = ref.reshape(-1).float()
    got_flat = got.reshape(-1).float()
    ref_norm = float(ref_flat.norm().item())
    got_norm = float(got_flat.norm().item())
    if ref_norm == 0.0 and got_norm == 0.0:
        return 1.0
    if ref_norm == 0.0 or got_norm == 0.0:
        return 0.0
    return float(F.cosine_similarity(ref_flat.unsqueeze(0), got_flat.unsqueeze(0)).item())


def _compare_tensor(
    name: str,
    ref: torch.Tensor,
    got: torch.Tensor,
    *,
    cos_threshold: float,
) -> Dict[str, Any]:
    ref_cpu = ref.detach().cpu()
    got_cpu = got.detach().cpu()
    out: Dict[str, Any] = {
        "name": name,
        "shape_ref": list(ref_cpu.shape),
        "shape_dump": list(got_cpu.shape),
        "dtype_ref": str(ref_cpu.dtype),
        "dtype_dump": str(got_cpu.dtype),
    }
    if ref_cpu.shape != got_cpu.shape:
        out["match"] = False
        out["reason"] = "shape_mismatch"
        return out

    if not ref_cpu.is_floating_point():
        diff = (ref_cpu.to(torch.int64) != got_cpu.to(torch.int64))
        out["match"] = bool(not diff.any())
        out["num_mismatch"] = int(diff.sum().item())
        if ref_cpu.numel() > 0:
            out["dump_sample"] = got_cpu.reshape(-1)[:8].tolist()
            out["ref_sample"] = ref_cpu.reshape(-1)[:8].tolist()
        return out

    ref_f = ref_cpu.float()
    got_f = got_cpu.float()
    delta = (ref_f - got_f).abs()
    out["cosine"] = _cosine_score(ref_f, got_f)
    out["match"] = bool(out["cosine"] >= cos_threshold)
    out["max_abs"] = float(delta.max().item()) if delta.numel() else 0.0
    out["mean_abs"] = float(delta.mean().item()) if delta.numel() else 0.0
    return out


def _load_dump(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected dump payload type: {type(payload)}")
    for key in ("metadata", "inputs", "outputs"):
        if key not in payload:
            raise KeyError(f"Dump missing top-level key: {key}")
    return payload


def _build_talker_reference(
    model,
    inputs_cpu: Dict[str, torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    fused, num_layers, _hidden, _kv_heads, _head_dim = build_talker_unified_fused_module(
        model, device=device
    )
    fused.eval()

    inputs = {
        key: _move_tensor(value, device, dtype)
        for key, value in inputs_cpu.items()
        if isinstance(value, torch.Tensor)
    }
    talker_past = inputs["talker_past_kv"]
    past_kv = [talker_past[:, i, :, :, :] for i in range(num_layers * 2)]
    cp_gumbel = inputs.get("cp_gumbel_noise")
    if cp_gumbel is None:
        cp_gumbel = torch.zeros(
            inputs["input_embeds"].shape[0],
            fused.cp.num_stages,
            inputs["gumbel_noise"].shape[-1],
            device=inputs["gumbel_noise"].device,
            dtype=inputs["gumbel_noise"].dtype,
        )

    with torch.no_grad():
        ref = fused(
            inputs["input_embeds"],
            inputs["position_ids"],
            inputs["token_counts"],
            inputs["gumbel_noise"],
            cp_gumbel,
            inputs["temperature"],
            inputs["penalty"],
            inputs["attention_bias"],
            *past_kv,
        )

    return {
        "codec_sum": ref[0].detach().cpu(),
        "full_codec": ref[1].detach().cpu(),
        "hidden": ref[2].detach().cpu(),
        "logits": ref[3].detach().cpu(),
        "updated_token_counts": ref[4].detach().cpu(),
        "talker_new_kv": torch.stack(ref[5:], dim=1).detach().cpu(),
    }


def _build_c2w_reference(
    tokenizer_path: Path,
    inputs_cpu: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
    full_codec_cpu: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    tokenizer_model = load_speech_tokenizer(tokenizer_path, device=device, dtype=torch.float32)
    decoder = tokenizer_model.decoder.to(device).eval()
    patch_decoder_transconv_for_trt(decoder)
    wrapper = Code2WavStreamingWrapper(decoder, emit_delta_kv=True).to(device).eval()
    n_c2w = num_code2wav_hidden_layers(decoder)

    inputs = {
        key: _move_tensor(value, device, dtype)
        for key, value in inputs_cpu.items()
        if isinstance(value, torch.Tensor)
    }
    c2w_past = inputs["c2w_past_kv"]
    state_inputs = [c2w_past[:, i, :, :, :] for i in range(n_c2w * 2)]
    state_inputs.extend(inputs[name] for name in meta.get("c2w_conv_input_names", []) if name in inputs)
    state_inputs.extend(inputs[name] for name in meta.get("c2w_transconv_input_names", []) if name in inputs)

    codes = full_codec_cpu.long().clamp(0, 2047).to(device=device).unsqueeze(-1)
    with torch.no_grad():
        ref = wrapper(
            codes,
            inputs["cache_position"],
            inputs["c2w_attention_bias"],
            *state_inputs,
        )

    out: Dict[str, torch.Tensor] = {
        "wav": ref[0].detach().cpu(),
        "c2w_new_kv": torch.stack(ref[1 : 1 + 2 * n_c2w], dim=1).detach().cpu(),
    }

    idx = 1 + 2 * n_c2w
    for name, tensor in zip(meta.get("c2w_conv_output_names", []), ref[idx : idx + len(meta.get("c2w_conv_output_names", []))]):
        out[name] = tensor.detach().cpu()
    idx += len(meta.get("c2w_conv_output_names", []))
    for name, tensor in zip(meta.get("c2w_transconv_output_names", []), ref[idx:]):
        out[name] = tensor.detach().cpu()
    return out


def _print_section(title: str, rows: list[Dict[str, Any]]) -> None:
    print(title)
    print("=" * len(title))
    for row in rows:
        name = row["name"]
        if row.get("reason") == "shape_mismatch":
            print(
                f"{name}: SHAPE_MISMATCH ref={row['shape_ref']} dump={row['shape_dump']}"
            )
            continue
        if "max_abs" in row:
            print(
                f"{name}: match={row['match']} cosine={row.get('cosine', 0.0):.9f} "
                f"max_abs={row['max_abs']:.6g} mean_abs={row['mean_abs']:.6g}"
            )
        else:
            print(
                f"{name}: match={row['match']} num_mismatch={row.get('num_mismatch', 0)} "
                f"ref_sample={row.get('ref_sample')} dump_sample={row.get('dump_sample')}"
            )
    print()


def main() -> None:
    args = _parse_args()
    dump_path = Path(args.dump).expanduser().resolve()
    payload = _load_dump(dump_path)
    meta = payload["metadata"]
    inputs_cpu = payload["inputs"]
    outputs_cpu = payload["outputs"]

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    model_path_str = args.model_path or meta.get("weights_dir") or ""
    if not model_path_str:
        raise ValueError("Model path is required; pass --model-path or ensure dump metadata has weights_dir")
    model_path = Path(model_path_str).expanduser().resolve()

    print("Dump")
    print("====")
    print(f"path: {dump_path}")
    print(f"stage: {meta.get('stage')}")
    print(f"sessions: {meta.get('slot_session_ids')}")
    print(f"slot_ids: {meta.get('slot_ids')}")
    print(f"batch_size: {meta.get('batch_size')} seq_len: {meta.get('seq_len')}")
    print(f"use_dummy_kv: {meta.get('use_dummy_kv')}")
    print(f"talker_past_before: {meta.get('original_talker_past_lens')}")
    print(f"c2w_past_before: {meta.get('slot_c2w_len_before')}")
    print()

    model = load_tts_model(model_path, device=device, dtype=dtype)

    talker_ref = _build_talker_reference(
        model,
        inputs_cpu,
        device=device,
        dtype=dtype,
    )
    talker_rows = []
    for name in (
        "codec_sum",
        "full_codec",
        "hidden",
        "logits",
        "updated_token_counts",
        "talker_new_kv",
    ):
        if name not in outputs_cpu:
            continue
        talker_rows.append(
            _compare_tensor(
                name,
                talker_ref[name],
                outputs_cpu[name],
                cos_threshold=args.cos_threshold,
            )
        )
    _print_section("Talker Replay", talker_rows)

    report: Dict[str, Any] = {
        "dump": str(dump_path),
        "metadata": meta,
        "talker": talker_rows,
    }

    if args.compare_c2w:
        tokenizer_path = Path(args.tokenizer_path).expanduser().resolve() if args.tokenizer_path else resolve_tokenizer_path()
        c2w_codec = outputs_cpu["full_codec"]
        if args.c2w_codes_source == "reference":
            c2w_codec = talker_ref["full_codec"]
        c2w_ref = _build_c2w_reference(
            tokenizer_path,
            inputs_cpu,
            meta,
            c2w_codec,
            device=device,
            dtype=dtype,
        )
        c2w_rows = [
            _compare_tensor("wav", c2w_ref["wav"], outputs_cpu["wav"], cos_threshold=args.cos_threshold),
            _compare_tensor("c2w_new_kv", c2w_ref["c2w_new_kv"], outputs_cpu["c2w_new_kv"], cos_threshold=args.cos_threshold),
        ]
        for name in meta.get("c2w_conv_output_names", []):
            if name in outputs_cpu and name in c2w_ref:
                c2w_rows.append(
                    _compare_tensor(
                        name,
                        c2w_ref[name],
                        outputs_cpu[name],
                        cos_threshold=args.cos_threshold,
                    )
                )
        for name in meta.get("c2w_transconv_output_names", []):
            if name in outputs_cpu and name in c2w_ref:
                c2w_rows.append(
                    _compare_tensor(
                        name,
                        c2w_ref[name],
                        outputs_cpu[name],
                        cos_threshold=args.cos_threshold,
                    )
                )
        _print_section("Code2Wav Replay", c2w_rows)
        report["c2w"] = c2w_rows

    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
