#!/usr/bin/env python3
# English comments only.
"""
Build trtexec --inputIOFormats / --outputIOFormats for talker_code2wav_fused.onnx.

Packed KV format: talker_past_kv and c2w_past_kv are each a single 5-D tensor.
Conv/transconv states remain individual I/O.

I/O order must match export_09_talker_code2wav_fused.py (input_names / output_names).
Integer tensors use int64:chw; float tensors use {fp32|fp16|bf16}:chw from manifest.
`cache_position` is forced to fp32 to avoid a TensorRT/Myelin cast fusion bug.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _normalize_float_io_token(s: str) -> str:
    x = (s or "fp32").lower().strip()
    if x in ("float32", "float"):
        return "fp32"
    if x in ("bfloat16",):
        return "bf16"
    if x in ("float16",):
        return "fp16"
    if x in ("fp32", "bf16", "fp16"):
        return x
    return "fp32"


def _normalize_engine_dtype(s: str) -> str:
    x = (s or "bf16").lower().strip()
    if x in ("bfloat16",):
        return "bf16"
    if x in ("float16",):
        return "fp16"
    if x in ("float32", "float"):
        return "fp32"
    if x in ("fp8", "float8"):
        return "fp8"
    if x in ("bf16", "fp16", "fp32", "fp8"):
        return x
    return "bf16"


def trtexec_precision_args(engine_dtype: str) -> List[str]:
    """Flags as argv tokens (e.g. ['--bf16']); empty for fp32."""
    ed = _normalize_engine_dtype(engine_dtype)
    if ed == "bf16":
        return ["--bf16"]
    if ed == "fp16":
        return ["--fp16"]
    if ed == "fp8":
        return ["--fp8"]
    return []


# Per-submodule layer-name prefixes in talker_code2wav_fused.onnx.  Each maps to
# a single trtexec --layerPrecisions wildcard (one '*' allowed per entry).
_SUBMODULE_WILDCARDS = {
    "cp": "/talker_fused/cp/*",
    "code2wav": "/code2wav/*",
}


def _submodule_precisions(manifest: Dict[str, Any]) -> Dict[str, str]:
    """Resolve normalized per-submodule precisions from the manifest.

    Falls back to the global engine_dtype when a submodule precision is absent
    (i.e. a uniform-precision engine).
    """
    global_dtype = _normalize_engine_dtype(str(manifest.get("engine_dtype", "bf16")))
    out = {}
    for key in ("backbone", "cp", "code2wav"):
        raw = manifest.get(f"{key}_precision")
        out[key] = _normalize_engine_dtype(str(raw)) if raw else global_dtype
    return out


def fused_precision_args(manifest: Dict[str, Any]) -> List[str]:
    """trtexec global precision flags = union of all submodule precisions.

    Enabling every precision present lets ``--precisionConstraints=obey`` honor
    the per-layer overrides (fp32 is always available, so it needs no flag).
    """
    precs = set(_submodule_precisions(manifest).values())
    flags: List[str] = []
    if "bf16" in precs:
        flags.append("--bf16")
    if "fp16" in precs:
        flags.append("--fp16")
    if "fp8" in precs:
        flags.append("--fp8")
    return flags


def fused_layer_precisions(manifest: Dict[str, Any]) -> str:
    """Return the trtexec --layerPrecisions value for a mixed-precision build.

    The backbone (the bulk of the graph) is treated as the global precision and
    floats to it via the enabled flags; only the well-prefixed cp / code2wav
    sub-graphs are pinned via wildcards when they differ.  Returns "" for a
    uniform engine (no override needed).

    Note: this relies on the backbone being the high-speed default precision
    (e.g. bf16); pinning the backbone itself is not expressible as a single
    wildcard because it is the catch-all prefix.
    """
    precs = _submodule_precisions(manifest)
    backbone = precs["backbone"]
    parts: List[str] = []
    for key in ("cp", "code2wav"):
        if precs[key] != backbone:
            parts.append(f"{_SUBMODULE_WILDCARDS[key]}:{precs[key]}")
    if parts and precs["cp"] == backbone:
        # A mixed build enables extra global precision flags (e.g. --fp16 for
        # code2wav), which would otherwise let unpinned talker/cp layers float
        # to the faster precision and silently change sampling numerics.  When
        # cp shares the backbone precision, one catch-all wildcard pins them
        # both without overlapping the cp pin (not expressible otherwise).
        parts.append(f"/talker_fused/*:{backbone}")
    return ",".join(parts)


def fused_input_output_io_format_strings(manifest: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (input_io_formats, output_io_formats) comma-separated for trtexec.

    Packed KV layout:
        Inputs:  input_embeds, position_ids(i64), attention_bias,
                 token_counts(i64), gumbel_noise(fp32), cp_gumbel_noise(fp32),
                 temperature(fp32), penalty(fp32),
                 cache_position(fp32), c2w_attention_bias,
                 talker_past_kv, c2w_past_kv,
                 c2w_conv_state_* (17), c2w_transconv_overlap_* (4)

        Outputs: wav, codec_sum, full_codec(i64), hidden, logits,
                 updated_token_counts(i64),
                 talker_new_kv, c2w_new_kv,
                 c2w_new_conv_state_* (17), c2w_new_transconv_overlap_* (4)
    """
    c2w = manifest.get("code2wav_fused") or {}
    c2w_in = list(c2w.get("c2w_state_input_names") or [])
    c2w_out = list(c2w.get("c2w_state_output_names") or [])

    raw_io = (
        manifest.get("triton_io_float_dtype") or manifest.get("onnx_io_dtype") or "fp32"
    )
    ft = _normalize_float_io_token(str(raw_io))
    fp_spec = f"{ft}:chw"
    i64 = "int64:chw"

    in_parts: List[str] = [
        fp_spec,  # input_embeds
        i64,  # position_ids
        fp_spec,  # attention_bias
        i64,  # token_counts
        "fp32:chw",  # gumbel_noise
        "fp32:chw",  # cp_gumbel_noise
        "fp32:chw",  # temperature
        "fp32:chw",  # penalty
        "fp32:chw",  # cache_position
        fp_spec,  # c2w_attention_bias
        fp_spec,  # talker_past_kv (packed)
        fp_spec,  # c2w_past_kv (packed)
    ]
    in_parts.extend([fp_spec] * len(c2w_in))

    out_parts: List[str] = [
        fp_spec,  # wav
        fp_spec,  # codec_sum
        i64,  # full_codec
        fp_spec,  # hidden
        fp_spec,  # logits
        i64,  # updated_token_counts
        fp_spec,  # talker_new_kv (packed delta)
        fp_spec,  # c2w_new_kv (packed delta)
    ]
    out_parts.extend([fp_spec] * len(c2w_out))

    return ",".join(in_parts), ",".join(out_parts)


def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "manifest",
        type=Path,
        help="Path to triton_manifest.json (variant export dir)",
    )
    p.add_argument(
        "--emit",
        choices=("input", "output", "prec", "layer-precisions", "all"),
        default="all",
        help="Print one value, or all lines (input, output, prec, layer-precisions)",
    )
    args = p.parse_args()
    m = load_manifest(args.manifest)

    inp, out = fused_input_output_io_format_strings(m)
    # Global precision flags = union of submodule precisions (mixed-aware);
    # falls back to a single flag for a uniform engine.
    prec_str = " ".join(fused_precision_args(m))
    layer_prec = fused_layer_precisions(m)

    if args.emit == "input":
        print(inp, end="")
    elif args.emit == "output":
        print(out, end="")
    elif args.emit == "prec":
        print(prec_str, end="")
    elif args.emit == "layer-precisions":
        print(layer_prec, end="")
    else:
        print(inp)
        print(out)
        print(prec_str)
        print(layer_prec)


if __name__ == "__main__":
    main()
