#!/usr/bin/env python3
"""Suggest TensorRT/runtime profile limits from exported model artifacts.

The old shell default was a coarse GPU-memory tier.  This helper keeps that as
fallback but prefers an export-aware estimate:

  fixed memory      = fused TRT/ONNX artifact + runtime embedding weights
  per-slot memory   = persistent KV/state pools kept by the engine
  batch peak memory = temporary batched KV/state tensors used during decode

The estimate intentionally rounds the final capacity down to the supported
profile tiers.  It is a sizing default, not a hard safety proof; explicit
--max-batch-size always wins.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


PROFILE_TIERS = (16, 32, 64, 128)
FUSED_CHUNK_T = 4
MIB = 1024 * 1024


def _dtype_bytes(dtype: str) -> int:
    normalized = (dtype or "bf16").strip().lower()
    if normalized in {"fp32", "float32"}:
        return 4
    if normalized in {"fp8", "float8", "int8"}:
        return 1
    return 2


def _prod(values: list[int] | tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _dir_file_size(root: Path, patterns: tuple[str, ...]) -> int:
    if not root.is_dir():
        return 0
    total = 0
    for pattern in patterns:
        for path in root.glob(pattern):
            total += _file_size(path)
    return total


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _discover_variants(exported_dir: Path) -> list[str]:
    variants: list[str] = []
    if not exported_dir.is_dir():
        return variants
    for child in sorted(exported_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "triton_manifest.json").is_file():
            variants.append(child.name)
    return variants


def _coarse_profile(memory_mib: int) -> tuple[int, int, int]:
    if memory_mib >= 76000:
        return (128, 128, 512)
    if memory_mib >= 45000:
        return (64, 128, 512)
    if memory_mib >= 29000:
        return (32, 128, 512)
    return (16, 96, 384)


def _artifact_fixed_bytes(variant_dir: Path) -> int:
    """Estimate fixed runtime model/weights memory from exported artifacts."""
    fused_engine = variant_dir / "talker_code2wav_fused.engine"
    if fused_engine.is_file():
        fused_bytes = _file_size(fused_engine)
    else:
        fused_bytes = (
            _file_size(variant_dir / "talker_code2wav_fused.onnx")
            + _file_size(variant_dir / "talker_code2wav_fused.onnx.data")
        )

    # EmbeddingWeights loads .pt tensors on GPU.  Keep .npz out because they are
    # export/intermediate compatibility files and are not loaded by runtime.
    weights_bytes = _dir_file_size(variant_dir / "weights", ("*.pt",))

    # Base/ICL variants may load these small peripheral TRT engines for
    # reference preprocessing.  If engines do not exist yet, use ONNX size as a
    # conservative proxy.
    peripheral_bytes = 0
    for stem in ("speaker_encoder", "speech_tokenizer_codec_fused"):
        engine = variant_dir / f"{stem}.engine"
        if engine.is_file():
            peripheral_bytes += _file_size(engine)
        else:
            peripheral_bytes += _file_size(variant_dir / f"{stem}.onnx")

    return fused_bytes + weights_bytes + peripheral_bytes


def _manifest_arch(manifest: dict[str, Any]) -> dict[str, Any]:
    arch = dict(manifest.get("architecture") or {})
    talker = manifest.get("talker") or {}
    c2w = manifest.get("code2wav_fused") or {}

    def pick(key: str, *fallbacks: Any, default: int) -> int:
        for value in (arch.get(key), *fallbacks):
            if value is not None:
                return int(value)
        return default

    return {
        "num_layers": pick("num_layers", talker.get("num_layers"), default=28),
        "kv_heads": pick("kv_heads", talker.get("num_kv_heads"), default=8),
        "head_dim": pick("head_dim", talker.get("head_dim"), default=128),
        "hidden_size": pick("hidden_size", talker.get("hidden_size"), default=2048),
        "codec_vocab_size": pick(
            "codec_vocab_size", talker.get("vocab_size"), default=3072
        ),
        "logits_topk": pick("logits_topk", c2w.get("logits_topk"), default=50),
        "cp_num_stages": pick(
            "cp_num_stages", c2w.get("cp_num_stages"), default=15
        ),
        "n_c2w_layers": pick(
            "n_c2w_layers", c2w.get("num_code2wav_hidden_layers"), default=8
        ),
        "c2w_kv_heads": pick(
            "c2w_kv_heads", c2w.get("c2w_kv_heads"), default=16
        ),
        "c2w_head_dim": pick(
            "c2w_head_dim", c2w.get("c2w_head_dim"), default=64
        ),
        "c2w_sliding_window": pick(
            "c2w_sliding_window", c2w.get("c2w_sliding_window"), default=72
        ),
        "initial_state_shapes": c2w.get("initial_state_shapes") or [],
    }


def _variant_estimate(
    variant_dir: Path,
    *,
    engine_dtype: str,
    max_seq_len: int,
) -> dict[str, float | str]:
    manifest = _load_json(variant_dir / "triton_manifest.json")
    arch = _manifest_arch(manifest)
    dtype_bytes = _dtype_bytes(engine_dtype or manifest.get("engine_dtype", "bf16"))

    num_layers = int(arch["num_layers"])
    kv_heads = int(arch["kv_heads"])
    head_dim = int(arch["head_dim"])
    hidden_size = int(arch["hidden_size"])
    vocab = int(arch["codec_vocab_size"])
    n_c2w = int(arch["n_c2w_layers"])
    c2w_heads = int(arch["c2w_kv_heads"])
    c2w_head = int(arch["c2w_head_dim"])
    c2w_window = int(arch["c2w_sliding_window"])
    c2w_past = max(1, c2w_window - 1)
    c2w_decode_past = max(1, c2w_window - FUSED_CHUNK_T)

    state_shapes = [
        tuple(int(dim) for dim in shape)
        for shape in arch.get("initial_state_shapes", [])
        if isinstance(shape, list) and len(shape) >= 2
    ]
    state_elems_per_lane = sum(_prod(shape[1:]) for shape in state_shapes)

    talker_kv = num_layers * 2 * kv_heads * max_seq_len * head_dim * dtype_bytes
    c2w_pool_kv = n_c2w * 2 * c2w_heads * c2w_window * c2w_head * dtype_bytes
    c2w_slot_kv = n_c2w * 2 * c2w_heads * c2w_past * c2w_head * dtype_bytes
    c2w_batch_kv = n_c2w * 2 * c2w_heads * c2w_decode_past * c2w_head * dtype_bytes
    c2w_states = state_elems_per_lane * dtype_bytes
    token_counts = vocab * 8
    tiny_slot = hidden_size * dtype_bytes * 4

    # Persistent: preallocated pools + per-slot c2w state/read-write buffers.
    persistent = (
        talker_kv
        + c2w_pool_kv
        + c2w_slot_kv
        + 2 * c2w_states
        + token_counts
        + tiny_slot
    )

    # Decode peak: the engine gathers talker KV into a contiguous batch input,
    # builds batched C2W KV/states, and keeps cached output buffers.
    talker_delta = num_layers * 2 * kv_heads * 1 * head_dim * dtype_bytes
    c2w_delta = n_c2w * 2 * c2w_heads * FUSED_CHUNK_T * c2w_head * dtype_bytes
    logits_scratch = (
        50 * 4
        + 15 * 50 * 4
        + hidden_size * dtype_bytes
        + vocab * 8
    )
    transient = (
        talker_kv
        + c2w_batch_kv
        + c2w_states
        + c2w_states
        + talker_delta
        + c2w_delta
        + logits_scratch
    )

    # TensorRT execution context / activation allocator overhead is not visible
    # in ONNX metadata.  Use a small per-lane cushion calibrated from runtime
    # profiling; callers can override for a site-specific deployment.
    trt_lane_overhead_mib = float(
        os.environ.get("QWEN3_PROFILE_TRT_LANE_OVERHEAD_MIB", "40")
    )
    per_lane_peak = persistent + transient + trt_lane_overhead_mib * MIB

    return {
        "variant": variant_dir.name,
        "fixed_mib": _artifact_fixed_bytes(variant_dir) / MIB,
        "persistent_mib": persistent / MIB,
        "transient_mib": transient / MIB,
        "trt_lane_overhead_mib": trt_lane_overhead_mib,
        "per_lane_peak_mib": per_lane_peak / MIB,
        "talker_kv_mib": talker_kv / MIB,
        "c2w_kv_mib": (c2w_pool_kv + c2w_slot_kv) / MIB,
        "c2w_state_mib": (2 * c2w_states) / MIB,
    }


def suggest_profile(
    *,
    memory_mib: int,
    exported_dir: Path,
    variants: list[str],
    engine_dtype: str,
    max_input_len: int,
    max_seq_len: int,
) -> dict[str, Any]:
    coarse_batch, coarse_input, coarse_seq = _coarse_profile(memory_mib)
    max_input_len = int(max_input_len or coarse_input)
    max_seq_len = int(max_seq_len or coarse_seq)

    selected_variants = variants or _discover_variants(exported_dir)
    selected_variants = [v for v in selected_variants if v and not v.startswith("all")]
    estimates = []
    for variant in selected_variants:
        variant_dir = exported_dir / variant
        manifest = variant_dir / "triton_manifest.json"
        if manifest.is_file():
            estimates.append(
                _variant_estimate(
                    variant_dir,
                    engine_dtype=engine_dtype,
                    max_seq_len=max_seq_len,
                )
            )

    if not estimates:
        return {
            "source": "coarse-fallback",
            "max_batch_size": coarse_batch,
            "max_input_len": max_input_len,
            "max_seq_len": max_seq_len,
            "memory_total_mib": memory_mib,
            "raw_capacity": coarse_batch,
            "fixed_mib": 0.0,
            "runtime_reserve_mib": 0.0,
            "per_lane_peak_mib": 0.0,
            "variants": [],
        }

    fixed_mib = max(float(e["fixed_mib"]) for e in estimates)
    per_lane_peak_mib = max(float(e["per_lane_peak_mib"]) for e in estimates)
    runtime_reserve_mib = float(
        os.environ.get(
            "QWEN3_PROFILE_RUNTIME_RESERVE_MIB",
            str(max(8192.0, memory_mib * 0.15)),
        )
    )
    usable_fraction = float(os.environ.get("QWEN3_PROFILE_USABLE_FRACTION", "0.90"))
    batch_cap = int(os.environ.get("QWEN3_PROFILE_MAX_BATCH_CAP", "128"))

    available_mib = memory_mib * usable_fraction - fixed_mib - runtime_reserve_mib
    raw_capacity = int(math.floor(available_mib / per_lane_peak_mib)) if per_lane_peak_mib > 0 else 0
    supported = [tier for tier in PROFILE_TIERS if tier <= batch_cap]
    batch = supported[0]
    for tier in supported:
        if raw_capacity >= tier:
            batch = tier

    return {
        "source": "export-manifest",
        "max_batch_size": int(batch),
        "max_input_len": max_input_len,
        "max_seq_len": max_seq_len,
        "memory_total_mib": memory_mib,
        "usable_fraction": usable_fraction,
        "available_mib": available_mib,
        "raw_capacity": raw_capacity,
        "fixed_mib": fixed_mib,
        "runtime_reserve_mib": runtime_reserve_mib,
        "per_lane_peak_mib": per_lane_peak_mib,
        "variants": estimates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-mib", type=int, required=True)
    parser.add_argument("--exported-dir", type=Path, required=True)
    parser.add_argument("--variants", default="", help="Comma-separated variants")
    parser.add_argument("--engine-dtype", default="bf16")
    parser.add_argument("--max-input-len", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--format", choices=("plain", "json", "summary"), default="plain")
    args = parser.parse_args()

    variants = [v for v in args.variants.split(",") if v]
    result = suggest_profile(
        memory_mib=args.memory_mib,
        exported_dir=args.exported_dir,
        variants=variants,
        engine_dtype=args.engine_dtype,
        max_input_len=args.max_input_len,
        max_seq_len=args.max_seq_len,
    )

    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.format == "summary":
        print(
            "source={source} memory={memory_total_mib:.0f}MiB "
            "fixed={fixed_mib:.0f}MiB reserve={runtime_reserve_mib:.0f}MiB "
            "per_lane_peak={per_lane_peak_mib:.1f}MiB raw_capacity={raw_capacity} "
            "profile={max_batch_size}/{max_input_len}/{max_seq_len}".format(**result)
        )
    else:
        print(
            f"{result['max_batch_size']} "
            f"{result['max_input_len']} "
            f"{result['max_seq_len']}"
        )


if __name__ == "__main__":
    main()
