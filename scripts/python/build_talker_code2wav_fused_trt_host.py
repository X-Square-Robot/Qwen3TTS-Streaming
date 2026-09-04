#!/usr/bin/env python3
"""Build talker_code2wav_fused.engine on the host with TensorRT Python API.

This is a Docker-free fallback for environments where `build_engines.sh`
cannot use `docker run --gpus all`, but the local Python environment already
has a matching TensorRT runtime/builder installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import tensorrt as trt

from trt_fused_io_formats import fused_input_output_io_format_strings
from trt_fused_talk_c2w_profiles import (
    C2W_HEAD_DIM,
    C2W_KV_HEADS,
    C2W_SLIDING_WINDOW,
    LOGITS_TOPK,
    VOCAB_SIZE,
    c2w_conv_transconv_specs,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant-dir",
        type=Path,
        default=Path("workspace/exported/custom-1.7b"),
        help="Exported variant directory containing talker_code2wav_fused.onnx and triton_manifest.json",
    )
    p.add_argument(
        "--engine-out",
        type=Path,
        default=None,
        help="Output engine path (default: <variant-dir>/talker_code2wav_fused.fixed.engine)",
    )
    p.add_argument("--max-batch-size", type=int, default=8)
    p.add_argument("--max-input-len", type=int, default=4096)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--workspace-mib", type=int, default=8192)
    p.add_argument(
        "--builder-optimization-level",
        type=int,
        default=None,
        help="Optional TensorRT builder_optimization_level override (0-5).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load_manifest(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _split_io_formats(spec: str) -> list[str]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise ValueError("Empty I/O format specification")
    return parts


def _dtype_from_format_token(token: str) -> trt.DataType:
    base = token.split(":", 1)[0].lower()
    if base == "bf16":
        return trt.DataType.BF16
    if base == "fp16":
        return trt.DataType.HALF
    if base == "fp32":
        return trt.DataType.FLOAT
    if base == "int64":
        return trt.DataType.INT64
    raise ValueError(f"Unsupported TensorRT I/O dtype token: {token}")


def _iter_profile_shapes(
    *,
    hidden_size: int,
    kv_heads: int,
    head_dim: int,
    num_layers: int,
    max_batch_size: int,
    max_input_len: int,
    max_seq_len: int,
    n_c2w_layers: int,
    cp_num_stages: int,
    cursor_enabled: bool = False,
    cursor_max_labels: int = 512,
    cursor_d: int = 256,
    cursor_history: int = 30,
) -> Iterable[Tuple[str, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
    b_opt = 1
    opt_s_past = 128
    talker_kv_dim1 = num_layers * 2

    fixed_specs = [
        (
            "input_embeds",
            (1, 1, hidden_size),
            (b_opt, 1, hidden_size),
            (max_batch_size, max_input_len, hidden_size),
        ),
        (
            "position_ids",
            (1, 3, 1, 1),
            (b_opt, 3, 1, 1),
            (max_batch_size, 3, max_input_len, 1),
        ),
        (
            "attention_bias",
            (1, 1, 1, 1),
            (b_opt, 1, 1, opt_s_past + 1),
            (max_batch_size, 1, max_input_len, max_seq_len + max_input_len),
        ),
        (
            "token_counts",
            (1, VOCAB_SIZE),
            (b_opt, VOCAB_SIZE),
            (max_batch_size, VOCAB_SIZE),
        ),
        (
            "gumbel_noise",
            (1, LOGITS_TOPK),
            (b_opt, LOGITS_TOPK),
            (max_batch_size, LOGITS_TOPK),
        ),
        (
            "cp_gumbel_noise",
            (1, cp_num_stages, LOGITS_TOPK),
            (b_opt, cp_num_stages, LOGITS_TOPK),
            (max_batch_size, cp_num_stages, LOGITS_TOPK),
        ),
        ("temperature", (1, 1), (b_opt, 1), (max_batch_size, 1)),
        ("penalty", (1, 1), (b_opt, 1), (max_batch_size, 1)),
        ("cache_position", (1, 1), (b_opt, 1), (max_batch_size, 1)),
        (
            "c2w_attention_bias",
            (1, 1, 1, 2),
            (b_opt, 1, 1, 5),
            (max_batch_size, 1, 1, C2W_SLIDING_WINDOW),
        ),
        (
            "talker_past_kv",
            (1, talker_kv_dim1, kv_heads, 0, head_dim),
            (b_opt, talker_kv_dim1, kv_heads, opt_s_past, head_dim),
            (max_batch_size, talker_kv_dim1, kv_heads, max_seq_len, head_dim),
        ),
        (
            "c2w_past_kv",
            (1, n_c2w_layers * 2, C2W_KV_HEADS, 1, C2W_HEAD_DIM),
            (b_opt, n_c2w_layers * 2, C2W_KV_HEADS, 4, C2W_HEAD_DIM),
            (
                max_batch_size,
                n_c2w_layers * 2,
                C2W_KV_HEADS,
                C2W_SLIDING_WINDOW - 1,
                C2W_HEAD_DIM,
            ),
        ),
    ]
    for item in fixed_specs:
        yield item

    for name, smin, sopt, smax in c2w_conv_transconv_specs(str(max_batch_size)):

        def _parse(spec: str) -> Tuple[int, ...]:
            return tuple(int(x) for x in spec.split("x"))

        yield (f"c2w_{name}", _parse(smin), _parse(sopt), _parse(smax))

    if cursor_enabled:
        cursor_specs = [
            ("cursor_label_ids", (1, cursor_max_labels), (b_opt, cursor_max_labels), (max_batch_size, cursor_max_labels)),
            ("cursor_label_count", (1,), (b_opt,), (max_batch_size,)),
            ("cursor_active", (1,), (b_opt,), (max_batch_size,)),
            ("cursor_mu_in", (1,), (b_opt,), (max_batch_size,)),
            ("cursor_frames_since_advance_in", (1,), (b_opt,), (max_batch_size,)),
            ("cursor_delta_history_in", (1, 8), (b_opt, 8), (max_batch_size, 8)),
            ("cursor_conv_history_in", (1, cursor_history, cursor_d), (b_opt, cursor_history, cursor_d), (max_batch_size, cursor_history, cursor_d)),
            ("cursor_last_trunk_input_in", (1, cursor_d), (b_opt, cursor_d), (max_batch_size, cursor_d)),
            ("cursor_seen_frames_in", (1,), (b_opt,), (max_batch_size,)),
            ("cursor_text_start_frame", (1,), (b_opt,), (max_batch_size,)),
            ("cursor_override_valid", (1,), (b_opt,), (max_batch_size,)),
            ("cursor_override_mu", (1,), (b_opt,), (max_batch_size,)),
        ]
        yield from cursor_specs


def _set_io_dtypes(
    network: trt.INetworkDefinition, input_formats: list[str], output_formats: list[str]
) -> None:
    if network.num_inputs != len(input_formats):
        raise ValueError(
            f"Input format count mismatch: network={network.num_inputs}, formats={len(input_formats)}"
        )
    if network.num_outputs != len(output_formats):
        raise ValueError(
            f"Output format count mismatch: network={network.num_outputs}, formats={len(output_formats)}"
        )

    linear_mask = 1 << int(trt.TensorFormat.LINEAR)
    for idx, fmt in enumerate(input_formats):
        tensor = network.get_input(idx)
        tensor.dtype = _dtype_from_format_token(fmt)
        tensor.allowed_formats = linear_mask

    for idx, fmt in enumerate(output_formats):
        tensor = network.get_output(idx)
        tensor.dtype = _dtype_from_format_token(fmt)
        tensor.allowed_formats = linear_mask


def _apply_precision_flags(config: trt.IBuilderConfig, engine_dtype: str) -> None:
    dtype = str(engine_dtype or "bf16").lower()
    if dtype == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif dtype == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif dtype == "fp8":
        config.set_flag(trt.BuilderFlag.FP8)


def main() -> None:
    args = _parse_args()
    variant_dir = args.variant_dir.resolve()
    onnx_path = variant_dir / "talker_code2wav_fused.onnx"
    manifest_path = variant_dir / "triton_manifest.json"
    engine_path = (
        args.engine_out or (variant_dir / "talker_code2wav_fused.fixed.engine")
    ).resolve()

    manifest = _load_manifest(manifest_path)
    arch = manifest["architecture"]

    input_formats, output_formats = fused_input_output_io_format_strings(manifest)
    input_formats = _split_io_formats(input_formats)
    output_formats = _split_io_formats(output_formats)

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX:\n{errors}")

    _set_io_dtypes(network, input_formats, output_formats)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    if args.builder_optimization_level is not None:
        config.builder_optimization_level = int(args.builder_optimization_level)
    _apply_precision_flags(config, str(manifest.get("engine_dtype", "bf16")))

    profile = builder.create_optimization_profile()
    for name, min_shape, opt_shape, max_shape in _iter_profile_shapes(
        hidden_size=int(arch["hidden_size"]),
        kv_heads=int(arch["kv_heads"]),
        head_dim=int(arch["head_dim"]),
        num_layers=int(arch["num_layers"]),
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_seq_len=args.max_seq_len,
        n_c2w_layers=int(manifest["code2wav_fused"]["num_code2wav_hidden_layers"]),
        cp_num_stages=int(arch.get("cp_num_stages", 15)),
        cursor_enabled=bool((manifest.get("native_cursor") or {}).get("enabled", False)),
        cursor_max_labels=int((manifest.get("native_cursor") or {}).get("max_labels", 512)),
        cursor_d=int((manifest.get("native_cursor") or {}).get("embedding_dim", 256)),
        cursor_history=int((manifest.get("native_cursor") or {}).get("history_width", 30)),
    ):
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build returned None")

    engine_path.write_bytes(bytes(serialized))
    size_gib = engine_path.stat().st_size / (1024**3)
    print(f"Wrote engine: {engine_path}")
    print(f"Size: {size_gib:.2f} GiB")


if __name__ == "__main__":
    main()
