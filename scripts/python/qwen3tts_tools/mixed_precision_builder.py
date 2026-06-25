"""TensorRT Python builder for mixed-precision engine compilation.

Builds ``talker_code2wav_fused.engine`` with per-submodule compute precision,
using the TensorRT Python API instead of trtexec.  This enables prefix-based
layer classification and fine-grained precision control.

Public interface:

- :func:`build_fused_mixed_precision` — build with per-submodule precision
- :func:`classify_layer_name` — classify a layer by prefix
- :func:`is_mixed_precision` — check if a precision config is heterogeneous
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
#  Precision helpers
# ---------------------------------------------------------------------------

_DTYPE_ALIASES: dict[str, str] = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float8": "fp8",
}

_VALID_DTYPES = {"fp32", "bf16", "fp16", "fp8"}


def _normalize_dtype(value: str) -> str:
    raw = (value or "").strip().lower()
    result = _DTYPE_ALIASES.get(raw, raw)
    if result not in _VALID_DTYPES:
        raise ValueError(f"Invalid dtype '{value}'; expected one of {_VALID_DTYPES}")
    return result


def _trt_dtype(dtype_str: str):
    """Convert dtype string to TensorRT dtype enum value."""
    import tensorrt as trt

    mapping = {
        "fp32": trt.float32,
        "fp16": trt.float16,
        "bf16": trt.bfloat16,
        "fp8": trt.fp8,
    }
    return mapping[dtype_str]


def _trt_builder_flag(dtype_str: str):
    """Convert dtype string to TensorRT BuilderFlag."""
    import tensorrt as trt

    mapping = {
        "bf16": trt.BuilderFlag.BF16,
        "fp16": trt.BuilderFlag.FP16,
        "fp8": trt.BuilderFlag.FP8,
    }
    return mapping.get(dtype_str)


def is_mixed_precision(
    backbone_precision: str,
    cp_precision: str,
    code2wav_precision: str,
) -> bool:
    """Check if the precision configuration is heterogeneous (mixed).

    Returns:
        True if the three submodules don't all use the same precision.
    """
    return len({backbone_precision, cp_precision, code2wav_precision}) > 1


# ---------------------------------------------------------------------------
#  Layer classification
# ---------------------------------------------------------------------------

# Canonical prefix rules for talker_code2wav_fused.onnx
#
# Classification order: first match wins.  More-specific prefixes first,
# broader fallback prefixes later.
#
# Fallback logic (per plan decision #4 and #2):
# - /talker_fused/ nodes NOT under cp/ are backbone.
# - Top-level nodes (e.g. /Constant, /Gather) are backbone — glue/bridge ops.
# - Nodes without / prefix (e.g. Constant_13004) are backbone — ONNX auto-generated.
_PREFIX_RULES: list[tuple[str, str]] = [
    ("/talker_fused/cp/", "cp"),
    ("/talker_fused/talker_unified/", "backbone"),
    ("/talker_fused/codec_sum/", "backbone"),
    ("/talker_fused/", "backbone"),
    ("/code2wav/", "code2wav"),
    ("/", "backbone"),
]


def classify_layer_name(name: str) -> str:
    """Classify a TRT/ONNX layer name into a submodel category.

    Args:
        name: Layer name (e.g. ``/talker_fused/cp/Linear_0``).

    Returns:
        Category string: ``"backbone"``, ``"cp"``, or ``"code2wav"``.
        Nodes that don't match any known prefix are classified as
        ``"backbone"`` (ONNX auto-generated glue ops).
    """
    for prefix, category in _PREFIX_RULES:
        if name.startswith(prefix):
            return category
    # Fallback: ONNX auto-generated names without / prefix are backbone
    return "backbone"


def _classify_layers(network) -> dict[str, list[int]]:
    """Classify all layers in a TRT network by submodel category.

    Returns:
        {category: [layer_indices]} mapping.
    """
    categories: dict[str, list[int]] = {
        "backbone": [],
        "cp": [],
        "code2wav": [],
    }

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        category = classify_layer_name(layer.name)
        categories.setdefault(category, []).append(i)

    return categories


def _log_classification_summary(categories: dict[str, list[int]], network) -> None:
    """Log a summary of layer classification."""
    total = sum(len(v) for v in categories.values())
    logger.info("Layer classification summary (%d total layers):", total)
    for cat, indices in sorted(categories.items()):
        count = len(indices)
        logger.info("  %s: %d layers", cat, count)


# ---------------------------------------------------------------------------
#  ONNX prefix audit (using layer_audit module)
# ---------------------------------------------------------------------------

def _run_prefix_audit(onnx_path: Path) -> None:
    """Run ONNX prefix audit before building.

    Raises LayerAuditError if too many nodes are unclassified.
    """
    sys_path = str(_repo_root() / "scripts" / "python")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    try:
        from qwen3tts_tools.layer_audit import audit_onnx_prefixes, LayerAuditError
    except ImportError:
        logger.warning(
            "layer_audit module not available; skipping ONNX prefix audit"
        )
        return

    try:
        report = audit_onnx_prefixes(onnx_path)
        if report.is_healthy:
            logger.info("ONNX prefix audit passed: %d nodes, %.1f%% unclassified",
                        report.total_nodes, report.unclassified_ratio * 100)
        else:
            raise LayerAuditError(
                f"ONNX prefix audit failed: {report.unclassified_ratio:.1%} "
                f"unclassified nodes ({len(report.categories.get('unclassified', []))} "
                f"/ {report.total_nodes})"
            )
    except FileNotFoundError:
        logger.warning("ONNX file not found for prefix audit: %s", onnx_path)
    except ImportError:
        logger.warning("onnx package not available; skipping prefix audit")


# ---------------------------------------------------------------------------
#  Build function
# ---------------------------------------------------------------------------

def build_fused_mixed_precision(
    onnx_path: Path,
    engine_path: Path,
    *,
    backbone_precision: str = "bf16",
    cp_precision: str = "fp32",
    code2wav_precision: str = "bf16",
    triton_io_float_dtype: str = "bf16",
    min_shapes: str = "",
    opt_shapes: str = "",
    max_shapes: str = "",
    gpu_device: str = "auto",
    workspace_size: int = 8192,
    dry_run: bool = False,
    skip_audit: bool = False,
) -> int:
    """Build talker_code2wav_fused.engine with per-submodule precision.

    Uses the TensorRT Python API to:
    1. Parse the ONNX graph
    2. Classify layers by prefix into backbone / cp / code2wav
    3. Apply per-category precision constraints
    4. Build and serialize the engine

    Args:
        onnx_path: Path to talker_code2wav_fused.onnx.
        engine_path: Output path for the TensorRT engine.
        backbone_precision: Compute precision for backbone (talker_unified + codec_sum).
        cp_precision: Compute precision for code_predictor.
        code2wav_precision: Compute precision for code2wav decoder.
        triton_io_float_dtype: External float I/O dtype (unified).
        min_shapes: --minShapes equivalent string.
        opt_shapes: --optShapes equivalent string.
        max_shapes: --maxShapes equivalent string.
        gpu_device: GPU device selection (auto|all|N|cuda:N).
        workspace_size: Workspace size in MiB.
        dry_run: If True, run classification and audit but skip engine build.
        skip_audit: If True, skip ONNX prefix audit.

    Returns:
        Exit code (0 = success).
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error(
            "tensorrt Python package is required for mixed-precision build. "
            "Run inside NGC container or install TensorRT Python bindings."
        )
        return 1

    # Normalize precision strings
    backbone_precision = _normalize_dtype(backbone_precision)
    cp_precision = _normalize_dtype(cp_precision)
    code2wav_precision = _normalize_dtype(code2wav_precision)
    triton_io_float_dtype = _normalize_dtype(triton_io_float_dtype)

    logger.info(
        "Mixed-precision build: backbone=%s, cp=%s, code2wav=%s, io=%s",
        backbone_precision, cp_precision, code2wav_precision, triton_io_float_dtype,
    )

    if not onnx_path.is_file():
        logger.error("ONNX file not found: %s", onnx_path)
        return 1

    # Step 0: Run ONNX prefix audit
    if not skip_audit:
        _run_prefix_audit(onnx_path)

    # Determine the global (default) builder precision: the most common precision
    # among the three submodules, falling back to backbone.
    precision_counts: dict[str, int] = {}
    for p in (backbone_precision, cp_precision, code2wav_precision):
        precision_counts[p] = precision_counts.get(p, 0) + 1
    global_precision = max(precision_counts, key=precision_counts.get)
    logger.info("Global builder precision: %s (most common among submodules)", global_precision)

    # Step 1: Create builder and parse ONNX
    logger.info("Parsing ONNX: %s", onnx_path)
    trt_logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(trt_logger)
    # EXPLICIT_BATCH was removed in TRT 10+ (always enabled); use 0 as default
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    else:
        network_flags = 0
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error [%d]: %s", i, parser.get_error(i))
        return 1

    logger.info("ONNX parsed successfully: %d layers, %d inputs, %d outputs",
                network.num_layers, network.num_inputs, network.num_outputs)

    # Step 2: Classify layers
    categories = _classify_layers(network)
    _log_classification_summary(categories, network)

    if dry_run:
        logger.info("[DRY RUN] Classification complete; skipping engine build")
        return 0

    # Step 3: Set global builder flags
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 20))

    # Set global precision flag
    global_flag = _trt_builder_flag(global_precision)
    if global_flag is not None:
        config.set_flag(global_flag)
        logger.info("Set global builder flag for %s", global_precision)

    # Step 4: Apply per-layer precision constraints
    precision_map = {
        "backbone": backbone_precision,
        "cp": cp_precision,
        "code2wav": code2wav_precision,
    }

    # Layer types that support float precision overrides (whitelist).
    # Only these layer types carry meaningful compute and can have their
    # precision changed.  All other layer types (Constant, Gather, Shape,
    # Reshape, etc.) are auxiliary / index-computation layers and must
    # remain in their native precision.
    _COMPUTE_LAYERS: set = set()
    for _attr in (
        "MATRIX_MULTIPLY", "CONVOLUTION", "ELEMENTWISE", "SCALE",
        "ACTIVATION", "PADDING", "POOLING", "SOFTMAX",
        "NORMALIZATION", "REDUCE", "UNARY", "LRN", "DECONVOLUTION",
    ):
        _val = getattr(trt.LayerType, _attr, None)
        if _val is not None:
            _COMPUTE_LAYERS.add(_val)

    constrained_count = 0
    skipped_count = 0
    for category, indices in categories.items():
        target_precision = precision_map.get(category, global_precision)
        if target_precision == global_precision:
            # No override needed — these layers use the global default
            continue

        trt_prec = _trt_dtype(target_precision)
        for idx in indices:
            layer = network.get_layer(idx)

            # Only set precision on compute layers (whitelist).
            # Auxiliary layers (Constant, Gather, Shape, Reshape, etc.)
            # must remain in their native precision.
            if hasattr(layer, 'type') and layer.type not in _COMPUTE_LAYERS:
                skipped_count += 1
                continue

            try:
                layer.precision = trt_prec
                constrained_count += 1
            except Exception as e:
                # Layer does not support precision setting — skip it
                skipped_count += 1
                logger.debug(
                    "Skipping precision override for layer %s (type=%s): %s",
                    layer.name, getattr(layer, 'type', '?'), e,
                )

    logger.info(
        "Applied precision overrides to %d layers, skipped %d non-compute "
        "(cp=%s, backbone=%s, code2wav=%s)",
        constrained_count, skipped_count,
        cp_precision, backbone_precision, code2wav_precision,
    )

    # Step 5: Set optimization profiles (shapes)
    if min_shapes or opt_shapes or max_shapes:
        profile = builder.create_optimization_profile()
        _apply_shapes_to_profile(profile, network, min_shapes, opt_shapes, max_shapes)
        config.add_optimization_profile(profile)

    # Step 6: Build engine
    logger.info("Building TensorRT engine (this may take a while)...")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        logger.error("TRT engine build failed: %s", e)
        return 1

    if serialized_engine is None:
        logger.error("TRT engine build returned None — check TRT logs above")
        return 1

    # Step 7: Save engine
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    # IHostMemory: use buffer() for bytes conversion
    engine_bytes = bytes(serialized_engine)
    engine_path.write_bytes(engine_bytes)
    engine_size_mb = len(engine_bytes) / (1 << 20)
    logger.info(
        "Engine saved: %s (%.1f MiB)",
        engine_path, engine_size_mb,
    )

    # Step 8: Post-build verification (optional)
    try:
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is not None:
            num_io = engine.num_io_tensors if hasattr(engine, 'num_io_tensors') else engine.num_bindings
            logger.info(
                "Post-build verification: engine has %d I/O tensors",
                num_io,
            )
            del engine
        else:
            logger.warning("Post-build verification: engine deserialization returned None")
    except Exception as e:
        logger.warning("Post-build verification failed: %s", e)

    return 0


# ---------------------------------------------------------------------------
#  Shape parsing helpers
# ---------------------------------------------------------------------------

def _parse_shape_string(shape_str: str) -> list[tuple[str, tuple[int, ...]]]:
    """Parse a trtexec-style shape string into a list of (name, dims).

    Input format: ``"tensor1:1x3x224x224,tensor2:1x10"``
    Output: ``[("tensor1", (1, 3, 224, 224)), ("tensor2", (1, 10))]``
    """
    if not shape_str:
        return []

    result = []
    for entry in shape_str.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        name, dims_str = entry.split(":", 1)
        dims = tuple(int(d) for d in dims_str.split("x"))
        result.append((name, dims))
    return result


def _apply_shapes_to_profile(
    profile,
    network,
    min_shapes: str,
    opt_shapes: str,
    max_shapes: str,
) -> None:
    """Apply shape specifications to a TRT optimization profile.

    If a tensor's shape specification is missing or has incompatible
    dimensions, falls back to the network's static shape.
    """
    import tensorrt as trt

    min_specs = dict(_parse_shape_string(min_shapes))
    opt_specs = dict(_parse_shape_string(opt_shapes))
    max_specs = dict(_parse_shape_string(max_shapes))

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        name = input_tensor.name

        if name in min_specs and name in opt_specs and name in max_specs:
            min_dims = min_specs[name]
            opt_dims = opt_specs[name]
            max_dims = max_specs[name]

            # Validate: all three must have the same rank and satisfy MIN<=OPT<=MAX
            if (len(min_dims) == len(opt_dims) == len(max_dims)
                    and all(mn <= op <= mx for mn, op, mx in zip(min_dims, opt_dims, max_dims))):
                profile.set_shape(
                    name,
                    trt.Dims(min_dims),
                    trt.Dims(opt_dims),
                    trt.Dims(max_dims),
                )
            else:
                # Shape mismatch (e.g. rank difference) — use max as static
                logger.warning(
                    "Shape spec mismatch for %s: min=%s opt=%s max=%s; "
                    "using max as static shape",
                    name, min_dims, opt_dims, max_dims,
                )
                profile.set_shape(
                    name,
                    trt.Dims(max_dims),
                    trt.Dims(max_dims),
                    trt.Dims(max_dims),
                )
        else:
            # Use static shape from network
            shape = input_tensor.shape
            dims = trt.Dims(shape)
            profile.set_shape(name, dims, dims, dims)


# ---------------------------------------------------------------------------
#  Manifest update helper
# ---------------------------------------------------------------------------

def update_manifest_mixed_precision(
    variant_dir: Path,
    variant: str,
    backbone_precision: str,
    cp_precision: str,
    code2wav_precision: str,
    triton_io_float_dtype: str,
    engine_dtype: str,
    max_batch_size: int,
    max_input_len: int,
    max_seq_len: int,
    ngc_image: str = "",
    mark_built: bool = False,
) -> None:
    """Update triton_manifest.json with mixed-precision build metadata."""
    manifest_path = variant_dir / "triton_manifest.json"
    if not manifest_path.is_file():
        logger.warning("No triton_manifest.json for %s; cannot record profile", variant)
        return

    sys_path = str(_repo_root() / "scripts" / "python")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    try:
        from update_triton_manifest_profile import update_manifest
        import argparse

        args = argparse.Namespace(
            manifest=str(manifest_path),
            engine_mode="trt",
            engine_dtype=_normalize_dtype(engine_dtype),
            triton_io_float_dtype=_normalize_dtype(triton_io_float_dtype),
            backbone_precision=_normalize_dtype(backbone_precision),
            cp_precision=_normalize_dtype(cp_precision),
            code2wav_precision=_normalize_dtype(code2wav_precision),
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            builder="python_trt",
            builder_image=ngc_image or "",
            target_driver="",
            ngc_tag=os.environ.get("NGC_TAG", ""),
            target_profile="",
            gpu_sm="",
            tensorrt_version="",
            skip_built_at=not mark_built,
        )
        update_manifest(args)
    except (ImportError, Exception) as e:
        logger.warning("Could not update manifest profile: %s", e)
