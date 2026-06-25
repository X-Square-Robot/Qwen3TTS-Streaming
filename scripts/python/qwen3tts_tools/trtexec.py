"""TRT engine compilation — Python wrapper for trtexec.

Replaces ``scripts/bash/lib/trtexec_runner.sh`` and
``scripts/bash/build_engines.sh`` core logic.

Public interface:

- :func:`resolve_runner` — determine docker or host trtexec mode
- :func:`run_trtexec` — run trtexec to compile ONNX → TensorRT engine
- :func:`suggest_profile` — suggest TRT profile from GPU memory
- :func:`build_all_engines` — build all engines for a variant
- :func:`build_talker_code2wav_fused` — build primary production engine
- :func:`build_talker_unified` — build talker unified engine (verification)
- :func:`build_speech_tokenizer_codec_fused` — build speech tokenizer codec
- :func:`build_speaker_encoder` — build speaker encoder engine
- :func:`build_peripheral_engines` — build tokenizer encoder + code2wav decoder
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
#  TrtProfile
# ---------------------------------------------------------------------------

@dataclass
class TrtProfile:
    """TensorRT optimization profile."""

    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 1
    min_seq: int = 1
    opt_seq: int = 128
    max_seq: int = 512


# ---------------------------------------------------------------------------
#  Precision helpers
# ---------------------------------------------------------------------------

def _normalize_dtype(value: str) -> str:
    raw = (value or "").strip().lower()
    aliases = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float8": "fp8",
    }
    return aliases.get(raw, raw)


def precision_flags(dtype: str) -> list[str]:
    """Return trtexec precision flag(s) for the given dtype.

    Args:
        dtype: Engine dtype (bf16, fp16, fp32, fp8).

    Returns:
        List of flag strings, e.g. ``["--bf16"]``.
    """
    d = _normalize_dtype(dtype)
    if d == "bf16":
        return ["--bf16"]
    if d == "fp16":
        return ["--fp16"]
    if d == "fp8":
        return ["--fp8"]
    return []


def io_format_string(dtype: str) -> str:
    """Return trtexec I/O format string for float tensors.

    Args:
        dtype: Engine dtype (bf16, fp16, fp32, fp8).

    Returns:
        Format string, e.g. ``"bf16:chw"``.
    """
    d = _normalize_dtype(dtype)
    mapping = {"bf16": "bf16:chw", "fp16": "fp16:chw", "fp8": "fp8:chw"}
    return mapping.get(d, "fp32:chw")


# ---------------------------------------------------------------------------
#  Runner detection
# ---------------------------------------------------------------------------

def resolve_runner() -> str:
    """Determine whether to use docker or host trtexec.

    Returns ``"docker"`` or ``"host"``.
    """
    runner = os.environ.get("BUILD_RUNNER", "").lower()
    if runner in ("docker", "host"):
        return runner

    if shutil.which("docker") and _docker_gpu_available():
        return "docker"

    if _host_trtexec_available():
        return "host"

    raise RuntimeError(
        "Cannot find trtexec. Either install TensorRT on the host, "
        "or ensure Docker + NVIDIA Container Toolkit are available."
    )


def _docker_gpu_available() -> bool:
    """Check if Docker + NVIDIA runtime are functional."""
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.4.0-base-ubuntu22.04", "nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _host_trtexec_available() -> bool:
    """Check if trtexec is available on the host."""
    candidates = [
        Path("/usr/src/tensorrt/bin/trtexec"),
        Path("/usr/local/bin/trtexec"),
    ]
    trtexec_env = os.environ.get("TRTEXEC_HOST", "")
    if trtexec_env:
        candidates.insert(0, Path(trtexec_env))

    for p in candidates:
        if p.is_file():
            return True

    return shutil.which("trtexec") is not None


def _find_trtexec_path() -> str:
    """Find host trtexec binary path."""
    trtexec_env = os.environ.get("TRTEXEC_HOST", "")
    if trtexec_env and Path(trtexec_env).is_file():
        return trtexec_env

    for p in ["/usr/src/tensorrt/bin/trtexec", "/usr/local/bin/trtexec"]:
        if Path(p).is_file():
            return p

    if shutil.which("trtexec"):
        return shutil.which("trtexec")

    raise FileNotFoundError("trtexec not found on host")


def _resolve_ngc_image() -> str:
    """Resolve NGC container image for trtexec in Docker mode."""
    from qwen3tts_tools.ngc_matrix import resolve_ngc_image
    from qwen3tts_tools.docker import detect_driver_version

    driver = detect_driver_version()
    if driver:
        image = resolve_ngc_image(driver)
        if image:
            return image

    return os.environ.get("NGC_IMAGE", "nvcr.io/nvidia/tritonserver:25.05-py3")


# ---------------------------------------------------------------------------
#  Docker GPU args detection
# ---------------------------------------------------------------------------

def _detect_docker_gpu_args(image: str, gpu_device: str = "auto") -> list[str]:
    """Detect Docker GPU arguments for trtexec.

    Tries ``--gpus device=N`` first, then falls back to
    ``--runtime=nvidia`` with environment variables.

    Args:
        image: Docker image to test with.
        gpu_device: GPU device (auto|all|N|cuda:N).

    Returns:
        List of Docker arguments for GPU passthrough.
    """
    docker_gpu_arg = "all"
    visible_devices = "all"

    if gpu_device not in ("auto", "all", ""):
        norm = gpu_device.removeprefix("cuda:")
        docker_gpu_arg = f"device={norm}"
        visible_devices = norm

    # Try --gpus first
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", docker_gpu_arg, image, "/bin/true"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logger.info("Docker GPU launch mode: --gpus %s", docker_gpu_arg)
            return ["--gpus", docker_gpu_arg]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: --runtime=nvidia
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--runtime=nvidia",
                "-e", f"NVIDIA_VISIBLE_DEVICES={visible_devices}",
                "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                image, "/bin/true",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logger.info(
                "Docker GPU launch fallback: --runtime=nvidia "
                "(NVIDIA_VISIBLE_DEVICES=%s)", visible_devices,
            )
            return [
                "--runtime=nvidia",
                "-e", f"NVIDIA_VISIBLE_DEVICES={visible_devices}",
                "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise RuntimeError(f"Docker GPU smoke test failed for image: {image}")


# ---------------------------------------------------------------------------
#  run_trtexec — full-featured trtexec invocation
# ---------------------------------------------------------------------------

def run_trtexec(
    onnx: str,
    engine: str,
    *,
    dtype: str = "bf16",
    runner: str = "auto",
    min_shapes: str = "",
    opt_shapes: str = "",
    max_shapes: str = "",
    input_io_formats: str = "",
    output_io_formats: str = "",
    mem_pool_size: str = "",
    extra_args: list[str] | None = None,
    gpu_device: str = "auto",
    ngc_image: str = "",
    profile: TrtProfile | None = None,
) -> int:
    """Run trtexec to compile an ONNX model into a TensorRT engine.

    Args:
        onnx: Path to the ONNX model file.
        engine: Output path for the TensorRT engine.
        dtype: Engine data type (bf16, fp16, fp32, fp8).
        runner: Execution mode (auto, docker, host).
        min_shapes: ``--minShapes`` argument string.
        opt_shapes: ``--optShapes`` argument string.
        max_shapes: ``--maxShapes`` argument string.
        input_io_formats: ``--inputIOFormats`` argument string.
        output_io_formats: ``--outputIOFormats`` argument string.
        mem_pool_size: ``--memPoolSize`` argument (e.g. ``"workspace:8192"``).
        extra_args: Additional trtexec arguments.
        gpu_device: GPU device for build (auto|all|N|cuda:N).
        ngc_image: Override NGC container image for docker mode.
        profile: Simplified TrtProfile for backward compat. Ignored if
                 min/opt/max_shapes are provided.

    Returns:
        Process exit code (0 = success).
    """
    if runner == "auto":
        runner = resolve_runner()

    # Build trtexec argument list
    args: list[str] = [
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
    ]

    # Precision flags
    args.extend(precision_flags(dtype))

    # Shapes
    if min_shapes:
        args.append(f"--minShapes={min_shapes}")
    elif profile:
        args.append(f"--minShapes=input_embeds:{profile.min_batch}x{profile.min_seq}")

    if opt_shapes:
        args.append(f"--optShapes={opt_shapes}")
    elif profile:
        args.append(f"--optShapes=input_embeds:{profile.opt_batch}x{profile.opt_seq}")

    if max_shapes:
        args.append(f"--maxShapes={max_shapes}")
    elif profile:
        args.append(f"--maxShapes=input_embeds:{profile.max_batch}x{profile.max_seq}")

    # I/O formats
    if input_io_formats:
        args.append(f"--inputIOFormats={input_io_formats}")
    if output_io_formats:
        args.append(f"--outputIOFormats={output_io_formats}")

    # Memory pool
    if mem_pool_size:
        args.append(f"--memPoolSize={mem_pool_size}")

    # Extra args
    if extra_args:
        args.extend(extra_args)

    if runner == "docker":
        return _run_docker(onnx, engine, args, gpu_device, ngc_image)
    else:
        return _run_host(args)


def _run_docker(
    onnx: str,
    engine: str,
    trtexec_args: list[str],
    gpu_device: str = "auto",
    ngc_image: str = "",
) -> int:
    """Run trtexec inside an NGC Docker container.

    Mirrors ``scripts/bash/lib/trtexec_runner.sh::_trtexec_run_docker``:
    a single bind mount of the model directory at ``/mnt/model`` with the
    ``--onnx`` / ``--saveEngine`` paths rewritten to that mount.  trtexec is
    invoked by absolute path because it is *not* on ``PATH`` in NGC
    tritonserver images (it lives under ``/usr/src/tensorrt/bin``); override
    with the ``TRTEXEC`` env var if needed.
    """
    image = ngc_image or _resolve_ngc_image()
    repo_root = _repo_root()

    onnx_path = Path(onnx)
    engine_path = Path(engine)
    onnx_dir = onnx_path.parent
    engine_dir = engine_path.parent

    # Detect GPU args
    try:
        gpu_args = _detect_docker_gpu_args(image, gpu_device)
    except RuntimeError:
        gpu_args = ["--gpus", "all"]

    bundle_root = os.environ.get("QWEN3_IN_BUNDLE_ROOT", "")

    # trtexec is not on PATH in NGC tritonserver images.
    trtexec_bin = os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec")

    # Rewrite the onnx/engine paths to the container mount(s).  The ONNX (with
    # its external-data files) and the output engine normally share one
    # directory → a single /mnt/model mount, matching the bash runner.  Fall
    # back to two mounts only if they somehow differ.
    if onnx_dir == engine_dir:
        mount_args = ["-v", f"{onnx_dir}:/mnt/model"]
        onnx_in = f"/mnt/model/{onnx_path.name}"
        engine_in = f"/mnt/model/{engine_path.name}"
    else:
        mount_args = [
            "-v", f"{onnx_dir}:/mnt/onnx",
            "-v", f"{engine_dir}:/mnt/output",
        ]
        onnx_in = f"/mnt/onnx/{onnx_path.name}"
        engine_in = f"/mnt/output/{engine_path.name}"

    rewritten_args = [
        f"--onnx={onnx_in}" if a.startswith("--onnx=")
        else f"--saveEngine={engine_in}" if a.startswith("--saveEngine=")
        else a
        for a in trtexec_args
    ]

    docker_args = ["docker", "run", "--rm"] + gpu_args + mount_args
    if bundle_root:
        docker_args += ["-v", f"{bundle_root}:/bundle_root"]
    else:
        docker_args += ["-v", f"{repo_root}/scripts:/scripts"]
    docker_args += [image, trtexec_bin] + rewritten_args

    logger.info("docker run: %s", " ".join(docker_args[:20]) + ("..." if len(docker_args) > 20 else ""))
    return subprocess.call(docker_args)


def _run_host(trtexec_args: list[str]) -> int:
    """Run trtexec on host."""
    trtexec_path = _find_trtexec_path()

    cmd = [trtexec_path] + trtexec_args
    logger.info("host trtexec: %s", " ".join(cmd))
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
#  suggest_profile
# ---------------------------------------------------------------------------

def suggest_profile(memory_mb: int) -> TrtProfile:
    """Suggest TRT build profile based on available GPU memory.

    Args:
        memory_mb: Available GPU memory in MiB.

    Returns:
        Suggested optimization profile.
    """
    if memory_mb >= 80000:
        return TrtProfile(max_batch=64, opt_seq=256, max_seq=1024)
    elif memory_mb >= 48000:
        return TrtProfile(max_batch=32, opt_seq=128, max_seq=512)
    elif memory_mb >= 24000:
        return TrtProfile(max_batch=16, opt_seq=64, max_seq=256)
    else:
        return TrtProfile(max_batch=4, opt_seq=32, max_seq=128)


def suggest_build_profile(memory_mb: int) -> tuple[int, int, int]:
    """Suggest (max_batch, max_input, max_seq) from GPU memory.

    This matches the tier logic from build_pipeline.sh.

    Args:
        memory_mb: Available GPU memory in MiB.

    Returns:
        Tuple of (max_batch_size, max_input_len, max_seq_len).
    """
    if memory_mb >= 76000:
        return (128, 128, 512)
    elif memory_mb >= 45000:
        return (64, 128, 512)
    elif memory_mb >= 29000:
        return (32, 128, 512)
    else:
        return (16, 96, 384)


# ---------------------------------------------------------------------------
#  Talker dimension helpers
# ---------------------------------------------------------------------------

_VARIANT_DIR_MAP: dict[str, str] = {
    "base-0.6b": "Qwen3-TTS-12Hz-0.6B-Base",
    "custom-0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base-1.7b": "Qwen3-TTS-12Hz-1.7B-Base",
    "custom-1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design-1.7b": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


def get_talker_dims(variant: str) -> tuple[int, int, int, int]:
    """Get talker dimensions (H, kv_heads, head_dim, num_layers).

    Reads from config.json if available, falls back to hardcoded defaults.

    Args:
        variant: Model variant name (e.g. "custom-1.7b").

    Returns:
        Tuple of (hidden_size, num_kv_heads, head_dim, num_layers).
    """
    repo = _repo_root()
    model_dir = _VARIANT_DIR_MAP.get(variant, "")
    if model_dir:
        cfg_path = repo / "workspace" / "models" / model_dir / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                tc = cfg.get("talker_config", cfg)
                h = tc.get("hidden_size", 1024)
                nkv = tc.get("num_key_value_heads", 8)
                natt = tc.get("num_attention_heads", 16)
                head_dim = tc.get("head_dim", h // natt)
                nlayers = tc.get("num_hidden_layers", 28)
                return h, nkv, head_dim, nlayers
            except (json.JSONDecodeError, KeyError):
                pass

    # Fallback defaults
    if variant in ("base-0.6b", "custom-0.6b"):
        return 1024, 8, 128, 28
    return 2048, 8, 128, 28


# ---------------------------------------------------------------------------
#  build_talker_code2wav_fused
# ---------------------------------------------------------------------------

def build_talker_code2wav_fused(
    exported_dir: Path,
    variant: str,
    *,
    dtype: str = "bf16",
    triton_io_float_dtype: str = "",
    backbone_precision: str = "",
    cp_precision: str = "",
    code2wav_precision: str = "",
    runner: str = "auto",
    max_batch_size: int = 64,
    max_input_len: int = 128,
    max_seq_len: int = 512,
    device: str = "auto",
    ngc_image: str = "",
    dry_run: bool = False,
) -> int:
    """Build the primary production TRT engine (talker_code2wav_fused).

    Args:
        exported_dir: Path to workspace/exported/ (or bundle exported/).
        variant: Model variant name.
        dtype: Engine dtype (global default for submodules without explicit precision).
        triton_io_float_dtype: Triton IO float dtype (default: same as dtype).
        backbone_precision: Compute precision for backbone sub-graph (default: dtype).
        cp_precision: Compute precision for CP sub-graph (default: dtype).
            Set to "fp32" to mitigate BF16 numerical sensitivity.
        code2wav_precision: Compute precision for code2wav sub-graph (default: dtype).
        runner: trtexec runner mode.
        max_batch_size: Max batch size for TRT profile.
        max_input_len: Max input length for prefill.
        max_seq_len: Max sequence length including KV cache.
        device: GPU device for build.
        ngc_image: Override NGC container image.
        dry_run: If True, skip actual compilation.

    Returns:
        Exit code (0 = success).
    """
    if not triton_io_float_dtype:
        triton_io_float_dtype = dtype

    # Resolve per-submodule precision defaults
    if not backbone_precision:
        backbone_precision = dtype
    if not cp_precision:
        cp_precision = dtype
    if not code2wav_precision:
        code2wav_precision = dtype

    # Normalize
    backbone_precision = _normalize_dtype(backbone_precision)
    cp_precision = _normalize_dtype(cp_precision)
    code2wav_precision = _normalize_dtype(code2wav_precision)

    # Detect mixed-precision mode
    is_mixed = len({backbone_precision, cp_precision, code2wav_precision}) > 1

    variant_dir = exported_dir / variant
    onnx_path = variant_dir / "talker_code2wav_fused.onnx"

    if not onnx_path.is_file():
        logger.error(
            "Missing %s — run: python scripts/export/export_09_talker_code2wav_fused.py --variant %s",
            onnx_path, variant,
        )
        return 1

    H, kv_heads, head_dim, num_layers = get_talker_dims(variant)

    # Resolve shape profiles
    n_c2w, n_cp = _resolve_c2w_dims(variant_dir)
    fused_min, fused_opt, fused_max = _compute_fused_shapes(
        H, kv_heads, head_dim, num_layers,
        max_batch_size, max_input_len, max_seq_len,
        n_c2w, n_cp,
    )

    logger.info(
        "Building talker_code2wav_fused.engine: %s "
        "(H=%d, kv_heads=%d, head_dim=%d, layers=%d, "
        "prec: backbone=%s, cp=%s, code2wav=%s)",
        variant, H, kv_heads, head_dim, num_layers,
        backbone_precision, cp_precision, code2wav_precision,
    )

    if dry_run:
        logger.info("[DRY RUN] talker_code2wav_fused (mixed=%s)", is_mixed)
        return 0

    engine_path = variant_dir / "talker_code2wav_fused.engine"

    if is_mixed:
        # Mixed-precision path: use trtexec with --layerPrecisions
        result = _build_fused_mixed_precision(
            onnx_path, engine_path,
            backbone_precision=backbone_precision,
            cp_precision=cp_precision,
            code2wav_precision=code2wav_precision,
            triton_io_float_dtype=triton_io_float_dtype,
            fused_min=fused_min,
            fused_opt=fused_opt,
            fused_max=fused_max,
            dtype=dtype,
            device=device,
            variant_dir=variant_dir,
            runner=runner,
            ngc_image=ngc_image,
        )
    else:
        # Uniform-precision path: use trtexec (backward compatible)
        result = _build_fused_trtexec(
            variant_dir, onnx_path, engine_path,
            dtype=dtype,
            triton_io_float_dtype=triton_io_float_dtype,
            runner=runner,
            fused_min=fused_min,
            fused_opt=fused_opt,
            fused_max=fused_max,
            device=device,
            ngc_image=ngc_image,
        )

    if result == 0:
        # Update manifest with precision info
        _update_manifest_profile_mixed(
            variant_dir, variant, dtype, triton_io_float_dtype,
            backbone_precision, cp_precision, code2wav_precision,
            max_batch_size, max_input_len, max_seq_len,
            ngc_image, mark_built=True,
        )
        logger.info("talker_code2wav_fused.engine built: %s", variant_dir)
    else:
        logger.error("talker_code2wav_fused build failed for %s", variant)

    return result


def _build_fused_trtexec(
    variant_dir: Path,
    onnx_path: Path,
    engine_path: Path,
    *,
    dtype: str = "bf16",
    triton_io_float_dtype: str = "",
    runner: str = "auto",
    fused_min: str = "",
    fused_opt: str = "",
    fused_max: str = "",
    device: str = "auto",
    ngc_image: str = "",
) -> int:
    """Build talker_code2wav_fused.engine using trtexec (uniform precision)."""
    # Resolve I/O formats from manifest
    input_io, output_io, prec_override = _resolve_fused_io_formats(
        variant_dir, dtype, triton_io_float_dtype,
    )

    prec = precision_flags(prec_override or dtype)

    return run_trtexec(
        onnx=str(onnx_path),
        engine=str(engine_path),
        dtype=prec_override or dtype,
        runner=runner,
        min_shapes=fused_min,
        opt_shapes=fused_opt,
        max_shapes=fused_max,
        input_io_formats=input_io,
        output_io_formats=output_io,
        mem_pool_size="workspace:8192",
        gpu_device=device,
        ngc_image=ngc_image,
    )


def _build_fused_mixed_precision(
    onnx_path: Path,
    engine_path: Path,
    *,
    backbone_precision: str = "bf16",
    cp_precision: str = "fp32",
    code2wav_precision: str = "bf16",
    triton_io_float_dtype: str = "bf16",
    fused_min: str = "",
    fused_opt: str = "",
    fused_max: str = "",
    dtype: str = "bf16",
    device: str = "auto",
    variant_dir: Path | None = None,
    runner: str = "auto",
    ngc_image: str = "",
) -> int:
    """Build talker_code2wav_fused.engine using trtexec with --layerPrecisions.

    Uses trtexec's ``--layerPrecisions`` and ``--precisionConstraints=obey``
    to enforce per-submodule compute precision.  This avoids depending on the
    local TensorRT Python API (which changed significantly in TRT 10+/11+).

    Strategy:
      1. Set the *global* builder flag to the most common precision among
         backbone / cp / code2wav (the "majority" precision).
      2. Enumerate ONNX node names whose prefix belongs to a submodule whose
         precision differs from the global default.
      3. Pass those layer names via ``--layerPrecisions`` to override.

    Args:
        onnx_path: Path to talker_code2wav_fused.onnx.
        engine_path: Output path for the TensorRT engine.
        backbone_precision: Compute precision for backbone sub-graph.
        cp_precision: Compute precision for code_predictor sub-graph.
        code2wav_precision: Compute precision for code2wav sub-graph.
        triton_io_float_dtype: External float I/O dtype.
        fused_min/opt/max: Shape profile strings.
        dtype: Global engine dtype (used as fallback).
        device: GPU device for build.
        variant_dir: Path to the variant directory (for manifest I/O formats).
        runner: trtexec runner mode.
        ngc_image: Override NGC container image.
    """
    from qwen3tts_tools.mixed_precision_builder import is_mixed_precision

    # Normalize
    backbone_precision = _normalize_dtype(backbone_precision)
    cp_precision = _normalize_dtype(cp_precision)
    code2wav_precision = _normalize_dtype(code2wav_precision)
    triton_io_float_dtype = _normalize_dtype(triton_io_float_dtype)

    if not is_mixed_precision(backbone_precision, cp_precision, code2wav_precision):
        # Not actually mixed — fall through to uniform path
        logger.info("Not mixed precision; delegating to uniform trtexec build")
        return _build_fused_trtexec(
            variant_dir or onnx_path.parent,
            onnx_path, engine_path,
            dtype=dtype,
            triton_io_float_dtype=triton_io_float_dtype,
            runner=runner,
            fused_min=fused_min,
            fused_opt=fused_opt,
            fused_max=fused_max,
            device=device,
            ngc_image=ngc_image,
        )

    # Use the backbone precision as the *global* builder precision.  The
    # backbone is by far the largest sub-graph (tens of thousands of nodes),
    # so making it the global default means we only ever have to *override*
    # the two smaller sub-graphs (cp / code2wav) when they differ.
    global_precision = backbone_precision
    logger.info(
        "Mixed-precision trtexec build: backbone=%s, cp=%s, code2wav=%s, "
        "global=%s (backbone), io=%s",
        backbone_precision, cp_precision, code2wav_precision,
        global_precision, triton_io_float_dtype,
    )

    # Express per-submodule overrides as *wildcard* patterns instead of
    # enumerating every layer name.  trtexec's --layerPrecisions accepts one
    # '*' wildcard per entry and matches against TRT layer names (which keep
    # their ONNX node-name prefix after parsing).  This keeps the argument a
    # few dozen bytes long; enumerating all layers produced a >170 KB string
    # that overflowed the kernel's per-argument limit (MAX_ARG_STRLEN, 128 KB)
    # and made execve fail regardless of whether it was passed inline or via a
    # script file.  Each sub-graph maps to a unique layer-name prefix
    # (see _PREFIX_RULES in mixed_precision_builder).
    _SUBMODULE_WILDCARDS: dict[str, list[str]] = {
        "cp": ["/talker_fused/cp/*"],
        "code2wav": ["/code2wav/*"],
    }

    layer_prec_parts: list[str] = []
    for category, prec in (("cp", cp_precision), ("code2wav", code2wav_precision)):
        if prec == global_precision:
            continue
        patterns = _SUBMODULE_WILDCARDS.get(category, [])
        for pat in patterns:
            layer_prec_parts.append(f"{pat}:{prec}")
            logger.info("  %s override: %s → %s", category, pat, prec)

    if not layer_prec_parts:
        logger.warning(
            "No sub-graph differs from the backbone precision — "
            "building as uniform %s",
            global_precision,
        )
        return _build_fused_trtexec(
            variant_dir or onnx_path.parent,
            onnx_path, engine_path,
            dtype=global_precision,
            triton_io_float_dtype=triton_io_float_dtype,
            runner=runner,
            fused_min=fused_min,
            fused_opt=fused_opt,
            fused_max=fused_max,
            device=device,
            ngc_image=ngc_image,
        )

    layer_precisions_str = ",".join(layer_prec_parts)
    logger.info(
        "Total --layerPrecisions entries: %d (%d chars)",
        len(layer_prec_parts), len(layer_precisions_str),
    )

    # Resolve I/O formats
    input_io, output_io, prec_override = "", "", ""
    if variant_dir:
        input_io, output_io, prec_override = _resolve_fused_io_formats(
            variant_dir, global_precision, triton_io_float_dtype,
        )

    # Build extra_args for trtexec
    extra_args: list[str] = [
        "--precisionConstraints=obey",
        f"--layerPrecisions={layer_precisions_str}",
    ]

    return run_trtexec(
        onnx=str(onnx_path),
        engine=str(engine_path),
        dtype=global_precision,
        runner=runner,
        min_shapes=fused_min,
        opt_shapes=fused_opt,
        max_shapes=fused_max,
        input_io_formats=input_io,
        output_io_formats=output_io,
        mem_pool_size="workspace:8192",
        extra_args=extra_args,
        gpu_device=device,
        ngc_image=ngc_image,
    )


def _update_manifest_profile_mixed(
    variant_dir: Path,
    variant: str,
    engine_dtype: str,
    triton_io_float_dtype: str,
    backbone_precision: str,
    cp_precision: str,
    code2wav_precision: str,
    max_batch_size: int,
    max_input_len: int,
    max_seq_len: int,
    ngc_image: str,
    mark_built: bool = False,
) -> None:
    """Update triton_manifest.json with engine profile + mixed-precision metadata."""
    manifest_path = variant_dir / "triton_manifest.json"
    if not manifest_path.is_file():
        logger.warning("No triton_manifest.json for %s; cannot record profile", variant)
        return

    sys_path = str(_repo_root() / "scripts" / "python")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    try:
        from update_triton_manifest_profile import update_manifest
        import argparse

        args = argparse.Namespace(
            manifest=str(manifest_path),
            engine_mode="trt",
            engine_dtype=_normalize_dtype(engine_dtype),
            triton_io_float_dtype=_normalize_dtype(triton_io_float_dtype or engine_dtype),
            backbone_precision=_normalize_dtype(backbone_precision),
            cp_precision=_normalize_dtype(cp_precision),
            code2wav_precision=_normalize_dtype(code2wav_precision),
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            builder="trtexec",
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


def _resolve_c2w_dims(variant_dir: Path) -> tuple[int, int]:
    """Resolve code2wav hidden layers and CP stages from manifest."""
    n_c2w = 8
    n_cp = 15
    manifest_path = variant_dir / "triton_manifest.json"
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            c2w = data.get("code2wav_fused") or {}
            n_c2w = int(c2w.get("num_code2wav_hidden_layers", 8))
            n_cp = int(data.get("architecture", {}).get("cp_num_stages", 15))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return n_c2w, n_cp


def _compute_fused_shapes(
    H: int,
    kv_heads: int,
    head_dim: int,
    num_layers: int,
    max_batch: int,
    max_input: int,
    max_seq: int,
    n_c2w: int,
    n_cp: int,
) -> tuple[str, str, str]:
    """Compute --minShapes/--optShapes/--maxShapes for fused model.

    Uses the same packed KV format as trt_fused_talk_c2w_profiles.py.
    """
    # Delegate to the canonical profile generator in
    # scripts/python/trt_fused_talk_c2w_profiles.py — the single source of
    # truth also used by the bash build pipeline (build_engines.sh).  Keeping
    # one implementation prevents the optimization profile from drifting
    # between the two code paths (a divergent copy here previously emitted a
    # 3-D attention_bias opt shape, which trtexec rejected).
    sys_path = str(_repo_root() / "scripts" / "python")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    from trt_fused_talk_c2w_profiles import compute_fused_profiles

    return compute_fused_profiles(
        H, kv_heads, head_dim, num_layers, max_batch,
        max_in=max_input, max_seq=max_seq,
        n_c2w=n_c2w or 8, n_cp=n_cp or 15,
    )


def _resolve_fused_io_formats(
    variant_dir: Path,
    dtype: str,
    triton_io_float_dtype: str,
) -> tuple[str, str, str]:
    """Resolve I/O format strings from triton_manifest.json.

    Returns:
        (input_io_formats, output_io_formats, precision_override_dtype).
        Empty strings mean "not specified" (trtexec defaults apply).
    """
    manifest_path = variant_dir / "triton_manifest.json"
    if not manifest_path.is_file():
        logger.warning(
            "No triton_manifest.json — using dtype for precision only; "
            "fused I/O formats omitted",
        )
        return "", "", dtype

    sys_path = str(_repo_root() / "scripts" / "python")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    try:
        from trt_fused_io_formats import (
            fused_input_output_io_format_strings,
            trtexec_precision_args,
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        inp, out = fused_input_output_io_format_strings(data)
        prec_list = trtexec_precision_args(str(data.get("engine_dtype", dtype)))
        prec_str = " ".join(prec_list) if prec_list else ""
        # Return the dtype that corresponds to the precision flags
        prec_dtype = dtype
        if "--bf16" in prec_str:
            prec_dtype = "bf16"
        elif "--fp16" in prec_str:
            prec_dtype = "fp16"
        elif "--fp8" in prec_str:
            prec_dtype = "fp8"
        else:
            prec_dtype = "fp32"
        return inp, out, prec_dtype
    except (ImportError, Exception) as e:
        logger.warning("Could not resolve fused I/O formats from manifest: %s", e)
        return "", "", dtype


# ---------------------------------------------------------------------------
#  build_talker_unified (verification)
# ---------------------------------------------------------------------------

def build_talker_unified(
    exported_dir: Path,
    variant: str,
    *,
    dtype: str = "bf16",
    runner: str = "auto",
    max_batch_size: int = 64,
    max_input_len: int = 128,
    max_seq_len: int = 512,
    device: str = "auto",
    ngc_image: str = "",
    dry_run: bool = False,
) -> int:
    """Build talker_unified TRT engine (verification / separate prefill+decode).

    Args:
        exported_dir: Path to workspace/exported/.
        variant: Model variant name.
        dtype: Engine dtype.
        runner: trtexec runner mode.
        max_batch_size: Max batch size.
        max_input_len: Max input length.
        max_seq_len: Max sequence length.
        device: GPU device.
        ngc_image: Override NGC container image.
        dry_run: If True, skip actual compilation.

    Returns:
        Exit code (0 = success).
    """
    variant_dir = exported_dir / variant
    onnx_path = variant_dir / "talker_unified.onnx"

    if not onnx_path.is_file():
        logger.error("Unified TRT: missing ONNX for %s (need talker_unified.onnx)", variant)
        return 1

    H, kv_heads, head_dim, num_layers = get_talker_dims(variant)

    logger.info(
        "Building Talker unified TRT engine: %s (H=%d, kv_heads=%d, "
        "head_dim=%d, layers=%d)",
        variant, H, kv_heads, head_dim, num_layers,
    )

    if dry_run:
        logger.info("[DRY RUN] Would run trtexec for talker_unified.engine")
        return 0

    # Build shape strings
    OPT_BATCH = 1
    OPT_S_PAST = 128
    io_fmt = io_format_string(dtype)

    unif_min = f"input_embeds:1x1x{H},position_ids:1x3x1x1"
    unif_opt = f"input_embeds:{OPT_BATCH}x1x{H},position_ids:{OPT_BATCH}x3x1x1"
    unif_max = f"input_embeds:{max_batch_size}x{max_input_len}x{H},position_ids:{max_batch_size}x3x{max_input_len}x1"

    for i in range(num_layers):
        unif_min += f",past_kv_{i}_k:1x{kv_heads}x0x{head_dim},past_kv_{i}_v:1x{kv_heads}x0x{head_dim}"
        unif_opt += f",past_kv_{i}_k:{OPT_BATCH}x{kv_heads}x{OPT_S_PAST}x{head_dim},past_kv_{i}_v:{OPT_BATCH}x{kv_heads}x{OPT_S_PAST}x{head_dim}"
        unif_max += f",past_kv_{i}_k:{max_batch_size}x{kv_heads}x{max_seq_len}x{head_dim},past_kv_{i}_v:{max_batch_size}x{kv_heads}x{max_seq_len}x{head_dim}"

    # I/O formats
    io_in = f"{io_fmt},int64:chw"
    io_out = f"{io_fmt},int64:chw,{io_fmt},{io_fmt}"
    for i in range(num_layers):
        io_in += f",{io_fmt},{io_fmt}"
        io_out += f",{io_fmt},{io_fmt}"

    engine_path = variant_dir / "talker_unified.engine"
    return run_trtexec(
        onnx=str(onnx_path),
        engine=str(engine_path),
        dtype=dtype,
        runner=runner,
        min_shapes=unif_min,
        opt_shapes=unif_opt,
        max_shapes=unif_max,
        input_io_formats=io_in,
        output_io_formats=io_out,
        mem_pool_size="workspace:8192",
        gpu_device=device,
        ngc_image=ngc_image,
    )


# ---------------------------------------------------------------------------
#  build_speech_tokenizer_codec_fused
# ---------------------------------------------------------------------------

def build_speech_tokenizer_codec_fused(
    exported_dir: Path,
    variant: str,
    *,
    dtype: str = "bf16",
    runner: str = "auto",
    device: str = "auto",
    ngc_image: str = "",
    dry_run: bool = False,
) -> int:
    """Build speech_tokenizer_codec_fused TRT engine.

    Args:
        exported_dir: Path to workspace/exported/.
        variant: Model variant name.
        dtype: Engine dtype.
        runner: trtexec runner mode.
        device: GPU device.
        ngc_image: Override NGC container image.
        dry_run: If True, skip actual compilation.

    Returns:
        Exit code (0 = success), 0 also if ONNX not found (skip).
    """
    variant_dir = exported_dir / variant
    onnx_path = variant_dir / "speech_tokenizer_codec_fused.onnx"

    if not onnx_path.is_file():
        return 0  # Not an error — just not present for this variant

    logger.info("Building speech_tokenizer_codec_fused.engine: %s", variant)

    if dry_run:
        logger.info("[DRY RUN] trtexec speech_tokenizer_codec_fused")
        return 0

    engine_path = variant_dir / "speech_tokenizer_codec_fused.engine"
    return run_trtexec(
        onnx=str(onnx_path),
        engine=str(engine_path),
        dtype=dtype,
        runner=runner,
        min_shapes="waveform:1x1x960",
        opt_shapes="waveform:1x1x48000",
        max_shapes="waveform:1x1x192000",
        mem_pool_size="workspace:6144",
        gpu_device=device,
        ngc_image=ngc_image,
    )


# ---------------------------------------------------------------------------
#  build_speaker_encoder
# ---------------------------------------------------------------------------

def build_speaker_encoder(
    exported_dir: Path,
    variant: str,
    *,
    dtype: str = "bf16",
    runner: str = "auto",
    max_batch_size: int = 64,
    device: str = "auto",
    ngc_image: str = "",
    dry_run: bool = False,
) -> int:
    """Build speaker_encoder TRT engine for a variant.

    Args:
        exported_dir: Path to workspace/exported/.
        variant: Model variant name (or directory name under exported/).
        dtype: Engine dtype.
        runner: trtexec runner mode.
        max_batch_size: Max batch size for the profile.
        device: GPU device.
        ngc_image: Override NGC container image.
        dry_run: If True, skip actual compilation.

    Returns:
        Exit code (0 = success), 0 also if ONNX not found (skip).
    """
    variant_dir = exported_dir / variant
    onnx_path = variant_dir / "speaker_encoder.onnx"

    if not onnx_path.is_file():
        return 0  # Not present — skip

    logger.info("Building speaker_encoder.engine for %s", variant)

    if dry_run:
        logger.info("[DRY RUN] Would build speaker_encoder engine")
        return 0

    io_fmt = io_format_string(dtype)
    engine_path = variant_dir / "speaker_encoder.engine"

    return run_trtexec(
        onnx=str(onnx_path),
        engine=str(engine_path),
        dtype=dtype,
        runner=runner,
        min_shapes="mel:1x1x128",
        opt_shapes="mel:1x300x128",
        max_shapes=f"mel:{max_batch_size}x1000x128",
        input_io_formats=io_fmt,
        output_io_formats=io_fmt,
        mem_pool_size="workspace:1024",
        gpu_device=device,
        ngc_image=ngc_image,
    )


# ---------------------------------------------------------------------------
#  build_peripheral_engines (verification)
# ---------------------------------------------------------------------------

def build_peripheral_engines(
    exported_dir: Path,
    *,
    dtype: str = "bf16",
    runner: str = "auto",
    max_batch_size: int = 64,
    device: str = "auto",
    ngc_image: str = "",
    dry_run: bool = False,
) -> int:
    """Build peripheral verification engines (tokenizer encoder + code2wav decoder).

    Args:
        exported_dir: Path to workspace/exported/.
        dtype: Engine dtype.
        runner: trtexec runner mode.
        max_batch_size: Max batch size.
        device: GPU device.
        ngc_image: Override NGC container image.
        dry_run: If True, skip actual compilation.

    Returns:
        Number of failed engines (0 = all success).
    """
    failed = 0
    tokenizer_dir = exported_dir / "tokenizer"

    # Speech tokenizer encoder
    ste_onnx = tokenizer_dir / "speech_tokenizer_encoder.onnx"
    if ste_onnx.is_file():
        logger.info("Building speech_tokenizer_encoder.engine ...")
        if dry_run:
            logger.info("[DRY RUN] Would build speech_tokenizer_encoder")
        else:
            result = run_trtexec(
                onnx=str(ste_onnx),
                engine=str(tokenizer_dir / "speech_tokenizer_encoder.engine"),
                dtype=dtype,
                runner=runner,
                min_shapes="waveform:1x1x960",
                opt_shapes="waveform:1x1x48000",
                max_shapes="waveform:1x1x192000",
                mem_pool_size="workspace:6144",
                gpu_device=device,
                ngc_image=ngc_image,
            )
            if result != 0:
                logger.error("speech_tokenizer_encoder trtexec failed")
                failed += 1
    else:
        logger.info("speech_tokenizer_encoder.onnx not found, skipping")

    # Code2Wav decoder
    c2w_onnx = tokenizer_dir / "code2wav_decoder.onnx"
    if c2w_onnx.is_file():
        logger.info("Building code2wav_decoder.engine (streaming, chunk_T=4) ...")
        if dry_run:
            logger.info("[DRY RUN] Would build code2wav_decoder")
        else:
            result = _build_code2wav_decoder(
                tokenizer_dir, dtype, runner, max_batch_size, device, ngc_image,
            )
            if result != 0:
                logger.error("code2wav_decoder trtexec failed")
                failed += 1
    else:
        logger.warning("code2wav_decoder.onnx not found, skipping (export step 06 / verification)")

    return failed


def _build_code2wav_decoder(
    tokenizer_dir: Path,
    dtype: str,
    runner: str,
    max_batch_size: int,
    device: str,
    ngc_image: str,
) -> int:
    """Build code2wav_decoder TRT engine with streaming shapes."""
    C2W_BATCH = max_batch_size or 8
    io_fmt = io_format_string(dtype)

    C2W_MIN = "codes:1x16x4,cache_position:1x4,c2w_attention_bias:1x1x4x5"
    C2W_OPT = "codes:1x16x4,cache_position:1x4,c2w_attention_bias:1x1x4x8"
    C2W_MAX = f"codes:{C2W_BATCH}x16x4,cache_position:{C2W_BATCH}x4,c2w_attention_bias:{C2W_BATCH}x1x4x76"

    for i in range(8):
        C2W_MIN += f",past_kv_{i}_k:1x16x1x64,past_kv_{i}_v:1x16x1x64"
        C2W_OPT += f",past_kv_{i}_k:1x16x4x64,past_kv_{i}_v:1x16x4x64"
        C2W_MAX += f",past_kv_{i}_k:{C2W_BATCH}x16x72x64,past_kv_{i}_v:{C2W_BATCH}x16x72x64"

    # Conv/transconv state shapes
    conv_specs = [
        ("conv_state_0", "1x512x2"), ("conv_state_1", "1x1024x6"),
        ("conv_state_2", "1x1024x6"), ("conv_state_3", "1x1024x6"),
        ("conv_state_4", "1x768x6"), ("conv_state_5", "1x768x18"),
        ("conv_state_6", "1x768x54"), ("conv_state_7", "1x384x6"),
        ("conv_state_8", "1x384x18"), ("conv_state_9", "1x384x54"),
        ("conv_state_10", "1x192x6"), ("conv_state_11", "1x192x18"),
        ("conv_state_12", "1x192x54"), ("conv_state_13", "1x96x6"),
        ("conv_state_14", "1x96x18"), ("conv_state_15", "1x96x54"),
        ("conv_state_16", "1x96x6"),
    ]
    tc_specs = [
        ("transconv_overlap_0", "1x768x8"), ("transconv_overlap_1", "1x384x5"),
        ("transconv_overlap_2", "1x192x4"), ("transconv_overlap_3", "1x96x3"),
    ]

    for name, s in conv_specs + tc_specs:
        rest = s[2:]
        C2W_MIN += f",{name}:{s}"
        C2W_OPT += f",{name}:{s}"
        C2W_MAX += f",{name}:{C2W_BATCH}x{rest}"

    # I/O formats
    c2w_io_in = f"int64:chw,fp32:chw,{io_fmt}"
    for _ in range(37):
        c2w_io_in += f",{io_fmt}"

    c2w_io_out = io_fmt
    for _ in range(37):
        c2w_io_out += f",{io_fmt}"

    return run_trtexec(
        onnx=str(tokenizer_dir / "code2wav_decoder.onnx"),
        engine=str(tokenizer_dir / "code2wav_decoder.engine"),
        dtype=dtype,
        runner=runner,
        min_shapes=C2W_MIN,
        opt_shapes=C2W_OPT,
        max_shapes=C2W_MAX,
        input_io_formats=c2w_io_in,
        output_io_formats=c2w_io_out,
        mem_pool_size="workspace:4096",
        gpu_device=device,
        ngc_image=ngc_image,
    )


# ---------------------------------------------------------------------------
#  build_all_engines — top-level orchestration
# ---------------------------------------------------------------------------

def build_all_engines(
    exported_dir: Path,
    variant: str,
    *,
    dtype: str = "bf16",
    triton_io_float_dtype: str = "",
    backbone_precision: str = "",
    cp_precision: str = "",
    code2wav_precision: str = "",
    runner: str = "auto",
    max_batch_size: int = 64,
    max_input_len: int = 128,
    max_seq_len: int = 512,
    device: str = "auto",
    ngc_image: str = "",
    dry_run: bool = False,
    build_verification: bool = False,
) -> int:
    """Build all production engines for a variant.

    Production engines:
    - talker_code2wav_fused (primary)
    - speaker_encoder
    - speech_tokenizer_codec_fused (base variants only)

    With build_verification=True, also builds:
    - talker_unified
    - speech_tokenizer_encoder
    - code2wav_decoder

    Args:
        exported_dir: Path to workspace/exported/.
        variant: Model variant name.
        dtype: Engine dtype (global default for submodules without explicit precision).
        triton_io_float_dtype: Triton IO float dtype.
        backbone_precision: Compute precision for backbone (default: dtype).
        cp_precision: Compute precision for CP (default: dtype).
        code2wav_precision: Compute precision for code2wav (default: dtype).
        runner: trtexec runner mode.
        max_batch_size: Max batch size.
        max_input_len: Max input length.
        max_seq_len: Max sequence length.
        device: GPU device.
        ngc_image: Override NGC container image.
        dry_run: If True, skip actual compilation.
        build_verification: If True, also build verification engines.

    Returns:
        Number of failed engines (0 = all success).
    """
    failed = 0

    # Speaker encoder (per variant)
    result = build_speaker_encoder(
        exported_dir, variant,
        dtype=dtype, runner=runner, max_batch_size=max_batch_size,
        device=device, ngc_image=ngc_image, dry_run=dry_run,
    )
    if result != 0:
        failed += 1

    # Speech tokenizer codec fused (base variants)
    if variant.startswith("base-"):
        result = build_speech_tokenizer_codec_fused(
            exported_dir, variant,
            dtype=dtype, runner=runner, device=device,
            ngc_image=ngc_image, dry_run=dry_run,
        )
        if result != 0:
            failed += 1

    # Primary production engine
    result = build_talker_code2wav_fused(
        exported_dir, variant,
        dtype=dtype, triton_io_float_dtype=triton_io_float_dtype,
        backbone_precision=backbone_precision,
        cp_precision=cp_precision,
        code2wav_precision=code2wav_precision,
        runner=runner, max_batch_size=max_batch_size,
        max_input_len=max_input_len, max_seq_len=max_seq_len,
        device=device, ngc_image=ngc_image, dry_run=dry_run,
    )
    if result != 0:
        failed += 1

    # Verification engines
    if build_verification:
        result = build_talker_unified(
            exported_dir, variant,
            dtype=dtype, runner=runner,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len, max_seq_len=max_seq_len,
            device=device, ngc_image=ngc_image, dry_run=dry_run,
        )
        if result != 0:
            failed += 1

        peripheral_failed = build_peripheral_engines(
            exported_dir,
            dtype=dtype, runner=runner, max_batch_size=max_batch_size,
            device=device, ngc_image=ngc_image, dry_run=dry_run,
        )
        failed += peripheral_failed

    # Write engine dtype marker
    if not dry_run and failed == 0:
        dtype_marker = exported_dir / ".engine_dtype"
        dtype_marker.write_text(_normalize_dtype(dtype))

    return failed


# ---------------------------------------------------------------------------
#  Manifest profile update
# ---------------------------------------------------------------------------

def _update_manifest_profile(
    variant_dir: Path,
    variant: str,
    engine_dtype: str,
    triton_io_float_dtype: str,
    max_batch_size: int,
    max_input_len: int,
    max_seq_len: int,
    ngc_image: str,
    mark_built: bool = False,
) -> None:
    """Update triton_manifest.json with engine profile metadata."""
    manifest_path = variant_dir / "triton_manifest.json"
    if not manifest_path.is_file():
        logger.warning("No triton_manifest.json for %s; cannot record engine profile", variant)
        return

    sys_path = str(_repo_root() / "scripts" / "python")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    try:
        from update_triton_manifest_profile import update_manifest
        import argparse

        args = argparse.Namespace(
            manifest=str(manifest_path),
            engine_mode="trt",
            engine_dtype=_normalize_dtype(engine_dtype),
            triton_io_float_dtype=_normalize_dtype(triton_io_float_dtype or engine_dtype),
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            builder="trtexec",
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


# ---------------------------------------------------------------------------
#  Legacy build_fused_engine (backward compat)
# ---------------------------------------------------------------------------

def build_fused_engine(
    exported_dir: Path,
    variant: str,
    *,
    dtype: str = "bf16",
    runner: str = "auto",
    max_batch: int = 0,
    max_seq: int = 0,
) -> int:
    """Build the primary production TRT engine (talker_code2wav_fused).

    Legacy interface for backward compatibility.
    Use :func:`build_talker_code2wav_fused` for the full-featured version.
    """
    from qwen3tts_tools.docker import detect_gpu_free_memory_mb

    onnx_path = exported_dir / "talker_code2wav_fused.onnx"
    engine_path = exported_dir / "talker_code2wav_fused.engine"

    if not onnx_path.is_file():
        onnx_path = exported_dir / "fused_model.onnx"

    if not onnx_path.is_file():
        logger.error("Fused ONNX model not found in %s", exported_dir)
        return 1

    if max_batch > 0 or max_seq > 0:
        profile = TrtProfile(
            max_batch=max_batch or 48,
            opt_batch=max(1, (max_batch or 48) // 4),
            max_seq=max_seq or 512,
            opt_seq=max(1, (max_seq or 512) // 4),
        )
    else:
        try:
            free_mb = detect_gpu_free_memory_mb()
            profile = suggest_profile(free_mb)
        except Exception:
            profile = TrtProfile(max_batch=48, opt_seq=128, max_seq=512)

    logger.info(
        "Building fused engine: %s → %s (profile: batch=%d/%d/%d, seq=%d/%d/%d)",
        onnx_path.name, engine_path.name,
        profile.min_batch, profile.opt_batch, profile.max_batch,
        profile.min_seq, profile.opt_seq, profile.max_seq,
    )

    return run_trtexec(
        onnx=str(onnx_path),
        engine=str(engine_path),
        dtype=dtype,
        runner=runner,
        profile=profile,
    )
