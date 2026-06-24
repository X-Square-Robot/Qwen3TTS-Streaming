"""Cross-host TensorRT engine build bundle lifecycle.

Ports logic from ``scripts/bash/lib/cross_host.sh`` and
``scripts/bash/lib/build_pipeline.sh`` to pure Python.

Public interface:

- :func:`make_build_bundle` — create engine build bundle for target host
- :func:`extract_artifact_bundle` — import compiled engines from artifact bundle
- :func:`fingerprint_check` — validate engine artifact manifest
- :func:`prepare_local_workspace` — prepare bundle workspace for local builds
- :func:`compile_engines_in_bundle` — compile engines inside a bundle workspace
- :func:`write_artifact_manifest` — write artifact_manifest.json with SHA256
- :func:`collect_engines_to_workspace` — copy engines from bundle to workspace
- :func:`pack_artifact_bundle` — pack engines into artifact tarball
- :func:`target_profile_memory_mb` — read GPU memory from target profile
- :func:`generate_build_on_target_script` — generate bash script for target host
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class BuildManifest:
    """Build manifest embedded in engine build bundles."""

    bundle_schema_version: int = 1
    created_at_utc: str = ""
    variants: list[str] = field(default_factory=list)
    engine_dtype: str = "bf16"
    triton_io_float_dtype: str = "bf16"
    backbone_precision: str = ""
    cp_precision: str = ""
    code2wav_precision: str = ""
    max_batch_size: int = 64
    max_input_len: int = 128
    max_seq_len: int = 512
    ngc_tag: str = ""
    ngc_image: str = ""
    build_gpu_device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_schema_version": self.bundle_schema_version,
            "created_at_utc": self.created_at_utc,
            "variants": self.variants,
            "engine_dtype": self.engine_dtype,
            "triton_io_float_dtype": self.triton_io_float_dtype,
            "backbone_precision": self.backbone_precision,
            "cp_precision": self.cp_precision,
            "code2wav_precision": self.code2wav_precision,
            "max_batch_size": self.max_batch_size,
            "max_input_len": self.max_input_len,
            "max_seq_len": self.max_seq_len,
            "ngc_tag": self.ngc_tag,
            "ngc_image": self.ngc_image,
            "build_gpu_device": self.build_gpu_device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildManifest:
        return cls(
            bundle_schema_version=data.get("bundle_schema_version", 1),
            created_at_utc=data.get("created_at_utc", ""),
            variants=data.get("variants", []),
            engine_dtype=data.get("engine_dtype", "bf16"),
            triton_io_float_dtype=data.get("triton_io_float_dtype", "bf16"),
            backbone_precision=data.get("backbone_precision", ""),
            cp_precision=data.get("cp_precision", ""),
            code2wav_precision=data.get("code2wav_precision", ""),
            max_batch_size=data.get("max_batch_size", 64),
            max_input_len=data.get("max_input_len", 128),
            max_seq_len=data.get("max_seq_len", 512),
            ngc_tag=data.get("ngc_tag", ""),
            ngc_image=data.get("ngc_image", ""),
            build_gpu_device=data.get("build_gpu_device", "auto"),
        )

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> BuildManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class ArtifactManifest:
    """Artifact manifest produced after engine compilation on target host."""

    artifact_schema_version: int = 1
    built_at_utc: str = ""
    build_host: str = ""
    ngc_tag: str = ""
    ngc_image: str = ""
    tensorrt_version: str = ""
    cuda_version: str = ""
    gpu_sm: str = ""
    gpu_name: str = ""
    driver_version: str = ""
    engine_dtype: str = ""
    triton_io_float_dtype: str = ""
    max_batch_size: int = 0
    max_input_len: int = 0
    max_seq_len: int = 0
    engines: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "built_at_utc": self.built_at_utc,
            "build_host": self.build_host,
            "ngc_tag": self.ngc_tag,
            "ngc_image": self.ngc_image,
            "tensorrt_version": self.tensorrt_version,
            "cuda_version": self.cuda_version,
            "gpu_sm": self.gpu_sm,
            "gpu_name": self.gpu_name,
            "driver_version": self.driver_version,
            "engine_dtype": self.engine_dtype,
            "triton_io_float_dtype": self.triton_io_float_dtype,
            "max_batch_size": self.max_batch_size,
            "max_input_len": self.max_input_len,
            "max_seq_len": self.max_seq_len,
            "engines": self.engines,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        return cls(
            artifact_schema_version=data.get("artifact_schema_version", 1),
            built_at_utc=data.get("built_at_utc", ""),
            build_host=data.get("build_host", ""),
            ngc_tag=data.get("ngc_tag", ""),
            ngc_image=data.get("ngc_image", ""),
            tensorrt_version=data.get("tensorrt_version", ""),
            cuda_version=data.get("cuda_version", ""),
            gpu_sm=data.get("gpu_sm", ""),
            gpu_name=data.get("gpu_name", ""),
            driver_version=data.get("driver_version", ""),
            engine_dtype=data.get("engine_dtype", ""),
            triton_io_float_dtype=data.get("triton_io_float_dtype", ""),
            max_batch_size=data.get("max_batch_size", 0),
            max_input_len=data.get("max_input_len", 0),
            max_seq_len=data.get("max_seq_len", 0),
            engines=data.get("engines", {}),
        )

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ArtifactManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _tar_cmd() -> list[str]:
    """Return tar command that supports zstd if available, else gzip."""
    try:
        result = subprocess.run(
            ["tar", "--help"], capture_output=True, text=True, timeout=5,
        )
        if "--zstd" in result.stdout:
            return ["tar", "--zstd"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ["tar", "-z"]


def _tar_ext_note() -> str:
    """Return 'zstd' or 'gzip' depending on tar support."""
    try:
        result = subprocess.run(
            ["tar", "--help"], capture_output=True, text=True, timeout=5,
        )
        if "--zstd" in result.stdout:
            return "zstd"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "gzip"


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink if possible, else copy."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _link_or_copy_glob(src_dir: Path, dst_dir: Path, pattern: str) -> None:
    """Hardlink or copy files matching pattern from src_dir to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(src_dir.glob(pattern)):
        _link_or_copy(src_file, dst_dir / src_file.name)


def _resolve_ngc_tag_from_profile(target_profile: Path) -> str:
    """Read recommended_ngc_tag from target_profile.json."""
    data = json.loads(target_profile.read_text(encoding="utf-8"))
    tag = data.get("recommended_ngc_tag", "")
    if not tag:
        raise ValueError(
            f"target_profile.json has no recommended_ngc_tag: {target_profile}"
        )
    return tag


def _resolve_ngc_image_from_tag(ngc_tag: str) -> str:
    """Resolve full NGC image URI from tag."""
    from qwen3tts_tools.ngc_matrix import resolve_ngc_image
    image = resolve_ngc_image(ngc_tag)
    if not image:
        raise ValueError(f"Cannot resolve NGC image for tag: {ngc_tag}")
    return image


# ---------------------------------------------------------------------------
#  target_profile_memory_mb
# ---------------------------------------------------------------------------

def target_profile_memory_mb(
    target_profile: Path,
    gpu_index: int = 0,
) -> int:
    """Read GPU memory (MiB) from target_profile.json.

    Args:
        target_profile: Path to target_profile.json.
        gpu_index: GPU index to read (default 0).

    Returns:
        GPU memory in MiB, or 0 if not found.
    """
    try:
        data = json.loads(target_profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0

    gpus = data.get("gpus") or []
    if not gpus:
        return 0

    selected = next(
        (g for g in gpus if int(g.get("index", -1)) == gpu_index),
        gpus[0],
    )
    return int(selected.get("memory_total_mib") or 0)


# ---------------------------------------------------------------------------
#  fingerprint_check
# ---------------------------------------------------------------------------

def fingerprint_check(
    artifact_manifest_path: Path,
    exported_dir_or_manifest: Path,
    target_profile: Path | None = None,
) -> list[str]:
    """Validate engine artifact manifest against triton manifests.

    Args:
        artifact_manifest_path: Path to artifact_manifest.json.
        exported_dir_or_manifest: Directory with triton_manifest.json files,
            or a single manifest file.
        target_profile: Optional target_profile.json for cross-validation.

    Returns:
        List of error strings. Empty list means OK.

    Raises:
        FileNotFoundError: If artifact_manifest_path does not exist.
    """
    if not artifact_manifest_path.is_file():
        raise FileNotFoundError(
            f"Engine artifact manifest missing: {artifact_manifest_path}\n"
            "  Run 'qwen3tts build' or 'qwen3tts build import-artifact <bundle>' first."
        )

    artifact = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))

    def _major_minor(v: str) -> str:
        parts = str(v or "").split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else str(v or "")

    # Find triton manifests to cross-check
    manifests: list[Path] = []
    if exported_dir_or_manifest.is_file():
        manifests = [exported_dir_or_manifest]
    elif exported_dir_or_manifest.is_dir():
        manifests = sorted(exported_dir_or_manifest.glob("*/triton_manifest.json"))

    errors: list[str] = []

    if not manifests:
        logger.debug(
            "No triton_manifest.json found under %s (artifact-only check)",
            exported_dir_or_manifest,
        )

    for manifest_path in manifests:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        profile = data.get("engine_profile") or {}
        expected = {
            "ngc_tag": profile.get("ngc_tag") or "",
            "engine_dtype": profile.get("engine_dtype") or data.get("engine_dtype") or "",
            "max_batch_size": profile.get("max_batch_size"),
            "max_input_len": profile.get("max_input_len"),
            "max_seq_len": profile.get("max_seq_len"),
        }

        for key in ("engine_dtype", "max_batch_size", "max_input_len", "max_seq_len"):
            exp_val = str(expected.get(key) or "")
            if exp_val and exp_val != str(artifact.get(key)):
                errors.append(
                    f"{manifest_path}: {key} expected={expected.get(key)} "
                    f"artifact={artifact.get(key)}"
                )

        exp_ngc = expected["ngc_tag"]
        if exp_ngc and exp_ngc != str(artifact.get("ngc_tag") or ""):
            errors.append(
                f"{manifest_path}: ngc_tag expected={exp_ngc} "
                f"artifact={artifact.get('ngc_tag')}"
            )

        exp_trt = profile.get("tensorrt_version") or ""
        if exp_trt and _major_minor(exp_trt) != _major_minor(artifact.get("tensorrt_version") or ""):
            errors.append(
                f"{manifest_path}: tensorrt_version expected={exp_trt} "
                f"artifact={artifact.get('tensorrt_version')}"
            )

        exp_sm = profile.get("gpu_sm") or ""
        if exp_sm and exp_sm != str(artifact.get("gpu_sm") or ""):
            errors.append(
                f"{manifest_path}: gpu_sm expected={exp_sm} "
                f"artifact={artifact.get('gpu_sm')}"
            )

    # Cross-validate against target_profile.json
    if target_profile and target_profile.is_file():
        tp_data = json.loads(target_profile.read_text(encoding="utf-8"))
        gpus = tp_data.get("gpus") or []
        tp_sm = (gpus[0] if gpus else {}).get("sm") or ""
        if tp_sm and tp_sm != str(artifact.get("gpu_sm") or ""):
            errors.append(
                f"target_profile: gpu_sm expected={tp_sm} "
                f"artifact={artifact.get('gpu_sm')}"
            )
        host_env = tp_data.get("host_environment") or {}
        tp_trt = host_env.get("trtexec_version") or ""
        if tp_trt and _major_minor(tp_trt) != _major_minor(artifact.get("tensorrt_version") or ""):
            errors.append(
                f"target_profile: trtexec_version expected={tp_trt} "
                f"artifact={artifact.get('tensorrt_version')}"
            )

    return errors


# ---------------------------------------------------------------------------
#  make_build_bundle
# ---------------------------------------------------------------------------

def make_build_bundle(
    repo_root: Path,
    exported_dir: Path,
    out: Path,
    target_profile: Path,
    variants: list[str],
    engine_dtype: str = "bf16",
    triton_io_float_dtype: str = "",
    backbone_precision: str = "",
    cp_precision: str = "",
    code2wav_precision: str = "",
    max_batch_size: int = 64,
    max_input_len: int = 128,
    max_seq_len: int = 512,
    build_gpu_device: str = "auto",
) -> None:
    """Create engine build bundle for cross-host compilation.

    Args:
        repo_root: Repository root path.
        exported_dir: Path to workspace/exported/ with ONNX inputs.
        out: Output path for the bundle tarball.
        target_profile: Path to target_profile.json.
        variants: List of variant names to include.
        engine_dtype: Engine dtype (bf16|fp16|fp32|fp8).
        triton_io_float_dtype: Triton IO float dtype (default: same as engine_dtype).
        backbone_precision: Compute precision for backbone (default: same as engine_dtype).
        cp_precision: Compute precision for CP (default: same as engine_dtype).
        code2wav_precision: Compute precision for code2wav (default: same as engine_dtype).
        max_batch_size: Max batch size.
        max_input_len: Max input length.
        max_seq_len: Max sequence length.
        build_gpu_device: GPU device for build (auto|N|cuda:N).

    Raises:
        FileNotFoundError: If target_profile or exported_dir not found.
        ValueError: If NGC tag/image cannot be resolved.
    """
    if not target_profile.is_file():
        raise FileNotFoundError(f"Target profile not found: {target_profile}")
    if not exported_dir.is_dir():
        raise FileNotFoundError(f"Exported dir not found: {exported_dir}")

    ngc_tag = _resolve_ngc_tag_from_profile(target_profile)
    ngc_image = _resolve_ngc_image_from_tag(ngc_tag)

    if not triton_io_float_dtype:
        triton_io_float_dtype = engine_dtype

    with tempfile.TemporaryDirectory() as tmp:
        bundle_root = Path(tmp) / "bundle"
        bundle_root.mkdir()

        # Copy target profile
        shutil.copy2(target_profile, bundle_root / "target_profile.json")

        # Generate build_on_target.sh (self-contained, no scripts/ dependency)
        script_path = bundle_root / "build_on_target.sh"
        script_path.write_text(
            generate_build_on_target_script(), encoding="utf-8"
        )
        script_path.chmod(0o755)

        # Copy ONNX files per variant
        for variant in variants:
            if not variant:
                continue
            variant_src = exported_dir / variant
            if not variant_src.is_dir():
                raise FileNotFoundError(
                    f"Variant export dir not found: {variant_src}"
                )
            variant_dst = bundle_root / "workspace" / "exported" / variant
            variant_dst.mkdir(parents=True, exist_ok=True)

            _link_or_copy_glob(variant_src, variant_dst, "*.onnx")
            _link_or_copy_glob(variant_src, variant_dst, "*.onnx.data")

            manifest_src = variant_src / "triton_manifest.json"
            if manifest_src.is_file():
                shutil.copy2(manifest_src, variant_dst / "triton_manifest.json")

            weights_src = variant_src / "weights"
            if weights_src.is_dir():
                shutil.copytree(weights_src, variant_dst / "weights", symlinks=True)

        # Copy tokenizer ONNX
        tokenizer_src = exported_dir / "tokenizer"
        if tokenizer_src.is_dir():
            tokenizer_dst = bundle_root / "workspace" / "exported" / "tokenizer"
            tokenizer_dst.mkdir(parents=True, exist_ok=True)
            _link_or_copy_glob(tokenizer_src, tokenizer_dst, "*.onnx")
            _link_or_copy_glob(tokenizer_src, tokenizer_dst, "*.onnx.data")

        # Write build_manifest.json
        build_manifest = BuildManifest(
            created_at_utc=_now_utc(),
            variants=variants,
            engine_dtype=engine_dtype,
            triton_io_float_dtype=triton_io_float_dtype,
            backbone_precision=backbone_precision,
            cp_precision=cp_precision,
            code2wav_precision=code2wav_precision,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            ngc_tag=ngc_tag,
            ngc_image=ngc_image,
            build_gpu_device=build_gpu_device,
        )
        build_manifest.write(bundle_root / "build_manifest.json")

        # Generate run.sh
        (bundle_root / "run.sh").write_text(
            _RUN_SH_CONTENT, encoding="utf-8"
        )
        (bundle_root / "run.sh").chmod(0o755)

        # Generate README.md
        (bundle_root / "README.md").write_text(
            _BUNDLE_README, encoding="utf-8"
        )

        # Pack into tarball
        out.parent.mkdir(parents=True, exist_ok=True)
        tar_cmd = _tar_cmd()
        subprocess.run(
            tar_cmd + ["-cf", str(out), "."],
            cwd=str(bundle_root),
            check=True,
        )

    logger.info("Engine build bundle written: %s (%s)", out, _tar_ext_note())


_RUN_SH_CONTENT = """\
#!/bin/bash
# ===========================================================================
#  run.sh — One-shot entry point for engine_build_bundle on a target host.
#  Usage:
#    bash run.sh                       # auto-detect runner (docker|host)
#    BUILD_RUNNER=host bash run.sh     # force host trtexec (DSW etc.)
#    BUILD_RUNNER=docker bash run.sh   # force docker NGC trtexec
#    TRTEXEC_HOST=/path/to/trtexec bash run.sh
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/build_on_target.sh" "$@"
"""

_BUNDLE_README = """\
# Qwen3-TTS TensorRT Engine Build Bundle

Run on the target production-like GPU host:

```bash
bash run.sh
```

This is a thin wrapper around `build_on_target.sh` that auto-detects
whether to compile via docker (NGC container) or directly with host
trtexec.  On Aliyun DSW or any container without docker, the bundle
will use the host trtexec found at `/usr/src/tensorrt/bin/trtexec`
(override with `TRTEXEC_HOST=`).

The script writes `engine_artifact_bundle.tar.zst` in the current directory.
Copy that artifact back to the packaging machine and run:

```bash
qwen3tts build import-artifact engine_artifact_bundle.tar.zst
```
"""


# ---------------------------------------------------------------------------
#  generate_build_on_target_script
# ---------------------------------------------------------------------------

def generate_build_on_target_script() -> str:
    """Generate the build_on_target.sh bash script for target machines.

    This script is self-contained: it does not require the qwen3tts Python
    package nor the scripts/bash/ directory. It only depends on:
    - python3 (available on all target hosts)
    - nvidia-smi (required for GPU detection)
    - docker (optional, for NGC container-based trtexec)

    Returns:
        Content of build_on_target.sh as a string.
    """
    return r"""#!/bin/bash
# ===========================================================================
#  build_on_target.sh — Compile TensorRT engines from a build bundle
#
#  Auto-generated by qwen3tts build make-bundle — do not edit.
#
#  This script runs on the target production-like GPU host. It reads
#  build_manifest.json and target_profile.json from the bundle root,
#  then invokes trtexec (via docker or host) to compile engines.
#
#  Environment variables:
#    BUILD_RUNNER   auto|docker|host (default: auto)
#    TRTEXEC_HOST   Path to host trtexec binary
# ===========================================================================
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info()  { echo "[INFO] $*"; }
log_error() { echo "[ERROR] $*" >&2; }
log_step()  { echo ""; echo "=== $* ==="; }

MANIFEST="$BUNDLE_ROOT/build_manifest.json"
TARGET_PROFILE="$BUNDLE_ROOT/target_profile.json"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: build_manifest.json not found in $BUNDLE_ROOT" >&2
    exit 1
fi
if [ ! -f "$TARGET_PROFILE" ]; then
    echo "ERROR: target_profile.json not found in $BUNDLE_ROOT" >&2
    exit 1
fi

# ── Read build parameters from manifest ──
ENGINE_DTYPE=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['engine_dtype'])" 2>/dev/null || echo "bf16")
IO_DTYPE=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['triton_io_float_dtype'])" 2>/dev/null || echo "$ENGINE_DTYPE")
MAX_BATCH=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['max_batch_size'])" 2>/dev/null || echo "64")
MAX_INPUT=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['max_input_len'])" 2>/dev/null || echo "128")
MAX_SEQ=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['max_seq_len'])" 2>/dev/null || echo "512")
NGC_TAG=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['ngc_tag'])" 2>/dev/null || echo "")
NGC_IMAGE=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['ngc_image'])" 2>/dev/null || echo "")
BUILD_DEVICE=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['build_gpu_device'])" 2>/dev/null || echo "auto")
VARIANTS_CSV=$(python3 -c "import json; print(','.join(json.load(open('$MANIFEST'))['variants']))" 2>/dev/null || echo "")

log_step "Building TensorRT engines on target host"
log_info "  engine_dtype:  $ENGINE_DTYPE"
log_info "  max_batch:     $MAX_BATCH"
log_info "  max_input:     $MAX_INPUT"
log_info "  max_seq:       $MAX_SEQ"
log_info "  ngc_image:     $NGC_IMAGE"
log_info "  variants:      $VARIANTS_CSV"

# ── Resolve runner ──
RUNNER="${BUILD_RUNNER:-auto}"
if [ "$RUNNER" = "auto" ]; then
    if command -v docker &>/dev/null && docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L >/dev/null 2>&1; then
        RUNNER="docker"
        log_info "  runner: docker (auto-detected)"
    else
        RUNNER="host"
        log_info "  runner: host (docker not available)"
    fi
fi

# ── Find trtexec binary ──
TRTEXEC_BIN=""
if [ "$RUNNER" = "host" ]; then
    if [ -n "${TRTEXEC_HOST:-}" ] && [ -f "$TRTEXEC_HOST" ]; then
        TRTEXEC_BIN="$TRTEXEC_HOST"
    elif [ -f "/usr/src/tensorrt/bin/trtexec" ]; then
        TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"
    elif command -v trtexec &>/dev/null; then
        TRTEXEC_BIN="trtexec"
    else
        log_error "trtexec not found. Set TRTEXEC_HOST=/path/to/trtexec"
        exit 1
    fi
    log_info "  trtexec: $TRTEXEC_BIN"
fi

# ── Resolve GPU device ──
if [ "$BUILD_DEVICE" = "auto" ] || [ -z "$BUILD_DEVICE" ]; then
    BUILD_DEVICE="0"
fi
BUILD_DEVICE="${BUILD_DEVICE#cuda:}"
log_info "  gpu_device: $BUILD_DEVICE"

# ── Normalize dtype ──
PREC_FLAG=""
case "$ENGINE_DTYPE" in
    bf16|bfloat16) PREC_FLAG="--bf16"; ENGINE_DTYPE="bf16" ;;
    fp16|float16)  PREC_FLAG="--fp16"; ENGINE_DTYPE="fp16" ;;
    fp32|float32)  PREC_FLAG="";       ENGINE_DTYPE="fp32" ;;
    fp8|float8)    PREC_FLAG="--fp8";   ENGINE_DTYPE="fp8" ;;
    *) log_error "Unknown ENGINE_DTYPE: $ENGINE_DTYPE"; exit 1 ;;
esac

# ── Run trtexec ──
_run_trtexec() {
    local onnx="$1" engine="$2"; shift 2
    if [ "$RUNNER" = "docker" ]; then
        local onnx_dir engine_dir
        onnx_dir="$(cd "$(dirname "$onnx")" && pwd)"
        engine_dir="$(cd "$(dirname "$engine")" && pwd)"
        docker run --rm --gpus "device=$BUILD_DEVICE" \
            -v "$onnx_dir:/models/onnx" \
            -v "$engine_dir:/models/output" \
            "$NGC_IMAGE" \
            /usr/src/tensorrt/bin/trtexec \
            --onnx=/models/onnx/$(basename "$onnx") \
            --saveEngine=/models/output/$(basename "$engine") \
            "$@"
    else
        CUDA_VISIBLE_DEVICES="$BUILD_DEVICE" \
        "$TRTEXEC_BIN" \
            --onnx="$onnx" \
            --saveEngine="$engine" \
            "$@"
    fi
}

FAILED=0
IFS=',' read -r -a VARIANT_ARRAY <<< "$VARIANTS_CSV"

for variant in "${VARIANT_ARRAY[@]}"; do
    [ -n "$variant" ] || continue
    VARIANT_DIR="$BUNDLE_ROOT/workspace/exported/$variant"
    log_step "Compiling: variant=$variant (runner=$RUNNER)"

    # ── speaker_encoder ──
    if [ -f "$VARIANT_DIR/speaker_encoder.onnx" ]; then
        log_info "  Building speaker_encoder.engine ..."
        IO_FMT="bf16:chw"
        [ "$ENGINE_DTYPE" = "fp16" ] && IO_FMT="fp16:chw"
        [ "$ENGINE_DTYPE" = "fp32" ] && IO_FMT="fp32:chw"
        if ! _run_trtexec \
            "$VARIANT_DIR/speaker_encoder.onnx" "$VARIANT_DIR/speaker_encoder.engine" \
            $PREC_FLAG \
            --inputIOFormats="$IO_FMT" \
            --outputIOFormats="$IO_FMT" \
            --minShapes=mel:1x1x128 \
            --optShapes=mel:1x300x128 \
            --maxShapes=mel:${MAX_BATCH}x1000x128 \
            --memPoolSize=workspace:1024; then
            log_error "speaker_encoder failed for $variant"
            FAILED=$((FAILED + 1))
        fi
    fi

    # ── speech_tokenizer_codec_fused (base variants only) ──
    if [[ "$variant" == base-* ]] && [ -f "$VARIANT_DIR/speech_tokenizer_codec_fused.onnx" ]; then
        log_info "  Building speech_tokenizer_codec_fused.engine ..."
        if ! _run_trtexec \
            "$VARIANT_DIR/speech_tokenizer_codec_fused.onnx" "$VARIANT_DIR/speech_tokenizer_codec_fused.engine" \
            --minShapes=waveform:1x1x960 \
            --optShapes=waveform:1x1x48000 \
            --maxShapes=waveform:1x1x192000 \
            --memPoolSize=workspace:6144; then
            log_error "speech_tokenizer_codec_fused failed for $variant"
            FAILED=$((FAILED + 1))
        fi
    fi

    # ── talker_code2wav_fused (primary production engine) ──
    if [ -f "$VARIANT_DIR/talker_code2wav_fused.onnx" ]; then
        log_info "  Building talker_code2wav_fused.engine ..."

        # Read dims from triton_manifest.json or fallback
        H=2048; KV=8; HD=128; NL=28
        if [ -f "$VARIANT_DIR/triton_manifest.json" ]; then
            read_dims=$(python3 - "$VARIANT_DIR/triton_manifest.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    c = d.get("talker_config", d)
    h = c.get("hidden_size", 2048)
    kv = c.get("num_key_value_heads", 8)
    hd = c.get("head_dim", 128)
    nl = c.get("num_hidden_layers", 28)
    print(h, kv, hd, nl)
except: print(2048, 8, 128, 28)
PY
            )
            if [ -n "$read_dims" ]; then
                read H KV HD NL <<< "$read_dims"
            fi
        fi
        log_info "    dims: H=$H KV=$KV HD=$HD NL=$NL"

        # Read code2wav dims from manifest
        N_C2W=8; N_CP=15
        if [ -f "$VARIANT_DIR/triton_manifest.json" ]; then
            N_C2W=$(python3 -c "import json; d=json.load(open('$VARIANT_DIR/triton_manifest.json')); print(int(d.get('code2wav_fused',{}).get('num_code2wav_hidden_layers',8)))" 2>/dev/null || echo "8")
            N_CP=$(python3 -c "import json; d=json.load(open('$VARIANT_DIR/triton_manifest.json')); print(int(d.get('architecture',{}).get('cp_num_stages',15)))" 2>/dev/null || echo "15")
        fi

        # Generate shape profiles using inline Python (same logic as trt_fused_talk_c2w_profiles.py)
        read FUSED_MIN FUSED_OPT FUSED_MAX <<< $(python3 - "$H" "$KV" "$HD" "$NL" "$MAX_BATCH" "$MAX_INPUT" "$MAX_SEQ" "$N_C2W" "$N_CP" <<'PYEOF'
import sys
H,KV,HD,NL = [int(x) for x in sys.argv[1:5]]
Bmax,MI,MS = sys.argv[5:8]
n_c2w,n_cp = int(sys.argv[8]), int(sys.argv[9])
V,K,SW = 3072,50,72
tkv1 = NL*2
c2w_kv = n_c2w*2
c2w_h,c2w_hd = 16,64
parts_min = [f"input_embeds:1x1x{H}",f"position_ids:1x3x1x1","attention_bias:1x1x1x1",
    f"token_counts:1x{V}",f"gumbel_noise:1x{K}",f"cp_gumbel_noise:1x{n_cp}x{K}",
    "temperature:1x1","penalty:1x1","cache_position:1x1","c2w_attention_bias:1x1x1x2",
    f"talker_past_kv:1x{tkv1}x{KV}x0x{HD}",f"c2w_past_kv:1x{c2w_kv}x{c2w_h}x1x{c2w_hd}"]
parts_opt = [f"input_embeds:1x1x{H}",f"position_ids:1x3x1x1","attention_bias:1x1x129",
    f"token_counts:1x{V}",f"gumbel_noise:1x{K}",f"cp_gumbel_noise:1x{n_cp}x{K}",
    "temperature:1x1","penalty:1x1","cache_position:1x1","c2w_attention_bias:1x1x1x5",
    f"talker_past_kv:1x{tkv1}x{KV}x128x{HD}",f"c2w_past_kv:1x{c2w_kv}x{c2w_h}x4x{c2w_hd}"]
parts_max = [f"input_embeds:{Bmax}x{MI}x{H}",f"position_ids:{Bmax}x3x{MI}x1",
    f"attention_bias:{Bmax}x1x{MI}x{int(MS)+int(MI)}",
    f"token_counts:{Bmax}x{V}",f"gumbel_noise:{Bmax}x{K}",f"cp_gumbel_noise:{Bmax}x{n_cp}x{K}",
    f"temperature:{Bmax}x1",f"penalty:{Bmax}x1",f"cache_position:{Bmax}x1",
    f"c2w_attention_bias:{Bmax}x1x1x{SW}",
    f"talker_past_kv:{Bmax}x{tkv1}x{KV}x{MS}x{HD}",f"c2w_past_kv:{Bmax}x{c2w_kv}x{c2w_h}x{SW-1}x{c2w_hd}"]
for specs in [
    [("conv_state_0","1x512x2"),("conv_state_1","1x1024x6"),("conv_state_2","1x1024x6"),("conv_state_3","1x1024x6"),
     ("conv_state_4","1x768x6"),("conv_state_5","1x768x18"),("conv_state_6","1x768x54"),("conv_state_7","1x384x6"),
     ("conv_state_8","1x384x18"),("conv_state_9","1x384x54"),("conv_state_10","1x192x6"),("conv_state_11","1x192x18"),
     ("conv_state_12","1x192x54"),("conv_state_13","1x96x6"),("conv_state_14","1x96x18"),("conv_state_15","1x96x54"),("conv_state_16","1x96x6")],
    [("transconv_overlap_0","1x768x8"),("transconv_overlap_1","1x384x5"),("transconv_overlap_2","1x192x4"),("transconv_overlap_3","1x96x3")]]:
    for n,s in specs:
        r=s[2:]; parts_min.append(f"c2w_{n}:{s}"); parts_opt.append(f"c2w_{n}:{s}"); parts_max.append(f"c2w_{n}:{Bmax}x{r}")
print(",".join(parts_min)); print(",".join(parts_opt)); print(",".join(parts_max))
PYEOF
)
        # I/O formats from manifest
        IO_IN="" IO_OUT=""
        if [ -f "$VARIANT_DIR/triton_manifest.json" ]; then
            read IO_IN IO_OUT <<< $(python3 - "$VARIANT_DIR/triton_manifest.json" <<'PYEOF2'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    raw = d.get("triton_io_float_dtype", d.get("onnx_io_dtype", "fp32")).lower().strip()
    if raw in ("bfloat16","bf16"): ft="bf16:chw"
    elif raw in ("float16","fp16"): ft="fp16:chw"
    else: ft="fp32:chw"
    i64="int64:chw"
    c2w = d.get("code2wav_fused") or {}
    c2w_in = list(c2w.get("c2w_state_input_names") or [])
    c2w_out = list(c2w.get("c2w_state_output_names") or [])
    inp = [ft,i64,ft,i64,"fp32:chw","fp32:chw","fp32:chw","fp32:chw","fp32:chw",ft,ft,ft]+[ft]*len(c2w_in)
    out = [ft,ft,i64,ft,ft,i64,ft,ft]+[ft]*len(c2w_out)
    print(",".join(inp)); print(",".join(out))
except: print(""); print("")
PYEOF2
)
        fi

        TRTEXEC_ARGS="$PREC_FLAG --memPoolSize=workspace:8192"
        TRTEXEC_ARGS="$TRTEXEC_ARGS --minShapes=$FUSED_MIN --optShapes=$FUSED_OPT --maxShapes=$FUSED_MAX"
        if [ -n "$IO_IN" ] && [ -n "$IO_OUT" ]; then
            TRTEXEC_ARGS="$TRTEXEC_ARGS --inputIOFormats=$IO_IN --outputIOFormats=$IO_OUT"
        fi
        if ! _run_trtexec \
            "$VARIANT_DIR/talker_code2wav_fused.onnx" "$VARIANT_DIR/talker_code2wav_fused.engine" \
            $TRTEXEC_ARGS; then
            log_error "talker_code2wav_fused failed for $variant"
            FAILED=$((FAILED + 1))
        fi
    else
        log_error "Missing talker_code2wav_fused.onnx for $variant"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    log_error "$FAILED variant(s) failed"
    exit 1
fi

# ── Write engine dtype marker ──
echo "$ENGINE_DTYPE" > "$BUNDLE_ROOT/workspace/exported/.engine_dtype"

# ── Write artifact manifest ──
log_step "Writing artifact manifest"
python3 - "$BUNDLE_ROOT" "$MANIFEST" "$TARGET_PROFILE" \
    "$(trtexec --version 2>/dev/null | head -1 || echo "")" \
    "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")" \
    "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.' | sed 's/^/sm_/' || echo "")" <<'PY'
import datetime as dt, hashlib, json, socket, sys
from pathlib import Path
root, bm, tp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
trt_ver, driver, sm = sys.argv[4:7]
build = json.loads(bm.read_text()); target = json.loads(tp.read_text())
engines = {}
for e in sorted((root / "workspace" / "exported").glob("**/*.engine")):
    rel = e.relative_to(root / "workspace" / "exported").as_posix()
    h = hashlib.sha256()
    with e.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): h.update(chunk)
    engines[rel] = {"sha256": h.hexdigest(), "size": e.stat().st_size}
gpus = target.get("gpus") or []; gpu = gpus[0] if gpus else {}
manifest = {
    "artifact_schema_version": 1,
    "built_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    "build_host": socket.gethostname(),
    "ngc_tag": build["ngc_tag"], "ngc_image": build["ngc_image"],
    "tensorrt_version": trt_ver, "gpu_sm": sm,
    "gpu_name": gpu.get("name", ""), "driver_version": driver,
    "engine_dtype": build["engine_dtype"],
    "triton_io_float_dtype": build["triton_io_float_dtype"],
    "max_batch_size": build["max_batch_size"],
    "max_input_len": build["max_input_len"],
    "max_seq_len": build["max_seq_len"],
    "engines": engines,
}
(root / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
PY

log_info "Artifact manifest written: $BUNDLE_ROOT/artifact_manifest.json"

# ── Pack artifact bundle ──
log_step "Packing artifact bundle"
ARTIFACT_OUT="$BUNDLE_ROOT/engine_artifact_bundle.tar.zst"
if command -v tar &>/dev/null && tar --help 2>/dev/null | grep -q -- "--zstd"; then
    TAR_CMD="tar --zstd"
else
    TAR_CMD="tar -z"
fi

TMP_PACK=$(mktemp -d)
trap 'rm -rf "$TMP_PACK"' EXIT
mkdir -p "$TMP_PACK/exported"

cp "$BUNDLE_ROOT/artifact_manifest.json" "$TMP_PACK/artifact_manifest.json"

for v_dir in "$BUNDLE_ROOT/workspace/exported"/*/; do
    [ -d "$v_dir" ] || continue
    v=$(basename "$v_dir")
    mkdir -p "$TMP_PACK/exported/$v"
    cp -a "$v_dir"*.engine "$TMP_PACK/exported/$v/" 2>/dev/null || true
    [ -f "$v_dir/triton_manifest.json" ] && cp -a "$v_dir/triton_manifest.json" "$TMP_PACK/exported/$v/"
done

echo "$ENGINE_DTYPE" > "$TMP_PACK/exported/.engine_dtype"

(cd "$TMP_PACK" && $TAR_CMD -cf "$ARTIFACT_OUT" .)
log_info "Engine artifact bundle written: $ARTIFACT_OUT"
log_step "Done. Copy $ARTIFACT_OUT back to the packaging machine."
"""


# ---------------------------------------------------------------------------
#  extract_artifact_bundle
# ---------------------------------------------------------------------------

def extract_artifact_bundle(
    repo_root: Path,
    exported_dir: Path,
    artifact: Path,
    strict: bool = True,
) -> None:
    """Import compiled engines from an artifact bundle.

    Args:
        repo_root: Repository root path.
        exported_dir: Path to workspace/exported/ to import into.
        artifact: Path to engine_artifact_bundle.tar.zst.
        strict: If True, run fingerprint check and fail on mismatch.

    Raises:
        FileNotFoundError: If artifact not found.
        RuntimeError: If fingerprint check fails in strict mode.
    """
    if not artifact.is_file():
        raise FileNotFoundError(f"Engine artifact bundle not found: {artifact}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tar_cmd = _tar_cmd()
        subprocess.run(
            tar_cmd + ["-xf", str(artifact), "-C", str(tmp_path)],
            check=True,
        )

        manifest_path = tmp_path / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("artifact_manifest.json missing in artifact bundle")

        if strict:
            errors = fingerprint_check(manifest_path, exported_dir)
            if errors:
                msg = "Engine fingerprint mismatch:\n" + "\n".join(f"  - {e}" for e in errors)
                raise RuntimeError(msg)
            logger.info("Engine fingerprint check: OK")

        # Copy exported files
        exported_tmp = tmp_path / "exported"
        if not exported_tmp.is_dir():
            raise RuntimeError("exported/ missing in artifact bundle")

        exported_dir.mkdir(parents=True, exist_ok=True)
        # Use rsync if available for clean delta copy
        if shutil.which("rsync"):
            subprocess.run(
                [
                    "rsync", "-a",
                    "--include=*/", "--include=*.engine",
                    "--include=triton_manifest.json",
                    "--include=.engine_dtype", "--exclude=*",
                    str(exported_tmp) + "/", str(exported_dir) + "/",
                ],
                check=True,
            )
        else:
            for d in sorted(exported_tmp.iterdir()):
                if not d.is_dir():
                    continue
                dst = exported_dir / d.name
                dst.mkdir(exist_ok=True)
                for f in d.glob("*.engine"):
                    shutil.copy2(f, dst / f.name)
                mf = d / "triton_manifest.json"
                if mf.is_file():
                    shutil.copy2(mf, dst / mf.name)

        # Copy artifact manifest
        shutil.copy2(manifest_path, exported_dir / "artifact_manifest.json")

        # Copy engine dtype marker
        dtype_marker = tmp_path / "exported" / ".engine_dtype"
        if dtype_marker.is_file():
            shutil.copy2(dtype_marker, exported_dir / ".engine_dtype")
        else:
            # Derive from manifest
            am = ArtifactManifest.load(manifest_path)
            (exported_dir / ".engine_dtype").write_text(am.engine_dtype or "bf16")

    logger.info("Imported TensorRT engine artifact into: %s", exported_dir)


# ---------------------------------------------------------------------------
#  prepare_local_workspace
# ---------------------------------------------------------------------------

def prepare_local_workspace(
    repo_root: Path,
    exported_dir: Path,
    target_profile: Path,
    bundle_root: Path,
    variants: list[str],
    engine_dtype: str = "bf16",
    triton_io_float_dtype: str = "",
    max_batch_size: int = 64,
    max_input_len: int = 128,
    max_seq_len: int = 512,
    build_gpu_device: str = "auto",
) -> None:
    """Prepare a bundle-shaped directory for local builds using hardlinks.

    Args:
        repo_root: Repository root path.
        exported_dir: Source workspace/exported/ with ONNX inputs.
        target_profile: Path to target_profile.json.
        bundle_root: Destination directory (created/cleaned).
        variants: List of variant names.
        engine_dtype: Engine dtype.
        triton_io_float_dtype: Triton IO float dtype.
        max_batch_size: Max batch size.
        max_input_len: Max input length.
        max_seq_len: Max sequence length.
        build_gpu_device: GPU device for build.
    """
    if not target_profile.is_file():
        raise FileNotFoundError(f"target_profile.json missing: {target_profile}")
    if not exported_dir.is_dir():
        raise FileNotFoundError(f"exported dir missing: {exported_dir}")

    # Clean and create
    if bundle_root.is_dir():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)

    (bundle_root / "workspace" / "exported").mkdir(parents=True)

    shutil.copy2(target_profile, bundle_root / "target_profile.json")

    # Resolve NGC tag/image from target_profile
    ngc_tag = _resolve_ngc_tag_from_profile(target_profile)
    ngc_image = _resolve_ngc_image_from_tag(ngc_tag)

    if not triton_io_float_dtype:
        triton_io_float_dtype = engine_dtype

    # Materialize ONNX inputs via hardlinks
    for variant in variants:
        if not variant:
            continue
        variant_src = exported_dir / variant
        if not variant_src.is_dir():
            raise FileNotFoundError(f"Variant export dir not found: {variant_src}")

        variant_dst = bundle_root / "workspace" / "exported" / variant
        variant_dst.mkdir(parents=True)

        _link_or_copy_glob(variant_src, variant_dst, "*.onnx")
        _link_or_copy_glob(variant_src, variant_dst, "*.onnx.data")

        manifest_src = variant_src / "triton_manifest.json"
        if manifest_src.is_file():
            shutil.copy2(manifest_src, variant_dst / "triton_manifest.json")

        weights_src = variant_src / "weights"
        if weights_src.is_dir():
            shutil.copytree(weights_src, variant_dst / "weights", symlinks=True)

    # Tokenizer
    tokenizer_src = exported_dir / "tokenizer"
    if tokenizer_src.is_dir():
        tokenizer_dst = bundle_root / "workspace" / "exported" / "tokenizer"
        tokenizer_dst.mkdir(parents=True)
        _link_or_copy_glob(tokenizer_src, tokenizer_dst, "*.onnx")
        _link_or_copy_glob(tokenizer_src, tokenizer_dst, "*.onnx.data")

    # Symlink repo root for local builds (used by trtexec.py to find scripts)
    repo_link = bundle_root / ".repo_root"
    if not repo_link.exists():
        repo_link.write_text(str(repo_root), encoding="utf-8")

    # Write build_manifest.json
    build_manifest = BuildManifest(
        created_at_utc=_now_utc(),
        variants=variants,
        engine_dtype=engine_dtype,
        triton_io_float_dtype=triton_io_float_dtype,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_seq_len=max_seq_len,
        ngc_tag=ngc_tag,
        ngc_image=ngc_image,
        build_gpu_device=build_gpu_device,
    )
    build_manifest.write(bundle_root / "build_manifest.json")

    logger.info("Local bundle workspace ready: %s", bundle_root)


# ---------------------------------------------------------------------------
#  compile_engines_in_bundle
# ---------------------------------------------------------------------------

def compile_engines_in_bundle(bundle_root: Path) -> None:
    """Compile engines inside a bundle workspace using Python-native trtexec.

    Args:
        bundle_root: Path to the bundle root directory.

    Raises:
        FileNotFoundError: If bundle structure is invalid.
        RuntimeError: If compilation fails.
    """
    manifest_path = bundle_root / "build_manifest.json"
    target_profile = bundle_root / "target_profile.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing build_manifest.json in {bundle_root}")
    if not target_profile.is_file():
        raise FileNotFoundError(f"Missing target_profile.json in {bundle_root}")

    manifest = BuildManifest.load(manifest_path)

    from qwen3tts_tools.trtexec import (
        build_all_engines,
        resolve_runner,
    )

    runner = resolve_runner()
    logger.info("compile_engines_in_bundle: runner=%s", runner)

    exported_dir = bundle_root / "workspace" / "exported"
    failed = 0

    for variant in manifest.variants:
        logger.info("Bundle compile: variant=%s (runner=%s)", variant, runner)
        result = build_all_engines(
            exported_dir=exported_dir,
            variant=variant,
            dtype=manifest.engine_dtype,
            triton_io_float_dtype=manifest.triton_io_float_dtype,
            backbone_precision=manifest.backbone_precision,
            cp_precision=manifest.cp_precision,
            code2wav_precision=manifest.code2wav_precision,
            runner=runner,
            max_batch_size=manifest.max_batch_size,
            max_input_len=manifest.max_input_len,
            max_seq_len=manifest.max_seq_len,
            device=manifest.build_gpu_device,
            ngc_image=manifest.ngc_image,
        )
        if result != 0:
            failed += 1

    if failed > 0:
        raise RuntimeError(f"compile_engines_in_bundle: {failed} variant(s) failed")


# ---------------------------------------------------------------------------
#  write_artifact_manifest
# ---------------------------------------------------------------------------

def write_artifact_manifest(bundle_root: Path) -> None:
    """Write artifact_manifest.json with SHA256 for each engine.

    Args:
        bundle_root: Path to the bundle root directory.
    """
    build_manifest_path = bundle_root / "build_manifest.json"
    target_profile_path = bundle_root / "target_profile.json"

    if not build_manifest_path.is_file():
        raise FileNotFoundError("Missing build_manifest.json")
    if not target_profile_path.is_file():
        raise FileNotFoundError("Missing target_profile.json")

    build_manifest = BuildManifest.load(build_manifest_path)
    target_data = json.loads(target_profile_path.read_text(encoding="utf-8"))

    # Compute SHA256 for each engine
    engines: dict[str, dict[str, Any]] = {}
    exported_dir = bundle_root / "workspace" / "exported"
    for engine_path in sorted(exported_dir.glob("**/*.engine")):
        rel = engine_path.relative_to(exported_dir).as_posix()
        h = hashlib.sha256()
        with engine_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        engines[rel] = {"sha256": h.hexdigest(), "size": engine_path.stat().st_size}

    # Resolve runtime metadata
    tensorrt_version = _probe_tensorrt_version(build_manifest.ngc_image)
    driver_version = _probe_driver_version()
    gpu_sm, gpu_name = _probe_gpu_info(build_manifest.build_gpu_device)
    cuda_version = str(target_data.get("cuda_runtime", "") or "")

    gpus = target_data.get("gpus") or []
    gpu = gpus[0] if gpus else {}

    artifact = ArtifactManifest(
        built_at_utc=_now_utc(),
        build_host=socket.gethostname(),
        ngc_tag=build_manifest.ngc_tag,
        ngc_image=build_manifest.ngc_image,
        tensorrt_version=tensorrt_version,
        cuda_version=cuda_version,
        gpu_sm=gpu_sm,
        gpu_name=gpu_name or gpu.get("name", ""),
        driver_version=driver_version,
        engine_dtype=build_manifest.engine_dtype,
        triton_io_float_dtype=build_manifest.triton_io_float_dtype,
        max_batch_size=build_manifest.max_batch_size,
        max_input_len=build_manifest.max_input_len,
        max_seq_len=build_manifest.max_seq_len,
        engines=engines,
    )
    artifact.write(bundle_root / "artifact_manifest.json")
    logger.info("Artifact manifest written: %s/artifact_manifest.json", bundle_root)


def _probe_tensorrt_version(ngc_image: str) -> str:
    """Probe TensorRT version from host or Docker."""
    # Try host trtexec
    trtexec_env = os.environ.get("TRTEXEC_HOST", "")
    candidates = []
    if trtexec_env:
        candidates.append(trtexec_env)
    candidates.extend([
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
    ])
    for path in candidates:
        if Path(path).is_file():
            try:
                result = subprocess.run(
                    [path, "--version"], capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "TensorRT" in line:
                            return line.strip()
            except (subprocess.TimeoutExpired, OSError):
                pass

    # Try Docker
    if ngc_image and shutil.which("docker"):
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm", ngc_image,
                    "/bin/bash", "-lc",
                    "/usr/src/tensorrt/bin/trtexec --version 2>/dev/null | head -20",
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "TensorRT" in line:
                        return line.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass

    return ""


def _probe_driver_version() -> str:
    """Probe NVIDIA driver version."""
    try:
        from qwen3tts_tools.docker import detect_driver_version
        return detect_driver_version() or ""
    except ImportError:
        pass
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0].strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
    return ""


def _probe_gpu_info(device: str = "auto") -> tuple[str, str]:
    """Probe GPU SM and name.

    Returns:
        (gpu_sm, gpu_name) tuple, e.g. ("sm_89", "NVIDIA GeForce RTX 4090").
    """
    probe_device = device
    if probe_device in ("auto", "all", ""):
        probe_device = "0"
    probe_device = probe_device.removeprefix("cuda:").removeprefix("device=")

    if not shutil.which("nvidia-smi"):
        return "", ""

    try:
        result = subprocess.run(
            [
                "nvidia-smi", f"--id={probe_device}",
                "--query-gpu=compute_cap,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                cc = parts[0].strip().replace(".", "")
                name = parts[1].strip()
                return f"sm_{cc}", name
    except (subprocess.TimeoutExpired, OSError):
        pass

    return "", ""


# ---------------------------------------------------------------------------
#  collect_engines_to_workspace
# ---------------------------------------------------------------------------

def collect_engines_to_workspace(bundle_root: Path, exported_dir: Path) -> None:
    """Copy compiled engines from bundle back to workspace/exported/.

    Args:
        bundle_root: Bundle root with workspace/exported/ containing engines.
        exported_dir: Target workspace/exported/ directory.
    """
    bundle_exported = bundle_root / "workspace" / "exported"
    if not bundle_exported.is_dir():
        raise RuntimeError(f"Bundle workspace/exported missing: {bundle_root}")

    exported_dir.mkdir(parents=True, exist_ok=True)

    # Use rsync if available
    if shutil.which("rsync"):
        subprocess.run(
            [
                "rsync", "-a",
                "--include=*/", "--include=*.engine",
                "--include=triton_manifest.json",
                "--include=.engine_dtype", "--exclude=*",
                str(bundle_exported) + "/", str(exported_dir) + "/",
            ],
            check=True,
        )
    else:
        for d in sorted(bundle_exported.iterdir()):
            if not d.is_dir():
                continue
            dst = exported_dir / d.name
            dst.mkdir(exist_ok=True)
            for f in d.glob("*.engine"):
                shutil.copy2(f, dst / f.name)
            mf = d / "triton_manifest.json"
            if mf.is_file():
                shutil.copy2(mf, dst / mf.name)

    # Copy artifact manifest and dtype marker
    artifact_manifest = bundle_root / "artifact_manifest.json"
    if artifact_manifest.is_file():
        shutil.copy2(artifact_manifest, exported_dir / "artifact_manifest.json")

    dtype_marker = bundle_exported / ".engine_dtype"
    if dtype_marker.is_file():
        shutil.copy2(dtype_marker, exported_dir / ".engine_dtype")

    logger.info("Engines collected into: %s", exported_dir)


# ---------------------------------------------------------------------------
#  pack_artifact_bundle
# ---------------------------------------------------------------------------

def pack_artifact_bundle(bundle_root: Path, out: Path) -> None:
    """Pack compiled engines into an artifact tarball.

    Args:
        bundle_root: Bundle root with artifact_manifest.json and engines.
        out: Output path for the artifact tarball.
    """
    artifact_manifest = bundle_root / "artifact_manifest.json"
    if not artifact_manifest.is_file():
        raise RuntimeError(f"artifact_manifest.json missing in bundle root: {bundle_root}")

    # Read engine dtype from build manifest
    build_manifest_path = bundle_root / "build_manifest.json"
    engine_dtype = "bf16"
    if build_manifest_path.is_file():
        bm = BuildManifest.load(build_manifest_path)
        engine_dtype = bm.engine_dtype

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        exported_dst = tmp_path / "exported"
        exported_dst.mkdir()

        shutil.copy2(artifact_manifest, tmp_path / "artifact_manifest.json")

        # Collect engines per variant
        bundle_exported = bundle_root / "workspace" / "exported"
        for v_dir in sorted(bundle_exported.iterdir()):
            if not v_dir.is_dir():
                continue
            v_dst = exported_dst / v_dir.name
            v_dst.mkdir()
            for f in v_dir.glob("*.engine"):
                shutil.copy2(f, v_dst / f.name)
            mf = v_dir / "triton_manifest.json"
            if mf.is_file():
                shutil.copy2(mf, v_dst / mf.name)

        (exported_dst / ".engine_dtype").write_text(engine_dtype)

        out.parent.mkdir(parents=True, exist_ok=True)
        tar_cmd = _tar_cmd()
        subprocess.run(
            tar_cmd + ["-cf", str(out), "."],
            cwd=str(tmp_path),
            check=True,
        )

    logger.info("Engine artifact bundle written: %s", out)
