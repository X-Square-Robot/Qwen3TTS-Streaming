"""Build subcommand — Phase B: compile TensorRT engines.

Supports local compilation and cross-host build workflows:

- ``qwen3tts build`` — local compilation (default)
- ``qwen3tts build make-bundle`` — create cross-host build bundle
- ``qwen3tts build import-artifact`` — import compiled engines from artifact bundle
- ``qwen3tts build remote-build`` — SSH-driven remote compilation
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_build(args: object) -> int:
    """Dispatch build subcommand based on ``build_action``."""
    action = getattr(args, "build_action", None) or "local"

    dispatch = {
        "local": _build_local,
        "make-bundle": _build_make_bundle,
        "import-artifact": _build_import_artifact,
        "remote-build": _build_remote_build,
    }
    handler = dispatch.get(action)
    if handler is None:
        print(f"Error: Unknown build action: {action}", file=sys.stderr)
        return 1
    return handler(args)


# ---------------------------------------------------------------------------
#  NGC tag resolution (mirrors dev branch autorun.sh build_forward_args)
# ---------------------------------------------------------------------------

def _resolve_ngc_tag(args: object, *, for_phase: str = "build") -> str:
    """Resolve the NGC tag with the same priority as dev branch autorun.sh.

    Priority order (matches build_forward_args in autorun.sh):
      1. --target-profile → read recommended_ngc_tag from target_profile.json
      2. --target-driver   → derive tag from the target driver version
      3. --ngc-tag         → user-specified tag directly
      4. local driver      → derive from the current host's driver (fallback)

    For ``deploy`` phase, we skip the local driver fallback — deploy.sh
    reads the tag from the Phase B manifest instead.
    """
    from qwen3tts_tools.ngc_matrix import (
        load_ngc_matrix,
        resolve_ngc_entry_by_tag,
        resolve_ngc_image,
        resolve_ngc_tag,
        resolve_ngc_tag_from_profile,
    )

    ngc_tag = getattr(args, "ngc_tag", "") or ""
    target_driver = getattr(args, "target_driver", "") or ""
    target_profile = getattr(args, "target_profile", "") or ""

    # Step 1: --target-profile takes highest priority
    if target_profile:
        tp_path = Path(target_profile)
        if tp_path.is_file():
            tag_from_profile = resolve_ngc_tag_from_profile(tp_path)
            if tag_from_profile:
                return tag_from_profile

    # Step 2: --target-driver
    if target_driver:
        tag = resolve_ngc_tag(target_driver) or ""
        if tag:
            return tag

    # Step 3: --ngc-tag (explicit user override)
    if ngc_tag:
        # Validate against the matrix (warn but don't block)
        entry = resolve_ngc_entry_by_tag(ngc_tag)
        if entry is None:
            print(f"Warning: --ngc-tag {ngc_tag} not found in NGC matrix. Using it anyway.", file=sys.stderr)
        return ngc_tag

    # Step 4: local driver fallback (only for build/setup/all phases)
    if for_phase in ("build", "setup", "all"):
        try:
            from qwen3tts_tools.docker import detect_driver_version
            driver = detect_driver_version()
            if driver:
                tag = resolve_ngc_tag(driver) or ""
                if tag:
                    return tag
        except Exception:
            pass

    return ""


def _resolve_ngc_image(ngc_tag: str) -> str:
    """Get the full NGC image URI from a tag."""
    if not ngc_tag:
        return ""
    from qwen3tts_tools.ngc_matrix import resolve_ngc_image
    image = resolve_ngc_image(ngc_tag) or ""
    if not image:
        # Construct from pattern if not in matrix
        from qwen3tts_tools.ngc_matrix import NGC_TRITON_BASE, NGC_PY3_SUFFIX
        image = f"{NGC_TRITON_BASE}:{ngc_tag}{NGC_PY3_SUFFIX}"
    return image


def _apply_ngc_env(ngc_tag: str) -> dict[str, str]:
    """Set NGC-related environment variables (mirrors dev branch build_forward_args).

    Returns a dict of env vars to export.
    """
    from qwen3tts_tools.ngc_matrix import (
        load_ngc_matrix,
        resolve_ngc_entry_by_tag,
        NGC_TRITON_BASE,
        NGC_PY3_SUFFIX,
    )

    env_updates: dict[str, str] = {}
    if not ngc_tag:
        return env_updates

    env_updates["NGC_TAG"] = ngc_tag

    entry = resolve_ngc_entry_by_tag(ngc_tag)
    if entry:
        env_updates["TRITON_BASE_IMAGE"] = f"{NGC_TRITON_BASE}:{ngc_tag}{NGC_PY3_SUFFIX}"
        env_updates["ENGINE_BASE_IMAGE"] = f"nvcr.io/nvidia/tensorrt:{ngc_tag}-py3"
        env_updates["TRITON_IMAGE"] = f"qwen3-tts-triton:{ngc_tag}"
        env_updates["ENGINE_IMAGE"] = f"qwen3-engine:{ngc_tag}"
        if entry.python_version:
            # Try to resolve torch CUDA tag
            try:
                from qwen3tts_tools.ngc_matrix import _DEFAULT_MATRIX_PATH
                # Read from matrix conf for torch tag
                import re
                conf_text = _DEFAULT_MATRIX_PATH.read_text(encoding="utf-8") if _DEFAULT_MATRIX_PATH.is_file() else ""
                for line in conf_text.splitlines():
                    parts = line.split()
                    if len(parts) >= 6 and parts[0] == ngc_tag:
                        torch_tag = parts[5] if len(parts) > 5 else ""
                        if torch_tag and torch_tag != "-":
                            env_updates["PYTORCH_CUDA_TAG"] = torch_tag
                            env_updates["ENGINE_PYTORCH_CUDA_TAG"] = torch_tag
                            env_updates["TRITON_PYTORCH_CUDA_TAG"] = torch_tag
                        break
            except Exception:
                pass
        if entry.tensorrt_version:
            env_updates["TRITON_TENSORRT_PYTHON_VERSION"] = entry.tensorrt_version
            env_updates["TRITON_TENSORRT_PIP_VERSION"] = entry.tensorrt_version
            env_updates["STANDALONE_ENGINE_TENSORRT_PIP_VERSION"] = entry.tensorrt_version
            # Resolve pip package name from CUDA version
            cuda_ver = entry.cuda_versions
            major = cuda_ver.split(".")[0] if cuda_ver else ""
            if major == "13":
                env_updates["STANDALONE_ENGINE_TENSORRT_PIP_PACKAGE"] = "tensorrt-cu13"
            elif major == "12":
                env_updates["STANDALONE_ENGINE_TENSORRT_PIP_PACKAGE"] = "tensorrt-cu12"
            else:
                env_updates["STANDALONE_ENGINE_TENSORRT_PIP_PACKAGE"] = "tensorrt"

    return env_updates


# ---------------------------------------------------------------------------
#  Local build
# ---------------------------------------------------------------------------

def _build_local(args: object) -> int:
    """Build engines locally using Python-native trtexec functions."""
    from qwen3tts_tools.common import REPO_ROOT
    from qwen3tts_tools.trtexec import (
        build_all_engines,
        resolve_runner,
        suggest_build_profile,
    )
    from qwen3tts_tools.docker import detect_gpu_total_memory_mb

    variant = getattr(args, "variant", "") or "custom-1.7b"
    dtype = getattr(args, "dtype", "bf16")
    triton_io_float_dtype = getattr(args, "triton_io_float_dtype", "") or dtype
    backbone_precision = getattr(args, "backbone_precision", "") or ""
    cp_precision = getattr(args, "cp_precision", "") or ""
    code2wav_precision = getattr(args, "code2wav_precision", "") or ""
    dry_run = getattr(args, "dry_run", False)
    max_batch_size = getattr(args, "max_batch_size", 0)
    max_seq_len = getattr(args, "max_seq_len", 0)
    max_input_len = getattr(args, "max_input_len", 0)
    device = getattr(args, "device", "auto")
    build_device = getattr(args, "build_device", "") or device
    ngc_image = getattr(args, "image", "")

    exported_dir = REPO_ROOT / "workspace" / "exported" / variant
    if not exported_dir.is_dir():
        print(f"Error: Exported models not found: {exported_dir}", file=sys.stderr)
        print("Run 'qwen3tts setup' first.", file=sys.stderr)
        return 1

    # Resolve build profile if not specified
    if max_batch_size == 0 or max_seq_len == 0:
        total_mb = detect_gpu_total_memory_mb()
        suggested_batch, suggested_input, suggested_seq = suggest_build_profile(total_mb)
        if max_batch_size == 0:
            max_batch_size = suggested_batch
        if max_input_len == 0:
            max_input_len = suggested_input
        if max_seq_len == 0:
            max_seq_len = suggested_seq

    # Resolve runner
    try:
        runner = resolve_runner()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Resolve NGC tag using dev-branch priority order:
    #   --target-profile > --target-driver > --ngc-tag > local driver
    ngc_tag = _resolve_ngc_tag(args, for_phase="build")
    if ngc_tag and not ngc_image:
        ngc_image = _resolve_ngc_image(ngc_tag)

    # Export NGC environment variables (for downstream tools)
    env_updates = _apply_ngc_env(ngc_tag)
    if env_updates:
        import os
        for key, value in env_updates.items():
            os.environ[key] = value
            logger.info("export %s=%s", key, value)

    print(f"Building TensorRT engines for variant: {variant}")
    print(f"  dtype: {dtype}")
    print(f"  triton_io_float_dtype: {triton_io_float_dtype}")
    if backbone_precision:
        print(f"  backbone_precision: {backbone_precision}")
    if cp_precision:
        print(f"  cp_precision: {cp_precision}")
    if code2wav_precision:
        print(f"  code2wav_precision: {code2wav_precision}")
    print(f"  max_batch_size: {max_batch_size}")
    print(f"  max_input_len: {max_input_len}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  device: {build_device}")
    print(f"  runner: {runner}")
    if ngc_tag:
        print(f"  ngc_tag: {ngc_tag}")
    if ngc_image:
        print(f"  ngc_image: {ngc_image}")

    if dry_run:
        print("[DRY RUN] Would compile TensorRT engines.")
        return 0

    # Try Python-native build first
    try:
        failed = build_all_engines(
            exported_dir=REPO_ROOT / "workspace" / "exported",
            variant=variant,
            dtype=dtype,
            triton_io_float_dtype=triton_io_float_dtype,
            backbone_precision=backbone_precision,
            cp_precision=cp_precision,
            code2wav_precision=code2wav_precision,
            runner=runner,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            device=build_device,
            ngc_image=ngc_image,
            dry_run=dry_run,
            build_verification=_should_build_verification(),
        )
        if failed == 0:
            print("Phase B (build) complete.")
            return 0
        else:
            print(f"Error: {failed} engine(s) failed to build.", file=sys.stderr)
            return 1

    except (ImportError, NotImplementedError) as e:
        logger.info("Python-native build not available (%s), falling back to Bash", e)

    # Fallback to Bash
    return _fallback_bash_build(args)


def _should_build_verification() -> bool:
    """Check if BUILD_VERIFICATION_ENGINES env var is set."""
    import os
    return os.environ.get("BUILD_VERIFICATION_ENGINES", "0") == "1"


def _fallback_bash_build(args: object) -> int:
    """Fall back to Bash build_engines.sh."""
    from qwen3tts_tools.common import REPO_ROOT

    variant = getattr(args, "variant", "") or "custom-1.7b"
    bash_script = REPO_ROOT / "scripts" / "bash" / "build_engines.sh"
    if not bash_script.is_file():
        print(f"Error: Build script not found: {bash_script}", file=sys.stderr)
        return 1

    print("(Falling back to Bash build script)")
    cmd = ["bash", str(bash_script), "build"]
    if variant:
        cmd += ["--variant", variant]
    if getattr(args, "max_batch_size", 0):
        cmd += ["--max-batch-size", str(args.max_batch_size)]
    if getattr(args, "max_seq_len", 0):
        cmd += ["--max-seq-len", str(args.max_seq_len)]
    if getattr(args, "dtype", "bf16"):
        cmd += ["--dtype", args.dtype]
    if getattr(args, "image", ""):
        cmd += ["--image", args.image]
    build_device = getattr(args, "build_device", "") or getattr(args, "device", "auto")
    if build_device != "auto":
        cmd += ["--device", build_device]
    # Forward NGC tag and target profile
    if getattr(args, "ngc_tag", ""):
        cmd += ["--ngc-tag", args.ngc_tag]
    if getattr(args, "target_profile", ""):
        cmd += ["--target-profile", args.target_profile]
    if getattr(args, "target_driver", ""):
        cmd += ["--target-driver", args.target_driver]

    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
#  make-bundle
# ---------------------------------------------------------------------------

def _build_make_bundle(args: object) -> int:
    """Create engine build bundle for cross-host compilation."""
    from qwen3tts_tools.common import REPO_ROOT
    from qwen3tts_tools.bundle import make_build_bundle

    variant = getattr(args, "variant", "") or "custom-1.7b"
    target_profile = Path(getattr(args, "target_profile", ""))
    out = Path(getattr(args, "out", "workspace/engine_build_bundle.tar.zst"))
    dtype = getattr(args, "dtype", "bf16")
    triton_io_float_dtype = getattr(args, "triton_io_float_dtype", "") or dtype
    backbone_precision = getattr(args, "backbone_precision", "") or ""
    cp_precision = getattr(args, "cp_precision", "") or ""
    code2wav_precision = getattr(args, "code2wav_precision", "") or ""
    max_batch_size = getattr(args, "max_batch_size", 64)
    max_input_len = getattr(args, "max_input_len", 128)
    max_seq_len = getattr(args, "max_seq_len", 512)
    device = getattr(args, "device", "auto")
    dry_run = getattr(args, "dry_run", False)

    if not target_profile.is_file():
        print(f"Error: Target profile not found: {target_profile}", file=sys.stderr)
        print("Run 'qwen3tts probe' on the target machine first.", file=sys.stderr)
        return 1

    exported_dir = REPO_ROOT / "workspace" / "exported"
    if not exported_dir.is_dir():
        print(f"Error: Exported models not found: {exported_dir}", file=sys.stderr)
        print("Run 'qwen3tts setup' first.", file=sys.stderr)
        return 1

    # Resolve build profile from target profile
    if max_batch_size == 0 or max_seq_len == 0:
        from qwen3tts_tools.bundle import target_profile_memory_mb
        from qwen3tts_tools.trtexec import suggest_build_profile
        mem_mb = target_profile_memory_mb(target_profile)
        s_batch, s_input, s_seq = suggest_build_profile(mem_mb)
        if max_batch_size == 0:
            max_batch_size = s_batch
        if max_input_len == 0:
            max_input_len = s_input
        if max_seq_len == 0:
            max_seq_len = s_seq

    variants = [v.strip() for v in variant.split(",") if v.strip()]

    print(f"Creating engine build bundle:")
    print(f"  variants: {', '.join(variants)}")
    print(f"  target_profile: {target_profile}")
    print(f"  dtype: {dtype}")
    print(f"  max_batch_size: {max_batch_size}")
    print(f"  max_input_len: {max_input_len}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  out: {out}")

    if dry_run:
        print("[DRY RUN] Would create build bundle.")
        return 0

    try:
        make_build_bundle(
            repo_root=REPO_ROOT,
            exported_dir=exported_dir,
            out=out if out.is_absolute() else REPO_ROOT / out,
            target_profile=target_profile,
            variants=variants,
            engine_dtype=dtype,
            triton_io_float_dtype=triton_io_float_dtype,
            backbone_precision=backbone_precision,
            cp_precision=cp_precision,
            code2wav_precision=code2wav_precision,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            build_gpu_device=device,
        )
        print(f"Build bundle created: {out}")
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
#  import-artifact
# ---------------------------------------------------------------------------

def _build_import_artifact(args: object) -> int:
    """Import compiled engines from artifact bundle."""
    from qwen3tts_tools.common import REPO_ROOT
    from qwen3tts_tools.bundle import extract_artifact_bundle

    artifact = Path(getattr(args, "artifact", ""))
    allow_mismatch = getattr(args, "allow_fingerprint_mismatch", False)

    if not artifact.is_file():
        print(f"Error: Artifact bundle not found: {artifact}", file=sys.stderr)
        return 1

    exported_dir = REPO_ROOT / "workspace" / "exported"

    print(f"Importing engine artifact bundle: {artifact}")
    try:
        extract_artifact_bundle(
            repo_root=REPO_ROOT,
            exported_dir=exported_dir,
            artifact=artifact,
            strict=not allow_mismatch,
        )
        print("Artifact imported successfully.")
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        if not allow_mismatch:
            print("Use --allow-fingerprint-mismatch to bypass.", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
#  remote-build
# ---------------------------------------------------------------------------

def _build_remote_build(args: object) -> int:
    """SSH-driven remote engine compilation."""
    from qwen3tts_tools.common import REPO_ROOT
    from qwen3tts_tools.bundle import (
        extract_artifact_bundle,
        make_build_bundle,
    )

    variant = getattr(args, "variant", "") or "custom-1.7b"
    target_profile = Path(getattr(args, "target_profile", ""))
    remote_host = getattr(args, "remote_host", "")
    remote_workdir = getattr(args, "remote_workdir", "/tmp/qwen3-tts-engine-build")
    dtype = getattr(args, "dtype", "bf16")
    triton_io_float_dtype = getattr(args, "triton_io_float_dtype", "") or dtype
    backbone_precision = getattr(args, "backbone_precision", "") or ""
    cp_precision = getattr(args, "cp_precision", "") or ""
    code2wav_precision = getattr(args, "code2wav_precision", "") or ""
    max_batch_size = getattr(args, "max_batch_size", 64)
    max_input_len = getattr(args, "max_input_len", 128)
    max_seq_len = getattr(args, "max_seq_len", 512)
    device = getattr(args, "device", "auto")
    dry_run = getattr(args, "dry_run", False)

    if not remote_host:
        print("Error: --remote-host is required for remote-build.", file=sys.stderr)
        return 1

    if not target_profile.is_file():
        print(f"Error: Target profile not found: {target_profile}", file=sys.stderr)
        return 1

    exported_dir = REPO_ROOT / "workspace" / "exported"
    if not exported_dir.is_dir():
        print(f"Error: Exported models not found: {exported_dir}", file=sys.stderr)
        return 1

    # Resolve build profile
    if max_batch_size == 0 or max_seq_len == 0:
        from qwen3tts_tools.bundle import target_profile_memory_mb
        from qwen3tts_tools.trtexec import suggest_build_profile
        mem_mb = target_profile_memory_mb(target_profile)
        s_batch, s_input, s_seq = suggest_build_profile(mem_mb)
        if max_batch_size == 0:
            max_batch_size = s_batch
        if max_input_len == 0:
            max_input_len = s_input
        if max_seq_len == 0:
            max_seq_len = s_seq

    variants = [v.strip() for v in variant.split(",") if v.strip()]

    # Step 1: Create build bundle
    tmp_bundle = REPO_ROOT / "workspace" / "engine_build_bundle.remote.tar.zst"
    artifact_local = REPO_ROOT / "workspace" / "engine_artifact_bundle.remote.tar.zst"

    print("Step 1/6: Creating build bundle...")
    try:
        make_build_bundle(
            repo_root=REPO_ROOT,
            exported_dir=exported_dir,
            out=tmp_bundle,
            target_profile=target_profile,
            variants=variants,
            engine_dtype=dtype,
            triton_io_float_dtype=triton_io_float_dtype,
            backbone_precision=backbone_precision,
            cp_precision=cp_precision,
            code2wav_precision=code2wav_precision,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            build_gpu_device=device,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error creating bundle: {e}", file=sys.stderr)
        return 1

    if dry_run:
        print("[DRY RUN] Would upload/run/download via SSH host:", remote_host)
        return 0

    # Step 2: Clean remote directory
    print(f"Step 2/6: Preparing remote directory on {remote_host}...")
    result = subprocess.run(
        ["ssh", remote_host, f"rm -rf '{remote_workdir}' && mkdir -p '{remote_workdir}'"],
    )
    if result.returncode != 0:
        print(f"Error: Failed to prepare remote directory.", file=sys.stderr)
        return 1

    # Step 3: Upload bundle
    print(f"Step 3/6: Uploading bundle to {remote_host}...")
    result = subprocess.run(
        ["scp", str(tmp_bundle), f"{remote_host}:{remote_workdir}/engine_build_bundle.tar.zst"],
    )
    if result.returncode != 0:
        print(f"Error: Failed to upload bundle.", file=sys.stderr)
        return 1

    # Step 4: Extract and build on remote
    print(f"Step 4/6: Running build on {remote_host}...")
    result = subprocess.run(
        [
            "ssh", remote_host,
            f"cd '{remote_workdir}' && "
            "(tar --zstd -xf engine_build_bundle.tar.zst 2>/dev/null || tar -z -xf engine_build_bundle.tar.zst) && "
            "bash build_on_target.sh",
        ],
    )
    if result.returncode != 0:
        print(f"Error: Remote build failed.", file=sys.stderr)
        return 1

    # Step 5: Download artifact
    print(f"Step 5/6: Downloading artifact from {remote_host}...")
    result = subprocess.run(
        ["scp", f"{remote_host}:{remote_workdir}/engine_artifact_bundle.tar.zst", str(artifact_local)],
    )
    if result.returncode != 0:
        print(f"Error: Failed to download artifact.", file=sys.stderr)
        return 1

    # Step 6: Import artifact
    print("Step 6/6: Importing artifact...")
    try:
        extract_artifact_bundle(
            repo_root=REPO_ROOT,
            exported_dir=exported_dir,
            artifact=artifact_local,
            strict=True,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Remote build complete. Engines imported successfully.")
    return 0
