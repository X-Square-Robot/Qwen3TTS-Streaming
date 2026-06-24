"""Deploy subcommand — Phase C: package, run, stop, assemble, pull."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_package(args: argparse.Namespace) -> int:
    """Phase C1: Assemble deployment artifacts."""
    from qwen3tts_tools.common import REPO_ROOT

    variant = getattr(args, "variant", "") or "custom-1.7b"
    gateway = getattr(args, "gateway", "standalone")
    engine_mode = getattr(args, "engine_mode", "trt")
    dry_run = getattr(args, "dry_run", False)

    try:
        from qwen3tts_tools.triton import TritonManager
    except ImportError:
        print("Error: qwen3tts_tools not available. Install with: pip install -e .", file=sys.stderr)
        return 1

    if gateway in ("triton", "standalone"):
        mgr = TritonManager(repo_root=REPO_ROOT)
        print(f"Assembling model repository for variant: {variant}, engine_mode: {engine_mode}")
        if dry_run:
            print("[DRY RUN] Would assemble model repository.")
            return 0
        result = mgr.assemble_model_repo(
            exported_dir=REPO_ROOT / "workspace" / "exported",
            variant=variant,
            model_repo_dir=REPO_ROOT / "workspace" / "model_repository",
            engine_mode=engine_mode,
            model_version=getattr(args, "model_version", 1),
        )
        if result:
            print(f"Model repository assembled: {REPO_ROOT / 'workspace' / 'model_repository'}")
            return 0
        else:
            print("Error: Model repository assembly failed.", file=sys.stderr)
            return 1
    else:
        # engine-docker
        try:
            from qwen3tts_tools.compose import ComposeManager
            mgr = ComposeManager(repo_root=REPO_ROOT)
            # For engine-docker, assemble first then compose
            mgr.prepare(
                variant=variant,
                engine_mode=engine_mode,
                model_version=getattr(args, "model_version", 1),
            )
            print(f"Model repository assembled for engine-docker.")
            return 0
        except (ImportError, NotImplementedError):
            print("Error: engine-docker gateway requires qwen3tts_tools.", file=sys.stderr)
            return 1


def run_deploy(args: argparse.Namespace) -> int:
    """Phase C2: Start the TTS service."""
    from qwen3tts_tools.common import REPO_ROOT

    variant = getattr(args, "variant", "") or "custom-1.7b"
    gateway = getattr(args, "gateway", "standalone")
    dry_run = getattr(args, "dry_run", False)
    port = getattr(args, "port", 50051)
    ws_port = getattr(args, "ws_port", 50052)
    device = getattr(args, "device", "auto")
    max_batch = getattr(args, "max_batch", 0)
    max_sessions = getattr(args, "max_sessions", 128)
    foreground = getattr(args, "foreground", False)
    engine_mode = getattr(args, "engine_mode", "trt")
    model_version = getattr(args, "model_version", 1)

    if gateway == "standalone":
        try:
            from qwen3tts_tools.engine import EngineManager
        except ImportError:
            print("Error: qwen3tts_tools not available. Install with: pip install -e .", file=sys.stderr)
            return 1

        mgr = EngineManager()
        if dry_run:
            print(f"[DRY RUN] Would start standalone engine on port {port}")
            return 0
        print(f"Starting standalone engine on port {port}...")
        return mgr.start(
            variant=variant,
            port=port,
            ws_port=ws_port,
            device=device,
            max_batch=max_batch or 128,
            max_sessions=max_sessions,
            foreground=foreground,
        )

    elif gateway == "triton":
        try:
            from qwen3tts_tools.compose import ComposeManager
        except ImportError:
            print("Error: qwen3tts_tools not available. Install with: pip install -e .", file=sys.stderr)
            return 1

        mgr = ComposeManager(repo_root=REPO_ROOT)
        return mgr.up(
            gateway="triton",
            variant=variant,
            engine_mode=engine_mode,
            device=device,
            max_batch=max_batch,
            model_version=model_version,
        )

    else:
        # engine-docker
        try:
            from qwen3tts_tools.compose import ComposeManager
        except ImportError:
            print("Error: qwen3tts_tools not available. Install with: pip install -e .", file=sys.stderr)
            return 1

        mgr = ComposeManager(repo_root=REPO_ROOT)
        return mgr.up(
            gateway="engine",
            variant=variant,
            port=port,
            ws_port=ws_port,
            device=device,
            max_batch=max_batch,
            model_version=model_version,
        )


def run_stop(args: argparse.Namespace) -> int:
    """Stop the TTS service."""
    try:
        from qwen3tts_tools.engine import EngineManager
    except ImportError:
        print("Error: qwen3tts_tools not available.", file=sys.stderr)
        return 1

    mgr = EngineManager()
    mgr.stop()

    # Also stop Docker containers
    try:
        from qwen3tts_tools.compose import ComposeManager
        compose_mgr = ComposeManager()
        compose_mgr.down(gateway="all")
    except (ImportError, NotImplementedError):
        pass

    print("TTS service stopped.")
    return 0


def run_assemble(args: argparse.Namespace) -> int:
    """Assemble Triton model_repository."""
    from qwen3tts_tools.common import REPO_ROOT

    variant = getattr(args, "variant", "") or "custom-1.7b"
    engine_mode = getattr(args, "engine_mode", "trt")
    model_version = getattr(args, "model_version", 1)

    try:
        from qwen3tts_tools.triton import TritonManager
    except ImportError:
        print("Error: qwen3tts_tools not available. Install with: pip install -e .", file=sys.stderr)
        return 1

    mgr = TritonManager(repo_root=REPO_ROOT)
    result = mgr.assemble_model_repo(
        exported_dir=REPO_ROOT / "workspace" / "exported",
        variant=variant,
        model_repo_dir=REPO_ROOT / "workspace" / "model_repository",
        engine_mode=engine_mode,
        model_version=model_version,
    )
    return 0 if result else 1


def run_pull(args: argparse.Namespace) -> int:
    """Pull NGC Triton container image."""
    try:
        from qwen3tts_tools.docker import ensure_image, detect_driver_version
        from qwen3tts_tools.ngc_matrix import resolve_ngc_image
    except ImportError:
        print("Error: qwen3tts_tools not available. Install with: pip install -e .", file=sys.stderr)
        return 1

    driver_ver = detect_driver_version()
    if not driver_ver:
        print("Error: Cannot detect NVIDIA driver version.", file=sys.stderr)
        return 1

    image = resolve_ngc_image(driver_ver)
    print(f"Pulling NGC image: {image}")
    return ensure_image(image)
