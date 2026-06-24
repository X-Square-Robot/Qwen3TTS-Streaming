"""Setup subcommand — Phase A: install environment, download/export models."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_setup(args: argparse.Namespace) -> int:
    """Run Phase A: environment setup, model download, and ONNX export."""
    from qwen3tts_tools.common import REPO_ROOT

    variant = getattr(args, "variant", "") or "custom-1.7b"
    dry_run = getattr(args, "dry_run", False)
    skip_deps = getattr(args, "skip_deps", False)
    skip_download = getattr(args, "skip_download", False)
    skip_export = getattr(args, "skip_export", False)

    # Step 1: Initialize git submodule
    if not skip_deps:
        third_party = REPO_ROOT / "third_party" / "Qwen3-TTS"
        if not third_party.is_dir():
            print("[1/6] Initializing git submodule...")
            if not dry_run:
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--depth", "1", "third_party/Qwen3-TTS"],
                    cwd=str(REPO_ROOT), check=True,
                )
        else:
            print("[1/6] Git submodule already initialized.")

    # Step 2: Install Qwen3-TTS upstream package (editable)
    # This pulls in all official deps: librosa, sox, torchaudio, soundfile, etc.
    if not skip_deps:
        print("[2/6] Installing Qwen3-TTS upstream package...")
        if not dry_run:
            qwen_tts_dir = REPO_ROOT / "third_party" / "Qwen3-TTS"
            if qwen_tts_dir.is_dir() and (qwen_tts_dir / "pyproject.toml").is_file():
                # Check if already installed
                already_installed = False
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", "import qwen_tts"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0:
                        already_installed = True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

                if already_installed:
                    print("  qwen-tts already installed, skipping.")
                else:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-e", str(qwen_tts_dir)],
                        check=False,
                    )
                    if result.returncode != 0:
                        print("Warning: Qwen3-TTS package install failed. Export may not work.", file=sys.stderr)
            else:
                print("Warning: Qwen3-TTS submodule not found. Run step 1 first.", file=sys.stderr)
    else:
        print("[2/6] Skipping Qwen3-TTS package install (--skip-deps).")

    # Step 3: Install project export dependencies
    if not skip_deps:
        print("[3/6] Installing export dependencies...")
        if not dry_run:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", ".[export]"],
                cwd=str(REPO_ROOT), check=False,
            )
            # Also install ONNX export extras (onnxscript, onnxsim)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade",
                 "onnx", "onnxscript", "onnxsim"],
                check=False,
            )
    else:
        print("[3/6] Skipping dependency installation (--skip-deps).")

    # Step 4: Download models
    if not skip_download:
        print("[4/6] Downloading models...")
        if not dry_run:
            result = subprocess.call(
                [sys.executable, "-m", "qwen3tts_cli.cmd_download"],
            )
            # Fallback to Bash if Python download not available
            if result != 0:
                bash_script = REPO_ROOT / "scripts" / "bash" / "download_models.sh"
                if bash_script.is_file():
                    result = subprocess.call(["bash", str(bash_script), "--variant", variant])
                    if result != 0:
                        print("Warning: Model download failed. You may need to download manually.", file=sys.stderr)
    else:
        print("[4/6] Skipping model download (--skip-download).")

    # Step 5: Export models to ONNX
    if not skip_export:
        print("[5/6] Exporting models to ONNX...")
        export_script = REPO_ROOT / "scripts" / "export" / "export_all.py"
        export_device = getattr(args, "export_device", "") or getattr(args, "device", "") or ""

        if export_script.is_file():
            if not dry_run:
                cmd = [sys.executable, str(export_script), "--variant", variant]
                if export_device and export_device != "auto":
                    cmd += ["--device", export_device]
                result = subprocess.call(
                    cmd,
                    cwd=str(REPO_ROOT),
                )
                if result != 0:
                    print("Error: ONNX export failed.", file=sys.stderr)
                    return 1
        else:
            print(f"Warning: Export script not found: {export_script}", file=sys.stderr)
            # Fallback to Bash
            bash_script = REPO_ROOT / "scripts" / "bash" / "export_models.sh"
            if bash_script.is_file():
                result = subprocess.call(["bash", str(bash_script), "--variant", variant])
                if result != 0:
                    return 1
    else:
        print("[5/6] Skipping ONNX export (--skip-export).")

    # Step 6: Create symlink for engine access
    if not skip_deps:
        print("[6/6] Setting up project links...")
        if not dry_run:
            # Ensure workspace dir exists
            workspace = REPO_ROOT / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
    else:
        print("[6/6] Skipping project setup (--skip-deps).")

    print("Phase A (setup) complete.")
    return 0
