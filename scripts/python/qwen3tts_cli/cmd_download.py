"""Download subcommand — download model weights."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_download(args: argparse.Namespace) -> int:
    """Download model weights from ModelScope or HuggingFace."""
    from qwen3tts_tools.common import REPO_ROOT

    variant = getattr(args, "variant", "")
    source = getattr(args, "source", "auto")

    # Ensure git-lfs
    try:
        subprocess.run(["git", "lfs", "version"], capture_output=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Installing git-lfs...", file=sys.stderr)
        try:
            subprocess.run(["apt-get", "install", "-y", "git-lfs"], check=True, timeout=60)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("Error: Cannot install git-lfs. Please install it manually.", file=sys.stderr)
            return 1

    third_party = REPO_ROOT / "third_party" / "Qwen3-TTS"
    if not third_party.is_dir():
        print("Initializing git submodule: third_party/Qwen3-TTS...", file=sys.stderr)
        subprocess.run(
            ["git", "submodule", "update", "--init", "--depth", "1", "third_party/Qwen3-TTS"],
            cwd=str(REPO_ROOT), check=True,
        )

    # Use modelscope or huggingface_hub for download
    download_script = third_party / "scripts" / "download_model.py"
    if download_script.is_file():
        cmd = [sys.executable, str(download_script)]
        if variant:
            cmd += ["--variant", variant]
        if source != "auto":
            cmd += ["--source", source]
        return subprocess.call(cmd)
    else:
        # Fallback: try modelscope CLI
        if shutil.which("modelscope"):
            return subprocess.call(["modelscope", "download", "--model", f"Qwen/Qwen3-TTS-{variant}"])
        else:
            print("Error: No download script found. Install modelscope or huggingface_hub.", file=sys.stderr)
            return 1
