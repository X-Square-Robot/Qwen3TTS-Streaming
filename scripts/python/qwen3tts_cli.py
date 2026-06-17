"""Qwen3-TTS CLI — thin Python wrapper around the Bash lifecycle scripts.

Usage::

    qwen3tts all -m custom-1.7b
    qwen3tts setup -m custom-1.7b
    qwen3tts build -m custom-1.7b --max-batch-size 64
    qwen3tts package -m custom-1.7b
    qwen3tts deploy -m custom-1.7b --gateway standalone
    qwen3tts status
    qwen3tts probe

Each subcommand forwards its arguments to the corresponding Bash script
under ``scripts/bash/``.  This is a thin convenience layer — all real
logic remains in the Bash scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASH_DIR = _REPO_ROOT / "scripts" / "bash"


def _run_bash(script: str, extra_args: list[str]) -> int:
    script_path = _BASH_DIR / script
    if not script_path.exists():
        print(f"Error: {script_path} not found", file=sys.stderr)
        return 1
    return subprocess.call(["bash", str(script_path)] + extra_args)


def _passthrough(args: argparse.Namespace) -> int:
    script_map = {
        "all": "autorun.sh",
        "setup": "autorun.sh",
        "build": "autorun.sh",
        "package": "autorun.sh",
        "deploy": "autorun.sh",
    }
    script = script_map[args.command]
    return _run_bash(script, [args.command] + args.rest)


def _probe(_args: argparse.Namespace) -> int:
    return _run_bash("probe_target.sh", [])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="qwen3tts",
        description="Qwen3-TTS Triton lifecycle CLI. Delegates to scripts/bash/.",
    )
    sub = parser.add_subparsers(dest="command")

    for name, help_text in [
        ("all", "Full pipeline: setup → build → package → deploy"),
        ("setup", "Phase A: install environment, download/export models"),
        ("build", "Phase B: compile TensorRT engines"),
        ("package", "Phase C1: assemble deployment artifacts"),
        ("deploy", "Phase C2: start service on current machine"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("rest", nargs=argparse.REMAINDER,
                        help="Arguments forwarded to the underlying script")

    sub.add_parser("status", help="Show current pipeline status")
    sub.add_parser("probe", help="Probe target GPU/driver profile")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "status":
        sys.exit(_run_bash("autorun.sh", ["status"]))
    if args.command == "probe":
        sys.exit(_probe())

    sys.exit(_passthrough(args))
