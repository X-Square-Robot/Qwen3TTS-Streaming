#!/usr/bin/env python3
"""Write a validated, read-only version identity sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.runtime.model_version import validate_model_version


def write_version_sidecar(output: Path, *, version: str) -> str:
    """Write one canonical version line and set mode ``0444``."""

    validated = validate_model_version(version, source=str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.chmod(0o644)
    output.write_text(f"{validated}\n", encoding="utf-8")
    output.chmod(0o444)
    return validated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_version_sidecar(args.output, version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
