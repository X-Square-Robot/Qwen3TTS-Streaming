#!/usr/bin/env python3
"""Write immutable package provenance for a model package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.runtime.package_info import PackageInfo, validate_packaged_on, validate_packager


def write_package_info(output: Path, *, packager: str, packaged_on: str) -> PackageInfo:
    """Validate and write canonical read-only ``PACKAGE_INFO.json``."""

    info = PackageInfo(
        packager=validate_packager(packager),
        packaged_on=validate_packaged_on(packaged_on),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.chmod(0o644)
    output.write_text(
        json.dumps(info.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output.chmod(0o444)
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packager", required=True)
    parser.add_argument("--packaged-on", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_package_info(
        args.output,
        packager=args.packager,
        packaged_on=args.packaged_on,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
