"""NGC compatibility matrix — Python replacement for ``scripts/bash/ngc_matrix.conf`` parsing.

Parses the NGC Triton container compatibility matrix and provides functions
to resolve the best NGC tag for a given NVIDIA driver version.

Usage from CLI::

    python -m qwen3tts_tools.ngc_matrix
    python -m qwen3tts_tools.ngc_matrix --driver 570.86
    python -m qwen3tts_tools.ngc_matrix --list

Usage from Python::

    from qwen3tts_tools.ngc_matrix import load_ngc_matrix, resolve_ngc_tag
    entries = load_ngc_matrix()
    tag = resolve_ngc_tag("570.86")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# NGC image constants (mirrored from docker.sh)
NGC_TRITON_BASE = "nvcr.io/nvidia/tritonserver"
NGC_PY3_SUFFIX = "-py3"

# Minimum driver for Qwen3-TTS (TensorRT 10+ / CUDA 12.4+)
QWEN3_MIN_DRIVER = "550.54"

# Default matrix path relative to repo root
_DEFAULT_MATRIX_PATH = Path(__file__).resolve().parents[3] / "scripts" / "bash" / "ngc_matrix.conf"

# Built-in fallback when the conf file is missing or empty
_FALLBACK_ENTRIES: list[dict[str, str]] = [
    {"tag": "25.11", "driver_version": "590.44", "cuda_versions": "13.1"},
    {"tag": "25.05", "driver_version": "575.51", "cuda_versions": "12.9"},
    {"tag": "25.03", "driver_version": "570.124", "cuda_versions": "12.8"},
    {"tag": "24.07", "driver_version": "555.42", "cuda_versions": "12.5"},
]


@dataclass
class NgcEntry:
    """A single row in the NGC compatibility matrix.

    Attributes:
        tag: NGC container tag (e.g. ``"25.03"``).
        driver_version: Minimum NVIDIA driver version (e.g. ``"570.124"``).
        cuda_versions: CUDA toolkit version(s) in the container (e.g. ``"12.8"``).
        tensorrt_version: TensorRT version (e.g. ``"10.9.0.34"``), or empty.
        python_version: Python version in the container (e.g. ``"3.12"``), or empty.
        size_gb: Approximate image size in GB, or ``"-"`` if unknown.
    """

    tag: str
    driver_version: str
    cuda_versions: str
    tensorrt_version: str = ""
    python_version: str = ""
    size_gb: str = "-"


def _parse_driver_version(version: str) -> tuple[int, int]:
    """Parse a driver version string into ``(major, minor)`` for comparison.

    Ignores the patch component (e.g. ``570.86.10`` → ``(570, 86)``).
    Missing minor defaults to 0.
    """
    parts = version.strip().split(".")
    major = int(parts[0]) if parts else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    return major, minor


def _driver_ge(installed: str, required: str) -> bool:
    """Return ``True`` if *installed* driver version >= *required*.

    Compares major.minor numerically; patch is ignored.
    """
    try:
        inst = _parse_driver_version(installed)
        req = _parse_driver_version(required)
    except (ValueError, IndexError):
        return False
    return inst >= req


def load_ngc_matrix(path: Path | str | None = None) -> list[NgcEntry]:
    """Parse the NGC compatibility matrix conf file.

    The file format is whitespace-separated columns::

        ngc_tag  min_driver  tensorrt_version  cuda_version  python_version  size_gb

    Comment lines (starting with ``#``) and blank lines are skipped.
    The matrix is expected to be sorted newest-first.

    Args:
        path: Path to the conf file. Defaults to
              ``scripts/bash/ngc_matrix.conf`` relative to the repo root.

    Returns:
        List of :class:`NgcEntry` objects, newest-first.
    """
    if path is None:
        path = _DEFAULT_MATRIX_PATH
    path = Path(path)

    entries: list[NgcEntry] = []

    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    continue
                entries.append(NgcEntry(
                    tag=parts[0],
                    driver_version=parts[1],
                    cuda_versions=parts[3] if len(parts) > 3 else parts[2],
                    tensorrt_version=parts[2] if len(parts) > 3 else "",
                    python_version=parts[4] if len(parts) > 4 else "",
                    size_gb=parts[5] if len(parts) > 5 else "-",
                ))

    # Fallback to built-in defaults if the file is missing or empty
    if not entries:
        for row in _FALLBACK_ENTRIES:
            entries.append(NgcEntry(
                tag=row["tag"],
                driver_version=row["driver_version"],
                cuda_versions=row["cuda_versions"],
            ))

    return entries


def resolve_ngc_tag(
    driver_version: str,
    matrix: Sequence[NgcEntry] | None = None,
) -> str | None:
    """Find the best compatible NGC tag for a given NVIDIA driver version.

    Scans the matrix (newest-first) and returns the first tag whose
    minimum driver version is satisfied by *driver_version*.

    Args:
        driver_version: Installed NVIDIA driver version (e.g. ``"570.86.10"``).
        matrix: Pre-loaded matrix entries. Loaded from the conf file if None.

    Returns:
        The compatible NGC tag (e.g. ``"25.03"``), or ``None`` if no
        compatible entry is found.
    """
    if matrix is None:
        matrix = load_ngc_matrix()

    for entry in matrix:
        if _driver_ge(driver_version, entry.driver_version):
            return entry.tag

    return None


def resolve_ngc_image(
    ngc_tag: str,
    matrix: Sequence[NgcEntry] | None = None,
) -> str | None:
    """Return the full nvcr.io image URI for a given NGC tag.

    The image URI follows the pattern::

        nvcr.io/nvidia/tritonserver:<tag>-py3

    Args:
        ngc_tag: NGC container tag (e.g. ``"25.03"``).
        matrix: Pre-loaded matrix entries. Loaded from the conf file if None.

    Returns:
        Full image URI, or ``None`` if the tag is not found in the matrix.
    """
    if matrix is None:
        matrix = load_ngc_matrix()

    for entry in matrix:
        if entry.tag == ngc_tag:
            return f"{NGC_TRITON_BASE}:{ngc_tag}{NGC_PY3_SUFFIX}"

    return None


def resolve_ngc_entry(
    driver_version: str,
    matrix: Sequence[NgcEntry] | None = None,
) -> NgcEntry | None:
    """Find the full matrix entry for a given driver version.

    Like :func:`resolve_ngc_tag` but returns the complete :class:`NgcEntry`
    with all fields (TensorRT version, CUDA version, Python version, etc.).

    Args:
        driver_version: Installed NVIDIA driver version.
        matrix: Pre-loaded matrix entries. Loaded from the conf file if None.

    Returns:
        The best compatible :class:`NgcEntry`, or ``None`` if not found.
    """
    if matrix is None:
        matrix = load_ngc_matrix()

    for entry in matrix:
        if _driver_ge(driver_version, entry.driver_version):
            return entry

    return None


def resolve_ngc_entry_by_tag(
    ngc_tag: str,
    matrix: Sequence[NgcEntry] | None = None,
) -> NgcEntry | None:
    """Look up a matrix entry by its NGC tag.

    Args:
        ngc_tag: NGC container tag (e.g. ``"25.03"``).
        matrix: Pre-loaded matrix entries. Loaded from the conf file if None.

    Returns:
        The matching :class:`NgcEntry`, or ``None`` if not found.
    """
    if matrix is None:
        matrix = load_ngc_matrix()

    for entry in matrix:
        if entry.tag == ngc_tag:
            return entry

    return None


def format_matrix_table(
    matrix: Sequence[NgcEntry],
    driver_version: str | None = None,
) -> str:
    """Format the matrix as a human-readable table.

    Args:
        matrix: Matrix entries to display.
        driver_version: If provided, mark compatibility status per row.

    Returns:
        Formatted table string.
    """
    default_tag = resolve_ngc_tag(driver_version, matrix) if driver_version else None

    header = (
        f"{'TAG':<7} {'MIN_DRIVER':<10} {'CUDA':<8} {'PYTHON':<7} "
        f"{'TENSORRT':<14} {'SIZE_GB':<8} {'STATUS':<12} IMAGE"
    )
    lines = [header]

    for entry in matrix:
        status = "unknown"
        if driver_version:
            if _driver_ge(driver_version, entry.driver_version):
                status = "ok"
            else:
                status = "needs-driver"
            if entry.tag == default_tag:
                status = "default"

        image = f"{NGC_TRITON_BASE}:{entry.tag}{NGC_PY3_SUFFIX}"
        py = entry.python_version or "3.12"
        trt = entry.tensorrt_version or "-"
        line = (
            f"{entry.tag:<7} {entry.driver_version:<10} {entry.cuda_versions:<8} "
            f"{py:<7} {trt:<14} {entry.size_gb:<8} {status:<12} {image}"
        )
        lines.append(line)

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NGC compatibility matrix utilities",
    )
    parser.add_argument(
        "--driver", type=str, default=None,
        help="NVIDIA driver version to check compatibility for",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all matrix entries with compatibility status",
    )
    parser.add_argument(
        "--matrix-path", type=Path, default=None,
        help="Path to ngc_matrix.conf (default: auto-detect)",
    )
    args = parser.parse_args()

    matrix = load_ngc_matrix(args.matrix_path)

    if args.list:
        print(format_matrix_table(matrix, args.driver))
        return

    if args.driver:
        entry = resolve_ngc_entry(args.driver, matrix)
        if entry is None:
            print(f"No compatible NGC container for driver {args.driver}")
            print(f"Minimum driver for Qwen3-TTS: >= {QWEN3_MIN_DRIVER}")
            raise SystemExit(1)
        image = resolve_ngc_image(entry.tag, matrix)
        print(f"Tag:  {entry.tag}")
        print(f"Image: {image}")
        print(f"CUDA: {entry.cuda_versions}")
        print(f"Python: {entry.python_version or '3.12'}")
        print(f"TensorRT: {entry.tensorrt_version or '-'}")
        return

    # Default: show the full matrix
    print(format_matrix_table(matrix))


if __name__ == "__main__":
    main()
