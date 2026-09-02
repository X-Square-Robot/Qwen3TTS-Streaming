"""Content-addressed evidence inventories for long-form reports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .preflight import sha256_file


def artifact_identity(output_root: Path, path: Path) -> dict[str, Any]:
    """Describe one report input by its output-relative path and content hash."""

    root = output_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"report input escapes output root: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"report input is not a regular file: {path}")
    return {
        "path": str(relative),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def runtime_evidence_inventory(output_root: Path) -> dict[str, Any]:
    """Fingerprint every entry under ``output_root/runtime`` without following links.

    Regular files are content-addressed.  Symlinks are recorded by hashing the
    literal link target rather than the target contents, so a link cannot make
    report generation silently traverse outside the experiment output.
    """

    runtime_root = output_root / "runtime"
    if not runtime_root.exists():
        return {
            "schema_version": 1,
            "root": "runtime",
            "exists": False,
            "audit_ready": False,
            "file_count": 0,
            "symlink_count": 0,
            "total_file_bytes": 0,
            "tree_sha256": hashlib.sha256(b"").hexdigest(),
            "entries": [],
            "unsupported_entries": [],
        }
    if runtime_root.is_symlink():
        raise ValueError(f"runtime evidence root must not be a symlink: {runtime_root}")
    if not runtime_root.is_dir():
        raise NotADirectoryError(
            f"runtime evidence root is not a directory: {runtime_root}"
        )

    entries: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for path in sorted(runtime_root.rglob("*"), key=lambda item: str(item)):
        relative = str(path.relative_to(output_root))
        if path.is_symlink():
            target = os.readlink(path)
            entries.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "link_target": target,
                    "bytes": len(os.fsencode(target)),
                    "sha256": hashlib.sha256(os.fsencode(target)).hexdigest(),
                    "hash_scope": "literal_symlink_target",
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "hash_scope": "file_contents",
                }
            )
        elif not path.is_dir():
            unsupported.append(relative)

    tree_digest = hashlib.sha256()
    for entry in entries:
        tree_digest.update(
            json.dumps(
                entry,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        tree_digest.update(b"\n")
    file_entries = [entry for entry in entries if entry["kind"] == "file"]
    return {
        "schema_version": 1,
        "root": "runtime",
        "exists": True,
        "audit_ready": bool(file_entries) and not unsupported,
        "file_count": len(file_entries),
        "symlink_count": len(entries) - len(file_entries),
        "total_file_bytes": sum(int(entry["bytes"]) for entry in file_entries),
        "tree_sha256": tree_digest.hexdigest(),
        "entries": entries,
        "unsupported_entries": unsupported,
    }


__all__ = ["artifact_identity", "runtime_evidence_inventory"]
