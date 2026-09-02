"""Atomic, content-addressed ASR sidecars for resumable long-form scoring."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .preflight import sha256_file


SCHEMA_VERSION = 1


def sidecar_path(wav_path: Path) -> Path:
    """Return the ASR evidence path owned by one WAV artifact."""

    return wav_path.with_suffix(f"{wav_path.suffix}.asr.json")


def request_identity(
    wav_path: Path,
    *,
    uri: str,
    language: str,
    chunk_ms: int,
) -> dict[str, Any]:
    """Fingerprint every input that can change a transcription result."""

    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    resolved = wav_path.resolve(strict=True)
    return {
        "wav": {
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        },
        "uri": str(uri),
        "language": str(language),
        "chunk_ms": int(chunk_ms),
        "protocol": {
            "delivery_mode": "offline",
            "pacing": "none",
            "partial_mode": "off",
            "hotwords": [],
            "vad": "fsmn",
        },
    }


def _reusable_result(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or value.get("status") != "ok":
        return None
    if not isinstance(value.get("transcript"), str):
        return None
    if not isinstance(value.get("segments"), list):
        return None
    if not isinstance(value.get("stream_done"), Mapping):
        return None
    return dict(value)


def load_reusable_sidecar(
    path: Path,
    *,
    expected_request: Mapping[str, Any],
    required_provenance_source: str | None = None,
) -> dict[str, Any] | None:
    """Load a matching successful result; corrupt/error sidecars are misses.

    ``required_provenance_source`` lets protocol-sensitive callers reject a
    result copied from another WAV even when the bytes happen to be identical.
    This is required by the long-form experiment's one-connection-per-WAV
    contract.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    if payload.get("request") != dict(expected_request):
        return None
    if required_provenance_source is not None:
        provenance = payload.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("source") != required_provenance_source
        ):
            return None
    return _reusable_result(payload.get("result"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_sidecar(
    path: Path,
    *,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Atomically retain one attempt; only successful payloads are reusable."""

    _atomic_write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "request": dict(request),
            "result": dict(result),
            "provenance": dict(provenance or {}),
        },
    )


__all__ = [
    "SCHEMA_VERSION",
    "load_reusable_sidecar",
    "request_identity",
    "sidecar_path",
    "write_sidecar",
]
