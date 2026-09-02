"""Stable on-disk evidence format for long-form synthesis arms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from qwen3tts_protocol.audio import pcm16_from_float_audio, save_wav


SCHEMA_VERSION = 2


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def pcm_hashes(samples: np.ndarray) -> dict[str, str]:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    return {
        "pcm_f32le_sha256": hashlib.sha256(
            audio.astype("<f4", copy=False).tobytes()
        ).hexdigest(),
        "pcm_s16le_sha256": hashlib.sha256(
            pcm16_from_float_audio(audio).astype("<i2", copy=False).tobytes()
        ).hexdigest(),
    }


def persist_collected_run(
    output_root: Path,
    collected: Any,
    *,
    seed: int,
    source_text: str,
    source_text_sha256: str,
    experiment_manifest_binding_sha256: str,
    collection_contract_sha256: str,
    arm_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a ``CollectedRun`` without coupling this module to its class."""

    arm = getattr(collected, "arm")
    arm_name = getattr(arm, "value", str(arm))
    session_id = str(getattr(collected, "session_id"))
    run_dir = output_root / "arms" / arm_name / f"seed_{int(seed):04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FileExistsError(f"run artifact directory is not empty: {run_dir}")

    samples = np.asarray(getattr(collected, "samples"), dtype=np.float32).reshape(-1)
    sample_rate = int(getattr(collected, "sample_rate"))
    wav_path = run_dir / "full.wav"
    pcm_path = run_dir / "full.pcm_f32le"
    if samples.size:
        save_wav(samples, wav_path, sample_rate=sample_rate)
        pcm_path.write_bytes(samples.astype("<f4", copy=False).tobytes())

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "arm": arm_name,
        "seed": int(seed),
        "session_id": session_id,
        "status": _jsonable(getattr(collected, "status")),
        "sample_rate": sample_rate,
        "sample_count": int(samples.size),
        "duration_s": (samples.size / sample_rate if sample_rate > 0 else 0.0),
        "source_text": source_text,
        "source_text_sha256": source_text_sha256,
        "experiment_manifest_binding_sha256": experiment_manifest_binding_sha256,
        "collection_contract_sha256": collection_contract_sha256,
        "events": _jsonable(getattr(collected, "events", [])),
        "audio_chunks": _jsonable(getattr(collected, "audio_chunks", [])),
        "error": getattr(collected, "error", None),
        "total_ms": getattr(collected, "total_ms", None),
        "ttft_ms": getattr(collected, "ttft_ms", None),
        "terminal_event": getattr(collected, "terminal_event", None),
        "eos_reason": getattr(collected, "eos_reason", None),
        "sampling_seed": getattr(collected, "sampling_seed", None),
        "arm_metadata": dict(arm_metadata or {}),
        "artifacts": {
            "wav": str(wav_path.relative_to(output_root)) if samples.size else None,
            "pcm_f32le": (
                str(pcm_path.relative_to(output_root)) if samples.size else None
            ),
            "run_json": str((run_dir / "run.json").relative_to(output_root)),
        },
    }
    if samples.size:
        record.update(pcm_hashes(samples))
    write_json(run_dir / "run.json", record)
    return record


def discover_run_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((output_root / "arms").glob("*/seed_*/run.json")):
        record = read_json(path)
        record["_record_path"] = str(path.relative_to(output_root))
        records.append(record)
    return records
