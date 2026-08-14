"""Artifact persistence and aggregate reporting for sweep trials."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qwen3tts_protocol.audio import pcm16_from_float_audio, save_wav

from .metrics import wilson_interval
from .models import ChunkPattern, SynthesisResult, TextPacket, TrialStatus


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def persist_trial(
    run_dir: Path,
    *,
    trial_index: int,
    session_id: str,
    pattern: ChunkPattern,
    packets: Sequence[TextPacket],
    result: SynthesisResult,
) -> dict[str, Any]:
    """Write one WAV plus JSON sidecar and return its serializable record."""

    trials_dir = run_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trial_{trial_index:04d}"
    wav_path = trials_dir / f"{stem}.wav"
    json_path = trials_dir / f"{stem}.json"
    record: dict[str, Any] = {
        "trial_index": trial_index,
        "session_id": session_id,
        "pattern": pattern.value,
        "packets": [
            {
                "text": packet.text,
                "text_repr": repr(packet.text),
                "delay_after_ms": packet.delay_after_s * 1000.0,
            }
            for packet in packets
        ],
        "status": result.status.value,
        "sample_rate": result.sample_rate,
        "duration_s": result.duration_s,
        "ttft_ms": result.ttft_ms,
        "total_ms": result.total_ms,
        "chunks": result.chunks,
        "terminal_event": result.terminal_event,
        "eos_reason": result.eos_reason,
        "events": result.events,
        "error": result.error,
        "artifacts": {"json": str(json_path.relative_to(run_dir))},
    }
    if result.samples.size:
        save_wav(result.samples, wav_path, sample_rate=result.sample_rate)
        pcm16 = pcm16_from_float_audio(result.samples)
        record["artifacts"]["wav"] = str(wav_path.relative_to(run_dir))
        record["wav_sha256"] = sha256_file(wav_path)
        record["pcm_s16le_sha256"] = hashlib.sha256(pcm16.tobytes()).hexdigest()
        record["pcm_f32le_sha256"] = hashlib.sha256(
            np.asarray(result.samples, dtype="<f4").tobytes()
        ).hexdigest()
    json_write(json_path, record)
    return record


def rewrite_trial_sidecar(run_dir: Path, record: Mapping[str, Any]) -> None:
    raw_path = record.get("artifacts", {}).get("json")
    if not isinstance(raw_path, str):
        raise TypeError("trial record is missing its JSON artifact")
    json_write(run_dir / raw_path, record)


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    confidence: float,
) -> dict[str, Any]:
    tts_ok = sum(record.get("status") == TrialStatus.OK.value for record in records)
    classified = [
        record
        for record in records
        if isinstance(record.get("screening", {}).get("is_suspect"), bool)
    ]
    suspects = sum(
        record.get("screening", {}).get("is_suspect") is True for record in classified
    )
    durations = [
        float(record["duration_s"])
        for record in records
        if record.get("status") == TrialStatus.OK.value
    ]
    return {
        "trials": len(records),
        "tts_ok": tts_ok,
        "tts_failed_or_no_audio": len(records) - tts_ok,
        "screened": len(classified),
        "screening_unknown": len(records) - len(classified),
        "asr_supported_suspects": suspects,
        "asr_supported_suspect_rate_wilson": wilson_interval(
            suspects, len(classified), confidence=confidence
        ),
        "duration_s": (
            {
                "min": min(durations),
                "median": statistics.median(durations),
                "max": max(durations),
            }
            if durations
            else None
        ),
        "interpretation": (
            "Automatic ASR/duration screen only; blind human review is required "
            "for a hallucination-rate claim."
        ),
    }
