"""Ungated, fail-closed replay for long-form engine root-cause diagnostics.

This path deliberately does not depend on listening-review state.  It answers
the narrower engineering question: did each backend synthesize the exact same
frozen text groups, and what did its engine/ASR telemetry report?
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qwen3tts_protocol.audio import save_wav

from tools.validation.hallucination.asr import transcribe_wav

from .artifacts import pcm_hashes, read_json, write_json
from .collection import logical_session_id
from .matched import (
    GREEDY_GENERATION,
    MATCHED_GENERATION,
    MatchedEndpointArm,
    MatchedOfficialArm,
)
from .metrics import character_error_metrics
from .reference_groups import validate_reference_groups


def load_frozen_replay(
    text_file: Path,
    groups_file: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Read exact UTF-8 text and validate that frozen groups cover every byte."""

    text = text_file.read_bytes().decode("utf-8")
    payload = read_json(groups_file)
    groups = validate_reference_groups(list(payload.get("groups") or []), text)
    return text, groups


def _persist_case(
    output_dir: Path,
    run: Any,
    *,
    source_text: str,
    groups: Sequence[Mapping[str, Any]],
    replay_contract: Mapping[str, Any],
) -> dict[str, Any]:
    case_dir = output_dir / str(run.arm.value) / f"seed_{int(run.seed):05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    if any(case_dir.iterdir()):
        raise FileExistsError(f"diagnostic case directory is not empty: {case_dir}")

    samples = np.asarray(run.samples, dtype=np.float32).reshape(-1)
    wav_path = case_dir / "full.wav"
    if samples.size:
        save_wav(samples, wav_path, sample_rate=int(run.sample_rate))
    commits = [
        event
        for event in run.events
        if event.get("type") == "text_boundary_commit"
    ]
    record = {
        "schema_version": 1,
        "diagnostic_only": True,
        "arm": run.arm,
        "seed": int(run.seed),
        "session_id": run.session_id,
        "status": run.status,
        "error": run.error,
        "terminal_event": run.terminal_event,
        "eos_reason": run.eos_reason,
        "sample_rate": int(run.sample_rate),
        "sample_count": int(samples.size),
        "duration_s": float(run.duration_s),
        "source_text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "groups": list(groups),
        "observed_commits": commits,
        "frozen_boundaries_preserved": (
            [str(event.get("text") or "") for event in commits]
            == [str(group["text"]) for group in groups]
            and [int(event.get("segment_id", -1)) for event in commits]
            == list(range(len(groups)))
        ),
        "events": run.events,
        "audio_chunks": run.audio_chunks,
        "replay_contract": dict(replay_contract),
        "artifacts": {"wav": str(wav_path.resolve()) if samples.size else None},
        **(pcm_hashes(samples) if samples.size else {}),
    }
    write_json(case_dir / "run.json", record)
    return record


def replay_endpoint(
    adapter: Any,
    *,
    text_file: Path,
    groups_file: Path,
    output_dir: Path,
    seeds: Sequence[int],
    runtime_mode: str,
) -> list[dict[str, Any]]:
    """Replay frozen packets through one endpoint session per seed."""

    if runtime_mode not in {"sample", "greedy"}:
        raise ValueError("runtime_mode must be sample or greedy")
    text, groups = load_frozen_replay(text_file, groups_file)
    arm = MatchedEndpointArm(adapter, groups)
    return [
        _persist_case(
            output_dir,
            arm.collect(text, session_id=logical_session_id(seed), seed=int(seed)),
            source_text=text,
            groups=groups,
            replay_contract={
                "input_mode": "long_segment",
                "group_policy": "auto",
                "vad_enabled": False,
                "delivery": "firehose",
                "runtime_mode": runtime_mode,
                "sampling_parameters_are_server_owned": True,
            },
        )
        for seed in seeds
    ]


def replay_official(
    adapter: Any,
    *,
    text_file: Path,
    groups_file: Path,
    output_dir: Path,
    seeds: Sequence[int],
    mode: str,
) -> list[dict[str, Any]]:
    """Replay the same groups with the official PyTorch implementation."""

    if mode not in {"sample", "greedy"}:
        raise ValueError("mode must be sample or greedy")
    text, groups = load_frozen_replay(text_file, groups_file)
    generation = MATCHED_GENERATION if mode == "sample" else GREEDY_GENERATION
    arm = MatchedOfficialArm(adapter, groups, generation_kwargs=generation)
    return [
        _persist_case(
            output_dir,
            arm.collect(text, session_id=logical_session_id(seed), seed=int(seed)),
            source_text=text,
            groups=groups,
            replay_contract={
                "mode": mode,
                "generation": generation,
                "sampling_seed_identity_matched": True,
                "sampling_random_stream_matched": False,
            },
        )
        for seed in seeds
    ]


def score_replay_cases(
    cases_dir: Path,
    *,
    text_file: Path,
    client_class: Any,
    asr_url: str,
    language: str = "中文",
    chunk_ms: int = 960,
) -> list[dict[str, Any]]:
    """ASR-score every diagnostic full WAV, one fresh connection per case."""

    reference = text_file.read_bytes().decode("utf-8")
    summaries: list[dict[str, Any]] = []
    run_paths = sorted(cases_dir.glob("*/seed_*/run.json"))
    if not run_paths:
        raise FileNotFoundError(f"no diagnostic run.json files under {cases_dir}")
    for run_path in run_paths:
        run = read_json(run_path)
        wav_value = dict(run.get("artifacts") or {}).get("wav")
        if not isinstance(wav_value, str) or not wav_value:
            raise RuntimeError(f"diagnostic run has no WAV: {run_path}")
        wav_path = Path(wav_value)
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        sidecar_path = run_path.with_name("full.asr.json")
        request = {
            "asr_url": asr_url,
            "language": language,
            "chunk_ms": int(chunk_ms),
            "delivery_mode": "offline",
            "pacing": "none",
            "partial_mode": "off",
            "vad": "fsmn",
            "wav_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
        }
        if sidecar_path.exists():
            payload = read_json(sidecar_path)
            existing_request = payload.get("asr_request")
            if existing_request is not None and existing_request != request:
                raise RuntimeError(
                    f"existing ASR sidecar has a different request: {sidecar_path}"
                )
        else:
            result = asyncio.run(
                transcribe_wav(
                    client_class,
                    wav_path,
                    uri=asr_url,
                    language=language,
                    chunk_ms=chunk_ms,
                    duration_s=float(run["duration_s"]),
                    strict_sdk_contract=True,
                )
            )
            payload = {
                **result,
                "diagnostic_only": True,
                "asr_request": request,
                "reference_text_sha256": hashlib.sha256(
                    reference.encode("utf-8")
                ).hexdigest(),
                "character_errors": (
                    character_error_metrics(reference, str(result["transcript"]))
                    if result.get("status") == "ok"
                    else None
                ),
            }
            write_json(sidecar_path, payload)
        errors = payload.get("character_errors") or {}
        summaries.append(
            {
                "arm": run["arm"],
                "seed": int(run["seed"]),
                "status": payload.get("status"),
                "cer": errors.get("cer"),
                "substitutions": errors.get("substitutions"),
                "deletions": errors.get("deletions"),
                "insertions": errors.get("insertions"),
                "sidecar": str(sidecar_path.resolve()),
            }
        )
    return summaries


__all__ = [
    "load_frozen_replay",
    "replay_endpoint",
    "replay_official",
    "score_replay_cases",
]
