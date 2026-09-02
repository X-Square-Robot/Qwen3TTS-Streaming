"""ASR diagnostics, delivered-segment evidence, and sentence review clips."""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.validation.hallucination.asr import transcribe_wav

from .acoustic import acoustic_diagnostics
from .alignment import align_reference_sentences
from .artifacts import discover_run_records, read_json, write_json
from .asr_sidecar import (
    load_reusable_sidecar,
    request_identity,
    sidecar_path,
    write_sidecar,
)
from .audio_io import load_mono_wav, slice_ms, write_mono_wav
from .metrics import character_error_metrics, normalize_transcript
from .models import ArmKind, RunStatus, parse_reference_sentences
from .reference_groups import freeze_reference_groups
from .segments import DeliveredSegment, extract_delivered_segments
from .telemetry import summarize_engine_events


def _asr_segments(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("status") != "ok":
        return []
    return [dict(item) for item in payload.get("segments", []) if isinstance(item, Mapping)]


def _overlapping_transcript(
    segments: Sequence[Mapping[str, Any]], start_ms: int, end_ms: int
) -> str:
    selected: list[str] = []
    for segment in segments:
        try:
            start = int(segment.get("start_ms", 0) or 0)
            end = int(segment.get("end_ms", start) or start)
        except (TypeError, ValueError):
            continue
        overlap_start = max(start, start_ms)
        overlap_end = min(end, end_ms)
        if overlap_end <= overlap_start:
            continue
        normalized = normalize_transcript(str(segment.get("text", "")))
        if not normalized:
            continue
        width = end - start
        if width <= 0:
            selected.append(normalized)
            continue
        relative_start = (overlap_start - start) / width
        relative_end = (overlap_end - start) / width
        text_start = max(0, math.floor(len(normalized) * relative_start))
        text_end = min(len(normalized), math.ceil(len(normalized) * relative_end))
        if text_end <= text_start:
            text_end = min(len(normalized), text_start + 1)
        selected.append(normalized[text_start:text_end])
    return "".join(selected)


def _fallback_segment(total_samples: int) -> DeliveredSegment:
    return DeliveredSegment(
        segment_id=0,
        text="",
        output_sample_start=0,
        output_sample_end=total_samples,
        segment_end_meta={},
        progress_meta={},
    )


async def _transcribe(
    client_class: Any,
    wav_path: Path,
    *,
    uri: str,
    language: str,
    chunk_ms: int,
    duration_s: float,
) -> dict[str, Any]:
    # transcribe_wav creates one fresh SDK context/connection per invocation.
    return await transcribe_wav(
        client_class,
        wav_path,
        uri=uri,
        language=language,
        chunk_ms=chunk_ms,
        duration_s=duration_s,
    )


def _transcribe_with_sidecar(
    client_class: Any,
    wav_path: Path,
    *,
    uri: str,
    language: str,
    chunk_ms: int,
    duration_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse one successful exact-input sidecar or persist a fresh attempt."""

    request = request_identity(
        wav_path,
        uri=uri,
        language=language,
        chunk_ms=chunk_ms,
    )
    cache_path = sidecar_path(wav_path)
    cached = load_reusable_sidecar(
        cache_path,
        expected_request=request,
        required_provenance_source="fresh_connection",
    )
    if cached is not None:
        return cached, {
            "cache_hit": True,
            "sidecar": str(cache_path.name),
            "wav_sha256": request["wav"]["sha256"],
            "source": "fresh_connection_sidecar_cache",
            "origin_source": "fresh_connection",
        }
    result = asyncio.run(
        _transcribe(
            client_class,
            wav_path,
            uri=uri,
            language=language,
            chunk_ms=chunk_ms,
            duration_s=duration_s,
        )
    )
    write_sidecar(
        cache_path,
        request=request,
        result=result,
        provenance={"source": "fresh_connection"},
    )
    return result, {
        "cache_hit": False,
        "sidecar": str(cache_path.name),
        "wav_sha256": request["wav"]["sha256"],
        "source": "fresh_connection",
        "origin_source": "fresh_connection",
    }


def _score_delivered_segments(
    client_class: Any,
    *,
    run_dir: Path,
    samples: np.ndarray,
    sample_rate: int,
    delivered: Sequence[DeliveredSegment],
    uri: str,
    language: str,
    chunk_ms: int,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    segment_dir = run_dir / "segments"
    for sequence_index, segment in enumerate(delivered):
        clip = samples[segment.output_sample_start : segment.output_sample_end]
        wav_path = segment_dir / f"segment_{sequence_index:03d}.wav"
        write_mono_wav(wav_path, clip, sample_rate)
        asr, asr_provenance = _transcribe_with_sidecar(
            client_class,
            wav_path,
            uri=uri,
            language=language,
            chunk_ms=chunk_ms,
            duration_s=clip.size / float(sample_rate),
        )
        hypothesis = str(asr.get("transcript", ""))
        reference = segment.text
        if asr.get("status") != "ok":
            character_errors = None
            character_errors_unavailable_reason = "asr_failed"
        elif not normalize_transcript(reference):
            character_errors = None
            character_errors_unavailable_reason = "empty_normalized_reference"
        else:
            character_errors = character_error_metrics(reference, hypothesis)
            character_errors_unavailable_reason = None
        evidence.append(
            {
                "sequence_index": sequence_index,
                **segment.to_dict(),
                "wav": str(wav_path.name),
                "duration_s": clip.size / float(sample_rate),
                "asr": asr,
                "asr_provenance": asr_provenance,
                "character_errors": character_errors,
                "character_errors_unavailable_reason": (
                    character_errors_unavailable_reason
                ),
            }
        )
    write_json(segment_dir / "index.json", {"segments": evidence})
    return evidence


def score_run(
    output_root: Path,
    record: Mapping[str, Any],
    *,
    client_class: Any,
    asr_url: str,
    language: str = "中文",
    chunk_ms: int = 960,
) -> list[dict[str, Any]]:
    """Score one complete WAV and emit exactly one row per reference sentence."""

    run_json = output_root / str(record["_record_path"])
    run_dir = run_json.parent
    result_path = run_dir / "scoring.json"
    observation_path = run_dir / "sentence_observations.json"
    reference_text = str(record["source_text"])
    references = parse_reference_sentences(reference_text)
    tts_complete = record.get("status") == RunStatus.OK.value
    wav_relative = record.get("artifacts", {}).get("wav")
    if not wav_relative:
        observations = [
            {
                "arm": record["arm"],
                "seed": int(record["seed"]),
                "session_id": record["session_id"],
                "sentence_ordinal": sentence.ordinal,
                "sentence_id": sentence.sentence_id,
                "reference_text": sentence.text,
                "status": RunStatus.TTS_FAILED.value,
                "tts_run_status": str(record.get("status", "error")),
                "tts_complete": False,
                "asr_status": "not_run",
                "asr_localization_ready": False,
                "valid_for_review": False,
                "invalid_reason": record.get("error") or "TTS produced no WAV",
            }
            for sentence in references
        ]
        write_json(observation_path, {"observations": observations})
        write_json(
            result_path,
            {"status": RunStatus.TTS_FAILED.value, "error": record.get("error")},
        )
        return observations

    wav_path = output_root / str(wav_relative)
    samples, sample_rate = load_mono_wav(wav_path)
    duration_s = samples.size / float(sample_rate)
    full_asr, full_asr_provenance = _transcribe_with_sidecar(
        client_class,
        wav_path,
        uri=asr_url,
        language=language,
        chunk_ms=chunk_ms,
        duration_s=duration_s,
    )
    full_asr_ok = full_asr.get("status") == "ok"
    full_segments = _asr_segments(full_asr)
    full_hypothesis = str(full_asr.get("transcript", ""))
    delivered = extract_delivered_segments(
        list(record.get("events") or []), total_samples=samples.size
    )
    if not delivered:
        delivered = [_fallback_segment(samples.size)]
    segment_evidence = _score_delivered_segments(
        client_class,
        run_dir=run_dir,
        samples=samples,
        sample_rate=sample_rate,
        delivered=delivered,
        uri=asr_url,
        language=language,
        chunk_ms=chunk_ms,
    )
    segment_asr_failures = sum(
        item.get("asr", {}).get("status") != "ok" for item in segment_evidence
    )

    timed = align_reference_sentences(
        reference_text,
        full_segments,
        audio_duration_ms=max(1, round(duration_s * 1000)),
    )
    observations: list[dict[str, Any]] = []
    sentence_dir = run_dir / "sentences"
    if not tts_complete:
        observation_status = RunStatus.TTS_FAILED.value
        invalid_reason = record.get("error") or "TTS run was not complete"
    else:
        observation_status = RunStatus.REVIEW_PENDING.value
        invalid_reason = None
    for index, (sentence, interval) in enumerate(zip(references, timed, strict=True)):
        clip = slice_ms(
            samples, sample_rate, interval.clip_start_ms, interval.clip_end_ms
        )
        context_start = max(0, interval.start_ms - 2500)
        context_end = min(round(duration_s * 1000), interval.end_ms + 2500)
        context = slice_ms(samples, sample_rate, context_start, max(context_start + 1, context_end))
        clip_path = sentence_dir / f"sentence_{sentence.ordinal:03d}.wav"
        context_path = sentence_dir / f"sentence_{sentence.ordinal:03d}_context.wav"
        write_mono_wav(clip_path, clip, sample_rate)
        write_mono_wav(context_path, context, sample_rate)
        sentence_hypothesis = _overlapping_transcript(
            full_segments, interval.start_ms, interval.end_ms
        )
        try:
            clip_acoustic: dict[str, Any] = acoustic_diagnostics(clip, sample_rate)
        except Exception as exc:  # noqa: BLE001 - diagnostic failure is traceable
            clip_acoustic = {"error": f"{type(exc).__name__}: {exc}", "diagnostic_only": True}
        audio_reviewable = tts_complete and clip.size > 0 and context.size > 0
        diagnostic_warnings = []
        if not full_asr_ok:
            diagnostic_warnings.append("full_wav_asr_failed")
        elif not full_segments:
            diagnostic_warnings.append("full_wav_asr_has_no_timed_segments")
        observations.append(
            {
                "arm": record["arm"],
                "seed": int(record["seed"]),
                "session_id": record["session_id"],
                "sentence_ordinal": sentence.ordinal,
                "sentence_id": sentence.sentence_id,
                "occurrence": sentence.occurrence,
                "reference_text": sentence.text,
                "previous_reference": references[index - 1].text if index else "",
                "next_reference": references[index + 1].text if index + 1 < len(references) else "",
                "status": observation_status,
                "tts_run_status": str(record.get("status", "error")),
                "tts_complete": tts_complete,
                "asr_status": str(full_asr.get("status", "error")),
                "asr_localization_ready": full_asr_ok and bool(full_segments),
                "valid_for_review": audio_reviewable,
                "invalid_reason": (
                    invalid_reason
                    if invalid_reason is not None
                    else None
                    if audio_reviewable
                    else "review clip is empty"
                ),
                "diagnostic_warnings": diagnostic_warnings,
                "clip": str(clip_path.relative_to(output_root)),
                "context_clip": str(context_path.relative_to(output_root)),
                "timing": interval.to_dict(),
                "asr_transcript": sentence_hypothesis,
                "character_errors": (
                    character_error_metrics(sentence.text, sentence_hypothesis)
                    if full_asr_ok
                    else None
                ),
                "acoustic": clip_acoustic,
            }
        )

    try:
        full_acoustic: dict[str, Any] = acoustic_diagnostics(samples, sample_rate)
    except Exception as exc:  # noqa: BLE001
        full_acoustic = {"error": f"{type(exc).__name__}: {exc}", "diagnostic_only": True}
    if not tts_complete:
        scoring_status = RunStatus.TTS_FAILED.value
    elif not full_asr_ok or segment_asr_failures:
        scoring_status = RunStatus.ASR_FAILED.value
    else:
        scoring_status = RunStatus.OK.value
    scoring = {
        "status": scoring_status,
        "full_wav": str(wav_path.relative_to(output_root)),
        "full_asr": full_asr,
        "full_asr_provenance": full_asr_provenance,
        "full_character_errors": (
            character_error_metrics(reference_text, full_hypothesis)
            if full_asr_ok
            else None
        ),
        "delivered_segments": segment_evidence,
        "asr_failures": {
            "full_wav": int(not full_asr_ok),
            "delivered_segments": segment_asr_failures,
        },
        "engine_telemetry": summarize_engine_events(list(record.get("events") or [])),
        "acoustic": full_acoustic,
        "asr_is_diagnostic_only": True,
        "human_review_is_ground_truth": True,
    }
    write_json(result_path, scoring)
    write_json(observation_path, {"observations": observations})
    return observations


def scoring_completion_summary(output_root: Path) -> dict[str, Any]:
    """Summarize formal run-grid and ASR readiness without treating gaps clean."""

    records = discover_run_records(output_root)
    arm_counts = Counter(str(record.get("arm", "")) for record in records)
    positions = Counter(
        (str(record.get("arm", "")), int(record.get("seed", -1)))
        for record in records
    )
    manifest_path = output_root / "manifest.json"
    expected_positions: set[tuple[str, int]] = set()
    expected_observations: int | None = None
    manifest_seed_grid_valid = False
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        seeds = [int(seed) for seed in manifest.get("seeds") or []]
        manifest_seed_grid_valid = len(seeds) == 3 and len(set(seeds)) == 3
        expected_positions = {
            (arm.value, seed) for arm in ArmKind for seed in seeds
        }
        source_text = str((manifest.get("text") or {}).get("text", ""))
        expected_observations = (
            len(expected_positions) * len(parse_reference_sentences(source_text))
        )

    tts_failed_runs: list[str] = []
    unscored_runs: list[str] = []
    full_asr_failed_runs: list[str] = []
    segment_asr_failed_runs: list[str] = []
    segment_asr_failure_count = 0
    invalid_asr_provenance_wavs: list[str] = []
    asr_wav_count = 0
    observations: list[dict[str, Any]] = []
    for record in records:
        record_path = str(record["_record_path"])
        if record.get("status") != RunStatus.OK.value:
            tts_failed_runs.append(record_path)
        run_dir = (output_root / record_path).parent
        scoring_path = run_dir / "scoring.json"
        observation_path = run_dir / "sentence_observations.json"
        try:
            scoring = read_json(scoring_path)
            run_observations = list(
                read_json(observation_path).get("observations") or []
            )
        except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
            unscored_runs.append(record_path)
            continue
        observations.extend(
            dict(item) for item in run_observations if isinstance(item, Mapping)
        )
        full_asr = scoring.get("full_asr")
        if record.get("artifacts", {}).get("wav"):
            asr_wav_count += 1
        if isinstance(full_asr, Mapping):
            if full_asr.get("status") != "ok":
                full_asr_failed_runs.append(record_path)
        elif record.get("artifacts", {}).get("wav"):
            full_asr_failed_runs.append(record_path)
        full_provenance = scoring.get("full_asr_provenance")
        if (
            not isinstance(full_provenance, Mapping)
            or full_provenance.get("origin_source") != "fresh_connection"
        ):
            invalid_asr_provenance_wavs.append(
                str(scoring.get("full_wav") or f"{record_path}:full_wav")
            )
        segment_failures = sum(
            not isinstance(item.get("asr"), Mapping)
            or item["asr"].get("status") != "ok"
            for item in scoring.get("delivered_segments") or []
            if isinstance(item, Mapping)
        )
        if segment_failures:
            segment_asr_failed_runs.append(record_path)
            segment_asr_failure_count += segment_failures
        for item in scoring.get("delivered_segments") or []:
            if not isinstance(item, Mapping):
                continue
            asr_wav_count += 1
            provenance = item.get("asr_provenance")
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("origin_source") != "fresh_connection"
            ):
                invalid_asr_provenance_wavs.append(
                    str(
                        (
                            run_dir
                            / "segments"
                            / str(item.get("wav") or "unknown_segment.wav")
                        ).relative_to(output_root)
                    )
                )

    missing_run_positions = sorted(expected_positions - positions.keys())
    unexpected_run_positions = sorted(positions.keys() - expected_positions)
    duplicate_run_positions = sorted(
        position for position, count in positions.items() if count != 1
    )
    run_grid_complete = manifest_seed_grid_valid and bool(expected_positions) and not (
        missing_run_positions
        or unexpected_run_positions
        or duplicate_run_positions
    )
    review_ready_count = sum(
        item.get("valid_for_review") is True for item in observations
    )
    blocked_observation_count = len(observations) - review_ready_count
    blocking_reasons: list[str] = []
    diagnostic_warnings: list[str] = []
    if not run_grid_complete:
        blocking_reasons.append("run_grid_incomplete")
    if tts_failed_runs:
        blocking_reasons.append("tts_failed_runs")
    if unscored_runs:
        blocking_reasons.append("unscored_runs")
    if full_asr_failed_runs:
        diagnostic_warnings.append("full_asr_failed_runs")
    if segment_asr_failure_count:
        diagnostic_warnings.append("segment_asr_failed_wavs")
    if invalid_asr_provenance_wavs:
        diagnostic_warnings.append("asr_wav_without_fresh_connection_origin")
    if blocked_observation_count:
        blocking_reasons.append("observations_not_review_ready")
    if (
        expected_observations is not None
        and len(observations) != expected_observations
    ):
        blocking_reasons.append("observation_count_mismatch")

    return {
        "schema_version": 2,
        "run_count": len(records),
        "expected_run_count": len(expected_positions) or None,
        "arm_run_counts": dict(sorted(arm_counts.items())),
        "manifest_seed_grid_valid": manifest_seed_grid_valid,
        "run_grid_complete": run_grid_complete,
        "missing_run_positions": [list(item) for item in missing_run_positions],
        "unexpected_run_positions": [
            list(item) for item in unexpected_run_positions
        ],
        "duplicate_run_positions": [
            list(item) for item in duplicate_run_positions
        ],
        "tts_failed_run_count": len(tts_failed_runs),
        "tts_failed_runs": tts_failed_runs,
        "unscored_run_count": len(unscored_runs),
        "unscored_runs": unscored_runs,
        "full_asr_failure_count": len(full_asr_failed_runs),
        "full_asr_failed_runs": full_asr_failed_runs,
        "segment_asr_failure_count": segment_asr_failure_count,
        "segment_asr_failed_runs": segment_asr_failed_runs,
        "asr_wav_count": asr_wav_count,
        "fresh_connection_origin_wav_count": (
            asr_wav_count - len(invalid_asr_provenance_wavs)
        ),
        "invalid_asr_provenance_wav_count": len(invalid_asr_provenance_wavs),
        "invalid_asr_provenance_wavs": invalid_asr_provenance_wavs,
        "observation_count": len(observations),
        "expected_observation_count": expected_observations,
        "review_ready_observation_count": review_ready_count,
        "blocked_observation_count": blocked_observation_count,
        "ready_for_review": not blocking_reasons,
        "diagnostic_degraded": bool(diagnostic_warnings),
        "blocking_reasons": blocking_reasons,
        "diagnostic_warnings": diagnostic_warnings,
        # Compatibility for callers that historically displayed ``reasons``
        # when readiness was false.  Diagnostic-only failures intentionally do
        # not appear here and therefore cannot block a blind-listening package.
        "reasons": blocking_reasons,
    }


def score_all_runs(
    output_root: Path,
    *,
    client_class: Any,
    asr_url: str,
    language: str = "中文",
    chunk_ms: int = 960,
) -> list[dict[str, Any]]:
    """Serially score every discovered arm WAV (ASR concurrency is always one)."""

    records = discover_run_records(output_root)
    observations: list[dict[str, Any]] = []
    for record in records:
        observations.extend(
            score_run(
                output_root,
                record,
                client_class=client_class,
                asr_url=asr_url,
                language=language,
                chunk_ms=chunk_ms,
            )
        )
    freeze_reference_groups(output_root)
    write_json(output_root / "sentence_observations.json", {"observations": observations})
    write_json(
        output_root / "scoring_summary.json",
        scoring_completion_summary(output_root),
    )
    return observations


__all__ = ["score_all_runs", "score_run", "scoring_completion_summary"]
