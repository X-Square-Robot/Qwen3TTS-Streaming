"""Row builders and arm aggregation for pre-review diagnostics."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Mapping, Sequence

from .diagnostic_grid import ARMS, DiagnosticInputs
from .diagnostic_metrics import (
    as_integer,
    as_number,
    character_error_fields,
    summarize_diagnostic_events,
    text_boundary_map,
)
from .text_normalization import normalize_transcript


def build_document_rows(inputs: DiagnosticInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in (item.value for item in ARMS):
        for seed in inputs.seeds:
            record = inputs.records[(arm, seed)]
            scoring = inputs.scoring[(arm, seed)]
            errors = character_error_fields(scoring.get("full_character_errors"))
            event_diagnostics = summarize_diagnostic_events(
                [
                    event
                    for event in record.get("events") or []
                    if isinstance(event, Mapping)
                ]
            )
            full_asr = scoring.get("full_asr")
            asr_status = (
                full_asr.get("status") if isinstance(full_asr, Mapping) else None
            )
            asr_provenance = scoring.get("full_asr_provenance")
            wav_sha256 = (
                asr_provenance.get("wav_sha256")
                if isinstance(asr_provenance, Mapping)
                else None
            )
            acoustic = scoring.get("acoustic")
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "session_id": record.get("session_id"),
                    "run_status": record.get("status"),
                    "scoring_status": scoring.get("status"),
                    "full_asr_status": asr_status,
                    "duration_s": as_number(record.get("duration_s")),
                    "sample_rate": as_integer(record.get("sample_rate")),
                    "sample_count": as_integer(record.get("sample_count")),
                    "pcm_f32le_sha256": record.get("pcm_f32le_sha256"),
                    "pcm_s16le_sha256": record.get("pcm_s16le_sha256"),
                    "wav_sha256": wav_sha256,
                    **errors,
                    "delivered_segment_count": len(
                        scoring.get("delivered_segments") or []
                    ),
                    **event_diagnostics,
                    "terminal_event": record.get("terminal_event"),
                    "acoustic_status": (
                        "error"
                        if isinstance(acoustic, Mapping) and acoustic.get("error")
                        else "available"
                        if isinstance(acoustic, Mapping)
                        else "missing"
                    ),
                    "acoustic": acoustic,
                    "diagnostic_only": True,
                }
            )
    return rows


def build_sentence_rows(inputs: DiagnosticInputs) -> list[dict[str, Any]]:
    group_for_sentence = {
        int(ordinal): int(group["group_index"])
        for group in inputs.reference_groups
        for ordinal in group["sentence_ordinals"]
    }
    rows: list[dict[str, Any]] = []
    for arm in (item.value for item in ARMS):
        for seed in inputs.seeds:
            for observation in inputs.observations[(arm, seed)]:
                metrics = observation.get("character_errors")
                flattened = character_error_fields(metrics)
                insertion_candidates = (
                    list(metrics.get("insertion_spans") or [])
                    if isinstance(metrics, Mapping)
                    else []
                )
                repeat_candidates = (
                    list(metrics.get("repetition_spans") or [])
                    if isinstance(metrics, Mapping)
                    else []
                )
                insertion_repeat_count = sum(
                    len(candidate.get("repetition_spans") or [])
                    for candidate in insertion_candidates
                    if isinstance(candidate, Mapping)
                )
                acoustic = observation.get("acoustic")
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "session_id": observation.get("session_id"),
                        "sentence_ordinal": int(observation["sentence_ordinal"]),
                        "sentence_id": observation.get("sentence_id"),
                        "reference_group": group_for_sentence[
                            int(observation["sentence_ordinal"])
                        ],
                        "reference_text": observation.get("reference_text"),
                        "asr_transcript": observation.get("asr_transcript"),
                        "status": observation.get("status"),
                        "tts_run_status": observation.get("tts_run_status"),
                        "asr_status": observation.get("asr_status"),
                        "valid_for_review": observation.get("valid_for_review"),
                        **flattened,
                        "insertion_candidate_count": len(insertion_candidates),
                        "insertion_candidates": insertion_candidates,
                        "repeat_candidate_count": len(repeat_candidates),
                        "repeat_candidates": repeat_candidates,
                        "insertion_repeat_candidate_count": insertion_repeat_count,
                        "clip": observation.get("clip"),
                        "context_clip": observation.get("context_clip"),
                        "timing": observation.get("timing"),
                        "acoustic_status": (
                            "error"
                            if isinstance(acoustic, Mapping) and acoustic.get("error")
                            else "available"
                            if isinstance(acoustic, Mapping)
                            else "missing"
                        ),
                        "acoustic": acoustic,
                        "candidate_interpretation": (
                            "diagnostic_only_requires_blind_review"
                        ),
                    }
                )
    return rows


def build_reference_group_rows(
    inputs: DiagnosticInputs,
    sentence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_position = {
        (str(row["arm"]), int(row["seed"]), int(row["sentence_ordinal"])): row
        for row in sentence_rows
    }
    rows: list[dict[str, Any]] = []
    for arm in (item.value for item in ARMS):
        for seed in inputs.seeds:
            for group in inputs.reference_groups:
                selected = [
                    by_position[(arm, seed, int(ordinal))]
                    for ordinal in group["sentence_ordinals"]
                ]
                available = [row for row in selected if row["metrics_available"]]
                reference_characters = sum(
                    int(row["reference_characters"] or 0) for row in available
                )
                distance = sum(int(row["distance"] or 0) for row in available)
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "reference_group": int(group["group_index"]),
                        "segment_id": int(group["segment_id"]),
                        "raw_start": int(group["raw_start"]),
                        "raw_end": int(group["raw_end"]),
                        "boundary_provenance": group.get("provenance"),
                        "sentence_ordinals": list(group["sentence_ordinals"]),
                        "sentence_count": len(selected),
                        "metrics_available_sentence_count": len(available),
                        "all_sentence_metrics_available": len(available)
                        == len(selected),
                        "reference_characters": reference_characters,
                        "distance": distance,
                        "substitutions": sum(
                            int(row["substitutions"] or 0) for row in available
                        ),
                        "deletions": sum(
                            int(row["deletions"] or 0) for row in available
                        ),
                        "insertions": sum(
                            int(row["insertions"] or 0) for row in available
                        ),
                        "cer": (
                            distance / reference_characters
                            if reference_characters
                            else None
                        ),
                        "insertion_candidate_count": sum(
                            int(row["insertion_candidate_count"]) for row in selected
                        ),
                        "repeat_candidate_count": sum(
                            int(row["repeat_candidate_count"]) for row in selected
                        ),
                        "diagnostic_only": True,
                    }
                )
    return rows


def build_segment_rows(inputs: DiagnosticInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frozen_groups = {
        int(group["segment_id"]): group for group in inputs.reference_groups
    }
    for arm in (item.value for item in ARMS):
        for seed in inputs.seeds:
            record = inputs.records[(arm, seed)]
            boundaries = text_boundary_map(record)
            scoring = inputs.scoring[(arm, seed)]
            for segment in scoring.get("delivered_segments") or []:
                if not isinstance(segment, Mapping):
                    continue
                segment_id = as_integer(segment.get("segment_id"))
                text = str(segment.get("text", ""))
                normalized = normalize_transcript(text)
                if text and not normalized:
                    reference_kind = "punctuation_only"
                    availability = "not_applicable_punctuation_only"
                    errors = character_error_fields(None)
                elif not normalized:
                    reference_kind = "empty"
                    availability = "not_applicable_empty_reference"
                    errors = character_error_fields(None)
                else:
                    reference_kind = "speech_text"
                    errors = character_error_fields(segment.get("character_errors"))
                    availability = (
                        "available"
                        if errors["metrics_available"]
                        else str(
                            segment.get("character_errors_unavailable_reason")
                            or "unavailable"
                        )
                    )
                boundary = (
                    boundaries.get(segment_id, {}) if segment_id is not None else {}
                )
                raw_start = boundary.get("raw_start")
                raw_end = boundary.get("raw_end")
                overlapping_groups = [
                    int(group["group_index"])
                    for group in inputs.reference_groups
                    if isinstance(raw_start, int)
                    and isinstance(raw_end, int)
                    and int(group["raw_end"]) > raw_start
                    and int(group["raw_start"]) < raw_end
                ]
                frozen_group = (
                    frozen_groups.get(segment_id)
                    if arm == "triton_0818" and segment_id is not None
                    else None
                )
                if frozen_group and frozen_group.get("text") != text:
                    frozen_group = None
                progress = dict(segment.get("progress_meta") or {})
                asr = segment.get("asr")
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "sequence_index": as_integer(segment.get("sequence_index")),
                        "segment_id": segment_id,
                        "reference_text": text,
                        "reference_kind": reference_kind,
                        "duration_s": as_number(segment.get("duration_s")),
                        "output_sample_start": as_integer(
                            segment.get("output_sample_start")
                        ),
                        "output_sample_end": as_integer(
                            segment.get("output_sample_end")
                        ),
                        "text_boundary_raw_start": raw_start,
                        "text_boundary_raw_end": raw_end,
                        "text_boundary_provenance": boundary.get(
                            "provenance",
                            "full_wav_fallback_no_text_boundary",
                        ),
                        "delivery_boundary_provenance": progress.get(
                            "delivery_boundary_source",
                            "full_wav_fallback_no_delivery_boundary",
                        ),
                        "frozen_reference_group_provenance": (
                            frozen_group.get("provenance") if frozen_group else None
                        ),
                        "overlapping_reference_groups": overlapping_groups,
                        "asr_status": (
                            asr.get("status") if isinstance(asr, Mapping) else None
                        ),
                        "asr_transcript": (
                            asr.get("transcript") if isinstance(asr, Mapping) else None
                        ),
                        "cer_availability": availability,
                        **errors,
                        "cer_display": (
                            "N/A" if errors["cer"] is None else errors["cer"]
                        ),
                        "diagnostic_only": True,
                    }
                )
    return rows


def aggregate_arms(
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for arm in (item.value for item in ARMS):
        selected = [row for row in documents if row["arm"] == arm]
        available = [
            row
            for row in selected
            if row["metrics_available"] and row["cer"] is not None
        ]
        reference_characters = sum(
            int(row["reference_characters"] or 0) for row in available
        )
        distance = sum(int(row["distance"] or 0) for row in available)
        durations = [
            float(row["duration_s"])
            for row in selected
            if row["duration_s"] is not None
        ]
        eos_counts: Counter[str] = Counter()
        for row in selected:
            eos_counts.update(dict(row["eos_reason_counts"]))
        f32_hashes = {
            str(row["pcm_f32le_sha256"]) for row in selected if row["pcm_f32le_sha256"]
        }
        summaries[arm] = {
            "run_count": len(selected),
            "duration_total_s": sum(durations),
            "duration_mean_s": mean(durations) if durations else None,
            "metrics_available_run_count": len(available),
            "reference_characters": reference_characters,
            "distance": distance,
            "substitutions": sum(int(row["substitutions"] or 0) for row in available),
            "deletions": sum(int(row["deletions"] or 0) for row in available),
            "insertions": sum(int(row["insertions"] or 0) for row in available),
            "weighted_cer": (
                distance / reference_characters if reference_characters else None
            ),
            "mean_document_cer": (
                mean(float(row["cer"]) for row in available) if available else None
            ),
            "delivered_segment_count": sum(
                int(row["delivered_segment_count"]) for row in selected
            ),
            "segment_end_event_count": sum(
                int(row["segment_end_event_count"]) for row in selected
            ),
            "eos_reason_counts": dict(sorted(eos_counts.items())),
            "loop_max_run": max(
                (int(row["loop_max_run"]) for row in selected), default=0
            ),
            "loop_suspect_count": sum(
                int(row["loop_suspect_count"]) for row in selected
            ),
            "loop_recovery_count": sum(
                int(row["loop_recovery_count"]) for row in selected
            ),
            "loop_abort_count": sum(int(row["loop_abort_count"]) for row in selected),
            "guard_run_count": sum(bool(row["guard_modes"]) for row in selected),
            "actual_retry_run_count": sum(
                bool(row["had_actual_retry"]) for row in selected
            ),
            "actual_retry_event_count": sum(
                int(row["actual_retry_event_count"]) for row in selected
            ),
            "unique_pcm_f32le_hash_count": len(f32_hashes),
            "all_pcm_f32le_hashes_identical": len(f32_hashes) == 1 and bool(selected),
            "acoustic_error_run_count": sum(
                row["acoustic_status"] == "error" for row in selected
            ),
            "diagnostic_only": True,
        }
    return summaries


__all__ = [
    "aggregate_arms",
    "build_document_rows",
    "build_reference_group_rows",
    "build_segment_rows",
    "build_sentence_rows",
]
