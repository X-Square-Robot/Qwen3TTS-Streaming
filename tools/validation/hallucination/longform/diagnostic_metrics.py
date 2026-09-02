"""Metric extraction for pre-review diagnostic rows."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def as_integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def character_error_fields(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        return {
            "metrics_available": False,
            "reference_characters": None,
            "hypothesis_characters": None,
            "distance": None,
            "substitutions": None,
            "deletions": None,
            "insertions": None,
            "cer": None,
        }
    return {
        "metrics_available": True,
        "reference_characters": as_integer(metrics.get("reference_characters")),
        "hypothesis_characters": as_integer(metrics.get("hypothesis_characters")),
        "distance": as_integer(metrics.get("distance")),
        "substitutions": as_integer(metrics.get("substitutions")),
        "deletions": as_integer(metrics.get("deletions")),
        "insertions": as_integer(metrics.get("insertions")),
        "cer": as_number(metrics.get("cer")),
    }


def is_actual_retry_event(event: Mapping[str, Any]) -> bool:
    """Accept only explicit retry events or strictly positive retry counters."""

    event_type = str(event.get("type", "")).strip().casefold()
    if "retry" in event_type:
        return True
    meta = dict(event.get("meta") or {})
    return any(
        (value := as_integer(meta.get(field))) is not None and value > 0
        for field in ("retry_idx", "retry_count")
    )


def summarize_diagnostic_events(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize EOS/loop/guard/retry without treating retry_idx=0 as retry."""

    segment_ends = [event for event in events if event.get("type") == "segment_end"]
    metric_events = segment_ends or list(events)
    eos_reasons = [
        str(dict(event.get("meta") or {}).get("eos_reason", "")).strip()
        for event in segment_ends
    ]
    eos_reasons = [reason for reason in eos_reasons if reason]
    retry_events = [event for event in events if is_actual_retry_event(event)]
    retry_indices = [
        value
        for event in events
        if (value := as_integer(dict(event.get("meta") or {}).get("retry_idx")))
        is not None
        and value > 0
    ]
    retry_counts = [
        value
        for event in events
        if (value := as_integer(dict(event.get("meta") or {}).get("retry_count")))
        is not None
        and value > 0
    ]

    def metric_values(name: str) -> list[int]:
        return [
            value
            for event in metric_events
            if (value := as_integer(dict(event.get("meta") or {}).get(name)))
            is not None
        ]

    guard_events = [
        event
        for event in events
        if str(dict(event.get("meta") or {}).get("guard_mode", "")).strip()
    ]
    guard_modes = sorted(
        {
            str(dict(event.get("meta") or {})["guard_mode"]).strip()
            for event in guard_events
        }
    )
    guard_segments = {
        segment_id
        for event in guard_events
        if (segment_id := as_integer(event.get("segment_id"))) is not None
        and segment_id >= 0
    }
    eos_counts = Counter(eos_reasons)
    loop_maxima = metric_values("loop_max_run")
    return {
        "segment_end_event_count": len(segment_ends),
        "text_boundary_commit_count": sum(
            event.get("type") == "text_boundary_commit" for event in events
        ),
        "eos_reasons": eos_reasons,
        "eos_reason_counts": dict(sorted(eos_counts.items())),
        "loop_abort_count": eos_counts["loop_abort"],
        "loop_max_run": max(loop_maxima, default=0),
        "loop_suspect_count": sum(metric_values("loop_suspect_count")),
        "loop_recovery_count": sum(metric_values("loop_recovery_count")),
        "guard_modes": guard_modes,
        "guard_event_count": len(guard_events),
        "guard_segment_count": len(guard_segments),
        "actual_retry_event_count": len(retry_events),
        "actual_retry_max_idx": max(retry_indices, default=0),
        "actual_retry_max_count": max(retry_counts, default=0),
        "had_actual_retry": bool(retry_events),
    }


def text_boundary_map(record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Resolve commit text to verified or exact-text-derived source offsets."""

    source_text = str(record.get("source_text", ""))
    commits = sorted(
        (
            event
            for event in record.get("events") or []
            if isinstance(event, Mapping)
            and event.get("type") == "text_boundary_commit"
        ),
        key=lambda event: as_integer(event.get("segment_id")) or 0,
    )
    ids = [as_integer(event.get("segment_id")) for event in commits]
    exact_partition = (
        ids == list(range(len(commits)))
        and "".join(str(event.get("text", "")) for event in commits) == source_text
    )
    result: dict[int, dict[str, Any]] = {}
    cursor = 0
    for event in commits:
        segment_id = as_integer(event.get("segment_id"))
        if segment_id is None or segment_id < 0:
            continue
        text = str(event.get("text", ""))
        meta = dict(event.get("meta") or {})
        raw_start = as_integer(meta.get("raw_codepoint_start"))
        raw_end = as_integer(meta.get("raw_codepoint_end"))
        offsets_verified = (
            raw_start is not None
            and raw_end is not None
            and 0 <= raw_start < raw_end <= len(source_text)
            and source_text[raw_start:raw_end] == text
        )
        if offsets_verified:
            provenance = "verified_event_raw_codepoint_offsets"
        elif exact_partition:
            raw_start = cursor
            raw_end = cursor + len(text)
            provenance = "derived_from_exact_commit_text"
        else:
            raw_start = raw_end = None
            provenance = "event_text_without_verified_document_offsets"
        result[segment_id] = {
            "raw_start": raw_start,
            "raw_end": raw_end,
            "provenance": provenance,
        }
        if exact_partition:
            cursor += len(text)
    return result


__all__ = [
    "as_integer",
    "as_number",
    "character_error_fields",
    "is_actual_retry_event",
    "summarize_diagnostic_events",
    "text_boundary_map",
]
