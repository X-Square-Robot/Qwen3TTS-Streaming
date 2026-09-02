"""Engine-event summaries used for diagnosis, never for human truth labels."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


_INTEGER_METRICS = (
    "loop_max_run",
    "loop_suspect_count",
    "loop_recovery_count",
    "loop_abort_count",
    "retry_idx",
    "retry_count",
    "abort_tail_frames",
    "generated_frames",
    "delivered_frames",
    "output_sample_end",
)


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_engine_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce raw events while retaining the fields needed for root-cause triage."""

    types: Counter[str] = Counter()
    maxima: dict[str, int] = {}
    segment_ids: set[int] = set()
    guard_modes: set[str] = set()
    eos_reasons: list[str] = []
    retry_events: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("type", ""))
        types[event_type] += 1
        raw_id = event.get("segment_id", -1)
        segment_id = _integer(raw_id)
        if segment_id is not None and segment_id >= 0:
            segment_ids.add(segment_id)
        meta = dict(event.get("meta") or {})
        for name in _INTEGER_METRICS:
            value = _integer(meta.get(name))
            if value is not None:
                maxima[name] = max(maxima.get(name, value), value)
        guard_mode = str(meta.get("guard_mode", "")).strip()
        if guard_mode:
            guard_modes.add(guard_mode)
        reason = str(meta.get("eos_reason") or meta.get("reason") or "").strip()
        if reason:
            eos_reasons.append(reason)
        retry_idx = _integer(meta.get("retry_idx")) or 0
        retry_count = _integer(meta.get("retry_count")) or 0
        if "retry" in event_type.lower() or retry_idx > 0 or retry_count > 0:
            retry_events.append(
                {
                    "type": event_type,
                    "segment_id": segment_id,
                    "message": str(event.get("message", "")),
                    "meta": {str(key): str(value) for key, value in meta.items()},
                }
            )
    return {
        "event_count": sum(types.values()),
        "event_type_counts": dict(sorted(types.items())),
        "segment_ids": sorted(segment_ids),
        "segment_count": len(segment_ids),
        "metric_maxima": maxima,
        "guard_modes": sorted(guard_modes),
        "eos_reasons": eos_reasons,
        "retry_events": retry_events,
        "diagnostic_only": True,
    }


__all__ = ["summarize_engine_events"]
