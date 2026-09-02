"""Recover delivered audio segment boundaries from canonical progress events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class DeliveredSegment:
    segment_id: int
    text: str
    output_sample_start: int
    output_sample_end: int
    segment_end_meta: dict[str, str]
    progress_meta: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_delivered_segments(
    events: Sequence[Mapping[str, Any]],
    *,
    total_samples: int,
) -> list[DeliveredSegment]:
    """Build monotonic slices from audio ends that were actually delivered.

    A guarded ``loop_abort`` deliberately has no final alignment event, but its
    preceding progress events still carry exact output-sample coordinates for
    audio already sent to the caller.  The maximum such coordinate is a real
    delivery boundary, not an inferred duration, and must not be merged into
    the following segment.
    """

    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    ends: dict[int, tuple[int, dict[str, str], bool]] = {}
    texts: dict[int, tuple[str, dict[str, str]]] = {}

    for event in events:
        event_type = str(event.get("type", ""))
        raw_id = event.get("segment_id", event.get("segment_idx", -1))
        try:
            segment_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if segment_id < 0:
            continue
        meta = {str(key): str(value) for key, value in dict(event.get("meta") or {}).items()}
        if event_type == "segment_end":
            texts[segment_id] = (str(event.get("text", "")), meta)
            continue
        if event_type != "text_progress":
            continue
        raw_end = meta.get("output_sample_end")
        if raw_end is None:
            continue
        try:
            output_end = int(raw_end)
        except (TypeError, ValueError):
            continue
        if 0 <= output_end <= total_samples:
            is_final = _truthy(meta.get("alignment_final")) or _truthy(
                meta.get("progress_final")
            )
            previous = ends.get(segment_id)
            if previous is None or output_end >= previous[0]:
                ends[segment_id] = (output_end, meta, is_final)

    if not ends:
        return []
    ordered = sorted(ends.items(), key=lambda item: (item[1][0], item[0]))
    result: list[DeliveredSegment] = []
    cursor = 0
    for segment_id, (output_end, progress_meta, is_final) in ordered:
        if output_end <= cursor:
            continue
        text, end_meta = texts.get(segment_id, ("", {}))
        boundary_meta = {
            **progress_meta,
            "delivery_boundary_source": (
                "final_progress_output_end"
                if is_final
                else "max_nonfinal_progress_output_end"
            ),
        }
        result.append(
            DeliveredSegment(
                segment_id=segment_id,
                text=text,
                output_sample_start=cursor,
                output_sample_end=output_end,
                segment_end_meta=end_meta,
                progress_meta=boundary_meta,
            )
        )
        cursor = output_end
    if result and cursor < total_samples:
        last = result[-1]
        result[-1] = DeliveredSegment(
            segment_id=last.segment_id,
            text=last.text,
            output_sample_start=last.output_sample_start,
            output_sample_end=total_samples,
            segment_end_meta=last.segment_end_meta,
            progress_meta=last.progress_meta,
        )
    return result
