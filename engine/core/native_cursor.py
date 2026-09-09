"""CPU-side contracts shared by streaming TN and the fused cursor graph.

The graph receives only padded numeric labels and recurrent state.  Raw and
normalized coordinates stay in these owner spans so the CPU projector can
publish conservative, monotonic progress after TensorRT returns ``mu``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class ProgressMode(str, Enum):
    """Requested text-progress route for one session."""

    AUTO = "auto"
    NATIVE = "native"
    EMA = "ema"
    DISABLED = "disabled"


# Stable recurrent-state ABI shared by manifest validation and TRT discovery.
CURSOR_RECURRENT_INPUT_BINDINGS = frozenset(
    {
        "cursor_mu_in",
        "cursor_frames_since_advance_in",
        "cursor_delta_history_in",
        "cursor_conv_history_in",
        "cursor_last_trunk_input_in",
        "cursor_seen_frames_in",
    }
)
CURSOR_RECURRENT_OUTPUT_BINDINGS = frozenset(
    {
        "cursor_mu",
        "cursor_frames_since_advance",
        "cursor_delta_history",
        "cursor_conv_history",
        "cursor_last_trunk_input",
        "cursor_seen_frames",
    }
)


@dataclass(frozen=True, slots=True)
class CursorOwnerSpan:
    """Stable provenance for a contiguous spoken-label owner."""

    owner_id: int
    label_start: int
    label_end: int
    normalized_start: int
    normalized_end: int
    raw_start: int
    raw_end: int

    def __post_init__(self) -> None:
        values = (
            self.owner_id,
            self.label_start,
            self.label_end,
            self.normalized_start,
            self.normalized_end,
            self.raw_start,
            self.raw_end,
        )
        if any(isinstance(value, bool) or int(value) != value for value in values):
            raise ValueError("cursor owner coordinates must be integers")
        if self.owner_id < 0:
            raise ValueError("owner_id must be non-negative")
        if self.label_start < 0 or self.label_end < self.label_start:
            raise ValueError("invalid label owner span")
        if self.normalized_start < 0 or self.normalized_end < self.normalized_start:
            raise ValueError("invalid normalized owner span")
        if self.raw_start < 0 or self.raw_end < self.raw_start:
            raise ValueError("invalid raw owner span")


@dataclass(frozen=True, slots=True)
class CursorLabelPlan:
    """One immutable TN-derived plan revision for a session/segment.

    ``label_ids`` are cursor vocabulary ids, never Talker BPE ids.  Each
    label must be covered by exactly one ordered owner span; an empty plan is
    valid for an inactive cursor before the first committed spoken text.
    """

    label_ids: tuple[int, ...] = ()
    owner_spans: tuple[CursorOwnerSpan, ...] = ()
    revision: int = 0
    final: bool = False
    # Exact source provenance for each cursor label, in global spoken
    # codepoints. Empty denotes a legacy labelizer without offset support.
    label_normalized_spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or int(self.revision) != self.revision:
            raise ValueError("plan revision must be an integer")
        if self.revision < 0:
            raise ValueError("plan revision must be non-negative")
        if any(
            isinstance(label_id, bool) or int(label_id) != label_id or int(label_id) < 0
            for label_id in self.label_ids
        ):
            raise ValueError("cursor label ids must be non-negative integers")

        previous_end = 0
        for index, owner in enumerate(self.owner_spans):
            if owner.label_start != previous_end:
                raise ValueError(
                    "owner spans must cover labels contiguously from zero "
                    f"(owner index {index})"
                )
            if owner.label_end <= owner.label_start:
                raise ValueError("owner spans must contain at least one label")
            previous_end = owner.label_end
        if previous_end != len(self.label_ids):
            raise ValueError("owner spans must cover the complete label plan")
        if self.label_normalized_spans:
            if len(self.label_normalized_spans) != len(self.label_ids):
                raise ValueError("label spans must align one-to-one with labels")
            last_start = last_end = 0
            for start, end in self.label_normalized_spans:
                if any(isinstance(v, bool) or int(v) != v for v in (start, end)):
                    raise ValueError("label spans must use integer codepoints")
                if start < last_start or end < last_end or end <= start:
                    raise ValueError("label spans must be nonempty and ordered")
                last_start, last_end = start, end
            for owner in self.owner_spans:
                if any(start < owner.normalized_start or end > owner.normalized_end
                       for start, end in self.label_normalized_spans[
                           owner.label_start:owner.label_end
                       ]):
                    raise ValueError("label provenance must stay inside its owner")

    @property
    def label_count(self) -> int:
        return len(self.label_ids)

    @property
    def active(self) -> bool:
        return bool(self.label_ids)


def reanchor_cursor_mu(
    previous: CursorLabelPlan,
    current: CursorLabelPlan,
    *,
    previous_mu: float,
) -> float:
    """Map a live cursor position onto a rewritten TN plan by owner identity.

    ``mu`` is a label-space coordinate, so it cannot be carried across a
    tail rewrite by index.  Completed owners are identified in the previous
    plan and looked up by their stable ``owner_id`` in the current plan.  If a
    rewrite removed those owners, the same confirmed raw/normalized frontier
    is used to find a replacement boundary; if none exists, the injected
    position is zero.  The public raw/normalized high-water remains owned by
    the CPU projector and never moves backwards.

    The helper intentionally does not interpolate or use text-length ratios.
    It is a CPU-side coordinate injection for the fused graph's next step.
    """

    try:
        position = float(previous_mu)
    except (TypeError, ValueError) as exc:
        raise ValueError("previous cursor mu must be numeric") from exc
    if not math.isfinite(position):
        raise ValueError("previous cursor mu must be finite")
    position = max(0.0, min(float(previous.label_count), position))
    if not current.active:
        return 0.0

    # Pure appends preserve a label coordinate only when both numerical ids
    # and exact source provenance through the live position are unchanged.
    prefix = math.ceil(position)
    if (previous.label_normalized_spans and current.label_normalized_spans
            and prefix <= current.label_count
            and previous.label_ids[:prefix] == current.label_ids[:prefix]
            and previous.label_normalized_spans[:prefix]
            == current.label_normalized_spans[:prefix]
            and all(
                any(new.owner_id == old.owner_id
                    and (new.raw_start, new.raw_end) == (old.raw_start, old.raw_end)
                    for new in current.owner_spans)
                for old in previous.owner_spans if old.label_start < prefix
            )):
        return position

    completed_ids = {
        owner.owner_id
        for owner in previous.owner_spans
        if position >= owner.label_end
    }
    if not completed_ids:
        return 0.0

    current_boundaries = [
        owner.label_end
        for owner in current.owner_spans
        if owner.owner_id in completed_ids
    ]
    if current_boundaries:
        return float(max(current_boundaries))

    # A normalizer may replace an owner id during a tail rewrite.  Retain the
    # confirmed coordinate only when a new owner ends within the same stable
    # raw/normalized frontier; this is still a boundary lookup, never a ratio.
    completed = [
        owner for owner in previous.owner_spans if owner.owner_id in completed_ids
    ]
    raw_frontier = max(owner.raw_end for owner in completed)
    normalized_frontier = max(owner.normalized_end for owner in completed)
    replacement_boundaries = [
        owner.label_end
        for owner in current.owner_spans
        if (
            owner.raw_end <= raw_frontier
            and owner.normalized_end <= normalized_frontier
        )
    ]
    return float(max(replacement_boundaries, default=0))


def slice_cursor_label_plan(
    plan: CursorLabelPlan,
    *,
    normalized_start: int,
    normalized_end: int,
) -> CursorLabelPlan | None:
    """Slice exact label provenance into a segment's global spoken window.

    Owners may straddle segments when exact label offsets are available. Raw
    owner spans are retained, not interpolated; the shared journal performs
    raw confirmation. Legacy plans lacking offsets only allow whole owners.
    """

    start = int(normalized_start)
    end = int(normalized_end)
    if start < 0 or end < start:
        raise ValueError("invalid normalized segment bounds")
    owners = plan.owner_spans
    if not owners:
        return CursorLabelPlan(revision=plan.revision, final=plan.final)

    if plan.label_normalized_spans:
        indices = [
            i for i, (left, right) in enumerate(plan.label_normalized_spans)
            if left < end and right > start
        ]
        if not indices:
            return CursorLabelPlan(revision=plan.revision, final=plan.final)
        first, last = indices[0], indices[-1] + 1
        # A label itself cannot be split without more precise provenance.
        if any(left < start or right > end
               for left, right in plan.label_normalized_spans[first:last]):
            return None
        sliced_owners = tuple(
            CursorOwnerSpan(
                owner.owner_id,
                max(owner.label_start, first) - first,
                min(owner.label_end, last) - first,
                max(owner.normalized_start, start),
                min(owner.normalized_end, end),
                owner.raw_start,
                owner.raw_end,
            )
            for owner in owners
            if owner.label_start < last and owner.label_end > first
        )
        return CursorLabelPlan(
            label_ids=plan.label_ids[first:last],
            owner_spans=sliced_owners,
            revision=plan.revision,
            final=plan.final,
            label_normalized_spans=plan.label_normalized_spans[first:last],
        )

    selected = []
    for owner in owners:
        overlaps = owner.normalized_start < end and owner.normalized_end > start
        inside = owner.normalized_start >= start and owner.normalized_end <= end
        if overlaps and not inside:
            return None
        if inside:
            selected.append(owner)
    if not selected:
        return CursorLabelPlan(revision=plan.revision, final=plan.final)

    label_start = selected[0].label_start
    label_end = selected[-1].label_end
    labels = plan.label_ids[label_start:label_end]
    rebased = tuple(
        CursorOwnerSpan(
            owner_id=owner.owner_id,
            label_start=owner.label_start - label_start,
            label_end=owner.label_end - label_start,
            normalized_start=owner.normalized_start,
            normalized_end=owner.normalized_end,
            raw_start=owner.raw_start,
            raw_end=owner.raw_end,
        )
        for owner in selected
    )
    return CursorLabelPlan(
        label_ids=labels,
        owner_spans=rebased,
        revision=plan.revision,
        final=plan.final,
    )


__all__ = (
    "CURSOR_RECURRENT_INPUT_BINDINGS",
    "CURSOR_RECURRENT_OUTPUT_BINDINGS",
    "CursorLabelPlan",
    "CursorOwnerSpan",
    "ProgressMode",
    "reanchor_cursor_mu",
    "slice_cursor_label_plan",
)
