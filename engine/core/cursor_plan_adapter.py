"""Build native-cursor plans from already committed streaming-TN records.

This module deliberately does not normalize text or choose cursor labels.  The
primary streaming TN owns spoken-form semantics; a model-owned labelizer is
injected at this boundary and returns numeric cursor-vocabulary ids.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .native_cursor import CursorLabelPlan, CursorOwnerSpan


Labelizer = Callable[[str], Sequence[int]]


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value:
        raise ValueError(f"{name} must be an integer")
    return result


def _owner_id(commit: Any, ordinal: int) -> int:
    commit_id = _integer(getattr(commit, "commit_id", 0), name="commit_id")
    span_id = _integer(getattr(commit, "span_id", 0), name="span_id")
    if commit_id > 0:
        return commit_id
    if span_id > 0:
        return span_id
    return ordinal + 1


def _raw_span(commit: Any) -> tuple[int, int]:
    start = _integer(getattr(commit, "raw_start", 0), name="raw_start")
    end = _integer(getattr(commit, "raw_end", start), name="raw_end")
    if start < 0 or end < start:
        raise ValueError("commit raw span is invalid")
    mapping = tuple(getattr(commit, "mapping", ()) or ())
    if not mapping:
        return start, end

    mapped: list[tuple[int, int]] = []
    for item in mapping:
        if len(item) != 2:
            raise ValueError("commit mapping entries must be [start, end]")
        mapped_start = _integer(item[0], name="mapping start")
        mapped_end = _integer(item[1], name="mapping end")
        if mapped_start < start or mapped_end < mapped_start or mapped_end > end:
            raise ValueError("commit mapping must stay inside the commit raw span")
        mapped.append((mapped_start, mapped_end))
    return min(start for start, _ in mapped), max(end for _, end in mapped)


def _check_committed_prefix(
    plan: CursorLabelPlan,
    previous: CursorLabelPlan | None,
    committed_label_count: int,
) -> None:
    if committed_label_count < 0:
        raise ValueError("committed_label_count must be non-negative")
    if previous is None or committed_label_count == 0:
        return
    if committed_label_count > len(previous.label_ids):
        raise ValueError("committed_label_count exceeds previous plan")
    if len(plan.label_ids) < committed_label_count:
        raise ValueError("cursor plan rewrites committed labels")
    if plan.label_ids[:committed_label_count] != previous.label_ids[:committed_label_count]:
        raise ValueError("cursor plan rewrites committed labels")

    previous_owners = tuple(
        owner for owner in previous.owner_spans if owner.label_start < committed_label_count
    )
    current_owners = tuple(
        owner for owner in plan.owner_spans if owner.label_start < committed_label_count
    )
    if previous_owners != current_owners:
        raise ValueError("cursor plan rewrites committed owner spans")


class CursorLabelPlanAdapter:
    """Convert committed TN records into monotonic cursor-plan revisions."""

    def __init__(self, labelize: Labelizer):
        if not callable(labelize):
            raise TypeError("labelize must be callable")
        self._labelize = labelize
        self._last_revision = -1

    @property
    def last_revision(self) -> int:
        return self._last_revision

    def reset(self) -> None:
        self._last_revision = -1

    def build(
        self,
        commits: Iterable[Any],
        *,
        spoken_texts: Sequence[str] | None = None,
        normalized_base: int = 0,
        revision: int,
        final: bool = False,
        previous: CursorLabelPlan | None = None,
        committed_label_count: int = 0,
    ) -> CursorLabelPlan:
        base = _integer(normalized_base, name="normalized_base")
        if base < 0:
            raise ValueError("normalized_base must be non-negative")
        revision_value = _integer(revision, name="revision")
        if revision_value < self._last_revision:
            raise ValueError("cursor plan revision cannot move backwards")

        commit_list = tuple(commits)
        if spoken_texts is not None and len(spoken_texts) != len(commit_list):
            raise ValueError("spoken_texts must align one-to-one with commits")

        label_ids: list[int] = []
        owners: list[CursorOwnerSpan] = []
        seen_owner_ids: set[int] = set()
        normalized_cursor = base
        for ordinal, commit in enumerate(commit_list):
            spoken = str(
                (
                    spoken_texts[ordinal]
                    if spoken_texts is not None
                    else getattr(commit, "tts_text", "")
                )
                or ""
            )
            if not spoken:
                continue
            owner_id = _owner_id(commit, ordinal)
            if owner_id in seen_owner_ids:
                raise ValueError(f"duplicate cursor owner id {owner_id}")
            seen_owner_ids.add(owner_id)
            raw_start, raw_end = _raw_span(commit)
            try:
                labels = tuple(self._labelize(spoken))
            except Exception as exc:
                raise ValueError("cursor labelizer failed") from exc
            for label in labels:
                label_value = _integer(label, name="cursor label id")
                if label_value < 0:
                    raise ValueError("cursor label ids must be non-negative")
                label_ids.append(label_value)

            normalized_end = normalized_cursor + len(spoken)
            if not labels:
                normalized_cursor = normalized_end
                continue
            label_start = len(label_ids) - len(labels)
            owners.append(
                CursorOwnerSpan(
                    owner_id=owner_id,
                    label_start=label_start,
                    label_end=len(label_ids),
                    normalized_start=normalized_cursor,
                    normalized_end=normalized_end,
                    raw_start=raw_start,
                    raw_end=raw_end,
                )
            )
            normalized_cursor = normalized_end

        plan = CursorLabelPlan(
            label_ids=tuple(label_ids),
            owner_spans=tuple(owners),
            revision=revision_value,
            final=bool(final),
        )
        _check_committed_prefix(plan, previous, committed_label_count)
        self._last_revision = revision_value
        return plan


__all__ = ("CursorLabelPlanAdapter", "Labelizer")
