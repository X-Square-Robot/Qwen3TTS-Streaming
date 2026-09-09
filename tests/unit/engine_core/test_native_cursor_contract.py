from __future__ import annotations

import pytest

from engine.core.native_cursor import (
    CursorLabelPlan,
    CursorOwnerSpan,
    ProgressMode,
    reanchor_cursor_mu,
    slice_cursor_label_plan,
)


def _owner(start: int, end: int, owner_id: int = 0) -> CursorOwnerSpan:
    return CursorOwnerSpan(
        owner_id=owner_id,
        label_start=start,
        label_end=end,
        normalized_start=start,
        normalized_end=end,
        raw_start=start,
        raw_end=end,
    )


def test_label_plan_validates_contiguous_owner_coverage():
    plan = CursorLabelPlan(
        label_ids=(11, 12, 13),
        owner_spans=(_owner(0, 2), _owner(2, 3, owner_id=1)),
        revision=4,
    )

    assert plan.label_count == 3
    assert plan.active is True
    assert plan.owner_spans[-1].raw_end == 3


@pytest.mark.parametrize(
    "owners",
    [
        (_owner(1, 2),),
        (_owner(0, 1), _owner(3, 4, owner_id=1)),
        (_owner(0, 3),),
    ],
)
def test_label_plan_rejects_uncovered_or_misaligned_labels(owners):
    with pytest.raises(ValueError):
        CursorLabelPlan(label_ids=(1, 2), owner_spans=owners)


def test_empty_plan_is_valid_for_inactive_cursor():
    plan = CursorLabelPlan()

    assert plan.active is False
    assert plan.label_count == 0


def test_progress_mode_is_a_stable_enum():
    assert [mode.value for mode in ProgressMode] == [
        "auto",
        "native",
        "ema",
        "disabled",
    ]


def test_segment_plan_slices_only_complete_tn_owners():
    plan = CursorLabelPlan(
        label_ids=(10, 11, 20),
        owner_spans=(
            CursorOwnerSpan(1, 0, 2, 0, 2, 0, 2),
            CursorOwnerSpan(2, 2, 3, 2, 3, 2, 3),
        ),
        revision=5,
    )

    segment = slice_cursor_label_plan(
        plan,
        normalized_start=0,
        normalized_end=2,
    )
    assert segment is not None
    assert segment.label_ids == (10, 11)
    assert segment.owner_spans[0].label_start == 0
    assert segment.owner_spans[0].label_end == 2

    assert (
        slice_cursor_label_plan(
            plan,
            normalized_start=1,
            normalized_end=3,
        )
        is None
    )


def test_reanchor_uses_stable_owner_ids_when_tail_label_count_changes():
    previous = CursorLabelPlan(
        label_ids=(1, 2, 3, 4),
        owner_spans=(
            CursorOwnerSpan(10, 0, 2, 0, 2, 0, 2),
            CursorOwnerSpan(11, 2, 4, 2, 4, 2, 4),
        ),
        revision=1,
    )
    current = CursorLabelPlan(
        label_ids=(1, 2, 8, 9, 10),
        owner_spans=(
            CursorOwnerSpan(10, 0, 2, 0, 2, 0, 2),
            CursorOwnerSpan(11, 2, 5, 2, 5, 2, 4),
        ),
        revision=2,
    )

    assert reanchor_cursor_mu(previous, current, previous_mu=4.0) == 5.0


def test_reanchor_uses_stable_spans_when_owner_id_was_rewritten():
    previous = CursorLabelPlan(
        label_ids=(1, 2),
        owner_spans=(CursorOwnerSpan(10, 0, 2, 0, 2, 0, 2),),
        revision=1,
    )
    current = CursorLabelPlan(
        label_ids=(7, 8, 9),
        owner_spans=(CursorOwnerSpan(20, 0, 3, 0, 2, 0, 2),),
        revision=2,
    )

    assert reanchor_cursor_mu(previous, current, previous_mu=2.0) == 3.0


def test_reanchor_does_not_guess_across_a_newer_stable_span():
    previous = CursorLabelPlan(
        label_ids=(1, 2),
        owner_spans=(CursorOwnerSpan(10, 0, 2, 0, 2, 0, 2),),
        revision=1,
    )
    current = CursorLabelPlan(
        label_ids=(7,),
        owner_spans=(CursorOwnerSpan(20, 0, 1, 0, 1, 2, 3),),
        revision=2,
    )

    assert reanchor_cursor_mu(previous, current, previous_mu=2.0) == 0.0
