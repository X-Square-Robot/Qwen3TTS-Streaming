from __future__ import annotations

import pytest

from engine.core.native_cursor import CursorLabelPlan, CursorOwnerSpan, ProgressMode


def _owner(
    *,
    owner_id: int,
    label_start: int,
    label_end: int,
    normalized_start: int,
    normalized_end: int,
    raw_start: int,
    raw_end: int,
) -> CursorOwnerSpan:
    return CursorOwnerSpan(
        owner_id=owner_id,
        label_start=label_start,
        label_end=label_end,
        normalized_start=normalized_start,
        normalized_end=normalized_end,
        raw_start=raw_start,
        raw_end=raw_end,
    )


def test_ordered_owner_spans_map_distinct_label_slices_to_one_plan():
    owners = (
        _owner(
            owner_id=17,
            label_start=0,
            label_end=2,
            normalized_start=0,
            normalized_end=4,
            raw_start=0,
            raw_end=2,
        ),
        _owner(
            owner_id=23,
            label_start=2,
            label_end=5,
            normalized_start=4,
            normalized_end=9,
            raw_start=2,
            raw_end=7,
        ),
    )
    plan = CursorLabelPlan(
        label_ids=(101, 102, 201, 202, 203),
        owner_spans=owners,
        revision=6,
    )

    assert plan.active is True
    assert plan.label_count == 5
    assert [plan.label_ids[span.label_start : span.label_end] for span in owners] == [
        (101, 102),
        (201, 202, 203),
    ]
    assert [(span.owner_id, span.raw_start, span.raw_end) for span in plan.owner_spans] == [
        (17, 0, 2),
        (23, 2, 7),
    ]


def test_final_empty_plan_remains_inactive():
    plan = CursorLabelPlan(revision=9, final=True)

    assert plan.active is False
    assert plan.label_count == 0
    assert plan.final is True


@pytest.mark.parametrize(
    "owners",
    [
        (
            _owner(
                owner_id=1,
                label_start=0,
                label_end=1,
                normalized_start=0,
                normalized_end=1,
                raw_start=0,
                raw_end=1,
            ),
            _owner(
                owner_id=2,
                label_start=3,
                label_end=4,
                normalized_start=1,
                normalized_end=2,
                raw_start=1,
                raw_end=2,
            ),
        ),
        (
            _owner(
                owner_id=1,
                label_start=0,
                label_end=2,
                normalized_start=0,
                normalized_end=2,
                raw_start=0,
                raw_end=2,
            ),
            _owner(
                owner_id=2,
                label_start=1,
                label_end=3,
                normalized_start=2,
                normalized_end=4,
                raw_start=2,
                raw_end=4,
            ),
        ),
    ],
    ids=("gap", "overlap"),
)
def test_label_plan_rejects_gapped_or_overlapping_owner_spans(owners):
    with pytest.raises(ValueError, match="owner spans"):
        CursorLabelPlan(label_ids=(10, 11, 12, 13), owner_spans=owners)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: CursorLabelPlan(
                label_ids=(-1,),
                owner_spans=(
                    _owner(
                        owner_id=1,
                        label_start=0,
                        label_end=1,
                        normalized_start=0,
                        normalized_end=1,
                        raw_start=0,
                        raw_end=1,
                    ),
                ),
            ),
            id="label-id",
        ),
        pytest.param(
            lambda: CursorOwnerSpan(
                owner_id=-1,
                label_start=0,
                label_end=1,
                normalized_start=0,
                normalized_end=1,
                raw_start=0,
                raw_end=1,
            ),
            id="owner-id",
        ),
        pytest.param(
            lambda: CursorLabelPlan(
                label_ids=(10,),
                owner_spans=(
                    _owner(
                        owner_id=1,
                        label_start=0,
                        label_end=1,
                        normalized_start=0,
                        normalized_end=1,
                        raw_start=0,
                        raw_end=1,
                    ),
                ),
                revision=-1,
            ),
            id="revision",
        ),
    ],
)
def test_cursor_contract_rejects_negative_ids_and_revisions(factory):
    with pytest.raises(ValueError):
        factory()


def test_progress_modes_round_trip_stable_wire_values():
    expected = {
        "auto": ProgressMode.AUTO,
        "native": ProgressMode.NATIVE,
        "ema": ProgressMode.EMA,
        "disabled": ProgressMode.DISABLED,
    }

    assert {mode.value: mode for mode in ProgressMode} == expected
    assert all(ProgressMode(mode.value) is mode for mode in ProgressMode)
