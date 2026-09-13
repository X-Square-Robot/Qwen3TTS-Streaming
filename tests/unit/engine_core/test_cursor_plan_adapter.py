from types import SimpleNamespace

import pytest

from engine.core.cursor_plan_adapter import CursorLabelPlanAdapter
from engine.core.native_cursor import CursorLabelPlan, CursorOwnerSpan


def _commit(
    text: str,
    *,
    raw_start: int = 0,
    raw_end: int | None = None,
    mapping=(),
    commit_id: int = 0,
    span_id: int = 0,
    raw_text: str | None = None,
    commit_kind: str | None = None,
):
    source = text if raw_text is None else raw_text
    return SimpleNamespace(
        tts_text=text,
        raw_text=source,
        commit_kind=commit_kind,
        raw_start=raw_start,
        raw_end=len(text) if raw_end is None else raw_end,
        mapping=mapping,
        commit_id=commit_id,
        span_id=span_id,
    )


def test_builds_multiple_commit_owners_and_stable_coordinates():
    adapter = CursorLabelPlanAdapter(lambda text: list(range(len(text))))
    plan = adapter.build(
        [
            _commit("你好", raw_start=4, raw_end=6, commit_id=11),
            _commit("abc", raw_start=6, raw_end=9, commit_id=12),
        ],
        normalized_base=3,
        revision=7,
        final=True,
    )

    assert plan == CursorLabelPlan(
        label_ids=(0, 1, 0, 1, 2),
        owner_spans=(
            CursorOwnerSpan(11, 0, 2, 3, 5, 4, 6),
            CursorOwnerSpan(12, 2, 5, 5, 8, 6, 9),
        ),
        revision=7,
        final=True,
    )
    assert [(span.owner_id, span.label_start, span.label_end) for span in plan.owner_spans] == [
        (11, 0, 2),
        (12, 2, 5),
    ]
    assert [(span.normalized_start, span.normalized_end) for span in plan.owner_spans] == [
        (3, 5),
        (5, 8),
    ]


def test_mapping_uses_conservative_union_without_length_guessing():
    adapter = CursorLabelPlanAdapter(lambda text: [9, 10, 11])
    plan = adapter.build(
        [_commit("spoken", raw_start=10, raw_end=15, mapping=((10, 11), (13, 15)))],
        revision=0,
    )
    assert (plan.owner_spans[0].raw_start, plan.owner_spans[0].raw_end) == (10, 15)


def test_literal_commit_exposes_fine_grained_raw_owners():
    class Labelizer:
        def __call__(self, text):
            return tuple(range(len(text)))

        def encode_with_spans(self, text, *, strict=False):
            return tuple(range(len(text))), tuple((i, i + 1) for i in range(len(text)))

    adapter = CursorLabelPlanAdapter(Labelizer())
    plan = adapter.build(
        [_commit("你好，世界。", raw_start=10, raw_end=16, commit_id=7, commit_kind="literal")],
        revision=0,
    )

    # The labelizer emits no gaps in this focused contract test.  Each label
    # has a stable owner, while the final owner absorbs the sentence-ending
    # boundary so raw high-water can finish at the commit end.
    assert len(plan.owner_spans) == 6
    assert [owner.raw_start for owner in plan.owner_spans] == [10, 11, 12, 13, 14, 15]
    assert plan.owner_spans[-1].raw_end == 16
    assert plan.owner_spans[0].owner_id == (7 << 32) | 1
    assert plan.owner_spans[-1].owner_id == (7 << 32) | 6


def test_long_literal_owner_does_not_freeze_progress_at_its_first_character():
    class Labelizer:
        def __call__(self, text):
            return tuple(range(len(text)))

        def encode_with_spans(self, text, *, strict=False):
            return tuple(range(len(text))), tuple((i, i + 1) for i in range(len(text)))

    text = "，都是学生自己用加工工具造出来的，这个工程化的实操能力。" + (
        "有些核心车队的成员在毕业以后甚至在比亚迪、蔚来这类主机厂的"
        "工程开发部拿到了正式的"
    )
    start = 703
    adapter = CursorLabelPlanAdapter(Labelizer())
    plan = adapter.build(
        [_commit(
            text,
            raw_start=start,
            raw_end=start + len(text),
            commit_id=27,
            commit_kind="literal",
        )],
        revision=0,
    )

    target = text.index("有些")
    target_owner = next(
        owner for owner in plan.owner_spans if owner.raw_start == start + target
    )
    assert target_owner.raw_end == start + target + 1
    assert plan.owner_spans[-1].raw_end == start + len(text)


def test_expanded_commit_remains_one_semantic_owner():
    adapter = CursorLabelPlanAdapter(lambda text: list(range(len(text))))
    plan = adapter.build(
        [_commit(
            "99%",
            raw_start=0,
            raw_end=3,
            commit_id=5,
            commit_kind="normalized",
            raw_text="99%",
        )],
        spoken_texts=["百分之九十九"],
        revision=0,
    )
    assert len(plan.owner_spans) == 1


def test_identity_commit_is_split_even_if_upstream_kind_is_normalized():
    class Labelizer:
        def __call__(self, text):
            return tuple(range(len(text)))

        def encode_with_spans(self, text, *, strict=False):
            return tuple(range(len(text))), tuple((i, i + 1) for i in range(len(text)))

    adapter = CursorLabelPlanAdapter(Labelizer())
    plan = adapter.build(
        [_commit("普通文本", raw_start=10, raw_end=14, commit_id=8, commit_kind="normalized")],
        revision=0,
    )
    assert len(plan.owner_spans) == len("普通文本")


def test_literal_sub_owner_ids_remain_stable_across_plan_revisions():
    class Labelizer:
        def __call__(self, text):
            return tuple(range(len(text)))

        def encode_with_spans(self, text, *, strict=False):
            return tuple(range(len(text))), tuple((i, i + 1) for i in range(len(text)))

    adapter = CursorLabelPlanAdapter(Labelizer())
    first = adapter.build(
        [_commit("你好", raw_start=4, raw_end=6, commit_id=11, commit_kind="literal")],
        revision=0,
    )
    second = adapter.build(
        [
            _commit("你好", raw_start=4, raw_end=6, commit_id=11, commit_kind="literal"),
            _commit("99%", raw_start=6, raw_end=9, commit_id=12, commit_kind="normalized"),
        ],
        revision=1,
        previous=first,
        committed_label_count=2,
    )
    assert second.owner_spans[:2] == first.owner_spans


def test_adapter_can_consume_the_journaled_spoken_projection():
    adapter = CursorLabelPlanAdapter(lambda text: list(range(len(text))))
    plan = adapter.build(
        [_commit("99%", raw_start=0, raw_end=3, commit_id=1)],
        spoken_texts=["百分之九十九"],
        revision=0,
    )
    assert plan.label_count == 6
    assert plan.owner_spans[0].normalized_end == 6


def test_spoken_projection_must_align_with_commits():
    adapter = CursorLabelPlanAdapter(lambda _text: [1])
    with pytest.raises(ValueError, match="align"):
        adapter.build([_commit("x")], spoken_texts=[], revision=0)


def test_empty_text_and_empty_plan_are_valid():
    adapter = CursorLabelPlanAdapter(lambda text: [])
    plan = adapter.build([_commit("", raw_start=0, raw_end=0)], revision=0)
    assert plan.label_ids == ()
    assert plan.owner_spans == ()
    assert plan.active is False


@pytest.mark.parametrize("labels", [(-1,), (1.5,), (True,)])
def test_rejects_invalid_labels(labels):
    adapter = CursorLabelPlanAdapter(lambda _text: labels)
    with pytest.raises(ValueError, match="label"):
        adapter.build([_commit("x")], revision=0)


def test_revision_cannot_move_back_and_equal_revision_is_idempotent():
    adapter = CursorLabelPlanAdapter(lambda _text: [1])
    adapter.build([_commit("x")], revision=3)
    adapter.build([_commit("x")], revision=3)
    with pytest.raises(ValueError, match="revision"):
        adapter.build([_commit("x")], revision=2)


def test_committed_prefix_and_owner_cannot_be_rewritten():
    adapter = CursorLabelPlanAdapter(lambda text: [ord(text[0])])
    previous = adapter.build([_commit("a", commit_id=9)], revision=0)
    with pytest.raises(ValueError, match="committed"):
        adapter.build(
            [_commit("b", commit_id=9)],
            revision=1,
            previous=previous,
            committed_label_count=1,
        )


def test_invalid_mapping_is_rejected_before_plan_is_published():
    adapter = CursorLabelPlanAdapter(lambda _text: [1])
    with pytest.raises(ValueError, match="mapping"):
        adapter.build(
            [_commit("x", raw_start=3, raw_end=4, mapping=((2, 4),))],
            revision=0,
        )
