from __future__ import annotations

from dataclasses import astuple

import pytest

from engine.frontend.spliter.driver import (
    ActionType,
    SplitThresholds,
    StreamingDriver,
    compute_thresholds,
)
from engine.frontend.spliter.event import SpliterEvent, SpliterEventType
from engine.frontend.spliter.ratio import RatioOutcome
from engine.frontend.spliter.spliter import Spliter


def _start(driver: StreamingDriver) -> None:
    driver.feed(SpliterEvent(type=SpliterEventType.START))


def _normal(token: int) -> SpliterEvent:
    return SpliterEvent(type=SpliterEventType.NORMAL_TOKEN, token=token)


def _punct(token: int, level: int) -> SpliterEvent:
    return SpliterEvent(
        type=SpliterEventType.PUNCTUATION_TOKEN,
        token=token,
        punct_level=level,
    )


def _has_flush(results) -> bool:
    return any(result.type == ActionType.FLUSH_EOS for result in results)


def _segment_text(actions, segment_idx: int) -> str:
    return "".join(
        action.token_text for action in actions if action.segment_idx == segment_idx
    )


def _token_ids(actions) -> list[int]:
    return [
        action.action.token
        for action in actions
        if action.action.type in (ActionType.PREFILL, ActionType.DECODE)
    ]


def _pending_group_lengths(spliter: Spliter) -> list[int]:
    lengths: list[int] = []
    current = 0
    for pending in spliter._pending:
        if pending.group_key is None:
            continue
        current += 1
        if pending.boundary:
            lengths.append(current)
            current = 0
    assert current == 0
    return lengths


@pytest.mark.parametrize(
    ("remaining_kv", "ema_ratio", "safety_margin", "expected"),
    [
        (500, 5.0, 8, (69, 79, 89, 98)),
        (88, 2.0, 8, (28, 32, 36, 40)),
        (88, 10.0, 8, (6, 7, 8, 8)),
        (18, 1.0, 8, (7, 8, 9, 10)),
        (9, 1.0, 8, (1, 1, 1, 1)),
    ],
)
def test_compute_thresholds_matches_capacity_formula(
    remaining_kv: int,
    ema_ratio: float,
    safety_margin: int,
    expected: tuple[int, int, int, int],
) -> None:
    thresholds = compute_thresholds(remaining_kv, ema_ratio, safety_margin)
    assert astuple(thresholds) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remaining_kv": 8, "ema_ratio": 1.0, "safety_margin": 8},
        {"remaining_kv": 100, "ema_ratio": 0.5},
        {"remaining_kv": 100, "ema_ratio": float("nan")},
        {"remaining_kv": 100, "ema_ratio": 2.0, "safety_margin": -1},
        {
            "remaining_kv": 100,
            "ema_ratio": 2.0,
            "l1_cap_ratio": 0.8,
            "l2_cap_ratio": 0.7,
        },
    ],
)
def test_compute_thresholds_rejects_invalid_capacity_inputs(kwargs) -> None:
    with pytest.raises(ValueError):
        compute_thresholds(**kwargs)


def test_hard_capacity_uses_post_append_length_without_c_plus_one() -> None:
    capacity = 6
    driver = StreamingDriver(SplitThresholds(4, 5, 6, capacity))
    _start(driver)

    token_actions = 0
    for token in range(capacity):
        results = driver.feed(_normal(token))
        token_actions += sum(
            result.type in (ActionType.PREFILL, ActionType.DECODE) for result in results
        )
        assert _has_flush(results) is (token == capacity - 1)

    assert token_actions == capacity


@pytest.mark.parametrize(("level", "threshold"), [(1, 7), (2, 8), (3, 9)])
def test_punctuation_splits_exactly_at_post_append_threshold(
    level: int, threshold: int
) -> None:
    thresholds = SplitThresholds(7, 8, 9, 10)

    early = StreamingDriver(thresholds)
    _start(early)
    for token in range(threshold - 2):
        early.feed(_normal(token))
    assert not _has_flush(early.feed(_punct(100, level)))

    exact = StreamingDriver(thresholds)
    _start(exact)
    for token in range(threshold - 1):
        exact.feed(_normal(token))
    assert _has_flush(exact.feed(_punct(100, level)))


def test_capacity_one_flushes_the_first_token() -> None:
    driver = StreamingDriver(SplitThresholds(1, 1, 1, 1))
    _start(driver)

    results = driver.feed(_normal(0))

    assert [result.type for result in results] == [
        ActionType.PREFILL,
        ActionType.FLUSH_EOS,
    ]


def test_duration_feedback_does_not_weaken_split_safety_capacity() -> None:
    spliter = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=4.5,
        safety_ratio_initial=5.5,
        ema_alpha=1.0,
        max_concurrent=2,
    )
    spliter.feed_tokens([(0, "a")])
    old_thresholds = spliter._drivers[0].thresholds

    spliter.update_ratio(actual_audio_steps=16, actual_text_tokens=8)

    assert spliter._drivers[0].thresholds == old_thresholds
    assert spliter.ema_ratio == pytest.approx(2.0)
    assert spliter.safety_ratio == pytest.approx(5.5)
    assert spliter._make_thresholds() == old_thresholds


def test_production_default_ratios_keep_duration_adaptive_and_cap_at_89() -> None:
    spliter = Spliter(
        engine_max_decode_len=512,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=4.5,
        safety_ratio_initial=5.5,
    )

    assert spliter.ema_ratio == pytest.approx(4.5)
    assert spliter.safety_ratio == pytest.approx(5.5)
    assert spliter._make_thresholds().force_split_at == 89


def test_direct_constructor_without_safety_override_preserves_old_semantics() -> None:
    spliter = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=2.0,
    )

    assert spliter.safety_ratio == pytest.approx(2.0)
    assert spliter._make_thresholds().force_split_at == 50


def test_failure_feedback_is_frozen_for_open_segment_and_used_by_next_stream() -> None:
    spliter = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=2.0,
        safety_ratio_initial=2.0,
        ema_overflow_alpha=1.0,
        safety_failure_multiplier=1.0,
        max_concurrent=2,
    )
    spliter.feed_tokens([(0, "a")])
    old_thresholds = spliter._drivers[0].thresholds

    spliter.observe_segment(
        300,
        50,
        outcome=RatioOutcome.KV_OVERFLOW,
        segment_idx=0,
    )

    assert spliter._drivers[0].thresholds == old_thresholds
    assert spliter._make_thresholds().force_split_at < old_thresholds.force_split_at

    # Fill segment 0 exactly to its frozen cap and leave one token for a newly
    # opened streaming segment. That segment uses the tightened safety state.
    spliter.feed_tokens(
        [(token, "a") for token in range(1, old_thresholds.force_split_at + 1)]
    )
    assert spliter._drivers[1].thresholds == spliter._make_thresholds()
    assert spliter._drivers[1].thresholds != old_thresholds


def test_auto_routing_uses_the_active_segments_frozen_capacity() -> None:
    spliter = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=2.0,
        safety_ratio_initial=2.0,
        ema_overflow_alpha=1.0,
        safety_failure_multiplier=1.0,
    )
    spliter.feed_auto([(0, "a")])
    frozen_capacity = spliter._drivers[0].thresholds.force_split_at
    spliter.update_ratio(
        actual_audio_steps=300,
        actual_text_tokens=50,
        overflow=True,
    )
    assert spliter._make_thresholds().force_split_at < frozen_capacity

    actions = spliter.feed_auto([(token, "a") for token in range(1, 21)])

    assert {action.segment_idx for action in actions} == {0}
    assert spliter._drivers[0].token_count == 21


def test_transition_log_records_the_frozen_threshold_snapshot() -> None:
    spliter = Spliter(
        engine_max_decode_len=40,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=2.0,
        safety_ratio_initial=2.0,
        ema_overflow_alpha=1.0,
        safety_failure_multiplier=1.0,
    )
    spliter.enable_decision_recording(True)
    spliter.feed_tokens([(0, "a")])
    frozen = spliter._drivers[0].thresholds

    spliter.update_ratio(
        actual_audio_steps=300,
        actual_text_tokens=50,
        overflow=True,
    )
    spliter.feed_tokens([(token, "a") for token in range(1, frozen.force_split_at)])

    transition = next(
        record
        for record in spliter.drain_split_decisions()
        if record["obs"] == "driver_transition"
    )
    assert transition["duration_ema_ratio"] == 2.0
    assert transition["safety_ratio"] == 2.0
    assert transition["ema_ratio"] == 2.0
    assert transition["thresholds"] == {
        "min_tokens_l1": frozen.min_tokens_l1,
        "force_split_at": frozen.force_split_at,
    }


def test_presplit_suffix_is_replanned_as_a_whole_after_safety_backoff() -> None:
    """Regression: do not turn every planned 50 group into a 47+3 pair."""
    spliter = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=2.0,
        safety_ratio_initial=2.0,
        ema_overflow_alpha=1.0,
        safety_failure_multiplier=1.0,
        max_concurrent=1,
    )
    tokens = [(token, "x") for token in range(150)]

    first = spliter.set_full_text(tokens)
    assert len(_segment_text(first, 0)) == 50
    assert spliter._drivers[0].thresholds.force_split_at == 50

    spliter.observe_segment(
        105,
        50,
        outcome=RatioOutcome.KV_OVERFLOW,
        segment_idx=0,
    )
    assert spliter._make_thresholds().force_split_at == 47
    assert _pending_group_lengths(spliter) == [47, 47, 6]

    second = spliter.on_segment_done(0)

    assert len(_segment_text(second, 1)) == 47
    assert spliter._drivers[1].thresholds.force_split_at == 47
    assert {(action.group_idx, action.local_idx) for action in second} == {(1, 0)}
    assert all(action.group_final for action in second)

    third = spliter.on_segment_done(1)
    fourth = spliter.on_segment_done(2)
    assert len(_segment_text(third, 2)) == 47
    assert len(_segment_text(fourth, 3)) == 6
    assert _token_ids(first + second + third + fourth) == list(range(150))


def test_concurrent_full_text_replans_only_unopened_suffix() -> None:
    spliter = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=2.0,
        safety_ratio_initial=2.0,
        ema_overflow_alpha=1.0,
        safety_failure_multiplier=1.0,
        max_concurrent=2,
    )

    first = spliter.set_full_text([(token, "x") for token in range(200)])
    assert len(_segment_text(first, 0)) == 50
    assert len(_segment_text(first, 1)) == 50

    spliter.observe_segment(
        105,
        50,
        outcome=RatioOutcome.KV_OVERFLOW,
        segment_idx=1,
    )

    assert spliter._drivers[0].thresholds.force_split_at == 50
    assert spliter._drivers[1].thresholds.force_split_at == 50
    assert _pending_group_lengths(spliter) == [47, 47, 6]

    # Segment 1 completes first.  The new slot opens only the replanned suffix;
    # segment 0 remains frozen and in flight.
    next_actions = spliter.on_segment_done(1)
    assert len(_segment_text(next_actions, 2)) == 47
    assert spliter._drivers[2].thresholds.force_split_at == 47


def test_each_presplit_packet_keeps_its_own_capacity_plan() -> None:
    """A later LONG_SEGMENT packet must not overwrite an older queued plan."""
    spliter = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=2.0,
        safety_ratio_initial=2.0,
        ema_overflow_alpha=1.0,
        safety_failure_multiplier=1.0,
        max_concurrent=1,
    )
    first = spliter.push_group_tokens([(token, "a") for token in range(50)])
    assert len(_segment_text(first, 0)) == 50

    # New packet is planned at ratio 2.1 (capacity 47) while segment 0 keeps
    # the original capacity 50. Tightening again must repack the still-hidden
    # packet as one 94-token suffix, not split each old group independently.
    spliter.observe_segment(
        105,
        50,
        outcome=RatioOutcome.KV_OVERFLOW,
        segment_idx=0,
    )
    assert spliter._make_thresholds().force_split_at == 47
    assert spliter.push_group_tokens([(token, "b") for token in range(94)]) == []
    spliter.observe_segment(
        110,
        50,
        outcome=RatioOutcome.KV_OVERFLOW,
        segment_idx=0,
    )
    assert spliter._make_thresholds().force_split_at == 45
    assert _pending_group_lengths(spliter) == [45, 45, 4]

    packet2_group0 = spliter.on_segment_done(0)
    assert len(_segment_text(packet2_group0, 1)) == 45
    assert spliter._drivers[1].thresholds.force_split_at == 45
    packet2_group1 = spliter.on_segment_done(1)
    assert len(_segment_text(packet2_group1, 2)) == 45
    assert spliter._drivers[2].thresholds.force_split_at == 45
    packet2_group2 = spliter.on_segment_done(2)
    assert len(_segment_text(packet2_group2, 3)) == 4


def test_replan_preserves_forced_commitment_boundaries() -> None:
    spliter = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=2.0,
        safety_ratio_initial=2.0,
        ema_overflow_alpha=1.0,
        safety_failure_multiplier=1.0,
        max_concurrent=1,
    )
    spliter.push_group_tokens([(token, "a") for token in range(50)])
    spliter.push_group_tokens(
        [(token, "b") for token in range(94)],
        force_boundary_before=True,
        force_boundary=True,
    )
    assert spliter._pending[0].boundary_before is True
    assert spliter._pending[-1].forced_boundary is True

    spliter.observe_segment(
        105,
        50,
        outcome=RatioOutcome.KV_OVERFLOW,
        segment_idx=0,
    )

    assert spliter._pending[0].boundary_before is True
    assert spliter._pending[-1].forced_boundary is True


def test_reset_restores_ratio_state_and_clears_plan_snapshots() -> None:
    spliter = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=4.5,
        safety_ratio_initial=5.5,
        ema_overflow_alpha=1.0,
    )
    spliter.enable_decision_recording(True)
    spliter.set_full_text([(token, "x") for token in range(100)])
    spliter.observe_segment(
        400,
        40,
        outcome=RatioOutcome.KV_OVERFLOW,
        segment_idx=0,
    )
    assert spliter.safety_ratio == pytest.approx(10.0)

    spliter.reset()

    assert spliter.ema_ratio == pytest.approx(4.5)
    assert spliter.safety_ratio == pytest.approx(5.5)
    assert spliter._seg_ema_ratio == {}
    assert spliter._seg_safety_ratio == {}
    assert list(spliter._pending) == []
    assert spliter.drain_split_decisions() == []

    actions = spliter.set_full_text([(token, "x") for token in range(100)])
    assert len(_segment_text(actions, 0)) == 18
    assert spliter._drivers[0].thresholds.force_split_at == 18
