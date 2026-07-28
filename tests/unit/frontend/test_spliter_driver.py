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


def test_ema_feedback_is_frozen_for_open_segment_and_used_by_next_segment() -> None:
    spliter = Spliter(
        engine_max_decode_len=120,
        prefill_len=12,
        safety_margin=8,
        ema_ratio=2.0,
        ema_alpha=1.0,
        max_concurrent=2,
    )
    spliter.feed_tokens([(0, "a")])
    old_thresholds = spliter._drivers[0].thresholds

    spliter.update_ratio(actual_audio_steps=100, actual_text_tokens=1)

    assert spliter._drivers[0].thresholds == old_thresholds
    assert spliter._make_thresholds().force_split_at < old_thresholds.force_split_at

    # Fill segment 0 exactly to its frozen cap and leave one token for a newly
    # opened segment.  That segment must use the updated EMA snapshot.
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
        ema_alpha=1.0,
    )
    spliter.feed_auto([(0, "a")])
    frozen_capacity = spliter._drivers[0].thresholds.force_split_at
    spliter.update_ratio(actual_audio_steps=100, actual_text_tokens=1)
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
        ema_alpha=1.0,
    )
    spliter.enable_decision_recording(True)
    spliter.feed_tokens([(0, "a")])
    frozen = spliter._drivers[0].thresholds

    spliter.update_ratio(actual_audio_steps=100, actual_text_tokens=1)
    spliter.feed_tokens([(token, "a") for token in range(1, frozen.force_split_at)])

    transition = next(
        record
        for record in spliter.drain_split_decisions()
        if record["obs"] == "driver_transition"
    )
    assert transition["ema_ratio"] == 2.0
    assert transition["thresholds"] == {
        "min_tokens_l1": frozen.min_tokens_l1,
        "force_split_at": frozen.force_split_at,
    }
