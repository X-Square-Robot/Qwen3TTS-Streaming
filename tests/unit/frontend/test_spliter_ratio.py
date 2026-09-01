from __future__ import annotations

import pytest

from engine.frontend.spliter.ratio import (
    RatioOutcome,
    SplitRatioController,
)


def _controller(**overrides) -> SplitRatioController:
    kwargs = {
        "duration_initial": 4.5,
        "safety_initial": 5.5,
        "duration_alpha": 0.1,
        "failure_alpha": 0.5,
        "min_ratio": 2.0,
        "max_ratio": 10.0,
        "min_duration_tokens": 8,
        "failure_multiplier": 1.25,
    }
    kwargs.update(overrides)
    return SplitRatioController(**kwargs)


def test_clean_duration_sample_can_fall_without_weakening_safety() -> None:
    ratios = _controller(duration_alpha=1.0)

    observation = ratios.observe(
        audio_steps=32,
        text_tokens=8,
        outcome=RatioOutcome.CODEC_EOS,
    )

    assert observation.duration_sample_accepted is True
    assert ratios.duration_ratio == pytest.approx(4.0)
    assert ratios.safety_ratio == pytest.approx(5.5)
    assert observation.safety_tightened is False


def test_slow_clean_sample_tightens_safety_without_replacing_duration_ema() -> None:
    ratios = _controller(duration_alpha=0.1)

    observation = ratios.observe(
        audio_steps=56,
        text_tokens=8,
        outcome=RatioOutcome.CODEC_EOS,
    )

    assert ratios.duration_ratio == pytest.approx(4.75)
    assert ratios.safety_ratio == pytest.approx(7.0)
    assert observation.duration_sample_accepted is True
    assert observation.safety_tightened is True


def test_tiny_clean_fragment_cannot_pollute_duration_or_safety() -> None:
    ratios = _controller(duration_alpha=1.0)

    observation = ratios.observe(
        audio_steps=37,
        text_tokens=1,
        outcome=RatioOutcome.CODEC_EOS,
    )

    assert observation.observed_ratio == pytest.approx(37.0)
    assert observation.duration_sample_accepted is False
    assert ratios.duration_ratio == pytest.approx(4.5)
    assert ratios.safety_ratio == pytest.approx(5.5)


@pytest.mark.parametrize(
    "outcome",
    [
        RatioOutcome.KV_OVERFLOW,
        RatioOutcome.LOOP_ABORT,
        RatioOutcome.LENGTH_ABORT,
    ],
)
def test_censored_or_runaway_failure_tightens_only_safety(
    outcome: RatioOutcome,
) -> None:
    ratios = _controller()

    observation = ratios.observe(
        audio_steps=504,
        text_tokens=108,
        outcome=outcome,
        segment_safety_ratio=5.5,
    )

    # target=5.5*1.25=6.875; alpha=.5 -> 6.1875.  The failure
    # observation must not enter the duration EMA.
    assert ratios.duration_ratio == pytest.approx(4.5)
    assert ratios.safety_ratio == pytest.approx(6.1875)
    assert observation.safety_tightened is True


@pytest.mark.parametrize(
    "outcome",
    [RatioOutcome.LOOP_ABORT, RatioOutcome.LENGTH_ABORT],
)
def test_runaway_failure_does_not_use_hallucination_inflated_ratio(
    outcome: RatioOutcome,
) -> None:
    ratios = _controller()

    observation = ratios.observe(
        audio_steps=300,
        text_tokens=8,
        outcome=outcome,
        segment_safety_ratio=5.5,
    )

    assert observation.observed_ratio == pytest.approx(37.5)
    assert ratios.duration_ratio == pytest.approx(4.5)
    assert ratios.safety_ratio == pytest.approx(6.1875)


def test_overflow_backoff_never_stays_below_censored_lower_bound() -> None:
    ratios = _controller()

    ratios.observe(
        audio_steps=72,
        text_tokens=8,
        outcome=RatioOutcome.KV_OVERFLOW,
        segment_safety_ratio=5.5,
    )

    # EMA-like backoff alone would be 7.25; 9.0 is an observed lower bound.
    assert ratios.safety_ratio == pytest.approx(9.0)


def test_concurrent_failures_from_same_snapshot_do_not_compound_backoff() -> None:
    ratios = _controller()

    for _ in range(2):
        ratios.observe(
            audio_steps=504,
            text_tokens=108,
            outcome=RatioOutcome.KV_OVERFLOW,
            segment_safety_ratio=5.5,
        )

    assert ratios.safety_ratio == pytest.approx(6.1875)


def test_silence_and_invalid_failures_are_ignored() -> None:
    ratios = _controller()

    observation = ratios.observe(
        audio_steps=50,
        text_tokens=20,
        outcome=RatioOutcome.IGNORED_FAILURE,
        segment_safety_ratio=5.5,
    )

    assert observation.duration_sample_accepted is False
    assert observation.safety_tightened is False
    assert ratios.duration_ratio == pytest.approx(4.5)
    assert ratios.safety_ratio == pytest.approx(5.5)


def test_external_duration_update_never_weakens_safety() -> None:
    ratios = _controller()

    ratios.set_duration_ratio(3.5)
    assert ratios.duration_ratio == pytest.approx(3.5)
    assert ratios.safety_ratio == pytest.approx(5.5)

    ratios.set_duration_ratio(7.0)
    assert ratios.duration_ratio == pytest.approx(7.0)
    assert ratios.safety_ratio == pytest.approx(7.0)


def test_safety_initial_above_configured_max_fails_fast() -> None:
    with pytest.raises(ValueError, match="safety_initial"):
        _controller(safety_initial=5.5, max_ratio=5.0)


def test_reason_mapping_is_typed_and_conservative() -> None:
    assert RatioOutcome.from_eos_reason("codec_eos") is RatioOutcome.CODEC_EOS
    assert RatioOutcome.from_eos_reason("kv_overflow") is RatioOutcome.KV_OVERFLOW
    assert RatioOutcome.from_eos_reason("", overflow=True) is RatioOutcome.KV_OVERFLOW
    assert RatioOutcome.from_eos_reason("loop_abort") is RatioOutcome.LOOP_ABORT
    assert RatioOutcome.from_eos_reason("length_abort") is RatioOutcome.LENGTH_ABORT
    assert (
        RatioOutcome.from_eos_reason("silent_audio_abort")
        is RatioOutcome.IGNORED_FAILURE
    )
    assert RatioOutcome.from_retry_reason("loop") is RatioOutcome.LOOP_ABORT
    assert RatioOutcome.from_retry_reason("length") is RatioOutcome.LENGTH_ABORT
    assert (
        RatioOutcome.from_retry_reason("silence")
        is RatioOutcome.IGNORED_FAILURE
    )
