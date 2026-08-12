from __future__ import annotations

import pytest

from engine.core.text_progress import EmaTextProgressEstimator


def test_ema_progress_maps_frames_to_monotonic_token_end() -> None:
    estimator = EmaTextProgressEstimator(segment_idx=3, ema_ratio=5.5)

    first = estimator.update(
        source_frame_start=0,
        source_frame_end=11,
        text_token_count=4,
    )
    second = estimator.update(
        source_frame_start=11,
        source_frame_end=22,
        text_token_count=4,
    )

    assert first.segment_idx == 3
    assert first.text_token_end == 2
    assert second.text_token_start == 2
    assert second.text_token_end == 4
    assert second.progress == 1.0


def test_streaming_token_append_does_not_move_cursor_backwards() -> None:
    estimator = EmaTextProgressEstimator(segment_idx=0, ema_ratio=2.0)

    before_append = estimator.update(
        source_frame_start=0,
        source_frame_end=6,
        text_token_count=3,
    )
    after_append = estimator.update(
        source_frame_start=6,
        source_frame_end=7,
        text_token_count=8,
    )

    assert before_append.text_token_end == 3
    assert after_append.text_token_end >= before_append.text_token_end
    assert after_append.progress < before_append.progress


def test_final_progress_closes_known_text_prefix() -> None:
    estimator = EmaTextProgressEstimator(segment_idx=1, ema_ratio=10.0)

    result = estimator.update(
        source_frame_start=0,
        source_frame_end=2,
        text_token_count=5,
        final=True,
    )

    assert result.text_token_end == 5
    assert result.progress == 1.0
    assert result.to_meta()["progress_final"] == "true"


def test_invalid_ratio_is_rejected() -> None:
    with pytest.raises(ValueError):
        EmaTextProgressEstimator(segment_idx=0, ema_ratio=0.0)
