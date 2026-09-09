from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from engine.core.native_cursor import CursorLabelPlan, CursorOwnerSpan
from engine.core.text_progress import (
    NATIVE_CURSOR_PROGRESS_BASIS,
    NativeCursorProgressProjector,
)
from engine.frontend.interface import FrontendInterface
from engine.backend.engine_loop import EngineLoop


def _plan(revision: int = 1) -> CursorLabelPlan:
    return CursorLabelPlan(
        label_ids=(11, 12, 21, 22, 23),
        owner_spans=(
            CursorOwnerSpan(
                owner_id=1,
                label_start=0,
                label_end=2,
                normalized_start=0,
                normalized_end=2,
                raw_start=0,
                raw_end=3,
            ),
            CursorOwnerSpan(
                owner_id=2,
                label_start=2,
                label_end=5,
                normalized_start=2,
                normalized_end=5,
                raw_start=3,
                raw_end=6,
            ),
        ),
        revision=revision,
    )


def test_invalid_lookahead_does_not_publish_native_progress() -> None:
    projector = NativeCursorProgressProjector(segment_idx=4, plan=_plan())

    assert projector.update(mu=0.0, valid=False) is None
    assert projector.last_normalized_end == 0
    assert projector.last_raw_end == 0


def test_owner_boundary_is_conservative_and_display_is_interpolated() -> None:
    projector = NativeCursorProgressProjector(segment_idx=4, plan=_plan())

    inside = projector.update(
        mu=1.25,
        valid=True,
        confidence=0.8,
        source_frame_start=2,
        source_frame_end=3,
    )
    assert inside is not None
    assert inside.normalized_codepoint_end == 0
    assert inside.raw_codepoint_end == 0
    assert inside.display_normalized_position == pytest.approx(1.25)
    assert inside.display_raw_position == pytest.approx(1.875)

    crossed = projector.update(mu=2.0, valid=True, confidence=0.9)
    assert crossed is not None
    assert crossed.normalized_codepoint_end == 2
    assert crossed.raw_codepoint_end == 3
    assert crossed.normalized_codepoint_start == 0
    assert crossed.progress == pytest.approx(2 / 5)
    assert crossed.to_meta()["progress_basis"] == NATIVE_CURSOR_PROGRESS_BASIS


def test_high_water_never_recedes_on_mu_regression_or_tail_revision() -> None:
    projector = NativeCursorProgressProjector(segment_idx=0, plan=_plan())
    first = projector.update(mu=2.1, valid=True)
    assert first is not None
    assert first.normalized_codepoint_end == 2

    regressed = projector.update(mu=0.1, valid=True)
    assert regressed is not None
    assert regressed.normalized_codepoint_end == 2
    assert regressed.raw_codepoint_end == 3

    projector.update_plan(_plan(revision=2))
    final = projector.update(valid=True, final=True)
    assert final is not None
    assert final.normalized_codepoint_end == 5
    assert final.raw_codepoint_end == 6
    assert final.final is True


def test_final_without_valid_mu_completes_only_an_active_plan() -> None:
    projector = NativeCursorProgressProjector(segment_idx=1, plan=_plan())

    result = projector.update(valid=False, final=True)

    assert result is not None
    assert result.normalized_codepoint_start == 0
    assert result.normalized_codepoint_end == 5
    assert result.raw_codepoint_end == 6


def test_frontend_publishes_native_event_and_falls_back_on_lookahead() -> None:
    session = SimpleNamespace(
        spliter=SimpleNamespace(ema_ratio_for_segment=lambda _segment_idx: 4.5),
        segment_progress_frames={},
        text_progress_estimators={},
        segment_token_spans={
            1: [
                {
                    "normalized_start": 0,
                    "normalized_end": 5,
                    "raw_start": 0,
                    "raw_end": 5,
                }
            ]
        },
        segment_token_emitted_count={},
        text_journal=None,
        input_complete=False,
        cursor_label_plan=_plan(),
        native_cursor_projectors={},
        session_id="test",
    )

    native = FrontendInterface._make_text_progress_event(
        None,
        session,
        1,
        {
            "source_frame_end": 3,
            "cursor_plan_revision": 1,
            "cursor_valid": 1,
            "cursor_mu": 2.2,
            "cursor_confidence": 0.9,
        },
    )
    assert native is not None
    assert native["meta"]["progress_basis"] == NATIVE_CURSOR_PROGRESS_BASIS
    # The shared contract confirms complete tokenizer spans, not a legacy
    # native-owner boundary inside this single five-character BPE token.
    assert native["meta"]["normalized_codepoint_end"] == "0"

    fallback = FrontendInterface._make_text_progress_event(
        None,
        session,
        1,
        {
            "source_frame_end": 4,
            "text_tokens": 1,
            "cursor_plan_revision": 1,
            "cursor_valid": 0,
            "cursor_mu": 2.2,
            "cursor_confidence": 0.0,
        },
    )
    assert fallback is not None
    assert fallback["meta"]["progress_basis"] == "ema_frame_ratio_v1"


def test_engine_does_not_publish_cursor_metrics_after_continuity_downgrade() -> None:
    output = SimpleNamespace(
        cursor_outputs={
            "cursor_valid": torch.tensor([[1]]),
            "cursor_mu": torch.tensor([[2.0]]),
            "cursor_confidence": torch.tensor([[0.9]]),
        }
    )
    segment = SimpleNamespace(cursor_progress_disabled=True, cursor_plan_revision=1)

    assert EngineLoop._cursor_progress_metrics(output, 0, segment) == {}
