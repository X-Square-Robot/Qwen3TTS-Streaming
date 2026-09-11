"""Coarse monotonic text progress estimation for streaming TTS.

The first implementation deliberately uses the segment's frozen audio-frame /
text-token EMA.  It is an estimate of where playback is likely to be in the
source text, not an alignment claim.  Keeping this as a small pure component
lets a future ASR or codec-text reviewer replace the estimator without
changing transport or client progress contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .native_cursor import CursorLabelPlan, CursorOwnerSpan


TEXT_PROGRESS_BASIS = "ema_frame_ratio_v1"
NATIVE_CURSOR_PROGRESS_BASIS = "native_cursor_v1"


@dataclass(frozen=True)
class TextProgressEstimate:
    """A monotonic, segment-local text progress estimate.

    All ranges are half-open. ``text_token_end`` is therefore the number of
    source tokens estimated to have been spoken, not a zero-based token index.
    """

    segment_idx: int
    source_frame_start: int
    source_frame_end: int
    text_token_start: int
    text_token_end: int
    text_token_count: int
    expected_audio_frames: float
    ema_audio_frames_per_token: float
    progress: float
    basis: str = TEXT_PROGRESS_BASIS
    quality: str = "rough"
    final: bool = False

    def to_meta(self) -> dict[str, str]:
        """Serialize the stable wire metadata without transport dependencies."""

        return {
            "progress_basis": self.basis,
            "progress_quality": self.quality,
            "source_frame_start": str(self.source_frame_start),
            "source_frame_end": str(self.source_frame_end),
            "text_token_start": str(self.text_token_start),
            "text_token_end": str(self.text_token_end),
            "text_token_count": str(self.text_token_count),
            "expected_audio_frames": f"{self.expected_audio_frames:.3f}",
            "ema_audio_frames_per_token": f"{self.ema_audio_frames_per_token:.3f}",
            "text_progress": f"{self.progress:.6f}",
            "progress_final": "true" if self.final else "false",
        }


@dataclass(frozen=True)
class NativeCursorProgressEstimate:
    """Conservative projection of one fused-cursor output.

    ``normalized_codepoint_end`` and ``raw_codepoint_end`` are owner-level
    high-water marks.  The optional display coordinates are deliberately
    separate: they may interpolate inside an owner, but are not protocol
    confirmation boundaries.
    """

    segment_idx: int
    source_frame_start: int
    source_frame_end: int
    normalized_codepoint_start: int
    normalized_codepoint_end: int
    raw_codepoint_start: int
    raw_codepoint_end: int
    label_position: float
    confidence: float
    progress: float
    display_normalized_position: float | None = None
    display_raw_position: float | None = None
    final: bool = False
    basis: str = NATIVE_CURSOR_PROGRESS_BASIS
    quality: str = "native"

    def to_meta(self) -> dict[str, str]:
        return {
            "progress_basis": self.basis,
            "progress_quality": self.quality,
            "source_frame_start": str(self.source_frame_start),
            "source_frame_end": str(self.source_frame_end),
            "normalized_codepoint_start": str(self.normalized_codepoint_start),
            "normalized_codepoint_end": str(self.normalized_codepoint_end),
            "raw_codepoint_start": str(self.raw_codepoint_start),
            "raw_codepoint_end": str(self.raw_codepoint_end),
            "cursor_mu": f"{self.label_position:.6f}",
            "cursor_confidence": f"{self.confidence:.6f}",
            "text_progress": f"{self.progress:.6f}",
            "progress_final": "true" if self.final else "false",
            **(
                {
                    "display_normalized_position": f"{self.display_normalized_position:.6f}",
                    "display_raw_position": f"{self.display_raw_position:.6f}",
                }
                if self.display_normalized_position is not None
                and self.display_raw_position is not None
                else {}
            ),
        }


class NativeCursorProgressProjector:
    """Map fused ``mu`` values to owner-level text high-water marks.

    This is intentionally CPU-only.  It does not normalize text, infer raw
    coordinates by ratio, or run a second cursor model.  A plan revision may
    replace a mutable tail while the already-published high-water remains
    monotonic.
    """

    def __init__(self, *, segment_idx: int, plan: CursorLabelPlan) -> None:
        self.segment_idx = int(segment_idx)
        self.plan = plan
        self._last_normalized_end = 0
        self._last_raw_end = 0
        self._last_display_normalized = 0.0
        self._last_display_raw = 0.0
        self._last_mu = 0.0

    @property
    def last_normalized_end(self) -> int:
        return self._last_normalized_end

    @property
    def last_raw_end(self) -> int:
        return self._last_raw_end

    def update_plan(self, plan: CursorLabelPlan) -> None:
        if plan.revision < self.plan.revision:
            raise ValueError("native cursor plan revision cannot move backwards")
        self.plan = plan
        self._last_mu = min(self._last_mu, float(plan.label_count))

    def update(
        self,
        *,
        mu: float | None = None,
        valid: bool = True,
        confidence: float = 0.0,
        source_frame_start: int = 0,
        source_frame_end: int = 0,
        final: bool = False,
    ) -> NativeCursorProgressEstimate | None:
        if not valid and not final:
            return None
        if not self.plan.active:
            return None

        if mu is None:
            position = float(self._last_mu)
        else:
            try:
                position = float(mu)
            except (TypeError, ValueError) as exc:
                raise ValueError("cursor mu must be numeric") from exc
            if not math.isfinite(position):
                raise ValueError("cursor mu must be finite")
        position = max(0.0, min(float(self.plan.label_count), position))
        position = max(self._last_mu, position)
        if final:
            position = float(self.plan.label_count)
        self._last_mu = position

        confidence_value = float(confidence)
        if not math.isfinite(confidence_value):
            raise ValueError("cursor confidence must be finite")
        confidence_value = max(0.0, min(1.0, confidence_value))
        frame_start = max(0, int(source_frame_start))
        frame_end = max(frame_start, int(source_frame_end))

        completed = self._completed_owner_count(position)
        if completed:
            boundary_owner = self.plan.owner_spans[completed - 1]
            normalized_end = boundary_owner.normalized_end
            raw_end = boundary_owner.raw_end
        else:
            normalized_end = self._last_normalized_end
            raw_end = self._last_raw_end
        normalized_end = max(self._last_normalized_end, normalized_end)
        raw_end = max(self._last_raw_end, raw_end)
        normalized_start = self._last_normalized_end
        raw_start = self._last_raw_end
        self._last_normalized_end = normalized_end
        self._last_raw_end = raw_end

        owner = self._owner_at(position)
        display_normalized = None
        display_raw = None
        if owner is not None:
            width = owner.label_end - owner.label_start
            fraction = 1.0 if position >= owner.label_end else (
                max(0.0, position - owner.label_start) / width
            )
            display_normalized = owner.normalized_start + fraction * (
                owner.normalized_end - owner.normalized_start
            )
            display_raw = owner.raw_start + fraction * (owner.raw_end - owner.raw_start)
            # Display interpolation is UI-only, but it must not visibly move
            # backwards when a mutable tail is re-anchored or an owner closes.
            # The integer protocol high-water above remains authoritative.
            display_normalized = max(
                self._last_display_normalized, display_normalized
            )
            display_raw = max(self._last_display_raw, display_raw)
            self._last_display_normalized = display_normalized
            self._last_display_raw = display_raw

        normalized_total = max(
            self.plan.owner_spans[-1].normalized_end,
            normalized_end,
        )
        progress = normalized_end / normalized_total if normalized_total else 0.0
        return NativeCursorProgressEstimate(
            segment_idx=self.segment_idx,
            source_frame_start=frame_start,
            source_frame_end=frame_end,
            normalized_codepoint_start=normalized_start,
            normalized_codepoint_end=normalized_end,
            raw_codepoint_start=raw_start,
            raw_codepoint_end=raw_end,
            label_position=position,
            confidence=confidence_value,
            progress=max(0.0, min(1.0, progress)),
            display_normalized_position=display_normalized,
            display_raw_position=display_raw,
            final=bool(final),
        )

    def _completed_owner_count(self, position: float) -> int:
        if position >= self.plan.label_count:
            return len(self.plan.owner_spans)
        return sum(1 for owner in self.plan.owner_spans if position >= owner.label_end)

    def _owner_at(self, position: float) -> CursorOwnerSpan | None:
        for owner in self.plan.owner_spans:
            if position < owner.label_end:
                return owner
        return self.plan.owner_spans[-1] if self.plan.owner_spans else None


class EmaTextProgressEstimator:
    """Estimate text progress from emitted source audio frames.

    The estimator is intentionally monotonic.  Streaming input may append
    tokens after audio has already started; increasing the denominator must
    not move the visible cursor backwards.
    """

    def __init__(self, *, segment_idx: int, ema_ratio: float) -> None:
        if not math.isfinite(ema_ratio) or ema_ratio <= 0.0:
            raise ValueError("ema_ratio must be finite and positive")
        self.segment_idx = int(segment_idx)
        self.ema_ratio = float(ema_ratio)
        self._last_token_end = 0
        self._last_frame_end = 0

    @property
    def last_token_end(self) -> int:
        return self._last_token_end

    @property
    def last_frame_end(self) -> int:
        return self._last_frame_end

    def update(
        self,
        *,
        source_frame_start: int,
        source_frame_end: int,
        text_token_count: int,
        final: bool = False,
    ) -> TextProgressEstimate:
        frame_start = max(0, int(source_frame_start))
        frame_end = max(frame_start, int(source_frame_end))
        frame_end = max(frame_end, self._last_frame_end)
        token_count = max(0, int(text_token_count))
        expected_frames = self.ema_ratio * token_count

        if final:
            token_end = token_count
        elif token_count == 0:
            token_end = 0
        else:
            token_end = min(token_count, max(0, math.floor(frame_end / self.ema_ratio)))
        token_end = max(self._last_token_end, token_end)
        token_end = min(token_count, token_end)
        token_start = min(self._last_token_end, token_end)

        self._last_frame_end = frame_end
        self._last_token_end = token_end
        progress = token_end / token_count if token_count else 0.0
        return TextProgressEstimate(
            segment_idx=self.segment_idx,
            source_frame_start=frame_start,
            source_frame_end=frame_end,
            text_token_start=token_start,
            text_token_end=token_end,
            text_token_count=token_count,
            expected_audio_frames=expected_frames,
            ema_audio_frames_per_token=self.ema_ratio,
            progress=max(0.0, min(1.0, progress)),
            final=bool(final),
        )


__all__ = [
    "EmaTextProgressEstimator",
    "NativeCursorProgressEstimate",
    "NativeCursorProgressProjector",
    "NATIVE_CURSOR_PROGRESS_BASIS",
    "TEXT_PROGRESS_BASIS",
    "TextProgressEstimate",
]
