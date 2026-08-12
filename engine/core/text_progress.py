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


TEXT_PROGRESS_BASIS = "ema_frame_ratio_v1"


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
    "TEXT_PROGRESS_BASIS",
    "TextProgressEstimate",
]
