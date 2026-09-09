"""Estimator-independent codec/token/segment/spoken/raw coordinates.

The estimator supplies a segment-local codec interval and token frontier.
Everything after that boundary is provenance lookup, never a length ratio.
PCM delivery coordinates are assigned later by the audio reorder/output layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

from .text_journal import CanonicalTextJournal


@dataclass(frozen=True, slots=True)
class CodecTokenProgress:
    segment_idx: int
    source_frame_start: int
    source_frame_end: int
    token_end: int

    def __post_init__(self) -> None:
        values = (self.segment_idx, self.source_frame_start,
                  self.source_frame_end, self.token_end)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
            raise ValueError("codec/token coordinates must be nonnegative integers")
        if self.source_frame_end < self.source_frame_start:
            raise ValueError("codec interval is reversed")


class SegmentTextCoordinates:
    """View of tokenizer provenance; offsets are session-global codepoints."""

    def __init__(self, spans: Sequence[Mapping[str, int]],
                 journal: CanonicalTextJournal | None) -> None:
        if not spans:
            raise ValueError("token provenance is required")
        self.spans = spans
        self.journal = journal

    def token_end_at(self, normalized_end: int) -> int:
        # Stop at the first uncompleted token, including overlapping Unicode
        # byte-piece spans. A later token never confirms an earlier gap.
        for index, span in enumerate(self.spans):
            if span["normalized_end"] > normalized_end:
                return index
        return len(self.spans)

    def token_end_from_labels(
        self, mu: float, label_spans: Sequence[tuple[int, int]],
    ) -> int:
        position = float(mu)
        if not math.isfinite(position):
            raise ValueError("cursor mu must be finite")
        completed = min(len(label_spans), max(0, math.floor(position)))
        if completed == 0:
            return 0
        return self.token_end_at(label_spans[completed - 1][1])

    def boundary(self, token_end: int) -> tuple[int, int]:
        if not 0 <= token_end <= len(self.spans):
            raise ValueError("token frontier exceeds segment provenance")
        span = self.spans[token_end - 1] if token_end else self.spans[0]
        edge = "end" if token_end else "start"
        normalized = span[f"normalized_{edge}"]
        if self.journal is not None:
            raw = self.journal.raw_span(normalized, normalized)[0]
        else:
            raw = span[f"raw_{edge}"]
        return normalized, raw


class TextProgressProjection:
    """Shared high-water state across estimator changes for one segment.

    Segment state stays independent while GPU batches interleave. The existing
    audio reorder publishes these attributed events in delivery order.
    """

    def __init__(self, segment_idx: int) -> None:
        self.segment_idx = segment_idx
        self.token_end = 0
        self._normalized_end: int | None = None
        self._raw_end: int | None = None
        self._frame_end = 0
        self.basis: str | None = None

    def project(self, estimate: CodecTokenProgress,
                coordinates: SegmentTextCoordinates) -> dict[str, str]:
        if estimate.segment_idx != self.segment_idx:
            raise ValueError("progress belongs to another segment")
        token_end = max(self.token_end, estimate.token_end)
        normalized_end, raw_end = coordinates.boundary(token_end)
        base_normalized, base_raw = coordinates.boundary(0)
        normalized_start = (base_normalized if self._normalized_end is None
                            else self._normalized_end)
        raw_start = base_raw if self._raw_end is None else self._raw_end
        normalized_end = max(normalized_start, normalized_end)
        raw_end = max(raw_start, raw_end)
        frame_start = max(self._frame_end, estimate.source_frame_start)
        frame_end = max(frame_start, estimate.source_frame_end)
        result = {
            "source_frame_start": str(frame_start),
            "source_frame_end": str(frame_end),
            "text_token_start": str(self.token_end),
            "text_token_end": str(token_end),
            "text_token_count": str(len(coordinates.spans)),
            "normalized_codepoint_start": str(normalized_start),
            "normalized_codepoint_end": str(normalized_end),
            "raw_codepoint_start": str(raw_start),
            "raw_codepoint_end": str(raw_end),
        }
        self.token_end = token_end
        self._normalized_end, self._raw_end = normalized_end, raw_end
        self._frame_end = frame_end
        return result
