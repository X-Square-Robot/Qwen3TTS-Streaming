"""Playback-clock based text progress for all client transports."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Iterable

from qwen3tts_protocol import AudioChunk, StreamEvent

from .exceptions import ProtocolError


def _int(meta: dict[str, str], key: str, default: int | None = None) -> int | None:
    try:
        value = meta.get(key)
        return default if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return default


def _float(meta: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default) or default)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TextProgressAnchor:
    anchor_seq: int
    output_sample_start: int
    output_sample_end: int
    output_sample_rate: int
    raw_codepoint_start: int
    raw_codepoint_end: int
    normalized_codepoint_start: int
    normalized_codepoint_end: int
    text_input_final: bool = False
    alignment_final: bool = False
    segment_id: int = -1
    progress: float = 0.0
    basis: str = ""
    quality: str = "rough"

    @classmethod
    def from_meta(cls, meta: dict[str, str], *, segment_id: int = -1) -> "TextProgressAnchor | None":
        seq = _int(meta, "anchor_seq")
        start = _int(meta, "output_sample_start")
        end = _int(meta, "output_sample_end")
        rate = _int(meta, "output_sample_rate", 24000)
        if seq is None or start is None or end is None or rate is None:
            return None
        def span(prefix: str) -> tuple[int, int]:
            span_start = _int(meta, f"{prefix}_start", 0) or 0
            span_end = _int(meta, f"{prefix}_end", 0) or 0
            return span_start, span_end

        raw_start, raw_end = span("raw_codepoint")
        normalized_start, normalized_end = span("normalized_codepoint")
        return cls(
            anchor_seq=seq,
            output_sample_start=start,
            output_sample_end=end,
            output_sample_rate=rate,
            raw_codepoint_start=raw_start,
            raw_codepoint_end=raw_end,
            normalized_codepoint_start=normalized_start,
            normalized_codepoint_end=normalized_end,
            text_input_final=meta.get("text_input_final", "false").lower() == "true",
            alignment_final=(
                meta.get("alignment_final", meta.get("progress_final", "false")).lower()
                == "true"
            ),
            segment_id=segment_id,
            progress=max(0.0, min(1.0, _float(meta, "text_progress"))),
            basis=str(meta.get("progress_basis", "") or ""),
            quality=str(meta.get("progress_quality", "rough") or "rough"),
        )


@dataclass(frozen=True)
class TextCursor:
    raw_codepoint: int = 0
    normalized_codepoint: int = 0
    anchor_seq: int = 0
    available: bool = False


@dataclass(frozen=True)
class PlaybackTextProgress:
    played_through_sample: int = 0
    buffered_through_sample: int = 0
    confirmed: TextCursor = TextCursor()
    estimated: TextCursor = TextCursor()
    playback_complete: bool = False
    available: bool = False


class PlaybackProgressTracker:
    """Thread-safe monotonic sample-to-text cursor.

    ``confirmed`` never interpolates. ``estimated`` interpolates only between
    received anchors and deliberately does not extrapolate past the newest
    anchor, because a synthesis pause is not evidence of playback progress.
    """

    def __init__(self, anchors: Iterable[TextProgressAnchor] = ()) -> None:
        self._lock = threading.RLock()
        self._anchors: dict[int, TextProgressAnchor] = {}
        self._received_sample = 0
        self._played_sample = 0
        self._buffered_sample = 0
        self._terminal = False
        self._final_sample: int | None = None
        self._output_sample_rate: int | None = None
        self._latest = PlaybackTextProgress()
        for anchor in anchors:
            self.add_anchor(anchor)

    @property
    def latest(self) -> PlaybackTextProgress:
        with self._lock:
            return self._latest

    @property
    def anchors(self) -> tuple[TextProgressAnchor, ...]:
        with self._lock:
            return tuple(sorted(self._anchors.values(), key=lambda item: (item.output_sample_end, item.anchor_seq)))

    def observe(self, message: StreamEvent | AudioChunk) -> PlaybackTextProgress:
        with self._lock:
            if isinstance(message, AudioChunk):
                start = message.output_sample_start
                end = message.output_sample_end
                if start is None:
                    start = self._received_sample
                if end is None:
                    channels = max(1, message.audio.channels)
                    width = {"pcm_f32": 4, "pcm_s16le": 2, "pcm_s16": 2, "pcm_u8": 1}.get(
                        message.audio.encoding.lower(), 4
                    )
                    end = start + len(message.pcm_bytes) // (width * channels)
                self._received_sample = max(self._received_sample, end)
                if message.output_sample_start is not None:
                    start = int(message.output_sample_start)
                if message.output_sample_end is not None:
                    end = int(message.output_sample_end)
                if start < 0 or end < start:
                    raise ProtocolError("audio output sample range is invalid")
                rate = int(message.audio.sample_rate or 0)
                if rate > 0:
                    self._remember_sample_rate_locked(rate)
                self._received_sample = max(self._received_sample, end)
                self._add_meta_anchor(message.meta, -1, require_received=False)
            else:
                self._add_meta_anchor(message.meta, message.segment_id, require_received=True)
                if message.type in {"done", "error"}:
                    self._terminal = True
                    self._final_sample = _int(message.meta, "final_output_sample", self._received_sample)
            return self._recompute_locked()

    def add_anchor(self, anchor: TextProgressAnchor) -> PlaybackTextProgress:
        with self._lock:
            self._validate_anchor_locked(anchor)
            old = self._anchors.get(anchor.anchor_seq)
            if old is not None:
                if old == anchor:
                    return self._recompute_locked()
                raise ProtocolError(
                    f"conflicting replay for text progress anchor {anchor.anchor_seq}"
                )
            self._anchors[anchor.anchor_seq] = anchor
            self._received_sample = max(self._received_sample, anchor.output_sample_end)
            return self._recompute_locked()

    def update_playback_progress(
        self,
        *,
        played_through_sample: int,
        buffered_through_sample: int | None = None,
    ) -> PlaybackTextProgress:
        with self._lock:
            requested_played = int(played_through_sample)
            if requested_played < self._played_sample:
                raise ProtocolError("played playback sample moved backwards")
            played = requested_played
            buffered = (
                max(self._buffered_sample, played)
                if buffered_through_sample is None
                else int(buffered_through_sample)
            )
            if played > self._received_sample or buffered > self._received_sample:
                raise ValueError("playback sample cannot exceed received output")
            if buffered < played:
                raise ValueError("buffered sample cannot be before played sample")
            self._played_sample = played
            if buffered < self._buffered_sample:
                raise ProtocolError("buffered playback sample moved backwards")
            self._buffered_sample = buffered
            return self._recompute_locked()

    def _add_meta_anchor(
        self,
        meta: dict[str, str],
        segment_id: int,
        *,
        require_received: bool,
    ) -> None:
        anchor = TextProgressAnchor.from_meta(meta, segment_id=segment_id)
        if anchor is not None:
            if require_received and anchor.output_sample_end > self._received_sample:
                raise ProtocolError(
                    "text progress anchor is ahead of received output audio"
                )
            self.add_anchor(anchor)

    def _remember_sample_rate_locked(self, sample_rate: int) -> None:
        if self._output_sample_rate is None:
            self._output_sample_rate = sample_rate
            return
        if self._output_sample_rate != sample_rate:
            raise ProtocolError("output sample rate changed within a playback stream")

    def _validate_anchor_locked(self, anchor: TextProgressAnchor) -> None:
        if (
            anchor.anchor_seq <= 0
            or anchor.output_sample_start < 0
            or anchor.output_sample_end < anchor.output_sample_start
            or anchor.output_sample_rate <= 0
            or anchor.raw_codepoint_start < 0
            or anchor.raw_codepoint_end < anchor.raw_codepoint_start
            or anchor.normalized_codepoint_start < 0
            or anchor.normalized_codepoint_end < anchor.normalized_codepoint_start
        ):
            raise ProtocolError("malformed text progress anchor")
        self._remember_sample_rate_locked(anchor.output_sample_rate)
        for old in self._anchors.values():
            if old.anchor_seq == anchor.anchor_seq:
                continue
            if anchor.anchor_seq < old.anchor_seq:
                raise ProtocolError("text progress anchor sequence moved backwards")
            if anchor.output_sample_start < old.output_sample_start:
                raise ProtocolError("text progress sample range moved backwards")
            if anchor.output_sample_end < old.output_sample_end:
                raise ProtocolError("text progress sample end moved backwards")
            if anchor.raw_codepoint_start < old.raw_codepoint_start:
                raise ProtocolError("text progress raw range moved backwards")
            if anchor.raw_codepoint_end < old.raw_codepoint_end:
                raise ProtocolError("text progress raw range moved backwards")
            if anchor.normalized_codepoint_start < old.normalized_codepoint_start:
                raise ProtocolError("text progress normalized range moved backwards")
            if anchor.normalized_codepoint_end < old.normalized_codepoint_end:
                raise ProtocolError("text progress normalized range moved backwards")

    def _recompute_locked(self) -> PlaybackTextProgress:
        ordered = sorted(self._anchors.values(), key=lambda item: (item.output_sample_end, item.anchor_seq))
        confirmed = TextCursor()
        previous = None
        estimated = TextCursor()
        for anchor in ordered:
            if self._played_sample >= anchor.output_sample_end:
                confirmed = TextCursor(anchor.raw_codepoint_end, anchor.normalized_codepoint_end, anchor.anchor_seq, True)
            if previous is None and anchor.output_sample_start <= self._played_sample <= anchor.output_sample_end:
                width = anchor.output_sample_end - anchor.output_sample_start
                ratio = (
                    1.0
                    if width <= 0
                    else (self._played_sample - anchor.output_sample_start) / width
                )
                estimated = TextCursor(
                    round(anchor.raw_codepoint_start + ratio * (anchor.raw_codepoint_end - anchor.raw_codepoint_start)),
                    round(anchor.normalized_codepoint_start + ratio * (anchor.normalized_codepoint_end - anchor.normalized_codepoint_start)),
                    anchor.anchor_seq,
                    True,
                )
            elif (
                previous is not None
                and previous.output_sample_end <= self._played_sample <= anchor.output_sample_end
            ):
                width = anchor.output_sample_end - previous.output_sample_end
                ratio = (
                    1.0
                    if width <= 0
                    else (self._played_sample - previous.output_sample_end) / width
                )
                estimated = TextCursor(
                    round(previous.raw_codepoint_end + ratio * (anchor.raw_codepoint_end - previous.raw_codepoint_end)),
                    round(previous.normalized_codepoint_end + ratio * (anchor.normalized_codepoint_end - previous.normalized_codepoint_end)),
                    anchor.anchor_seq,
                    True,
                )
            previous = anchor
        if not estimated.available and confirmed.available:
            estimated = confirmed
        self._latest = PlaybackTextProgress(
            played_through_sample=self._played_sample,
            buffered_through_sample=self._buffered_sample,
            confirmed=confirmed,
            estimated=estimated,
            playback_complete=bool(
                self._terminal and self._final_sample is not None and self._played_sample >= self._final_sample
            ),
            available=bool(ordered),
        )
        return self._latest


__all__ = (
    "PlaybackProgressTracker",
    "PlaybackTextProgress",
    "TextCursor",
    "TextProgressAnchor",
)
