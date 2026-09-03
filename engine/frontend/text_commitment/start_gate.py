"""Semantic start/release gate, separate from guarded delivery."""
from __future__ import annotations

import time
from collections import deque
from typing import Callable, Iterable


class SemanticStartGate:
    """Hold the initial audio tail while a high-ambiguity span resolves.

    The gate is intentionally one-way: once the first effective chunk is
    released, subsequent chunks pass through immediately.  Unlike
    ``DeliveryHoldWindow`` it is not a hallucination rollback buffer; its only
    job is to hide a short semantic wait at the beginning of playback.
    """

    def __init__(
        self,
        *,
        bytes_per_sec: float,
        min_audio_ms: float = 160.0,
        max_hold_ms: float = 300.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bytes_per_sec = max(float(bytes_per_sec), 1.0)
        self.min_audio_ms = max(float(min_audio_ms), 0.0)
        self.max_hold_ms = max(float(max_hold_ms), 0.0)
        self._time = time_fn
        self._started_at: float | None = None
        self._held: deque[bytes] = deque()
        self._held_bytes = 0
        self._released = False

    @property
    def held_bytes(self) -> int:
        return self._held_bytes

    @property
    def released(self) -> bool:
        return self._released

    @property
    def held_audio_ms(self) -> float:
        return self._held_bytes / self.bytes_per_sec * 1000.0

    def push(self, chunks: Iterable[bytes]) -> None:
        for chunk in chunks:
            if not chunk:
                continue
            if self._started_at is None:
                self._started_at = self._time()
            if self._released:
                continue
            self._held.append(chunk)
            self._held_bytes += len(chunk)

    def release_due(self, *, semantic_pending: bool, safe_wait_ms: float = 0.0) -> list[bytes]:
        if not self._held:
            return []
        if self._released:
            return self._drain()
        now = self._time()
        elapsed_ms = 0.0 if self._started_at is None else (now - self._started_at) * 1000.0
        timed_out = elapsed_ms >= self.max_hold_ms
        enough_audio = self.held_audio_ms >= self.min_audio_ms
        enough_credit = safe_wait_ms >= min(self.max_hold_ms, self.held_audio_ms)
        if timed_out or (not semantic_pending and (enough_audio or enough_credit)):
            self._released = True
            return self._drain()
        return []

    def flush(self) -> list[bytes]:
        self._released = True
        return self._drain()

    def discard_segment(self, segment_idx: int, *, discard_all: bool = False, discard_bytes: int = 0) -> int:
        """Discard buffered chunks belonging to an aborted segment."""
        if not self._held:
            return 0
        items = list(self._held)
        dropped = 0
        remaining = max(int(discard_bytes), 0)
        kept: list[bytes] = []
        for chunk in reversed(items):
            if getattr(chunk, "segment_idx", None) != segment_idx:
                kept.append(chunk)
                continue
            if discard_all or remaining > 0:
                dropped += 1
                remaining -= len(chunk)
            else:
                kept.append(chunk)
        kept.reverse()
        self._held = deque(kept)
        self._held_bytes = sum(len(item) for item in kept)
        return dropped

    def _drain(self) -> list[bytes]:
        out = list(self._held)
        self._held.clear()
        self._held_bytes = 0
        return out
