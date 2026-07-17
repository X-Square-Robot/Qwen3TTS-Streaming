"""Confidence-gated delivery window ("guarded delivery").

The engine synthesizes ~2x faster than realtime, so under plain firehose
delivery a hallucinated tail reaches the client long before playback gets
there — and once sent it cannot be recalled. This window keeps the
synthesized-ahead excess inside the server, where segment verdicts can still
act on it:

- while a segment is in flight, release only up to (wall time elapsed since
  the first release + ``window_sec``) worth of audio; hold the rest;
- codec EOS validates the tail → ``flush()`` everything held (burst; the
  final chunk leaves at the same instant firehose would have emitted it,
  because the held frames were synthesized before EOS anyway);
- loop/silence abort → the held tail is hallucination garbage; ``discard()``.

The playhead estimate needs no client feedback: a client cannot play faster
than wall clock from the first byte it received, so elapsed time
upper-bounds its playhead and (elapsed + window) upper-bounds what it can
have played plus buffered. Release is whole-chunk granular (one engine chunk
is one 80ms frame), so the window overshoots by at most one chunk.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Iterable, List


class DeliveryHoldWindow:
    """Session-scoped hold buffer between the reorder stage and the gateway.

    Chunks must be pushed in final (post-reorder) delivery order; the window
    preserves it. Byte counts are session-cumulative, matching a client that
    plays the session stream continuously.
    """

    __slots__ = (
        "_window_sec",
        "_bytes_per_sec",
        "_time",
        "_clock_start",
        "_released_bytes",
        "_held",
        "_held_bytes",
    )

    def __init__(
        self,
        window_sec: float,
        bytes_per_sec: float,
        *,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window_sec = float(window_sec)
        self._bytes_per_sec = float(bytes_per_sec)
        self._time = time_fn
        self._clock_start: float | None = None
        self._released_bytes: int = 0
        self._held: deque[bytes] = deque()
        self._held_bytes: int = 0

    @property
    def held_bytes(self) -> int:
        return self._held_bytes

    @property
    def held_chunks(self) -> int:
        return len(self._held)

    def push(self, chunks: Iterable[bytes]) -> None:
        """Add in-order chunks to the window (no release; call release_due)."""
        for chunk in chunks:
            if chunk:
                self._held.append(chunk)
                self._held_bytes += len(chunk)

    def release_due(self) -> List[bytes]:
        """Release chunks the client's playback window is entitled to."""
        if not self._held:
            return []
        now = self._time()
        if self._clock_start is None:
            self._clock_start = now
        budget_bytes = (
            (now - self._clock_start) + self._window_sec
        ) * self._bytes_per_sec
        out: List[bytes] = []
        while self._held and self._released_bytes < budget_bytes:
            chunk = self._held.popleft()
            self._held_bytes -= len(chunk)
            self._released_bytes += len(chunk)
            out.append(chunk)
        return out

    def flush(self) -> List[bytes]:
        """Release everything held (segment tail validated by codec EOS)."""
        out = list(self._held)
        self._held.clear()
        for chunk in out:
            self._released_bytes += len(chunk)
        self._held_bytes = 0
        return out

    def discard(self) -> int:
        """Drop everything held (segment tail condemned by an abort verdict).

        Returns the number of chunks dropped. Released-byte accounting is
        untouched: the client's playback clock keeps running regardless."""
        dropped = len(self._held)
        self._held.clear()
        self._held_bytes = 0
        return dropped

    def discard_tail(self, discard_bytes: int) -> int:
        """Drop up to ``discard_bytes`` from the newest end of the hold.

        Abort verdicts condemn the *tail* of a segment (the loop run, the
        silent pad run, the length beyond the expected sentence end); the
        older held audio is legitimate speech the playback window simply had
        not reached yet and must still be flushed. Whole-chunk granularity.
        Returns the number of chunks dropped."""
        dropped = 0
        remaining = int(discard_bytes)
        while self._held and remaining > 0:
            chunk = self._held.pop()
            self._held_bytes -= len(chunk)
            remaining -= len(chunk)
            dropped += 1
        return dropped


__all__ = ("DeliveryHoldWindow",)
