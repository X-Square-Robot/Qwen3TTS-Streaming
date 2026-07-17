"""Per-session audio chunk reorder buffer for Level 2 segment pipelining.

When multiple segments of the same session decode in parallel, audio chunks
may arrive out of order. For offline pre-split mode, ordering is hierarchical:
``group_idx`` preserves top-level text order while ``local_idx`` preserves the
Driver-created sub-segment order inside each group.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List


class AudioReorder:
    """Reorder buffer that emits audio in text order.

    Usage::

        reorder = AudioReorder()
        out = reorder.push(0, 1, b"g0s1")      # buffered
        out = reorder.push(0, 0, b"g0s0")      # emitted
        out = reorder.mark_done(0, 0)           # drains g0s1 if ready
    """

    __slots__ = ("_next_group", "_next_local", "_buffers", "_done", "_final_locals")

    def __init__(self) -> None:
        self._next_group: int = 0
        self._next_local: int = 0
        self._buffers: dict[tuple[int, int], list[bytes]] = defaultdict(list)
        self._done: set[tuple[int, int]] = set()
        self._final_locals: dict[int, int] = {}

    @property
    def next_emit_segment(self) -> tuple[int, int]:
        return (self._next_group, self._next_local)

    def pending_state(self) -> dict:
        """Introspection for L2 ``reorder_state`` observability: how much audio
        is buffered waiting for an earlier segment (reorder stall risk)."""
        return {
            "next_emit": [self._next_group, self._next_local],
            "buffered_keys": len(self._buffers),
            "buffered_chunks": sum(len(v) for v in self._buffers.values()),
        }

    def push(self, group_idx: int, local_idx: int, audio: bytes) -> List[bytes]:
        """Push an audio chunk; return chunks ready for emission (in order)."""
        key = (group_idx, local_idx)
        self._buffers[key].append(audio)
        if key == (self._next_group, self._next_local):
            return self._try_drain()
        return []

    def mark_done(
        self, group_idx: int, local_idx: int, *, group_final: bool = False
    ) -> List[bytes]:
        """Mark a segment as fully complete; drain any contiguous completions."""
        return [
            chunk
            for _, chunks, _ in self.mark_done_ex(
                group_idx, local_idx, group_final=group_final
            )
            for chunk in chunks
        ]

    def mark_done_ex(
        self, group_idx: int, local_idx: int, *, group_final: bool = False
    ) -> List[tuple]:
        """Like :meth:`mark_done`, but preserves segment attribution.

        Returns ``[(key, chunks, fully_passed)]`` in drain order — one entry
        per segment the drain touched. ``fully_passed`` is True when the
        playhead advanced past that segment (all of its audio is out of this
        buffer). Guarded delivery needs this: segment verdicts must be applied
        to that segment's own audio, and a single mark_done can chain-drain
        several already-done segments."""
        key = (group_idx, local_idx)
        self._done.add(key)
        if group_final:
            self._final_locals[group_idx] = local_idx
        return self._try_drain_ex()

    def _try_drain(self) -> List[bytes]:
        return [chunk for _, chunks, _ in self._try_drain_ex() for chunk in chunks]

    def _try_drain_ex(self) -> List[tuple]:
        out: List[tuple] = []
        while True:
            key = (self._next_group, self._next_local)
            buf = self._buffers.get(key)
            if buf is None and key not in self._done:
                break
            chunks: List[bytes] = []
            if buf:
                chunks = list(buf)
                buf.clear()
            if key in self._done:
                if key in self._buffers:
                    del self._buffers[key]
                self._done.discard(key)
                final_local = self._final_locals.get(self._next_group)
                if final_local is not None and self._next_local >= final_local:
                    self._final_locals.pop(self._next_group, None)
                    self._next_group += 1
                    self._next_local = 0
                else:
                    self._next_local += 1
                out.append((key, chunks, True))
            else:
                if chunks:
                    out.append((key, chunks, False))
                break
        return out

    def discard(self, group_idx: int, local_idx: int) -> int:
        """Drop the buffered (not yet drained) chunks of one segment.

        Used when the engine reruns a hallucinated lookahead segment: the
        buffered attempt is garbage and the rerun pushes fresh chunks under
        the same key. Playhead and done-set are untouched, so this is only
        meaningful for segments the drain has not reached yet; chunks already
        drained to the client cannot be recalled. Returns the chunk count
        dropped."""
        buf = self._buffers.pop((group_idx, local_idx), None)
        return len(buf) if buf else 0

    def reset(self) -> None:
        self._next_group = 0
        self._next_local = 0
        self._buffers.clear()
        self._done.clear()
        self._final_locals.clear()


__all__ = ("AudioReorder",)
