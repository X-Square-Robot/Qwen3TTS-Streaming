"""Text segmentation orchestrator for Qwen3-TTS streaming/offline synthesis.

Responsibilities:
  1. Offline pre-split: split at L1 punctuation (。！？) for optimal
     prosody; forced-cut fallback uses L1 > L2 > L3 priority.
  2. Text buffering: accumulate upstream text chunks, smooth input rate.
  3. Drive Driver: tokenize buffered text, classify punct level, feed
     events to per-segment StreamingDrivers, collect actions.

Architecture (Level 2 pipelining)::

    Upstream → Spliter → Driver[seg0] → SegmentActions
                       → Driver[seg1] → SegmentActions   (parallel)
                       → ...

The Spliter creates a new Driver when the previous segment produces a
FLUSH action.  In offline mode, all segments can be driven immediately.
In streaming mode, segments are driven as tokens arrive.
"""

from __future__ import annotations

from collections import deque
import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from ...core.types import SegmentToken
from .driver import (
    StreamingDriver,
    ActionType,
    ActionResult,
    SplitThresholds,
    compute_thresholds,
)
from .event import SpliterEvent, SpliterEventType as ET
from .defines import LEVEL1_PUNCTIONS, LEVEL2_PUNCTIONS, LEVEL3_PUNCTIONS

logger = logging.getLogger(__name__)

_TRAILING_PUNCT_CLOSER_CATEGORIES = frozenset({"Pe", "Pf"})
_AMBIGUOUS_TRAILING_QUOTES = frozenset({"'", '"', "＇", "＂"})


def _tier_puncts(
    puncts: Tuple[str, ...],
    *lower_tiers: Tuple[str, ...],
) -> Tuple[str, ...]:
    lower = set().union(*lower_tiers) if lower_tiers else set()
    unique = {p for p in puncts if p not in lower}
    return tuple(sorted(unique, key=len, reverse=True))


_LEVEL1_TERMINALS = _tier_puncts(LEVEL1_PUNCTIONS)
_LEVEL2_TERMINALS = _tier_puncts(LEVEL2_PUNCTIONS, LEVEL1_PUNCTIONS)
_LEVEL3_TERMINALS = _tier_puncts(LEVEL3_PUNCTIONS, LEVEL2_PUNCTIONS)
_LEVEL3_BREAK_WHITESPACE = frozenset(p for p in _LEVEL3_TERMINALS if p.isspace())
_PUNCT_TERMINALS_BY_LEVEL = (
    (1, _LEVEL1_TERMINALS),
    (2, _LEVEL2_TERMINALS),
    (3, _LEVEL3_TERMINALS),
)


def _is_trailing_punct_closer(ch: str) -> bool:
    """True for quote/bracket chars that may close after terminal punctuation."""
    return (
        ch in _AMBIGUOUS_TRAILING_QUOTES
        or unicodedata.category(ch) in _TRAILING_PUNCT_CLOSER_CATEGORIES
    )


def _match_terminal_punct(text: str) -> int:
    for level, puncts in _PUNCT_TERMINALS_BY_LEVEL:
        if any(text.endswith(punct) for punct in puncts):
            return level
    return 0


# ---------------------------------------------------------------------------
# Output type — actions tagged with segment index
# ---------------------------------------------------------------------------

@dataclass
class SegmentAction:
    """One Driver action tagged with the segment it belongs to."""
    segment_idx: int
    action: ActionResult
    group_idx: int = -1
    local_idx: int = 0
    group_final: bool = True
    token_text: str = ""


@dataclass
class _PendingToken:
    """One queued token awaiting a driver slot in the unified pipeline.

    ``group_idx`` is the offline pre-split group id, or ``None`` for a streaming
    token (each streaming segment is its own top-level group — see Option ①).
    ``boundary`` marks the last token of an offline group: the driver is
    force-flushed right after it.
    """
    token: SegmentToken
    group_idx: Optional[int]
    boundary: bool = False


# ---------------------------------------------------------------------------
# Spliter
# ---------------------------------------------------------------------------

class Spliter:
    """Orchestrates text splitting and multi-segment Driver management.

    Parameters
    ----------
    engine_max_decode_len : int
        Maximum KV cache steps per segment.
    prefill_len : int
        Estimated prefill length (prompt tokens) for threshold computation.
    ema_ratio : float
        Initial audio:text step ratio (updated via update_ratio).
    max_concurrent : int
        Maximum segments driven in parallel (Level 2).  Set to 1 for
        Level 1 (serial) behavior.
    """

    def __init__(
        self,
        *,
        engine_max_decode_len: int = 512,
        prefill_len: int = 12,
        ema_ratio: float = 5.0,
        safety_margin: int = 8,
        max_concurrent: int = 2,
        ema_alpha: float = 0.1,
        ema_overflow_alpha: float = 0.5,
        ema_min_ratio: float = 2.0,
        ema_max_ratio: float = 10.0,
        l1_split_cap_ratio: float = 0.70,
        l2_split_cap_ratio: float = 0.80,
        l3_split_cap_ratio: float = 0.90,
    ) -> None:
        self._engine_max = engine_max_decode_len
        self._prefill_len = prefill_len
        self._ema_ratio = ema_ratio
        self._safety_margin = safety_margin
        self._max_concurrent = max_concurrent
        self._ema_alpha = ema_alpha
        self._ema_overflow_alpha = ema_overflow_alpha
        self._ema_min_ratio = ema_min_ratio
        self._ema_max_ratio = ema_max_ratio
        self._l1_split_cap_ratio = l1_split_cap_ratio
        self._l2_split_cap_ratio = l2_split_cap_ratio
        self._l3_split_cap_ratio = l3_split_cap_ratio

        # Unified pending-token queue. Replaces the old streaming token buffer
        # and offline pre-split group deque: streaming and offline now feed one
        # queue of `_PendingToken` and one driving core (`_drive_events`).
        self._pending: Deque[_PendingToken] = deque()
        self._presplit_thresholds: Optional[SplitThresholds] = None
        self._next_group_idx: int = 0
        self._input_complete: bool = False

        # Per-segment drivers; key = segment_idx
        self._drivers: dict[int, StreamingDriver] = {}
        self._next_segment_idx: int = 0

        # Coordinate metadata assigned when a segment is opened (Option ①):
        self._seg_coords: dict[int, Tuple[int, int]] = {}   # segment_idx -> (group_idx, local_idx)
        self._seg_group_key: dict[int, Optional[int]] = {}  # segment_idx -> offline group id (None = streaming)
        self._group_next_local: dict[int, int] = {}         # offline group id -> next local_idx

        # Segments that have entered flush (engine still decoding pad)
        self._flushing: set[int] = set()
        # Segments fully done (engine reported SEGMENT_END)
        self._done: set[int] = set()

        # L2 observability side-channel: when recording is enabled, each split
        # decision is buffered here (path/trigger/thresholds/remaining_kv/ema/
        # reason/text_preview) and drained by the caller, which owns the session
        # context and emits the structured event. Keeps the spliter decoupled
        # from lifecycle logging. Off by default ⇒ zero cost at L1/daily.
        self._record_decisions: bool = False
        self._split_decisions: List[dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ema_ratio(self) -> float:
        return self._ema_ratio

    @property
    def current_segment_idx(self) -> int:
        """Index that will be assigned to the next segment created."""
        return self._next_segment_idx

    @property
    def active_segment_count(self) -> int:
        return len(self._drivers)

    # ------------------------------------------------------------------
    # L2 split-decision observability (side-channel; see __init__)
    # ------------------------------------------------------------------

    def enable_decision_recording(self, flag: bool) -> None:
        """Toggle split-decision recording (caller sets this from the session's
        effective observability level)."""
        self._record_decisions = flag

    def drain_split_decisions(self) -> List[dict]:
        """Return and clear buffered split-decision records."""
        if not self._split_decisions:
            return []
        out = self._split_decisions
        self._split_decisions = []
        return out

    def _record_split(
        self, trigger: str, th: SplitThresholds, seg_tokens: List[SegmentToken],
        last_l1_pos: int, chosen_level: int, reason: str,
    ) -> None:
        if not self._record_decisions:
            return
        preview = "".join(getattr(t, "text", "") or "" for t in seg_tokens)
        self._split_decisions.append({
            "obs": "split_decision",
            "path": "offline_pre_split",
            "trigger": trigger,
            "remaining_kv": self._engine_max - self._prefill_len,
            "prefill_len": self._prefill_len,
            "ema_ratio": round(self._ema_ratio, 2),
            "thresholds": {
                "min_tokens_l1": th.min_tokens_l1,
                "force_split_at": th.force_split_at,
            },
            "token_count_at_split": len(seg_tokens),
            "last_l1_pos": last_l1_pos,
            "chosen_level": chosen_level,
            "reason": reason,
            "text_preview": preview,
        })

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    def _make_thresholds(self) -> SplitThresholds:
        remaining_kv = self._engine_max - self._prefill_len
        return compute_thresholds(
            remaining_kv,
            self._ema_ratio,
            self._safety_margin,
            l1_cap_ratio=self._l1_split_cap_ratio,
            l2_cap_ratio=self._l2_split_cap_ratio,
            l3_cap_ratio=self._l3_split_cap_ratio,
        )

    def _create_driver(
        self, thresholds: Optional[SplitThresholds] = None,
    ) -> Tuple[int, StreamingDriver]:
        idx = self._next_segment_idx
        self._next_segment_idx += 1
        driver = StreamingDriver(thresholds or self._make_thresholds())
        self._drivers[idx] = driver
        return idx, driver

    def _results_to_actions(
        self,
        idx: int,
        results: List[ActionResult],
        *,
        token_text: str = "",
        group_idx: Optional[int] = None,
        local_idx: int = 0,
        group_final: bool = True,
    ) -> Tuple[List[SegmentAction], bool]:
        """Map one ``driver.feed()`` result list to SegmentActions.

        PREFILL/DECODE results carry ``token_text``; FLUSH and structural
        results carry ``""``. Marks the segment flushing on FLUSH. Returns
        ``(actions, flushed)`` — callers own the post-flush control flow
        (start-next / break / buffer bookkeeping). Relies on the Driver FSM
        emitting FLUSH as the terminal action of a ``feed()`` call.

        ``group_idx`` defaults to ``idx``: a streaming segment is its own
        top-level group (local_idx=0, group_final=True), so the streaming and
        offline paths share one explicit ``(group_idx, local_idx)`` coordinate
        system and no ``-1`` sentinel reaches the dispatcher.
        """
        if group_idx is None:
            group_idx = idx
        out: List[SegmentAction] = []
        flushed = False
        for r in results:
            tt = token_text if r.type in (ActionType.PREFILL, ActionType.DECODE) else ""
            out.append(SegmentAction(idx, r, group_idx, local_idx, group_final, token_text=tt))
            if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                self._flushing.add(idx)
                flushed = True
                if self._record_decisions:
                    th = self._make_thresholds()
                    self._split_decisions.append({
                        "obs": "driver_transition",
                        "path": "streaming_driver",
                        "segment_idx": idx,
                        "flush_type": r.type.name,
                        "ema_ratio": round(self._ema_ratio, 2),
                        "thresholds": {
                            "min_tokens_l1": th.min_tokens_l1,
                            "force_split_at": th.force_split_at,
                        },
                    })
        return out, flushed

    # ------------------------------------------------------------------
    # Token classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_punct_level(text: str) -> int:
        """Determine punctuation level from token text.

        Returns 0 (none), 1 (L1), 2 (L2), or 3 (L3).
        Classifies punctuation at the token's right boundary. Closing quotes
        or brackets after a boundary punctuation are ignored, while opening
        quotes are kept as normal trailing text.
        """
        if not text:
            return 0

        i = len(text) - 1
        saw_level3_break = False

        while i >= 0:
            ch = text[i]
            if ch.isspace():
                if ch in _LEVEL3_BREAK_WHITESPACE:
                    saw_level3_break = True
                i -= 1
                continue
            if _is_trailing_punct_closer(ch):
                i -= 1
                continue
            break

        if i < 0:
            return 3 if saw_level3_break else 0

        level = _match_terminal_punct(text[: i + 1])
        if level:
            return level
        return 3 if saw_level3_break else 0

    def _make_event(
        self, token_id: int, text: str, punct_level: int,
    ) -> SpliterEvent:
        if punct_level > 0:
            return SpliterEvent(
                type=ET.PUNCTUATION_TOKEN,
                token=token_id,
                text=text,
                punct_level=punct_level,
            )
        return SpliterEvent(
            type=ET.NORMAL_TOKEN,
            token=token_id,
            text=text,
            punct_level=0,
        )

    # ------------------------------------------------------------------
    # Offline pre-split
    # ------------------------------------------------------------------

    def pre_split(
        self, tokens: List[SegmentToken],
    ) -> List[List[SegmentToken]]:
        """Split a fully-known token sequence at L1 punctuation boundaries.

        Returns list of segments, each segment is ``SegmentToken`` sequence.

        Unlike the streaming Driver (which uses L1/L2/L3 thresholds because it
        lacks global visibility), offline pre-split only cuts at L1 (。！？)
        for optimal prosody and fewer segments.

        Algorithm:
          1. Greedy scan; split at L1 punctuation when token_count >= min_tokens_l1.
          2. If force_split_at is reached without an L1 split, prefer L1 only;
             otherwise hard-cut at the current position instead of snapping
             to L2/L3 punctuation. This avoids exaggerated prosodic breaks
             on commas / formatting newlines in fully-known long sentences.
        """
        if not tokens:
            return []

        th = self._make_thresholds()
        source_tokens = self._coerce_tokens(tokens)
        segments: List[List[SegmentToken]] = []
        current: List[SegmentToken] = []
        last_l1: int = -1
        def _flush_at(pos: int) -> None:
            nonlocal current, last_l1
            split_at = pos + 1
            segments.append(current[:split_at])
            remaining = current[split_at:]
            current = remaining
            last_l1 = -1
            for j, token in enumerate(current):
                if token.punct_level == 1:
                    last_l1 = j

        for token in source_tokens:
            current.append(token)
            n = len(current)

            if token.punct_level == 1:
                last_l1 = n - 1

            if token.punct_level == 1 and n >= th.min_tokens_l1:
                _flush_at(n - 1)
                self._record_split(
                    "l1_punct", th, segments[-1], n - 1, 1,
                    f"L1 punct at {n} tokens >= min_tokens_l1({th.min_tokens_l1})",
                )
            elif n >= th.force_split_at:
                if last_l1 >= 0:
                    prev_last_l1 = last_l1
                    _flush_at(last_l1)
                    self._record_split(
                        "force_fallback_l1", th, segments[-1], prev_last_l1, 1,
                        f"force_split_at({th.force_split_at}) reached without fresh L1; "
                        f"fell back to last L1 at pos {prev_last_l1}",
                    )
                else:
                    segments.append(current)
                    self._record_split(
                        "force_hard_cut", th, segments[-1], -1, 0,
                        f"force_split_at({th.force_split_at}) reached, no L1 since "
                        f"segment start; hard cut",
                    )
                    current = []
                    last_l1 = -1

        if current:
            segments.append(current)

        return segments

    # ------------------------------------------------------------------
    # Public API: offline
    # ------------------------------------------------------------------

    def set_full_text(
        self, tokens: List[SegmentToken],
    ) -> List[SegmentAction]:
        """Offline mode: set complete token sequence, pre-split, drive all.

        Returns SegmentActions for up to max_concurrent segments.
        Remaining work is queued and driven as previous segments flush.
        """
        self._pending.clear()
        self._next_group_idx = 0
        self._enqueue_presplit_groups(tokens)
        self._input_complete = True

        return self._drive_events()

    def _enqueue_presplit_groups(
        self, tokens: List[SegmentToken],
    ) -> None:
        """Pre-split tokens into L1 groups and enqueue them as pending tokens.

        Each group gets a monotonic ``group_idx``; the group's last token is
        marked ``boundary`` so the driving core force-flushes at the group end.
        """
        self._presplit_thresholds = self._make_thresholds()
        for seg_tokens in self.pre_split(tokens):
            if not seg_tokens:
                continue
            gid = self._next_group_idx
            self._next_group_idx += 1
            last = len(seg_tokens) - 1
            for i, tok in enumerate(seg_tokens):
                self._pending.append(_PendingToken(tok, gid, boundary=(i == last)))

    def push_group_tokens(
        self, tokens: List[SegmentToken],
    ) -> List[SegmentAction]:
        """Queue one complete long-segment unit for group-level pre-splitting.

        Unlike ``set_full_text()``, this does not mark the session text-complete.
        Each incoming long segment is treated as a self-contained unit that may
        be further pre-split into one or more groups before the second-layer
        driver takes over.
        """
        if not tokens:
            return []
        self._enqueue_presplit_groups(tokens)
        return self._drive_events()

    # ------------------------------------------------------------------
    # Unified driving core
    # ------------------------------------------------------------------

    def _drive_events(self) -> List[SegmentAction]:
        """Drive the pending queue through the active-driver chain.

        Streaming and offline share this core: one active driver accumulates
        tokens across calls; offline ``boundary`` tokens force a flush;
        concurrency gates how many segments are in flight. Loops until no
        further progress is possible (queue drained / backpressure / an open
        streaming segment waiting for more text).
        """
        actions: List[SegmentAction] = []
        while True:
            seg_actions = self._drive_one_segment()
            if not seg_actions:
                break
            actions.extend(seg_actions)
        return actions

    def _drive_one_segment(self) -> List[SegmentAction]:
        """Advance a single segment by one driving step.

        Opens a new segment if there is none active and a slot is free, feeds
        pending tokens of the active segment's group until it flushes / a
        boundary forces a flush / the queue drains / end-of-input flushes it.
        Returns the actions produced, or ``[]`` when no progress is possible.
        """
        out: List[SegmentAction] = []

        active_idx = self._get_active_driver_idx()
        if active_idx is None:
            if not self._pending or self.active_segment_count >= self._max_concurrent:
                return out  # nothing to do, or backpressure
            active_idx = self._open_segment(out)

        driver = self._drivers[active_idx]
        group_idx, local_idx = self._seg_coords[active_idx]
        cur_key = self._seg_group_key[active_idx]
        is_stream = cur_key is None
        flushed = False

        while self._pending and self._pending[0].group_idx == cur_key:
            pt = self._pending.popleft()
            evt = self._make_event(pt.token.token_id, pt.token.text, pt.token.punct_level)
            tok_actions, flushed = self._results_to_actions(
                active_idx, driver.feed(evt), token_text=pt.token.text,
                group_idx=group_idx, local_idx=local_idx, group_final=is_stream,
            )
            out.extend(tok_actions)
            if flushed:
                self._finalize_segment(active_idx, out, group_exhausted=pt.boundary)
                break
            if pt.boundary:
                end_actions, _ = self._results_to_actions(
                    active_idx, driver.feed(SpliterEvent(type=ET.END)),
                    group_idx=group_idx, local_idx=local_idx, group_final=is_stream,
                )
                out.extend(end_actions)
                self._finalize_segment(active_idx, out, group_exhausted=True)
                flushed = True
                break

        # Auto handoff (Step 3): if the next pending token belongs to a different
        # group than the open segment, close the open segment at a clean
        # boundary so the new group starts fresh. Does not trigger in pure
        # streaming/offline sessions (one group key throughout).
        if not flushed and self._pending and self._pending[0].group_idx != cur_key:
            end_actions, _ = self._results_to_actions(
                active_idx, driver.feed(SpliterEvent(type=ET.END)),
                group_idx=group_idx, local_idx=local_idx, group_final=is_stream,
            )
            out.extend(end_actions)
            self._finalize_segment(active_idx, out, group_exhausted=True)
            flushed = True

        # End-of-input: flush the open streaming segment once the queue drains.
        if not flushed and self._input_complete and not self._pending:
            end_actions, _ = self._results_to_actions(
                active_idx, driver.feed(SpliterEvent(type=ET.END)),
                group_idx=group_idx, local_idx=local_idx, group_final=is_stream,
            )
            out.extend(end_actions)
            self._finalize_segment(active_idx, out, group_exhausted=True)

        return out

    def _open_segment(self, out: List[SegmentAction]) -> int:
        """Create a driver for the next pending token's group, assign its
        ``(group_idx, local_idx)`` coordinate (Option ①), emit START, and
        return the new segment index."""
        key = self._pending[0].group_idx
        # Recompute thresholds with the latest EMA before each new segment so
        # long offline queues benefit from ratio learning by earlier segments.
        thresholds = self._make_thresholds()
        if key is not None:
            self._presplit_thresholds = thresholds
        idx, driver = self._create_driver(thresholds)
        if key is None:
            group_idx, local_idx = idx, 0            # streaming: each segment its own group
        else:
            group_idx, local_idx = key, self._group_next_local.get(key, 0)
        self._seg_coords[idx] = (group_idx, local_idx)
        self._seg_group_key[idx] = key
        start_actions, _ = self._results_to_actions(
            idx, driver.feed(SpliterEvent(type=ET.START)),
            group_idx=group_idx, local_idx=local_idx, group_final=(key is None),
        )
        out.extend(start_actions)
        return idx

    def _finalize_segment(
        self, idx: int, actions: List[SegmentAction], *, group_exhausted: bool,
    ) -> None:
        """Stamp ``group_final`` on a just-flushed segment's actions and advance
        its group's local counter. Streaming segments are always final; an
        offline segment is final only when its group is exhausted."""
        key = self._seg_group_key[idx]
        group_final = key is None or group_exhausted
        for sa in actions:
            if sa.segment_idx == idx:
                sa.group_final = group_final
        if key is not None and not group_exhausted:
            self._group_next_local[key] = self._seg_coords[idx][1] + 1

    # ------------------------------------------------------------------
    # Public API: streaming
    # ------------------------------------------------------------------

    def feed_tokens(
        self, tokens: List[SegmentToken],
    ) -> List[SegmentAction]:
        """Streaming mode: queue tokens (each its own group) and drive.

        Tokens accumulate into the active Driver across calls; when it flushes,
        the next segment opens if concurrency allows, otherwise tokens wait in
        the shared pending queue. The driver decides flush points (no boundary).
        """
        for tok in self._coerce_tokens(tokens):
            self._pending.append(_PendingToken(tok, None, boundary=False))
        return self._drive_events()

    def input_done(self) -> List[SegmentAction]:
        """Signal that no more tokens will arrive; flush the open segment."""
        self._input_complete = True
        return self._drive_events()

    def _get_active_driver_idx(self) -> Optional[int]:
        """Find the most recent non-flushing, non-done driver."""
        for idx in range(self._next_segment_idx - 1, -1, -1):
            if idx not in self._flushing and idx not in self._done:
                return idx
        return None

    # ------------------------------------------------------------------
    # Engine feedback
    # ------------------------------------------------------------------

    def on_segment_done(self, segment_idx: int) -> List[SegmentAction]:
        """Called when the engine reports SEGMENT_END for a segment.

        Frees the segment's driver/slot and drives any pending work that the
        freed concurrency slot now allows (offline groups or streaming).
        """
        self._done.add(segment_idx)
        self._flushing.discard(segment_idx)
        self._drivers.pop(segment_idx, None)
        self._seg_coords.pop(segment_idx, None)
        self._seg_group_key.pop(segment_idx, None)

        return self._drive_events()

    def update_ratio(
        self, actual_audio_steps: int, actual_text_tokens: int,
        *, overflow: bool = False,
    ) -> None:
        """Update EMA audio:text ratio from engine feedback.

        After EMA update, refreshes thresholds for all active (non-flushing)
        drivers so subsequent token classification uses the latest ratio.

        Parameters
        ----------
        overflow : bool
            When True, uses ``ema_overflow_alpha`` (default 0.5) for faster
            convergence after a KV cache overflow event.
        """
        if actual_text_tokens <= 0:
            return
        observed = actual_audio_steps / actual_text_tokens
        old_ratio = self._ema_ratio
        if overflow:
            alpha = self._ema_overflow_alpha
            self._ema_ratio = (1.0 - alpha) * self._ema_ratio + alpha * observed
            self._ema_ratio = max(self._ema_min_ratio, min(self._ema_max_ratio, self._ema_ratio))
            logger.warning(
                "EMA overflow update: observed=%.3f ema=%.3f→%.3f (steps=%d tokens=%d)",
                observed, old_ratio, self._ema_ratio, actual_audio_steps, actual_text_tokens,
            )
        else:
            alpha = self._ema_alpha
            self._ema_ratio = (1.0 - alpha) * self._ema_ratio + alpha * observed
            self._ema_ratio = max(self._ema_min_ratio, min(self._ema_max_ratio, self._ema_ratio))
            logger.debug(
                "EMA update: observed=%.3f ema=%.3f→%.3f (steps=%d tokens=%d)",
                observed, old_ratio, self._ema_ratio, actual_audio_steps, actual_text_tokens,
            )

        if abs(self._ema_ratio - old_ratio) > 0.01:
            self._refresh_active_thresholds()

    def _refresh_active_thresholds(self) -> None:
        """Recompute thresholds for all non-flushing active drivers.

        Called after EMA ratio changes so that drivers currently
        accumulating tokens use up-to-date split thresholds.
        """
        new_th = self._make_thresholds()
        refreshed = 0
        for idx, driver in self._drivers.items():
            if idx in self._flushing:
                continue
            driver.thresholds = new_th
            refreshed += 1
        if refreshed:
            logger.debug(
                "Refreshed thresholds for %d active driver(s): "
                "L1=%d L2=%d L3=%d force=%d",
                refreshed,
                new_th.min_tokens_l1,
                new_th.min_tokens_l2,
                new_th.min_tokens_l3,
                new_th.force_split_at,
            )

    # ------------------------------------------------------------------
    # Full reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._pending.clear()
        self._presplit_thresholds = None
        self._next_group_idx = 0
        self._input_complete = False
        self._drivers.clear()
        self._next_segment_idx = 0
        self._seg_coords.clear()
        self._seg_group_key.clear()
        self._group_next_local.clear()
        self._flushing.clear()
        self._done.clear()

    def _coerce_tokens(self, tokens) -> List[SegmentToken]:
        """Accept legacy tuple tokens at the API edge, normalize internally."""
        normalized: List[SegmentToken] = []
        for token in tokens:
            if isinstance(token, SegmentToken):
                normalized.append(token)
                continue
            token_id, text = token[:2]
            punct_level = token[2] if len(token) > 2 else self.classify_punct_level(text)
            normalized.append(
                SegmentToken(
                    token_id=token_id,
                    text=text,
                    punct_level=punct_level,
                )
            )
        return normalized


__all__ = ("Spliter", "SegmentAction")
