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
class PendingGroup:
    """One offline pre-split group with a resumable read cursor."""
    group_idx: int
    tokens: List[SegmentToken]
    cursor: int = 0
    next_local_idx: int = 0


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

        # Offline pre-split groups. Each group may still yield multiple
        # backend segments because the Driver remains the final decider.
        self._presplit_groups: Deque[PendingGroup] = deque()
        self._presplit_thresholds: Optional[SplitThresholds] = None
        self._next_group_idx: int = 0

        # Streaming token buffer (streaming mode)
        self._token_buffer: List[SegmentToken] = []
        self._input_complete: bool = False

        # Per-segment drivers; key = segment_idx
        self._drivers: dict[int, StreamingDriver] = {}
        self._next_segment_idx: int = 0

        # Segments that have entered flush (engine still decoding pad)
        self._flushing: set[int] = set()
        # Segments fully done (engine reported SEGMENT_END)
        self._done: set[int] = set()

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
            elif n >= th.force_split_at:
                if last_l1 >= 0:
                    _flush_at(last_l1)
                else:
                    segments.append(current)
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
        Remaining work is queued by group and driven as previous segments flush.
        """
        self._presplit_groups.clear()
        self._next_group_idx = 0
        self._enqueue_presplit_groups(tokens)
        self._input_complete = True

        return self._drive_presplit_batch()

    def _enqueue_presplit_groups(
        self, tokens: List[SegmentToken],
    ) -> None:
        """Pre-split a complete long segment and append its groups."""
        self._presplit_thresholds = self._make_thresholds()
        for seg_tokens in self.pre_split(tokens):
            self._presplit_groups.append(
                PendingGroup(self._next_group_idx, seg_tokens),
            )
            self._next_group_idx += 1

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
        return self._drive_presplit_batch()

    def _drive_presplit_batch(self) -> List[SegmentAction]:
        """Drive as many queued offline groups as concurrency allows."""
        actions: List[SegmentAction] = []
        while self._presplit_groups and self.active_segment_count < self._max_concurrent:
            group = self._presplit_groups.popleft()
            group_actions, has_remaining = self._drive_group(group)
            actions.extend(group_actions)
            if has_remaining:
                self._presplit_groups.append(group)
        return actions

    def _drive_group(
        self,
        group: PendingGroup,
    ) -> tuple[List[SegmentAction], bool]:
        """Drive one offline group until it flushes or runs out of tokens.

        Returns ``(actions, has_remaining_tokens)``. A pre-split group may
        still yield multiple backend segments because the Driver keeps the
        final authority to flush inside the group.
        """
        # Recompute thresholds with the latest EMA before starting each new
        # offline segment so long pre-split queues can benefit from ratio
        # learning accumulated by earlier groups.
        thresholds = self._make_thresholds()
        self._presplit_thresholds = thresholds
        idx, driver = self._create_driver(thresholds)
        actions: List[SegmentAction] = []
        flushed = False
        local_idx = group.next_local_idx
        group.next_local_idx += 1

        start_evt = SpliterEvent(type=ET.START)
        for r in driver.feed(start_evt):
            actions.append(SegmentAction(idx, r, group.group_idx, local_idx, False))

        while group.cursor < len(group.tokens):
            token = group.tokens[group.cursor]
            group.cursor += 1
            evt = self._make_event(token.token_id, token.text, token.punct_level)
            for r in driver.feed(evt):
                actions.append(SegmentAction(
                    idx,
                    r,
                    group.group_idx,
                    local_idx,
                    False,
                    token_text=token.text if r.type in (ActionType.PREFILL, ActionType.DECODE) else "",
                ))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(idx)
                    flushed = True
                    break
            if flushed:
                break

        if not flushed:
            end_evt = SpliterEvent(type=ET.END)
            for r in driver.feed(end_evt):
                actions.append(SegmentAction(idx, r, group.group_idx, local_idx, False))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(idx)

        group_final = group.cursor >= len(group.tokens)
        for sa in actions:
            sa.group_final = group_final

        return actions, group.cursor < len(group.tokens)

    # ------------------------------------------------------------------
    # Public API: streaming
    # ------------------------------------------------------------------

    def feed_tokens(
        self, tokens: List[SegmentToken],
    ) -> List[SegmentAction]:
        """Streaming mode: feed tokenized text, return actions.

        Tokens are classified and fed to the current active Driver.
        If the Driver produces a FLUSH, a new Driver is created for the
        next segment (if concurrency allows).
        """
        classified = self._coerce_tokens(tokens)

        actions: List[SegmentAction] = []

        for token_idx, token in enumerate(classified):
            active_idx = self._get_active_driver_idx()
            if active_idx is None:
                if self.active_segment_count < self._max_concurrent:
                    boot = self._try_start_next()
                    actions.extend(boot)
                    active_idx = self._get_active_driver_idx()
                if active_idx is None:
                    self._token_buffer.append(token)
                    continue

            driver = self._drivers[active_idx]
            evt = self._make_event(token.token_id, token.text, token.punct_level)
            results = driver.feed(evt)

            for r in results:
                actions.append(SegmentAction(
                    active_idx,
                    r,
                    token_text=token.text if r.type in (ActionType.PREFILL, ActionType.DECODE) else "",
                ))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(active_idx)
                    # Only pre-create the follow-up segment if we already know
                    # there is more text to feed. Creating an empty placeholder
                    # driver at the end of a chunk can block SESSION_TOKENS_DONE
                    # and leave the transport waiting forever for a terminal
                    # event.
                    has_more_classified = token_idx < len(classified) - 1
                    if has_more_classified or self._token_buffer:
                        actions.extend(self._try_start_next())

        return actions

    def input_done(self) -> List[SegmentAction]:
        """Signal that no more tokens will arrive (streaming mode)."""
        self._input_complete = True
        actions: List[SegmentAction] = []

        active_idx = self._get_active_driver_idx()
        if active_idx is not None:
            driver = self._drivers[active_idx]
            end_evt = SpliterEvent(type=ET.END)
            for r in driver.feed(end_evt):
                actions.append(SegmentAction(active_idx, r))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(active_idx)

        return actions

    def _get_active_driver_idx(self) -> Optional[int]:
        """Find the most recent non-flushing, non-done driver."""
        for idx in range(self._next_segment_idx - 1, -1, -1):
            if idx not in self._flushing and idx not in self._done:
                return idx
        return None

    def _try_start_next(self) -> List[SegmentAction]:
        """After a FLUSH, start a new segment if concurrency allows."""
        if self.active_segment_count >= self._max_concurrent:
            return []

        actions: List[SegmentAction] = []
        idx, driver = self._create_driver()

        start_evt = SpliterEvent(type=ET.START)
        for r in driver.feed(start_evt):
            actions.append(SegmentAction(idx, r))

        # Drain any buffered tokens into the new driver
        remaining: List[SegmentToken] = []
        for i, token in enumerate(self._token_buffer):
            evt = self._make_event(token.token_id, token.text, token.punct_level)
            results = driver.feed(evt)
            for r in results:
                actions.append(SegmentAction(
                    idx,
                    r,
                    token_text=token.text if r.type in (ActionType.PREFILL, ActionType.DECODE) else "",
                ))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(idx)
                    remaining = self._token_buffer[i + 1:]
                    break
            else:
                continue
            break
        self._token_buffer = remaining

        if self._input_complete and not self._token_buffer:
            end_evt = SpliterEvent(type=ET.END)
            for r in driver.feed(end_evt):
                actions.append(SegmentAction(idx, r))
                if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                    self._flushing.add(idx)

        return actions

    # ------------------------------------------------------------------
    # Engine feedback
    # ------------------------------------------------------------------

    def on_segment_done(self, segment_idx: int) -> List[SegmentAction]:
        """Called when the engine reports SEGMENT_END for a segment.

        Frees the segment's driver and may start the next queued segment
        (offline pre-split) or drain the token buffer (streaming).
        """
        self._done.add(segment_idx)
        self._flushing.discard(segment_idx)
        self._drivers.pop(segment_idx, None)

        if self._presplit_thresholds is not None:
            return self._drive_presplit_batch()

        if self._token_buffer:
            return self._try_start_next()
        return []

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
        self._presplit_groups.clear()
        self._presplit_thresholds = None
        self._next_group_idx = 0
        self._token_buffer.clear()
        self._input_complete = False
        self._drivers.clear()
        self._next_segment_idx = 0
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
