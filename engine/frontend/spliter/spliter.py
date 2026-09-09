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
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Tuple

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
from .ratio import (
    RatioObservation,
    RatioOutcome,
    SplitRatioController,
)

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
    normalized_start: int = 0
    normalized_end: int = 0
    raw_start: int = 0
    raw_end: int = 0


@dataclass
class _PendingToken:
    """One queued token awaiting a driver slot in the unified pipeline.

    ``group_key`` is an internal offline pre-split key, or ``None`` for a
    streaming token.  It is deliberately not the public reorder ``group_idx``:
    public ids are allocated only when a segment opens, in FIFO text order.
    ``boundary`` marks the last token of an offline group: the driver is
    force-flushed right after it.
    """

    token: SegmentToken
    group_key: Optional[int]
    boundary: bool = False
    boundary_before: bool = False
    forced_boundary: bool = False
    plan: Optional["_PresplitPlan"] = None


@dataclass(frozen=True)
class _PresplitPlan:
    """Immutable Stage-1 capacity contract shared by one planning epoch.

    A packet may yield several groups.  Every visible group must be opened with
    the same thresholds used to choose its boundary; otherwise a later ratio
    update can silently turn a planned ``N``-token group into ``cap + tiny
    remainder`` during Stage 2.  A monotonic safety tightening may replace the
    plan only by re-running Stage 1 over an entirely unopened packet suffix.
    """

    safety_ratio: float
    thresholds: SplitThresholds


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
        Initial audio:text duration ratio (updated from natural completions).
        This ratio is used for progress/duration estimation, not as an
        unconstrained hard-capacity estimate.
    safety_ratio_initial : float, optional
        Conservative audio:text ratio used for text-capacity planning.  It is
        initialized independently and never decreases within a session.  When
        omitted, it inherits ``ema_ratio`` for compatibility with direct
        callers; the production server passes its explicit calibrated value.
    max_concurrent : int
        Maximum segments driven in parallel (Level 2).  Set to 1 for
        Level 1 (serial) behavior.
    """

    def __init__(
        self,
        *,
        engine_max_decode_len: int = 512,
        prefill_len: int = 12,
        ema_ratio: float = 4.5,
        safety_ratio_initial: Optional[float] = None,
        safety_margin: int = 8,
        max_concurrent: int = 2,
        ema_alpha: float = 0.1,
        ema_overflow_alpha: float = 0.5,
        ema_min_ratio: float = 2.0,
        ema_max_ratio: float = 10.0,
        ema_min_observation_tokens: int = 8,
        safety_failure_multiplier: float = 1.25,
        l1_split_cap_ratio: float = 0.70,
        l2_split_cap_ratio: float = 0.80,
        l3_split_cap_ratio: float = 0.90,
    ) -> None:
        self._engine_max = engine_max_decode_len
        self._prefill_len = prefill_len
        self._safety_margin = safety_margin
        self._max_concurrent = max_concurrent
        self._ratios = SplitRatioController(
            duration_initial=ema_ratio,
            safety_initial=(
                ema_ratio
                if safety_ratio_initial is None
                else safety_ratio_initial
            ),
            duration_alpha=ema_alpha,
            # Keep the existing overflow-alpha knob backward compatible, but
            # apply it to the independent failure-safety controller instead
            # of mixing censored failures into the duration EMA.
            failure_alpha=ema_overflow_alpha,
            min_ratio=ema_min_ratio,
            max_ratio=ema_max_ratio,
            min_duration_tokens=ema_min_observation_tokens,
            failure_multiplier=safety_failure_multiplier,
        )
        self._l1_split_cap_ratio = l1_split_cap_ratio
        self._l2_split_cap_ratio = l2_split_cap_ratio
        self._l3_split_cap_ratio = l3_split_cap_ratio

        # Unified pending-token queue. Replaces the old streaming token buffer
        # and offline pre-split group deque: streaming and offline now feed one
        # queue of `_PendingToken` and one driving core (`_drive_events`).
        self._pending: Deque[_PendingToken] = deque()
        # Offline planning keys identify queued Stage-1 groups. Public reorder
        # ids live in a separate namespace and are allocated only at open time;
        # preallocating public ids for offline work behind pending streaming
        # text can otherwise invert AudioReorder order under backpressure.
        self._next_group_key: int = 0
        self._next_output_group_idx: int = 0
        self._input_complete: bool = False

        # Per-segment drivers; key = segment_idx
        self._drivers: dict[int, StreamingDriver] = {}
        self._next_segment_idx: int = 0

        # Coordinate metadata assigned when a segment is opened (Option ①):
        self._seg_coords: dict[
            int, Tuple[int, int]
        ] = {}  # segment_idx -> (group_idx, local_idx)
        self._seg_group_key: dict[
            int, Optional[int]
        ] = {}  # segment_idx -> offline group id (None = streaming)
        self._seg_ema_ratio: dict[int, float] = {}  # EMA snapshot at segment open
        self._seg_safety_ratio: dict[
            int, float
        ] = {}  # capacity ratio snapshot at segment open
        self._group_next_local: dict[
            int, int
        ] = {}  # internal offline group key -> next local_idx
        self._group_output_idx: dict[
            int, int
        ] = {}  # internal offline group key -> public reorder group_idx

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
        """Current duration EMA (never the unconstrained split budget)."""
        return self._ratios.duration_ratio

    @property
    def safety_ratio(self) -> float:
        """Current monotonic ratio used for future capacity planning."""
        return self._ratios.safety_ratio

    @property
    def current_thresholds(self) -> SplitThresholds:
        """Return the thresholds used for the next newly opened segment."""

        return self._make_thresholds()

    def ema_ratio_for_segment(self, segment_idx: int) -> float:
        """Return the EMA snapshot frozen when *segment_idx* opened.

        Progress estimation must use the segment-local snapshot.  Using the
        session's latest EMA would make a long-running segment change its
        denominator halfway through playback and could move the estimate
        backwards when a later segment teaches the splitter a new ratio.
        """

        return self._seg_ema_ratio.get(int(segment_idx), self.ema_ratio)

    def safety_ratio_for_segment(self, segment_idx: int) -> float:
        """Return the capacity ratio frozen when *segment_idx* opened."""
        return self._seg_safety_ratio.get(int(segment_idx), self.safety_ratio)

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
        self,
        trigger: str,
        th: SplitThresholds,
        seg_tokens: List[SegmentToken],
        last_l1_pos: int,
        chosen_level: int,
        reason: str,
    ) -> None:
        if not self._record_decisions:
            return
        preview = "".join(getattr(t, "text", "") or "" for t in seg_tokens)
        self._split_decisions.append(
            {
                "obs": "split_decision",
                "path": "offline_pre_split",
                "trigger": trigger,
                "remaining_kv": self._engine_max - self._prefill_len,
                "prefill_len": self._prefill_len,
                # Compatibility alias: this field historically meant the
                # ratio that planned the split capacity.
                "ema_ratio": round(self.safety_ratio, 3),
                "duration_ema_ratio": round(self.ema_ratio, 3),
                "safety_ratio": round(self.safety_ratio, 3),
                "thresholds": {
                    "min_tokens_l1": th.min_tokens_l1,
                    "force_split_at": th.force_split_at,
                },
                "token_count_at_split": len(seg_tokens),
                "last_l1_pos": last_l1_pos,
                "chosen_level": chosen_level,
                "reason": reason,
                "text_preview": preview,
            }
        )

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    def _make_thresholds(self) -> SplitThresholds:
        remaining_kv = self._engine_max - self._prefill_len
        return compute_thresholds(
            remaining_kv,
            self.safety_ratio,
            self._safety_margin,
            l1_cap_ratio=self._l1_split_cap_ratio,
            l2_cap_ratio=self._l2_split_cap_ratio,
            l3_cap_ratio=self._l3_split_cap_ratio,
        )

    def _make_presplit_plan(self) -> _PresplitPlan:
        return _PresplitPlan(
            safety_ratio=self.safety_ratio,
            thresholds=self._make_thresholds(),
        )

    def _create_driver(
        self,
        thresholds: Optional[SplitThresholds] = None,
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
        token: Optional[SegmentToken] = None,
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
            out.append(
                SegmentAction(
                    idx,
                    r,
                    group_idx,
                    local_idx,
                    group_final,
                    token_text=tt,
                    normalized_start=(token.normalized_start if token is not None else 0),
                    normalized_end=(token.normalized_end if token is not None else 0),
                    raw_start=(token.raw_start if token is not None else 0),
                    raw_end=(token.raw_end if token is not None else 0),
                )
            )
            if r.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                self._flushing.add(idx)
                flushed = True
                if self._record_decisions:
                    th = self._drivers[idx].thresholds
                    self._split_decisions.append(
                        {
                            "obs": "driver_transition",
                            "path": "streaming_driver",
                            "segment_idx": idx,
                            "flush_type": r.type.name,
                            # Compatibility alias for existing telemetry
                            # consumers; capacity now uses safety_ratio.
                            "ema_ratio": round(
                                self._seg_safety_ratio.get(idx, self.safety_ratio),
                                3,
                            ),
                            "duration_ema_ratio": round(
                                self._seg_ema_ratio.get(idx, self.ema_ratio), 3
                            ),
                            "safety_ratio": round(
                                self._seg_safety_ratio.get(idx, self.safety_ratio),
                                3,
                            ),
                            "thresholds": {
                                "min_tokens_l1": th.min_tokens_l1,
                                "force_split_at": th.force_split_at,
                            },
                        }
                    )
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
        self,
        token_id: int,
        text: str,
        punct_level: int,
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
        self,
        tokens: List[SegmentToken],
        *,
        thresholds: Optional[SplitThresholds] = None,
    ) -> List[List[SegmentToken]]:
        """Pack a fully-known token sequence into capacity-sized segments,
        cutting at the LATEST safe boundary (hierarchical bin-packing).

        Unlike the streaming Driver (first-fit past a low threshold, no
        foresight), offline pre-split sees the whole sequence, so it packs L1
        units up to one segment's capacity and cuts at the *latest* boundary
        that fits — preferring L1 (。！？), falling back to L2 (，；：) then L3,
        and hard-cutting only when a single unit alone exceeds capacity. This
        yields fewer, fuller, prosody-continuous segments than cutting at the
        first L1, and avoids orphaning a trailing punctuation token.

        Capacity comes from one immutable Stage-1 planning snapshot.  Every
        resulting group carries that snapshot into obey mode (see
        ``_open_segment``), so Stage 2 cannot re-fragment a group whose boundary
        was selected under a different hard cap.  Monotonic safety feedback is
        handled separately by re-running Stage 1 over only the unopened suffix.
        """
        if not tokens:
            return []

        th = thresholds or self._make_thresholds()
        capacity = th.force_split_at
        source_tokens = self._coerce_tokens(tokens)
        segments: List[List[SegmentToken]] = []
        current: List[SegmentToken] = []
        last = {1: -1, 2: -1, 3: -1}  # latest index of each punct tier in `current`

        def _flush_at(pos: int) -> None:
            nonlocal current
            segments.append(current[: pos + 1])
            current = current[pos + 1 :]
            last[1] = last[2] = last[3] = -1
            for j, tok in enumerate(current):
                if tok.punct_level in last:
                    last[tok.punct_level] = j

        for token in source_tokens:
            current.append(token)
            n = len(current)
            if token.punct_level in last:
                last[token.punct_level] = n - 1

            if n >= capacity:
                # Must cut: take the latest boundary that fits, by tier; only
                # hard-cut when a single unit has no usable punctuation.
                # (positions captured before _flush_at resets the `last` map.)
                if last[1] >= 0:
                    pos = last[1]
                    _flush_at(pos)
                    self._record_split(
                        "l1",
                        th,
                        segments[-1],
                        pos,
                        1,
                        f"packed to capacity({capacity}); cut at latest L1 boundary pos {pos}",
                    )
                elif last[2] >= 0:
                    pos = last[2]
                    _flush_at(pos)
                    self._record_split(
                        "l2",
                        th,
                        segments[-1],
                        pos,
                        2,
                        f"packed to capacity({capacity}); no L1, cut at latest L2 boundary pos {pos}",
                    )
                elif last[3] >= 0:
                    pos = last[3]
                    _flush_at(pos)
                    self._record_split(
                        "l3",
                        th,
                        segments[-1],
                        pos,
                        3,
                        f"packed to capacity({capacity}); no L1/L2, cut at latest L3 boundary pos {pos}",
                    )
                else:
                    segments.append(current)
                    self._record_split(
                        "hard_cut",
                        th,
                        segments[-1],
                        -1,
                        0,
                        f"packed to capacity({capacity}); single unit has no usable "
                        f"punctuation; hard cut",
                    )
                    current = []
                    last[1] = last[2] = last[3] = -1

        if current:
            segments.append(current)

        return segments

    # ------------------------------------------------------------------
    # Public API: offline
    # ------------------------------------------------------------------

    def set_full_text(
        self,
        tokens: List[SegmentToken],
        *,
        force_boundary_indices: Optional[Iterable[int]] = None,
        force_boundary_before_indices: Optional[Iterable[int]] = None,
    ) -> List[SegmentAction]:
        """Offline mode: set complete token sequence, pre-split, drive all.

        Returns SegmentActions for up to max_concurrent segments.
        Remaining work is queued and driven as previous segments flush.
        """
        self._pending.clear()
        self._next_group_key = 0
        self._next_output_group_idx = 0
        self._group_output_idx.clear()
        self._enqueue_presplit_groups(tokens)
        pending = list(self._pending)
        for index in force_boundary_indices or ():
            if 0 <= int(index) < len(pending):
                pending[int(index)].boundary = True
                pending[int(index)].forced_boundary = True
        for index in force_boundary_before_indices or ():
            if 0 <= int(index) < len(pending):
                pending[int(index)].boundary_before = True
        self._pending = deque(pending)
        self._input_complete = True

        return self._drive_events()

    def _enqueue_presplit_groups(
        self,
        tokens: List[SegmentToken],
    ) -> None:
        """Pre-split tokens into L1 groups and enqueue them as pending tokens.

        Each group gets an internal planning key; its public ``group_idx`` is
        allocated later in FIFO open order. The group's last token is marked
        ``boundary`` so the driving core force-flushes at the group end.
        """
        plan = self._make_presplit_plan()
        pending, next_group_key = self._build_planned_pending(
            tokens,
            plan=plan,
            first_group_key=self._next_group_key,
        )
        self._pending.extend(pending)
        self._next_group_key = next_group_key

    def _build_planned_pending(
        self,
        tokens: List[SegmentToken],
        *,
        plan: _PresplitPlan,
        first_group_key: int,
    ) -> Tuple[List[_PendingToken], int]:
        """Build queued groups whose Stage-1 and Stage-2 caps agree."""
        pending: List[_PendingToken] = []
        group_key = int(first_group_key)
        for seg_tokens in self.pre_split(tokens, thresholds=plan.thresholds):
            if not seg_tokens:
                continue
            if len(seg_tokens) > plan.thresholds.force_split_at:
                raise RuntimeError(
                    "pre-split group exceeds its immutable capacity contract: "
                    f"tokens={len(seg_tokens)} "
                    f"capacity={plan.thresholds.force_split_at}"
                )
            last = len(seg_tokens) - 1
            for i, tok in enumerate(seg_tokens):
                pending.append(
                    _PendingToken(
                        tok,
                        group_key,
                        boundary=(i == last),
                        plan=plan,
                    )
                )
            group_key += 1
        return pending, group_key

    def _tighten_pending_presplit_plans(self) -> None:
        """Re-plan only unopened offline suffixes after safety tightens.

        Open/flushing drivers retain their frozen cap.  For each still-hidden
        packet cohort, all remaining tokens are packed together under the new,
        smaller cap.  Repacking the whole suffix is essential: applying the
        live cap independently to every old group was the source of repeated
        ``108+1`` / ``106+3`` residual segments.

        Safety is monotonic, so this path never merges groups after a duration
        observation falls.  Packet boundaries remain intact because one
        `_PresplitPlan` object is shared only by groups from the same packet.
        """
        if not self._pending:
            return

        live_plan = self._make_presplit_plan()
        pending = list(self._pending)
        offline = [pt for pt in pending if pt.group_key is not None]
        if not offline:
            return
        if not any(
            pt.plan is not None
            and pt.plan.thresholds.force_split_at
            > live_plan.thresholds.force_split_at
            for pt in offline
        ):
            return

        # A pending continuation of an already exposed group must never be
        # renumbered.  Obey-mode planned groups cannot enter this state, but
        # retain a defensive fail-safe for mixed/legacy traffic.
        protected = set(self._group_next_local)
        if any(pt.group_key in protected for pt in offline):
            logger.warning(
                "Skipped pending safety replan because an exposed group "
                "continuation is still queued"
            )
            return

        first_key = min(
            int(pt.group_key) for pt in offline if pt.group_key is not None
        )
        rebuilt: List[_PendingToken] = []
        next_key = first_key
        i = 0
        while i < len(pending):
            head = pending[i]
            if head.group_key is None:
                rebuilt.append(head)
                i += 1
                continue
            old_plan = head.plan
            if old_plan is None:
                raise RuntimeError(
                    f"pre-split group {head.group_key} is missing its capacity contract"
                )

            # Object identity is the packet/cohort identity.  Equal-valued
            # plans from two LONG_SEGMENT packets must remain separate.
            block: List[_PendingToken] = []
            while (
                i < len(pending)
                and pending[i].group_key is not None
                and pending[i].plan is old_plan
            ):
                block.append(pending[i])
                i += 1

            if old_plan.thresholds.force_split_at > live_plan.thresholds.force_split_at:
                block_plan = _PresplitPlan(
                    safety_ratio=live_plan.safety_ratio,
                    thresholds=live_plan.thresholds,
                )
                block_tokens = [pt.token for pt in block]
                block_pending, next_key = self._build_planned_pending(
                    block_tokens,
                    plan=block_plan,
                    first_group_key=next_key,
                )
                for old_pending, new_pending in zip(block, block_pending):
                    new_pending.boundary_before = old_pending.boundary_before
                    new_pending.forced_boundary = old_pending.forced_boundary
                    if new_pending.forced_boundary:
                        new_pending.boundary = True
                rebuilt.extend(block_pending)
                if self._record_decisions:
                    self._split_decisions.append(
                        {
                            "obs": "presplit_replan",
                            "old_capacity": old_plan.thresholds.force_split_at,
                            "new_capacity": live_plan.thresholds.force_split_at,
                            "token_count": len(block_tokens),
                            "old_group_count": sum(pt.boundary for pt in block),
                            "new_group_count": sum(pt.boundary for pt in block_pending),
                            "safety_ratio": round(live_plan.safety_ratio, 3),
                        }
                    )
                continue

            # This packet was already planned at least as conservatively.
            current_group: List[_PendingToken] = []
            for pt in block:
                current_group.append(pt)
                if not pt.boundary:
                    continue
                last = len(current_group) - 1
                rebuilt.extend(
                    _PendingToken(
                        pending.token,
                        next_key,
                        boundary=(index == last) or pending.forced_boundary,
                        boundary_before=pending.boundary_before,
                        forced_boundary=pending.forced_boundary,
                        plan=old_plan,
                    )
                    for index, pending in enumerate(current_group)
                )
                next_key += 1
                current_group = []
            if current_group:
                raise RuntimeError("unterminated pre-split group in pending queue")

        self._pending = deque(rebuilt)
        self._next_group_key = max(self._next_group_key, next_key)

    def push_group_tokens(
        self,
        tokens: List[SegmentToken],
        *,
        force_boundary: bool = False,
        force_boundary_before: bool = False,
    ) -> List[SegmentAction]:
        """Queue one complete long-segment unit for group-level pre-splitting.

        Unlike ``set_full_text()``, this does not mark the session text-complete.
        Each incoming long segment is treated as a self-contained unit that may
        be further pre-split into one or more groups before the second-layer
        driver takes over.
        """
        if not tokens:
            return []
        pending_start = len(self._pending)
        self._enqueue_presplit_groups(tokens)
        added = list(self._pending)[pending_start:]
        if added and force_boundary_before:
            added[0].boundary_before = True
        if added and force_boundary:
            added[-1].boundary = True
            added[-1].forced_boundary = True
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

        while self._pending and self._pending[0].group_key == cur_key:
            if self._pending[0].boundary_before and driver.token_count > 0:
                end_actions, _ = self._results_to_actions(
                    active_idx,
                    driver.feed(SpliterEvent(type=ET.END)),
                    group_idx=group_idx,
                    local_idx=local_idx,
                    group_final=is_stream,
                )
                out.extend(end_actions)
                self._finalize_segment(active_idx, out, group_exhausted=True)
                flushed = True
                break
            pt = self._pending.popleft()
            evt = self._make_event(
                pt.token.token_id, pt.token.text, pt.token.punct_level
            )
            tok_actions, flushed = self._results_to_actions(
                active_idx,
                driver.feed(evt),
                token_text=pt.token.text,
                token=pt.token,
                group_idx=group_idx,
                local_idx=local_idx,
                group_final=is_stream,
            )
            out.extend(tok_actions)
            if flushed:
                self._finalize_segment(active_idx, out, group_exhausted=pt.boundary)
                break
            if pt.boundary:
                end_actions, _ = self._results_to_actions(
                    active_idx,
                    driver.feed(SpliterEvent(type=ET.END)),
                    group_idx=group_idx,
                    local_idx=local_idx,
                    group_final=is_stream,
                )
                out.extend(end_actions)
                self._finalize_segment(active_idx, out, group_exhausted=True)
                flushed = True
                break

        # Auto handoff (Step 3): if the next pending token belongs to a different
        # group than the open segment, close the open segment at a clean
        # boundary so the new group starts fresh. Does not trigger in pure
        # streaming/offline sessions (one group key throughout).
        if not flushed and self._pending and self._pending[0].group_key != cur_key:
            end_actions, _ = self._results_to_actions(
                active_idx,
                driver.feed(SpliterEvent(type=ET.END)),
                group_idx=group_idx,
                local_idx=local_idx,
                group_final=is_stream,
            )
            out.extend(end_actions)
            self._finalize_segment(active_idx, out, group_exhausted=True)
            flushed = True

        # End-of-input: flush the open streaming segment once the queue drains.
        if not flushed and self._input_complete and not self._pending:
            end_actions, _ = self._results_to_actions(
                active_idx,
                driver.feed(SpliterEvent(type=ET.END)),
                group_idx=group_idx,
                local_idx=local_idx,
                group_final=is_stream,
            )
            out.extend(end_actions)
            self._finalize_segment(active_idx, out, group_exhausted=True)

        return out

    def _open_segment(self, out: List[SegmentAction]) -> int:
        """Create a driver for the next pending token's group, assign its
        ``(group_idx, local_idx)`` coordinate (Option ①), emit START, and
        return the new segment index."""
        pending_head = self._pending[0]
        key = pending_head.group_key
        if key is not None:
            plan = pending_head.plan
            if plan is None:
                raise RuntimeError(
                    f"pre-split group {key} is missing its capacity contract"
                )
            # Obey mode: Stage 1 (pre_split bin-packing) already chose this
            # group's boundary, fed as a forced END at the group's last token.
            # Raise the driver's L1/L2/L3 thresholds to capacity so it does not
            # re-fragment the packed group at internal punctuation.  Crucially,
            # use the exact Stage-1 plan rather than a live ratio: changing the
            # cap here used to split 109-token groups into 108+1 / 106+3 tails.
            cap = plan.thresholds.force_split_at
            thresholds = SplitThresholds(
                min_tokens_l1=cap,
                min_tokens_l2=cap,
                min_tokens_l3=cap,
                force_split_at=cap,
            )
            segment_safety_ratio = plan.safety_ratio
        else:
            # True streaming has no Stage-1 boundary contract.  A newly opened
            # segment may therefore use the latest monotonic safety ratio.
            thresholds = self._make_thresholds()
            segment_safety_ratio = self.safety_ratio
        idx, driver = self._create_driver(thresholds)
        self._seg_ema_ratio[idx] = self.ema_ratio
        self._seg_safety_ratio[idx] = segment_safety_ratio
        if key is None:
            # Streaming: every opened segment is one public group.
            group_idx, local_idx = self._next_output_group_idx, 0
            self._next_output_group_idx += 1
        else:
            # Offline planning keys are intentionally private. Allocate the
            # public reorder coordinate only when this FIFO group actually
            # opens; all local fragments of the same planned group reuse it.
            if key not in self._group_output_idx:
                self._group_output_idx[key] = self._next_output_group_idx
                self._next_output_group_idx += 1
            group_idx = self._group_output_idx[key]
            local_idx = self._group_next_local.get(key, 0)
        self._seg_coords[idx] = (group_idx, local_idx)
        self._seg_group_key[idx] = key
        start_actions, _ = self._results_to_actions(
            idx,
            driver.feed(SpliterEvent(type=ET.START)),
            group_idx=group_idx,
            local_idx=local_idx,
            group_final=(key is None),
        )
        out.extend(start_actions)
        return idx

    def _finalize_segment(
        self,
        idx: int,
        actions: List[SegmentAction],
        *,
        group_exhausted: bool,
    ) -> None:
        """Stamp ``group_final`` on a just-flushed segment's actions and advance
        its group's local counter. Streaming segments are always final; an
        offline segment is final only when its group is exhausted."""
        key = self._seg_group_key[idx]
        group_final = key is None or group_exhausted
        for sa in actions:
            if sa.segment_idx == idx:
                sa.group_final = group_final
        if key is not None:
            if not group_exhausted:
                self._group_next_local[key] = self._seg_coords[idx][1] + 1
            else:
                self._group_next_local.pop(key, None)
                self._group_output_idx.pop(key, None)

    # ------------------------------------------------------------------
    # Public API: streaming
    # ------------------------------------------------------------------

    def feed_tokens(
        self,
        tokens: List[SegmentToken],
        *,
        force_boundary: bool = False,
        force_boundary_before: bool = False,
    ) -> List[SegmentAction]:
        """Streaming mode: queue tokens (each its own group) and drive.

        Tokens accumulate into the active Driver across calls; when it flushes,
        the next segment opens if concurrency allows, otherwise tokens wait in
        the shared pending queue. The driver decides flush points (no boundary).
        """
        coerced = self._coerce_tokens(tokens)
        for tok in coerced:
            self._pending.append(_PendingToken(tok, None, boundary=False))
        if coerced and force_boundary_before:
            self._pending[-len(coerced)].boundary_before = True
        if coerced and force_boundary:
            self._pending[-1].boundary = True
            self._pending[-1].forced_boundary = True
        return self._drive_events()

    def feed_auto(
        self,
        tokens: List[SegmentToken],
        *,
        force_boundary: bool = False,
        force_boundary_before: bool = False,
    ) -> List[SegmentAction]:
        """Auto mode: route a packet by size; Stage 1 engages only when long.

        Stage 1 (offline pre-split) adds value only through *global foresight*,
        which exists only when a packet is longer than one segment can hold. For
        a small streaming packet, Stage 1 sees no more than the driver would, so
        it is transparent: the tokens stream (``group_idx=None``) and coalesce
        across packets exactly like plain streaming, and the driver segments
        them. A long packet (more tokens than one segment's capacity) is
        pre-split with foresight, yielding offline-quality L1 boundaries.

        Quality therefore scales with packet size, with no buffering latency:
        the client implicitly chooses the optimization scope by how much text it
        hands over per packet.
        """
        coerced = self._coerce_tokens(tokens)
        if not coerced:
            return []
        # Route against the active segment's frozen capacity.  A delayed EMA
        # update may change the capacity for future segments, but must not make
        # this gate disagree with the already-open driver's hard limit.
        active = self._get_active_driver_idx()
        if active is None:
            capacity = self._make_thresholds().force_split_at
            occupied = 0
        else:
            driver = self._drivers[active]
            capacity = driver.thresholds.force_split_at
            occupied = driver.token_count
        if len(coerced) > capacity - occupied:
            self._enqueue_presplit_groups(
                coerced
            )  # won't fit remaining room: Stage 1 foresight
        else:
            for tok in coerced:  # fits: transparent stream (coalesce)
                self._pending.append(_PendingToken(tok, None, boundary=False))
        if coerced and force_boundary_before:
            self._pending[-len(coerced)].boundary_before = True
        if coerced and force_boundary:
            self._pending[-1].boundary = True
            self._pending[-1].forced_boundary = True
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
        self._seg_ema_ratio.pop(segment_idx, None)
        self._seg_safety_ratio.pop(segment_idx, None)

        return self._drive_events()

    def set_duration_ratio(self, ratio: float) -> None:
        """Apply legacy external duration feedback without weakening safety."""
        safety_before = self.safety_ratio
        self._ratios.set_duration_ratio(ratio)
        if self.safety_ratio > safety_before:
            self._tighten_pending_presplit_plans()

    def observe_segment(
        self,
        actual_audio_steps: int,
        actual_text_tokens: int,
        *,
        outcome: RatioOutcome,
        segment_idx: Optional[int] = None,
    ) -> RatioObservation:
        """Route one result to duration estimation or safety backoff.

        Natural ``codec_eos`` observations may update the duration EMA.
        Overflow/loop/length observations are censored or hallucination-
        inflated, so they only tighten the independent capacity controller.
        Other failures affect neither estimator.
        """
        segment_safety = (
            self.safety_ratio_for_segment(segment_idx)
            if segment_idx is not None
            else None
        )
        observation = self._ratios.observe(
            audio_steps=actual_audio_steps,
            text_tokens=actual_text_tokens,
            outcome=outcome,
            segment_safety_ratio=segment_safety,
        )
        if observation.safety_tightened:
            self._tighten_pending_presplit_plans()

        if outcome is RatioOutcome.KV_OVERFLOW:
            logger.warning(
                "Safety ratio overflow backoff: observed_lower_bound=%.3f "
                "safety=%.3f→%.3f (steps=%d tokens=%d segment=%s)",
                observation.observed_ratio,
                observation.safety_before,
                observation.safety_after,
                observation.audio_steps,
                observation.text_tokens,
                segment_idx,
            )
        elif outcome in (RatioOutcome.LOOP_ABORT, RatioOutcome.LENGTH_ABORT):
            logger.warning(
                "Safety ratio failure backoff: outcome=%s safety=%.3f→%.3f "
                "(steps=%d tokens=%d segment=%s)",
                outcome.value,
                observation.safety_before,
                observation.safety_after,
                observation.audio_steps,
                observation.text_tokens,
                segment_idx,
            )
        elif observation.duration_sample_accepted:
            logger.debug(
                "Duration EMA update: observed=%.3f duration=%.3f→%.3f "
                "safety=%.3f (steps=%d tokens=%d segment=%s)",
                observation.observed_ratio,
                observation.duration_before,
                observation.duration_after,
                observation.safety_after,
                observation.audio_steps,
                observation.text_tokens,
                segment_idx,
            )

        return observation

    def update_ratio(
        self,
        actual_audio_steps: int,
        actual_text_tokens: int,
        *,
        overflow: bool = False,
    ) -> None:
        """Backward-compatible wrapper around typed segment observation.

        ``overflow=True`` no longer feeds a censored sample into the duration
        EMA.  It tightens only the independent safety controller.

        Parameters
        ----------
        overflow : bool
            When True, uses ``ema_overflow_alpha`` (default 0.5) for safety
            backoff and also respects the overflow ratio as a censored lower
            bound.  It never updates the duration EMA.
        """
        self.observe_segment(
            actual_audio_steps,
            actual_text_tokens,
            outcome=(
                RatioOutcome.KV_OVERFLOW if overflow else RatioOutcome.CODEC_EOS
            ),
        )

    # ------------------------------------------------------------------
    # Full reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._pending.clear()
        self._next_group_key = 0
        self._next_output_group_idx = 0
        self._input_complete = False
        self._drivers.clear()
        self._next_segment_idx = 0
        self._seg_coords.clear()
        self._seg_group_key.clear()
        self._seg_ema_ratio.clear()
        self._seg_safety_ratio.clear()
        self._group_next_local.clear()
        self._group_output_idx.clear()
        self._flushing.clear()
        self._done.clear()
        self._ratios.reset()
        self._split_decisions.clear()

    def _coerce_tokens(self, tokens) -> List[SegmentToken]:
        """Accept legacy tuple tokens at the API edge, normalize internally."""
        normalized: List[SegmentToken] = []
        for token in tokens:
            if isinstance(token, SegmentToken):
                normalized.append(token)
                continue
            token_id, text = token[:2]
            punct_level = (
                token[2] if len(token) > 2 else self.classify_punct_level(text)
            )
            normalized.append(
                SegmentToken(
                    token_id=token_id,
                    text=text,
                    punct_level=punct_level,
                )
            )
        return normalized


__all__ = ("Spliter", "SegmentAction")
