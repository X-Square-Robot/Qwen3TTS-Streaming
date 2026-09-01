from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List

from .core import FSM, Rule, ALWAYS
from .state import SpliterState as St
from .event import SpliterEvent, SpliterEventType as ET

logger = logging.getLogger(__name__)

_TRANSIENT = frozenset({St.PREFILL, St.PAD_TEXT_EOS, St.PAD_TEXT_NOP})

# ---------------------------------------------------------------------------
# Action results — returned to the caller via feed()
# ---------------------------------------------------------------------------


class ActionType(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    FLUSH_EOS = "flush_eos"
    FLUSH_NOP = "flush_nop"


@dataclass
class ActionResult:
    type: ActionType
    token: int = -1


# ---------------------------------------------------------------------------
# Threshold computation — mirrors decode_fsm.py logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitThresholds:
    """Min text-token counts before split at punct tier + forced upper bound.

    Aligns with ``Spliter.classify_punct_level`` / ``SpliterEvent.punct_level``:
    L1 (。！？), L2 (，；：), L3 (weaker breaks).  ``force_split_at`` caps segment
    length regardless of punctuation.
    """

    min_tokens_l1: int
    min_tokens_l2: int
    min_tokens_l3: int
    force_split_at: int


def compute_thresholds(
    remaining_kv: int,
    ema_ratio: float,
    safety_margin: int = 8,
    l1_cap_ratio: float = 0.70,
    l2_cap_ratio: float = 0.80,
    l3_cap_ratio: float = 0.90,
) -> SplitThresholds:
    """Compute the frozen token thresholds for one segment.

    ``remaining_kv`` is the KV budget after prefill.  ``ema_ratio`` is the
    caller's conservative planning ratio (kept as the historical parameter
    name for API compatibility), not necessarily the duration EMA.
    ``safety_margin`` is reserved before converting that budget to the
    predicted text-token capacity ``cap``.  The punctuation thresholds are
    ceilings of their configured capacity ratios, while ``force_split_at``
    remains exactly ``cap``; tier spacing must never enlarge the
    resource-derived capacity.

    For small capacities two tiers may coincide.  That is intentional: the
    punctuation preference still determines which boundary is chosen, while
    the common hard capacity remains enforceable.
    """
    ratios = (l1_cap_ratio, l2_cap_ratio, l3_cap_ratio)
    if not all(math.isfinite(ratio) for ratio in ratios):
        raise ValueError("split capacity ratios must be finite")
    if not 0.0 < ratios[0] <= ratios[1] <= ratios[2] <= 1.0:
        raise ValueError("split capacity ratios must satisfy 0 < L1 <= L2 <= L3 <= 1")
    if safety_margin < 0:
        raise ValueError("safety_margin must be nonnegative")
    if not math.isfinite(ema_ratio) or ema_ratio < 1.0:
        raise ValueError("ema_ratio must be finite and at least 1.0")

    usable_kv = remaining_kv - safety_margin
    cap = math.floor(usable_kv / ema_ratio)
    if cap < 1:
        raise ValueError(
            "remaining KV budget cannot accommodate one predicted text token "
            "after the safety margin"
        )

    return SplitThresholds(
        min_tokens_l1=math.ceil(cap * l1_cap_ratio),
        min_tokens_l2=math.ceil(cap * l2_cap_ratio),
        min_tokens_l3=math.ceil(cap * l3_cap_ratio),
        force_split_at=cap,
    )


# ---------------------------------------------------------------------------
# FSM design (mermaid reference)
# ---------------------------------------------------------------------------
"""
Three-tier punctuation threshold state machine
================================================

The Driver uses 4 thresholds (min_tokens_l1 <= … <= force_split_at) computed
from remaining KV budget and a conservative audio:text planning ratio.

Guards use the length *after* the triggering token is appended:
  - next_count >= min_tokens_l1 AND punct_level == 1 (L1) → split
  - next_count >= min_tokens_l2 AND punct_level == 2 (L2) → split
  - next_count >= min_tokens_l3 AND punct_level == 3 (L3) → split
  - next_count >= force_split_at (any token)              → forced split

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE --> IDLE : unknown / ()
    IDLE --> IDLE : START signal / reset_context()
    IDLE --> HALT : END signal / ()
    IDLE --> PREFILL : start/normal/punct token / prefill()
    IDLE --> IDLE : end_token / ()

    PREFILL --> PAD_TEXT_EOS : first token reaches a threshold / ()
    PREFILL --> TEXT_INPUTING : otherwise / ()

    TEXT_INPUTING --> TEXT_INPUTING : unknown/START/start_token / ()
    TEXT_INPUTING --> PAD_TEXT_EOS : END signal / set_final()
    TEXT_INPUTING --> PAD_TEXT_NOP : end_token / decode()
    TEXT_INPUTING --> PAD_TEXT_EOS : normal [next_count>=force_split_at] / decode()
    TEXT_INPUTING --> TEXT_INPUTING : normal [next_count<force_split_at] / decode()
    TEXT_INPUTING --> PAD_TEXT_EOS : punct [next threshold met] / decode()
    TEXT_INPUTING --> TEXT_INPUTING : punct [threshold not met] / decode()

    PAD_TEXT_EOS --> PAD_TEXT_NOP : Always / ()

    PAD_TEXT_NOP --> HALT : [is_final] / flush_nop()
    PAD_TEXT_NOP --> IDLE : [else] / flush_nop()
```
"""


# ---------------------------------------------------------------------------
# StreamingDriver
# ---------------------------------------------------------------------------


class StreamingDriver:
    """Token-driven FSM for streaming text segmentation with 3-tier thresholds.

    The caller (Spliter) creates one Driver per segment with thresholds
    computed from the current KV budget and EMA ratio.

    Usage::

        thresholds = compute_thresholds(remaining_kv=500, ema_ratio=5.0)
        driver = StreamingDriver(thresholds)
        for event in token_stream:
            results = driver.feed(event)
            ...
    """

    def __init__(self, thresholds: SplitThresholds) -> None:
        self.thresholds = thresholds

        self._is_final = False
        self._token_count = 0

        self._fsm = self._build_fsm()

    # ---- internal punct logic ------------------------------------------

    def _meets_split_threshold(self, e: SpliterEvent, token_count: int) -> bool:
        pl = e.punct_level
        if pl == 1 and token_count >= self.thresholds.min_tokens_l1:
            return True
        if pl == 2 and token_count >= self.thresholds.min_tokens_l2:
            return True
        if pl == 3 and token_count >= self.thresholds.min_tokens_l3:
            return True
        return False

    # ---- guards --------------------------------------------------------

    def _evt(self, *types: ET) -> Callable[[SpliterEvent], bool]:
        s = frozenset(types)
        return lambda e: e.type in s

    def _normal_overflow(self, e: SpliterEvent) -> bool:
        return (
            e.type == ET.NORMAL_TOKEN
            and self._token_count + 1 >= self.thresholds.force_split_at
        )

    def _normal_no_overflow(self, e: SpliterEvent) -> bool:
        return (
            e.type == ET.NORMAL_TOKEN
            and self._token_count + 1 < self.thresholds.force_split_at
        )

    def _punct_should_split(self, e: SpliterEvent) -> bool:
        if e.type != ET.PUNCTUATION_TOKEN:
            return False
        next_count = self._token_count + 1
        return (
            next_count >= self.thresholds.force_split_at
            or self._meets_split_threshold(e, next_count)
        )

    def _punct_no_split(self, e: SpliterEvent) -> bool:
        return e.type == ET.PUNCTUATION_TOKEN and not self._punct_should_split(e)

    def _prefill_should_split(self, e: SpliterEvent) -> bool:
        """Handle the C=1 and first-token punctuation edge cases."""
        if self._token_count >= self.thresholds.force_split_at:
            return True
        return e.type == ET.PUNCTUATION_TOKEN and self._meets_split_threshold(
            e, self._token_count
        )

    def _check_final(self, _e: SpliterEvent) -> bool:
        return self._is_final

    # ---- actions -------------------------------------------------------

    def _nop(self, _e: SpliterEvent) -> None:
        return None

    def _reset_context(self, _e: SpliterEvent) -> None:
        self._is_final = False
        self._token_count = 0
        return None

    def _prefill(self, e: SpliterEvent) -> ActionResult:
        self._token_count = 1
        return ActionResult(ActionType.PREFILL, e.token)

    def _set_final(self, _e: SpliterEvent) -> None:
        self._is_final = True
        return None

    def _decode(self, e: SpliterEvent) -> ActionResult:
        self._token_count += 1
        return ActionResult(ActionType.DECODE, e.token)

    def _flush_eos(self, _e: SpliterEvent) -> ActionResult:
        self._token_count = 0
        return ActionResult(ActionType.FLUSH_EOS)

    def _flush_nop(self, _e: SpliterEvent) -> ActionResult:
        self._token_count = 0
        return ActionResult(ActionType.FLUSH_NOP)

    # ---- FSM construction ----------------------------------------------

    def _build_fsm(self) -> FSM:
        table = {
            St.IDLE: [
                Rule(
                    St.HALT, guard=self._evt(ET.END), action=self._nop, name="idle→halt"
                ),
                Rule(
                    St.IDLE,
                    guard=self._evt(ET.START),
                    action=self._reset_context,
                    name="idle:start",
                ),
                Rule(
                    St.PREFILL,
                    guard=self._evt(
                        ET.START_TOKEN, ET.NORMAL_TOKEN, ET.PUNCTUATION_TOKEN
                    ),
                    action=self._prefill,
                    name="idle→prefill",
                ),
                Rule(
                    St.IDLE,
                    guard=self._evt(ET.END_TOKEN),
                    action=self._nop,
                    name="idle:end_tok_ignore",
                ),
                Rule(
                    St.IDLE,
                    guard=self._evt(ET.UNKNOWN),
                    action=self._nop,
                    name="idle:unknown",
                ),
            ],
            St.PREFILL: [
                Rule(
                    St.PAD_TEXT_EOS,
                    guard=self._prefill_should_split,
                    action=self._nop,
                    name="prefill→pad_eos(threshold)",
                ),
                Rule(
                    St.TEXT_INPUTING,
                    guard=ALWAYS,
                    action=self._nop,
                    name="prefill→input",
                ),
            ],
            St.TEXT_INPUTING: [
                Rule(
                    St.PAD_TEXT_EOS,
                    guard=self._evt(ET.END),
                    action=self._set_final,
                    name="input→pad_eos(final)",
                ),
                Rule(
                    St.PAD_TEXT_NOP,
                    guard=self._evt(ET.END_TOKEN),
                    action=self._decode,
                    name="input→pad_nop",
                ),
                Rule(
                    St.PAD_TEXT_EOS,
                    guard=self._normal_overflow,
                    action=self._decode,
                    name="input→pad_eos(overflow)",
                ),
                Rule(
                    St.TEXT_INPUTING,
                    guard=self._normal_no_overflow,
                    action=self._decode,
                    name="input:normal",
                ),
                Rule(
                    St.PAD_TEXT_EOS,
                    guard=self._punct_should_split,
                    action=self._decode,
                    name="input→pad_eos(punct)",
                ),
                Rule(
                    St.TEXT_INPUTING,
                    guard=self._punct_no_split,
                    action=self._decode,
                    name="input:punct_acc",
                ),
                Rule(
                    St.TEXT_INPUTING,
                    guard=self._evt(ET.START_TOKEN, ET.START, ET.UNKNOWN),
                    action=self._nop,
                    name="input:ignore",
                ),
            ],
            St.PAD_TEXT_EOS: [
                Rule(
                    St.HALT,
                    guard=self._check_final,
                    action=self._flush_eos,
                    name="pad_eos→halt",
                ),
                Rule(
                    St.IDLE, guard=ALWAYS, action=self._flush_eos, name="pad_eos→idle"
                ),
            ],
            St.PAD_TEXT_NOP: [
                Rule(
                    St.HALT,
                    guard=self._check_final,
                    action=self._flush_nop,
                    name="pad_nop→halt",
                ),
                Rule(
                    St.IDLE, guard=ALWAYS, action=self._flush_nop, name="pad_nop→idle"
                ),
            ],
        }

        return FSM.from_table(initial=St.IDLE, table=table)

    # ---- public API ----------------------------------------------------

    @property
    def state(self) -> St:
        return self._fsm.state

    @property
    def is_final(self) -> bool:
        return self._is_final

    @property
    def token_count(self) -> int:
        return self._token_count

    def feed(self, event: SpliterEvent) -> List[ActionResult]:
        """Feed one event; returns action results (may traverse transient states)."""
        results: List[ActionResult] = []
        state, result = self._fsm.step(event)
        if result is not None:
            results.append(result)

        while state in _TRANSIENT:
            prev = state
            state, result = self._fsm.step(event)
            if result is not None:
                results.append(result)
            if state == prev:
                break

        return results

    def reset(self) -> None:
        """Reset FSM and all context to initial state."""
        self._fsm.reset()
        self._is_final = False
        self._token_count = 0


__all__ = (
    "StreamingDriver",
    "ActionType",
    "ActionResult",
    "SplitThresholds",
    "compute_thresholds",
)
