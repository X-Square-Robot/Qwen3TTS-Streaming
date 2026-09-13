"""Table-driven lifecycle FSM for incremental semantic spans.

The committer remains the owner of lexical scanning and normalization.  This
module only describes the control-plane contract around it: typed events,
session context, and lifecycle transitions.  Keeping that boundary explicit
allows callers to drive or observe commitment state without duplicating TN
rules in another state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from ..spliter.core import FSM, Rule
from ..text_commitment.types import (
    CommitmentState,
    CommitDecision,
    SpanKind,
)
from .events import SpanEvent, SpanEventType

SpanState = CommitmentState


def _type_guard(event_type: SpanEventType) -> SpanGuard:
    return lambda _context, event: event.type is event_type


@dataclass(frozen=True)
class SpanContext:
    """Small, serializable snapshot shared by guards and transition actions."""

    state: CommitmentState = CommitmentState.SCAN
    pending_raw: str = ""
    pending_kind: SpanKind | None = None
    raw_cursor: int = 0
    committed_raw_end: int = 0
    finalized: bool = False
    sequence: int = 0
    last_event: SpanEventType | None = None

    def apply_decision(self, decision: CommitDecision) -> "SpanContext":
        """Project a committer decision into the control-plane snapshot."""

        return replace(
            self,
            state=decision.state,
            pending_raw=decision.pending_raw,
            pending_kind=decision.pending_kind,
            committed_raw_end=decision.committed_raw_end,
            finalized=decision.state is CommitmentState.DONE,
        )


SpanGuard = Callable[[SpanContext, SpanEvent], bool]
SpanAction = Callable[[SpanContext, SpanEvent], SpanContext | None]


@dataclass(frozen=True)
class SpanTransition:
    """One row in a declarative span transition table."""

    source: CommitmentState
    event: SpanEventType
    target: CommitmentState
    guard: SpanGuard = lambda _context, _event: True
    action: SpanAction | None = None
    name: str = ""


@dataclass(frozen=True)
class SpanStep:
    """Result of feeding one event to :class:`SpanFSM`."""

    previous: CommitmentState
    state: CommitmentState
    event: SpanEvent
    context: SpanContext
    transitioned: bool
    action_result: Any = None


def _decision_state(state: CommitmentState) -> SpanGuard:
    def guard(_context: SpanContext, event: SpanEvent) -> bool:
        return event.type is SpanEventType.DECISION and event.decision is not None and event.decision.state is state

    return guard


def default_span_transitions() -> tuple[SpanTransition, ...]:
    """Return conservative lifecycle rows used by ``SpanFSM()``.

    The DECISION rows make this useful as an observer around
    ``IncrementalTextCommitter.feed``/``poll``.  Explicit lifecycle events are
    retained for integrations that run normalization asynchronously.
    """

    rows: list[SpanTransition] = []
    for state in CommitmentState:
        rows.append(
            SpanTransition(
                state,
                SpanEventType.RESET,
                CommitmentState.SCAN,
                guard=_type_guard(SpanEventType.RESET),
                name="reset",
            )
        )

    # A decision is authoritative because the committer owns semantic policy.
    for source in CommitmentState:
        for target in CommitmentState:
            rows.append(
                SpanTransition(
                    source,
                    SpanEventType.DECISION,
                    target,
                    guard=_decision_state(target),
                    name=f"decision_{target.value}",
                )
            )

    rows.extend(
        (
            SpanTransition(CommitmentState.SCAN, SpanEventType.INPUT, CommitmentState.OPEN, guard=lambda _c, e: bool(e.text), name="open"),
            SpanTransition(CommitmentState.OPEN, SpanEventType.INPUT, CommitmentState.OPEN, name="append"),
            SpanTransition(CommitmentState.OPEN, SpanEventType.SPAN_READY, CommitmentState.READY, name="ready"),
            SpanTransition(CommitmentState.READY, SpanEventType.NORMALIZE_START, CommitmentState.NORMALIZE, name="normalize"),
            SpanTransition(CommitmentState.NORMALIZE, SpanEventType.NORMALIZE_FINISHED, CommitmentState.COMMIT, name="normalized"),
            SpanTransition(CommitmentState.COMMIT, SpanEventType.INPUT, CommitmentState.OPEN, guard=lambda _c, e: bool(e.text), name="next_span"),
            SpanTransition(CommitmentState.OPEN, SpanEventType.TIMEOUT, CommitmentState.FALLBACK, name="timeout"),
            SpanTransition(CommitmentState.OPEN, SpanEventType.FALLBACK, CommitmentState.FALLBACK, name="fallback"),
            SpanTransition(CommitmentState.FALLBACK, SpanEventType.INPUT, CommitmentState.OPEN, guard=lambda _c, e: bool(e.text), name="late_extension"),
            SpanTransition(CommitmentState.OPEN, SpanEventType.FINALIZE, CommitmentState.DONE, name="finalize"),
            SpanTransition(CommitmentState.COMMIT, SpanEventType.FINALIZE, CommitmentState.DONE, name="finalize"),
            SpanTransition(CommitmentState.FALLBACK, SpanEventType.FINALIZE, CommitmentState.DONE, name="finalize"),
            SpanTransition(CommitmentState.SCAN, SpanEventType.FINALIZE, CommitmentState.DONE, name="empty_finalize"),
        )
    )
    return tuple(rows)


class SpanFSM:
    """Typed facade over the repository's generic table-driven ``FSM``."""

    def __init__(
        self,
        *,
        initial: CommitmentState = CommitmentState.SCAN,
        transitions: Sequence[SpanTransition] | None = None,
        context: SpanContext | None = None,
    ) -> None:
        self._initial = initial
        self._context = context or SpanContext(state=initial)
        rows = tuple(transitions or default_span_transitions())
        self._transitions = rows
        table: dict[CommitmentState, list[Rule]] = {}
        for row in rows:
            def guard(event: SpanEvent, row: SpanTransition = row) -> bool:
                # ``FSM`` tables are keyed by state, while this contract is
                # keyed by state *and* event.  Always gate on the declared
                # event first so an unconditional row for ``INPUT`` cannot
                # swallow ``FINALIZE`` or another lifecycle signal.
                return event.type is row.event and row.guard(self._context, event)

            def action(event: SpanEvent, row: SpanTransition = row) -> Any:
                if row.action is None:
                    return None
                updated = row.action(self._context, event)
                if updated is not None:
                    self._context = updated
                return updated

            table.setdefault(row.source, []).append(
                Rule(target=row.target, guard=guard, action=action, name=row.name)
            )
        self._fsm: FSM[CommitmentState, SpanEvent] = FSM.from_table(initial, table)

    @property
    def state(self) -> CommitmentState:
        return self._fsm.state

    @property
    def context(self) -> SpanContext:
        return self._context

    @property
    def transitions(self) -> tuple[SpanTransition, ...]:
        return self._transitions

    def step(self, event: SpanEvent) -> SpanStep:
        previous = self.state
        state, result = self._fsm.step(event)
        if event.decision is not None:
            self._context = self._context.apply_decision(event.decision)
        self._context = replace(
            self._context,
            state=state,
            raw_cursor=self._context.raw_cursor + len(event.text),
            sequence=self._context.sequence + 1,
            last_event=event.type,
            finalized=state is CommitmentState.DONE,
        )
        return SpanStep(previous, state, event, self._context, state is not previous, result)

    def reset(self) -> None:
        self._fsm.reset()
        self._context = SpanContext(state=self._initial)

    def transitions_from(self, state: CommitmentState) -> tuple[tuple[str, CommitmentState], ...]:
        return tuple((row.name, row.target) for row in self._transitions if row.source is state)


__all__ = (
    "SpanEventType",
    "SpanState",
    "SpanEvent",
    "SpanContext",
    "SpanTransition",
    "SpanStep",
    "SpanFSM",
    "default_span_transitions",
)
