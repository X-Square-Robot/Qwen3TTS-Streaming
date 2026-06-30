"""
Generic guard-based finite state machine.

Transition rules are evaluated in registration order; first matching
guard wins.  Each transition optionally carries an action callback
whose return value is forwarded to the caller of ``step()``.

Usage (chaining)::

    fsm = FSM(State.A)
    fsm.when(State.A, target=State.B, guard=lambda e: e.x == 1)
    fsm.when(State.A, target=State.C, guard=lambda e: e.x == 2)
    new_state, result = fsm.step(event)

Usage (declarative table)::

    fsm = FSM.from_table(
        initial=State.A,
        table={
            State.A: [Rule(State.B)],
            State.B: [Rule(State.C, guard=cond1, action=f2),
                       Rule(State.D, guard=cond2, action=f3)],
        },
        on_enter={State.B: f1},
    )
    new_state, result = fsm.step(event)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=Enum)
E = TypeVar("E")


def ALWAYS(_event: Any) -> bool:
    """Sentinel guard that always returns True (unconditional transition)."""
    return True


@dataclass
class Rule:
    """One transition rule: guard → target state, with optional action.

    Parameters
    ----------
    target : S
        Destination state.
    guard : callable
        ``guard(event) -> bool``.  Defaults to ``ALWAYS`` (unconditional).
    action : callable, optional
        ``action(event) -> Any``.  Return value is forwarded via ``step()``.
    name : str
        Label for logging / introspection.
    """

    target: Any
    guard: Callable[..., bool] = ALWAYS
    action: Optional[Callable[..., Any]] = None
    name: str = ""


class FSM(Generic[S, E]):
    """
    Table-driven FSM with guard predicates.

    Parameters
    ----------
    initial_state : S
        The state the machine starts in.
    """

    def __init__(self, initial_state: S) -> None:
        self._state: S = initial_state
        self._initial: S = initial_state
        self._table: Dict[S, List[Rule]] = {}
        self._on_enter: Dict[S, Callable[[S, S, E], None]] = {}
        self._on_exit: Dict[S, Callable[[S, S, E], None]] = {}

    # ---- class method constructors -----------------------------------------

    @classmethod
    def from_table(
        cls,
        initial: S,
        table: Dict[S, List[Rule]],
        *,
        on_enter: Optional[Dict[S, Callable]] = None,
        on_exit: Optional[Dict[S, Callable]] = None,
    ) -> "FSM[S, E]":
        """Build an FSM from a declarative transition table.

        Parameters
        ----------
        initial : S
            Initial state.
        table : dict
            ``{source_state: [Rule(...), ...], ...}``
        on_enter : dict, optional
            ``{state: callback(old, new, event), ...}``
        on_exit : dict, optional
            ``{state: callback(old, new, event), ...}``
        """
        fsm = cls(initial)
        for source, rules in table.items():
            fsm._table[source] = list(rules)
        if on_enter:
            fsm._on_enter.update(on_enter)
        if on_exit:
            fsm._on_exit.update(on_exit)
        return fsm

    # ---- properties --------------------------------------------------------

    @property
    def state(self) -> S:
        return self._state

    # ---- registration API (imperative, for backward compat) ----------------

    def when(
        self,
        from_state: S,
        *,
        target: S,
        guard: Callable[[E], bool] = ALWAYS,
        action: Optional[Callable[[E], Any]] = None,
        name: str = "",
    ) -> "FSM[S, E]":
        """Register a guarded transition.

        Returns *self* for chaining.
        """
        self._table.setdefault(from_state, []).append(
            Rule(target=target, guard=guard, action=action, name=name)
        )
        return self

    def on_enter(self, state: S, callback: Callable[[S, S, E], None]) -> "FSM[S, E]":
        """Register a callback invoked when *state* is entered.

        Signature: ``callback(old_state, new_state, event)``
        """
        self._on_enter[state] = callback
        return self

    def on_exit(self, state: S, callback: Callable[[S, S, E], None]) -> "FSM[S, E]":
        """Register a callback invoked when *state* is exited.

        Signature: ``callback(old_state, new_state, event)``
        """
        self._on_exit[state] = callback
        return self

    # ---- runtime -----------------------------------------------------------

    def step(self, event: E) -> Tuple[S, Any]:
        """Advance the FSM by one event.

        Returns ``(new_state, action_result)``.  If no guard matches,
        the state is unchanged and ``action_result`` is ``None``.
        """
        rules = self._table.get(self._state, [])
        for rule in rules:
            if rule.guard(event):
                old = self._state
                new = rule.target

                if old in self._on_exit:
                    self._on_exit[old](old, new, event)

                result = rule.action(event) if rule.action else None
                self._state = new

                if new in self._on_enter:
                    self._on_enter[new](old, new, event)

                logger.debug(
                    "%s -[%s]-> %s",
                    old.value if isinstance(old, Enum) else old,
                    rule.name or "?",
                    new.value if isinstance(new, Enum) else new,
                )
                return new, result

        return self._state, None

    def reset(self) -> None:
        """Reset to the initial state."""
        self._state = self._initial

    # ---- introspection -----------------------------------------------------

    def transitions_from(self, state: S) -> List[Tuple[str, S]]:
        """Return ``[(rule_name, target), ...]`` for debugging."""
        return [(r.name, r.target) for r in self._table.get(state, [])]

    def __repr__(self) -> str:
        s = self._state.value if isinstance(self._state, Enum) else self._state
        return f"FSM(state={s})"


__all__ = ("FSM", "Rule", "ALWAYS")
