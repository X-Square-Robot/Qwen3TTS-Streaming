"""Generic causal-frontier primitives used by the X2 text committer.

The module contains no TN rules.  It only turns a set of possible spoken
outputs into an append-only, unit-aligned frontier.  A WeText/FST adapter can
provide the candidates; deadlines, fallbacks and raw-span ownership remain in
the session controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


class FrontierViolation(RuntimeError):
    """Raised when a backend attempts to retract an already emitted prefix."""


_UNIT_RE = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)?|"
    r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U000323af]|"
    r"\s+|[^\w\s]",
    re.UNICODE,
)


def spoken_units(text: str) -> tuple[str, ...]:
    """Tokenize spoken output into commit-safe lexical units.

    CJK ideographs are individual units; Latin words and numeric strings are
    kept whole.  Whitespace is retained for reconstruction but is never
    emitted by itself as the last unit of a frontier.
    """

    if not text:
        return ()
    return tuple(_UNIT_RE.findall(text))


def longest_common_unit_prefix(outputs: Iterable[str]) -> tuple[str, ...]:
    """Return the longest common *complete-unit* prefix of candidate outputs."""

    tokenized = [spoken_units(value) for value in outputs]
    if not tokenized:
        return ()
    prefix: list[str] = []
    for columns in zip(*tokenized):
        if len(set(columns)) != 1:
            break
        prefix.append(columns[0])
    # Never release a dangling separator.  It can change when the next word is
    # appended (and it creates awkward TTS chunks such as ``word ``).
    while prefix and prefix[-1].isspace():
        prefix.pop()
    return tuple(prefix)


@dataclass(frozen=True)
class FrontierUpdate:
    """Result of advancing one candidate frontier."""

    stable_delta: str = ""
    stable_text: str = ""
    pending_text: str = ""
    extendable: bool = True
    closed: bool = False


class CausalFrontier:
    """Monotonic spoken-prefix controller.

    The controller is deliberately independent of raw offsets.  The owning
    span controller maps ``stable_delta`` back to a ``TextCommit`` and freezes
    its raw fence.  Calling ``advance`` with a candidate set that retracts the
    prior frontier raises instead of silently rewriting output.
    """

    def __init__(self) -> None:
        self._committed_units: tuple[str, ...] = ()
        self._committed_text = ""

    @property
    def committed_text(self) -> str:
        return self._committed_text

    @property
    def committed_units(self) -> tuple[str, ...]:
        return self._committed_units

    def reset(self) -> None:
        self._committed_units = ()
        self._committed_text = ""

    def observe_committed(self, text: str) -> FrontierUpdate:
        """Record an already-selected append-only commit by character prefix.

        Closed-span controllers often split a Latin word at a transport or
        lexical boundary.  Re-tokenizing the cumulative text can therefore
        regroup units (``ab`` + ``c`` → ``abc``) even though no character was
        retracted.  This method supplies the strict character-prefix guard for
        that path; :meth:`advance` remains the n-best/unit frontier API.
        """

        value = str(text or "")
        if not value.startswith(self._committed_text):
            raise FrontierViolation(
                "committed text retracts an already committed spoken prefix"
            )
        old = self._committed_text
        self._committed_text = value
        self._committed_units = spoken_units(value)
        return FrontierUpdate(
            stable_delta=value[len(old) :],
            stable_text=value,
            extendable=False,
            closed=True,
        )

    def advance(
        self,
        candidates: Iterable[str],
        *,
        pending_text: str = "",
        extendable: bool = True,
        closed: bool = False,
    ) -> FrontierUpdate:
        values = tuple(str(value) for value in candidates if value is not None)
        if not values:
            return FrontierUpdate(
                stable_text=self._committed_text,
                pending_text=pending_text,
                extendable=extendable,
                closed=closed,
            )
        common = longest_common_unit_prefix(values)
        old = self._committed_units
        if len(common) < len(old) or common[: len(old)] != old:
            raise FrontierViolation(
                "candidate frontier retracts an already committed spoken prefix"
            )
        # If the span is closed, its selected candidate is final even when the
        # candidates disagree.  The caller should pass one deterministic
        # candidate in that case; keeping this guard here prevents accidental
        # emission of a partial word.
        if closed and not common:
            common = spoken_units(values[0])
        delta_units = common[len(old) :]
        delta = "".join(delta_units)
        self._committed_units = common
        self._committed_text = "".join(common)
        return FrontierUpdate(
            stable_delta=delta,
            stable_text=self._committed_text,
            pending_text=pending_text,
            extendable=extendable,
            closed=closed,
        )


__all__ = [
    "CausalFrontier",
    "FrontierUpdate",
    "FrontierViolation",
    "longest_common_unit_prefix",
    "spoken_units",
]
