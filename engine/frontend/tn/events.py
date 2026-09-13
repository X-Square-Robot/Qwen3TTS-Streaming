"""Typed events for the frontend TN span lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..text_commitment.types import CommitDecision, TextInputMetadata


class SpanEventType(str, Enum):
    INPUT = "input"
    FEED = "input"
    APPEND = "input"
    DECISION = "decision"
    POLL = "decision"
    SPAN_READY = "span_ready"
    NORMALIZE_START = "normalize_start"
    NORMALIZE_FINISHED = "normalize_finished"
    NORMALIZE_DONE = "normalize_finished"
    COMMIT = "commit"
    FALLBACK = "fallback"
    TIMEOUT = "timeout"
    FINALIZE = "finalize"
    CLOSE = "finalize"
    RESET = "reset"


@dataclass(frozen=True)
class SpanEvent:
    type: SpanEventType
    text: str = ""
    final: bool = False
    now: float | None = None
    metadata: TextInputMetadata = TextInputMetadata()
    decision: CommitDecision | None = None
    payload: Any = None

    @classmethod
    def from_decision(
        cls,
        decision: CommitDecision,
        *,
        text: str = "",
        final: bool = False,
        now: float | None = None,
        metadata: TextInputMetadata | None = None,
    ) -> "SpanEvent":
        return cls(
            SpanEventType.DECISION,
            text=text,
            final=final,
            now=now,
            metadata=metadata or TextInputMetadata(),
            decision=decision,
        )


__all__ = ("SpanEvent", "SpanEventType")
