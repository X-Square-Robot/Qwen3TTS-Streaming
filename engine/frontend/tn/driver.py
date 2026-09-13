"""Outer driver connecting an existing committer to the TN span FSM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..text_commitment.types import (
    CommitDecision,
    TextInputMetadata,
)
from .fsm import SpanEvent, SpanEventType, SpanFSM, SpanStep
from .recognizer import RecognizerContext, RecognizerRegistry, SpanRecognition
from .recognizers import CandidateRecognizer, ClassifierRecognizer


class CommitterLike(Protocol):
    """Minimal input contract required by :class:`SpanDriver`."""

    def feed(
        self,
        text: str,
        *,
        final: bool = False,
        now: float | None = None,
        metadata: TextInputMetadata | None = None,
    ) -> CommitDecision: ...

    def poll(self, *, now: float | None = None) -> CommitDecision: ...


@dataclass(frozen=True)
class SpanDriverResult:
    """Decision plus the corresponding lifecycle transition."""

    decision: CommitDecision
    step: SpanStep


class SpanDriver:
    """Drive an ``IncrementalTextCommitter`` and record each decision in FSM.

    Lexical/TN behavior stays in the supplied committer.  This wrapper only
    translates feed/poll calls into typed FSM events, making the control-plane
    state inspectable and independently testable.
    """

    def __init__(
        self,
        committer: CommitterLike,
        *,
        fsm: SpanFSM | None = None,
        recognizers: RecognizerRegistry | None = None,
    ) -> None:
        self.committer = committer
        self.fsm = fsm or SpanFSM()
        if recognizers is None:
            detector = getattr(committer, "_detector", None)
            classifier = getattr(detector, "classify", None)
            candidate_resolver = getattr(committer, "_candidate_resolver", None)
            provider = getattr(candidate_resolver, "backend", None)
            if callable(classifier) and provider is not None:
                recognizer = CandidateRecognizer(classifier, provider)
            elif callable(classifier):
                recognizer = ClassifierRecognizer(classifier)
            else:
                recognizer = None
            recognizers = RecognizerRegistry((recognizer,) if recognizer else ())
        self.recognizers = recognizers
        self.last_result: SpanDriverResult | None = None

    def feed(
        self,
        text: str,
        *,
        final: bool = False,
        now: float | None = None,
        metadata: TextInputMetadata | None = None,
    ) -> CommitDecision:
        if text:
            self.fsm.step(
                SpanEvent(
                    SpanEventType.INPUT,
                    text=text,
                    final=final,
                    now=now,
                    metadata=metadata or TextInputMetadata(),
                )
            )
        decision = self.committer.feed(
            text,
            final=final,
            now=now,
            metadata=metadata,
        )
        self.last_result = SpanDriverResult(
            decision,
            self.fsm.step(
                SpanEvent.from_decision(
                    decision,
                    # INPUT owns the raw cursor increment.  The decision is
                    # a state projection and must not count the same packet
                    # twice.
                    text="",
                    final=final,
                    now=now,
                    metadata=metadata,
                )
            ),
        )
        return decision

    def poll(self, *, now: float | None = None) -> CommitDecision:
        decision = self.committer.poll(now=now)
        self.last_result = SpanDriverResult(
            decision,
            self.fsm.step(SpanEvent.from_decision(decision, now=now)),
        )
        return decision

    def recognize(
        self,
        text: str,
        *,
        context: RecognizerContext | None = None,
    ) -> SpanRecognition | None:
        return self.recognizers.recognize(text, context=context)


__all__ = ("CommitterLike", "SpanDriverResult", "SpanDriver")
