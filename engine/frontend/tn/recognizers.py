"""Recognizer adapters used by the span FSM.

The recognizers deliberately do not copy TN regular expressions.  The
existing lexical detector remains the single rule owner; this module exposes
that detector through the narrow ``SpanRecognizer`` contract so the FSM can
be extended without growing another collection of ``if`` branches.
"""

from __future__ import annotations

from typing import Callable

from ..text_commitment.types import LanguageKind, SpanKind
from .candidates import WfstCandidateProvider
from .recognizer import RecognitionKind, RecognizerContext, SpanRecognition


class ClassifierRecognizer:
    """Adapt one existing classifier into a lossless span recognizer."""

    def __init__(self, classifier: Callable[[str], SpanKind], *, name: str = "classifier"):
        self._classifier = classifier
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def recognize(
        self,
        text: str,
        *,
        context: RecognizerContext,
    ) -> SpanRecognition | None:
        if not text:
            return None
        kind = self._classifier(text)
        if kind is SpanKind.PLAIN:
            return None
        start = int(context.raw_offset)
        return SpanRecognition(
            kind=kind,
            raw_start=start,
            raw_end=start + len(text),
            result=RecognitionKind.MATCH,
            closed=False,
            extendable=True,
            recognizer=self.name,
        )


class CandidateRecognizer(ClassifierRecognizer):
    """Classifier that attaches n-best WFST candidates to its match.

    Candidate generation is optional and never decides closure.  The FSM and
    committer still own WAIT/COMMIT/FALLBACK; the candidate list is merely
    typed evidence available to that policy.
    """

    def __init__(
        self,
        classifier: Callable[[str], SpanKind],
        provider: WfstCandidateProvider,
        *,
        name: str = "candidate",
        nbest: int = 8,
    ):
        super().__init__(classifier, name=name)
        self._provider = provider
        self._nbest = max(1, int(nbest))

    def recognize(self, text: str, *, context: RecognizerContext) -> SpanRecognition | None:
        match = super().recognize(text, context=context)
        if match is None or context.language not in (LanguageKind.ZH, LanguageKind.EN):
            return match
        candidates = self._provider.candidates(
            text,
            language=context.language,
            domain=match.kind,
            nbest=self._nbest,
        )
        return SpanRecognition(
            kind=match.kind,
            raw_start=match.raw_start,
            raw_end=match.raw_end,
            result=match.result,
            closed=match.closed,
            extendable=match.extendable,
            recognizer=self.name,
            payload=tuple(candidates),
        )


__all__ = ("ClassifierRecognizer", "CandidateRecognizer")
