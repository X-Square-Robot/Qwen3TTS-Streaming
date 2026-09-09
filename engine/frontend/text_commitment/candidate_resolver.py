"""Semantic-family candidate resolution for streaming TN.

The resolver deliberately does not decide stream closure.  It only turns a
closed semantic span into typed WeText candidates and exposes relative
ambiguity diagnostics.  WFST costs are comparable only within one request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalizer_backend import BackendCandidate, NormalizerBackend
from .types import LanguageKind, SemanticFamily, SemioticSpan, SpanKind


@dataclass(frozen=True)
class CandidateSet:
    candidates: tuple[BackendCandidate, ...] = ()
    best_cost: float | None = None
    second_cost: float | None = None
    cost_margin: float | None = None
    margin_per_input_unit: float | None = None
    source: str = ""
    backend_version: str = ""
    unique: bool = False

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


def family_for_kind(kind: SpanKind) -> SemanticFamily:
    if kind in (SpanKind.NUMBER, SpanKind.ORDINAL):
        return SemanticFamily.QUANTITY
    if kind in (SpanKind.IDENTIFIER, SpanKind.VERSION, SpanKind.ID_CARD):
        return SemanticFamily.IDENTIFIER
    if kind in (SpanKind.PHONE, SpanKind.EMAIL, SpanKind.URL):
        return SemanticFamily.CONTACT
    if kind is SpanKind.MATH:
        return SemanticFamily.FORMULA
    if kind in (SpanKind.MARKDOWN, SpanKind.JSON, SpanKind.LITERAL):
        return SemanticFamily.STRUCTURED
    return SemanticFamily.PROSE


class CandidateResolver:
    def __init__(self, backend: NormalizerBackend, *, nbest: int = 8) -> None:
        self.backend = backend
        self.nbest = max(1, min(int(nbest), 16))

    def resolve(
        self,
        span: SemioticSpan,
        *,
        language: LanguageKind,
        backend_text: str | None = None,
        domain: SpanKind | str | None = None,
    ) -> CandidateSet:
        if language not in (LanguageKind.ZH, LanguageKind.EN):
            return CandidateSet(source="unresolved_language")
        text = span.raw_text if backend_text is None else backend_text
        candidates = tuple(
            self.backend.candidates(
                text,
                language=language,
                domain=domain or span.kind,
                nbest=self.nbest,
            )
        )
        costs = [c.cost for c in candidates if c.cost is not None]
        best = costs[0] if costs else None
        second = costs[1] if len(costs) > 1 else None
        margin = second - best if best is not None and second is not None else None
        units = max(1, len(text))
        return CandidateSet(
            candidates=candidates,
            best_cost=best,
            second_cost=second,
            cost_margin=margin,
            margin_per_input_unit=margin / units if margin is not None else None,
            source="wetext" if candidates else "empty",
            backend_version=getattr(self.backend, "backend_name", ""),
            unique=len(candidates) == 1,
        )
