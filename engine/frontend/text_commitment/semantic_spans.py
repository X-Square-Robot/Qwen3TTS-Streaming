"""Typed semantic boundary for the incremental lexer."""
from __future__ import annotations

from typing import Callable

from .candidate_resolver import family_for_kind
from .types import SemanticFamily, SpanKind


class SpanDetector:
    def __init__(self, classifier: Callable[[str], SpanKind]):
        self._classifier = classifier

    def classify(self, raw: str) -> SpanKind:
        return self._classifier(raw)

    def family(self, kind: SpanKind) -> SemanticFamily:
        return family_for_kind(kind)
