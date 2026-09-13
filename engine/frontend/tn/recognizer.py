"""Recognizer contracts used by the TN span boundary.

Recognizers classify an input island; they do not normalize it or emit spoken
text.  The registry is intentionally ordered so a future lexer can compose
specialized recognizers without copying rules into a second committer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol, Sequence

from ..text_commitment.types import (
    ContentKind,
    LanguageKind,
    SpanKind,
)


class RecognitionKind(str, Enum):
    MATCH = "match"
    CONTINUE = "continue"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class RecognizerContext:
    """Read-only context supplied to a recognizer."""

    raw_text: str = ""
    raw_offset: int = 0
    language: LanguageKind = LanguageKind.UNKNOWN
    content_kind: ContentKind = ContentKind.PROSE
    previous_kind: SpanKind | None = None
    payload: Any = None


@dataclass(frozen=True)
class SpanRecognition:
    """Lossless lexical result; TN remains a later concern."""

    kind: SpanKind
    raw_start: int
    raw_end: int
    result: RecognitionKind = RecognitionKind.MATCH
    closed: bool = False
    extendable: bool = True
    recognizer: str = ""
    payload: Any = None


class SpanRecognizer(Protocol):
    """Narrow protocol implemented by one lexical recognizer."""

    @property
    def name(self) -> str: ...

    def recognize(
        self,
        text: str,
        *,
        context: RecognizerContext,
    ) -> SpanRecognition | None: ...


class RecognizerRegistry:
    """Ordered recognizer registry with deterministic first-match semantics."""

    def __init__(self, recognizers: Sequence[SpanRecognizer] = ()) -> None:
        self._recognizers: list[SpanRecognizer] = list(recognizers)

    @property
    def recognizers(self) -> tuple[SpanRecognizer, ...]:
        return tuple(self._recognizers)

    def register(self, recognizer: SpanRecognizer, *, prepend: bool = False) -> None:
        if prepend:
            self._recognizers.insert(0, recognizer)
        else:
            self._recognizers.append(recognizer)

    def recognize(
        self,
        text: str,
        *,
        context: RecognizerContext | None = None,
    ) -> SpanRecognition | None:
        context = context or RecognizerContext(raw_text=text)
        for recognizer in self._recognizers:
            match = recognizer.recognize(text, context=context)
            if match is not None:
                if match.result is RecognitionKind.NO_MATCH:
                    continue
                if not match.recognizer:
                    return replace(match, recognizer=recognizer.name)
                return match
        return None


__all__ = (
    "RecognitionKind",
    "RecognizerContext",
    "SpanRecognition",
    "SpanRecognizer",
    "RecognizerRegistry",
)
