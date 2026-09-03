from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SpanKind(str, Enum):
    PLAIN = "plain"
    NUMBER = "number"
    ENGLISH_WORD = "english_word"
    ORDINAL = "ordinal"
    MATH = "math"
    URL = "url"
    EMAIL = "email"
    MARKDOWN = "markdown"
    JSON = "json"
    EMOJI = "emoji"
    LITERAL = "literal"


class CommitKind(str, Enum):
    NORMALIZED = "normalized"
    FALLBACK = "fallback"
    LITERAL = "literal"


class FallbackPolicy(str, Enum):
    CARDINAL_OR_LITERAL = "cardinal_or_literal"
    LITERAL = "literal"


@dataclass(frozen=True)
class TextNormalizationConfig:
    enabled: bool = True
    language: str = "mixed_zh_en"
    semantic_max_wait_ms: float = 200.0
    semantic_idle_wait_ms: float = 80.0
    fallback: FallbackPolicy = FallbackPolicy.CARDINAL_OR_LITERAL
    projection: str = "readable_values"
    max_pending_chars: int = 512


@dataclass(frozen=True)
class TextCommit:
    raw_start: int
    raw_end: int
    tts_text: str
    span_kind: SpanKind = SpanKind.PLAIN
    commit_kind: CommitKind = CommitKind.NORMALIZED
    fence: int = 0
    mapping: tuple[tuple[int, int], ...] = ()
    raw_text: str = ""


@dataclass(frozen=True)
class CommitDecision:
    commits: tuple[TextCommit, ...] = ()
    pending_raw: str = ""
    pending_kind: Optional[SpanKind] = None
    reason: str = ""
    fallback: bool = False
    events: tuple[str, ...] = ()
