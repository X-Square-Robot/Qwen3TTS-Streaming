"""Typed contracts shared by the long-form hallucination evaluation tools.

The contracts in this module intentionally contain no engine or ASR imports.  They
are used by synthesis arms, offline analysis, and review/report tooling alike.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ArmKind(str, Enum):
    """The three frozen implementations compared by the 0818 experiment."""

    CURRENT_HEAD = "current_head"
    TRITON_0818 = "triton_0818"
    PYTORCH_0818 = "pytorch_0818"

    # Readable compatibility aliases for callers that name the runtime role.
    CURRENT_STANDALONE = "current_head"
    STANDALONE = "current_head"
    CURRENT = "current_head"
    TRITON = "triton_0818"
    PYTORCH = "pytorch_0818"

    @classmethod
    def _missing_(cls, value: object) -> ArmKind | None:
        if not isinstance(value, str):
            return None
        alias = {
            "current": cls.CURRENT_HEAD,
            "standalone": cls.CURRENT_HEAD,
            "current_standalone": cls.CURRENT_HEAD,
            "current_head_standalone": cls.CURRENT_HEAD,
            "triton": cls.TRITON_0818,
            "0818_triton": cls.TRITON_0818,
            "pytorch": cls.PYTORCH_0818,
            "official": cls.PYTORCH_0818,
            "official_pytorch": cls.PYTORCH_0818,
        }.get(value.strip().lower())
        return alias


class ReviewLabel(str, Enum):
    """Blind-review labels; values match the review package protocol."""

    OK = "OK"
    SINGLE_UNIT_LOOP = "SINGLE_UNIT_LOOP"
    ABNORMAL_NOISE = "ABNORMAL_NOISE"
    UNSUPPORTED_SPEECH = "UNSUPPORTED_SPEECH"
    OMISSION = "OMISSION"
    MISPRONUNCIATION = "MISPRONUNCIATION"
    UNSCORABLE = "UNSCORABLE"

    @classmethod
    def _missing_(cls, value: object) -> ReviewLabel | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        return cls.__members__.get(normalized)


class RunStatus(str, Enum):
    """Lifecycle and terminal states for one arm/session execution."""

    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    TTS_FAILED = "tts_failed"
    ASR_FAILED = "asr_failed"
    REVIEW_PENDING = "review_pending"
    REVIEWED = "reviewed"
    INVALID = "invalid"
    ERROR = "error"

    @classmethod
    def _missing_(cls, value: object) -> RunStatus | None:
        if not isinstance(value, str):
            return None
        alias = {
            "success": cls.OK,
            "complete": cls.OK,
            "completed": cls.OK,
            "failed": cls.ERROR,
            "failure": cls.ERROR,
            "unscorable": cls.INVALID,
        }.get(value.strip().lower())
        return alias


@dataclass(frozen=True)
class ArmRunRecord:
    """Small shared arm result; detailed engine telemetry stays in metadata."""

    arm: ArmKind
    seed: int
    session_id: str
    status: RunStatus
    wav_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", ArmKind(self.arm))
        object.__setattr__(self, "status", RunStatus(self.status))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not self.session_id:
            raise ValueError("session_id must not be empty")


@dataclass(frozen=True)
class ReferenceSentence:
    """One occurrence-aware sentence in the original reference document."""

    ordinal: int
    text: str
    start: int
    end: int
    occurrence: int
    sentence_id: str

    def __post_init__(self) -> None:
        if self.ordinal < 1:
            raise ValueError("sentence ordinal is one-based and must be positive")
        if self.occurrence < 1:
            raise ValueError("sentence occurrence must be positive")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("sentence offsets must describe a non-empty span")
        if not self.text:
            raise ValueError("sentence text must not be empty")

    @property
    def index(self) -> int:
        """Zero-based index for slicing APIs."""

        return self.ordinal - 1


@dataclass(frozen=True)
class SentenceOutcome:
    """A paired sentence-level truth observation used by rate statistics."""

    arm: ArmKind
    seed: int
    sentence_ordinal: int
    severe_hallucination: bool | None
    status: RunStatus = RunStatus.REVIEWED
    sentence_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", ArmKind(self.arm))
        object.__setattr__(self, "status", RunStatus(self.status))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if self.sentence_ordinal < 1:
            raise ValueError("sentence_ordinal must be positive")
        if self.severe_hallucination not in (True, False, None):
            raise TypeError("severe_hallucination must be bool or None")

    @property
    def valid(self) -> bool:
        return isinstance(self.severe_hallucination, bool)


# A terminal mark plus immediately following closing quotation/bracket characters.
_SENTENCE_END_RE = re.compile(
    r"(?:(?:[。！？!?]+|…{2,}))+"
    r"[”’」』】》〕〗〙〛）)\]]*"
)


def _occurrence_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def parse_reference_sentences(text: str) -> tuple[ReferenceSentence, ...]:
    """Split a document without globally stripping or mutating its contents.

    Sentence identity includes the one-based occurrence count of normalized text,
    so repeated legitimate reference sentences remain distinct and alignable.
    Character offsets are half-open offsets into the supplied string.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        return ()

    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.end()
        if text[cursor:end].strip():
            spans.append((cursor, end))
        cursor = end
    if cursor < len(text):
        if text[cursor:].strip():
            spans.append((cursor, len(text)))
        elif spans:
            # Preserve trailing newlines/spaces without creating a phantom sentence.
            spans[-1] = (spans[-1][0], len(text))

    occurrences: dict[str, int] = {}
    sentences: list[ReferenceSentence] = []
    for ordinal, (start, end) in enumerate(spans, start=1):
        sentence_text = text[start:end]
        key = _occurrence_key(sentence_text)
        occurrences[key] = occurrences.get(key, 0) + 1
        occurrence = occurrences[key]
        sentences.append(
            ReferenceSentence(
                ordinal=ordinal,
                text=sentence_text,
                start=start,
                end=end,
                occurrence=occurrence,
                sentence_id=f"sentence-{ordinal:03d}-occurrence-{occurrence:02d}",
            )
        )
    return tuple(sentences)


def split_reference_sentences(text: str) -> tuple[str, ...]:
    """Return only sentence text while retaining parser ordering and boundaries."""

    return tuple(sentence.text for sentence in parse_reference_sentences(text))


__all__ = [
    "ArmKind",
    "ArmRunRecord",
    "ReferenceSentence",
    "ReviewLabel",
    "RunStatus",
    "SentenceOutcome",
    "parse_reference_sentences",
    "split_reference_sentences",
]
