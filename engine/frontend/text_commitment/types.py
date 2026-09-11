from __future__ import annotations

from dataclasses import dataclass
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
    IDENTIFIER = "identifier"
    VERSION = "version"
    PHONE = "phone"
    ID_CARD = "id_card"
    MARKDOWN = "markdown"
    JSON = "json"
    EMOJI = "emoji"
    LITERAL = "literal"


class SemanticFamily(str, Enum):
    QUANTITY = "quantity"
    IDENTIFIER = "identifier"
    CONTACT = "contact"
    FORMULA = "formula"
    STRUCTURED = "structured"
    PROSE = "prose"


class ContentKind(str, Enum):
    """The structural projection selected for an input island.

    This is deliberately separate from :class:`SpanKind`: ``ContentKind``
    describes the outer syntax (for example a Markdown document), while
    ``SpanKind`` describes the semiotic unit handed to the TN backend.
    """

    PROSE = "prose"
    MARKDOWN = "markdown"
    JSON = "json"
    CODE = "code"
    FORMULA = "formula"
    UNKNOWN = "unknown"


class LanguageKind(str, Enum):
    UNKNOWN = "unknown"
    ZH = "zh"
    EN = "en"


class LanguageEvidence(str, Enum):
    """Why a language hypothesis was produced.

    Evidence is observable metadata, not a commit decision.  In particular,
    script or LID evidence must not turn a numeric-only span into a committed
    language without the controller's safety checks.
    """

    EXPLICIT = "explicit"
    SESSION = "session"
    HAN_SCRIPT = "han_script"
    LATIN_SCRIPT = "latin_script"
    CONTEXT = "context"
    LID = "lid"
    FALLBACK = "fallback"


class CommitmentState(str, Enum):
    """Observable state of the causal commitment controller."""

    SCAN = "scan"
    OPEN = "open"
    READY = "ready"
    NORMALIZE = "normalize"
    COMMIT = "commit"
    FALLBACK = "fallback"
    DONE = "done"


class CommitKind(str, Enum):
    NORMALIZED = "normalized"
    FALLBACK = "fallback"
    LITERAL = "literal"


class FallbackPolicy(str, Enum):
    CARDINAL_OR_LITERAL = "cardinal_or_literal"
    LITERAL = "literal"


@dataclass(frozen=True)
class LanguageHypothesis:
    """A candidate language for one semiotic span.

    ``confidence`` is a ranking hint only.  A low-confidence or conflicting
    set remains ``UNKNOWN`` from the commit controller's point of view.
    """

    language: LanguageKind
    confidence: float
    evidence: tuple[LanguageEvidence, ...] = ()


@dataclass(frozen=True)
class TextInputMetadata:
    """Optional typed metadata attached to a raw transport delta.

    Existing transports construct the default value, so adding this type is
    backward compatible.  A future wire protocol can map ``language_hint``
    and ``read_as`` directly without changing the commitment contract.
    """

    language_hint: LanguageKind = LanguageKind.UNKNOWN
    content_kind: ContentKind = ContentKind.PROSE
    read_as: SpanKind | None = None
    closed: bool = False


@dataclass(frozen=True)
class SemioticSpan:
    """Lossless raw span passed from the incremental lexer to a backend."""

    span_id: int
    raw_start: int
    raw_end: int
    raw_text: str
    kind: SpanKind
    language_hypotheses: tuple[LanguageHypothesis, ...] = ()
    closed: bool = False
    content_kind: ContentKind = ContentKind.PROSE
    family: SemanticFamily = SemanticFamily.PROSE
    closure_reason: str = ""
    extendable: bool = True


@dataclass(frozen=True)
class NormalizationCandidate:
    """One backend verbalization candidate with source alignment."""

    text: str
    language: LanguageKind
    mapping: tuple[tuple[int, int], ...] = ()
    weight: float = 0.0


@dataclass(frozen=True)
class PrefixResult:
    """Result of advancing a backend prefix oracle.

    ``stable_units`` are whole spoken lexical units, never an arbitrary
    character suffix.  ``extendable`` means a future raw suffix can still
    change the current span and therefore the controller must retain it.
    """

    stable_units: tuple[str, ...] = ()
    candidates: tuple[NormalizationCandidate, ...] = ()
    extendable: bool = True
    closed: bool = False
    pending_raw: str = ""
    error_code: str | None = None


@dataclass(frozen=True)
class NormalizationResult:
    """Final verbalization of one closed span."""

    text: str
    language: LanguageKind
    mapping: tuple[tuple[int, int], ...] = ()
    backend: str = ""
    fallback: bool = False
    warning: str | None = None


@dataclass(frozen=True)
class CommitmentUpdate:
    """Typed result returned by the causal controller."""

    commits: tuple["TextCommit", ...] = ()
    pending_raw: str = ""
    pending_kind: SpanKind | None = None
    state: CommitmentState = CommitmentState.SCAN
    reason: str = ""
    fallback: bool = False
    events: tuple[str, ...] = ()
    committed_raw_end: int = 0


@dataclass(frozen=True)
class TextNormalizationConfig:
    enabled: bool = True
    language: str = "mixed_zh_en"
    # A semantic span often arrives over several transport packets.  The
    # previous 200 ms cap split ordinary URLs/phone numbers when clients sent
    # 4-character packets every ~45 ms, forcing a literal fallback before the
    # span's closing boundary arrived.  Keep the idle deadline as the
    # low-latency guard while allowing a normal burst to complete.
    semantic_max_wait_ms: float = 1000.0
    # Leave enough idle grace for the documented 90 ms slow preset; otherwise
    # a URL/phone span is committed literally just before its next packet.
    semantic_idle_wait_ms: float = 120.0
    fallback: FallbackPolicy = FallbackPolicy.CARDINAL_OR_LITERAL
    projection: str = "readable_values"
    max_pending_chars: int = 512
    # ``closed_span`` is the conservative production mode.  ``prefix_oracle``
    # may be enabled only after the pinned wetext prefix contract passes its
    # compatibility tests; it never permits snapshot-diff commits.
    commit_mode: str = "closed_span"
    candidate_nbest: int = 8
    calibration_profile: str = ""
    ambiguity_policy: str = "wait"
    margin_threshold: float | None = None
    family_margin_thresholds: tuple[tuple[str, float], ...] = ()


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
    language: LanguageKind = LanguageKind.ZH
    # Stable identifiers are appended fields so existing positional callers
    # remain source-compatible.  They are assigned by the session controller.
    commit_id: int = 0
    span_id: int = 0
    semantic_family: SemanticFamily = SemanticFamily.PROSE
    decision_source: str = ""
    candidate_count: int = 0
    best_cost: float | None = None
    cost_margin: float | None = None
    calibrated_confidence: float | None = None
    closure_reason: str = ""
    fallback_reason: str = ""
    # Optional detailed alignment: absolute raw start/end followed by output
    # (spoken-text) start/end.  ``mapping`` remains the legacy raw-only view.
    output_mapping: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True)
class CommitDecision:
    commits: tuple[TextCommit, ...] = ()
    pending_raw: str = ""
    pending_kind: Optional[SpanKind] = None
    reason: str = ""
    fallback: bool = False
    events: tuple[str, ...] = ()
    state: CommitmentState = CommitmentState.SCAN
    committed_raw_end: int = 0
