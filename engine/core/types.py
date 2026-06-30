"""Shared types for asyncio frontend ↔ engine thread communication.

Design principle: these objects cross the thread boundary via thread-safe queues.
They carry only plain data / small CPU tensors — never raw GPU tensors.

Level 2 pipelining: each session may have multiple in-flight segments.
Requests and results are tagged with (session_id, segment_idx) to route
correctly.  The scheduler uses RequestPriority to order prefill work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from .observability import ObsLevel


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class SessionState(Enum):
    PENDING = auto()
    PREFILL = auto()
    DECODING = auto()
    SEGMENT = auto()
    DONE = auto()


class InputMode(Enum):
    AUTO = "auto"  # engine decides per-packet: Stage 1 engages only for long packets
    TOKEN = "token"
    CLAUSE = "clause"
    LONG_SEGMENT = "long_segment"
    FULL_TEXT = "full_text"


class GroupPolicy(Enum):
    NONE = "none"
    AUTO = "auto"


class AudioEncoding(Enum):
    PCM_F32 = "pcm_f32"
    PCM_S16LE = "pcm_s16le"


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    encoding: AudioEncoding = AudioEncoding.PCM_F32
    channels: int = 1


@dataclass
class VADConfig:
    enabled: bool = False
    strategy: str = "disabled"  # "disabled" | "energy" | "tenvad"
    implementation: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    # Direct VAD parameters (populated from protocol layer)
    chunk_ms: int = 16
    begin_threshold: float = 0.6
    begin_count: int = 5
    end_threshold: float = 0.35
    end_count: int = 31
    start_margin_ms: int = 20


@dataclass
class OutputPolicyConfig:
    vad: VADConfig = field(default_factory=VADConfig)
    chunk_ms: int = 0
    packet_format: str = "raw_pcm"
    emit_text_events: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingConfig:
    request_id: str = ""
    turn_id: str = ""
    client_request_ts_ms: int = 0
    client_text_ts_ms: int = 0
    client_end_ts_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizedText:
    """Canonical text payload: normalized text plus its token IDs."""

    text: str = ""
    token_ids: list[int] = field(default_factory=list)


@dataclass
class SegmentToken:
    """One canonical text token used by the frontend spliter."""

    token_id: int
    text: str
    punct_level: int = 0


@dataclass
class SessionConfig:
    task_type: str = ""
    language: str = "auto"
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    instruct_spec: Optional[TokenizedText] = None
    ref_audio: Optional[bytes] = None
    ref_text: Optional[str] = None
    ref_text_spec: Optional[TokenizedText] = None
    spk_embedding: Optional[Any] = None
    ref_codec_sum_vec: Optional[Any] = None
    ref_audio_codes: Optional[Any] = None
    ref_c2w_kv: Optional[Any] = None
    ref_c2w_conv_states: Optional[list[Any]] = None
    ref_c2w_transconv_states: Optional[list[Any]] = None
    ref_c2w_frame_idx: int = 0
    ref_source: str = ""
    ref_id: Optional[str] = None
    ref_audio_sha256: str = ""
    ref_text_hash: str = ""
    ref_feature_cache_key: str = ""
    ref_warnings: list[str] = field(default_factory=list)
    ref_preprocess_runtime: str = ""
    x_vector_only: bool = False
    input_mode: InputMode = InputMode.AUTO
    group_policy: GroupPolicy = GroupPolicy.AUTO
    audio: AudioConfig = field(default_factory=AudioConfig)
    output_policy: OutputPolicyConfig = field(default_factory=OutputPolicyConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    # Resolved per-session observability level (raise-only override of the global
    # floor, clamped to max_session_level). None ⇒ use the global level.
    observability_level: Optional[ObsLevel] = None


# ---------------------------------------------------------------------------
# Frontend → Engine thread  (via engine_inbox)
# ---------------------------------------------------------------------------


class RequestType(Enum):
    NEW_SESSION = auto()
    START_TOKENS = auto()  # begin a new token segment within an existing session
    APPEND_TOKENS = auto()
    SEGMENT_TOKENS_DONE = auto()  # per-segment: no more tokens for this segment
    SESSION_TOKENS_DONE = auto()  # session-level: upstream has finished all tokens
    CANCEL_SESSION = auto()


class RequestPriority(Enum):
    """Lower numeric value = higher urgency."""

    FIRST_SEGMENT = 0  # new session, first segment — TTFB critical
    CONTINUATION = 1  # next segment while previous is flushing
    PREFETCHED = 2  # offline pre-split, ahead-of-time preparation


@dataclass
class EngineRequest:
    """A single message from the asyncio world to the engine thread."""

    type: RequestType
    session_id: str
    segment_idx: int = 0
    priority: RequestPriority = RequestPriority.FIRST_SEGMENT
    session_config: Optional[SessionConfig] = None
    # NEW_SESSION payload
    speaker_key: Optional[str] = None
    task_type: Optional[str] = None  # "custom_voice" | "voice_design" | "voice_clone"
    ref_audio: Optional[bytes] = None
    # APPEND_TOKENS / START_TOKENS payload
    token_ids: Optional[list[int]] = None
    append_eos: bool = True
    # back-reference so engine thread can push results to the right queue
    result_queue: Optional[asyncio.Queue] = None

    # -- Lifecycle timestamps (monotonic clock) --
    enqueued_at: Optional[float] = None  # set by Dispatcher before put()
    dequeued_at: Optional[float] = None  # set by EngineLoop after get()


# ---------------------------------------------------------------------------
# Engine thread → Frontend  (via per-session result_queue)
# ---------------------------------------------------------------------------


class ResultType(Enum):
    PREFILL_DONE = auto()
    AUDIO_CHUNK = auto()
    WARNING = auto()
    SEGMENT_END = auto()
    SESSION_DONE = auto()
    RATIO_UPDATE = auto()  # EMA audio:text ratio feedback
    ERROR = auto()


@dataclass
class EngineResult:
    type: ResultType
    session_id: str
    segment_idx: int = 0
    audio_bytes: Optional[bytes] = None  # for AUDIO_CHUNK
    warning_msg: Optional[str] = None  # for WARNING
    error_msg: Optional[str] = None  # for ERROR
    metrics: dict = field(default_factory=dict)  # step_count, rtf, etc.
    ema_ratio: float = 0.0  # for RATIO_UPDATE
