"""Async-side session: lives entirely in the asyncio event loop.

Holds the per-request Spliter (text segmentation orchestrator), an
asyncio.Queue for receiving results from the engine thread, an
AudioReorder for multi-segment pipelining, and stream-output state.

No GPU tensors, no torch imports.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .speech_state import SpeechStateCapability
from .types import EngineResult, SessionConfig, SessionState
from .text_journal import CanonicalTextJournal
from .native_cursor import CursorLabelPlan, slice_cursor_label_plan
from .text_coordinates import TextProgressProjection

logger = logging.getLogger(__name__)

_SESSION_RESULT_QUEUE_MAXSIZE = int(
    os.environ.get("ENGINE_SESSION_RESULT_QUEUE_MAXSIZE", "4096") or "4096"
)


@dataclass
class SegmentOrderMeta:
    group_idx: int
    local_idx: int
    group_final: bool = True


@dataclass
class Session:
    session_id: str
    config: SessionConfig = field(default_factory=SessionConfig)

    state: SessionState = SessionState.PENDING

    # -- Lifecycle timestamps (monotonic clock) --
    # Each field corresponds to a canonical lifecycle phase from timing_semantics.py.

    created_at: float = field(default_factory=time.monotonic)
    """Monotonic timestamp for phase ``session.created``."""

    first_text_enqueued_at: Optional[float] = None
    """Monotonic timestamp for phase ``text.first_enqueued``."""

    first_text_dequeued_at: Optional[float] = None
    """Monotonic timestamp for phase ``text.first_dequeued``."""

    prefill_started_at: Optional[float] = None
    """Monotonic timestamp for phase ``engine.prefill.started``."""

    prefill_completed_at: Optional[float] = None
    """Monotonic timestamp for phase ``engine.prefill.completed``."""

    first_raw_audio_at: Optional[float] = None
    """Monotonic timestamp for phase ``engine.audio.first_raw``."""

    first_effective_audio_at: Optional[float] = None
    """Monotonic timestamp for phase ``output.audio.first_effective``."""

    # Spliter (text segmentation orchestrator) — set by Dispatcher
    spliter: Any = None

    # AudioReorder for multi-segment pipelining — set by Dispatcher
    reorder: Any = None

    # Engine thread pushes EngineResult here; asyncio consumer reads them
    result_queue: asyncio.Queue[EngineResult] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_SESSION_RESULT_QUEUE_MAXSIZE)
    )

    # Accumulated text that hasn't been tokenized yet (streaming buffer)
    _text_buffer: str = ""
    _input_complete: bool = False
    # Stage-0 cross-packet carry: a trailing suffix held back because it may be
    # the start of an emoji sequence split across packets (see split_pending_emoji).
    _emoji_carry: str = ""
    text_journal: Optional[CanonicalTextJournal] = None
    text_committer: Any = None
    # Table-driven TN span controller wrapping the committer.  Kept separate
    # from the raw committer so lifecycle state and lexical normalization do
    # not share one mutable object.
    tn_driver: Any = None
    audio_credit_estimator: Any = None
    # Optional CPU-owned native-cursor plan.  It is populated only when the
    # frontend is given a model-compatible labelizer; the default path remains
    # EMA-only and carries no cursor labels across the thread boundary.
    cursor_commits: list[Any] = field(default_factory=list)
    cursor_spoken_texts: list[str] = field(default_factory=list)
    cursor_label_plan: Optional[CursorLabelPlan] = None
    cursor_plan_revision: int = 0
    # Native cursor is an optional progress route.  Once plan construction or
    # labelization fails, the session stays alive and the shared progress
    # publisher uses EMA for the remainder of the session.
    native_cursor_disabled: bool = False
    native_cursor_fallback_reason: str = ""
    # Frontend-side watchdog state.  A native cursor that keeps returning the
    # same neural position must not hold text progress forever; after the
    # bounded grace period the session uses the shared EMA route.  The
    # continuous position is separate from the discrete token frontier because
    # a token may span many audio frames before its public boundary advances.
    native_cursor_stall_frames: dict[int, int] = field(default_factory=dict)
    native_cursor_last_frame_end: dict[int, int] = field(default_factory=dict)
    native_cursor_last_mu: dict[int, float] = field(default_factory=dict)
    native_cursor_stalled_segments: set[int] = field(default_factory=set)
    cursor_segment_plans: dict[int, CursorLabelPlan] = field(default_factory=dict)
    # Payloads already handed to the engine, keyed by segment.  Revisions that
    # only advance the CPU-side plan counter do not require another GPU buffer
    # copy when the label/provenance payload is unchanged.
    cursor_segment_published_payloads: dict[int, tuple] = field(default_factory=dict)
    cursor_segment_bounds: dict[int, tuple[int, int]] = field(default_factory=dict)
    def cursor_plan_for_segment(self, segment_idx: int) -> CursorLabelPlan | None:
        """Build an owner-safe session-global plan for one segment."""
        plan = self.cursor_label_plan
        bounds = self.cursor_segment_bounds.get(int(segment_idx))
        if plan is None:
            return None
        if bounds is None:
            return CursorLabelPlan(revision=plan.revision, final=plan.final)
        sliced = slice_cursor_label_plan(
            plan,
            normalized_start=bounds[0],
            normalized_end=bounds[1],
        )
        if sliced is None:
            # Only a legacy plan without precise offsets, or an unsplittable
            # individual label, needs this fallback. Never guess label offsets.
            return CursorLabelPlan(revision=plan.revision, final=plan.final)
        return sliced

    # Segment tracking
    segments_submitted: int = 0
    segments_done: int = 0
    segment_order: dict[int, SegmentOrderMeta] = field(default_factory=dict)
    segment_texts: dict[int, str] = field(default_factory=dict)
    text_boundary_emitted: set[int] = field(default_factory=set)
    segment_token_emitted_count: dict[int, int] = field(default_factory=dict)
    segment_token_spans: dict[int, list[dict[str, int]]] = field(default_factory=dict)
    # Retry reruns the same segment actions. Keep token identity separate from
    # the public coordinate spans so a failed attempt can be discarded without
    # duplicating the segment's canonical text provenance.
    segment_token_keys: dict[int, set[tuple[Any, ...]]] = field(default_factory=dict)
    next_progress_anchor_seq: int = 1
    # Frontend-owned coarse text progress state.  The estimator is deliberately
    # transport-neutral so OpenAI Realtime, WebSocket, gRPC and demo clients
    # observe the same source-frame contract.
    text_progress_estimators: dict[int, Any] = field(default_factory=dict)
    segment_progress_frames: dict[int, int] = field(default_factory=dict)
    # Per-segment CPU projection state.  The projector never owns neural
    # state; it only remembers the last published owner-level high-water.
    native_cursor_projectors: dict[int, Any] = field(default_factory=dict)
    text_coordinate_projectors: dict[int, TextProgressProjection] = field(default_factory=dict)
    engine_tokens_done_sent: bool = False

    # Optional transport-layer callback hook (e.g. gRPC / Triton adapters)
    event_callback: Any = None

    # Metrics
    total_steps: int = 0
    total_audio_bytes: int = 0

    # Static, fail-closed model capability copied at session creation.  Keep it
    # at the end to preserve the positional constructor ABI of legacy callers.
    # The asyncio-side Session never owns a checkpoint/opaque handle; those
    # remain on the engine thread and backend adapter.
    speech_state_capability: SpeechStateCapability = field(
        default_factory=SpeechStateCapability.disabled
    )
    # Optional X2 commitment bridge. It observes only main-TN commits; raw
    # text and spoken-form ownership remain with the frontend committer/journal.
    # Keep this after the existing capability field to preserve positional ABI.
    commitment_adapter: Any = None

    @property
    def extension_commitment(self) -> Any:
        """Compatibility view of the optional X2 policy, never serialized."""

        return getattr(self.commitment_adapter, "_policy", None)

    def append_text(self, text: str) -> None:
        self._text_buffer += text

    def drain_text(self) -> str:
        t = self._text_buffer
        self._text_buffer = ""
        return t

    @property
    def has_pending_text(self) -> bool:
        return len(self._text_buffer) > 0

    def mark_input_complete(self) -> None:
        self._input_complete = True

    @property
    def input_complete(self) -> bool:
        return self._input_complete

    def record_first_audio(self, timestamp: Optional[float] = None) -> None:
        """Record the first raw audio arrival.

        ``timestamp`` may be supplied by the engine result so the session
        keeps the engine's timestamp.  Subsequent segment audio is ignored;
        this is a session-level first-event metric.
        """
        if self.first_raw_audio_at is None:
            self.first_raw_audio_at = (
                time.monotonic() if timestamp is None else timestamp
            )

    @property
    def first_audio_at(self) -> Optional[float]:
        """Deprecated: use ``first_raw_audio_at`` instead."""
        import warnings

        warnings.warn(
            "Session.first_audio_at is deprecated; use first_raw_audio_at",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.first_raw_audio_at

    @first_audio_at.setter
    def first_audio_at(self, value: Optional[float]) -> None:
        """Deprecated setter: forwards to ``first_raw_audio_at``."""
        self.first_raw_audio_at = value

    @property
    def speaker_key(self) -> Optional[str]:
        return self.config.speaker

    @property
    def task_type(self) -> str:
        return self.config.task_type

    @property
    def session_create_to_first_raw_audio_ms(self) -> Optional[float]:
        """Derived metric: session.created → engine.audio.first_raw (ms)."""
        if self.first_raw_audio_at is None:
            return None
        return (self.first_raw_audio_at - self.created_at) * 1000

    @property
    def first_audio_latency_ms(self) -> Optional[float]:
        """Deprecated: use ``session_create_to_first_raw_audio_ms`` instead."""
        import warnings

        warnings.warn(
            "Session.first_audio_latency_ms is deprecated; "
            "use session_create_to_first_raw_audio_ms",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.session_create_to_first_raw_audio_ms
