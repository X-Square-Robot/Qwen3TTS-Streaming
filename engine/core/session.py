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

from .types import EngineResult, SessionConfig, SessionState

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

    # Segment tracking
    segments_submitted: int = 0
    segments_done: int = 0
    segment_order: dict[int, SegmentOrderMeta] = field(default_factory=dict)
    segment_texts: dict[int, str] = field(default_factory=dict)
    text_boundary_emitted: set[int] = field(default_factory=set)
    segment_token_emitted_count: dict[int, int] = field(default_factory=dict)
    engine_tokens_done_sent: bool = False

    # Optional transport-layer callback hook (e.g. gRPC / Triton adapters)
    event_callback: Any = None

    # Metrics
    total_steps: int = 0
    total_audio_bytes: int = 0

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

    def record_first_audio(self) -> None:
        """Record the first raw audio arrival (phase ``engine.audio.first_raw``)."""
        if self.first_raw_audio_at is None:
            self.first_raw_audio_at = time.monotonic()

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
