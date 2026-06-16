"""Server-side timing accumulator for cross-thread observability.

Collects lifecycle timestamps as they become available across the
asyncio frontend and the engine thread.  The accumulator is created in
the asyncio thread (gateway), passed by reference through
``EngineRequest.session_config.timing.extra``, and progressively
populated by both threads.

Thread safety
-------------
Under CPython's GIL, individual float/int attribute writes are atomic.
The accumulator is written from the engine thread (dequeue, prefill,
raw audio timestamps) and read from the asyncio thread
(``OutputPipeline.done_meta``).  This is safe under CPython.  For
non-GIL implementations, add ``threading.Lock`` guards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ServerTimingAccumulator:
    """Collects server-side lifecycle timestamps as they become available.

    Created in the asyncio thread when a session starts, passed by
    reference to the engine thread, and read back in the asyncio thread
    when the output pipeline produces the ``done`` event.
    """

    # -- Reference timestamps for monotonic → epoch conversion --
    base_monotonic: float = field(default_factory=time.monotonic)
    base_epoch_ms: int = field(default_factory=lambda: int(round(time.time() * 1000.0)))

    # -- Monotonic timestamps (set by various threads) --
    session_created_monotonic: Optional[float] = None
    first_text_enqueued_monotonic: Optional[float] = None
    first_text_dequeued_monotonic: Optional[float] = None
    prefill_started_monotonic: Optional[float] = None
    prefill_completed_monotonic: Optional[float] = None
    first_raw_audio_monotonic: Optional[float] = None
    first_effective_audio_monotonic: Optional[float] = None

    # -- Epoch timestamps (set by asyncio thread for protocol output) --
    request_received_epoch_ms: Optional[int] = None
    session_created_epoch_ms: Optional[int] = None
    first_text_received_epoch_ms: Optional[int] = None

    # -- Policy / result fields --
    cache_hit: bool = False
    cache_tokens_reused: int = 0
    prefix_trim_applied: bool = False
    prefix_trimmed_ms: float = 0.0
    vad_policy: str = "disabled"
    output_gating_mode: str = ""

    # -- Text observability --
    raw_first_text_preview: str = ""
    normalized_first_text_preview: str = ""
    text_input_mode: str = ""
    text_coalesced: bool = False
    text_progress_protected: bool = False

    # -- Segment stats --
    total_segments: int = 0
    total_audio_ms: float = 0.0

    def monotonic_to_epoch_ms(self, monotonic: float) -> int:
        """Convert a monotonic timestamp to epoch milliseconds."""
        return int(self.base_epoch_ms + (monotonic - self.base_monotonic) * 1000)

    def _derived_ms(self, start: Optional[float], end: Optional[float]) -> Optional[float]:
        """Compute a derived duration in milliseconds."""
        if start is None or end is None:
            return None
        return (end - start) * 1000.0

    def to_meta_dict(self) -> dict[str, str]:
        """Produce the complete server timing meta dict for protocol output.

        Returns a dict of string keys and string values, suitable for
        inclusion in ``AudioChunk.meta`` or ``StreamEvent.meta``.
        """
        meta: dict[str, str] = {}

        # -- Epoch timestamps --
        epoch_fields: list[tuple[str, Optional[float]]] = [
            ("server_session_created_epoch_ms", self.session_created_monotonic),
            ("server_first_text_enqueued_epoch_ms", self.first_text_enqueued_monotonic),
            ("server_first_text_dequeued_epoch_ms", self.first_text_dequeued_monotonic),
            ("server_prefill_started_epoch_ms", self.prefill_started_monotonic),
            ("server_prefill_completed_epoch_ms", self.prefill_completed_monotonic),
            ("server_first_raw_audio_epoch_ms", self.first_raw_audio_monotonic),
            ("server_first_effective_audio_epoch_ms", self.first_effective_audio_monotonic),
        ]
        for key, mono in epoch_fields:
            if mono is not None:
                meta[key] = str(self.monotonic_to_epoch_ms(mono))

        # Explicitly set epoch timestamps (from asyncio thread)
        if self.request_received_epoch_ms is not None:
            meta["server_request_received_epoch_ms"] = str(self.request_received_epoch_ms)
        if self.session_created_epoch_ms is not None:
            meta["server_session_created_epoch_ms"] = str(self.session_created_epoch_ms)
        if self.first_text_received_epoch_ms is not None:
            meta["server_first_text_received_epoch_ms"] = str(self.first_text_received_epoch_ms)

        # -- Derived durations --
        derived_fields: list[tuple[str, Optional[float], Optional[float]]] = [
            ("server_session_create_to_first_raw_audio_ms",
             self.session_created_monotonic, self.first_raw_audio_monotonic),
            ("server_session_create_to_first_effective_audio_ms",
             self.session_created_monotonic, self.first_effective_audio_monotonic),
            ("server_first_text_enqueue_to_first_raw_audio_ms",
             self.first_text_enqueued_monotonic, self.first_raw_audio_monotonic),
            ("server_first_text_enqueue_to_first_effective_audio_ms",
             self.first_text_enqueued_monotonic, self.first_effective_audio_monotonic),
            ("server_first_text_dequeue_to_first_raw_audio_ms",
             self.first_text_dequeued_monotonic, self.first_raw_audio_monotonic),
            ("server_first_text_dequeue_to_first_effective_audio_ms",
             self.first_text_dequeued_monotonic, self.first_effective_audio_monotonic),
            ("server_engine_queue_wait_ms",
             self.first_text_enqueued_monotonic, self.first_text_dequeued_monotonic),
            ("server_engine_prefill_ms",
             self.prefill_started_monotonic, self.prefill_completed_monotonic),
            ("server_first_raw_to_first_effective_audio_ms",
             self.first_raw_audio_monotonic, self.first_effective_audio_monotonic),
        ]
        for key, start, end in derived_fields:
            val = self._derived_ms(start, end)
            if val is not None:
                meta[key] = f"{val:.3f}"

        # -- Policy / result fields --
        meta["server_prefix_trim_applied"] = "true" if self.prefix_trim_applied else "false"
        if self.prefix_trim_applied:
            meta["server_prefix_trimmed_ms"] = f"{self.prefix_trimmed_ms:.3f}"
        meta["server_vad_policy"] = self.vad_policy
        if self.output_gating_mode:
            meta["server_output_gating_mode"] = self.output_gating_mode
        meta["server_cache_hit"] = "true" if self.cache_hit else "false"
        if self.cache_tokens_reused > 0:
            meta["server_cache_tokens_reused"] = str(self.cache_tokens_reused)

        # -- Text observability --
        if self.text_input_mode:
            meta["server_text_input_mode"] = self.text_input_mode
        if self.raw_first_text_preview:
            meta["server_raw_first_text_preview"] = self.raw_first_text_preview
        if self.normalized_first_text_preview:
            meta["server_normalized_first_text_preview"] = self.normalized_first_text_preview
        if self.text_coalesced:
            meta["server_text_coalesced"] = "true"
        if self.text_progress_protected:
            meta["server_text_progress_protected"] = "true"

        # -- Segment stats --
        if self.total_segments > 0:
            meta["server_total_segments"] = str(self.total_segments)
        if self.total_audio_ms > 0:
            meta["server_total_audio_ms"] = f"{self.total_audio_ms:.1f}"

        return meta
