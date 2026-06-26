"""Client-side server timing report parser.

Parses server timing metadata from protocol responses (done event meta)
into a structured object with computed latency properties.

Usage::

    from qwen3tts.timing import ServerTimingReport

    report = ServerTimingReport.from_done_meta(done_event_meta)
    print(report.explain_latency())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _safe_int(val: str | None) -> int | None:
    """Parse an optional string to int."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val: str | None) -> float | None:
    """Parse an optional string to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


@dataclass
class ServerTimingReport:
    """Parsed server-side timing from protocol metadata.

    All epoch timestamps are in milliseconds. All derived durations
    are in milliseconds.
    """

    # -- Server epoch timestamps --
    server_request_received_epoch_ms: Optional[int] = None
    server_session_created_epoch_ms: Optional[int] = None
    server_first_text_received_epoch_ms: Optional[int] = None
    server_first_text_enqueued_epoch_ms: Optional[int] = None
    server_first_text_dequeued_epoch_ms: Optional[int] = None
    server_prefill_started_epoch_ms: Optional[int] = None
    server_prefill_completed_epoch_ms: Optional[int] = None
    server_first_raw_audio_epoch_ms: Optional[int] = None
    server_first_effective_audio_epoch_ms: Optional[int] = None
    server_done_epoch_ms: Optional[int] = None

    # -- Client timestamps --
    client_request_ts_ms: Optional[int] = None
    client_text_ts_ms: Optional[int] = None
    client_end_ts_ms: Optional[int] = None

    # -- Server-derived durations --
    server_session_create_to_first_raw_audio_ms: Optional[float] = None
    server_session_create_to_first_effective_audio_ms: Optional[float] = None
    server_first_text_enqueue_to_first_raw_audio_ms: Optional[float] = None
    server_first_text_enqueue_to_first_effective_audio_ms: Optional[float] = None
    server_first_text_dequeue_to_first_raw_audio_ms: Optional[float] = None
    server_first_text_dequeue_to_first_effective_audio_ms: Optional[float] = None
    server_engine_queue_wait_ms: Optional[float] = None
    server_engine_prefill_ms: Optional[float] = None
    server_first_raw_to_first_effective_audio_ms: Optional[float] = None
    server_total_latency_ms: Optional[float] = None

    # -- Policy / result fields --
    server_prefix_trim_applied: bool = False
    server_prefix_trimmed_ms: Optional[float] = None
    server_vad_policy: str = ""
    server_cache_hit: bool = False
    server_cache_tokens_reused: int = 0

    # -- Segment stats --
    server_total_segments: Optional[int] = None
    server_total_audio_ms: Optional[float] = None

    # -- Text observability --
    server_text_input_mode: str = ""
    server_raw_first_text_preview: str = ""
    server_normalized_first_text_preview: str = ""
    server_text_coalesced: bool = False
    server_text_progress_protected: bool = False

    @classmethod
    def from_done_meta(cls, meta: dict[str, str]) -> "ServerTimingReport":
        """Parse server timing from the done event's meta dict."""
        return cls(
            # Epoch timestamps
            server_request_received_epoch_ms=_safe_int(meta.get("server_request_received_epoch_ms")),
            server_session_created_epoch_ms=_safe_int(meta.get("server_session_created_epoch_ms")),
            server_first_text_received_epoch_ms=_safe_int(meta.get("server_first_text_received_epoch_ms")),
            server_first_text_enqueued_epoch_ms=_safe_int(meta.get("server_first_text_enqueued_epoch_ms")),
            server_first_text_dequeued_epoch_ms=_safe_int(meta.get("server_first_text_dequeued_epoch_ms")),
            server_prefill_started_epoch_ms=_safe_int(meta.get("server_prefill_started_epoch_ms")),
            server_prefill_completed_epoch_ms=_safe_int(meta.get("server_prefill_completed_epoch_ms")),
            server_first_raw_audio_epoch_ms=_safe_int(meta.get("server_first_raw_audio_epoch_ms")),
            server_first_effective_audio_epoch_ms=_safe_int(meta.get("server_first_effective_audio_epoch_ms")),
            server_done_epoch_ms=_safe_int(meta.get("server_done_epoch_ms")),
            # Client timestamps
            client_request_ts_ms=_safe_int(meta.get("client_request_ts_ms")),
            client_text_ts_ms=_safe_int(meta.get("client_text_ts_ms")),
            client_end_ts_ms=_safe_int(meta.get("client_end_ts_ms")),
            # Server-derived durations
            server_session_create_to_first_raw_audio_ms=_safe_float(meta.get("server_session_create_to_first_raw_audio_ms")),
            server_session_create_to_first_effective_audio_ms=_safe_float(meta.get("server_session_create_to_first_effective_audio_ms")),
            server_first_text_enqueue_to_first_raw_audio_ms=_safe_float(meta.get("server_first_text_enqueue_to_first_raw_audio_ms")),
            server_first_text_enqueue_to_first_effective_audio_ms=_safe_float(meta.get("server_first_text_enqueue_to_first_effective_audio_ms")),
            server_first_text_dequeue_to_first_raw_audio_ms=_safe_float(meta.get("server_first_text_dequeue_to_first_raw_audio_ms")),
            server_first_text_dequeue_to_first_effective_audio_ms=_safe_float(meta.get("server_first_text_dequeue_to_first_effective_audio_ms")),
            server_engine_queue_wait_ms=_safe_float(meta.get("server_engine_queue_wait_ms")),
            server_engine_prefill_ms=_safe_float(meta.get("server_engine_prefill_ms")),
            server_first_raw_to_first_effective_audio_ms=_safe_float(meta.get("server_first_raw_to_first_effective_audio_ms")),
            server_total_latency_ms=_safe_float(meta.get("server_total_latency_ms")),
            # Policy / result
            server_prefix_trim_applied=meta.get("server_prefix_trim_applied") == "true",
            server_prefix_trimmed_ms=_safe_float(meta.get("server_prefix_trimmed_ms")),
            server_vad_policy=meta.get("server_vad_policy", ""),
            server_cache_hit=meta.get("server_cache_hit") == "true",
            server_cache_tokens_reused=_safe_int(meta.get("server_cache_tokens_reused")) or 0,
            # Segment stats
            server_total_segments=_safe_int(meta.get("server_total_segments")),
            server_total_audio_ms=_safe_float(meta.get("server_total_audio_ms")),
            # Text observability
            server_text_input_mode=meta.get("server_text_input_mode", ""),
            server_raw_first_text_preview=meta.get("server_raw_first_text_preview", ""),
            server_normalized_first_text_preview=meta.get("server_normalized_first_text_preview", ""),
            server_text_coalesced=meta.get("server_text_coalesced") == "true",
            server_text_progress_protected=meta.get("server_text_progress_protected") == "true",
        )

    @property
    def client_request_to_server_first_audio_ms(self) -> Optional[float]:
        """Contextual: client request to server first effective audio."""
        if self.client_request_ts_ms and self.server_first_effective_audio_epoch_ms:
            return self.server_first_effective_audio_epoch_ms - self.client_request_ts_ms
        return None

    @property
    def client_request_to_server_first_raw_audio_ms(self) -> Optional[float]:
        """Contextual: client request to server first raw audio."""
        if self.client_request_ts_ms and self.server_first_raw_audio_epoch_ms:
            return self.server_first_raw_audio_epoch_ms - self.client_request_ts_ms
        return None

    def summary(self) -> dict[str, float | int | bool | str | None]:
        """Return all computed metrics as a dict."""
        return {
            "session_create_to_first_raw_audio_ms": self.server_session_create_to_first_raw_audio_ms,
            "session_create_to_first_effective_audio_ms": self.server_session_create_to_first_effective_audio_ms,
            "first_text_enqueue_to_first_raw_audio_ms": self.server_first_text_enqueue_to_first_raw_audio_ms,
            "first_text_enqueue_to_first_effective_audio_ms": self.server_first_text_enqueue_to_first_effective_audio_ms,
            "first_text_dequeue_to_first_raw_audio_ms": self.server_first_text_dequeue_to_first_raw_audio_ms,
            "first_text_dequeue_to_first_effective_audio_ms": self.server_first_text_dequeue_to_first_effective_audio_ms,
            "engine_queue_wait_ms": self.server_engine_queue_wait_ms,
            "engine_prefill_ms": self.server_engine_prefill_ms,
            "first_raw_to_first_effective_audio_ms": self.server_first_raw_to_first_effective_audio_ms,
            "total_latency_ms": self.server_total_latency_ms,
            "prefix_trim_applied": self.server_prefix_trim_applied,
            "prefix_trimmed_ms": self.server_prefix_trimmed_ms,
            "cache_hit": self.server_cache_hit,
            "cache_tokens_reused": self.server_cache_tokens_reused,
        }

    def explain_latency(self) -> str:
        """Human-readable breakdown of where time was spent.

        Identifies the dominant latency component and provides
        a structured breakdown.
        """
        lines = ["Request latency breakdown:"]
        components: list[tuple[str, float]] = []

        if self.server_engine_queue_wait_ms is not None:
            components.append(("Engine queue wait", self.server_engine_queue_wait_ms))
        if self.server_engine_prefill_ms is not None:
            components.append(("Prefill", self.server_engine_prefill_ms))
        if self.server_first_text_dequeue_to_first_raw_audio_ms is not None and self.server_engine_prefill_ms is not None:
            decode_ms = self.server_first_text_dequeue_to_first_raw_audio_ms - self.server_engine_prefill_ms
            if decode_ms > 0:
                components.append(("Decode (first raw audio)", decode_ms))
        if self.server_first_raw_to_first_effective_audio_ms is not None and self.server_first_raw_to_first_effective_audio_ms > 0.01:
            components.append(("Output gating (raw→effective)", self.server_first_raw_to_first_effective_audio_ms))
        if self.server_prefix_trimmed_ms is not None and self.server_prefix_trimmed_ms > 0:
            components.append(("Prefix trim", self.server_prefix_trimmed_ms))

        for name, ms in components:
            lines.append(f"  {name}: {ms:.1f}ms")

        if components:
            dominant = max(components, key=lambda c: c[1])
            lines.append(f"  → Dominant: {dominant[0]} ({dominant[1]:.1f}ms)")
        else:
            lines.append("  (No detailed latency breakdown available)")

        if self.server_total_latency_ms is not None:
            lines.append(f"  Total: {self.server_total_latency_ms:.1f}ms")
        if self.server_cache_hit:
            lines.append(f"  Cache hit: yes ({self.server_cache_tokens_reused} tokens reused)")

        return "\n".join(lines)
