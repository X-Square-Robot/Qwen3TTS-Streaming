"""Canonical timing metric definitions for the Qwen3-TTS engine.

This module is the single source of truth for all timing metric semantics.
Every metric that the engine exposes must be registered here with a precise
definition of its start event, end event, clock domain, classification,
and type.

Canonical lifecycle phase names
-------------------------------

    1. request.accepted            Gateway receives request
    2. session.config.validated    Configuration validated
    3. session.created             Session object created
    4. session.registered          Session registered to backend
    5. text.first_received         First text arrives at gateway/worker
    6. text.first_sent             First text sent toward engine
    7. text.first_enqueued         First text enqueued to engine inbox
    8. text.first_dequeued         First text dequeued by engine thread
    9. engine.prefill.started      Prefill begins
   10. engine.prefill.completed    Prefill completes
   11. engine.decode.first_step    First decode step starts
   12. engine.audio.first_raw      First raw audio produced by engine
   13. output.audio.first_effective  First effective audio published to output
   14. session.completed           Session complete

Error phases:

   E1. session.timeout            Session timed out
   E2. session.evicted            Session evicted
   E3. engine.prefill.failed      Prefill failed
   E4. session.cancelled          Session cancelled
   E5. session.error              Generic error
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimingMetricDef:
    """Definition of a timing metric with explicit semantics.

    Attributes:
        name: Canonical metric name (e.g. "session_create_to_first_raw_audio_ms").
        start_event: Lifecycle phase name for the start of the interval.
        end_event: Lifecycle phase name for the end of the interval.
        clock_domain: "server_monotonic", "server_epoch", or "cross_domain".
        classification: "strong" (single clock domain, suitable for SLOs)
            or "contextual" (cross-domain, diagnostic only).
        metric_type: "derived" (computed duration) or "raw_timestamp"
            (original event timestamp).
        deprecated_alias: Old name that this metric replaces, if any.
        description: Human-readable description.
    """

    name: str
    start_event: str
    end_event: str
    clock_domain: str
    classification: str  # "strong" | "contextual"
    metric_type: str     # "derived" | "raw_timestamp"
    deprecated_alias: Optional[str] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Raw timestamp metrics (strong, server_monotonic)
# ---------------------------------------------------------------------------

_RAW_TIMESTAMP_METRICS: list[TimingMetricDef] = [
    TimingMetricDef(
        name="server_session_created_monotonic",
        start_event="session.created",
        end_event="session.created",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="raw_timestamp",
        description="Monotonic timestamp when session was created.",
    ),
    TimingMetricDef(
        name="server_first_text_enqueued_monotonic",
        start_event="text.first_enqueued",
        end_event="text.first_enqueued",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="raw_timestamp",
        description="Monotonic timestamp when first text was enqueued to engine inbox.",
    ),
    TimingMetricDef(
        name="server_first_text_dequeued_monotonic",
        start_event="text.first_dequeued",
        end_event="text.first_dequeued",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="raw_timestamp",
        description="Monotonic timestamp when first text was dequeued by engine thread.",
    ),
    TimingMetricDef(
        name="server_prefill_started_monotonic",
        start_event="engine.prefill.started",
        end_event="engine.prefill.started",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="raw_timestamp",
        description="Monotonic timestamp when prefill started.",
    ),
    TimingMetricDef(
        name="server_prefill_completed_monotonic",
        start_event="engine.prefill.completed",
        end_event="engine.prefill.completed",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="raw_timestamp",
        description="Monotonic timestamp when prefill completed.",
    ),
    TimingMetricDef(
        name="server_first_raw_audio_monotonic",
        start_event="engine.audio.first_raw",
        end_event="engine.audio.first_raw",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="raw_timestamp",
        description="Monotonic timestamp when first raw audio was produced.",
    ),
    TimingMetricDef(
        name="server_first_effective_audio_monotonic",
        start_event="output.audio.first_effective",
        end_event="output.audio.first_effective",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="raw_timestamp",
        description="Monotonic timestamp when first effective audio was published.",
    ),
]

# ---------------------------------------------------------------------------
# Raw timestamp metrics (epoch milliseconds, for protocol)
# ---------------------------------------------------------------------------

_EPOCH_TIMESTAMP_METRICS: list[TimingMetricDef] = [
    TimingMetricDef(
        name="server_request_received_epoch_ms",
        start_event="request.accepted",
        end_event="request.accepted",
        clock_domain="server_epoch",
        classification="contextual",
        metric_type="raw_timestamp",
        description="Epoch timestamp when gateway received the request.",
    ),
    TimingMetricDef(
        name="server_session_created_epoch_ms",
        start_event="session.created",
        end_event="session.created",
        clock_domain="server_epoch",
        classification="strong",
        metric_type="raw_timestamp",
        description="Epoch timestamp when session was created.",
    ),
    TimingMetricDef(
        name="server_first_text_received_epoch_ms",
        start_event="text.first_received",
        end_event="text.first_received",
        clock_domain="server_epoch",
        classification="contextual",
        metric_type="raw_timestamp",
        description="Epoch timestamp when first text arrived at gateway.",
    ),
    TimingMetricDef(
        name="server_first_text_enqueued_epoch_ms",
        start_event="text.first_enqueued",
        end_event="text.first_enqueued",
        clock_domain="server_epoch",
        classification="strong",
        metric_type="raw_timestamp",
        description="Epoch timestamp when first text was enqueued.",
    ),
    TimingMetricDef(
        name="server_first_text_dequeued_epoch_ms",
        start_event="text.first_dequeued",
        end_event="text.first_dequeued",
        clock_domain="server_epoch",
        classification="strong",
        metric_type="raw_timestamp",
        description="Epoch timestamp when first text was dequeued.",
    ),
    TimingMetricDef(
        name="server_prefill_started_epoch_ms",
        start_event="engine.prefill.started",
        end_event="engine.prefill.started",
        clock_domain="server_epoch",
        classification="strong",
        metric_type="raw_timestamp",
        description="Epoch timestamp when prefill started.",
    ),
    TimingMetricDef(
        name="server_prefill_completed_epoch_ms",
        start_event="engine.prefill.completed",
        end_event="engine.prefill.completed",
        clock_domain="server_epoch",
        classification="strong",
        metric_type="raw_timestamp",
        description="Epoch timestamp when prefill completed.",
    ),
    TimingMetricDef(
        name="server_first_raw_audio_epoch_ms",
        start_event="engine.audio.first_raw",
        end_event="engine.audio.first_raw",
        clock_domain="server_epoch",
        classification="strong",
        metric_type="raw_timestamp",
        description="Epoch timestamp when first raw audio was produced.",
    ),
    TimingMetricDef(
        name="server_first_effective_audio_epoch_ms",
        start_event="output.audio.first_effective",
        end_event="output.audio.first_effective",
        clock_domain="server_epoch",
        classification="strong",
        metric_type="raw_timestamp",
        description="Epoch timestamp when first effective audio was published.",
    ),
    TimingMetricDef(
        name="server_done_epoch_ms",
        start_event="session.completed",
        end_event="session.completed",
        clock_domain="server_epoch",
        classification="strong",
        metric_type="raw_timestamp",
        description="Epoch timestamp when session completed.",
    ),
]

# ---------------------------------------------------------------------------
# Derived duration metrics (strong, server_monotonic)
# ---------------------------------------------------------------------------

_DERIVED_DURATION_METRICS: list[TimingMetricDef] = [
    TimingMetricDef(
        name="session_create_to_first_raw_audio_ms",
        start_event="session.created",
        end_event="engine.audio.first_raw",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="derived",
        deprecated_alias="first_audio_latency_ms",
        description="Time from session creation to first raw audio. "
                    "Replaces the ambiguous 'first_audio' metric.",
    ),
    TimingMetricDef(
        name="session_create_to_first_effective_audio_ms",
        start_event="session.created",
        end_event="output.audio.first_effective",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="derived",
        description="Time from session creation to first effective audio.",
    ),
    TimingMetricDef(
        name="first_text_enqueue_to_first_raw_audio_ms",
        start_event="text.first_enqueued",
        end_event="engine.audio.first_raw",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="derived",
        description="Time from first text enqueue to first raw audio. "
                    "Captures engine queue wait + prefill + decode.",
    ),
    TimingMetricDef(
        name="first_text_enqueue_to_first_effective_audio_ms",
        start_event="text.first_enqueued",
        end_event="output.audio.first_effective",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="derived",
        description="Time from first text enqueue to first effective audio.",
    ),
    TimingMetricDef(
        name="first_text_dequeue_to_first_raw_audio_ms",
        start_event="text.first_dequeued",
        end_event="engine.audio.first_raw",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="derived",
        description="Pure inference latency: text dequeued to first raw audio.",
    ),
    TimingMetricDef(
        name="first_text_dequeue_to_first_effective_audio_ms",
        start_event="text.first_dequeued",
        end_event="output.audio.first_effective",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="derived",
        description="Inference + gating latency: text dequeued to first effective audio.",
    ),
    TimingMetricDef(
        name="engine_queue_wait_ms",
        start_event="text.first_enqueued",
        end_event="text.first_dequeued",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="derived",
        description="Time first text spent waiting in the engine inbox queue.",
    ),
    TimingMetricDef(
        name="engine_prefill_ms",
        start_event="engine.prefill.started",
        end_event="engine.prefill.completed",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="derived",
        description="Prefill duration.",
    ),
    TimingMetricDef(
        name="first_raw_to_first_effective_audio_ms",
        start_event="engine.audio.first_raw",
        end_event="output.audio.first_effective",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="derived",
        description="Gating latency: raw audio to effective audio. "
                    "Non-zero when prefix trim / VAD gating is applied.",
    ),
    TimingMetricDef(
        name="total_latency_ms",
        start_event="request.accepted",
        end_event="session.completed",
        clock_domain="server_monotonic",
        classification="strong",
        metric_type="derived",
        deprecated_alias="server_total_latency_ms",
        description="Total request latency from acceptance to completion.",
    ),
]

# ---------------------------------------------------------------------------
# Cross-domain (contextual) metrics
# ---------------------------------------------------------------------------

_CONTEXTUAL_METRICS: list[TimingMetricDef] = [
    TimingMetricDef(
        name="client_request_to_server_first_audio_ms",
        start_event="request.accepted",
        end_event="output.audio.first_effective",
        clock_domain="cross_domain",
        classification="contextual",
        metric_type="derived",
        description="Client request timestamp to server first effective audio. "
                    "Cross-domain: involves both client and server clocks.",
    ),
    TimingMetricDef(
        name="client_request_to_server_first_raw_audio_ms",
        start_event="request.accepted",
        end_event="engine.audio.first_raw",
        clock_domain="cross_domain",
        classification="contextual",
        metric_type="derived",
        description="Client request timestamp to server first raw audio. "
                    "Cross-domain: involves both client and server clocks.",
    ),
]

# ---------------------------------------------------------------------------
# All metrics combined
# ---------------------------------------------------------------------------

ALL_METRICS: list[TimingMetricDef] = (
    _RAW_TIMESTAMP_METRICS
    + _EPOCH_TIMESTAMP_METRICS
    + _DERIVED_DURATION_METRICS
    + _CONTEXTUAL_METRICS
)

_METRICS_BY_NAME: dict[str, TimingMetricDef] = {m.name: m for m in ALL_METRICS}

_DEPRECATED_ALIASES: dict[str, str] = {
    m.deprecated_alias: m.name
    for m in ALL_METRICS
    if m.deprecated_alias is not None
}


def get_metric(name: str) -> Optional[TimingMetricDef]:
    """Look up a metric definition by name."""
    return _METRICS_BY_NAME.get(name)


def resolve_name(name: str) -> str:
    """Resolve a possibly-deprecated name to its canonical name.

    If *name* is a deprecated alias, returns the canonical name.
    Otherwise returns *name* unchanged.
    """
    return _DEPRECATED_ALIASES.get(name, name)


def is_deprecated(name: str) -> bool:
    """Return True if *name* is a deprecated alias."""
    return name in _DEPRECATED_ALIASES


def strong_metrics() -> list[TimingMetricDef]:
    """Return all strong (single clock domain) metrics."""
    return [m for m in ALL_METRICS if m.classification == "strong"]


def contextual_metrics() -> list[TimingMetricDef]:
    """Return all contextual (cross-domain) metrics."""
    return [m for m in ALL_METRICS if m.classification == "contextual"]


def derived_metrics() -> list[TimingMetricDef]:
    """Return all derived duration metrics."""
    return [m for m in ALL_METRICS if m.metric_type == "derived"]


def raw_timestamp_metrics() -> list[TimingMetricDef]:
    """Return all raw timestamp metrics."""
    return [m for m in ALL_METRICS if m.metric_type == "raw_timestamp"]
