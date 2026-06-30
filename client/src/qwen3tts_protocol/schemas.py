"""Trace and benchmark schemas shared by demo_api and tools/validation.

These types describe a single TTS run's events, metrics, and result summary.
They are intentionally pure-data (no numpy / triton / engine dependency).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any


# ---------------------------------------------------------------------------
# Known backend identifiers
# ---------------------------------------------------------------------------

BACKENDS = (
    "triton_streaming",
    "triton_offline",
    "triton_trt_streaming",
    "engine_grpc",
    "engine_websocket",
)


# ---------------------------------------------------------------------------
# Trace event
# ---------------------------------------------------------------------------


@dataclass
class TraceEvent:
    """A single event in a TTS trace timeline."""

    run_id: str
    backend: str
    type: str
    t_ms: float
    server_t_ms: float | None = None
    stream_id: str | None = None
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


# ---------------------------------------------------------------------------
# Run metrics
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """Quantitative metrics for a single TTS run."""

    first_playable_ms: float | None = None
    total_ms: float | None = None
    server_ttft_ms: float | None = None
    triton_adapter_ttft_ms: float | None = None
    engine_internal_ttft_ms: float | None = None
    client_ttfb_ms: float | None = None
    first_audible_ms: float | None = None
    full_audio_ready_ms: float | None = None
    audio_duration_ms: float | None = None
    simulated_llm_complete_ms: float | None = None
    chunks: int = 0
    cache_hit: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Complete result of a single TTS run: metrics + events + raw audio."""

    run_id: str
    backend: str
    label: str
    mode: str
    source: str
    metrics: RunMetrics
    events: list[TraceEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audio_format: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    raw_audio: bytes | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "backend": self.backend,
            "label": self.label,
            "mode": self.mode,
            "source": self.source,
            "metrics": self.metrics.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "warnings": list(self.warnings),
            "audio_format": dict(self.audio_format),
            "audio": dict(self.audio),
        }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float | None:
    """Compute the *pct*-th percentile (0..1) of *values*."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def summarize_ttft(values: list[float]) -> dict[str, float | int | None]:
    """Summarize a list of TTFT measurements."""
    if not values:
        return {
            "count": 0,
            "avg_ttft_ms": None,
            "p50_ttft_ms": None,
            "p90_ttft_ms": None,
            "p99_ttft_ms": None,
            "max_ttft_ms": None,
        }
    return {
        "count": len(values),
        "avg_ttft_ms": mean(values),
        "p50_ttft_ms": percentile(values, 0.50),
        "p90_ttft_ms": percentile(values, 0.90),
        "p99_ttft_ms": percentile(values, 0.99),
        "max_ttft_ms": max(values),
    }


def normalize_backend_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw backend trace dict into a canonical shape."""
    backend = str(raw.get("backend") or "")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend in trace result: {backend!r}")
    events = []
    for event in raw.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        event = dict(event)
        event.setdefault("run_id", raw.get("run_id", ""))
        event.setdefault("backend", backend)
        events.append(event)
    normalized = dict(raw)
    normalized["events"] = events
    normalized.setdefault("warnings", [])
    normalized.setdefault(
        "audio_format", {"encoding": "pcm_f32", "sample_rate": 24000, "channels": 1}
    )
    normalized.setdefault("metrics", {})
    return normalized


__all__ = [
    "BACKENDS",
    "TraceEvent",
    "RunMetrics",
    "RunResult",
    "percentile",
    "summarize_ttft",
    "normalize_backend_result",
]
