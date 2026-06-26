"""Timeline reconstruction and latency analysis tools.

Provides:
- ``TimelineReconstructor``: Reconstruct a request timeline from saved protocol events.
- ``LatencyAnalyzer``: Aggregate timing across multiple sessions with percentile
  computation and regression detection.
"""

from __future__ import annotations

import json
import statistics
from typing import Any, Optional

from .timing import ServerTimingReport
from .segment_timing import SegmentTimingReport
from .error_report import ErrorTimingReport


class TimelineReconstructor:
    """Reconstruct a request timeline from saved protocol events.

    Feed events and audio chunks as they are received. After the session
    completes, call ``build_timeline()`` to get a complete timeline with
    all computed latencies.

    Usage::

        recon = TimelineReconstructor()
        for event in session_events:
            recon.feed_event(event)
        timeline = recon.build_timeline()
    """

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._audio_chunks: list[dict[str, Any]] = []
        self._segments: list[SegmentTimingReport] = []
        self._error: Optional[ErrorTimingReport] = None
        self._done_meta: Optional[dict[str, str]] = None
        self._start_meta: Optional[dict[str, str]] = None

    def feed_event(self, event: dict[str, Any]) -> None:
        """Record a received event.

        Args:
            event: Event dict with 'type', 'meta', etc.
        """
        event_type = event.get("type", "")
        meta = event.get("meta", {})

        self._events.append(event)

        if event_type == "start":
            self._start_meta = meta
        elif event_type == "segment_end":
            seg_report = SegmentTimingReport.from_segment_end_meta(meta)
            self._segments.append(seg_report)
        elif event_type == "error":
            self._error = ErrorTimingReport.from_error_meta(meta)
        elif event_type == "done":
            self._done_meta = meta

    def feed_audio_chunk(self, chunk: dict[str, Any]) -> None:
        """Record a received audio chunk.

        Args:
            chunk: Audio chunk dict with 'meta', etc.
        """
        self._audio_chunks.append(chunk)

    def build_timeline(self) -> dict[str, Any]:
        """Build a complete timeline with all computed latencies.

        Returns:
            Dict with 'server_timing', 'segments', 'error', and
            'audio_chunk_count' keys.
        """
        result: dict[str, Any] = {
            "audio_chunk_count": len(self._audio_chunks),
            "segments": [s.summary() for s in self._segments],
        }

        if self._done_meta is not None:
            report = ServerTimingReport.from_done_meta(self._done_meta)
            result["server_timing"] = report.summary()
            result["explain_latency"] = report.explain_latency()

        if self._error is not None:
            result["error"] = self._error.summary()

        return result

    def to_json(self) -> str:
        """Serialize timeline for offline analysis."""
        return json.dumps(self.build_timeline(), indent=2, default=str)


class LatencyAnalyzer:
    """Analyze latency across multiple sessions.

    Collects ``ServerTimingReport`` instances from multiple requests
    and provides aggregated statistics.

    Usage::

        analyzer = LatencyAnalyzer()
        for session in sessions:
            report = ServerTimingReport.from_done_meta(session.done_meta)
            analyzer.add_session(report)

        p50 = analyzer.percentile("server_session_create_to_first_raw_audio_ms", 0.5)
        p99 = analyzer.percentile("server_session_create_to_first_raw_audio_ms", 0.99)
    """

    def __init__(self) -> None:
        self._sessions: list[ServerTimingReport] = []

    def add_session(self, report: ServerTimingReport) -> None:
        """Add a session's timing report."""
        self._sessions.append(report)

    @property
    def session_count(self) -> int:
        """Number of sessions analyzed."""
        return len(self._sessions)

    def percentile(self, metric: str, p: float) -> Optional[float]:
        """Compute percentile for a named metric across all sessions.

        Args:
            metric: Attribute name on ``ServerTimingReport``.
            p: Percentile (0.0-1.0).

        Returns:
            The percentile value, or None if no data.
        """
        values = []
        for report in self._sessions:
            val = getattr(report, metric, None)
            if val is not None:
                values.append(float(val))
        if not values:
            return None
        sorted_values = sorted(values)
        idx = p * (len(sorted_values) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_values) - 1)
        frac = idx - lower
        return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac

    def mean(self, metric: str) -> Optional[float]:
        """Compute mean for a named metric across all sessions."""
        values = []
        for report in self._sessions:
            val = getattr(report, metric, None)
            if val is not None:
                values.append(float(val))
        if not values:
            return None
        return statistics.mean(values)

    def stdev(self, metric: str) -> Optional[float]:
        """Compute standard deviation for a named metric."""
        values = []
        for report in self._sessions:
            val = getattr(report, metric, None)
            if val is not None:
                values.append(float(val))
        if len(values) < 2:
            return None
        return statistics.stdev(values)

    def regression_check(
        self,
        baseline: dict[str, float],
        threshold_pct: float = 20.0,
    ) -> list[str]:
        """Check for latency regressions vs a baseline.

        Args:
            baseline: Dict mapping metric names to baseline mean values.
            threshold_pct: Percentage increase that constitutes a regression.

        Returns:
            List of regression description strings.
        """
        regressions: list[str] = []
        for metric, baseline_val in baseline.items():
            current = self.mean(metric)
            if current is None:
                continue
            increase_pct = ((current - baseline_val) / baseline_val) * 100.0
            if increase_pct > threshold_pct:
                regressions.append(
                    f"{metric}: {current:.1f}ms vs baseline {baseline_val:.1f}ms "
                    f"(+{increase_pct:.1f}%)"
                )
        return regressions

    def summary(self) -> dict[str, Any]:
        """Return aggregated summary statistics for all sessions."""
        metrics = [
            "server_session_create_to_first_raw_audio_ms",
            "server_first_text_enqueue_to_first_raw_audio_ms",
            "server_first_text_dequeue_to_first_raw_audio_ms",
            "server_engine_prefill_ms",
            "server_first_raw_to_first_effective_audio_ms",
            "server_total_latency_ms",
        ]
        result: dict[str, Any] = {
            "session_count": self.session_count,
        }
        for metric in metrics:
            p50 = self.percentile(metric, 0.5)
            p99 = self.percentile(metric, 0.99)
            mean_val = self.mean(metric)
            if any(v is not None for v in [p50, p99, mean_val]):
                result[metric] = {
                    "p50": round(p50, 3) if p50 is not None else None,
                    "p99": round(p99, 3) if p99 is not None else None,
                    "mean": round(mean_val, 3) if mean_val is not None else None,
                }
        return result
