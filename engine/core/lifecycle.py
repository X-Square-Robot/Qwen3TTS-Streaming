"""Structured lifecycle event logger for the Qwen3-TTS engine.

Emits machine-parseable JSON log lines for each canonical lifecycle phase.
A downstream consumer can grep for a session_id and reconstruct the full
request timeline from these structured events alone.

Usage::

    from engine.core.lifecycle import LifecycleLogger

    LifecycleLogger.emit(
        session_id="abc123",
        phase="session.created",
        speaker_id="spk_001",
    )
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .observability import ObsLevel, is_enabled

try:
    # Rust JSON serializer, ~5-10x faster than stdlib for these small dicts.
    # Lifecycle events serialize on hot threads (engine loop / gateway loop),
    # so serialization cost is burst-TTFT relevant.
    import orjson

    def _dumps(event: dict) -> str:
        return orjson.dumps(event, default=str).decode()
except ImportError:  # pragma: no cover - orjson ships in the runtime image

    def _dumps(event: dict) -> str:
        return json.dumps(event, ensure_ascii=False, default=str)


logger = logging.getLogger("engine.lifecycle")


class LifecycleLogger:
    """Emits structured lifecycle events to the standard logger.

    Each event is a JSON object logged at INFO level with a ``"lifecycle": true``
    marker, enabling downstream consumers to filter lifecycle events from
    ordinary log noise.
    """

    @staticmethod
    def emit(
        session_id: str,
        phase: str,
        *,
        segment_idx: int | None = None,
        request_id: str | None = None,
        turn_id: str | None = None,
        monotonic_ts: float | None = None,
        epoch_ts_ms: int | None = None,
        min_level: ObsLevel = ObsLevel.DAILY,
        session_level: ObsLevel | None = None,
        **kwargs: Any,
    ) -> None:
        """Emit a structured lifecycle log line.

        Args:
            session_id: Session unique identifier.
            phase: Canonical lifecycle phase name (e.g. ``session.created``).
            segment_idx: Segment index, if phase is segment-scoped.
            request_id: Request unique identifier, if available.
            turn_id: Turn identifier, if available.
            monotonic_ts: Monotonic timestamp. Defaults to ``time.monotonic()``.
            epoch_ts_ms: Epoch timestamp in ms. Defaults to ``time.time() * 1000``.
            min_level: Observability tier at which this event fires (default
                ``DAILY`` so existing call sites always emit at the daily floor).
            session_level: This session's resolved override level, if any;
                gating uses it instead of the global level when present.
            **kwargs: Additional phase-specific fields.
        """
        if not is_enabled(min_level, session_level):
            return
        if monotonic_ts is None:
            monotonic_ts = time.monotonic()
        if epoch_ts_ms is None:
            epoch_ts_ms = int(round(time.time() * 1000.0))

        event: dict[str, Any] = {
            "lifecycle": True,
            "session_id": session_id,
            "phase": phase,
            "monotonic_ts": monotonic_ts,
            "epoch_ts_ms": epoch_ts_ms,
        }
        if segment_idx is not None:
            event["segment_id"] = segment_idx
        if request_id:
            event["request_id"] = request_id
        if turn_id:
            event["turn_id"] = turn_id
        if kwargs:
            event.update(kwargs)

        logger.info(_dumps(event))
