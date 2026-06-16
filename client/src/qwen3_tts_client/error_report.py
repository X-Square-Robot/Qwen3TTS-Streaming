"""Error timing report parser.

Parses structured error metadata from error event meta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _safe_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val: str | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


@dataclass
class ErrorTimingReport:
    """Structured error report parsed from error event meta."""

    error_phase: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    segments_completed: Optional[int] = None
    audio_produced_ms: Optional[float] = None

    @classmethod
    def from_error_meta(cls, meta: dict[str, str]) -> "ErrorTimingReport":
        """Parse error timing from an error event's meta dict."""
        return cls(
            error_phase=meta.get("error_phase"),
            error_type=meta.get("error_type"),
            error_message=meta.get("error_message"),
            segments_completed=_safe_int(meta.get("segments_completed")),
            audio_produced_ms=_safe_float(meta.get("audio_produced_ms")),
        )

    def summary(self) -> dict[str, Optional[str | int | float]]:
        """Return all error metrics as a dict."""
        return {
            "error_phase": self.error_phase,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "segments_completed": self.segments_completed,
            "audio_produced_ms": self.audio_produced_ms,
        }
