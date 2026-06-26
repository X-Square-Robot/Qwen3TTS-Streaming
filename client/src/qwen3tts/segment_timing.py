"""Per-segment timing report parser.

Parses segment-level timing metadata from segment_end event meta.
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
class SegmentTimingReport:
    """Per-segment timing report parsed from segment_end event meta."""

    segment_id: int
    text_preview: str
    prefill_ms: Optional[float] = None
    decode_steps: Optional[int] = None
    audio_ms: Optional[float] = None
    cache_hit: Optional[bool] = None
    text_tokens: Optional[int] = None

    @classmethod
    def from_segment_end_meta(cls, meta: dict[str, str]) -> "SegmentTimingReport":
        """Parse segment timing from a segment_end event's meta dict."""
        cache_hit_val = meta.get("segment_cache_hit")
        return cls(
            segment_id=_safe_int(meta.get("segment_id")) or 0,
            text_preview=meta.get("segment_text_preview", ""),
            prefill_ms=_safe_float(meta.get("segment_prefill_ms")),
            decode_steps=_safe_int(meta.get("segment_decode_steps")),
            audio_ms=_safe_float(meta.get("segment_audio_ms")),
            cache_hit=(cache_hit_val == "true") if cache_hit_val is not None else None,
            text_tokens=_safe_int(meta.get("segment_text_tokens")),
        )

    def summary(self) -> dict[str, Optional[float | int | bool | str]]:
        """Return all segment metrics as a dict."""
        return {
            "segment_id": self.segment_id,
            "text_preview": self.text_preview,
            "prefill_ms": self.prefill_ms,
            "decode_steps": self.decode_steps,
            "audio_ms": self.audio_ms,
            "cache_hit": self.cache_hit,
            "text_tokens": self.text_tokens,
        }
