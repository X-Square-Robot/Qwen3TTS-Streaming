"""Observability level control plane for the Qwen3-TTS engine.

Defines the four-tier observability model's *level* concept and the process-wide
control plane that gates how verbose logging/dumping is.

Tiers (see ``docs/dev/design/observability_tiers.md``)::

    L1 DAILY  常开低量      —— production daily logs
    L2 DEBUG  开发机/定点    —— decision-rationale logs ("why")
    L3 DUMP   疑难杂症       —— raw tensor / posterior evidence

Levels are accumulative: ``DUMP`` implies ``DEBUG`` implies ``DAILY``.

Control precedence (highest first)::

    per-session override  >  ENGINE_OBSERVABILITY_LEVEL  >  engine.yaml  >  default(daily)

The per-session override is *raise-only*: a session may escalate its own level
above the global floor (for targeted debugging in production) but never lower it
below the operator-configured global level. A session escalation is additionally
clamped to ``max_session_level`` so untrusted clients cannot crank a session to
``DUMP`` and amplify dump IO into a DoS / leak text (see design doc §3.3).
"""

from __future__ import annotations

import hashlib
import logging
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


class ObsLevel(IntEnum):
    """Observability verbosity level. Higher = more detail. Accumulative."""

    DAILY = 1
    DEBUG = 2
    DUMP = 3


_ALIASES = {
    "daily": ObsLevel.DAILY,
    "l1": ObsLevel.DAILY,
    "info": ObsLevel.DAILY,
    "debug": ObsLevel.DEBUG,
    "l2": ObsLevel.DEBUG,
    "dump": ObsLevel.DUMP,
    "l3": ObsLevel.DUMP,
}


def parse_level(value: object, *, default: ObsLevel = ObsLevel.DAILY) -> ObsLevel:
    """Parse a level from a string/enum, falling back to *default* on unknown input."""
    if isinstance(value, ObsLevel):
        return value
    if value is None:
        return default
    key = str(value).strip().lower()
    if not key:
        return default
    level = _ALIASES.get(key)
    if level is None:
        logger.warning("Unknown observability level %r, using %s", value, default.name)
        return default
    return level


def to_logging_level(level: ObsLevel) -> int:
    """Map an observability level to a stdlib ``logging`` level.

    DAILY → INFO; DEBUG/DUMP → DEBUG (so decision logs are emitted).
    """
    return logging.INFO if level <= ObsLevel.DAILY else logging.DEBUG


# ---------------------------------------------------------------------------
# Process-wide control plane (set once at startup from EngineConfig)
# ---------------------------------------------------------------------------

_global_level: ObsLevel = ObsLevel.DAILY
_max_session_level: ObsLevel = ObsLevel.DAILY
_text_capture: str = "preview"  # disabled | preview | hashed | full
_text_preview_chars: int = 64
_health_interval_sec: float = 30.0


def configure(
    global_level: object,
    max_session_level: object,
    *,
    text_capture: str = "preview",
    text_preview_chars: int = 64,
    health_interval_sec: float = 30.0,
) -> None:
    """Install the process-wide observability levels. Called once at startup."""
    global _global_level, _max_session_level, _text_capture, _text_preview_chars
    global _health_interval_sec
    _global_level = parse_level(global_level)
    _max_session_level = parse_level(max_session_level)
    _text_capture = str(text_capture or "preview").strip().lower()
    _text_preview_chars = int(text_preview_chars)
    _health_interval_sec = float(health_interval_sec)
    logger.info(
        "Observability configured: global=%s max_session=%s text_capture=%s",
        _global_level.name,
        _max_session_level.name,
        _text_capture,
    )


def text_preview(text: str) -> str:
    """Apply the configured text-capture privacy policy to a synthesized string.

    disabled → "" · preview → first N chars · hashed → sha256[:12] · full → text.
    """
    if not text:
        return ""
    if _text_capture == "disabled":
        return ""
    if _text_capture == "full":
        return text
    if _text_capture == "hashed":
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    # default: preview
    n = _text_preview_chars
    return text[:n] + "…" if len(text) > n else text


def global_level() -> ObsLevel:
    return _global_level


def max_session_level() -> ObsLevel:
    return _max_session_level


def health_interval_sec() -> float:
    return _health_interval_sec


def resolve_session_level(requested: object) -> tuple[ObsLevel, bool]:
    """Resolve a per-session level request against the global floor and the
    ``max_session_level`` ceiling.

    Returns ``(effective_level, clamped)`` where ``effective_level`` is
    ``max(global, min(requested, max_session_level))`` — raise-only and clamped —
    and ``clamped`` is True when the requested level exceeded the ceiling.
    """
    req = parse_level(requested, default=_global_level)
    clamped = req > _max_session_level
    ceiled = ObsLevel(min(int(req), int(_max_session_level)))
    effective = ObsLevel(max(int(_global_level), int(ceiled)))
    return effective, clamped


def is_enabled(min_level: ObsLevel, session_level: Optional[ObsLevel] = None) -> bool:
    """Whether an observation at *min_level* should fire.

    Effective level is the session override if present (already resolved to be
    >= global floor), otherwise the global level.
    """
    effective = session_level if session_level is not None else _global_level
    return effective >= min_level
