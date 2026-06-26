"""Demo API schemas — thin re-export layer.

All types are now defined in ``qwen3tts_protocol.schemas``.
This module re-exports them for backward compatibility with code that
imports from ``demo_api.schemas``.
"""

from qwen3tts_protocol.schemas import (  # noqa: F401
    BACKENDS,
    TraceEvent,
    RunMetrics,
    RunResult,
    normalize_backend_result,
    percentile,
    summarize_ttft,
)

__all__ = [
    "BACKENDS",
    "TraceEvent",
    "RunMetrics",
    "RunResult",
    "normalize_backend_result",
    "percentile",
    "summarize_ttft",
]
