"""Optional latency / timing diagnostics for advanced users.

These tools reconstruct a request timeline from saved protocol events and
aggregate server-side latency metrics across requests. They are **not** needed
for normal synthesis, so they are kept out of the top-level ``qwen3tts``
namespace. Import them explicitly when you need them::

    from qwen3tts.diagnostics import LatencyAnalyzer, ServerTimingReport

    analyzer = LatencyAnalyzer()
    for session in sessions:
        analyzer.add_session(ServerTimingReport.from_done_meta(session.done_meta))
    print(analyzer.percentile("server_session_create_to_first_effective_audio_ms", 95))
"""

from .analyzers import LatencyAnalyzer, TimelineReconstructor
from .error_report import ErrorTimingReport
from .segment_timing import SegmentTimingReport
from .timing import ServerTimingReport

__all__ = (
    "ErrorTimingReport",
    "LatencyAnalyzer",
    "SegmentTimingReport",
    "ServerTimingReport",
    "TimelineReconstructor",
)
