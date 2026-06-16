"""Unified lightweight Python SDK for Qwen3-TTS deployments.

Public entry points:

- `TTSClient.connect(...)`
- `AsyncTTSClient.connect(...)`
- shared protocol types from `qwen3_tts_protocol`
"""

from qwen3_tts_protocol import (
    ArrayResult,
    AudioChunk,
    AudioFormat,
    BytesResult,
    Capabilities,
    DetectedTransport,
    OutputPolicy,
    SessionEndRequest,
    SessionStartRequest,
    StreamCancelRequest,
    StreamEvent,
    StreamTextChunk,
    SynthesisConfig,
    TimingContext,
    VADPolicy,
)

from .async_client import AsyncTTSClient
from .client import TTSClient
from .exceptions import (
    DependencyMissingError,
    ProtocolError,
    StreamClosedError,
    TTSClientError,
    TransportNotSupportedError,
    TransportProbeError,
)
from .analyzers import LatencyAnalyzer, TimelineReconstructor
from .error_report import ErrorTimingReport
from .realtime import RealtimeAudioStream, TimedAudio
from .segment_timing import SegmentTimingReport
from .timing import ServerTimingReport

__all__ = (
    "ArrayResult",
    "AsyncTTSClient",
    "AudioChunk",
    "AudioFormat",
    "BytesResult",
    "Capabilities",
    "DependencyMissingError",
    "DetectedTransport",
    "ErrorTimingReport",
    "LatencyAnalyzer",
    "OutputPolicy",
    "ProtocolError",
    "RealtimeAudioStream",
    "SegmentTimingReport",
    "ServerTimingReport",
    "SessionEndRequest",
    "SessionStartRequest",
    "StreamCancelRequest",
    "StreamClosedError",
    "StreamEvent",
    "StreamTextChunk",
    "SynthesisConfig",
    "TimedAudio",
    "TTSClient",
    "TTSClientError",
    "TimelineReconstructor",
    "TimingContext",
    "TransportNotSupportedError",
    "TransportProbeError",
    "VADPolicy",
)
