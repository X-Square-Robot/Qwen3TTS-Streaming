"""Unified lightweight Python SDK for Qwen3-TTS deployments.

One package, one import root — everything a client needs lives here::

    from qwen3tts import TTSClient, SynthesisConfig

Public entry points:

- `TTSClient.connect(...)` — synchronous client
- `AsyncTTSClient.connect(...)` — asyncio client
- `RealtimeAudioStream` — wall-clock-aligned frames for playback / WebRTC
- protocol types (`SynthesisConfig`, `SessionStartRequest`, `AudioChunk`, …)
- exceptions (`TTSClientError`, …)

Optional latency/timing diagnostics live in `qwen3tts.diagnostics`.
"""

from qwen3tts_protocol import (
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
from .realtime import RealtimeAudioStream, TimedAudio

__all__ = (
    "ArrayResult",
    "AsyncTTSClient",
    "AudioChunk",
    "AudioFormat",
    "BytesResult",
    "Capabilities",
    "DependencyMissingError",
    "DetectedTransport",
    "OutputPolicy",
    "ProtocolError",
    "RealtimeAudioStream",
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
    "TimingContext",
    "TransportNotSupportedError",
    "TransportProbeError",
    "VADPolicy",
)
