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
    "SessionEndRequest",
    "SessionStartRequest",
    "StreamCancelRequest",
    "StreamClosedError",
    "StreamEvent",
    "StreamTextChunk",
    "SynthesisConfig",
    "TTSClient",
    "TTSClientError",
    "TimingContext",
    "TransportNotSupportedError",
    "TransportProbeError",
    "VADPolicy",
)
