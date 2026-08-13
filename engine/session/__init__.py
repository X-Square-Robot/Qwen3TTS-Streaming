"""Transport-neutral logical TTS session contracts.

Wire adapters (native WebSocket, OpenAI Realtime, and future transports) use
this package to talk to an execution backend.  The package intentionally does
not import aiohttp, JSON wire shapes, or Triton clients.
"""

from .service import (
    AppendText,
    CompleteInput,
    ExecutionBackend,
    ExecutionHandle,
    InputAck,
    SessionHandle,
    SessionProtocolError,
    SessionService,
)
from .types import (
    AudioOutput,
    AudioFormat,
    EventOutput,
    SessionOutput,
    StartedOutput,
    TerminalOutput,
    TerminalStatus,
)

__all__ = [
    "AppendText",
    "AudioFormat",
    "AudioOutput",
    "CompleteInput",
    "EventOutput",
    "ExecutionBackend",
    "ExecutionHandle",
    "InputAck",
    "SessionHandle",
    "SessionOutput",
    "SessionProtocolError",
    "SessionService",
    "StartedOutput",
    "TerminalOutput",
    "TerminalStatus",
]
