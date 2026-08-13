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
from .reliability import (
    AttachmentFailure,
    AttachmentFence,
    DeliveryAttachment,
    DeliveryLedger,
    LedgerError,
    ReliableDelivery,
)
from .types import (
    AudioOutput,
    AudioFormat,
    EventOutput,
    ProtocolKind,
    SessionCommandKind,
    SessionOutput,
    SessionOutputKind,
    StartedOutput,
    TerminalOutput,
    TerminalStatus,
)

__all__ = [
    "AppendText",
    "AttachmentFailure",
    "AttachmentFence",
    "AudioFormat",
    "AudioOutput",
    "CompleteInput",
    "DeliveryAttachment",
    "DeliveryLedger",
    "EventOutput",
    "ExecutionBackend",
    "ExecutionHandle",
    "InputAck",
    "LedgerError",
    "ProtocolKind",
    "SessionCommandKind",
    "SessionHandle",
    "SessionOutput",
    "SessionOutputKind",
    "SessionProtocolError",
    "SessionService",
    "ReliableDelivery",
    "StartedOutput",
    "TerminalOutput",
    "TerminalStatus",
]
