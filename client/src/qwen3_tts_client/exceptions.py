from __future__ import annotations


class TTSClientError(RuntimeError):
    """Base SDK error."""


class TransportNotSupportedError(TTSClientError):
    """Raised when a transport or required extra is unavailable."""


class TransportProbeError(TTSClientError):
    """Raised when endpoint auto-detection fails."""

    def __init__(self, message: str, *, probe_report: list[dict] | None = None):
        super().__init__(message)
        self.probe_report = list(probe_report or [])


class ProtocolError(TTSClientError):
    """Raised when the remote side returns malformed protocol data."""


class DependencyMissingError(TransportNotSupportedError):
    """Raised when an optional dependency extra is required but unavailable."""


class StreamClosedError(TTSClientError):
    """Raised when a stream session is already closed."""
