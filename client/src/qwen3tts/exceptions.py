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


class ProtocolVersionMismatchError(TTSClientError):
    """Raised when the server speaks a different protocol generation.

    Engine and client SDK are version-paired: install the wheel the engine
    serves at ``GET /sdk/`` on its health port, or the same git tag as the
    deployed engine (``/health`` reports it in the ``version`` field). Set
    ``QWEN3TTS_SKIP_PROTOCOL_CHECK=1`` to downgrade this error to a warning.
    """


class DependencyMissingError(TransportNotSupportedError):
    """Raised when an optional dependency extra is required but unavailable."""


class StreamClosedError(TTSClientError):
    """Raised when a stream session is already closed."""
