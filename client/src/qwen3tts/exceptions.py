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
    """Raised when the server speaks an incompatible protocol family or major.

    Revisions within one protocol major are compatible. Install the wheel the
    engine serves at ``GET /sdk/`` on its health port, or the matching wheel
    from the GitHub/GitLab Release or Package Registry. Set
    ``QWEN3TTS_SKIP_PROTOCOL_CHECK=1`` to downgrade this error to a warning.
    """


class EngineVersionMismatchError(TTSClientError):
    """Legacy exception retained for API compatibility.

    Built-in clients no longer raise this exception: ``engine_version`` is
    diagnostic metadata, so release skew emits a ``RuntimeWarning``. Only an
    incompatible wire-protocol family or major rejects a connection via
    :class:`ProtocolVersionMismatchError`.
    """


class DependencyMissingError(TransportNotSupportedError):
    """Raised when an optional dependency extra is required but unavailable."""


class StreamClosedError(TTSClientError):
    """Raised when a stream session is already closed."""


class StreamRecoveryError(StreamClosedError):
    """Raised when an interrupted websocket stream cannot be resumed safely.

    This is intentionally a :class:`StreamClosedError` subclass so existing
    callers keep taking their established closed-stream path, while callers
    that care about resumability can distinguish an exhausted/rejected resume
    from an ordinary local close.
    """


class ConnectionPoolError(TTSClientError):
    """Base error for bounded websocket connection-pool admission."""


class PoolSaturatedError(ConnectionPoolError):
    """Raised when the websocket pool's bounded wait queue is full."""


class PoolAcquireTimeoutError(ConnectionPoolError, TimeoutError):
    """Raised when no websocket lease becomes available before its deadline."""
