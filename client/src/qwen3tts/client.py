from __future__ import annotations

from dataclasses import replace
import os
import threading
import warnings

from qwen3tts_protocol import (
    ArrayResult,
    BytesResult,
    SessionStartRequest,
    SynthesisConfig,
)

from ._adapters.engine_grpc import EngineGrpcAdapter
from ._adapters.engine_websocket import EngineWebSocketAdapter
from ._adapters.openai_realtime import OpenAIRealtimeAdapter
from ._adapters.triton_grpc import TritonGrpcAdapter
from ._adapters.triton_http import TritonHttpAdapter
from ._internal.auth import apply_bearer_key
from ._internal.tls import TLSConfig, TLSVerify
from .audio import decode_audio_bytes_to_array
from .constants import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_TRITON_GRPC_MODEL,
    DEFAULT_TRITON_HTTP_MODEL,
    TRANSPORT_ENGINE_GRPC,
    TRANSPORT_ENGINE_WEBSOCKET,
    TRANSPORT_OPENAI_REALTIME,
    TRANSPORT_TRITON_GRPC,
    TRANSPORT_TRITON_HTTP,
)
from .detect import detect_transport
from .exceptions import TransportNotSupportedError


class TTSClient:
    def __init__(self, *, endpoint: str, adapter, detected) -> None:
        self.endpoint = endpoint
        self._adapter = adapter
        self.resolved_transport = detected.transport
        self.probe_report = list(detected.probe_report or [])
        self.detected_transport = detected

    @classmethod
    def connect(
        cls,
        endpoint,
        *,
        transport: str = "auto",
        model_name: str | None = None,
        model_version: str = DEFAULT_MODEL_VERSION,
        timeout: float = 30.0,
        connect_timeout: float | None = None,
        reconnect_attempts: int = 1,
        active_stream_resume: bool = True,
        stream_resume_attempts: int = 2,
        stream_resume_timeout: float = 10.0,
        stream_resume_ack_interval: int = 8,
        max_connections: int = 32,
        max_idle_connections: int = 8,
        max_pending_acquires: int = 256,
        acquire_timeout: float | None = 30.0,
        idle_ttl: float | None = None,
        max_lifetime: float | None = None,
        keepalive_interval: float = 15.0,
        keepalive_jitter: float = 0.2,
        key: str | None = None,
        headers: dict[str, str] | None = None,
        metadata=None,
        verify: bool | None = None,
        verify_protocol: bool | None = None,
        tls_verify: TLSVerify = True,
    ):
        """Connect to a TTS endpoint.

        For ``engine-websocket``, physical connections are exclusive leases
        from a bounded FIFO pool. Defaults allow 32 total connections, retain
        eight idle connections, queue at most 256 acquires, and wait 30 seconds
        for capacity. Active streams request resumable delivery by default and
        make up to two bounded resume attempts after a transport failure. A
        gateway that does not negotiate the delivery sidecar automatically
        keeps the historical fail-fast behavior. ``idle_ttl`` and
        ``max_lifetime`` are disabled by default; either ``None`` or ``0``
        disables them explicitly. Keepalive cycles use 20% timing jitter by
        default to avoid synchronized gateway probes.
        """

        if verify_protocol is None:
            verify_protocol = True if verify is None else bool(verify)
        elif verify is not None and bool(verify) != bool(verify_protocol):
            raise ValueError("verify and verify_protocol must not disagree")
        tls = TLSConfig.from_value(tls_verify)
        headers, metadata = apply_bearer_key(headers, metadata, key)
        detected = detect_transport(
            endpoint,
            transport=transport,
            model_name=model_name,
            model_version=model_version,
            timeout=timeout,
            connect_timeout=connect_timeout,
            headers=headers,
            metadata=metadata,
            **tls.forwarding_kwargs(),
        )
        _warn_legacy_transport(detected.transport)
        adapter = _build_adapter(
            detected.transport,
            endpoint=detected.resolved_endpoint,
            model_name=detected.model_name or model_name,
            model_version=detected.model_version or model_version,
            timeout=timeout,
            connect_timeout=connect_timeout,
            reconnect_attempts=reconnect_attempts,
            active_stream_resume=active_stream_resume,
            stream_resume_attempts=stream_resume_attempts,
            stream_resume_timeout=stream_resume_timeout,
            stream_resume_ack_interval=stream_resume_ack_interval,
            max_connections=max_connections,
            max_idle_connections=max_idle_connections,
            max_pending_acquires=max_pending_acquires,
            acquire_timeout=acquire_timeout,
            idle_ttl=idle_ttl,
            max_lifetime=max_lifetime,
            keepalive_interval=keepalive_interval,
            keepalive_jitter=keepalive_jitter,
            headers=headers,
            metadata=metadata,
            tls_verify=tls,
        )
        client = cls(endpoint=endpoint, adapter=adapter, detected=detected)
        # Connect-time compatibility guard via the versioned capabilities
        # surface: protocol major compatibility + engine/SDK release diagnostics. Auto-detect
        # already exchanged capabilities (and validated), so only the explicit-
        # transport path needs an extra in-band fetch here to close that gap.
        # ``verify_protocol=False`` skips it for a lazy connect; the historical
        # ``verify`` keyword remains a compatibility alias. A mismatch raises
        # ProtocolVersionMismatchError; release skew only emits a warning.
        if verify_protocol and transport != "auto":
            client.get_capabilities()
        elif transport == "auto" and detected.transport == TRANSPORT_ENGINE_WEBSOCKET:
            # Auto-detection used a short-lived probe socket.  Warm the actual
            # adapter pool now so the first synthesis does not pay a second
            # PaaS/LB websocket handshake.
            adapter.connect()
        return client

    def get_capabilities(self, *, timeout: float | None = None):
        """Return backend capabilities with an optional per-call timeout.

        Omitting ``timeout`` deliberately keeps the historical no-keyword
        adapter call.  That preserves compatibility with third-party adapters
        and older test doubles whose ``get_capabilities`` method does not yet
        accept a timeout override.
        """

        if timeout is None:
            return self._adapter.get_capabilities()
        return self._adapter.get_capabilities(timeout=timeout)

    def prewarm(
        self,
        connections: int = 1,
        *,
        timeout: float | None = None,
    ) -> int:
        """Fill a supported transport's idle connection pool to ``connections``.

        Websocket adapters establish the missing sockets concurrently and
        return the actual idle pool size.  Other transports may not expose an
        explicit connection pool and therefore reject this operation.
        """

        prewarm = getattr(self._adapter, "prewarm", None)
        if not callable(prewarm):
            raise TransportNotSupportedError(
                f"transport {self.resolved_transport!r} does not support prewarm"
            )
        # Preserve compatibility with adapters that implemented prewarm before
        # the optional per-call timeout keyword was added.
        if timeout is None:
            return prewarm(connections)
        return prewarm(connections, timeout=timeout)

    def synthesize_bytes(
        self, text: str, *, request: SynthesisConfig | None = None
    ) -> BytesResult:
        config = request or SynthesisConfig()
        start = SessionStartRequest(
            session_id="",
            config=config,
            output_policy=config.output_policy,
            timing=config.timing_context,
        )
        return self._adapter.synthesize_bytes(text, request=start)

    def synthesize_array(
        self, text: str, *, request: SynthesisConfig | None = None
    ) -> ArrayResult:
        result = self.synthesize_bytes(text, request=request)
        audio_array = decode_audio_bytes_to_array(
            result.audio_bytes, encoding=result.audio_format.encoding
        )
        return ArrayResult(
            audio_format=result.audio_format,
            session_id=result.session_id,
            transport=result.transport,
            events=list(result.events),
            warnings=list(result.warnings),
            details=dict(result.details),
            audio_array=audio_array,
        )

    def open_stream(self, start_request: SessionStartRequest):
        # Convenience: default the per-session output policy / timing from the
        # config so callers can pass just
        # ``SessionStartRequest(session_id=..., config=SynthesisConfig(...))``.
        if start_request.output_policy is None or start_request.timing is None:
            start_request = replace(
                start_request,
                output_policy=start_request.output_policy
                or start_request.config.output_policy,
                timing=start_request.timing or start_request.config.timing_context,
            )
        return self._adapter.open_stream(start_request)

    def close(self) -> None:
        """Release any transport resources held by the adapter (e.g. a shared
        gRPC channel kept warm across sessions)."""
        close = getattr(self._adapter, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "TTSClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _build_adapter(
    transport: str,
    *,
    endpoint: str,
    model_name: str | None,
    model_version: str,
    timeout: float,
    connect_timeout: float | None,
    headers,
    metadata,
    reconnect_attempts: int = 1,
    active_stream_resume: bool = True,
    stream_resume_attempts: int = 2,
    stream_resume_timeout: float = 10.0,
    stream_resume_ack_interval: int = 8,
    max_connections: int = 32,
    max_idle_connections: int = 8,
    max_pending_acquires: int = 256,
    acquire_timeout: float | None = 30.0,
    idle_ttl: float | None = None,
    max_lifetime: float | None = None,
    keepalive_interval: float = 15.0,
    keepalive_jitter: float = 0.2,
    tls_verify: TLSVerify | TLSConfig = True,
):
    tls = TLSConfig.from_value(tls_verify)
    if transport == TRANSPORT_ENGINE_WEBSOCKET:
        return EngineWebSocketAdapter(
            endpoint,
            timeout=timeout,
            connect_timeout=connect_timeout,
            headers=headers,
            reconnect_attempts=reconnect_attempts,
            active_stream_resume=active_stream_resume,
            stream_resume_attempts=stream_resume_attempts,
            stream_resume_timeout=stream_resume_timeout,
            stream_resume_ack_interval=stream_resume_ack_interval,
            max_connections=max_connections,
            max_idle_connections=max_idle_connections,
            max_pending_acquires=max_pending_acquires,
            acquire_timeout=acquire_timeout,
            idle_ttl=idle_ttl,
            max_lifetime=max_lifetime,
            keepalive_interval=keepalive_interval,
            keepalive_jitter=keepalive_jitter,
            tls_verify=tls,
        )
    if transport == TRANSPORT_OPENAI_REALTIME:
        return OpenAIRealtimeAdapter(
            endpoint,
            timeout=timeout,
            connect_timeout=connect_timeout,
            headers=headers,
            model_name=model_name,
            reconnect_attempts=reconnect_attempts,
            active_stream_resume=active_stream_resume,
            stream_resume_attempts=stream_resume_attempts,
            stream_resume_timeout=stream_resume_timeout,
            stream_resume_ack_interval=stream_resume_ack_interval,
            max_connections=max_connections,
            max_idle_connections=max_idle_connections,
            max_pending_acquires=max_pending_acquires,
            acquire_timeout=acquire_timeout,
            tls_verify=tls,
        )
    if transport == TRANSPORT_ENGINE_GRPC:
        return EngineGrpcAdapter(
            endpoint, timeout=timeout, metadata=metadata, headers=headers
        )
    if transport == TRANSPORT_TRITON_GRPC:
        return TritonGrpcAdapter(
            endpoint,
            model_name=model_name or DEFAULT_TRITON_GRPC_MODEL,
            model_version=model_version,
            timeout=timeout,
            metadata=metadata,
            headers=headers,
        )
    if transport == TRANSPORT_TRITON_HTTP:
        return TritonHttpAdapter(
            endpoint,
            model_name=model_name or DEFAULT_TRITON_HTTP_MODEL,
            model_version=model_version,
            timeout=timeout,
            headers=headers,
            tls_verify=tls,
        )
    raise ValueError(f"unsupported transport: {transport!r}")


_LEGACY_TRANSPORTS = {
    TRANSPORT_ENGINE_GRPC,
    TRANSPORT_TRITON_GRPC,
    TRANSPORT_TRITON_HTTP,
}
_WARNED_LEGACY_TRANSPORTS: set[str] = set()
_LEGACY_WARNING_LOCK = threading.Lock()


def _warn_legacy_transport(transport: str) -> None:
    if transport not in _LEGACY_TRANSPORTS:
        return
    if os.environ.get("QWEN3TTS_SUPPRESS_LEGACY_TRANSPORT_WARNING", "") == "1":
        return
    with _LEGACY_WARNING_LOCK:
        if transport in _WARNED_LEGACY_TRANSPORTS:
            return
        _WARNED_LEGACY_TRANSPORTS.add(transport)
    warnings.warn(
        f"The {transport!r} transport is a compatibility path and will be "
        "removed in a future major release. Prefer transport='engine-websocket' "
        "or transport='auto' against a server that advertises the native "
        "WebSocket protocol.",
        FutureWarning,
        stacklevel=3,
    )
