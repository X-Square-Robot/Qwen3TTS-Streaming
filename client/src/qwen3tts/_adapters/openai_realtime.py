from __future__ import annotations

import base64
import binascii
import json
import socket
import threading
import time
import uuid
import weakref
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from qwen3tts_protocol import (
    AudioChunk,
    AudioFormat,
    BytesResult,
    Capabilities,
    SessionStartRequest,
    StreamEvent,
    capabilities_from_mapping,
)

from .._internal.raw_websocket import (
    RawWebSocketConnection,
    RawWebSocketError,
    ws_close,
    ws_connect,
    ws_recv_frame,
    ws_send_json,
)
from .._internal.utils import (
    advertised_protocols,
    build_bytes_result,
    check_engine_version,
    maybe_base64,
)
from .._session import BaseStreamSession
from ..constants import (
    DEFAULT_OPENAI_REALTIME_MODEL,
    OPENAI_REALTIME_PROTOCOL,
    QWEN_TEXT_BUFFER_EXTENSION,
    QWEN_TEXT_PROGRESS_EXTENSION,
    TRANSPORT_OPENAI_REALTIME,
)
from ..exceptions import ProtocolError, StreamClosedError
from ..exceptions import PoolAcquireTimeoutError, PoolSaturatedError


class OpenAIRealtimeAdapter:
    """Synchronous SDK adapter for the OpenAI Realtime TTS endpoint."""

    transport_name = TRANSPORT_OPENAI_REALTIME

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float,
        connect_timeout: float | None = None,
        headers: dict[str, str] | None = None,
        model_name: str | None = None,
        reconnect_attempts: int = 1,
        active_stream_resume: bool = True,
        stream_resume_attempts: int = 2,
        stream_resume_timeout: float = 10.0,
        stream_resume_ack_interval: int = 8,
        max_connections: int = 32,
        max_idle_connections: int = 0,
        max_pending_acquires: int = 256,
        acquire_timeout: float | None = 30.0,
    ) -> None:
        self.model_name = str(model_name or DEFAULT_OPENAI_REALTIME_MODEL)
        self.endpoint = _with_model_query(endpoint, self.model_name)
        self.timeout = max(0.1, float(timeout))
        self.connect_timeout = (
            self.timeout
            if connect_timeout is None
            else max(0.1, float(connect_timeout))
        )
        self.headers = dict(headers or {})
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.active_stream_resume = bool(active_stream_resume)
        self.stream_resume_attempts = max(0, int(stream_resume_attempts))
        self.stream_resume_timeout = max(0.1, float(stream_resume_timeout))
        self.stream_resume_ack_interval = max(1, int(stream_resume_ack_interval))
        self.max_connections = int(max_connections)
        self.max_idle_connections = int(max_idle_connections)
        if self.max_idle_connections < 0 or self.max_idle_connections > self.max_connections:
            raise ValueError("max_idle_connections must be between 0 and max_connections")
        self.max_pending_acquires = int(max_pending_acquires)
        if self.max_connections <= 0:
            raise ValueError("max_connections must be greater than zero")
        if self.max_pending_acquires < 0:
            raise ValueError("max_pending_acquires must be non-negative")
        if acquire_timeout is not None and float(acquire_timeout) < 0:
            raise ValueError("acquire_timeout must be non-negative or None")
        self.acquire_timeout = (
            None if acquire_timeout is None else float(acquire_timeout)
        )
        self._slot_semaphore = threading.BoundedSemaphore(self.max_connections)
        self._slot_lock = threading.Lock()
        self._pending_acquires = 0
        self._sessions: weakref.WeakSet[OpenAIRealtimeStreamSession] = weakref.WeakSet()
        self._sessions_lock = threading.Lock()
        self._closed = False
        self._pool_lock = threading.Condition(threading.Lock())
        self._idle_connections: list[RawWebSocketConnection] = []
        self._physical_connections = 0

    def get_capabilities(self, *, timeout: float | None = None) -> Capabilities:
        request_timeout = self.timeout if timeout is None else max(0.1, float(timeout))
        response = requests.get(
            _capabilities_url(self.endpoint),
            timeout=request_timeout,
            headers=self.headers,
        )
        if response.status_code != 200:
            raise ProtocolError(
                f"Realtime capabilities request failed: HTTP {response.status_code}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise ProtocolError("Realtime capabilities response must be a JSON object")
        protocols = advertised_protocols(payload)
        if protocols and OPENAI_REALTIME_PROTOCOL not in protocols:
            raise ProtocolError(
                f"server does not advertise {OPENAI_REALTIME_PROTOCOL!r}"
            )
        check_engine_version(payload.get("engine_version"))
        return capabilities_from_mapping(payload)

    def synthesize_bytes(self, text: str, *, request) -> BytesResult:
        if not str(text or ""):
            raise ValueError("synthesis text must not be empty")
        session = self._open_session(request, initial_text=str(text))
        audio_parts: list[bytes] = []
        events: list[StreamEvent] = []
        warnings: list[str] = []
        audio_format = session.audio_format
        terminal_seen = False
        for message in session.iter_messages(post_send_idle_timeout=self.timeout):
            if isinstance(message, AudioChunk):
                audio_parts.append(message.pcm_bytes)
                audio_format = message.audio
                continue
            events.append(message)
            if message.type == "warning" and message.message:
                warnings.append(message.message)
            if message.type in {"done", "error"}:
                terminal_seen = True
        if not terminal_seen:
            raise ProtocolError("Realtime connection ended without response.done")
        return build_bytes_result(
            audio_bytes=b"".join(audio_parts),
            audio_format=audio_format,
            session_id=session.session_id,
            transport=self.transport_name,
            events=events,
            warnings=warnings,
            details={
                "usage": dict(session.usage),
                "response_id": session.response_id,
                "status": session.response_status,
            },
        )

    def open_stream(self, start_request: SessionStartRequest):
        return self._open_session(start_request, initial_text=None)

    def prewarm(self, connections: int = 1, *, timeout: float | None = None) -> int:
        """Establish idle Realtime sockets for short logical responses."""
        target = max(0, int(connections))
        if self.max_idle_connections == 0:
            raise ValueError("OpenAI Realtime pooling is disabled (max_idle_connections=0)")
        target = min(target, self.max_idle_connections)
        deadline = time.monotonic() + (
            self.connect_timeout if timeout is None else max(0.1, float(timeout))
        )
        while True:
            with self._pool_lock:
                if len(self._idle_connections) >= target:
                    return len(self._idle_connections)
                if self._physical_connections >= self.max_connections:
                    return len(self._idle_connections)
                self._physical_connections += 1
            conn = None
            try:
                conn = ws_connect(
                    self.endpoint,
                    timeout=self.connect_timeout,
                    headers=self.headers,
                )
                _receive_event(
                    conn,
                    expected_type="session.created",
                    timeout=max(0.1, deadline - time.monotonic()),
                )
                with self._pool_lock:
                    self._idle_connections.append(conn)
                    self._pool_lock.notify()
            except BaseException:
                if conn is not None:
                    ws_close(conn)
                self._discard_physical_slot()
                raise
            if time.monotonic() >= deadline:
                return len(self._idle_connections)

    def close(self) -> None:
        with self._sessions_lock:
            if self._closed:
                return
            self._closed = True
            sessions = list(self._sessions)
        for session in sessions:
            session.close(reason="client closed")
        with self._pool_lock:
            idle = list(self._idle_connections)
            self._idle_connections.clear()
            self._physical_connections -= len(idle)
            self._pool_lock.notify_all()
        for conn in idle:
            ws_close(conn)

    def _open_session(
        self,
        start_request: SessionStartRequest,
        *,
        initial_text: str | None,
    ) -> "OpenAIRealtimeStreamSession":
        with self._sessions_lock:
            if self._closed:
                raise StreamClosedError("Realtime adapter is closed")
        conn, reused = self._checkout_connection()
        slot_held = self.max_idle_connections == 0
        last_error: BaseException | None = None
        try:
            for attempt in range(self.reconnect_attempts + 1):
                try:
                    if conn is None:
                        conn = ws_connect(
                            self.endpoint,
                            timeout=self.connect_timeout,
                            headers=self.headers,
                        )
                    session = OpenAIRealtimeStreamSession(
                        adapter=self,
                        start_request=start_request,
                        conn=conn,
                        initial_text=initial_text,
                        reused_connection=reused,
                    )
                    with self._sessions_lock:
                        if self._closed:
                            session.close(reason="adapter closed during connect")
                            raise StreamClosedError("Realtime adapter is closed")
                        self._sessions.add(session)
                    session._slot_held = self.max_idle_connections == 0
                    slot_held = False
                    session._start_reader()
                    return session
                except BaseException as exc:
                    last_error = exc
                    if conn is not None:
                        ws_close(conn)
                        self._discard_connection(conn)
                        conn = None
                        reused = False
                    elif not slot_held:
                        self._discard_physical_slot()
                    if not isinstance(exc, (OSError, RawWebSocketError, TimeoutError)):
                        raise
                    if attempt >= self.reconnect_attempts:
                        raise
        finally:
            if slot_held:
                self._release_slot()
        assert last_error is not None  # pragma: no cover
        raise last_error

    def _checkout_connection(self) -> tuple[RawWebSocketConnection | None, bool]:
        if self.max_idle_connections == 0:
            self._acquire_slot()
            return None, False
        deadline = None if self.acquire_timeout is None else time.monotonic() + self.acquire_timeout
        with self._pool_lock:
            while True:
                if self._idle_connections:
                    return self._idle_connections.pop(), True
                if self._physical_connections < self.max_connections:
                    self._physical_connections += 1
                    return None, False
                if self._pending_acquires >= self.max_pending_acquires:
                    raise PoolSaturatedError(
                        "Realtime websocket connection pool pending queue is full"
                    )
                self._pending_acquires += 1
                try:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise PoolAcquireTimeoutError(
                            "timed out waiting for a Realtime websocket connection"
                        )
                    self._pool_lock.wait(remaining)
                finally:
                    self._pending_acquires -= 1

    def _release_connection(self, conn: RawWebSocketConnection) -> None:
        if self.max_idle_connections == 0:
            ws_close(conn)
            self._release_slot()
            return
        with self._pool_lock:
            if self._closed or len(self._idle_connections) >= self.max_idle_connections:
                self._physical_connections -= 1
                self._pool_lock.notify_all()
                close = True
            else:
                self._idle_connections.append(conn)
                self._pool_lock.notify()
                close = False
        if close:
            ws_close(conn)

    def _discard_connection(self, conn: RawWebSocketConnection) -> None:
        if self.max_idle_connections == 0:
            return
        with self._pool_lock:
            self._physical_connections = max(0, self._physical_connections - 1)
            self._pool_lock.notify_all()

    def _discard_physical_slot(self) -> None:
        with self._pool_lock:
            self._physical_connections = max(0, self._physical_connections - 1)
            self._pool_lock.notify_all()

    def _retire(self, session: "OpenAIRealtimeStreamSession") -> None:
        with self._sessions_lock:
            self._sessions.discard(session)
        if getattr(session, "_slot_held", False):
            session._slot_held = False
            self._release_slot()

    def _acquire_slot(self) -> None:
        with self._slot_lock:
            if self._pending_acquires >= self.max_pending_acquires:
                raise PoolSaturatedError(
                    "Realtime websocket connection pool pending queue is full"
                )
            self._pending_acquires += 1
        try:
            acquired = self._slot_semaphore.acquire(timeout=self.acquire_timeout)
        finally:
            with self._slot_lock:
                self._pending_acquires -= 1
        if not acquired:
            raise PoolAcquireTimeoutError(
                "timed out waiting for a Realtime websocket connection slot"
            )

    def _release_slot(self) -> None:
        self._slot_semaphore.release()


class OpenAIRealtimeStreamSession(BaseStreamSession):
    def __init__(
        self,
        *,
        adapter: OpenAIRealtimeAdapter,
        start_request: SessionStartRequest,
        conn: RawWebSocketConnection,
        initial_text: str | None,
        reused_connection: bool = False,
    ) -> None:
        session_id = start_request.session_id or f"client_{uuid.uuid4().hex}"
        super().__init__(session_id=session_id, transport=adapter.transport_name)
        self._adapter = adapter
        self._conn = conn
        self._send_lock = threading.Lock()
        self._transport_lock = threading.Lock()
        self._transport_closed = False
        self._slot_held = False
        self._input_closed = initial_text is not None
        self._next_sequence = 1
        self._chunk_index = 0
        self._audio_sample_cursor = 0
        self._last_error = ""
        self._response_terminal = False
        self.realtime_session_id = ""
        self.response_id = ""
        self.response_status = "in_progress"
        self.usage: dict = {}
        self.audio_format = AudioFormat(
            encoding="pcm_s16le",
            sample_rate=int(start_request.config.audio.sample_rate or 24000),
            channels=1,
        )
        if self.audio_format.sample_rate not in (16000, 24000):
            raise ValueError(
                "OpenAI Realtime output sample_rate must be 16000 or 24000"
            )
        if int(start_request.config.audio.channels or 1) != 1:
            raise ValueError("OpenAI Realtime output is mono only")

        created = None
        if not reused_connection:
            created = _receive_event(
                conn,
                expected_type="session.created",
                timeout=adapter.connect_timeout,
            )
        self.realtime_session_id = str(
            ((created or {}).get("session") or {}).get("id") or ""
        )
        ws_send_json(conn, _session_update(start_request, adapter.model_name))
        updated = _receive_event(
            conn,
            expected_type="session.updated",
            timeout=adapter.connect_timeout,
        )
        if not self.realtime_session_id:
            self.realtime_session_id = str(
                ((updated.get("session") or {}).get("id") or "")
            )
        if initial_text is None:
            extensions = ((updated.get("session") or {}).get("qwen") or {}).get(
                "text_buffer_extension"
            )
            if extensions != QWEN_TEXT_BUFFER_EXTENSION:
                raise ProtocolError(
                    "incremental text streaming requires server extension "
                    f"{QWEN_TEXT_BUFFER_EXTENSION!r}"
                )
            # The extension is additive. Older servers may not advertise it;
            # the session remains usable, but its tracker stays unavailable
            # until an anchor carrying output samples is received.
            self._text_progress_supported = (
                QWEN_TEXT_PROGRESS_EXTENSION
                == ((updated.get("session") or {}).get("qwen") or {}).get(
                    "text_progress_extension"
                )
            )
            ws_send_json(conn, {"type": "response.create"})
        else:
            ws_send_json(
                conn,
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": initial_text}],
                    },
                },
            )
            ws_send_json(conn, {"type": "response.create"})
            self._mark_send_closed()

        self._reader = threading.Thread(
            target=self._reader_loop,
            name=f"openai-realtime-{session_id}",
            daemon=True,
        )

    def _start_reader(self) -> None:
        self._reader.start()

    def send_text(
        self,
        text: str,
        *,
        seq_no: int | None = None,
        client_timestamp_ms: int | None = None,
    ) -> None:
        del client_timestamp_ms  # Realtime extension has sequence ordering instead.
        value = str(text or "")
        if not value:
            raise ValueError("stream text must not be empty")
        with self._send_lock:
            self._check_input_open()
            sequence = self._next_sequence if seq_no is None else int(seq_no)
            if sequence != self._next_sequence:
                raise ValueError(
                    f"text seq_no must be contiguous: expected "
                    f"{self._next_sequence}, got {sequence}"
                )
            ws_send_json(
                self._conn,
                {
                    "type": "qwen.input_text_buffer.append",
                    "sequence": sequence,
                    "text": value,
                },
            )
            self._next_sequence += 1

    def end(self, *, client_timestamp_ms: int | None = None) -> None:
        del client_timestamp_ms
        with self._send_lock:
            self._check_input_open()
            self._input_closed = True
            self._mark_send_closed()
            ws_send_json(self._conn, {"type": "qwen.input_text_buffer.commit"})

    def stop(self, *, client_timestamp_ms: int | None = None) -> None:
        self.end(client_timestamp_ms=client_timestamp_ms)

    def cancel(self, reason: str = "") -> None:
        del reason  # OpenAI response.cancel has no reason field.
        with self._send_lock:
            if self._transport_closed or self.response_status != "in_progress":
                return
            self._input_closed = True
            self._mark_send_closed()
            try:
                ws_send_json(self._conn, {"type": "response.cancel"})
            except (OSError, RawWebSocketError):
                self._finish_transport()

    def close(self, reason: str = "client closed") -> None:
        del reason
        try:
            self.cancel()
        finally:
            self._finish_transport()
            self._close_message_queue()

    def _check_input_open(self) -> None:
        if self._input_closed or self._transport_closed:
            raise StreamClosedError(
                f"Realtime stream session {self.session_id} is closed for input"
            )

    def _reader_loop(self) -> None:
        idle_deadline = time.perf_counter() + self._adapter.timeout
        try:
            while not self._transport_closed:
                remaining = idle_deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Realtime stream idle for {self._adapter.timeout:.0f}s"
                    )
                self._conn.settimeout(max(0.02, min(0.5, remaining)))
                try:
                    opcode, payload = ws_recv_frame(self._conn)
                except socket.timeout:
                    continue
                idle_deadline = time.perf_counter() + self._adapter.timeout
                if opcode == 0x8:
                    raise RawWebSocketError(
                        "Realtime connection closed without response.done"
                    )
                if opcode != 0x1:
                    continue
                event = json.loads(payload.decode("utf-8"))
                if not isinstance(event, dict):
                    continue
                if self._handle_event(event):
                    return
        except Exception as exc:
            if self.response_status == "in_progress":
                self.response_status = "failed"
                self._put_message(
                    StreamEvent(
                        type="error",
                        session_id=self.session_id,
                        message=str(exc),
                        audio=self.audio_format,
                        meta={"response_id": self.response_id},
                    )
                )
        finally:
            self._finish_transport()

    def _handle_event(self, event: dict) -> bool:
        event_type = str(event.get("type") or "")
        if event_type == "response.created":
            response = event.get("response") or {}
            self.response_id = str(response.get("id") or "")
            self._put_message(
                StreamEvent(
                    type="start",
                    session_id=self.session_id,
                    audio=self.audio_format,
                    meta={"response_id": self.response_id},
                )
            )
            return False
        if event_type == "response.output_audio.delta":
            try:
                pcm = base64.b64decode(str(event.get("delta") or ""), validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ProtocolError("invalid Base64 Realtime audio delta") from exc
            if pcm:
                sample_start = self._audio_sample_cursor
                sample_end = sample_start + len(pcm) // (
                    2 * max(1, self.audio_format.channels)
                )
                self._audio_sample_cursor = sample_end
                self._chunk_index += 1
                self._put_message(
                    AudioChunk(
                        pcm_bytes=pcm,
                        audio=self.audio_format,
                        chunk_index=self._chunk_index,
                        first_chunk=self._chunk_index == 1,
                        meta={
                            "response_id": self.response_id,
                        },
                        output_sample_start=sample_start,
                        output_sample_end=sample_end,
                    )
                )
            return False
        if event_type == "error":
            error = event.get("error") or {}
            self._last_error = str(error.get("message") or "Realtime request failed")
            self._put_message(
                StreamEvent(
                    type="warning",
                    session_id=self.session_id,
                    message=self._last_error,
                    meta={
                        "code": str(error.get("code") or ""),
                        "response_id": self.response_id,
                    },
                )
            )
            return False
        if event_type in {
            "qwen.text_token",
            "qwen.text_boundary_commit",
            "qwen.text_progress",
        }:
            meta = {
                str(key): str(value)
                for key, value in dict(event.get("meta") or {}).items()
            }
            meta["response_id"] = self.response_id
            self._put_message(
                StreamEvent(
                    type=event_type.removeprefix("qwen."),
                    session_id=self.session_id,
                    segment_id=int(event.get("segment_id", -1)),
                    text=str(event.get("text") or ""),
                    audio=self.audio_format,
                    meta=meta,
                )
            )
            return False
        if event_type != "response.done":
            return False

        response = event.get("response") or {}
        self.response_id = str(response.get("id") or self.response_id)
        self.response_status = str(response.get("status") or "completed")
        self._response_terminal = True
        usage = response.get("usage") or {}
        if isinstance(usage, dict):
            self.usage = dict(usage)
        details = response.get("status_details") or {}
        response_metadata = response.get("metadata") or {}
        failure = details.get("error") if isinstance(details, dict) else {}
        message = (
            str((failure or {}).get("message") if isinstance(failure, dict) else "")
            or self._last_error
        )
        terminal_type = "error" if self.response_status == "failed" else "done"
        self._put_message(
            StreamEvent(
                type=terminal_type,
                session_id=self.session_id,
                message=message,
                audio=self.audio_format,
                meta={
                    "response_id": self.response_id,
                    "status": self.response_status,
                    "final_output_sample": str(self._audio_sample_cursor),
                    **{
                        str(key): str(value)
                        for key, value in response_metadata.items()
                        if value is not None
                    },
                    "usage": json.dumps(
                        self.usage, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            )
        )
        return True

    def _finish_transport(self) -> None:
        with self._transport_lock:
            if self._transport_closed:
                return
            self._transport_closed = True
        if self._response_terminal and self._adapter.max_idle_connections > 0:
            self._adapter._release_connection(self._conn)
            self._adapter._retire(self)
        else:
            ws_close(self._conn)
            self._adapter._retire(self)


def _receive_event(
    conn: RawWebSocketConnection,
    *,
    expected_type: str,
    timeout: float,
) -> dict:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        conn.settimeout(max(0.02, min(0.2, deadline - time.perf_counter())))
        try:
            opcode, payload = ws_recv_frame(conn)
        except socket.timeout:
            continue
        if opcode == 0x8:
            raise RawWebSocketError(
                f"Realtime connection closed waiting for {expected_type}"
            )
        if opcode != 0x1:
            continue
        event = json.loads(payload.decode("utf-8"))
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error":
            error = event.get("error") or {}
            raise ProtocolError(
                str(error.get("message") or "Realtime handshake failed")
            )
        if event.get("type") == expected_type:
            return event
    raise TimeoutError(f"Realtime handshake timed out waiting for {expected_type}")


def _session_update(
    start_request: SessionStartRequest,
    model_name: str,
) -> dict:
    config = start_request.config
    qwen = {
        "task_type": config.task_type,
        "language": config.language,
        "speaker": config.speaker,
        "instruct": config.instruct,
        "ref_audio": maybe_base64(config.ref_audio or None),
        "ref_text": config.ref_text,
        "x_vector_only": bool(config.x_vector_only),
        "group_policy": config.group_policy or "auto",
        "output_policy": _output_policy_mapping(start_request),
        "timing": _timing_mapping(start_request),
    }
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": model_name,
            "instructions": config.instruct or "",
            "output_modalities": ["audio"],
            "audio": {
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": int(config.audio.sample_rate or 24000),
                    },
                    "voice": config.speaker or None,
                }
            },
            "qwen": {
                key: value for key, value in qwen.items() if value not in (None, "")
            },
        },
    }


def _output_policy_mapping(start_request: SessionStartRequest) -> dict:
    from qwen3tts_protocol import serialize_output_policy

    policy = start_request.output_policy or start_request.config.output_policy
    return serialize_output_policy(policy)


def _timing_mapping(start_request: SessionStartRequest) -> dict:
    from qwen3tts_protocol import serialize_timing_context

    timing = start_request.timing or start_request.config.timing_context
    return serialize_timing_context(timing)


def _with_model_query(endpoint: str, model_name: str) -> str:
    parsed = urlparse(endpoint)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("model", model_name)
    return urlunparse(parsed._replace(query=urlencode(query)))


def _capabilities_url(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    scheme = "https" if parsed.scheme == "wss" else "http"
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/realtime"):
        path = f"{path[: -len('/v1/realtime')]}/v1/capabilities"
    else:
        path = "/v1/capabilities"
    return urlunparse(parsed._replace(scheme=scheme, path=path, query="", fragment=""))


__all__ = ["OpenAIRealtimeAdapter", "OpenAIRealtimeStreamSession"]
