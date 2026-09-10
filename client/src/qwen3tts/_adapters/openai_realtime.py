from __future__ import annotations

import base64
import binascii
import json
import socket
import threading
import time
import uuid
import weakref
from collections import deque
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
from .._internal.tls import TLSConfig, TLSVerify
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
    QWEN_RESPONSE_RESUME_EXTENSION,
    TRANSPORT_OPENAI_REALTIME,
)
from ..exceptions import (
    ProtocolError,
    StreamClosedError,
    StreamRecoveryError,
    SynthesisError,
)
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
        tls_verify: TLSVerify | TLSConfig = True,
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
        self._tls = TLSConfig.from_value(tls_verify)
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.active_stream_resume = bool(active_stream_resume)
        self.stream_resume_attempts = max(0, int(stream_resume_attempts))
        self.stream_resume_timeout = max(0.1, float(stream_resume_timeout))
        self.stream_resume_ack_interval = max(1, int(stream_resume_ack_interval))
        self.max_connections = int(max_connections)
        self.max_idle_connections = int(max_idle_connections)
        if (
            self.max_idle_connections < 0
            or self.max_idle_connections > self.max_connections
        ):
            raise ValueError(
                "max_idle_connections must be between 0 and max_connections"
            )
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
            **self._tls.requests_kwargs(),
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

    def _connect_websocket(self, *, timeout: float) -> RawWebSocketConnection:
        return ws_connect(
            self.endpoint,
            timeout=timeout,
            headers=self.headers,
            **self._tls.forwarding_kwargs(),
        )

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
        terminal_event = next(
            (event for event in reversed(events) if event.type in {"done", "error"}),
            None,
        )
        if terminal_event is not None and terminal_event.type == "error":
            raise SynthesisError(
                terminal_event.meta.get("code", "synthesis_failed"),
                terminal_event.message or "the Realtime server returned an error event",
            )
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
            raise ValueError(
                "OpenAI Realtime pooling is disabled (max_idle_connections=0)"
            )
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
                conn = self._connect_websocket(timeout=self.connect_timeout)
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
                        conn = self._connect_websocket(timeout=self.connect_timeout)
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
        deadline = (
            None
            if self.acquire_timeout is None
            else time.monotonic() + self.acquire_timeout
        )
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
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
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
        self._start_request = start_request
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
        self._terminal_ack_sent = False
        self._resume_token = uuid.uuid4().hex
        self._resume_supported = False
        self._last_delivery_seq = 0
        self._audio_through_sample = 0
        self._deliveries_since_ack = 0
        self._acked_text_seq = 0
        self._text_journal: deque[tuple[int, dict]] = deque()
        self._commit_sent = False
        self._recovery_lock = threading.Lock()
        self._hard_closed = False
        self._recovery_error: StreamRecoveryError | None = None
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
            qwen_extensions = (updated.get("session") or {}).get("qwen") or {}
            extensions = qwen_extensions.get("text_buffer_extension")
            if extensions != QWEN_TEXT_BUFFER_EXTENSION:
                raise ProtocolError(
                    "incremental text streaming requires server extension "
                    f"{QWEN_TEXT_BUFFER_EXTENSION!r}"
                )
            # The extension is additive. Older servers may not advertise it;
            # the session remains usable, but its tracker stays unavailable
            # until an anchor carrying output samples is received.
            self._text_progress_supported = QWEN_TEXT_PROGRESS_EXTENSION == (
                (updated.get("session") or {}).get("qwen") or {}
            ).get("text_progress_extension")
            self._resume_supported = (
                adapter.active_stream_resume
                and qwen_extensions.get("response_resume_extension")
                == QWEN_RESPONSE_RESUME_EXTENSION
            )
            ws_send_json(conn, self._response_create_payload())
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
            qwen_extensions = (updated.get("session") or {}).get("qwen") or {}
            self._resume_supported = (
                adapter.active_stream_resume
                and qwen_extensions.get("response_resume_extension")
                == QWEN_RESPONSE_RESUME_EXTENSION
            )
            ws_send_json(conn, self._response_create_payload())
            self._mark_send_closed()

        self._reader = threading.Thread(
            target=self._reader_loop,
            name=f"openai-realtime-{session_id}",
            daemon=True,
        )

    def _start_reader(self) -> None:
        self._reader.start()

    def _response_create_payload(self) -> dict:
        payload: dict = {"type": "response.create"}
        if self._resume_supported:
            payload["response"] = {
                "metadata": {"qwen_resume_token": self._resume_token}
            }
        return payload

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
            payload = {
                "type": "qwen.input_text_buffer.append",
                "sequence": sequence,
                "text": value,
            }
            if self._resume_supported:
                self._text_journal.append((sequence, dict(payload)))
            self._next_sequence += 1
            try:
                ws_send_json(self._conn, payload)
            except (OSError, RawWebSocketError) as exc:
                if not self._recover_connection(exc):
                    raise

    def end(self, *, client_timestamp_ms: int | None = None) -> None:
        del client_timestamp_ms
        with self._send_lock:
            self._check_input_open()
            self._input_closed = True
            self._commit_sent = True
            self._mark_send_closed()
            try:
                ws_send_json(self._conn, {"type": "qwen.input_text_buffer.commit"})
            except (OSError, RawWebSocketError) as exc:
                if not self._recover_connection(exc):
                    raise

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
            except (OSError, RawWebSocketError) as exc:
                if self._recover_connection(exc):
                    ws_send_json(self._conn, {"type": "response.cancel"})
                else:
                    self._finish_transport()

    def close(self, reason: str = "client closed") -> None:
        del reason
        try:
            self.cancel()
        finally:
            self._hard_closed = True
            self._finish_transport()
            self._close_message_queue()

    def update_playback_progress(
        self,
        *,
        played_through_sample: int,
        buffered_through_sample: int | None = None,
        report: bool = False,
    ):
        progress = super().update_playback_progress(
            played_through_sample=played_through_sample,
            buffered_through_sample=buffered_through_sample,
            report=False,
        )
        if not report or not self.response_id:
            return progress
        buffered = (
            played_through_sample
            if buffered_through_sample is None
            else buffered_through_sample
        )
        with self._transport_lock:
            if self._transport_closed or self._hard_closed:
                return progress
            try:
                ws_send_json(
                    self._conn,
                    {
                        "type": "qwen.playback.ack",
                        "response_id": self.response_id,
                        "played_through_sample": int(played_through_sample),
                        "buffered_through_sample": int(buffered),
                        "observed_delivery_seq": int(self._last_delivery_seq),
                        "client_monotonic_ms": int(time.monotonic() * 1000),
                    },
                )
            except (OSError, RawWebSocketError):
                # Telemetry is best effort. Recovery remains driven by the
                # synthesis stream, not by a lost playback report.
                pass
        return progress

    def _check_input_open(self) -> None:
        if self._input_closed or self._transport_closed:
            raise StreamClosedError(
                f"Realtime stream session {self.session_id} is closed for input"
            )

    def _reader_loop(self) -> None:
        idle_deadline = time.perf_counter() + self._adapter.timeout
        try:
            while not self._transport_closed:
                try:
                    remaining = idle_deadline - time.perf_counter()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Realtime stream idle for {self._adapter.timeout:.0f}s"
                        )
                    self._conn.settimeout(max(0.02, min(0.5, remaining)))
                    opcode, payload = ws_recv_frame(self._conn)
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
                except socket.timeout:
                    continue
                except Exception as exc:
                    if self._recover_connection(exc):
                        idle_deadline = time.perf_counter() + self._adapter.timeout
                        continue
                    if self.response_status == "in_progress":
                        self.response_status = "failed"
                        failure = self._recovery_error or exc
                        self._put_message(
                            StreamEvent(
                                type="error",
                                session_id=self.session_id,
                                message=str(failure),
                                audio=self.audio_format,
                                meta={"response_id": self.response_id},
                            )
                        )
                    return
        finally:
            self._finish_transport()

    def _handle_event(self, event: dict) -> bool:
        event_type = str(event.get("type") or "")
        if event_type == "response.created":
            response = event.get("response") or {}
            response_id = str(response.get("id") or "")
            if self.response_id and response_id == self.response_id:
                return False
            self.response_id = response_id
            self._put_message(
                StreamEvent(
                    type="start",
                    session_id=self.session_id,
                    audio=self.audio_format,
                    meta={"response_id": self.response_id},
                )
            )
            return False
        if event_type == "qwen.input_text_buffer.ack":
            self._handle_text_ack(event)
            return False
        if event_type == "qwen.response.delivery":
            delivery_seq = self._begin_delivery(event)
            if delivery_seq is not None:
                self._commit_delivery(
                    delivery_seq,
                    int(
                        event.get("qwen_output_sample_end", self._audio_through_sample)
                    ),
                )
            return False
        if event_type == "qwen.session.event":
            delivery_seq = self._begin_delivery(event)
            if delivery_seq is not None:
                self._commit_delivery(
                    delivery_seq,
                    int(
                        event.get("qwen_output_sample_end", self._audio_through_sample)
                    ),
                )
            return False
        if event_type == "response.output_audio.delta":
            delivery_seq = self._begin_delivery(event)
            if self._resume_supported and delivery_seq is None:
                return False
            try:
                pcm = base64.b64decode(str(event.get("delta") or ""), validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ProtocolError("invalid Base64 Realtime audio delta") from exc
            if pcm:
                sample_start = int(
                    event.get("qwen_output_sample_start", self._audio_sample_cursor)
                )
                sample_end = int(
                    event.get(
                        "qwen_output_sample_end",
                        sample_start
                        + len(pcm) // (2 * max(1, self.audio_format.channels)),
                    )
                )
                expected_end = sample_start + len(pcm) // (
                    2 * max(1, self.audio_format.channels)
                )
                if (
                    sample_start != self._audio_sample_cursor
                    or sample_end != expected_end
                ):
                    raise ProtocolError(
                        "Realtime audio sample cursor is not contiguous"
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
                if delivery_seq is not None:
                    self._commit_delivery(delivery_seq, sample_end)
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
            delivery_seq = self._begin_delivery(event)
            if self._resume_supported and delivery_seq is None:
                return False
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
            if delivery_seq is not None:
                self._commit_delivery(
                    delivery_seq,
                    int(
                        event.get("qwen_output_sample_end", self._audio_through_sample)
                    ),
                )
            return False
        if event_type != "response.done":
            return False

        response = event.get("response") or {}
        delivery_seq = self._begin_delivery(event)
        if self._resume_supported and delivery_seq is None:
            return False
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
        if delivery_seq is not None:
            self._commit_delivery(
                delivery_seq,
                int(event.get("qwen_output_sample_end", self._audio_sample_cursor)),
                terminal=True,
            )
        return True

    def _begin_delivery(self, event: dict) -> int | None:
        raw = event.get("qwen_delivery_seq")
        if raw is None:
            if self._resume_supported:
                raise ProtocolError(
                    "resumable Realtime event is missing delivery sequence"
                )
            return None
        try:
            delivery_seq = int(raw)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("qwen_delivery_seq must be an integer") from exc
        if delivery_seq <= self._last_delivery_seq:
            return None
        if delivery_seq != self._last_delivery_seq + 1:
            raise ProtocolError(
                f"Realtime delivery gap: expected {self._last_delivery_seq + 1}, "
                f"got {delivery_seq}"
            )
        return delivery_seq

    def _commit_delivery(
        self, delivery_seq: int, audio_sample: int, *, terminal: bool = False
    ) -> None:
        self._last_delivery_seq = delivery_seq
        self._audio_through_sample = audio_sample
        if terminal:
            self._send_delivery_ack(terminal=True)
            return
        self._deliveries_since_ack += 1
        if self._deliveries_since_ack >= self._adapter.stream_resume_ack_interval:
            self._send_delivery_ack(terminal=False)

    def _send_delivery_ack(self, *, terminal: bool) -> None:
        if not self._resume_supported:
            return
        ws_send_json(
            self._conn,
            {
                "type": (
                    "qwen.response.terminal_ack" if terminal else "qwen.response.ack"
                ),
                "resume_token": self._resume_token,
                "through_delivery_seq": self._last_delivery_seq,
                "audio_through_sample": self._audio_through_sample,
            },
        )
        if terminal:
            self._terminal_ack_sent = True
        self._deliveries_since_ack = 0

    def _handle_text_ack(self, event: dict) -> None:
        try:
            sequence = int(event.get("sequence", 0))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                "Realtime text ACK sequence must be an integer"
            ) from exc
        if sequence < self._acked_text_seq or sequence >= self._next_sequence:
            raise ProtocolError("Realtime text ACK is outside the sent sequence range")
        self._acked_text_seq = sequence
        while self._text_journal and self._text_journal[0][0] <= sequence:
            self._text_journal.popleft()

    def _recover_connection(self, cause: BaseException) -> bool:
        if (
            not self._resume_supported
            or self._adapter.stream_resume_attempts <= 0
            or self._response_terminal
            or self._hard_closed
        ):
            return False
        failed_conn = self._conn
        with self._recovery_lock:
            if self._conn is not failed_conn:
                return True
            deadline = time.monotonic() + self._adapter.stream_resume_timeout
            last_error: BaseException = cause
            ws_close(failed_conn)
            for attempt in range(self._adapter.stream_resume_attempts):
                replacement = None
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Realtime resume deadline expired")
                    replacement = self._adapter._connect_websocket(
                        timeout=min(self._adapter.connect_timeout, remaining),
                    )
                    _receive_event(
                        replacement,
                        expected_type="session.created",
                        timeout=min(self._adapter.connect_timeout, remaining),
                    )
                    ws_send_json(
                        replacement,
                        _session_update(self._start_request, self._adapter.model_name),
                    )
                    updated = _receive_event(
                        replacement,
                        expected_type="session.updated",
                        timeout=min(self._adapter.connect_timeout, remaining),
                    )
                    extension = ((updated.get("session") or {}).get("qwen") or {}).get(
                        "response_resume_extension"
                    )
                    if extension != QWEN_RESPONSE_RESUME_EXTENSION:
                        raise StreamRecoveryError(
                            "replacement Realtime connection does not support response resume"
                        )
                    ws_send_json(
                        replacement,
                        {
                            "type": "qwen.response.resume",
                            "resume_token": self._resume_token,
                            "last_delivery_seq": self._last_delivery_seq,
                            "audio_through_sample": self._audio_through_sample,
                        },
                    )
                    resumed = _receive_event(
                        replacement,
                        expected_type="qwen.response.resumed",
                        timeout=min(self._adapter.connect_timeout, remaining),
                    )
                    self._replay_unacked_input(replacement, resumed)
                    self._conn = replacement
                    self.realtime_session_id = str(
                        ((updated.get("session") or {}).get("id") or "")
                    )
                    self._deliveries_since_ack = 0
                    self._recovery_error = None
                    return True
                except StreamRecoveryError as exc:
                    last_error = exc
                    if replacement is not None:
                        ws_close(replacement)
                    break
                except (OSError, RawWebSocketError, TimeoutError, ProtocolError) as exc:
                    last_error = exc
                    if replacement is not None:
                        ws_close(replacement)
                    if attempt + 1 < self._adapter.stream_resume_attempts:
                        time.sleep(min(0.2, 0.05 * (2**attempt)))
            self._recovery_error = StreamRecoveryError(
                f"Realtime response resume failed: {last_error}"
            )
            return False

    def _replay_unacked_input(
        self, conn: RawWebSocketConnection, resumed: dict
    ) -> None:
        try:
            acked = int(resumed.get("acked_text_seq", 0))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("resumed acked_text_seq must be an integer") from exc
        highest_sent = self._next_sequence - 1
        if acked < self._acked_text_seq or acked > highest_sent:
            raise ProtocolError(
                f"invalid resumed text ACK {acked}; sent through {highest_sent}"
            )
        self._acked_text_seq = acked
        while self._text_journal and self._text_journal[0][0] <= acked:
            self._text_journal.popleft()
        for _sequence, payload in self._text_journal:
            ws_send_json(conn, payload)
        server_input_closed = bool(resumed.get("input_closed", False))
        if self._commit_sent and not server_input_closed:
            ws_send_json(conn, {"type": "qwen.input_text_buffer.commit"})

    def _finish_transport(self) -> None:
        with self._transport_lock:
            if self._transport_closed:
                return
            self._transport_closed = True
        reusable_terminal = self._response_terminal and (
            not self._resume_supported or self._terminal_ack_sent
        )
        if reusable_terminal and self._adapter.max_idle_connections > 0:
            self._adapter._release_connection(self._conn)
            self._adapter._retire(self)
        else:
            ws_close(self._conn)
            if self._adapter.max_idle_connections > 0:
                self._adapter._discard_connection(self._conn)
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
        "input_mode": config.input_mode or "auto",
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
