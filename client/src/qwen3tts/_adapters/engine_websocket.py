from __future__ import annotations

import json
import socket
import threading
import time
import weakref

from qwen3tts_protocol import (
    AudioChunk,
    AudioFormat,
    BytesResult,
    Capabilities,
    SessionStartRequest,
    StreamCancelRequest,
    StreamEvent,
    StreamTextChunk,
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
    build_bytes_result,
    capabilities_from_payload,
    decode_stream_event,
    stream_text_chunk_to_mapping,
    synthesis_config_to_mapping,
)
from .._session import BaseStreamSession, _is_terminal_message
from ..constants import TRANSPORT_ENGINE_WEBSOCKET
from ..exceptions import ProtocolError, StreamClosedError

_WEBSOCKET_REUSABLE_META_KEY = "websocket_connection_reusable"


def _finalize_connection_pool(
    pool_lock: threading.Lock,
    connections: set[RawWebSocketConnection],
    idle_connections: list[RawWebSocketConnection],
    keepalive_stop: threading.Event,
    close_connection,
) -> None:
    """Release sockets without retaining the adapter from a daemon thread."""

    keepalive_stop.set()
    with pool_lock:
        tracked = list(connections)
        connections.clear()
        idle_connections.clear()
    for conn in tracked:
        close_connection(conn)


class EngineWebSocketAdapter:
    transport_name = TRANSPORT_ENGINE_WEBSOCKET

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float,
        connect_timeout: float | None = None,
        headers: dict[str, str] | None = None,
        reconnect_attempts: int = 1,
        max_idle_connections: int = 8,
        keepalive_interval: float = 15.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        # Keep the historical behavior when omitted, while allowing callers
        # to bound a stalled handshake independently from a long stream's idle
        # timeout.  A single timeout previously made a 120-second stream budget
        # also occupy a worker thread for up to 120 seconds during connect.
        self.connect_timeout = timeout if connect_timeout is None else connect_timeout
        self.headers = dict(headers or {})
        # A websocket carries one logical TTS session at a time because audio
        # frames are raw binary data without a session id.  Finished sockets
        # are therefore pooled and reused serially; concurrent sessions simply
        # check out different sockets from the pool.
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.max_idle_connections = max(0, int(max_idle_connections))
        self.keepalive_interval = max(0.0, float(keepalive_interval))
        self._pool_lock = threading.Lock()
        self._idle_connections: list[RawWebSocketConnection] = []
        self._connections: set[RawWebSocketConnection] = set()
        self._closed = False
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        self._finalizer = weakref.finalize(
            self,
            _finalize_connection_pool,
            self._pool_lock,
            self._connections,
            self._idle_connections,
            self._keepalive_stop,
            ws_close,
        )

    def connect(self) -> None:
        """Eagerly establish and retain one authenticated websocket.

        The adapter otherwise connects lazily on the first request.  Long-lived
        workers can call this during startup so a PaaS/LB handshake never sits
        on the first user's latency path.
        """
        with self._pool_lock:
            if self._closed:
                raise StreamClosedError("websocket adapter is closed")
            if self._idle_connections:
                return
        conn = self._new_connection()
        try:
            _capabilities, reusable = self._request_capabilities(
                conn, timeout=self.connect_timeout
            )
        except Exception:
            self._discard_connection(conn)
            raise
        if reusable:
            self._release_connection(conn)
        else:
            # A legacy gateway answers capabilities and then closes the
            # socket. It is healthy, but cannot be prewarmed safely.
            self._discard_connection(conn)

    def close(self) -> None:
        """Close all idle and active websocket connections held by the pool."""
        with self._pool_lock:
            if self._closed:
                return
            self._closed = True
            connections = list(self._connections)
            self._connections.clear()
            self._idle_connections.clear()
        self._keepalive_stop.set()
        self._finalizer.detach()
        for conn in connections:
            ws_close(conn)

    def _new_connection(self) -> RawWebSocketConnection:
        last_error: BaseException | None = None
        for attempt in range(self.reconnect_attempts + 1):
            with self._pool_lock:
                if self._closed:
                    raise StreamClosedError("websocket adapter is closed")
            try:
                conn = ws_connect(
                    self.endpoint,
                    timeout=self.connect_timeout,
                    headers=self.headers,
                )
            except (OSError, RawWebSocketError) as exc:
                last_error = exc
                if attempt >= self.reconnect_attempts:
                    raise
                time.sleep(min(0.2, 0.05 * (2**attempt)))
                continue
            with self._pool_lock:
                closed = self._closed
                if not closed:
                    self._connections.add(conn)
            if closed:
                # close() may have returned while the network handshake was
                # still in flight and therefore could not see this socket.
                ws_close(conn)
                raise StreamClosedError("websocket adapter closed during connect")
            return conn
        assert last_error is not None  # pragma: no cover - loop always returns/raises
        raise last_error

    def _checkout_connection(self) -> tuple[RawWebSocketConnection, bool]:
        with self._pool_lock:
            if self._closed:
                raise StreamClosedError("websocket adapter is closed")
            while self._idle_connections:
                conn = self._idle_connections.pop()
                if conn in self._connections:
                    return conn, True
        return self._new_connection(), False

    def _acquire_connection(self) -> RawWebSocketConnection:
        """Return a live exclusive connection, reconnecting stale idle ones.

        The background keepalive validates idle sockets when enabled.  If it is
        disabled, a capabilities round-trip before reuse detects connections
        reaped by aiohttp, a proxy, or the remote peer.  In either mode an
        initial ``start`` write failure is retried by ``open_stream``.
        """
        while True:
            conn, reused = self._checkout_connection()
            if not reused or self.keepalive_interval > 0:
                return conn
            try:
                _capabilities, reusable_protocol = self._request_capabilities(conn)
                conn.settimeout(self.connect_timeout)
            except Exception:
                self._discard_connection(conn)
                continue
            if not reusable_protocol:
                self._discard_connection(conn)
                continue
            return conn

    def _release_connection(self, conn: RawWebSocketConnection) -> None:
        try:
            # Receive loops poll with 200/500 ms socket timeouts. Do not leak
            # that transport detail into the next session's initial write.
            conn.settimeout(self.connect_timeout)
        except Exception:
            self._discard_connection(conn)
            return
        close = False
        retained = False
        with self._pool_lock:
            if self._closed or conn not in self._connections:
                close = True
            elif len(self._idle_connections) >= self.max_idle_connections:
                self._connections.discard(conn)
                close = True
            elif conn not in self._idle_connections:
                self._idle_connections.append(conn)
                retained = True
        if close:
            ws_close(conn)
        elif retained:
            self._ensure_keepalive_thread()

    def _ensure_keepalive_thread(self) -> None:
        if self.keepalive_interval <= 0:
            return
        with self._pool_lock:
            if self._closed:
                return
            if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
                return
            thread = threading.Thread(
                target=self._keepalive_worker,
                args=(
                    weakref.ref(self),
                    self._keepalive_stop,
                    self.keepalive_interval,
                ),
                name="qwen3tts-ws-keepalive",
                daemon=True,
            )
            self._keepalive_thread = thread
            thread.start()

    @staticmethod
    def _keepalive_worker(
        adapter_ref,
        stop: threading.Event,
        interval: float,
    ) -> None:
        while not stop.wait(interval):
            adapter = adapter_ref()
            if adapter is None:
                return
            adapter._keepalive_once(stop)
            # Do not retain the adapter while sleeping; otherwise a forgotten
            # client.close() leaks both this daemon thread and every idle socket.
            del adapter

    def _keepalive_once(self, stop: threading.Event) -> None:
        with self._pool_lock:
            idle = list(self._idle_connections)
            self._idle_connections.clear()
        for conn in idle:
            if stop.is_set():
                return
            with self._pool_lock:
                tracked = conn in self._connections
            if not tracked:
                continue
            try:
                # This both creates application traffic for idle-reaping
                # proxies and drains/responds to websocket ping frames.
                _capabilities, reusable_protocol = self._request_capabilities(conn)
            except Exception:
                self._discard_connection(conn)
            else:
                if reusable_protocol:
                    self._release_connection(conn)
                else:
                    self._discard_connection(conn)

    def _discard_connection(self, conn: RawWebSocketConnection) -> None:
        with self._pool_lock:
            self._connections.discard(conn)
            try:
                self._idle_connections.remove(conn)
            except ValueError:
                pass
        ws_close(conn)

    def _request_capabilities(
        self,
        conn: RawWebSocketConnection,
        *,
        timeout: float | None = None,
    ) -> tuple[Capabilities, bool]:
        request_timeout = self.timeout if timeout is None else max(0.0, float(timeout))
        conn.settimeout(request_timeout)
        ws_send_json(conn, {"type": "get_capabilities"})
        deadline = time.perf_counter() + request_timeout
        while time.perf_counter() < deadline:
            conn.settimeout(max(0.05, min(0.2, deadline - time.perf_counter())))
            try:
                opcode, payload = ws_recv_frame(conn)
            except socket.timeout:
                continue
            if opcode == 0x8:
                raise RawWebSocketError("websocket closed during capabilities request")
            if opcode != 0x1:
                # A well-behaved persistent server has no leftover session
                # frames after its terminal event.  Ignore any stale binary
                # frame defensively while waiting for the probe response.
                continue
            message = json.loads(payload.decode("utf-8"))
            if message.get("type") == "capabilities":
                reusable = _is_truthy(message.get(_WEBSOCKET_REUSABLE_META_KEY, False))
                return (
                    capabilities_from_payload(message.get("capabilities", {})),
                    reusable,
                )
        raise TimeoutError("websocket capabilities request timed out")

    def get_capabilities(self, *, timeout: float | None = None) -> Capabilities:
        last_error: BaseException | None = None
        for attempt in range(self.reconnect_attempts + 1):
            conn, _reused = self._checkout_connection()
            try:
                capabilities, reusable = self._request_capabilities(
                    conn, timeout=timeout
                )
            except Exception as exc:
                last_error = exc
                self._discard_connection(conn)
                if attempt >= self.reconnect_attempts:
                    raise
                continue
            if reusable:
                self._release_connection(conn)
            else:
                self._discard_connection(conn)
            return capabilities
        assert last_error is not None  # pragma: no cover
        raise last_error

    def synthesize_bytes(self, text: str, *, request) -> BytesResult:
        session_id = request.session_id or ""
        payload = {
            "type": "oneshot",
            "session_id": session_id,
            "text": text,
            "config": synthesis_config_to_mapping(request.config),
        }
        conn: RawWebSocketConnection | None = None
        last_error: BaseException | None = None
        for attempt in range(self.reconnect_attempts + 1):
            candidate = self._acquire_connection()
            try:
                ws_send_json(candidate, payload)
            except (OSError, RawWebSocketError) as exc:
                last_error = exc
                self._discard_connection(candidate)
                if attempt >= self.reconnect_attempts:
                    raise
                continue
            conn = candidate
            break
        if conn is None:  # pragma: no cover - loop always assigns or raises
            assert last_error is not None
            raise last_error
        events: list[StreamEvent] = []
        warnings: list[str] = []
        audio_format = request.config.audio
        audio_parts: list[bytes] = []
        reusable = False
        try:
            terminal_seen = False
            for message in _iter_conn_messages(conn, timeout=self.timeout):
                if isinstance(message, AudioChunk):
                    audio_parts.append(message.pcm_bytes)
                    audio_format = message.audio
                    continue
                events.append(message)
                if message.type == "warning" and message.message:
                    warnings.append(message.message)
                if message.type in {"done", "error"}:
                    terminal_seen = True
                    reusable = _terminal_allows_connection_reuse(message)
                    break
            if not terminal_seen:
                # Connection closed (opcode 0x8) before done/error: the
                # audio collected so far is silently truncated. Fail loudly
                # instead of returning a partial result with no signal.
                raise ProtocolError(
                    "websocket stream closed without terminal event "
                    f"({len(audio_parts)} audio chunks received)"
                )
        finally:
            if reusable:
                self._release_connection(conn)
            else:
                self._discard_connection(conn)
        return build_bytes_result(
            audio_bytes=b"".join(audio_parts),
            audio_format=audio_format,
            session_id=session_id,
            transport=self.transport_name,
            events=events,
            warnings=warnings,
            details={},
        )

    def open_stream(self, start_request: SessionStartRequest):
        last_error: BaseException | None = None
        for attempt in range(self.reconnect_attempts + 1):
            conn = self._acquire_connection()
            try:
                return EngineWebSocketStreamSession(
                    adapter=self,
                    start_request=start_request,
                    conn=conn,
                )
            except (OSError, RawWebSocketError) as exc:
                # No text has been submitted yet.  Retrying a failed initial
                # start write on a fresh connection is safe; once the session
                # object is returned, mid-stream replay is deliberately left
                # to the caller because it could duplicate audio.
                last_error = exc
                self._discard_connection(conn)
                if attempt >= self.reconnect_attempts:
                    raise
        assert last_error is not None  # pragma: no cover
        raise last_error


def _iter_conn_messages(
    conn: RawWebSocketConnection,
    *,
    timeout: float,
):
    # ``timeout`` is an *idle* limit: the clock re-arms on every received
    # frame, so a healthy long synthesis can stream for arbitrarily long
    # while a silent link still fails within ``timeout``.  It used to be an
    # absolute deadline for the whole stream, which truncated any synthesis
    # whose wall time exceeded it (~230 chars of text at the observed
    # generation speed with the default 120 s).
    idle_deadline = time.perf_counter() + timeout
    current_audio = AudioFormat()
    while time.perf_counter() < idle_deadline:
        conn.settimeout(max(0.02, min(0.5, idle_deadline - time.perf_counter())))
        try:
            opcode, payload = ws_recv_frame(conn)
        except socket.timeout:
            continue
        idle_deadline = time.perf_counter() + timeout
        if opcode == 0x2:
            yield AudioChunk(
                pcm_bytes=payload,
                audio=current_audio,
                meta={},
            )
            continue
        if opcode == 0x8:
            return
        if opcode != 0x1:
            continue
        message = json.loads(payload.decode("utf-8"))
        if message.get("type") != "event":
            continue
        event = decode_stream_event(message.get("event") or {})
        if event.audio is not None:
            current_audio = event.audio
        yield event
        if event.type in {"done", "error"}:
            return
    raise TimeoutError(
        f"websocket stream idle for {timeout:.0f}s waiting for terminal event"
    )


def _terminal_allows_connection_reuse(message: StreamEvent) -> bool:
    """Only pool sockets when the gateway advertises the persistent protocol.

    Legacy gateways close the physical websocket after ``done``. Treating a
    marker-less terminal event as reusable creates a race where the next
    ``start`` can be written just before the peer's close frame arrives.
    """

    return _is_truthy(message.meta.get(_WEBSOCKET_REUSABLE_META_KEY, ""))


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class EngineWebSocketStreamSession(BaseStreamSession):
    def __init__(
        self,
        *,
        adapter: EngineWebSocketAdapter,
        start_request: SessionStartRequest,
        conn: RawWebSocketConnection,
    ) -> None:
        super().__init__(
            session_id=start_request.session_id, transport=adapter.transport_name
        )
        self._adapter = adapter
        self._start_request = start_request
        self._conn = conn
        # Serializes every send with terminal release. Without this lock a
        # caller can pass the send-open check, then the reader can return the
        # socket to the pool and a late send can corrupt the next session.
        self._transport_lock = threading.RLock()
        self._transport_finished = False
        try:
            ws_send_json(
                self._conn,
                {
                    "type": "start",
                    "session_id": start_request.session_id,
                    "config": synthesis_config_to_mapping(start_request.config),
                },
            )
        except BaseException:
            raise
        self._reader = threading.Thread(
            target=self._reader_loop, name=f"ws-session-{self.session_id}", daemon=True
        )
        self._reader.start()

    def _reader_loop(self) -> None:
        terminal_seen = False
        try:
            for message in _iter_conn_messages(
                self._conn, timeout=self._adapter.timeout
            ):
                if _is_terminal_message(message):
                    terminal_seen = True
                    with self._transport_lock:
                        self._mark_send_closed()
                        # Return the physical websocket before publishing the
                        # terminal event. A consumer that immediately calls
                        # close() after seeing done must not race and tear down
                        # a healthy connection that is already reusable.
                        self._finish_transport(
                            reusable=_terminal_allows_connection_reuse(message)
                        )
                self._put_message(message)
        except Exception as exc:
            terminal_seen = True
            self._finish_transport(reusable=False)
            self._put_message(
                StreamEvent(
                    type="error",
                    session_id=self.session_id,
                    message=str(exc),
                )
            )
        finally:
            if not terminal_seen:
                self._finish_transport(reusable=False)
                # Clean reader exit without done/error — e.g. the gateway
                # closed the connection mid-redeploy (close frame, opcode
                # 0x8). Without a terminal event the queue sentinel is never
                # enqueued and iter_messages() blocks forever, permanently
                # pinning the caller's thread (this starved a relay worker
                # pool in production). Surface it as an error so callers can
                # log and run their error path.
                self._put_message(
                    StreamEvent(
                        type="error",
                        session_id=self.session_id,
                        message="connection closed without terminal event",
                    )
                )
            # Last-resort unblock: idempotent, and covers any exit path the
            # branches above might miss.
            self._close_message_queue()

    def _finish_transport(self, *, reusable: bool) -> None:
        with self._transport_lock:
            if self._transport_finished:
                return
            self._transport_finished = True
        if reusable:
            self._adapter._release_connection(self._conn)
        else:
            self._adapter._discard_connection(self._conn)

    def send_text(
        self,
        text: str,
        *,
        seq_no: int | None = None,
        client_timestamp_ms: int | None = None,
    ) -> None:
        chunk = StreamTextChunk(
            text=text,
            seq_no=int(seq_no or 0),
            client_timestamp_ms=int(client_timestamp_ms or 0),
        )
        payload = {"type": "text"}
        payload.update(stream_text_chunk_to_mapping(chunk))
        self._send_or_close(payload)

    def end(self, *, client_timestamp_ms: int | None = None) -> None:
        self._finish_input("end", client_timestamp_ms=client_timestamp_ms)

    def stop(self, *, client_timestamp_ms: int | None = None) -> None:
        """Gracefully stop text input and drain generated audio.

        ``end()`` remains the compatibility spelling; persistent gateways also
        accept the explicit ``stop`` control message.
        """
        self._finish_input("stop", client_timestamp_ms=client_timestamp_ms)

    def _finish_input(
        self, message_type: str, *, client_timestamp_ms: int | None = None
    ) -> None:
        payload = {"type": message_type}
        if client_timestamp_ms is not None:
            payload["client_timestamp_ms"] = int(client_timestamp_ms)
        self._send_or_close(payload, close_send=True)

    def cancel(self, reason: str = "") -> None:
        request = StreamCancelRequest(reason=reason)
        try:
            self._send_or_close(
                {"type": "cancel", "reason": request.reason},
                close_send=True,
            )
        except StreamClosedError:
            # Best-effort: a dead connection already achieves what cancel
            # wanted (the server tears the session down on disconnect).
            pass

    def close(self, reason: str = "client closed") -> None:
        """Cancel and force-close the websocket from any caller thread.

        This is intentionally stronger than ``cancel()``: a stalled or broken
        link may never deliver a terminal event, so relays need a public way to
        unblock both the SDK reader and ``iter_messages()`` without reaching
        into ``session._conn`` or importing private websocket helpers.
        """
        try:
            super().close(reason=reason)
        finally:
            # Normal terminal events release the connection before they are
            # visible to consumers.  A close while the session is still active
            # remains a hard stop and discards that one physical connection so
            # a blocked reader is interrupted immediately.
            self._finish_transport(reusable=False)

    def _send_or_close(self, payload: dict, *, close_send: bool = False) -> None:
        """Send a control/text payload, mapping a dead connection to
        ``StreamClosedError``.

        From the caller's perspective a connection that died mid-stream is
        the same condition as sending after ``end()`` — the stream is closed
        for sending — so both surface the same exception type and existing
        handlers cover both.  The terminal error event still arrives through
        the reader path.
        """
        with self._transport_lock:
            self._check_send_open()
            if close_send:
                self._mark_send_closed()
            try:
                self._conn.settimeout(self._adapter.connect_timeout)
                ws_send_json(self._conn, payload)
            except (OSError, RawWebSocketError) as exc:
                self._mark_send_closed()
                self._finish_transport(reusable=False)
                raise StreamClosedError(
                    f"stream session {self.session_id} connection closed while sending"
                ) from exc
