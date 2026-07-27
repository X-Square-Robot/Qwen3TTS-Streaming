from __future__ import annotations

import json
import gc
import queue
import socket
import threading
import time
import weakref

import pytest
import qwen3tts._adapters.engine_websocket as _ew

from qwen3tts_protocol import (
    BytesResult,
    Capabilities,
    SessionStartRequest,
    SynthesisConfig,
)
from qwen3tts._adapters.engine_websocket import (
    EngineWebSocketAdapter,
    EngineWebSocketStreamSession,
    _iter_conn_messages,
)
from qwen3tts._internal.raw_websocket import RawWebSocketError
from qwen3tts._session import BaseStreamSession
from qwen3tts.constants import TRANSPORT_ENGINE_WEBSOCKET
from qwen3tts.exceptions import StreamClosedError


class FakeRawWebSocketConnection:
    """Simulates a raw websocket connection."""

    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False
        self._timeout = 5.0

    def settimeout(self, value):
        self._timeout = value

    def close(self):
        self.closed = True


class PoolFakeWebSocketConnection(FakeRawWebSocketConnection):
    def __init__(self):
        super().__init__()
        self.responses: queue.Queue[dict] = queue.Queue()
        self.dead = False


def _make_ws_connect(fake_conn):
    def ws_connect(url, *, timeout, headers=None):
        return fake_conn

    return ws_connect


def _make_ws_send_json(tracker: list[dict]):
    def ws_send_json(conn, payload):
        tracker.append(payload)

    return ws_send_json


def _make_ws_recv_frame(responses):
    idx = [0]

    def ws_recv_frame(conn):
        if idx[0] >= len(responses):
            raise ConnectionError("no more frames")
        item = responses[idx[0]]
        idx[0] += 1
        if isinstance(item, bytes):
            return 0x2, item
        return 0x1, json.dumps(item).encode("utf-8")

    return ws_recv_frame


def _make_ws_close(tracker: list[bool]):
    def ws_close(conn):
        tracker[0] = True

    return ws_close


class TestEngineWebSocketAdapter:
    def test_transport_name(self):
        adapter = EngineWebSocketAdapter("ws://localhost:50052/v1/ws", timeout=5.0)
        assert adapter.transport_name == TRANSPORT_ENGINE_WEBSOCKET

    def test_get_capabilities(self, monkeypatch):
        caps_response = {
            "type": "capabilities",
            "websocket_connection_reusable": True,
            "capabilities": {
                "variant": "standalone",
                "loaded_model_type": "custom_voice",
            },
        }
        fake_conn = FakeRawWebSocketConnection()
        sent: list[dict] = []
        closed = [False]

        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_connect",
            _make_ws_connect(fake_conn),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_send_json",
            _make_ws_send_json(sent),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_recv_frame",
            _make_ws_recv_frame([caps_response]),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_close",
            _make_ws_close(closed),
        )

        adapter = EngineWebSocketAdapter("ws://localhost:50052/v1/ws", timeout=5.0)
        caps = adapter.get_capabilities()

        assert isinstance(caps, Capabilities)
        assert caps.loaded_model_type == "custom_voice"
        assert sent[0]["type"] == "get_capabilities"
        assert not closed[0], "capabilities connection should stay warm in the pool"
        adapter.close()
        assert closed[0]

    def test_legacy_capabilities_socket_is_not_pooled(self, monkeypatch):
        fake_conn = FakeRawWebSocketConnection()
        closed = [False]
        monkeypatch.setattr(_ew, "ws_connect", _make_ws_connect(fake_conn))
        monkeypatch.setattr(_ew, "ws_send_json", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            _ew,
            "ws_recv_frame",
            _make_ws_recv_frame(
                [
                    {
                        "type": "capabilities",
                        "capabilities": {"variant": "legacy"},
                    }
                ]
            ),
        )
        monkeypatch.setattr(_ew, "ws_close", _make_ws_close(closed))

        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws", timeout=1.0, reconnect_attempts=0
        )
        capabilities = adapter.get_capabilities()

        assert capabilities.variant == "legacy"
        assert closed[0]
        assert adapter._idle_connections == []

    def test_connect_timeout_is_distinct_from_request_idle_timeout(self, monkeypatch):
        caps_response = {
            "type": "capabilities",
            "websocket_connection_reusable": True,
            "capabilities": {"variant": "standalone"},
        }
        fake_conn = FakeRawWebSocketConnection()
        connect_timeouts: list[float] = []

        def connect(url, *, timeout, headers=None):
            connect_timeouts.append(timeout)
            return fake_conn

        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_connect",
            connect,
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_send_json",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_recv_frame",
            _make_ws_recv_frame([caps_response]),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_close",
            lambda _conn: None,
        )
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=120.0,
            connect_timeout=3.5,
        )

        adapter.get_capabilities()

        assert connect_timeouts == [3.5]
        assert adapter.timeout == 120.0
        assert fake_conn._timeout == 3.5

    def test_connect_timeout_defaults_to_request_timeout(self):
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=17.0,
        )

        assert adapter.connect_timeout == 17.0

    def test_get_capabilities_retries_short_receive_timeouts(self, monkeypatch):
        fake_conn = FakeRawWebSocketConnection()
        responses = iter(
            [
                socket.timeout("not ready yet"),
                {
                    "type": "capabilities",
                    "websocket_connection_reusable": True,
                    "capabilities": {"variant": "standalone"},
                },
            ]
        )

        def recv_frame(_conn):
            response = next(responses)
            if isinstance(response, BaseException):
                raise response
            return 0x1, json.dumps(response).encode("utf-8")

        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_connect",
            _make_ws_connect(fake_conn),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_send_json",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_recv_frame",
            recv_frame,
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_close",
            lambda _conn: None,
        )

        capabilities = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
        ).get_capabilities()

        assert capabilities.variant == "standalone"

    def test_synthesize_bytes_oneshot(self, monkeypatch):
        done_event = {
            "type": "event",
            "event": {"type": "done", "session_id": "s1"},
        }
        fake_conn = FakeRawWebSocketConnection()
        sent: list[dict] = []
        closed = [False]

        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_connect",
            _make_ws_connect(fake_conn),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_send_json",
            _make_ws_send_json(sent),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_recv_frame",
            _make_ws_recv_frame([done_event]),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_close",
            _make_ws_close(closed),
        )

        adapter = EngineWebSocketAdapter("ws://localhost:50052/v1/ws", timeout=5.0)
        start = SessionStartRequest(
            session_id="s1", config=SynthesisConfig(task_type="custom_voice")
        )
        result = adapter.synthesize_bytes("hello", request=start)

        assert isinstance(result, BytesResult)
        assert result.transport == TRANSPORT_ENGINE_WEBSOCKET
        assert sent[0]["type"] == "oneshot"
        assert sent[0]["text"] == "hello"

    def test_open_stream_sends_start_and_text(self, monkeypatch):
        fake_conn = FakeRawWebSocketConnection()
        sent: list[dict] = []
        closed = [False]
        responses: queue.Queue[dict] = queue.Queue()

        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_connect",
            _make_ws_connect(fake_conn),
        )

        def send(_conn, payload):
            sent.append(payload)
            if payload["type"] == "start":
                responses.put(
                    {"type": "event", "event": {"type": "start", "session_id": "s3"}}
                )
            elif payload["type"] == "end":
                responses.put(
                    {"type": "event", "event": {"type": "done", "session_id": "s3"}}
                )

        def recv(_conn):
            try:
                response = responses.get_nowait()
            except queue.Empty as exc:
                raise socket.timeout("not ready") from exc
            return 0x1, json.dumps(response).encode("utf-8")

        monkeypatch.setattr("qwen3tts._adapters.engine_websocket.ws_send_json", send)
        monkeypatch.setattr("qwen3tts._adapters.engine_websocket.ws_recv_frame", recv)
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_close",
            _make_ws_close(closed),
        )

        adapter = EngineWebSocketAdapter("ws://localhost:50052/v1/ws", timeout=5.0)
        start = SessionStartRequest(
            session_id="s3", config=SynthesisConfig(task_type="custom_voice")
        )
        session = adapter.open_stream(start)

        assert session.session_id == "s3"
        assert session.transport == TRANSPORT_ENGINE_WEBSOCKET
        # The start message should have been sent
        assert sent[0]["type"] == "start"

        session.send_text("hello")
        assert sent[1]["type"] == "text"
        assert sent[1]["text"] == "hello"

        session.end()
        assert sent[2]["type"] == "end"
        assert [message.type for message in session.iter_messages()] == [
            "start",
            "done",
        ]

    def test_open_stream_start_send_failure_closes_connection(self, monkeypatch):
        fake_conn = FakeRawWebSocketConnection()
        closed = [False]
        connect_timeouts: list[float] = []

        def connect(url, *, timeout, headers=None):
            connect_timeouts.append(timeout)
            return fake_conn

        def fail_start_send(conn, payload):
            raise OSError("start send failed")

        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_connect",
            connect,
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_send_json",
            fail_start_send,
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_close",
            _make_ws_close(closed),
        )
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=120.0,
            connect_timeout=5.0,
        )

        with pytest.raises(OSError, match="start send failed"):
            adapter.open_stream(
                SessionStartRequest(
                    session_id="s-start-failed",
                    config=SynthesisConfig(task_type="custom_voice"),
                )
            )

        assert connect_timeouts == [5.0, 5.0]
        assert closed[0]

    def test_completed_streams_reuse_one_physical_connection(self, monkeypatch):
        connections: list[PoolFakeWebSocketConnection] = []
        sent: list[tuple[PoolFakeWebSocketConnection, dict]] = []
        send_timeouts: list[tuple[str, float]] = []

        def connect(_url, *, timeout, headers=None):
            conn = PoolFakeWebSocketConnection()
            connections.append(conn)
            return conn

        def send(conn, payload):
            if conn.dead:
                raise RawWebSocketError("stale pooled connection")
            sent.append((conn, payload))
            message_type = payload["type"]
            send_timeouts.append((message_type, conn._timeout))
            if message_type == "get_capabilities":
                conn.responses.put(
                    {
                        "type": "capabilities",
                        "websocket_connection_reusable": True,
                        "capabilities": {"variant": "standalone"},
                    }
                )
            elif message_type == "start":
                conn.responses.put(
                    {
                        "type": "event",
                        "event": {
                            "type": "start",
                            "session_id": payload["session_id"],
                        },
                    }
                )
            elif message_type in {"end", "stop"}:
                conn.responses.put(
                    {
                        "type": "event",
                        "event": {
                            "type": "done",
                            "session_id": "done",
                            "meta": {"websocket_connection_reusable": "true"},
                        },
                    }
                )

        def recv(conn):
            try:
                response = conn.responses.get_nowait()
            except queue.Empty as exc:
                raise socket.timeout("not ready") from exc
            return 0x1, json.dumps(response).encode("utf-8")

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", recv)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: setattr(conn, "closed", True))

        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            connect_timeout=0.75,
            reconnect_attempts=0,
            keepalive_interval=0,
        )
        for index in range(2):
            session = adapter.open_stream(
                SessionStartRequest(
                    session_id=f"reuse-{index}",
                    config=SynthesisConfig(task_type="custom_voice"),
                )
            )
            session.send_text("hello")
            if index == 0:
                session.end()
            else:
                session.stop()
            assert [event.type for event in session.iter_messages()] == [
                "start",
                "done",
            ]

        assert len(connections) == 1
        assert [payload["type"] for _conn, payload in sent] == [
            "start",
            "text",
            "end",
            "get_capabilities",
            "start",
            "text",
            "stop",
        ]
        assert send_timeouts[4] == ("start", 0.75)
        assert not connections[0].closed
        adapter.close()
        assert connections[0].closed

    def test_legacy_terminal_without_reuse_marker_discards_connection(
        self, monkeypatch
    ):
        conn = PoolFakeWebSocketConnection()

        def send(_conn, payload):
            if payload["type"] == "start":
                conn.responses.put(
                    {
                        "type": "event",
                        "event": {
                            "type": "start",
                            "session_id": payload["session_id"],
                        },
                    }
                )
            elif payload["type"] == "end":
                # Legacy gateways sent an unmarked terminal event and then
                # closed the physical websocket.
                conn.responses.put(
                    {
                        "type": "event",
                        "event": {"type": "done", "session_id": "legacy"},
                    }
                )

        def recv(_conn):
            try:
                response = conn.responses.get_nowait()
            except queue.Empty as exc:
                raise socket.timeout("not ready") from exc
            return 0x1, json.dumps(response).encode("utf-8")

        monkeypatch.setattr(_ew, "ws_connect", _make_ws_connect(conn))
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", recv)
        monkeypatch.setattr(_ew, "ws_close", lambda raw: setattr(raw, "closed", True))

        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws", timeout=1.0, reconnect_attempts=0
        )
        session = adapter.open_stream(
            SessionStartRequest(session_id="legacy", config=SynthesisConfig())
        )
        session.end()
        assert [message.type for message in session.iter_messages()] == [
            "start",
            "done",
        ]

        assert conn.closed
        assert adapter._idle_connections == []

    def test_stale_idle_connection_is_reconnected_before_next_start(self, monkeypatch):
        connections: list[PoolFakeWebSocketConnection] = []

        def connect(_url, *, timeout, headers=None):
            conn = PoolFakeWebSocketConnection()
            connections.append(conn)
            return conn

        def send(conn, payload):
            if conn.dead:
                raise RawWebSocketError("proxy reaped idle connection")
            if payload["type"] == "get_capabilities":
                conn.responses.put(
                    {"type": "capabilities", "capabilities": {"variant": "standalone"}}
                )
            elif payload["type"] == "start":
                conn.responses.put(
                    {
                        "type": "event",
                        "event": {"type": "start", "session_id": payload["session_id"]},
                    }
                )
            elif payload["type"] == "end":
                conn.responses.put(
                    {
                        "type": "event",
                        "event": {
                            "type": "done",
                            "session_id": "s",
                            "meta": {"websocket_connection_reusable": "true"},
                        },
                    }
                )

        def recv(conn):
            try:
                response = conn.responses.get_nowait()
            except queue.Empty as exc:
                raise socket.timeout("not ready") from exc
            return 0x1, json.dumps(response).encode("utf-8")

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", recv)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: setattr(conn, "closed", True))

        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            keepalive_interval=0,
        )
        first = adapter.open_stream(
            SessionStartRequest(session_id="first", config=SynthesisConfig())
        )
        first.end()
        list(first.iter_messages())
        connections[0].dead = True

        second = adapter.open_stream(
            SessionStartRequest(session_id="second", config=SynthesisConfig())
        )
        second.end()
        list(second.iter_messages())

        assert len(connections) == 2
        assert connections[0].closed
        assert not connections[1].closed

    def test_oneshot_retries_initial_write_on_a_stale_pooled_connection(
        self, monkeypatch
    ):
        stale = PoolFakeWebSocketConnection()
        stale.dead = True
        fresh = PoolFakeWebSocketConnection()
        connect_calls = []

        def connect(_url, *, timeout, headers=None):
            connect_calls.append(headers)
            return fresh

        def send(conn, payload):
            if conn.dead:
                raise RawWebSocketError("stale")
            if payload["type"] == "oneshot":
                conn.responses.put(
                    {
                        "type": "event",
                        "event": {
                            "type": "done",
                            "session_id": "oneshot",
                            "meta": {"websocket_connection_reusable": "true"},
                        },
                    }
                )

        def recv(conn):
            try:
                response = conn.responses.get_nowait()
            except queue.Empty as exc:
                raise socket.timeout("not ready") from exc
            return 0x1, json.dumps(response).encode("utf-8")

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", recv)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: setattr(conn, "closed", True))

        adapter = EngineWebSocketAdapter(
            "wss://tts.example/v1/ws",
            timeout=1.0,
            headers={"Authorization": "Bearer secret"},
        )
        adapter._connections.add(stale)
        adapter._idle_connections.append(stale)
        result = adapter.synthesize_bytes(
            "hello",
            request=SessionStartRequest(session_id="oneshot", config=SynthesisConfig()),
        )

        assert result.events[-1].type == "done"
        assert stale.closed
        assert connect_calls == [{"Authorization": "Bearer secret"}]
        assert fresh in adapter._idle_connections

    def test_terminal_waits_for_inflight_send_before_returning_socket_to_pool(
        self, monkeypatch
    ):
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            keepalive_interval=0,
        )
        conn = FakeRawWebSocketConnection()
        adapter._connections.add(conn)
        session = EngineWebSocketStreamSession.__new__(EngineWebSocketStreamSession)
        BaseStreamSession.__init__(
            session, session_id="race", transport="engine-websocket"
        )
        session._adapter = adapter
        session._conn = conn
        session._transport_lock = threading.RLock()
        session._transport_finished = False

        send_entered = threading.Event()
        allow_send = threading.Event()
        terminal_started = threading.Event()
        terminal_finished = threading.Event()

        def blocking_send(_conn, payload):
            assert payload["type"] == "text"
            send_entered.set()
            assert allow_send.wait(timeout=1.0)

        monkeypatch.setattr(_ew, "ws_send_json", blocking_send)
        monkeypatch.setattr(_ew, "ws_close", lambda _conn: None)

        sender = threading.Thread(target=session.send_text, args=("late",))

        def finish_terminal():
            terminal_started.set()
            with session._transport_lock:
                session._mark_send_closed()
                session._finish_transport(reusable=True)
            terminal_finished.set()

        terminal = threading.Thread(target=finish_terminal)
        sender.start()
        assert send_entered.wait(timeout=1.0)
        terminal.start()
        assert terminal_started.wait(timeout=1.0)
        assert not terminal_finished.wait(timeout=0.05)
        assert adapter._idle_connections == []

        allow_send.set()
        sender.join(timeout=1.0)
        terminal.join(timeout=1.0)
        assert not sender.is_alive()
        assert terminal_finished.is_set()
        checked_out, reused = adapter._checkout_connection()
        assert reused is True
        assert checked_out is conn

    def test_close_wins_against_inflight_connection_handshake(self, monkeypatch):
        conn = FakeRawWebSocketConnection()
        connect_entered = threading.Event()
        allow_connect = threading.Event()
        closed = []
        errors = []

        def blocking_connect(_url, *, timeout, headers=None):
            connect_entered.set()
            assert allow_connect.wait(timeout=1.0)
            return conn

        monkeypatch.setattr(_ew, "ws_connect", blocking_connect)
        monkeypatch.setattr(_ew, "ws_close", lambda raw: closed.append(raw))
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws", timeout=1.0, reconnect_attempts=0
        )

        def run_connect():
            try:
                adapter.connect()
            except Exception as exc:
                errors.append(exc)

        connector = threading.Thread(target=run_connect)
        connector.start()
        assert connect_entered.wait(timeout=1.0)
        adapter.close()
        allow_connect.set()
        connector.join(timeout=1.0)

        assert not connector.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], StreamClosedError)
        assert conn in closed
        assert adapter._connections == set()
        assert adapter._idle_connections == []

    def test_keepalive_thread_does_not_retain_forgotten_adapter(self, monkeypatch):
        conn = FakeRawWebSocketConnection()
        closed = []
        monkeypatch.setattr(_ew, "ws_close", lambda raw: closed.append(raw))
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            keepalive_interval=60.0,
        )
        adapter._connections.add(conn)
        adapter._idle_connections.append(conn)
        adapter._ensure_keepalive_thread()
        thread = adapter._keepalive_thread
        reference = weakref.ref(adapter)

        del adapter
        gc.collect()
        assert reference() is None
        assert thread is not None
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert closed == [conn]


class TestConnectionClosedWithoutTerminal:
    """Close frame (0x8) without done/error must not hang iter_messages().

    A gateway redeploy tears connections down with a bare close frame; the
    reader used to exit cleanly without enqueueing the queue sentinel, so
    iter_messages() blocked forever and permanently pinned the caller's
    thread (starved a relay worker pool in production).
    """

    @staticmethod
    def _recv_with_close(responses):
        idx = [0]

        def ws_recv_frame(conn):
            if idx[0] >= len(responses):
                raise ConnectionError("no more frames")
            item = responses[idx[0]]
            idx[0] += 1
            if item == "__close__":
                return 0x8, b""
            if isinstance(item, bytes):
                return 0x2, item
            return 0x1, json.dumps(item).encode("utf-8")

        return ws_recv_frame

    def _patch(self, monkeypatch, responses):
        fake_conn = FakeRawWebSocketConnection()
        sent: list[dict] = []
        closed = [False]
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_connect",
            _make_ws_connect(fake_conn),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_send_json",
            _make_ws_send_json(sent),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_recv_frame",
            self._recv_with_close(responses),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_close",
            _make_ws_close(closed),
        )

    def test_stream_close_frame_yields_error_and_terminates(self, monkeypatch):
        import threading

        from qwen3tts_protocol import AudioChunk, StreamEvent

        self._patch(monkeypatch, [b"\x00\x01\x02\x03", "__close__"])
        adapter = EngineWebSocketAdapter("ws://localhost:50052/v1/ws", timeout=5.0)
        session = adapter.open_stream(
            SessionStartRequest(
                session_id="s-close",
                config=SynthesisConfig(task_type="custom_voice"),
            )
        )

        messages: list = []

        def consume():
            for msg in session.iter_messages():
                messages.append(msg)

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        consumer.join(timeout=5.0)
        assert not consumer.is_alive(), "iter_messages() hung after close frame"
        assert isinstance(messages[0], AudioChunk)
        assert isinstance(messages[-1], StreamEvent)
        assert messages[-1].type == "error"
        assert "without terminal event" in messages[-1].message

    def test_oneshot_close_frame_raises_instead_of_truncating(self, monkeypatch):
        from qwen3tts.exceptions import ProtocolError

        self._patch(monkeypatch, [b"\x00\x01\x02\x03", "__close__"])
        adapter = EngineWebSocketAdapter("ws://localhost:50052/v1/ws", timeout=5.0)
        start = SessionStartRequest(
            session_id="s-oneshot-close",
            config=SynthesisConfig(task_type="custom_voice"),
        )
        try:
            adapter.synthesize_bytes("hello", request=start)
            raise AssertionError("expected ProtocolError on truncated stream")
        except ProtocolError as exc:
            assert "without terminal event" in str(exc)


# ---------------------------------------------------------------------------
# Idle-timeout semantics (_iter_conn_messages) & dead-connection send mapping
# ---------------------------------------------------------------------------


class TestIterConnMessagesIdleTimeout:
    def test_slow_but_alive_stream_outlives_timeout(self, monkeypatch):
        """timeout 是空闲上限而非整流总死线：帧间隔 < timeout 但总时长 > timeout
        的健康长流必须完整走完（旧的绝对死线语义会在 timeout 处拦腰截断）。"""
        frames = [b"\x00\x01"] * 4 + [
            {"type": "event", "event": {"type": "done", "session_id": "s"}}
        ]
        idx = [0]

        def fake_recv(conn):
            if idx[0] >= len(frames):
                raise AssertionError("read past terminal event")
            time.sleep(0.15)  # 每帧间隔 0.15s < timeout=0.3s；总时长 0.75s > 0.3s
            item = frames[idx[0]]
            idx[0] += 1
            if isinstance(item, bytes):
                return 0x2, item
            return 0x1, json.dumps(item).encode("utf-8")

        monkeypatch.setattr(_ew, "ws_recv_frame", fake_recv)
        conn = FakeRawWebSocketConnection()

        messages = list(_iter_conn_messages(conn, timeout=0.3))

        audio = [m for m in messages if getattr(m, "pcm_bytes", None)]
        assert len(audio) == 4
        assert messages[-1].type == "done"

    def test_silent_link_fails_within_idle_timeout(self, monkeypatch):
        def fake_recv(conn):
            time.sleep(0.02)
            raise socket.timeout("no data")

        monkeypatch.setattr(_ew, "ws_recv_frame", fake_recv)
        conn = FakeRawWebSocketConnection()

        start = time.perf_counter()
        with pytest.raises(TimeoutError, match="idle"):
            list(_iter_conn_messages(conn, timeout=0.3))
        assert time.perf_counter() - start < 2.0


class TestSendOnDeadConnection:
    @staticmethod
    def _make_session() -> EngineWebSocketStreamSession:
        session = EngineWebSocketStreamSession.__new__(EngineWebSocketStreamSession)
        BaseStreamSession.__init__(
            session, session_id="s-dead", transport="engine-websocket"
        )
        session._conn = FakeRawWebSocketConnection()
        session._adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws", timeout=5.0, reconnect_attempts=0
        )
        session._adapter._connections.add(session._conn)
        session._transport_lock = threading.RLock()
        session._transport_finished = False
        return session

    @staticmethod
    def _raise_raw(conn, payload):
        raise RawWebSocketError("websocket closed")

    def test_send_text_maps_to_stream_closed(self, monkeypatch):
        monkeypatch.setattr(_ew, "ws_send_json", self._raise_raw)
        session = self._make_session()
        with pytest.raises(StreamClosedError, match="connection closed"):
            session.send_text("你好")
        # 之后的 send 走 _check_send_open 短路，同一异常类型
        with pytest.raises(StreamClosedError):
            session.send_text("再见")

    def test_end_maps_to_stream_closed(self, monkeypatch):
        monkeypatch.setattr(_ew, "ws_send_json", self._raise_raw)
        session = self._make_session()
        with pytest.raises(StreamClosedError, match="connection closed"):
            session.end()

    def test_cancel_on_dead_connection_is_silent(self, monkeypatch):
        monkeypatch.setattr(_ew, "ws_send_json", self._raise_raw)
        session = self._make_session()
        session.cancel(reason="interrupt")  # 不抛：死连接下 cancel 目的已达成

    def test_close_is_public_hard_stop_and_unblocks_consumer(self, monkeypatch):
        session = self._make_session()
        sent: list[dict] = []
        closed = [False]
        monkeypatch.setattr(_ew, "ws_send_json", _make_ws_send_json(sent))
        monkeypatch.setattr(_ew, "ws_close", _make_ws_close(closed))

        session.close(reason="worker shutdown")

        assert sent == [{"type": "cancel", "reason": "worker shutdown"}]
        assert closed[0]
        assert list(session.iter_messages()) == []
