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
from qwen3tts.exceptions import (
    EngineVersionMismatchError,
    PoolAcquireTimeoutError,
    PoolSaturatedError,
    ProtocolVersionMismatchError,
    StreamClosedError,
    StreamRecoveryError,
)


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


class RecoveryFakeWebSocketConnection(FakeRawWebSocketConnection):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.frames: queue.Queue[object] = queue.Queue()


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

    def test_bounded_pool_defaults_and_disabled_lifetime_aliases(self):
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=5.0,
            idle_ttl=0,
            max_lifetime=0,
        )

        assert adapter.max_connections == 32
        assert adapter.max_idle_connections == 8
        assert adapter.max_pending_acquires == 256
        assert adapter.acquire_timeout == 30.0
        assert adapter.idle_ttl is None
        assert adapter.max_lifetime is None
        assert adapter.keepalive_jitter == 0.2

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_connections": 0}, "max_connections"),
            (
                {"max_connections": 2, "max_idle_connections": 3},
                "max_idle_connections",
            ),
            ({"max_pending_acquires": -1}, "max_pending_acquires"),
            ({"acquire_timeout": -1}, "acquire_timeout"),
            ({"idle_ttl": -1}, "idle_ttl"),
            ({"max_lifetime": float("inf")}, "max_lifetime"),
            ({"keepalive_jitter": 1.0}, "keepalive_jitter"),
        ],
    )
    def test_rejects_invalid_pool_configuration(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            EngineWebSocketAdapter("ws://localhost:50052/v1/ws", timeout=5.0, **kwargs)

    def test_unexpected_replacement_dial_error_releases_reserved_slot(
        self, monkeypatch
    ):
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=1,
            max_idle_connections=1,
            keepalive_interval=0,
        )
        old = FakeRawWebSocketConnection()
        adapter._connections.add(old)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())

        def unexpected_dial(*_args, **_kwargs):
            raise ValueError("unexpected websocket constructor failure")

        monkeypatch.setattr(_ew, "ws_connect", unexpected_dial)

        with pytest.raises(ValueError, match="constructor failure"):
            adapter._replace_connection(
                old,
                connect_timeout=0.1,
                preserve_reservation_on_failure=True,
            )

        assert old.closed
        assert adapter._connecting == 0
        assert adapter._connections == set()

    def test_adapter_close_during_replacement_releases_reserved_slot_once(
        self, monkeypatch
    ):
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=1,
            max_idle_connections=1,
            keepalive_interval=0,
        )
        old = FakeRawWebSocketConnection()
        adapter._connections.add(old)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())

        def close_during_dial(*_args, **_kwargs):
            adapter.close()
            raise RawWebSocketError("dial interrupted by adapter close")

        monkeypatch.setattr(_ew, "ws_connect", close_during_dial)

        with pytest.raises(RawWebSocketError, match="adapter close"):
            adapter._replace_connection(
                old,
                connect_timeout=0.1,
                preserve_reservation_on_failure=True,
            )

        assert old.closed
        assert adapter._closed
        assert adapter._connecting == 0
        assert adapter._connections == set()

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

    def test_get_capabilities_uses_per_call_timeout(self, monkeypatch):
        fake_conn = FakeRawWebSocketConnection()
        timeouts = []

        def settimeout(value):
            timeouts.append(value)
            fake_conn._timeout = value

        fake_conn.settimeout = settimeout
        monkeypatch.setattr(_ew, "ws_connect", _make_ws_connect(fake_conn))
        monkeypatch.setattr(_ew, "ws_send_json", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            _ew,
            "ws_recv_frame",
            _make_ws_recv_frame(
                [
                    {
                        "type": "capabilities",
                        "capabilities": {"variant": "standalone"},
                    }
                ]
            ),
        )
        monkeypatch.setattr(_ew, "ws_close", lambda _conn: None)

        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=120.0,
            connect_timeout=5.0,
            reconnect_attempts=0,
        )
        adapter.get_capabilities(timeout=2.5)

        assert timeouts[0] == 2.5

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

    def test_prewarm_establishes_target_connections_concurrently(self, monkeypatch):
        connections: list[PoolFakeWebSocketConnection] = []
        barrier = threading.Barrier(3)
        state_lock = threading.Lock()
        active = 0
        max_active = 0

        def connect(_url, *, timeout, headers=None):
            nonlocal active, max_active
            conn = PoolFakeWebSocketConnection()
            with state_lock:
                connections.append(conn)
                active += 1
                max_active = max(max_active, active)
            barrier.wait(timeout=2.0)
            with state_lock:
                active -= 1
            return conn

        def send(conn, payload):
            assert payload == {"type": "get_capabilities"}
            conn.responses.put(
                {
                    "type": "capabilities",
                    "websocket_connection_reusable": True,
                    "capabilities": {"variant": "standalone"},
                }
            )

        def recv(conn):
            response = conn.responses.get_nowait()
            return 0x1, json.dumps(response).encode("utf-8")

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", recv)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=120.0,
            connect_timeout=0.75,
            reconnect_attempts=0,
            max_idle_connections=3,
            keepalive_interval=0,
        )

        idle_count = adapter.prewarm(3, timeout=0.5)

        assert idle_count == 3
        assert max_active == 3
        assert set(adapter._idle_connections) == set(connections)
        assert adapter._connections == set(connections)
        assert adapter.prewarm(10) == 3
        assert len(connections) == 3

    def test_prewarm_keeps_partial_success_and_reports_shortfall(self, monkeypatch):
        connections: list[PoolFakeWebSocketConnection] = []
        state_lock = threading.Lock()

        def connect(_url, *, timeout, headers=None):
            conn = PoolFakeWebSocketConnection()
            with state_lock:
                conn.reusable = not connections
                connections.append(conn)
            return conn

        def send(conn, payload):
            response = {
                "type": "capabilities",
                "capabilities": {"variant": "standalone"},
            }
            if conn.reusable:
                response["websocket_connection_reusable"] = True
            conn.responses.put(response)

        def recv(conn):
            response = conn.responses.get_nowait()
            return 0x1, json.dumps(response).encode("utf-8")

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", recv)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_idle_connections=2,
            keepalive_interval=0,
        )

        with pytest.raises(RuntimeError, match=r"reached 1/2") as excinfo:
            adapter.prewarm(2)

        assert "websocket_connection_reusable=true" in str(excinfo.value)
        assert adapter._idle_connections == [connections[0]]
        assert not connections[0].closed
        assert connections[1].closed

    @pytest.mark.parametrize(
        "error_type",
        [ProtocolVersionMismatchError, EngineVersionMismatchError],
    )
    def test_prewarm_preserves_version_mismatch_error(self, monkeypatch, error_type):
        conn = FakeRawWebSocketConnection()
        monkeypatch.setattr(_ew, "ws_connect", _make_ws_connect(conn))
        monkeypatch.setattr(_ew, "ws_close", lambda raw: raw.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            keepalive_interval=0,
        )

        def fail_version_check(_conn, *, timeout=None):
            raise error_type("version mismatch")

        monkeypatch.setattr(adapter, "_request_capabilities", fail_version_check)

        with pytest.raises(error_type, match="version mismatch"):
            adapter.prewarm(1)

        assert conn.closed

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
            elif message_type in {"end", "stop", "cancel"}:
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
        for index in range(3):
            session = adapter.open_stream(
                SessionStartRequest(
                    session_id=f"reuse-{index}",
                    config=SynthesisConfig(task_type="custom_voice"),
                )
            )
            session.send_text("hello")
            if index == 0:
                session.end()
            elif index == 1:
                session.cancel(reason="interrupt")
            else:
                session.stop()
            assert [event.type for event in session.iter_messages()] == [
                "start",
                "done",
            ]
            # A relay's delayed hard-close fallback may fire after the cancel
            # terminal has already returned this socket to the pool.  It must
            # be transport-idempotent and leave the pooled socket reusable.
            if index == 1:
                session.close(reason="cancel fallback")

        assert len(connections) == 1
        assert [payload["type"] for _conn, payload in sent] == [
            "start",
            "text",
            "end",
            "get_capabilities",
            "start",
            "text",
            "cancel",
            "get_capabilities",
            "start",
            "text",
            "stop",
        ]
        assert send_timeouts[4] == ("start", 0.75)
        assert send_timeouts[3] == ("get_capabilities", 0.75)
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

    def test_keepalive_probe_uses_connect_timeout(self, monkeypatch):
        conn = FakeRawWebSocketConnection()
        seen_timeouts = []
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=120.0,
            connect_timeout=1.25,
            keepalive_interval=0,
        )
        adapter._connections.add(conn)
        adapter._idle_connections.append(conn)

        def capabilities(_conn, *, timeout=None):
            seen_timeouts.append(timeout)
            return Capabilities(), True

        monkeypatch.setattr(adapter, "_request_capabilities", capabilities)
        monkeypatch.setattr(_ew, "ws_close", lambda raw: raw.close())

        adapter._keepalive_once(threading.Event())

        assert seen_timeouts == [1.25]
        assert adapter._idle_connections == [conn]

    def test_pool_bounds_connecting_active_and_idle_connections(self, monkeypatch):
        connections = []
        active = 0
        max_active = 0
        state_lock = threading.Lock()
        start = threading.Barrier(13)
        release = threading.Event()
        errors = []

        def connect(_url, *, timeout, headers=None):
            conn = FakeRawWebSocketConnection()
            with state_lock:
                connections.append(conn)
            return conn

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=3,
            max_idle_connections=3,
            max_pending_acquires=12,
            acquire_timeout=1.0,
            keepalive_interval=0,
        )

        def lease():
            nonlocal active, max_active
            try:
                start.wait(timeout=1.0)
                conn, _reused = adapter._checkout_connection()
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                assert release.wait(timeout=1.0)
                with state_lock:
                    active -= 1
                adapter._release_connection(conn)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=lease) for _ in range(12)]
        for thread in threads:
            thread.start()
        start.wait(timeout=1.0)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with state_lock:
                if active == 3:
                    break
            time.sleep(0.005)
        release.set()
        for thread in threads:
            thread.join(timeout=2.0)

        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert len(connections) == 3
        assert max_active == 3
        assert adapter._connecting == 0
        assert len(adapter._connections) == 3
        assert len(adapter._idle_connections) == 3

    def test_connecting_handshake_consumes_the_only_pool_slot(self, monkeypatch):
        connect_entered = threading.Event()
        allow_connect = threading.Event()
        first_conn = FakeRawWebSocketConnection()
        connect_calls = 0

        def connect(*_args, **_kwargs):
            nonlocal connect_calls
            connect_calls += 1
            connect_entered.set()
            assert allow_connect.wait(timeout=1.0)
            return first_conn

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=1,
            max_idle_connections=1,
            acquire_timeout=0.03,
            keepalive_interval=0,
        )
        leased = []

        def first_acquire():
            conn, _ = adapter._checkout_connection()
            leased.append(conn)

        connector = threading.Thread(target=first_acquire)
        connector.start()
        assert connect_entered.wait(timeout=1.0)
        with pytest.raises(PoolAcquireTimeoutError):
            adapter._checkout_connection()

        assert adapter._connecting == 1
        assert adapter._connections == set()
        assert connect_calls == 1
        allow_connect.set()
        connector.join(timeout=1.0)
        assert leased == [first_conn]
        assert adapter._connecting == 0
        adapter._release_connection(first_conn)

    def test_pool_queue_is_bounded_and_times_out(self, monkeypatch):
        monkeypatch.setattr(
            _ew,
            "ws_connect",
            lambda *_args, **_kwargs: FakeRawWebSocketConnection(),
        )
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=1,
            max_idle_connections=1,
            max_pending_acquires=1,
            acquire_timeout=0.05,
            keepalive_interval=0,
        )
        held, _ = adapter._checkout_connection()
        waiter_error = []

        def wait_for_lease():
            try:
                adapter._checkout_connection()
            except Exception as exc:
                waiter_error.append(exc)

        waiter = threading.Thread(target=wait_for_lease)
        waiter.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with adapter._pool_lock:
                if len(adapter._acquire_waiters) == 1:
                    break
            time.sleep(0.005)

        with pytest.raises(PoolSaturatedError, match="max_pending_acquires=1"):
            adapter._checkout_connection()

        waiter.join(timeout=1.0)
        assert len(waiter_error) == 1
        assert isinstance(waiter_error[0], PoolAcquireTimeoutError)
        adapter._release_connection(held)

    def test_pool_waiters_receive_released_connection_in_fifo_order(self, monkeypatch):
        created = []

        def connect(*_args, **_kwargs):
            conn = FakeRawWebSocketConnection()
            created.append(conn)
            return conn

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=1,
            max_idle_connections=0,
            max_pending_acquires=3,
            acquire_timeout=1.0,
            keepalive_interval=0,
        )
        held, _ = adapter._checkout_connection()
        order = []

        def lease(index):
            conn, _ = adapter._checkout_connection()
            order.append(index)
            adapter._release_connection(conn)

        threads = []
        for index in range(3):
            thread = threading.Thread(target=lease, args=(index,))
            thread.start()
            threads.append(thread)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                with adapter._pool_lock:
                    if len(adapter._acquire_waiters) == index + 1:
                        break
                time.sleep(0.005)

        adapter._release_connection(held)
        for thread in threads:
            thread.join(timeout=1.0)

        assert order == [0, 1, 2]
        assert len(created) == 1
        assert created[0].closed

    def test_close_wakes_all_pool_waiters(self, monkeypatch):
        monkeypatch.setattr(
            _ew,
            "ws_connect",
            lambda *_args, **_kwargs: FakeRawWebSocketConnection(),
        )
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=1,
            max_idle_connections=1,
            max_pending_acquires=2,
            acquire_timeout=None,
            keepalive_interval=0,
        )
        adapter._checkout_connection()
        errors = []

        def wait_for_lease():
            try:
                adapter._checkout_connection()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=wait_for_lease) for _ in range(2)]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with adapter._pool_lock:
                if len(adapter._acquire_waiters) == 2:
                    break
            time.sleep(0.005)

        adapter.close()
        for thread in threads:
            thread.join(timeout=1.0)

        assert len(errors) == 2
        assert all(isinstance(error, StreamClosedError) for error in errors)
        assert all(not thread.is_alive() for thread in threads)

    def test_idle_ttl_and_max_lifetime_retire_only_idle_or_released_connections(
        self, monkeypatch
    ):
        connections = []

        def connect(*_args, **_kwargs):
            conn = FakeRawWebSocketConnection()
            connections.append(conn)
            return conn

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=1,
            max_idle_connections=1,
            idle_ttl=1.0,
            max_lifetime=2.0,
            keepalive_interval=0,
        )
        old, _ = adapter._checkout_connection()
        adapter._release_connection(old)
        adapter._connection_idle_since[old] = time.monotonic() - 1.1

        replacement, reused = adapter._checkout_connection()

        assert not reused
        assert old.closed
        assert replacement is connections[1]
        # Active leases live past max_lifetime; retirement happens at release,
        # never asynchronously in the middle of a synthesis.
        adapter._connection_created_at[replacement] = time.monotonic() - 2.1
        assert not replacement.closed
        adapter._release_connection(replacement)
        assert replacement.closed
        assert adapter._connections == set()

    def test_prewarm_adds_only_missing_physical_connections(self, monkeypatch):
        existing = PoolFakeWebSocketConnection()
        created = []

        def connect(*_args, **_kwargs):
            conn = PoolFakeWebSocketConnection()
            created.append(conn)
            return conn

        def send(conn, payload):
            conn.responses.put(
                {
                    "type": "capabilities",
                    "websocket_connection_reusable": True,
                    "capabilities": {"variant": "standalone"},
                }
            )

        def recv(conn):
            return 0x1, json.dumps(conn.responses.get_nowait()).encode("utf-8")

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", recv)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: conn.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=4,
            max_idle_connections=4,
            keepalive_interval=0,
        )
        adapter._connections.add(existing)
        adapter._connection_created_at[existing] = time.monotonic()
        adapter._idle_connections.append(existing)
        adapter._connection_idle_since[existing] = time.monotonic()

        assert adapter.prewarm(4, timeout=0.5) == 4
        assert len(created) == 3
        assert existing in adapter._idle_connections
        assert len(adapter._connections) == 4

    def test_keepalive_jitter_and_probe_preserve_idle_age(self, monkeypatch):
        conn = FakeRawWebSocketConnection()
        stop = threading.Event()
        delays = []
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            connect_timeout=0.1,
            keepalive_interval=0.01,
            keepalive_jitter=0.2,
            idle_ttl=10.0,
        )
        adapter._connections.add(conn)
        adapter._connection_created_at[conn] = time.monotonic()
        adapter._idle_connections.append(conn)
        original_idle_since = time.monotonic() - 3.0
        adapter._connection_idle_since[conn] = original_idle_since
        monkeypatch.setattr(
            adapter,
            "_request_capabilities",
            lambda *_args, **_kwargs: (Capabilities(), True),
        )
        monkeypatch.setattr(
            _ew.random,
            "uniform",
            lambda lower, upper: delays.append((lower, upper)) or lower,
        )

        def probe_once(_stop, *, probe=True):
            EngineWebSocketAdapter._keepalive_once(adapter, _stop, probe=probe)
            stop.set()

        monkeypatch.setattr(adapter, "_keepalive_once", probe_once)
        EngineWebSocketAdapter._keepalive_worker(
            weakref.ref(adapter), stop, 0.01, 0.2, True
        )

        assert delays[0] == pytest.approx((0.008, 0.012))
        assert adapter._connection_idle_since[conn] == original_idle_since


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
# Active stream resume
# ---------------------------------------------------------------------------


class TestActiveStreamResume:
    @staticmethod
    def _recv(conn: RecoveryFakeWebSocketConnection):
        try:
            item = conn.frames.get(timeout=min(0.02, conn._timeout))
        except queue.Empty as exc:
            raise socket.timeout("not ready") from exc
        if item == "__close__":
            return 0x8, b""
        if isinstance(item, bytes):
            return 0x2, item
        return 0x1, json.dumps(item).encode("utf-8")

    def test_reader_disconnect_resumes_without_duplicate_audio_and_replays_stop(
        self, monkeypatch
    ):
        connections: list[RecoveryFakeWebSocketConnection] = []
        sent: list[tuple[str, dict]] = []
        pool_sizes: list[int] = []
        adapter: EngineWebSocketAdapter

        def connect(_url, *, timeout, headers=None):
            pool_sizes.append(len(adapter._connections) + adapter._connecting)
            conn = RecoveryFakeWebSocketConnection(f"conn-{len(connections) + 1}")
            connections.append(conn)
            return conn

        def send(conn: RecoveryFakeWebSocketConnection, payload):
            sent.append((conn.name, dict(payload)))
            message_type = payload["type"]
            if conn.name == "conn-1" and message_type == "start":
                conn.frames.put(
                    {
                        "type": "event",
                        "delivery_seq": 1,
                        "event": {"type": "start", "session_id": "resume-1"},
                    }
                )
            elif conn.name == "conn-1" and message_type == "text":
                if payload["seq_no"] == 1:
                    conn.frames.put({"type": "text_ack", "through_seq": 1})
            elif conn.name == "conn-1" and message_type == "stop":
                conn.frames.put(
                    {
                        "type": "audio_header",
                        "delivery_seq": 2,
                        "start_sample": 0,
                        "end_sample": 2,
                        "audio": {
                            "encoding": "pcm_f32",
                            "sample_rate": 24000,
                            "channels": 1,
                        },
                    }
                )
                conn.frames.put(b"A" * 8)
                conn.frames.put("__close__")
            elif conn.name == "conn-2" and message_type == "resume":
                conn.frames.put(
                    {
                        "type": "resumed",
                        "acked_text_seq": 1,
                        "input_closed": False,
                    }
                )
            elif conn.name == "conn-2" and message_type == "text":
                conn.frames.put({"type": "text_ack", "through_seq": 2})
            elif conn.name == "conn-2" and message_type == "stop":
                # ACK loss can make the server replay delivery 2. The client
                # consumes its paired binary frame but must not enqueue it.
                conn.frames.put(
                    {
                        "type": "audio_header",
                        "delivery_seq": 2,
                        "start_sample": 0,
                        "end_sample": 2,
                        "audio": {"encoding": "pcm_f32"},
                    }
                )
                conn.frames.put(b"A" * 8)
                conn.frames.put(
                    {
                        "type": "audio_header",
                        "delivery_seq": 3,
                        "start_sample": 2,
                        "end_sample": 4,
                        "audio": {"encoding": "pcm_f32"},
                    }
                )
                conn.frames.put(b"B" * 8)
                conn.frames.put(
                    {
                        "type": "event",
                        "delivery_seq": 4,
                        "event": {
                            "type": "done",
                            "session_id": "resume-1",
                            "meta": {"websocket_connection_reusable": "true"},
                        },
                    }
                )

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", self._recv)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: setattr(conn, "closed", True))
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            connect_timeout=0.2,
            reconnect_attempts=0,
            max_connections=1,
            max_idle_connections=1,
            keepalive_interval=0,
            stream_resume_attempts=2,
            stream_resume_timeout=1.0,
            stream_resume_ack_interval=8,
        )

        session = adapter.open_stream(
            SessionStartRequest(session_id="resume-1", config=SynthesisConfig())
        )
        session.send_text("你好")
        session.send_text("世界")
        session.stop()
        messages = list(session.iter_messages())

        assert [message.type for message in messages if hasattr(message, "type")] == [
            "start",
            "done",
        ]
        audio = [
            message.pcm_bytes for message in messages if hasattr(message, "pcm_bytes")
        ]
        assert audio == [b"A" * 8, b"B" * 8]
        assert len(connections) == 2
        assert max(pool_sizes) <= 1

        start_payload = next(payload for name, payload in sent if name == "conn-1")
        token = start_payload["resume"]["token"]
        assert token
        resume_payload = next(
            payload
            for name, payload in sent
            if name == "conn-2" and payload["type"] == "resume"
        )
        assert resume_payload == {
            "type": "resume",
            "token": token,
            "last_delivery_seq": 2,
            "audio_through_sample": 2,
        }
        replayed_text = [
            payload
            for name, payload in sent
            if name == "conn-2" and payload["type"] == "text"
        ]
        assert [payload["seq_no"] for payload in replayed_text] == [2]
        replayed_stop = next(
            payload
            for name, payload in sent
            if name == "conn-2" and payload["type"] == "stop"
        )
        assert replayed_stop["final_seq_no"] == 2
        assert any(payload["type"] == "terminal_ack" for _name, payload in sent)

    def test_resume_dial_exhaustion_publishes_one_terminal_error_and_releases_slot(
        self, monkeypatch
    ):
        first = RecoveryFakeWebSocketConnection("initial")
        connect_calls = 0

        def connect(_url, *, timeout, headers=None):
            nonlocal connect_calls
            connect_calls += 1
            if connect_calls == 1:
                return first
            raise RawWebSocketError("gateway unavailable")

        def send(conn, payload):
            if payload["type"] == "start":
                conn.frames.put(
                    {
                        "type": "event",
                        "delivery_seq": 1,
                        "event": {"type": "start", "session_id": "resume-fail"},
                    }
                )
            elif payload["type"] == "stop":
                conn.frames.put("__close__")

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", self._recv)
        monkeypatch.setattr(_ew, "ws_close", lambda conn: setattr(conn, "closed", True))
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            connect_timeout=0.05,
            reconnect_attempts=0,
            max_connections=1,
            max_idle_connections=0,
            keepalive_interval=0,
            stream_resume_attempts=2,
            stream_resume_timeout=0.5,
        )
        session = adapter.open_stream(
            SessionStartRequest(session_id="resume-fail", config=SynthesisConfig())
        )
        session.stop()
        messages = list(session.iter_messages())

        assert [message.type for message in messages] == ["start", "error"]
        assert "gateway unavailable" in messages[-1].message
        assert connect_calls == 3  # initial + two atomic resume dial attempts
        assert adapter._connecting == 0
        assert adapter._connections == set()
        assert isinstance(session._recovery_error, StreamRecoveryError)

    def test_failed_terminal_ack_discards_physical_connection(self, monkeypatch):
        conn = RecoveryFakeWebSocketConnection("terminal-ack")

        def send(_conn, payload):
            if payload["type"] == "start":
                conn.frames.put(
                    {
                        "type": "event",
                        "delivery_seq": 1,
                        "event": {"type": "start", "session_id": "terminal"},
                    }
                )
            elif payload["type"] == "stop":
                conn.frames.put(
                    {
                        "type": "event",
                        "delivery_seq": 2,
                        "event": {
                            "type": "done",
                            "session_id": "terminal",
                            "meta": {"websocket_connection_reusable": "true"},
                        },
                    }
                )
            elif payload["type"] == "terminal_ack":
                raise RawWebSocketError("ack write failed")

        monkeypatch.setattr(_ew, "ws_connect", _make_ws_connect(conn))
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", self._recv)
        monkeypatch.setattr(_ew, "ws_close", lambda raw: raw.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            keepalive_interval=0,
        )
        session = adapter.open_stream(
            SessionStartRequest(session_id="terminal", config=SynthesisConfig())
        )
        session.stop()

        assert [message.type for message in session.iter_messages()] == [
            "start",
            "done",
        ]
        assert conn.closed
        assert adapter._idle_connections == []

    def test_close_interrupts_resume_handshake_and_releases_replacement(
        self, monkeypatch
    ):
        connections: list[RecoveryFakeWebSocketConnection] = []
        resume_sent = threading.Event()

        def connect(_url, *, timeout, headers=None):
            conn = RecoveryFakeWebSocketConnection(f"close-{len(connections) + 1}")
            connections.append(conn)
            return conn

        def send(conn, payload):
            if payload["type"] == "start":
                conn.frames.put(
                    {
                        "type": "event",
                        "delivery_seq": 1,
                        "event": {"type": "start", "session_id": "close-resume"},
                    }
                )
            elif conn.name == "close-1" and payload["type"] == "stop":
                conn.frames.put("__close__")
            elif payload["type"] == "resume":
                resume_sent.set()

        def recv(conn):
            if conn.closed:
                raise RawWebSocketError("closed by local hard stop")
            return self._recv(conn)

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", recv)
        monkeypatch.setattr(_ew, "ws_close", lambda raw: raw.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=1,
            max_idle_connections=1,
            keepalive_interval=0,
            stream_resume_attempts=2,
            stream_resume_timeout=2.0,
        )
        session = adapter.open_stream(
            SessionStartRequest(session_id="close-resume", config=SynthesisConfig())
        )
        session.stop()
        assert resume_sent.wait(timeout=1.0)

        started = time.monotonic()
        session.close(reason="caller shutdown")
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        session._reader.join(timeout=1.0)
        assert not session._reader.is_alive()
        assert len(connections) == 2
        assert connections[1].closed
        assert adapter._connecting == 0
        assert adapter._connections == set()

    def test_terminal_is_not_visible_until_connection_is_reusable(self, monkeypatch):
        connections: list[RecoveryFakeWebSocketConnection] = []

        def connect(_url, *, timeout, headers=None):
            conn = RecoveryFakeWebSocketConnection(f"serial-{len(connections) + 1}")
            connections.append(conn)
            return conn

        def send(conn, payload):
            if payload["type"] == "start":
                conn.frames.put(
                    {
                        "type": "event",
                        "delivery_seq": 1,
                        "event": {
                            "type": "start",
                            "session_id": payload["session_id"],
                        },
                    }
                )
            elif payload["type"] == "stop":
                conn.frames.put(
                    {
                        "type": "event",
                        "delivery_seq": 2,
                        "event": {
                            "type": "done",
                            "session_id": "serial",
                            "meta": {"websocket_connection_reusable": "true"},
                        },
                    }
                )

        monkeypatch.setattr(_ew, "ws_connect", connect)
        monkeypatch.setattr(_ew, "ws_send_json", send)
        monkeypatch.setattr(_ew, "ws_recv_frame", self._recv)
        monkeypatch.setattr(_ew, "ws_close", lambda raw: raw.close())
        adapter = EngineWebSocketAdapter(
            "ws://localhost:50052/v1/ws",
            timeout=1.0,
            reconnect_attempts=0,
            max_connections=2,
            max_idle_connections=2,
        )
        first = adapter.open_stream(
            SessionStartRequest(session_id="serial-1", config=SynthesisConfig())
        )
        terminal_published = threading.Event()
        unblock_publish = threading.Event()
        original_put = first._put_message

        def blocking_put(message):
            original_put(message)
            if getattr(message, "type", "") == "done":
                terminal_published.set()
                unblock_publish.wait(timeout=1.0)

        first._put_message = blocking_put
        first.stop()
        assert terminal_published.wait(timeout=1.0)

        # The first reader is deliberately paused inside terminal publication.
        # Its socket must nevertheless already be back in the idle pool.
        second = adapter.open_stream(
            SessionStartRequest(session_id="serial-2", config=SynthesisConfig())
        )
        assert len(connections) == 1

        unblock_publish.set()
        second.stop()
        assert [message.type for message in second.iter_messages()] == [
            "start",
            "done",
        ]
        assert [message.type for message in first.iter_messages()] == [
            "start",
            "done",
        ]


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
