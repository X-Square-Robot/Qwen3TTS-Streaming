from __future__ import annotations

import json

from qwen3tts_protocol import (
    BytesResult,
    Capabilities,
    SessionStartRequest,
    SynthesisConfig,
)
from qwen3tts._adapters.engine_websocket import (
    EngineWebSocketAdapter,
)
from qwen3tts.constants import TRANSPORT_ENGINE_WEBSOCKET


class FakeRawWebSocketConnection:
    """Simulates a raw websocket connection."""

    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False
        self.buffer = bytearray()
        self.sock = self
        self._timeout = 5.0

    def settimeout(self, value):
        self._timeout = value

    def sendall(self, data: bytes):
        pass

    def close(self):
        self.closed = True


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
        assert closed[0]

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
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_send_frame",
            lambda conn, *, opcode, payload: None,
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
            _make_ws_recv_frame(
                [
                    {"type": "event", "event": {"type": "done", "session_id": "s3"}},
                ]
            ),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_close",
            _make_ws_close(closed),
        )
        monkeypatch.setattr(
            "qwen3tts._adapters.engine_websocket.ws_send_frame",
            lambda conn, *, opcode, payload: None,
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
