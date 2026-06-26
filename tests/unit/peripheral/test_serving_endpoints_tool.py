from __future__ import annotations

import socket
import time

import pytest

import sys
from pathlib import Path

# Bootstrap tools/validation/ onto sys.path so we can import serving_endpoints
_tools_dir = Path(__file__).resolve().parents[3] / "tools" / "validation"
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

import serving_endpoints


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _FakeConnection:
    def __init__(self) -> None:
        self.sock = _FakeSocket()


def test_websocket_oneshot_waits_through_short_socket_idle(monkeypatch):
    frames = [
        socket.timeout(),
        (
            0x1,
            b'{"type":"event","event":{"type":"done","meta":{}}}',
        ),
    ]

    def fake_recv_frame(_conn):
        item = frames.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(serving_endpoints, "ws_recv_frame", fake_recv_frame)

    transport = serving_endpoints.EngineWebSocketTransport("ws://example.test/v1/ws")
    result = serving_endpoints.SynthesisResult("engine-websocket", "sid", "text")

    terminal = transport._read_until_timeout(
        _FakeConnection(),
        result,
        [],
        [],
        {"first_ts": None, "encoding": "pcm_f32"},
        deadline=time.perf_counter() + 1.0,
        stop_on_idle=False,
    )

    assert terminal is True
    assert result.error is None
    assert result.events == ["done"]


def test_websocket_idle_drain_still_returns_on_socket_idle(monkeypatch):
    def fake_recv_frame(_conn):
        raise socket.timeout()

    monkeypatch.setattr(serving_endpoints, "ws_recv_frame", fake_recv_frame)

    transport = serving_endpoints.EngineWebSocketTransport("ws://example.test/v1/ws")
    result = serving_endpoints.SynthesisResult("engine-websocket", "sid", "text")

    terminal = transport._read_until_timeout(
        _FakeConnection(),
        result,
        [],
        [],
        {"first_ts": None, "encoding": "pcm_f32"},
        deadline=time.perf_counter() + 1.0,
        stop_on_idle=True,
    )

    assert terminal is False
    assert result.error is None
    assert result.events == []


def test_websocket_oneshot_replies_to_ping_and_keeps_waiting(monkeypatch):
    frames = [
        (0x9, b"heartbeat"),
        (
            0x1,
            b'{"type":"event","event":{"type":"done","meta":{}}}',
        ),
    ]
    sent_frames = []

    def fake_recv_frame(_conn):
        return frames.pop(0)

    def fake_send_frame(_conn, *, opcode, payload):
        sent_frames.append((opcode, payload))

    monkeypatch.setattr(serving_endpoints, "ws_recv_frame", fake_recv_frame)
    monkeypatch.setattr(serving_endpoints, "ws_send_frame", fake_send_frame)

    transport = serving_endpoints.EngineWebSocketTransport("ws://example.test/v1/ws")
    result = serving_endpoints.SynthesisResult("engine-websocket", "sid", "text")

    terminal = transport._read_until_timeout(
        _FakeConnection(),
        result,
        [],
        [],
        {"first_ts": None, "encoding": "pcm_f32"},
        deadline=time.perf_counter() + 1.0,
        stop_on_idle=False,
    )

    assert terminal is True
    assert sent_frames == [(0xA, b"heartbeat")]
    assert result.error is None
    assert result.events == ["done"]


def test_decode_audio_bytes_rejects_partial_float32_frame():
    with pytest.raises(ValueError, match="not a multiple of 4"):
        serving_endpoints._decode_audio_bytes(b"\x00\x00\x00", "pcm_f32")


def test_websocket_text_opcode_can_recover_audio_payload():
    transport = serving_endpoints.EngineWebSocketTransport("ws://example.test/v1/ws")
    result = serving_endpoints.SynthesisResult("engine-websocket", "sid", "text")
    chunks = []
    timestamps = []

    terminal = transport._consume_frame(
        result,
        chunks,
        timestamps,
        {"first_ts": None, "encoding": "pcm_f32"},
        0x1,
        b"\x00\x00\x00\x00",
    )

    assert terminal is False
    assert len(chunks) == 1
    assert chunks[0].tolist() == [0.0]
    assert result.warnings == [
        "received websocket audio payload in a text frame; decoded as audio"
    ]


def test_websocket_binary_opcode_can_recover_event_payload():
    transport = serving_endpoints.EngineWebSocketTransport("ws://example.test/v1/ws")
    result = serving_endpoints.SynthesisResult("engine-websocket", "sid", "text")

    terminal = transport._consume_frame(
        result,
        [],
        [],
        {"first_ts": None, "encoding": "pcm_f32"},
        0x2,
        b'{"type":"event","event":{"type":"done","meta":{}}}',
    )

    assert terminal is True
    assert result.events == ["done"]
    assert result.warnings == [
        "received websocket event payload in a binary frame; decoded as event"
    ]
