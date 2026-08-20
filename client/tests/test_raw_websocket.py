"""Transport-level regression tests for the websocket layer.

The scenarios mirror a production incident: a slow/lumpy network stalls
delivery mid-frame while the client polls with a short socket timeout.
The transport must resume the interrupted frame instead of
desynchronizing (the old hand-rolled parser lost frame state on
``socket.timeout`` and eventually crashed with ``OverflowError``).

Each test runs end-to-end against a scripted RFC6455 server on a real
localhost socket, exercising the actual ``websocket-client`` stack.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import socket
import ssl
import struct
import threading
import time

import pytest

from qwen3tts_protocol import AudioChunk
from qwen3tts._adapters.engine_websocket import _iter_conn_messages
from qwen3tts._internal.raw_websocket import (
    RawWebSocketConnection,
    RawWebSocketError,
    ws_connect,
    ws_recv_frame,
    ws_close,
    ws_send_json,
)

_WS_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def test_send_socket_error_is_normalized_to_transport_error():
    class _ResetSocket:
        def send(self, _payload):
            raise ConnectionResetError("peer reset")

    conn = RawWebSocketConnection(_ResetSocket())

    with pytest.raises(RawWebSocketError, match="send failed"):
        ws_send_json(conn, {"type": "text", "text": "hello"})


@pytest.mark.parametrize(
    ("tls_verify", "expected"),
    [
        (
            False,
            {"cert_reqs": ssl.CERT_NONE, "check_hostname": False},
        ),
    ],
)
def test_ws_connect_supports_local_tls_without_verification(
    monkeypatch, tls_verify, expected
):
    captured = {}
    fake_socket = object()

    def create_connection(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return fake_socket

    monkeypatch.setattr(
        "qwen3tts._internal.raw_websocket.websocket.create_connection",
        create_connection,
    )

    conn = ws_connect(
        "wss://localhost:50052/v1/realtime",
        timeout=2.0,
        tls_verify=tls_verify,
    )

    assert conn.ws is fake_socket
    assert captured["sslopt"] == expected


def test_ws_connect_uses_explicit_ca_bundle(monkeypatch, tmp_path):
    ca_file = tmp_path / "cert.local.pem"
    ca_file.write_text("test certificate", encoding="utf-8")
    captured = {}

    def create_connection(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return object()

    monkeypatch.setattr(
        "qwen3tts._internal.raw_websocket.websocket.create_connection",
        create_connection,
    )

    ws_connect(
        "wss://localhost:50052/v1/realtime",
        timeout=2.0,
        tls_verify=ca_file,
    )

    assert captured["sslopt"] == {
        "cert_reqs": ssl.CERT_REQUIRED,
        "check_hostname": True,
        "ca_certs": str(ca_file.resolve()),
    }


def _server_frame(opcode: int, payload: bytes) -> bytes:
    """Build an unmasked server->client frame."""
    first = 0x80 | (opcode & 0x0F)
    length = len(payload)
    if length < 126:
        header = bytes([first, length])
    elif length < (1 << 16):
        header = bytes([first, 126]) + struct.pack("!H", length)
    else:
        header = bytes([first, 127]) + struct.pack("!Q", length)
    return header + payload


def _read_client_frame(conn: socket.socket) -> tuple[int, bytes]:
    """Parse one masked client->server frame (test helper, small frames only)."""

    def read_exact(n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                raise ConnectionError("client closed")
            data += chunk
        return data

    first, second = read_exact(2)
    opcode = first & 0x0F
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", read_exact(2))[0]
    mask = read_exact(4)
    payload = read_exact(length)
    return opcode, bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


class _ScriptedServer:
    """Accepts one websocket connection and plays back a frame script."""

    def __init__(self, script: list[bytes], gap: float = 0.0) -> None:
        self._script = script
        self._gap = gap
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port = self._listener.getsockname()[1]
        self.url = f"ws://127.0.0.1:{self.port}/v1/ws"
        self.client_frames: list[tuple[int, bytes]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        conn, _ = self._listener.accept()
        try:
            conn.settimeout(5.0)
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = conn.recv(4096)
                if not chunk:
                    raise ConnectionError("client closed during handshake")
                request += chunk
            key = re.search(rb"Sec-WebSocket-Key: *(.+?)\r\n", request).group(1)
            accept = base64.b64encode(hashlib.sha1(key + _WS_GUID).digest())
            conn.sendall(
                b"HTTP/1.1 101 Switching Protocols\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n"
                b"Sec-WebSocket-Accept: " + accept + b"\r\n\r\n"
            )
            for chunk in self._script:
                conn.sendall(chunk)
                time.sleep(self._gap)
            # Keep collecting client frames (pong, close, ...) until the
            # client goes away, so tests can assert on them.
            conn.settimeout(5.0)
            while True:
                self.client_frames.append(_read_client_frame(conn))
        except Exception:
            pass
        finally:
            conn.close()

    def join(self) -> None:
        self._thread.join(timeout=10.0)
        self._listener.close()


class TestFrameResumeAcrossTimeouts:
    def test_frame_split_across_poll_timeouts_does_not_desync(self):
        """A frame dripped slower than the poll timeout parses intact."""
        # PCM-like payload of high bytes; > 65535 forces the 8-byte
        # extended-length path that desynchronized in production.
        pcm = bytes([0xFF, 0x7F]) * 35_000
        frame1 = _server_frame(0x2, pcm)
        frame2 = _server_frame(0x1, b'{"type":"event"}')
        server = _ScriptedServer(
            # Split points: after the 2-byte header, inside the 8-byte
            # extended length, and inside the payload.
            [frame1[:2], frame1[2:7], frame1[7:40_000], frame1[40_000:], frame2],
            gap=0.12,
        )
        conn = ws_connect(server.url, timeout=5.0)
        try:
            conn.settimeout(0.05)
            frames: list[tuple[int, bytes]] = []
            timeouts = 0
            deadline = time.perf_counter() + 10.0
            while len(frames) < 2:
                assert time.perf_counter() < deadline, "test frames never arrived"
                try:
                    frames.append(ws_recv_frame(conn))
                except socket.timeout:
                    timeouts += 1

            assert timeouts >= 2, "expected the drip-feed to interrupt mid-frame"
            assert frames[0] == (0x2, pcm)
            assert frames[1] == (0x1, b'{"type":"event"}')
        finally:
            ws_close(conn)
            server.join()

    def test_ping_is_answered_and_not_surfaced(self):
        server = _ScriptedServer(
            [
                _server_frame(0x9, b"keepalive"),
                _server_frame(0x1, b'{"type":"event"}'),
            ]
        )
        conn = ws_connect(server.url, timeout=5.0)
        try:
            assert ws_recv_frame(conn) == (0x1, b'{"type":"event"}')
        finally:
            ws_close(conn)
            server.join()
        pongs = [f for f in server.client_frames if f[0] == 0xA]
        assert pongs and pongs[0][1] == b"keepalive"

    def test_server_close_frame_normalizes_to_0x8(self):
        server = _ScriptedServer(
            [
                _server_frame(0x1, b'{"type":"event"}'),
                # Standard status 1000 + reason, as the engine gateway sends.
                _server_frame(0x8, struct.pack("!H", 1000) + b"bye"),
            ]
        )
        conn = ws_connect(server.url, timeout=5.0)
        try:
            assert ws_recv_frame(conn) == (0x1, b'{"type":"event"}')
            assert ws_recv_frame(conn) == (0x8, b"")
        finally:
            ws_close(conn)
            server.join()

    def test_mid_frame_disconnect_raises_transport_error(self):
        frame = _server_frame(0x2, bytes([0xFF, 0x7F]) * 400)
        server = _ScriptedServer([frame[:100]])  # then the server closes TCP
        conn = ws_connect(server.url, timeout=5.0)
        try:
            with pytest.raises((RawWebSocketError, OSError)):
                while True:
                    ws_recv_frame(conn)
        finally:
            ws_close(conn)
            server.join()

    def test_ws_close_unblocks_blocked_reader(self):
        """speechserver's force-close path relies on this contract."""
        server = _ScriptedServer([], gap=0.0)
        conn = ws_connect(server.url, timeout=30.0)
        outcome: list[object] = []

        def reader() -> None:
            try:
                outcome.append(ws_recv_frame(conn))
            except Exception as exc:
                outcome.append(exc)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        time.sleep(0.2)
        ws_close(conn)
        thread.join(timeout=5.0)
        server.join()

        assert not thread.is_alive(), "reader stayed blocked after ws_close"
        assert len(outcome) == 1
        # Either a normalized transport error or the server's close echo.
        assert isinstance(outcome[0], (RawWebSocketError, OSError)) or (
            isinstance(outcome[0], tuple) and outcome[0][0] == 0x8
        )


class TestIterConnMessagesSlowServer:
    def test_audio_intact_across_mid_frame_stalls(self):
        """The adapter poll loop resumes an interrupted frame without data loss."""
        chunk_a = bytes([0xFF, 0x7F]) * 400
        chunk_b = bytes([0x80, 0x00]) * 300
        frame_b = _server_frame(0x2, chunk_b)
        done = json.dumps(
            {"type": "event", "event": {"type": "done", "session_id": "s1"}}
        ).encode("utf-8")
        server = _ScriptedServer(
            # frame_b is split mid-payload with gaps longer than the
            # adapter's poll-timeout cap (min(0.5, ...) in
            # _iter_conn_messages) so the stall interrupts mid-frame;
            # a ping is interleaved.  gap must stay > that cap.
            [
                _server_frame(0x2, chunk_a),
                frame_b[:100],
                frame_b[100:],
                _server_frame(0x9, b"ping"),
                _server_frame(0x1, done),
            ],
            gap=0.55,
        )
        conn = ws_connect(server.url, timeout=10.0)
        try:
            audio = bytearray()
            events = []
            for message in _iter_conn_messages(conn, timeout=10.0):
                if isinstance(message, AudioChunk):
                    audio.extend(message.pcm_bytes)
                else:
                    events.append(message)

            assert bytes(audio) == chunk_a + chunk_b
            assert events and events[-1].type == "done"
        finally:
            ws_close(conn)
            server.join()

    def test_ws_close_unblocks_iter_conn_messages(self):
        """speechserver force-closes the connection from another thread to
        break its reader out of the message iterator mid-session."""
        server = _ScriptedServer([])  # silent server: iterator stays blocked
        conn = ws_connect(server.url, timeout=30.0)
        outcome: list[object] = []

        def reader() -> None:
            try:
                outcome.append(list(_iter_conn_messages(conn, timeout=30.0)))
            except Exception as exc:
                outcome.append(exc)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        time.sleep(0.3)
        started = time.perf_counter()
        ws_close(conn)
        thread.join(timeout=5.0)
        server.join()

        assert not thread.is_alive(), "iterator stayed blocked after ws_close"
        assert time.perf_counter() - started < 3.0
        assert len(outcome) == 1
        assert isinstance(outcome[0], (RawWebSocketError, OSError, list))

    def test_fragmented_text_message_is_reassembled(self):
        """Websocket intermediaries may re-fragment messages."""
        done = json.dumps(
            {"type": "event", "event": {"type": "done", "session_id": "s2"}}
        ).encode("utf-8")
        fragment1 = bytes([0x01, len(done[:10])]) + done[:10]  # FIN=0, text
        fragment2 = bytes([0x80, len(done[10:])]) + done[10:]  # FIN=1, cont
        server = _ScriptedServer([fragment1, fragment2])
        conn = ws_connect(server.url, timeout=5.0)
        try:
            events = list(_iter_conn_messages(conn, timeout=5.0))
            assert events and events[-1].type == "done"
        finally:
            ws_close(conn)
            server.join()
