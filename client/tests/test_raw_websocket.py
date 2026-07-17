"""Regression tests for buffered RFC6455 frame parsing.

Prior to the buffered rewrite, ``ws_recv_frame`` consumed bytes destructively
across multiple ``recv`` calls: a ``socket.timeout`` between the frame header
and its payload dropped the already-consumed header, so the retry re-parsed
from the middle of the frame and the stream desynchronized permanently.  In
production the desync made PCM high bytes (0x7F/0xFF) parse as length=127,
whose 8-byte extended length overflowed ``sock.recv`` with ``OverflowError``.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time

import pytest

from qwen3tts_protocol import AudioChunk
from qwen3tts._adapters.engine_websocket import _iter_conn_messages
from qwen3tts._internal.raw_websocket import (
    RawWebSocketConnection,
    RawWebSocketError,
    ws_recv_frame,
)


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


def _drip_send(sock: socket.socket, chunks: list[bytes], gap: float) -> None:
    for chunk in chunks:
        sock.sendall(chunk)
        time.sleep(gap)


class TestBufferedFrameParsing:
    def test_frame_split_across_poll_timeouts_does_not_desync(self):
        """A frame dripped slower than the poll timeout parses intact."""
        client_sock, server_sock = socket.socketpair()
        try:
            client_sock.settimeout(0.05)
            conn = RawWebSocketConnection(sock=client_sock, buffer=bytearray())

            # PCM-like payload of high bytes; > 65535 forces the 8-byte
            # extended-length path that overflowed in production.
            pcm = bytes([0xFF, 0x7F]) * 35_000
            frame1 = _server_frame(0x2, pcm)
            frame2 = _server_frame(0x1, b'{"type":"event"}')
            server = threading.Thread(
                target=_drip_send,
                args=(
                    server_sock,
                    # Split points: after the 2-byte header, inside the
                    # 8-byte extended length, and inside the payload.
                    [frame1[:2], frame1[2:7], frame1[7:40_000], frame1[40_000:], frame2],
                    0.12,
                ),
                daemon=True,
            )
            server.start()

            frames: list[tuple[int, bytes]] = []
            timeouts = 0
            deadline = time.perf_counter() + 10.0
            while len(frames) < 2:
                assert time.perf_counter() < deadline, "test frames never arrived"
                try:
                    frames.append(ws_recv_frame(conn))
                except socket.timeout:
                    timeouts += 1

            server.join(timeout=5.0)
            assert timeouts >= 2, "expected the drip-feed to interrupt mid-frame"
            assert frames[0] == (0x2, pcm)
            assert frames[1] == (0x1, b'{"type":"event"}')
        finally:
            client_sock.close()
            server_sock.close()

    def test_masked_frame_parses_from_buffer(self):
        mask = bytes([0x01, 0x02, 0x03, 0x04])
        payload = b"masked-payload"
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        raw = bytes([0x81, 0x80 | len(payload)]) + mask + masked
        conn = RawWebSocketConnection(sock=None, buffer=bytearray(raw))

        assert ws_recv_frame(conn) == (0x1, payload)
        assert not conn.buffer

    def test_desynced_pcm_length_raises_protocol_error(self):
        """PCM bytes misread as a frame header must not escape as OverflowError."""
        # Second byte 0xFF => masked + length=127; the next 8 PCM bytes decode
        # to a length >= 2**63, which used to reach sock.recv and raise
        # OverflowError.
        raw = bytes([0x82, 0xFF]) + bytes([0xF3, 0x22, 0xEE, 0x10, 0x9C, 0x55, 0x7F, 0x01])
        conn = RawWebSocketConnection(sock=None, buffer=bytearray(raw))

        with pytest.raises(RawWebSocketError, match="desynchronized"):
            ws_recv_frame(conn)


class TestIterConnMessagesSlowServer:
    def test_audio_intact_across_mid_frame_stalls(self):
        """The adapter poll loop resumes an interrupted frame without data loss."""
        client_sock, server_sock = socket.socketpair()
        try:
            conn = RawWebSocketConnection(sock=client_sock, buffer=bytearray())

            chunk_a = bytes([0xFF, 0x7F]) * 400
            chunk_b = bytes([0x80, 0x00]) * 300
            frame_b = _server_frame(0x2, chunk_b)
            done = json.dumps(
                {"type": "event", "event": {"type": "done", "session_id": "s1"}}
            ).encode("utf-8")
            server = threading.Thread(
                target=_drip_send,
                args=(
                    server_sock,
                    # frame_b is split mid-payload with gaps longer than the
                    # adapter's 0.5s poll timeout; a ping is interleaved.
                    [
                        _server_frame(0x2, chunk_a),
                        frame_b[:100],
                        frame_b[100:],
                        _server_frame(0x9, b"ping"),
                        _server_frame(0x1, done),
                    ],
                    0.55,
                ),
                daemon=True,
            )
            server.start()

            audio = bytearray()
            events = []
            for message in _iter_conn_messages(conn, timeout=10.0):
                if isinstance(message, AudioChunk):
                    audio.extend(message.pcm_bytes)
                else:
                    events.append(message)

            server.join(timeout=5.0)
            assert bytes(audio) == chunk_a + chunk_b
            assert events and events[-1].type == "done"
        finally:
            client_sock.close()
            server_sock.close()
