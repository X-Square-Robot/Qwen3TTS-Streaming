"""Minimal RFC6455 client helpers for the lightweight SDK."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse


class RawWebSocketError(RuntimeError):
    """Expected websocket transport error."""


@dataclass
class RawWebSocketConnection:
    sock: socket.socket
    buffer: bytearray

    def recv_exact(self, n: int) -> bytes:
        while len(self.buffer) < n:
            chunk = self.sock.recv(max(4096, n - len(self.buffer)))
            if not chunk:
                raise RawWebSocketError("websocket closed before enough data was received")
            self.buffer.extend(chunk)
        data = bytes(self.buffer[:n])
        del self.buffer[:n]
        return data


def ws_connect(
    url: str,
    *,
    timeout: float,
    headers: Mapping[str, str] | None = None,
) -> RawWebSocketConnection:
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"}:
        raise RawWebSocketError(f"unsupported websocket scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise RawWebSocketError(f"invalid websocket URL: {url!r}")
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    raw_sock = socket.create_connection((host, port), timeout=timeout)
    raw_sock.settimeout(timeout)
    if parsed.scheme == "wss":
        context = ssl.create_default_context()
        sock = context.wrap_socket(raw_sock, server_hostname=host)
    else:
        sock = raw_sock

    ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
    host_header = host if parsed.port is None else f"{host}:{port}"
    extra_headers = ""
    for key, value in dict(headers or {}).items():
        extra_headers += f"{key}: {value}\r\n"
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host_header}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"{extra_headers}"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))

    response = bytearray()
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise RawWebSocketError("websocket handshake failed: connection closed")
        response.extend(chunk)

    header_bytes, _, leftover = bytes(response).partition(b"\r\n\r\n")
    header_lines = header_bytes.decode("latin1").split("\r\n")
    if not header_lines or "101" not in header_lines[0]:
        raise RawWebSocketError(
            f"websocket handshake failed: {header_lines[0] if header_lines else '<empty>'}"
        )

    headers_map: dict[str, str] = {}
    for line in header_lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers_map[key.strip().lower()] = value.strip()

    accept = headers_map.get("sec-websocket-accept", "")
    expected = base64.b64encode(
        hashlib.sha1((ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
    ).decode("ascii")
    if accept != expected:
        raise RawWebSocketError("websocket handshake failed: invalid Sec-WebSocket-Accept")

    return RawWebSocketConnection(sock=sock, buffer=bytearray(leftover))


def ws_send_json(conn: RawWebSocketConnection, payload: dict[str, Any]) -> None:
    ws_send_frame(conn, opcode=0x1, payload=json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def ws_send_frame(conn: RawWebSocketConnection, *, opcode: int, payload: bytes) -> None:
    first = 0x80 | (opcode & 0x0F)
    length = len(payload)
    if length < 126:
        header = bytes([first, 0x80 | length])
    elif length < (1 << 16):
        header = bytes([first, 0x80 | 126]) + struct.pack("!H", length)
    else:
        header = bytes([first, 0x80 | 127]) + struct.pack("!Q", length)

    mask = os.urandom(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    conn.sock.sendall(header + mask + masked)


def ws_recv_frame(conn: RawWebSocketConnection) -> tuple[int, bytes]:
    header = conn.recv_exact(2)
    first, second = header[0], header[1]
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F

    if length == 126:
        length = struct.unpack("!H", conn.recv_exact(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", conn.recv_exact(8))[0]

    mask = conn.recv_exact(4) if masked else b""
    payload = conn.recv_exact(length)
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def ws_close(conn: RawWebSocketConnection) -> None:
    try:
        ws_send_frame(conn, opcode=0x8, payload=b"")
    except Exception:
        pass
    try:
        conn.sock.close()
    except Exception:
        pass
