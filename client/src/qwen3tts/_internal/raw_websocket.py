"""Websocket transport for the SDK, backed by ``websocket-client``.

This module exposes the minimal surface the SDK is built on
(``ws_connect`` / ``ws_send_json`` / ``ws_recv_frame`` / ``ws_close``).
``ws_close`` is also imported by external callers (e.g. speechserver's
force-close path), so its signature and "callable from any thread,
unblocks a concurrent ``ws_recv_frame`` immediately" contract must hold.

Handshake, RFC6455 framing, fragmentation reassembly and ping/pong are
delegated to ``websocket-client``, whose frame buffer keeps partial-frame
state across socket timeouts — callers may poll ``ws_recv_frame`` with a
short timeout and retry on ``socket.timeout`` without desynchronizing.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Mapping

import websocket
from websocket import (
    ABNF,
    WebSocketConnectionClosedException,
    WebSocketException,
    WebSocketTimeoutException,
)

from .tls import TLSConfig, TLSVerify


class RawWebSocketError(RuntimeError):
    """Expected websocket transport error."""


class RawWebSocketConnection:
    """A connected client websocket."""

    def __init__(self, ws: websocket.WebSocket) -> None:
        self.ws = ws

    def settimeout(self, value: float | None) -> None:
        self.ws.settimeout(value)


def ws_connect(
    url: str,
    *,
    timeout: float,
    headers: Mapping[str, str] | None = None,
    tls_verify: TLSVerify | TLSConfig = True,
) -> RawWebSocketConnection:
    tls = TLSConfig.from_value(tls_verify)
    tls_options = {} if tls.is_default else {"sslopt": tls.websocket_sslopt()}
    try:
        ws = websocket.create_connection(
            url,
            timeout=timeout,
            header=dict(headers or {}),
            # send_text()/end()/cancel() run on caller threads while the
            # session reader thread receives on the same socket.
            enable_multithread=True,
            skip_utf8_validation=True,
            **tls_options,
        )
    except WebSocketTimeoutException as exc:
        raise socket.timeout(str(exc)) from exc
    except (WebSocketException, ValueError) as exc:
        raise RawWebSocketError(f"websocket handshake failed: {exc}") from exc
    return RawWebSocketConnection(ws)


def ws_send_json(conn: RawWebSocketConnection, payload: dict[str, Any]) -> None:
    try:
        conn.ws.send(json.dumps(payload, ensure_ascii=False))
    except WebSocketTimeoutException as exc:
        raise RawWebSocketError(f"websocket send timed out: {exc}") from exc
    except WebSocketConnectionClosedException as exc:
        raise RawWebSocketError("websocket closed") from exc
    except OSError as exc:
        # websocket-client deliberately lets non-timeout socket failures such
        # as ECONNRESET escape unchanged. Normalize them so stream sessions
        # consistently surface StreamClosedError and stop accepting sends.
        raise RawWebSocketError(f"websocket send failed: {exc}") from exc


def ws_recv_frame(conn: RawWebSocketConnection) -> tuple[int, bytes]:
    """Receive the next data or close frame as ``(opcode, payload)``.

    Ping/pong frames are answered inside the library and never surfaced;
    fragmented messages arrive reassembled, so ``opcode`` is one of
    0x1 (text), 0x2 (binary) or 0x8 (close).  ``socket.timeout`` is safe
    to retry.

    Two mapping caveats: a close frame carrying a non-RFC6455 status code
    fails the library's frame validation and surfaces as
    ``RawWebSocketError`` instead of ``(0x8, b"")``, and raw ``OSError``
    from the socket layer (e.g. ECONNRESET) propagates unmapped.
    """
    try:
        opcode, data = conn.ws.recv_data(control_frame=False)
    except WebSocketTimeoutException as exc:
        raise socket.timeout(str(exc)) from exc
    except WebSocketConnectionClosedException as exc:
        raise RawWebSocketError(
            "websocket closed before enough data was received"
        ) from exc
    except WebSocketException as exc:
        raise RawWebSocketError(f"websocket protocol error: {exc}") from exc
    if opcode == ABNF.OPCODE_CLOSE:
        return 0x8, b""
    return int(opcode), bytes(data)


def ws_close(conn: RawWebSocketConnection) -> None:
    try:
        # timeout=0 sends the close frame and shuts the socket down without
        # waiting for the peer's ack; shutdown(SHUT_RDWR) unblocks a reader
        # thread stuck in recv.
        conn.ws.close(timeout=0)
    except Exception:
        pass
    try:
        conn.ws.shutdown()
    except Exception:
        pass
