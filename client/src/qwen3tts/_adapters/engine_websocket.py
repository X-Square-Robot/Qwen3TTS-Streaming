from __future__ import annotations

import json
import socket
import threading
import time

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
    ws_close,
    ws_connect,
    ws_recv_frame,
    ws_send_frame,
    ws_send_json,
)
from .._internal.utils import (
    build_bytes_result,
    capabilities_from_payload,
    decode_stream_event,
    stream_text_chunk_to_mapping,
    synthesis_config_to_mapping,
)
from .._session import BaseStreamSession
from ..constants import TRANSPORT_ENGINE_WEBSOCKET
from ..exceptions import ProtocolError


class EngineWebSocketAdapter:
    transport_name = TRANSPORT_ENGINE_WEBSOCKET

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.headers = dict(headers or {})

    def get_capabilities(self) -> Capabilities:
        conn = ws_connect(self.endpoint, timeout=self.timeout, headers=self.headers)
        try:
            ws_send_json(conn, {"type": "get_capabilities"})
            deadline = time.perf_counter() + self.timeout
            while time.perf_counter() < deadline:
                conn.sock.settimeout(
                    max(0.05, min(0.2, deadline - time.perf_counter()))
                )
                opcode, payload = ws_recv_frame(conn)
                if opcode == 0x9:
                    ws_send_frame(conn, opcode=0xA, payload=payload)
                    continue
                if opcode != 0x1:
                    continue
                message = json.loads(payload.decode("utf-8"))
                if message.get("type") != "capabilities":
                    raise ProtocolError(
                        f"unexpected websocket response type: {message.get('type')!r}"
                    )
                return capabilities_from_payload(message.get("capabilities", {}))
            raise TimeoutError("websocket capabilities request timed out")
        finally:
            ws_close(conn)

    def synthesize_bytes(self, text: str, *, request) -> BytesResult:
        session_id = request.session_id or ""
        conn = ws_connect(self.endpoint, timeout=self.timeout, headers=self.headers)
        events: list[StreamEvent] = []
        warnings: list[str] = []
        audio_format = request.config.audio
        audio_parts: list[bytes] = []
        try:
            ws_send_json(
                conn,
                {
                    "type": "oneshot",
                    "session_id": session_id,
                    "text": text,
                    "config": synthesis_config_to_mapping(request.config),
                },
            )
            for message in _iter_conn_messages(conn, timeout=self.timeout):
                if isinstance(message, AudioChunk):
                    audio_parts.append(message.pcm_bytes)
                    audio_format = message.audio
                    continue
                events.append(message)
                if message.type == "warning" and message.message:
                    warnings.append(message.message)
                if message.type in {"done", "error"}:
                    break
        finally:
            ws_close(conn)
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
        return EngineWebSocketStreamSession(
            adapter=self,
            start_request=start_request,
        )


def _iter_conn_messages(
    conn: RawWebSocketConnection,
    *,
    timeout: float,
):
    deadline = time.perf_counter() + timeout
    current_audio = AudioFormat()
    while time.perf_counter() < deadline:
        conn.sock.settimeout(max(0.02, min(0.5, deadline - time.perf_counter())))
        try:
            opcode, payload = ws_recv_frame(conn)
        except socket.timeout:
            continue
        if opcode == 0x9:
            ws_send_frame(conn, opcode=0xA, payload=payload)
            continue
        if opcode == 0xA:
            continue
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
    raise TimeoutError("websocket stream timed out waiting for terminal event")


class EngineWebSocketStreamSession(BaseStreamSession):
    def __init__(
        self, *, adapter: EngineWebSocketAdapter, start_request: SessionStartRequest
    ) -> None:
        super().__init__(
            session_id=start_request.session_id, transport=adapter.transport_name
        )
        self._adapter = adapter
        self._start_request = start_request
        self._conn = ws_connect(
            adapter.endpoint, timeout=adapter.timeout, headers=adapter.headers
        )
        ws_send_json(
            self._conn,
            {
                "type": "start",
                "session_id": start_request.session_id,
                "config": synthesis_config_to_mapping(start_request.config),
            },
        )
        self._reader = threading.Thread(
            target=self._reader_loop, name=f"ws-session-{self.session_id}", daemon=True
        )
        self._reader.start()

    def _reader_loop(self) -> None:
        try:
            for message in _iter_conn_messages(
                self._conn, timeout=self._adapter.timeout
            ):
                self._put_message(message)
        except Exception as exc:
            self._put_message(
                StreamEvent(
                    type="error",
                    session_id=self.session_id,
                    message=str(exc),
                )
            )
        finally:
            ws_close(self._conn)

    def send_text(
        self,
        text: str,
        *,
        seq_no: int | None = None,
        client_timestamp_ms: int | None = None,
    ) -> None:
        self._check_send_open()
        chunk = StreamTextChunk(
            text=text,
            seq_no=int(seq_no or 0),
            client_timestamp_ms=int(client_timestamp_ms or 0),
        )
        payload = {"type": "text"}
        payload.update(stream_text_chunk_to_mapping(chunk))
        ws_send_json(self._conn, payload)

    def end(self, *, client_timestamp_ms: int | None = None) -> None:
        self._check_send_open()
        self._mark_send_closed()
        payload = {"type": "end"}
        if client_timestamp_ms is not None:
            payload["client_timestamp_ms"] = int(client_timestamp_ms)
        ws_send_json(self._conn, payload)

    def cancel(self, reason: str = "") -> None:
        if self._send_closed:
            return
        self._mark_send_closed()
        request = StreamCancelRequest(reason=reason)
        ws_send_json(self._conn, {"type": "cancel", "reason": request.reason})
