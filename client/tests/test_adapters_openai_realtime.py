from __future__ import annotations

import base64
import json
import queue
import socket
from pathlib import Path

import pytest
import qwen3tts._adapters.openai_realtime as realtime_module
from qwen3tts._adapters.openai_realtime import OpenAIRealtimeAdapter
from qwen3tts.constants import TRANSPORT_OPENAI_REALTIME
from qwen3tts.exceptions import ProtocolError
from qwen3tts_protocol import (
    AudioChunk,
    AudioFormat,
    SessionStartRequest,
    SynthesisConfig,
    TimingContext,
)


class FakeRealtimeConnection:
    def __init__(self, *, advertise_text_buffer: bool = True) -> None:
        self.frames: queue.Queue[dict | None] = queue.Queue()
        self.frames.put(
            {
                "type": "session.created",
                "session": {"id": "sess_server", "type": "realtime"},
            }
        )
        self.sent: list[dict] = []
        self.timeout = 0.1
        self.closed = False
        self.advertise_text_buffer = advertise_text_buffer

    def settimeout(self, timeout):
        self.timeout = timeout

    def on_send(self, payload: dict) -> None:
        self.sent.append(payload)
        event_type = payload["type"]
        if event_type == "session.update":
            qwen = dict(payload["session"]["qwen"])
            if self.advertise_text_buffer:
                qwen["text_buffer_extension"] = "qwen.input_text_buffer.v1"
            self.frames.put(
                {
                    "type": "session.updated",
                    "session": {
                        **payload["session"],
                        "id": "sess_server",
                        "qwen": qwen,
                    },
                }
            )
        elif event_type == "response.create":
            self.frames.put(
                {
                    "type": "response.created",
                    "response": {"id": "resp_server", "status": "in_progress"},
                }
            )
            if any(event["type"] == "conversation.item.create" for event in self.sent):
                self._emit_audio_and_done(status="completed")
        elif event_type == "qwen.input_text_buffer.append":
            self.frames.put(
                {
                    "type": "qwen.input_text_buffer.ack",
                    "sequence": payload["sequence"],
                    "duplicate": False,
                }
            )
            self.frames.put(
                {
                    "type": "response.output_audio.delta",
                    "response_id": "resp_server",
                    "delta": base64.b64encode(b"\x01\x00" * 1200).decode("ascii"),
                }
            )
        elif event_type == "qwen.input_text_buffer.commit":
            self._emit_done(status="completed")
        elif event_type == "response.cancel":
            self._emit_done(status="cancelled")

    def _emit_audio_and_done(self, *, status: str) -> None:
        self.frames.put(
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_server",
                "delta": base64.b64encode(b"\x01\x00" * 1200).decode("ascii"),
            }
        )
        self._emit_done(status=status)

    def _emit_done(self, *, status: str) -> None:
        self.frames.put(
            {
                "type": "response.done",
                "response": {
                    "id": "resp_server",
                    "status": status,
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                        "output_token_details": {"audio_tokens": 1},
                    },
                },
            }
        )


def _patch_transport(monkeypatch, connection: FakeRealtimeConnection) -> None:
    monkeypatch.setattr(
        realtime_module,
        "ws_connect",
        lambda *_args, **_kwargs: connection,
    )
    monkeypatch.setattr(
        realtime_module,
        "ws_send_json",
        lambda conn, payload: conn.on_send(payload),
    )

    def recv(conn):
        try:
            event = conn.frames.get(timeout=max(0.001, conn.timeout))
        except queue.Empty as exc:
            raise socket.timeout("no frame") from exc
        if event is None:
            return 0x8, b""
        return 0x1, json.dumps(event).encode("utf-8")

    monkeypatch.setattr(realtime_module, "ws_recv_frame", recv)
    monkeypatch.setattr(
        realtime_module,
        "ws_close",
        lambda conn: setattr(conn, "closed", True),
    )


def _request(session_id="client-session") -> SessionStartRequest:
    return SessionStartRequest(
        session_id=session_id,
        config=SynthesisConfig(
            task_type="custom_voice",
            language="zh",
            speaker="Vivian",
            instruct="calm",
            audio=AudioFormat(encoding="pcm_f32", sample_rate=24000, channels=1),
        ),
    )


def test_oneshot_uses_standard_realtime_events_and_returns_usage(monkeypatch):
    connection = FakeRealtimeConnection(advertise_text_buffer=False)
    _patch_transport(monkeypatch, connection)
    adapter = OpenAIRealtimeAdapter(
        "ws://example.test/v1/realtime",
        model_name="qwen3-tts-realtime",
        timeout=1.0,
        reconnect_attempts=0,
    )

    result = adapter.synthesize_bytes("你好", request=_request())

    assert result.transport == TRANSPORT_OPENAI_REALTIME
    assert result.audio_bytes == b"\x01\x00" * 1200
    assert result.audio_format.encoding == "pcm_s16le"
    assert result.details["usage"]["total_tokens"] == 3
    assert result.details["response_id"] == "resp_server"
    assert [event["type"] for event in connection.sent] == [
        "session.update",
        "conversation.item.create",
        "response.create",
    ]
    item = connection.sent[1]["item"]
    assert item["content"] == [{"type": "input_text", "text": "你好"}]
    assert connection.closed is True


def test_python_sdk_matches_browser_golden_session_core():
    golden = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "protocol/contracts/golden/realtime-session-core.json"
        ).read_text()
    )
    request = SessionStartRequest(
        session_id="golden",
        config=SynthesisConfig(
            task_type="custom_voice",
            speaker="Serena",
            input_mode="full_text",
            audio=AudioFormat(encoding="pcm_s16le", sample_rate=24000, channels=1),
            timing_context=TimingContext(
                request_id="golden-request", client_request_ts_ms=123456
            ),
        ),
    )
    session = realtime_module._session_update(request, "custom-1.7b")["session"]
    assert {
        "model": session["model"],
        "task_type": session["qwen"]["task_type"],
        "speaker": session["audio"]["output"]["voice"],
        "input_mode": session["qwen"]["input_mode"],
        "sample_rate": session["audio"]["output"]["format"]["rate"],
        "audio_format": session["audio"]["output"]["format"]["type"],
        "vad_enabled": session["qwen"]["output_policy"]["vad_policy"]["enabled"],
        "vad_strategy": session["qwen"]["output_policy"]["vad_policy"]["strategy"],
        "delivery": session["qwen"]["output_policy"]["config"].get(
            "delivery", "guarded"
        ),
        "request_id": session["qwen"]["timing"]["request_id"],
        "client_request_ts_ms": session["qwen"]["timing"][
            "client_request_ts_ms"
        ],
    } == golden


def test_python_realtime_session_update_validates_against_shared_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "protocol/contracts/realtime-business.schema.json"
        ).read_text()
    )
    request = SessionStartRequest(
        session_id="schema",
        config=SynthesisConfig(
            task_type="custom_voice",
            input_mode="full_text",
            audio=AudioFormat(encoding="pcm_s16le", sample_rate=24000, channels=1),
        ),
    )
    jsonschema.validate(realtime_module._session_update(request, "custom-1.7b"), schema)
    jsonschema.validate(
        {"type": "qwen.input_text_buffer.append", "sequence": 1, "text": "你好"},
        schema,
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"type": "qwen.input_text_buffer.append", "sequence": 0, "text": ""},
            schema,
        )


def test_streaming_requires_advertised_text_buffer_extension(monkeypatch):
    connection = FakeRealtimeConnection(advertise_text_buffer=False)
    _patch_transport(monkeypatch, connection)
    adapter = OpenAIRealtimeAdapter(
        "ws://example.test/v1/realtime", timeout=1.0, reconnect_attempts=0
    )

    with pytest.raises(ProtocolError, match="incremental text streaming requires"):
        adapter.open_stream(_request())

    assert [event["type"] for event in connection.sent] == ["session.update"]
    assert connection.closed is True


def test_streaming_uses_namespaced_append_commit_and_exposes_usage(monkeypatch):
    connection = FakeRealtimeConnection()
    _patch_transport(monkeypatch, connection)
    adapter = OpenAIRealtimeAdapter(
        "ws://example.test/v1/realtime", timeout=1.0, reconnect_attempts=0
    )
    session = adapter.open_stream(_request())

    session.send_text("你", seq_no=1)
    session.send_text("好", seq_no=2)
    session.end()
    messages = list(session.iter_messages(post_send_idle_timeout=1.0))

    chunks = [message for message in messages if isinstance(message, AudioChunk)]
    assert len(chunks) == 2
    assert chunks[0].first_chunk is True
    assert session.usage["input_tokens"] == 2
    assert session.response_id == "resp_server"
    assert session.response_status == "completed"
    assert [event["type"] for event in connection.sent] == [
        "session.update",
        "response.create",
        "qwen.input_text_buffer.append",
        "qwen.input_text_buffer.append",
        "qwen.input_text_buffer.commit",
    ]
    assert connection.sent[2]["sequence"] == 1
    assert connection.sent[3]["sequence"] == 2


def test_stream_cancel_maps_to_response_cancel_and_preserves_partial_usage(monkeypatch):
    connection = FakeRealtimeConnection()
    _patch_transport(monkeypatch, connection)
    adapter = OpenAIRealtimeAdapter(
        "ws://example.test/v1/realtime", timeout=1.0, reconnect_attempts=0
    )
    session = adapter.open_stream(_request())

    session.send_text("partial")
    session.cancel(reason="caller stopped")
    list(session.iter_messages(post_send_idle_timeout=1.0))

    assert connection.sent[-1] == {"type": "response.cancel"}
    assert session.response_status == "cancelled"
    assert session.usage["total_tokens"] == 3


def test_realtime_pool_reuses_physical_socket_for_serial_responses(monkeypatch):
    connection = FakeRealtimeConnection()
    _patch_transport(monkeypatch, connection)
    adapter = OpenAIRealtimeAdapter(
        "ws://example.test/v1/realtime",
        timeout=1.0,
        reconnect_attempts=0,
        max_connections=1,
        max_idle_connections=1,
    )

    first = adapter.open_stream(_request("first"))
    first.send_text("一")
    first.end()
    list(first.iter_messages(post_send_idle_timeout=1.0))
    second = adapter.open_stream(_request("second"))
    second.send_text("二")
    second.end()
    list(second.iter_messages(post_send_idle_timeout=1.0))

    assert connection.closed is False
    assert connection.sent.count({"type": "session.update"}) == 0
    assert [payload["type"] for payload in connection.sent].count("session.update") == 2
    adapter.close()
    assert connection.closed is True


def test_capabilities_use_http_sibling_of_realtime_path(monkeypatch):
    seen = []

    class Response:
        status_code = 200

        def json(self):
            return {
                "supported_api_protocols": ["openai-realtime-v1"],
                "openai_realtime_path": "/v1/realtime",
                "backend": "triton-grpc",
            }

    def get(url, **kwargs):
        seen.append((url, kwargs))
        return Response()

    monkeypatch.setattr(realtime_module.requests, "get", get)
    adapter = OpenAIRealtimeAdapter(
        "wss://example.test/tenant/v1/realtime",
        timeout=5.0,
        headers={"Authorization": "Bearer secret"},
    )

    capabilities = adapter.get_capabilities(timeout=2.0)

    assert capabilities.extra["backend"] == "triton-grpc"
    assert seen == [
        (
            "https://example.test/tenant/v1/capabilities",
            {"timeout": 2.0, "headers": {"Authorization": "Bearer secret"}},
        )
    ]
