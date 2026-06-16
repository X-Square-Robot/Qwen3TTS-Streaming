from __future__ import annotations

import json
import time

import pytest

from qwen3_tts_protocol import (
    AudioChunk,
    BytesResult,
    Capabilities,
    OutputPolicy,
    SessionStartRequest,
    StreamEvent,
    SynthesisConfig,
    VADPolicy,
)
from qwen3_tts_client._adapters.triton_http import TritonHttpAdapter, TritonHttpBufferedSession
from qwen3_tts_client.constants import TRANSPORT_TRITON_HTTP


class TestTritonHttpAdapter:
    def test_transport_name(self):
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        assert adapter.transport_name == TRANSPORT_TRITON_HTTP

    def test_infer_url_format(self):
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            model_version="1",
            timeout=5.0,
        )
        url = adapter._infer_url()
        assert url == "http://localhost:8000/v2/models/tts_orchestrator_http/versions/1/infer"

    def test_infer_payload_structure(self):
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        payload = adapter._infer_payload({"action": "capabilities"})
        assert "inputs" in payload
        assert payload["inputs"][0]["name"] == "request"
        assert payload["inputs"][0]["datatype"] == "BYTES"
        assert "outputs" in payload
        output_names = [o["name"] for o in payload["outputs"]]
        assert "audio_chunk" in output_names
        assert "event_type" in output_names

    def test_synthesize_bytes(self, monkeypatch):
        caps_json = json.dumps({"loaded_model_type": "custom_voice"}).encode()
        audio_data = b"\x00\x01\x02\x03"

        def fake_post(url, json=None, timeout=None, headers=None):
            class FakeResponse:
                status_code = 200

                def json(self):
                    return {
                        "outputs": [
                            {"name": "audio_chunk", "data": [audio_data.hex()]},
                            {"name": "event_type", "data": ["done"]},
                            {"name": "event_json", "data": ["{}"]},
                            {"name": "is_final", "data": [True]},
                        ]
                    }

                @property
                def text(self):
                    return json.dumps(self.json())

            return FakeResponse()

        monkeypatch.setattr("qwen3_tts_client._adapters.triton_http.requests.post", fake_post)

        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(session_id="s1", config=SynthesisConfig(task_type="custom_voice"))
        result = adapter.synthesize_bytes("hello", request=start)

        assert isinstance(result, BytesResult)
        assert result.transport == TRANSPORT_TRITON_HTTP
        assert result.details.get("degraded_to_oneshot") is False


class TestTritonHttpBufferedSession:
    def test_degraded_to_oneshot_flag(self):
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(session_id="s1", config=SynthesisConfig(task_type="custom_voice"))
        session = adapter.open_stream(start)

        assert isinstance(session, TritonHttpBufferedSession)
        assert session.degraded_to_oneshot is True

    def test_buffered_stream_accumulates_text(self):
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(session_id="s1", config=SynthesisConfig(task_type="custom_voice"))
        session = adapter.open_stream(start)

        session.send_text("hello, ")
        session.send_text("world")
        assert session._text_parts == ["hello, ", "world"]

    def test_end_triggers_degraded_oneshot_and_delivers_events(self, monkeypatch):
        audio_data = b"\x00\x01\x02\x03" * 100

        def fake_post(url, json=None, timeout=None, headers=None):
            class FakeResponse:
                status_code = 200

                def json(self):
                    return {
                        "outputs": [
                            {"name": "audio_chunk", "data": [audio_data.hex()]},
                            {"name": "event_type", "data": ["done"]},
                            {"name": "event_json", "data": [json.dumps({"session_id": "s1"})]},
                            {"name": "is_final", "data": [True]},
                        ]
                    }

                @property
                def text(self):
                    return ""

            return FakeResponse()

        monkeypatch.setattr("qwen3_tts_client._adapters.triton_http.requests.post", fake_post)

        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(session_id="s1", config=SynthesisConfig(task_type="custom_voice"))
        session = adapter.open_stream(start)

        session.send_text("hello")
        session.end()

        messages = []
        for msg in session.iter_messages():
            messages.append(msg)

        types = [m.type if isinstance(m, StreamEvent) else "audio_chunk" for m in messages]
        assert "start" in types
        assert "done" in types

        # Start event should carry degraded_to_oneshot meta
        start_ev = [m for m in messages if isinstance(m, StreamEvent) and m.type == "start"][0]
        assert start_ev.meta.get("degraded_to_oneshot") == "true"

    def test_cancel_just_puts_done(self):
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(session_id="s1", config=SynthesisConfig(task_type="custom_voice"))
        session = adapter.open_stream(start)

        session.cancel(reason="user abort")

        messages = list(session.iter_messages())
        assert len(messages) == 1
        assert isinstance(messages[0], StreamEvent)
        assert messages[0].type == "done"
        assert messages[0].message == "user abort"
