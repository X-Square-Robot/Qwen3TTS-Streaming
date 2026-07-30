from __future__ import annotations

import json


from qwen3tts_protocol import (
    BytesResult,
    OutputPolicy,
    SessionStartRequest,
    StreamEvent,
    SynthesisConfig,
    VADPolicy,
)
from qwen3tts._adapters.triton_http import TritonHttpAdapter, TritonHttpBufferedSession
from qwen3tts.constants import TRANSPORT_TRITON_HTTP


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
        assert (
            url
            == "http://localhost:8000/v2/models/tts_orchestrator_http/versions/1/infer"
        )

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

    def test_get_capabilities_uses_per_call_timeout(self, monkeypatch):
        seen_timeouts = []

        class FakeResponse:
            status_code = 200
            text = 'CAPABILITIES:{"loaded_model_type":"custom_voice"}'

        def fake_post(url, json=None, timeout=None, headers=None):
            seen_timeouts.append(timeout)
            return FakeResponse()

        monkeypatch.setattr("qwen3tts._adapters.triton_http.requests.post", fake_post)
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=30.0,
        )

        capabilities = adapter.get_capabilities(timeout=1.25)

        assert capabilities.loaded_model_type == "custom_voice"
        assert seen_timeouts == [1.25]

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

        monkeypatch.setattr("qwen3tts._adapters.triton_http.requests.post", fake_post)

        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(
            session_id="s1", config=SynthesisConfig(task_type="custom_voice")
        )
        result = adapter.synthesize_bytes("hello", request=start)

        assert isinstance(result, BytesResult)
        assert result.transport == TRANSPORT_TRITON_HTTP
        assert result.details.get("degraded_to_oneshot") is False

    def test_request_payload_carries_tenvad_policy(self):
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(
            session_id="s-vad",
            config=SynthesisConfig(task_type="custom_voice"),
            output_policy=OutputPolicy(
                vad=VADPolicy(
                    enabled=True,
                    strategy="tenvad",
                    implementation="onnx",
                    config={"tenvad_threshold": 0.73},
                    chunk_ms=8,
                    begin_threshold=0.91,
                    begin_count=3,
                    end_threshold=0.21,
                    end_count=47,
                    start_margin_ms=12,
                ),
                chunk_ms=40,
                emit_text_events=False,
                config={"delivery": "guarded", "delivery_window_ms": "160"},
            ),
        )
        payload = adapter._request_payload_for_text(start, "hello")

        vad = payload["output_policy"]["vad_policy"]
        assert vad["strategy"] == "tenvad"
        assert vad["enabled"] is True
        assert vad["implementation"] == "onnx"
        assert vad["config"] == {"tenvad_threshold": 0.73}
        assert vad["chunk_ms"] == 8
        assert vad["begin_threshold"] == 0.91
        assert vad["begin_count"] == 3
        assert vad["end_threshold"] == 0.21
        assert vad["end_count"] == 47
        assert vad["start_margin_ms"] == 12
        assert payload["output_policy"]["chunk_ms"] == 40
        assert payload["output_policy"]["emit_text_events"] is False
        assert payload["output_policy"]["config"] == {
            "delivery": "guarded",
            "delivery_window_ms": "160",
        }


class TestTritonHttpBufferedSession:
    def test_degraded_to_oneshot_flag(self):
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(
            session_id="s1", config=SynthesisConfig(task_type="custom_voice")
        )
        session = adapter.open_stream(start)

        assert isinstance(session, TritonHttpBufferedSession)
        assert session.degraded_to_oneshot is True

    def test_buffered_stream_accumulates_text(self):
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(
            session_id="s1", config=SynthesisConfig(task_type="custom_voice")
        )
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
                            # NB: the `json` param above shadows the module, so use a literal.
                            {"name": "event_json", "data": ['{"session_id": "s1"}']},
                            {"name": "is_final", "data": [True]},
                        ]
                    }

                @property
                def text(self):
                    return ""

            return FakeResponse()

        monkeypatch.setattr("qwen3tts._adapters.triton_http.requests.post", fake_post)

        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(
            session_id="s1", config=SynthesisConfig(task_type="custom_voice")
        )
        session = adapter.open_stream(start)

        session.send_text("hello")
        session.end()

        messages = []
        for msg in session.iter_messages():
            messages.append(msg)

        types = [
            m.type if isinstance(m, StreamEvent) else "audio_chunk" for m in messages
        ]
        assert "start" in types
        assert "done" in types

        # Start event should carry degraded_to_oneshot meta
        start_ev = [
            m for m in messages if isinstance(m, StreamEvent) and m.type == "start"
        ][0]
        assert start_ev.meta.get("degraded_to_oneshot") == "true"

    def test_cancel_just_puts_done(self):
        adapter = TritonHttpAdapter(
            "http://localhost:8000",
            model_name="tts_orchestrator_http",
            timeout=5.0,
        )
        start = SessionStartRequest(
            session_id="s1", config=SynthesisConfig(task_type="custom_voice")
        )
        session = adapter.open_stream(start)

        session.cancel(reason="user abort")

        messages = list(session.iter_messages())
        assert len(messages) == 1
        assert isinstance(messages[0], StreamEvent)
        assert messages[0].type == "done"
        assert messages[0].message == "user abort"
