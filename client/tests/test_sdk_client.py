from __future__ import annotations

import pytest

from qwen3tts import SessionStartRequest, SynthesisConfig, TTSClient
from qwen3tts.audio import decode_audio_bytes_to_array
from qwen3tts.client import _build_adapter
from qwen3tts.constants import TRANSPORT_ENGINE_WEBSOCKET


class _FakeAdapter:
    def __init__(self):
        self.start_requests = []

    def get_capabilities(self):
        return {"variant": "fake"}

    def synthesize_bytes(self, text: str, *, request):
        self.start_requests.append((text, request))
        from qwen3tts_protocol import AudioFormat, BytesResult, StreamEvent

        return BytesResult(
            audio_bytes=(b"\x00\x00\x00\x00" * 2),
            audio_format=AudioFormat(),
            session_id=request.session_id,
            transport="fake",
            events=[StreamEvent(type="done", session_id=request.session_id)],
            warnings=[],
            details={},
        )

    def open_stream(self, start_request):
        self.start_requests.append(start_request)

        class _Session:
            def __init__(self):
                self.session_id = start_request.session_id
                self.transport = "fake"

            def send_text(self, text, *, seq_no=None, client_timestamp_ms=None):
                pass

            def end(self, *, client_timestamp_ms=None):
                pass

            def cancel(self, reason=""):
                pass

            def iter_messages(self):
                return iter(())

        return _Session()


def test_synthesize_bytes_uses_session_start_request():
    client = TTSClient(
        endpoint="fake",
        adapter=_FakeAdapter(),
        detected=type("D", (), {"transport": "fake", "probe_report": []})(),
    )
    result = client.synthesize_bytes(
        "hello", request=SynthesisConfig(task_type="custom_voice")
    )
    assert result.transport == "fake"
    assert len(result.audio_bytes) == 8


def test_synthesize_array_returns_numpy_when_available():
    client = TTSClient(
        endpoint="fake",
        adapter=_FakeAdapter(),
        detected=type("D", (), {"transport": "fake", "probe_report": []})(),
    )
    result = client.synthesize_array(
        "hello", request=SynthesisConfig(task_type="custom_voice")
    )
    assert result.audio_array.shape[0] == 2


def test_decode_audio_bytes_to_array_rejects_unknown_encoding():
    with pytest.raises(ValueError, match="unsupported audio encoding"):
        decode_audio_bytes_to_array(b"", encoding="mp3")


def test_open_stream_delegates_to_adapter():
    adapter = _FakeAdapter()
    client = TTSClient(
        endpoint="fake",
        adapter=adapter,
        detected=type("D", (), {"transport": "fake", "probe_report": []})(),
    )
    session = client.open_stream(
        SessionStartRequest(session_id="sid", config=SynthesisConfig())
    )
    assert session.session_id == "sid"


def test_build_websocket_adapter_forwards_connect_timeout():
    adapter = _build_adapter(
        TRANSPORT_ENGINE_WEBSOCKET,
        endpoint="ws://localhost:50052/v1/ws",
        model_name=None,
        model_version="1",
        timeout=120.0,
        connect_timeout=5.0,
        headers={"X-Test": "1"},
        metadata=None,
    )

    assert adapter.timeout == 120.0
    assert adapter.connect_timeout == 5.0
    assert adapter.headers == {"X-Test": "1"}
