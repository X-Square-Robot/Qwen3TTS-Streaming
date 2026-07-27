from __future__ import annotations

import pytest

from qwen3tts import SessionStartRequest, SynthesisConfig, TTSClient
from qwen3tts.audio import decode_audio_bytes_to_array
from qwen3tts.client import _build_adapter
from qwen3tts.constants import (
    TRANSPORT_ENGINE_GRPC,
    TRANSPORT_ENGINE_WEBSOCKET,
    TRANSPORT_TRITON_GRPC,
    TRANSPORT_TRITON_HTTP,
)


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
        reconnect_attempts=3,
        max_idle_connections=4,
        keepalive_interval=9.0,
        headers={"X-Test": "1"},
        metadata=None,
    )

    assert adapter.timeout == 120.0
    assert adapter.connect_timeout == 5.0
    assert adapter.reconnect_attempts == 3
    assert adapter.max_idle_connections == 4
    assert adapter.keepalive_interval == 9.0
    assert adapter.headers == {"X-Test": "1"}


@pytest.mark.parametrize(
    ("transport", "endpoint"),
    [
        (TRANSPORT_ENGINE_WEBSOCKET, "ws://localhost:50052/v1/ws"),
        (TRANSPORT_TRITON_HTTP, "http://localhost:8000"),
    ],
)
def test_connect_key_adds_bearer_header_for_http_transports(transport, endpoint):
    client = TTSClient.connect(
        endpoint,
        transport=transport,
        key="secret",
        verify=False,
    )

    assert client._adapter.headers == {"Authorization": "Bearer secret"}


@pytest.mark.parametrize(
    ("transport", "endpoint"),
    [
        (TRANSPORT_ENGINE_GRPC, "localhost:50051"),
        (TRANSPORT_TRITON_GRPC, "localhost:8001"),
    ],
)
def test_connect_key_adds_lowercase_grpc_metadata(transport, endpoint):
    client = TTSClient.connect(
        endpoint,
        transport=transport,
        key="secret",
        verify=False,
    )

    assert ("authorization", "Bearer secret") in client._adapter.metadata


def test_connect_key_overrides_authorization_without_mutating_inputs():
    headers = {"X-Test": "1", "authorization": "Bearer old-header"}
    metadata = [("x-meta", "2"), ("Authorization", "Bearer old-metadata")]

    client = TTSClient.connect(
        "localhost:50051",
        transport=TRANSPORT_ENGINE_GRPC,
        key="new-key",
        headers=headers,
        metadata=metadata,
        verify=False,
    )

    assert headers == {"X-Test": "1", "authorization": "Bearer old-header"}
    assert metadata == [
        ("x-meta", "2"),
        ("Authorization", "Bearer old-metadata"),
    ]
    assert client._adapter.metadata == (
        ("x-test", "1"),
        ("x-meta", "2"),
        ("authorization", "Bearer new-key"),
    )


def test_connect_key_none_does_not_inject_authorization():
    client = TTSClient.connect(
        "ws://localhost:50052/v1/ws",
        transport=TRANSPORT_ENGINE_WEBSOCKET,
        key=None,
        headers={"X-Test": "1"},
        verify=False,
    )

    assert client._adapter.headers == {"X-Test": "1"}
