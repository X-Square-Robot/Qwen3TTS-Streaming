from __future__ import annotations

import warnings

import pytest

from qwen3tts import SessionStartRequest, SynthesisConfig, TTSClient
from qwen3tts.audio import decode_audio_bytes_to_array
from qwen3tts.client import _build_adapter
from qwen3tts import client as client_module
from qwen3tts.constants import (
    TRANSPORT_ENGINE_GRPC,
    TRANSPORT_ENGINE_WEBSOCKET,
    TRANSPORT_OPENAI_REALTIME,
    TRANSPORT_TRITON_GRPC,
    TRANSPORT_TRITON_HTTP,
)


class _FakeAdapter:
    def __init__(self):
        self.start_requests = []
        self.prewarm_requests = []

    def get_capabilities(self):
        return {"variant": "fake"}

    def prewarm(self, connections):
        self.prewarm_requests.append(connections)
        return connections

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


def test_get_capabilities_timeout_none_preserves_legacy_adapter_call():
    client = TTSClient(
        endpoint="fake",
        adapter=_FakeAdapter(),
        detected=type("D", (), {"transport": "fake", "probe_report": []})(),
    )

    assert client.get_capabilities(timeout=None) == {"variant": "fake"}


def test_get_capabilities_forwards_explicit_timeout():
    seen: list[float | None] = []

    class _TimedAdapter(_FakeAdapter):
        def get_capabilities(self, *, timeout=None):
            seen.append(timeout)
            return {"variant": "timed"}

    client = TTSClient(
        endpoint="fake",
        adapter=_TimedAdapter(),
        detected=type("D", (), {"transport": "fake", "probe_report": []})(),
    )

    assert client.get_capabilities(timeout=2.5) == {"variant": "timed"}
    assert seen == [2.5]


def test_prewarm_timeout_none_preserves_legacy_adapter_call():
    adapter = _FakeAdapter()
    client = TTSClient(
        endpoint="fake",
        adapter=adapter,
        detected=type("D", (), {"transport": "fake", "probe_report": []})(),
    )

    assert client.prewarm(3, timeout=None) == 3
    assert adapter.prewarm_requests == [3]


def test_prewarm_forwards_explicit_timeout():
    seen = []

    class _TimedAdapter(_FakeAdapter):
        def prewarm(self, connections, *, timeout=None):
            seen.append((connections, timeout))
            return connections

    client = TTSClient(
        endpoint="fake",
        adapter=_TimedAdapter(),
        detected=type("D", (), {"transport": "fake", "probe_report": []})(),
    )

    assert client.prewarm(4, timeout=2.5) == 4
    assert seen == [(4, 2.5)]


def test_build_websocket_adapter_forwards_connect_timeout():
    adapter = _build_adapter(
        TRANSPORT_ENGINE_WEBSOCKET,
        endpoint="ws://localhost:50052/v1/ws",
        model_name=None,
        model_version="1",
        timeout=120.0,
        connect_timeout=5.0,
        reconnect_attempts=3,
        max_connections=12,
        max_idle_connections=4,
        max_pending_acquires=40,
        acquire_timeout=2.5,
        idle_ttl=60.0,
        max_lifetime=900.0,
        keepalive_interval=9.0,
        keepalive_jitter=0.1,
        headers={"X-Test": "1"},
        metadata=None,
    )

    assert adapter.timeout == 120.0
    assert adapter.connect_timeout == 5.0
    assert adapter.reconnect_attempts == 3
    assert adapter.max_connections == 12
    assert adapter.max_idle_connections == 4
    assert adapter.max_pending_acquires == 40
    assert adapter.acquire_timeout == 2.5
    assert adapter.idle_ttl == 60.0
    assert adapter.max_lifetime == 900.0
    assert adapter.keepalive_interval == 9.0
    assert adapter.keepalive_jitter == 0.1
    assert adapter.headers == {"X-Test": "1"}


def test_build_openai_realtime_adapter_forwards_model_and_auth():
    adapter = _build_adapter(
        TRANSPORT_OPENAI_REALTIME,
        endpoint="ws://localhost:50053/v1/realtime",
        model_name="qwen3-tts-realtime",
        model_version="1",
        timeout=120.0,
        connect_timeout=5.0,
        reconnect_attempts=3,
        headers={"Authorization": "Bearer secret"},
        metadata=None,
    )

    assert adapter.transport_name == TRANSPORT_OPENAI_REALTIME
    assert adapter.endpoint.endswith("?model=qwen3-tts-realtime")
    assert adapter.connect_timeout == 5.0
    assert adapter.reconnect_attempts == 3
    assert adapter.headers == {"Authorization": "Bearer secret"}


def test_build_websocket_adapter_forwards_tls_policy():
    adapter = _build_adapter(
        TRANSPORT_OPENAI_REALTIME,
        endpoint="wss://localhost:50052/v1/realtime",
        model_name="qwen3-tts-realtime",
        model_version="1",
        timeout=30.0,
        connect_timeout=5.0,
        headers=None,
        metadata=None,
        tls_verify=False,
    )

    assert adapter._tls.verify is False


def test_connect_forwards_one_tls_policy_to_detection_and_adapter(monkeypatch):
    seen = {}

    def detect(*args, **kwargs):
        seen["detect_tls"] = kwargs["tls_verify"]
        return type(
            "Detected",
            (),
            {
                "transport": TRANSPORT_OPENAI_REALTIME,
                "resolved_endpoint": "wss://localhost:50052/v1/realtime",
                "model_name": "qwen3-tts-realtime",
                "model_version": "1",
                "probe_report": [],
            },
        )()

    def build(*args, **kwargs):
        seen["adapter_tls"] = kwargs["tls_verify"]
        return _FakeAdapter()

    monkeypatch.setattr(client_module, "detect_transport", detect)
    monkeypatch.setattr(client_module, "_build_adapter", build)

    TTSClient.connect(
        "wss://localhost:50052/v1/realtime",
        tls_verify=False,
    )

    assert seen["detect_tls"] is seen["adapter_tls"]
    assert seen["detect_tls"].verify is False


def test_verify_protocol_replaces_legacy_verify_keyword(monkeypatch):
    class _NoCapabilityProbeAdapter(_FakeAdapter):
        def get_capabilities(self):
            raise AssertionError("verify_protocol=False should skip capabilities")

    monkeypatch.setattr(
        client_module,
        "detect_transport",
        lambda *args, **kwargs: type(
            "Detected",
            (),
            {
                "transport": TRANSPORT_OPENAI_REALTIME,
                "resolved_endpoint": "ws://localhost:50052/v1/realtime",
                "model_name": "qwen3-tts-realtime",
                "model_version": "1",
                "probe_report": [],
            },
        )(),
    )
    adapter = _NoCapabilityProbeAdapter()
    monkeypatch.setattr(
        client_module, "_build_adapter", lambda *args, **kwargs: adapter
    )

    TTSClient.connect(
        "ws://localhost:50052/v1/realtime",
        transport=TRANSPORT_OPENAI_REALTIME,
        verify_protocol=False,
    )


def test_verify_and_verify_protocol_cannot_disagree():
    with pytest.raises(ValueError, match="must not disagree"):
        TTSClient.connect(
            "ws://localhost:50052/v1/realtime",
            verify=False,
            verify_protocol=True,
        )


def test_legacy_transport_warning_is_emitted_once(monkeypatch):
    client_module._WARNED_LEGACY_TRANSPORTS.clear()
    monkeypatch.delenv("QWEN3TTS_SUPPRESS_LEGACY_TRANSPORT_WARNING", raising=False)

    with pytest.warns(FutureWarning, match="compatibility path"):
        TTSClient.connect(
            "ws://localhost:50052/v1/ws",
            transport=TRANSPORT_ENGINE_WEBSOCKET,
            verify=False,
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TTSClient.connect(
            "ws://localhost:50052/v1/ws",
            transport=TRANSPORT_ENGINE_WEBSOCKET,
            verify=False,
        )
    assert caught == []


def test_openai_realtime_transport_has_no_legacy_warning(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "detect_transport",
        lambda *args, **kwargs: type(
            "Detected",
            (),
            {
                "transport": TRANSPORT_OPENAI_REALTIME,
                "resolved_endpoint": "ws://localhost:50053/v1/realtime",
                "model_name": "qwen3-tts-realtime",
                "model_version": "1",
                "probe_report": [],
            },
        )(),
    )
    monkeypatch.setattr(
        client_module, "_build_adapter", lambda *args, **kwargs: _FakeAdapter()
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TTSClient.connect("localhost", transport="auto", verify=False)
    assert caught == []


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
