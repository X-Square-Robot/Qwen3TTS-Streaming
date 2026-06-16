from __future__ import annotations

import pytest

from qwen3_tts_client.constants import (
    DEFAULT_TRITON_HTTP_MODEL,
    DEFAULT_TRITON_GRPC_MODEL,
    TRANSPORT_ENGINE_GRPC,
    TRANSPORT_ENGINE_WEBSOCKET,
    TRANSPORT_TRITON_GRPC,
    TRANSPORT_TRITON_HTTP,
)
from qwen3_tts_client.detect import detect_transport


def test_explicit_ws_transport_resolves_ws_endpoint():
    detected = detect_transport(
        "localhost:50052",
        transport=TRANSPORT_ENGINE_WEBSOCKET,
        model_name=None,
        timeout=1.0,
    )
    assert detected.transport == TRANSPORT_ENGINE_WEBSOCKET
    assert detected.resolved_endpoint == "ws://localhost:50052/v1/ws"


def test_explicit_triton_http_resolves_http_endpoint():
    detected = detect_transport(
        "localhost:8000",
        transport=TRANSPORT_TRITON_HTTP,
        model_name=None,
        timeout=1.0,
    )
    assert detected.transport == TRANSPORT_TRITON_HTTP
    assert detected.resolved_endpoint == "http://localhost:8000"
    assert detected.model_name == DEFAULT_TRITON_HTTP_MODEL


def test_explicit_engine_grpc_keeps_host_port():
    detected = detect_transport(
        "localhost:50051",
        transport=TRANSPORT_ENGINE_GRPC,
        model_name=None,
        timeout=1.0,
    )
    assert detected.transport == TRANSPORT_ENGINE_GRPC
    assert detected.resolved_endpoint == "localhost:50051"


def test_explicit_triton_grpc_default_model_name():
    detected = detect_transport(
        "localhost:8001",
        transport=TRANSPORT_TRITON_GRPC,
        model_name=None,
        timeout=1.0,
    )
    assert detected.transport == TRANSPORT_TRITON_GRPC
    assert detected.model_name == DEFAULT_TRITON_GRPC_MODEL


def test_ws_scheme_short_circuit(monkeypatch):
    calls = []

    def fake_probe(url, *, timeout, headers):
        calls.append((url, timeout))

    monkeypatch.setattr("qwen3_tts_client.detect._probe_engine_websocket", fake_probe)
    detected = detect_transport(
        "ws://example.test/v1/ws",
        transport="auto",
        model_name=None,
        timeout=2.0,
    )
    assert detected.transport == TRANSPORT_ENGINE_WEBSOCKET
    assert calls == [("ws://example.test/v1/ws", 2.0)]


def test_http_scheme_prefers_standalone_capabilities(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"loaded_model_type": "custom_voice", "variant": "custom-1.7b"}

    monkeypatch.setattr("qwen3_tts_client.detect.requests.get", lambda *args, **kwargs: _Resp())
    detected = detect_transport(
        "http://example.test:50052",
        transport="auto",
        model_name=None,
        timeout=2.0,
    )
    assert detected.transport == TRANSPORT_ENGINE_WEBSOCKET
    assert detected.resolved_endpoint == "http://example.test:50052/v1/ws"


def test_host_port_prefers_engine_grpc(monkeypatch):
    def fake_engine_grpc(endpoint, *, timeout):
        assert endpoint == "host.test:50051"

    monkeypatch.setattr("qwen3_tts_client.detect._probe_engine_grpc", fake_engine_grpc)
    monkeypatch.setattr("qwen3_tts_client.detect._probe_triton_grpc", lambda *args, **kwargs: pytest.fail("should not probe triton grpc"))
    detected = detect_transport(
        "host.test:50051",
        transport="auto",
        model_name=None,
        timeout=2.0,
    )
    assert detected.transport == TRANSPORT_ENGINE_GRPC


def test_host_without_port_expands_candidates(monkeypatch):
    seen = []

    def fake_engine_ws(url, *, timeout, headers):
        seen.append(url)

    monkeypatch.setattr("qwen3_tts_client.detect._probe_engine_websocket", fake_engine_ws)
    detected = detect_transport(
        "host.test",
        transport="auto",
        model_name=None,
        timeout=2.0,
    )
    assert detected.transport == TRANSPORT_ENGINE_WEBSOCKET
    assert seen == ["ws://host.test:50052/v1/ws"]
