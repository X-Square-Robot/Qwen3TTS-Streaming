from __future__ import annotations

import json
import socket

import pytest

from qwen3tts.constants import (
    DEFAULT_TRITON_HTTP_MODEL,
    DEFAULT_TRITON_GRPC_MODEL,
    TRANSPORT_OPENAI_REALTIME,
    TRANSPORT_ENGINE_GRPC,
    TRANSPORT_ENGINE_WEBSOCKET,
    TRANSPORT_TRITON_GRPC,
    TRANSPORT_TRITON_HTTP,
)
from qwen3tts.detect import detect_transport
from qwen3tts import detect as detect_module


def test_explicit_ws_transport_resolves_ws_endpoint():
    detected = detect_transport(
        "localhost:50052",
        transport=TRANSPORT_ENGINE_WEBSOCKET,
        model_name=None,
        timeout=1.0,
    )
    assert detected.transport == TRANSPORT_ENGINE_WEBSOCKET
    assert detected.resolved_endpoint == "ws://localhost:50052/v1/ws"


def test_explicit_openai_realtime_resolves_default_path():
    detected = detect_transport(
        "localhost:50053",
        transport=TRANSPORT_OPENAI_REALTIME,
        model_name=None,
        timeout=1.0,
    )
    assert detected.transport == TRANSPORT_OPENAI_REALTIME
    assert detected.resolved_endpoint == "ws://localhost:50053/v1/realtime"


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

    def fake_probe(url, *, timeout, connect_timeout, headers):
        calls.append((url, timeout, connect_timeout, headers))

    monkeypatch.setattr("qwen3tts.detect._probe_engine_websocket", fake_probe)
    detected = detect_transport(
        "ws://example.test/v1/ws",
        transport="auto",
        model_name=None,
        timeout=2.0,
        connect_timeout=0.5,
        headers={"Authorization": "Bearer secret"},
    )
    assert detected.transport == TRANSPORT_ENGINE_WEBSOCKET
    assert calls == [
        (
            "ws://example.test/v1/ws",
            2.0,
            0.5,
            {"Authorization": "Bearer secret"},
        )
    ]


def test_realtime_ws_scheme_short_circuit(monkeypatch):
    calls = []

    def fake_probe(url, *, timeout, connect_timeout, headers):
        calls.append((url, timeout, connect_timeout, headers))

    monkeypatch.setattr("qwen3tts.detect._probe_openai_realtime", fake_probe)
    detected = detect_transport(
        "wss://example.test/v1/realtime",
        transport="auto",
        model_name="private-realtime-model",
        timeout=2.0,
        connect_timeout=0.5,
        headers={"Authorization": "Bearer secret"},
    )
    assert detected.transport == TRANSPORT_OPENAI_REALTIME
    assert detected.model_name == "private-realtime-model"
    assert calls == [
        (
            "wss://example.test/v1/realtime",
            2.0,
            0.5,
            {"Authorization": "Bearer secret"},
        )
    ]


def test_websocket_probe_retries_short_receive_timeouts(monkeypatch):
    class _Connection:
        def settimeout(self, _timeout):
            pass

    connection = _Connection()
    connect_timeouts: list[float] = []
    closed: list[object] = []
    responses = iter(
        [
            socket.timeout("not ready yet"),
            {
                "type": "capabilities",
                "capabilities": {},
            },
        ]
    )

    def connect(_url, *, timeout, headers):
        connect_timeouts.append(timeout)
        return connection

    def recv_frame(_conn):
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return 0x1, json.dumps(response).encode("utf-8")

    monkeypatch.setattr(detect_module, "ws_connect", connect)
    monkeypatch.setattr(detect_module, "ws_send_json", lambda *_args: None)
    monkeypatch.setattr(detect_module, "ws_recv_frame", recv_frame)
    monkeypatch.setattr(detect_module, "ws_close", closed.append)
    monkeypatch.setattr(detect_module, "check_capabilities_pairing", lambda _caps: None)

    detect_module._probe_engine_websocket(
        "ws://example.test/v1/ws",
        timeout=1.0,
        connect_timeout=0.25,
        headers=None,
    )

    assert connect_timeouts == [0.25]
    assert closed == [connection]


def test_http_scheme_prefers_standalone_capabilities(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"loaded_model_type": "custom_voice", "variant": "custom-1.7b"}

    monkeypatch.setattr("qwen3tts.detect.requests.get", lambda *args, **kwargs: _Resp())
    detected = detect_transport(
        "http://example.test:50052",
        transport="auto",
        model_name=None,
        timeout=2.0,
    )
    assert detected.transport == TRANSPORT_ENGINE_WEBSOCKET
    assert detected.resolved_endpoint == "ws://example.test:50052/v1/ws"


def test_http_capabilities_prefer_openai_realtime_when_advertised(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {
                "loaded_model_type": "custom_voice",
                "supported_api_protocols": [
                    "openai-realtime-v1",
                    "tts-session-v2alpha1",
                ],
                "openai_realtime_path": "/v1/realtime",
            }

    monkeypatch.setattr("qwen3tts.detect.requests.get", lambda *args, **kwargs: _Resp())
    detected = detect_transport(
        "https://example.test/tts",
        transport="auto",
        model_name=None,
        timeout=2.0,
    )

    assert detected.transport == TRANSPORT_OPENAI_REALTIME
    assert detected.resolved_endpoint == "wss://example.test/tts/v1/realtime"


def test_https_capabilities_resolves_secure_websocket(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"loaded_model_type": "custom_voice", "variant": "custom-1.7b"}

    monkeypatch.setattr("qwen3tts.detect.requests.get", lambda *args, **kwargs: _Resp())
    detected = detect_transport(
        "https://example.test/tts",
        transport="auto",
        model_name=None,
        timeout=2.0,
    )

    assert detected.resolved_endpoint == "wss://example.test/tts/v1/ws"


def test_host_port_prefers_engine_grpc(monkeypatch):
    calls = []

    def fake_engine_grpc(endpoint, *, timeout, headers, metadata):
        assert endpoint == "host.test:50051"
        calls.append((headers, metadata))

    monkeypatch.setattr("qwen3tts.detect._probe_engine_grpc", fake_engine_grpc)
    monkeypatch.setattr(
        "qwen3tts.detect._probe_triton_grpc",
        lambda *args, **kwargs: pytest.fail("should not probe triton grpc"),
    )
    detected = detect_transport(
        "host.test:50051",
        transport="auto",
        model_name=None,
        timeout=2.0,
        headers={"Authorization": "Bearer secret"},
        metadata=(("authorization", "Bearer secret"),),
    )
    assert detected.transport == TRANSPORT_ENGINE_GRPC
    assert calls == [
        (
            {"Authorization": "Bearer secret"},
            (("authorization", "Bearer secret"),),
        )
    ]


def test_host_without_port_expands_candidates(monkeypatch):
    seen = []
    realtime_seen = []

    def fake_engine_ws(url, *, timeout, connect_timeout, headers):
        seen.append((url, headers))

    def fake_realtime(url, **_kwargs):
        realtime_seen.append(url)
        raise RuntimeError("not realtime")

    monkeypatch.setattr("qwen3tts.detect._probe_openai_realtime", fake_realtime)
    monkeypatch.setattr("qwen3tts.detect._probe_engine_websocket", fake_engine_ws)
    detected = detect_transport(
        "host.test",
        transport="auto",
        model_name=None,
        timeout=2.0,
        headers={"Authorization": "Bearer secret"},
    )
    assert detected.transport == TRANSPORT_ENGINE_WEBSOCKET
    assert seen == [
        (
            "ws://host.test:50052/v1/ws",
            {"Authorization": "Bearer secret"},
        )
    ]
    assert realtime_seen == [
        "ws://host.test:50052/v1/realtime",
        "ws://host.test:50053/v1/realtime",
    ]


def test_host_without_port_prefers_realtime_before_legacy(monkeypatch):
    seen = []

    def fake_realtime(url, *, timeout, connect_timeout, headers):
        seen.append(url)

    monkeypatch.setattr("qwen3tts.detect._probe_openai_realtime", fake_realtime)
    monkeypatch.setattr(
        "qwen3tts.detect._probe_engine_websocket",
        lambda *_args, **_kwargs: pytest.fail("legacy websocket should not be probed"),
    )

    detected = detect_transport(
        "host.test",
        transport="auto",
        model_name=None,
        timeout=2.0,
    )

    assert detected.transport == TRANSPORT_OPENAI_REALTIME
    assert detected.resolved_endpoint == "ws://host.test:50052/v1/realtime"
    assert seen == ["ws://host.test:50052/v1/realtime"]


def test_host_without_port_tries_triton_realtime_sidecar_second(monkeypatch):
    seen = []

    def fake_realtime(url, *, timeout, connect_timeout, headers):
        seen.append(url)
        if ":50052/" in url:
            raise RuntimeError("standalone unavailable")

    monkeypatch.setattr("qwen3tts.detect._probe_openai_realtime", fake_realtime)
    monkeypatch.setattr(
        "qwen3tts.detect._probe_engine_websocket",
        lambda *_args, **_kwargs: pytest.fail(
            "legacy must not be probed before the Realtime sidecar"
        ),
    )

    detected = detect_transport(
        "host.test",
        transport="auto",
        model_name=None,
        timeout=2.0,
    )

    assert detected.transport == TRANSPORT_OPENAI_REALTIME
    assert detected.resolved_endpoint == "ws://host.test:50053/v1/realtime"
    assert seen == [
        "ws://host.test:50052/v1/realtime",
        "ws://host.test:50053/v1/realtime",
    ]


def test_bare_http_probe_forwards_headers(monkeypatch):
    seen = []

    def fake_http(
        base_url,
        *,
        timeout,
        headers,
        report,
        model_name,
        model_version,
    ):
        seen.append((base_url, headers))
        return type(
            "Detected",
            (),
            {
                "transport": TRANSPORT_TRITON_HTTP,
                "resolved_endpoint": base_url,
            },
        )()

    monkeypatch.setattr(detect_module, "_detect_http_url", fake_http)
    detected = detect_transport(
        "host.test:8000",
        transport="auto",
        model_name=None,
        timeout=2.0,
        headers={"Authorization": "Bearer secret"},
    )

    assert detected.transport == TRANSPORT_TRITON_HTTP
    assert seen == [
        (
            "http://host.test:8000",
            {"Authorization": "Bearer secret"},
        )
    ]


def test_bare_triton_grpc_probe_forwards_auth(monkeypatch):
    seen = []

    def fake_triton_grpc(
        endpoint,
        *,
        timeout,
        model_name,
        headers,
        metadata,
    ):
        seen.append((endpoint, headers, metadata))

    monkeypatch.setattr(detect_module, "_probe_triton_grpc", fake_triton_grpc)
    detected = detect_transport(
        "host.test:8001",
        transport="auto",
        model_name=None,
        timeout=2.0,
        headers={"Authorization": "Bearer secret"},
        metadata=(("authorization", "Bearer secret"),),
    )

    assert detected.transport == TRANSPORT_TRITON_GRPC
    assert seen == [
        (
            "host.test:8001",
            {"Authorization": "Bearer secret"},
            (("authorization", "Bearer secret"),),
        )
    ]
