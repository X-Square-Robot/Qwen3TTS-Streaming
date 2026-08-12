from __future__ import annotations

import json
import socket
import time
from urllib.parse import urlparse

import requests

from qwen3tts_protocol import DetectedTransport

from ._internal.auth import grpc_metadata_as_headers, normalize_grpc_metadata
from ._internal.raw_websocket import ws_close, ws_connect, ws_recv_frame, ws_send_json
from ._internal.utils import check_capabilities_pairing, check_engine_version
from .constants import (
    DEFAULT_ENGINE_CAPABILITIES_PATH,
    DEFAULT_ENGINE_GRPC_PORT,
    DEFAULT_ENGINE_WS_PATH,
    DEFAULT_ENGINE_WS_PORT,
    DEFAULT_OPENAI_REALTIME_MODEL,
    DEFAULT_OPENAI_REALTIME_PATH,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TRITON_REALTIME_PORT,
    DEFAULT_TRITON_GRPC_MODEL,
    DEFAULT_TRITON_GRPC_PORT,
    DEFAULT_TRITON_HTTP_MODEL,
    DEFAULT_TRITON_HTTP_PORT,
    SUPPORTED_TRANSPORTS,
    TRANSPORT_OPENAI_REALTIME,
    TRANSPORT_ENGINE_GRPC,
    TRANSPORT_ENGINE_WEBSOCKET,
    TRANSPORT_TRITON_GRPC,
    TRANSPORT_TRITON_HTTP,
)
from .exceptions import (
    DependencyMissingError,
    ProtocolVersionMismatchError,
    TransportProbeError,
)


def detect_transport(
    endpoint: str,
    *,
    transport: str,
    model_name: str | None,
    model_version: str = DEFAULT_MODEL_VERSION,
    timeout: float,
    connect_timeout: float | None = None,
    headers: dict[str, str] | None = None,
    metadata=None,
) -> DetectedTransport:
    if transport != "auto":
        if transport not in SUPPORTED_TRANSPORTS:
            raise TransportProbeError(f"unsupported transport: {transport!r}")
        resolved_endpoint = _resolve_explicit_endpoint(endpoint, transport)
        return DetectedTransport(
            requested_endpoint=endpoint,
            resolved_endpoint=resolved_endpoint,
            transport=transport,
            model_name=model_name or _default_model_for_transport(transport),
            model_version=model_version,
            probe_report=[
                {
                    "transport": transport,
                    "endpoint": resolved_endpoint,
                    "ok": True,
                    "reason": "explicit",
                }
            ],
        )

    report: list[dict] = []
    parsed = urlparse(endpoint if "://" in endpoint else "")
    if parsed.scheme in {"ws", "wss"}:
        return _detect_websocket_url(
            endpoint,
            timeout=timeout,
            connect_timeout=connect_timeout,
            headers=headers,
            report=report,
            model_name=model_name,
            model_version=model_version,
        )
    if parsed.scheme in {"http", "https"}:
        return _detect_http_url(
            endpoint,
            timeout=timeout,
            headers=headers,
            report=report,
            model_name=model_name,
            model_version=model_version,
        )
    return _detect_bare_endpoint(
        endpoint,
        timeout=timeout,
        connect_timeout=connect_timeout,
        headers=headers,
        metadata=metadata,
        report=report,
        model_name=model_name,
        model_version=model_version,
    )


def _detect_websocket_url(
    url: str,
    *,
    timeout: float,
    connect_timeout: float | None,
    headers,
    report: list[dict],
    model_name: str | None,
    model_version: str,
) -> DetectedTransport:
    parsed = urlparse(url)
    if not parsed.path.endswith(DEFAULT_ENGINE_WS_PATH):
        realtime_url = url
        if parsed.path in ("", "/"):
            realtime_url = url.rstrip("/") + DEFAULT_OPENAI_REALTIME_PATH
        try:
            _probe_openai_realtime(
                realtime_url,
                timeout=timeout,
                connect_timeout=connect_timeout,
                headers=headers,
            )
        except Exception as exc:
            report.append(
                {
                    "transport": TRANSPORT_OPENAI_REALTIME,
                    "endpoint": realtime_url,
                    "ok": False,
                    "reason": str(exc),
                }
            )
            if parsed.path.endswith(DEFAULT_OPENAI_REALTIME_PATH):
                raise TransportProbeError(
                    f"OpenAI Realtime endpoint probe failed: {exc}",
                    probe_report=report,
                ) from exc
        else:
            report.append(
                {
                    "transport": TRANSPORT_OPENAI_REALTIME,
                    "endpoint": realtime_url,
                    "ok": True,
                }
            )
            return DetectedTransport(
                requested_endpoint=url,
                resolved_endpoint=realtime_url,
                transport=TRANSPORT_OPENAI_REALTIME,
                model_name=model_name or DEFAULT_OPENAI_REALTIME_MODEL,
                model_version=model_version,
                probe_report=report,
            )

    legacy_url = url
    if parsed.path in ("", "/"):
        legacy_url = url.rstrip("/") + DEFAULT_ENGINE_WS_PATH
    try:
        _probe_engine_websocket(
            legacy_url,
            timeout=timeout,
            connect_timeout=connect_timeout,
            headers=headers,
        )
    except ProtocolVersionMismatchError:
        # Definitive answer: we reached a live engine, wrong SDK pairing.
        raise
    except Exception as exc:
        report.append(
            {
                "transport": TRANSPORT_ENGINE_WEBSOCKET,
                "endpoint": legacy_url,
                "ok": False,
                "reason": str(exc),
            }
        )
        raise TransportProbeError(
            f"websocket endpoint probe failed: {exc}", probe_report=report
        ) from exc
    report.append(
        {
            "transport": TRANSPORT_ENGINE_WEBSOCKET,
            "endpoint": legacy_url,
            "ok": True,
        }
    )
    return DetectedTransport(
        requested_endpoint=url,
        resolved_endpoint=legacy_url,
        transport=TRANSPORT_ENGINE_WEBSOCKET,
        model_name="",
        model_version=model_version,
        probe_report=report,
    )


def _detect_http_url(
    base_url: str,
    *,
    timeout: float,
    headers,
    report: list[dict],
    model_name: str | None,
    model_version: str,
) -> DetectedTransport:
    capabilities_url = f"{base_url.rstrip('/')}{DEFAULT_ENGINE_CAPABILITIES_PATH}"
    try:
        response = requests.get(capabilities_url, timeout=timeout, headers=headers)
        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, dict):
                protocols = list(payload.get("supported_api_protocols") or [])
                if "openai-realtime-v1" in protocols:
                    check_engine_version(payload.get("engine_version"))
                    report.append(
                        {
                            "transport": "openai-realtime-capabilities",
                            "endpoint": capabilities_url,
                            "ok": True,
                        }
                    )
                    realtime_path = str(
                        payload.get("openai_realtime_path")
                        or DEFAULT_OPENAI_REALTIME_PATH
                    )
                    return DetectedTransport(
                        requested_endpoint=base_url,
                        resolved_endpoint=(
                            f"{_http_base_to_websocket(base_url).rstrip('/')}"
                            f"{realtime_path}"
                        ),
                        transport=TRANSPORT_OPENAI_REALTIME,
                        model_name=model_name or DEFAULT_OPENAI_REALTIME_MODEL,
                        model_version=model_version,
                        probe_report=report,
                    )
                if "loaded_model_type" in payload:
                    check_capabilities_pairing(payload)
                    report.append(
                        {
                            "transport": "engine-http-capabilities",
                            "endpoint": capabilities_url,
                            "ok": True,
                        }
                    )
                    return DetectedTransport(
                        requested_endpoint=base_url,
                        resolved_endpoint=(
                            f"{_http_base_to_websocket(base_url).rstrip('/')}"
                            f"{DEFAULT_ENGINE_WS_PATH}"
                        ),
                        transport=TRANSPORT_ENGINE_WEBSOCKET,
                        model_name="",
                        model_version=model_version,
                        probe_report=report,
                    )
        report.append(
            {
                "transport": "engine-http-capabilities",
                "endpoint": capabilities_url,
                "ok": False,
                "reason": f"http {response.status_code}",
            }
        )
    except ProtocolVersionMismatchError:
        raise
    except Exception as exc:
        report.append(
            {
                "transport": "engine-http-capabilities",
                "endpoint": capabilities_url,
                "ok": False,
                "reason": str(exc),
            }
        )

    triton_model = model_name or DEFAULT_TRITON_HTTP_MODEL
    for path in (
        "/v2/health/live",
        "/v2/health/ready",
        f"/v2/models/{triton_model}/ready",
    ):
        try:
            response = requests.get(
                f"{base_url.rstrip('/')}{path}", timeout=timeout, headers=headers
            )
            ok = response.status_code == 200
            report.append(
                {
                    "transport": TRANSPORT_TRITON_HTTP,
                    "endpoint": f"{base_url.rstrip('/')}{path}",
                    "ok": ok,
                    "reason": "" if ok else f"http {response.status_code}",
                }
            )
            if not ok:
                raise RuntimeError(f"http {response.status_code}")
        except Exception as exc:
            if path.endswith("/ready"):
                raise TransportProbeError(
                    f"http endpoint did not match standalone capabilities or Triton HTTP ready checks: {exc}",
                    probe_report=report,
                ) from exc
    return DetectedTransport(
        requested_endpoint=base_url,
        resolved_endpoint=base_url.rstrip("/"),
        transport=TRANSPORT_TRITON_HTTP,
        model_name=triton_model,
        model_version=model_version,
        probe_report=report,
    )


def _http_base_to_websocket(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"expected HTTP endpoint, got {base_url!r}")
    websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
    return parsed._replace(scheme=websocket_scheme).geturl()


def _detect_bare_endpoint(
    endpoint: str,
    *,
    timeout: float,
    connect_timeout: float | None,
    headers,
    metadata,
    report: list[dict],
    model_name: str | None,
    model_version: str,
) -> DetectedTransport:
    candidates: list[tuple[str, str]]
    has_explicit_port = ":" in endpoint and not endpoint.endswith("]")
    if has_explicit_port:
        candidates = [(endpoint, "single")]
    else:
        for port in (DEFAULT_ENGINE_WS_PORT, DEFAULT_TRITON_REALTIME_PORT):
            detected = _probe_bare_realtime_candidate(
                requested_endpoint=endpoint,
                host=endpoint,
                port=port,
                timeout=timeout,
                connect_timeout=connect_timeout,
                headers=headers,
                report=report,
                model_name=model_name,
                model_version=model_version,
            )
            if detected is not None:
                return detected
        candidates = [
            (f"{endpoint}:{DEFAULT_ENGINE_WS_PORT}", "ws-port"),
            (f"{endpoint}:{DEFAULT_ENGINE_GRPC_PORT}", "grpc-port"),
            (f"{endpoint}:{DEFAULT_TRITON_GRPC_PORT}", "triton-grpc-port"),
            (f"http://{endpoint}:{DEFAULT_TRITON_HTTP_PORT}", "triton-http-port"),
        ]
    for candidate, _label in candidates:
        if candidate.startswith("http://") or candidate.startswith("https://"):
            try:
                return _detect_http_url(
                    candidate,
                    timeout=timeout,
                    headers=headers,
                    report=report,
                    model_name=model_name,
                    model_version=model_version,
                )
            except TransportProbeError:
                pass
            continue
        host, port = _split_host_port(candidate)
        if has_explicit_port and port in (
            DEFAULT_ENGINE_WS_PORT,
            DEFAULT_TRITON_REALTIME_PORT,
        ):
            detected = _probe_bare_realtime_candidate(
                requested_endpoint=endpoint,
                host=host,
                port=port,
                timeout=timeout,
                connect_timeout=connect_timeout,
                headers=headers,
                report=report,
                model_name=model_name,
                model_version=model_version,
            )
            if detected is not None:
                return detected
        if port == DEFAULT_ENGINE_GRPC_PORT:
            try:
                _probe_engine_grpc(
                    candidate,
                    timeout=timeout,
                    headers=headers,
                    metadata=metadata,
                )
                report.append(
                    {
                        "transport": TRANSPORT_ENGINE_GRPC,
                        "endpoint": candidate,
                        "ok": True,
                    }
                )
                return DetectedTransport(
                    requested_endpoint=endpoint,
                    resolved_endpoint=candidate,
                    transport=TRANSPORT_ENGINE_GRPC,
                    model_name="",
                    model_version=model_version,
                    probe_report=report,
                )
            except ProtocolVersionMismatchError:
                raise
            except Exception as exc:
                report.append(
                    {
                        "transport": TRANSPORT_ENGINE_GRPC,
                        "endpoint": candidate,
                        "ok": False,
                        "reason": str(exc),
                    }
                )
        if port == DEFAULT_TRITON_GRPC_PORT:
            try:
                triton_model = model_name or DEFAULT_TRITON_GRPC_MODEL
                _probe_triton_grpc(
                    candidate,
                    timeout=timeout,
                    model_name=triton_model,
                    headers=headers,
                    metadata=metadata,
                )
                report.append(
                    {
                        "transport": TRANSPORT_TRITON_GRPC,
                        "endpoint": candidate,
                        "ok": True,
                    }
                )
                return DetectedTransport(
                    requested_endpoint=endpoint,
                    resolved_endpoint=candidate,
                    transport=TRANSPORT_TRITON_GRPC,
                    model_name=triton_model,
                    model_version=model_version,
                    probe_report=report,
                )
            except Exception as exc:
                report.append(
                    {
                        "transport": TRANSPORT_TRITON_GRPC,
                        "endpoint": candidate,
                        "ok": False,
                        "reason": str(exc),
                    }
                )
        if port == DEFAULT_ENGINE_WS_PORT:
            ws_url = f"ws://{host}:{port}{DEFAULT_ENGINE_WS_PATH}"
            try:
                _probe_engine_websocket(
                    ws_url,
                    timeout=timeout,
                    connect_timeout=connect_timeout,
                    headers=headers,
                )
                report.append(
                    {
                        "transport": TRANSPORT_ENGINE_WEBSOCKET,
                        "endpoint": ws_url,
                        "ok": True,
                    }
                )
                return DetectedTransport(
                    requested_endpoint=endpoint,
                    resolved_endpoint=ws_url,
                    transport=TRANSPORT_ENGINE_WEBSOCKET,
                    model_name="",
                    model_version=model_version,
                    probe_report=report,
                )
            except ProtocolVersionMismatchError:
                raise
            except Exception as exc:
                report.append(
                    {
                        "transport": TRANSPORT_ENGINE_WEBSOCKET,
                        "endpoint": ws_url,
                        "ok": False,
                        "reason": str(exc),
                    }
                )
        http_fallback = f"http://{host}:{port}"
        try:
            return _detect_http_url(
                http_fallback,
                timeout=timeout,
                headers=headers,
                report=report,
                model_name=model_name,
                model_version=model_version,
            )
        except TransportProbeError:
            pass
    raise TransportProbeError(
        "auto-detect could not resolve a supported transport", probe_report=report
    )


def _probe_bare_realtime_candidate(
    *,
    requested_endpoint: str,
    host: str,
    port: int,
    timeout: float,
    connect_timeout: float | None,
    headers,
    report: list[dict],
    model_name: str | None,
    model_version: str,
) -> DetectedTransport | None:
    realtime_url = f"ws://{host}:{port}{DEFAULT_OPENAI_REALTIME_PATH}"
    try:
        _probe_openai_realtime(
            realtime_url,
            timeout=timeout,
            connect_timeout=connect_timeout,
            headers=headers,
        )
    except Exception as exc:
        report.append(
            {
                "transport": TRANSPORT_OPENAI_REALTIME,
                "endpoint": realtime_url,
                "ok": False,
                "reason": str(exc),
            }
        )
        return None
    report.append(
        {
            "transport": TRANSPORT_OPENAI_REALTIME,
            "endpoint": realtime_url,
            "ok": True,
        }
    )
    return DetectedTransport(
        requested_endpoint=requested_endpoint,
        resolved_endpoint=realtime_url,
        transport=TRANSPORT_OPENAI_REALTIME,
        model_name=model_name or DEFAULT_OPENAI_REALTIME_MODEL,
        model_version=model_version,
        probe_report=report,
    )


def _probe_engine_websocket(
    url: str,
    *,
    timeout: float,
    connect_timeout: float | None = None,
    headers,
) -> None:
    conn = ws_connect(
        url,
        timeout=timeout if connect_timeout is None else connect_timeout,
        headers=headers,
    )
    try:
        ws_send_json(conn, {"type": "get_capabilities"})
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            conn.settimeout(max(0.05, min(0.2, deadline - time.perf_counter())))
            try:
                opcode, payload = ws_recv_frame(conn)
            except socket.timeout:
                continue
            if opcode != 0x1:
                continue
            message = json.loads(payload.decode("utf-8"))
            if message.get("type") == "capabilities":
                check_capabilities_pairing(message.get("capabilities") or {})
                return
        raise TimeoutError("websocket probe timed out")
    finally:
        ws_close(conn)


def _probe_openai_realtime(
    url: str,
    *,
    timeout: float,
    connect_timeout: float | None = None,
    headers,
) -> None:
    conn = ws_connect(
        url,
        timeout=timeout if connect_timeout is None else connect_timeout,
        headers=headers,
    )
    try:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            conn.settimeout(max(0.05, min(0.2, deadline - time.perf_counter())))
            try:
                opcode, payload = ws_recv_frame(conn)
            except socket.timeout:
                continue
            if opcode == 0x8:
                raise RuntimeError("websocket closed before session.created")
            if opcode != 0x1:
                continue
            message = json.loads(payload.decode("utf-8"))
            if message.get("type") == "session.created":
                return
            if message.get("type") == "error":
                error = message.get("error") or {}
                raise RuntimeError(str(error.get("message") or "Realtime probe failed"))
        raise TimeoutError("OpenAI Realtime probe timed out")
    finally:
        ws_close(conn)


def _probe_engine_grpc(
    endpoint: str,
    *,
    timeout: float,
    headers=None,
    metadata=None,
) -> None:
    try:
        import grpc
    except ImportError as exc:
        raise DependencyMissingError(
            "auto-detect for engine-grpc requires the 'grpc' extra"
        ) from exc
    from ._proto import tts_pb2, tts_pb2_grpc

    channel = grpc.insecure_channel(endpoint)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        stub = tts_pb2_grpc.TTSServiceStub(channel)
        response = stub.GetCapabilities(
            tts_pb2.GetCapabilitiesRequest(),
            timeout=timeout,
            metadata=normalize_grpc_metadata(metadata, headers),
        )
        check_capabilities_pairing(
            {
                "protocol_version": getattr(response, "protocol_version", ""),
                "engine_version": getattr(response, "engine_version", ""),
            }
        )
    finally:
        channel.close()


def _probe_triton_grpc(
    endpoint: str,
    *,
    timeout: float,
    model_name: str,
    headers=None,
    metadata=None,
) -> None:
    try:
        import tritonclient.grpc as grpcclient
    except ImportError as exc:
        raise DependencyMissingError(
            "auto-detect for triton-grpc requires the 'triton' extra"
        ) from exc
    client = grpcclient.InferenceServerClient(url=endpoint)
    grpc_headers = grpc_metadata_as_headers(metadata, headers)
    if not client.is_server_live(headers=grpc_headers, client_timeout=timeout):
        raise RuntimeError("server_live=false")
    if not client.is_server_ready(headers=grpc_headers, client_timeout=timeout):
        raise RuntimeError("server_ready=false")
    if model_name and not client.is_model_ready(
        model_name, headers=grpc_headers, client_timeout=timeout
    ):
        raise RuntimeError(f"model_ready[{model_name}]=false")


def _resolve_explicit_endpoint(endpoint: str, transport: str) -> str:
    if transport == TRANSPORT_OPENAI_REALTIME:
        if endpoint.startswith("ws://") or endpoint.startswith("wss://"):
            parsed = urlparse(endpoint)
            if parsed.path in ("", "/"):
                return endpoint.rstrip("/") + DEFAULT_OPENAI_REALTIME_PATH
            return endpoint
        host, port = _split_host_port(endpoint, default_port=DEFAULT_ENGINE_WS_PORT)
        return f"ws://{host}:{port}{DEFAULT_OPENAI_REALTIME_PATH}"
    if transport == TRANSPORT_ENGINE_WEBSOCKET:
        if endpoint.startswith("ws://") or endpoint.startswith("wss://"):
            return endpoint
        host, port = _split_host_port(endpoint, default_port=DEFAULT_ENGINE_WS_PORT)
        return f"ws://{host}:{port}{DEFAULT_ENGINE_WS_PATH}"
    if transport == TRANSPORT_TRITON_HTTP:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint.rstrip("/")
        host, port = _split_host_port(endpoint, default_port=DEFAULT_TRITON_HTTP_PORT)
        return f"http://{host}:{port}"
    if "://" in endpoint:
        parsed = urlparse(endpoint)
        host = parsed.hostname or endpoint
        port = parsed.port or (
            DEFAULT_ENGINE_GRPC_PORT
            if transport == TRANSPORT_ENGINE_GRPC
            else DEFAULT_TRITON_GRPC_PORT
        )
        return f"{host}:{port}"
    return endpoint


def _default_model_for_transport(transport: str) -> str:
    if transport == TRANSPORT_OPENAI_REALTIME:
        return DEFAULT_OPENAI_REALTIME_MODEL
    if transport == TRANSPORT_TRITON_GRPC:
        return DEFAULT_TRITON_GRPC_MODEL
    if transport == TRANSPORT_TRITON_HTTP:
        return DEFAULT_TRITON_HTTP_MODEL
    return ""


def _split_host_port(endpoint: str, default_port: int | None = None) -> tuple[str, int]:
    value = endpoint.strip()
    host, sep, port_text = value.rpartition(":")
    if not sep:
        if default_port is None:
            raise ValueError(f"missing port in endpoint: {endpoint!r}")
        return value, default_port
    return host, int(port_text)
