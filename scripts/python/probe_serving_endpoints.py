#!/usr/bin/env python3
"""Unified availability probe for standalone engine and Triton endpoints.

Checks:
  - standalone engine gRPC: GetCapabilities
  - standalone engine WebSocket: get_capabilities control frame
  - Triton HTTP: live/ready/model-ready
  - Triton gRPC: live/ready/model-ready

This consolidates the repo's scattered readiness checks into one CLI that can
be run after activating the deployment environment:

    eval "$(mamba shell hook --shell zsh)"
    mamba activate qwen3-tts
    python scripts/python/probe_serving_endpoints.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PYTHON = REPO_ROOT / "scripts" / "python"
for import_path in (SCRIPTS_PYTHON, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from qwen3tts_tools.common import (
    DEFAULT_ENGINE_GRPC,
    DEFAULT_ENGINE_WS,
    DEFAULT_PROBE_TARGETS,
    DEFAULT_TRITON_GRPC,
    DEFAULT_TRITON_HTTP,
    DEFAULT_TRITON_HTTP_MODEL,
    DEFAULT_TRITON_MODEL,
    dedupe_keep_order,
    normalize_http_base,
    parse_host_port,
)
from raw_websocket import (
    RawWebSocketError,
    RawWebSocketConnection,
    ws_close,
    ws_connect,
    ws_recv_frame,
    ws_recv_json,
    ws_send_frame,
    ws_send_json,
)

@dataclass
class ProbeResult:
    name: str
    endpoint: str
    ok: bool
    latency_ms: float | None = None
    summary: str = ""
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class ProbeFailure(RuntimeError):
    """Expected endpoint-level probe failure."""


def _timed(name: str, endpoint: str, fn) -> ProbeResult:
    started = time.perf_counter()
    try:
        summary, details = fn()
        elapsed = (time.perf_counter() - started) * 1000.0
        return ProbeResult(
            name=name,
            endpoint=endpoint,
            ok=True,
            latency_ms=elapsed,
            summary=summary,
            details=details,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        error = str(exc).strip() or exc.__class__.__name__
        return ProbeResult(
            name=name,
            endpoint=endpoint,
            ok=False,
            latency_ms=elapsed,
            error=error,
        )


def probe_engine_grpc(endpoint: str, timeout: float) -> ProbeResult:
    def _run():
        import grpc
        from engine.gateway import tts_pb2, tts_pb2_grpc

        try:
            host, port = parse_host_port(endpoint, default_port=50051)
        except ValueError as exc:
            raise ProbeFailure(str(exc)) from exc
        channel = grpc.insecure_channel(f"{host}:{port}")
        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
            stub = tts_pb2_grpc.TTSServiceStub(channel)
            response = stub.GetCapabilities(tts_pb2.GetCapabilitiesRequest(), timeout=timeout)
        finally:
            channel.close()

        details = {
            "variant": response.variant,
            "loaded_model_type": response.loaded_model_type,
            "supported_task_types": list(response.declared_supported_task_types),
            "audio_formats": [
                {
                    "encoding": fmt.encoding,
                    "sample_rate": fmt.sample_rate,
                    "channels": fmt.channels,
                }
                for fmt in response.supported_audio_formats
            ],
        }
        summary = (
            f"variant={response.variant or '<unknown>'}, "
            f"model_type={response.loaded_model_type or '<unknown>'}"
        )
        return summary, details

    return _timed("engine-grpc", endpoint, _run)


def probe_engine_websocket(url: str, timeout: float) -> ProbeResult:
    def _run():
        parsed = urlparse(url)
        if parsed.scheme not in {"ws", "wss"}:
            raise ProbeFailure(f"unsupported websocket scheme: {parsed.scheme or '<empty>'}")
        conn = ws_connect(url, timeout=timeout)
        try:
            ws_send_json(conn, {"type": "get_capabilities"})
            message = ws_recv_json(conn)
        finally:
            ws_close(conn)

        if not isinstance(message, dict):
            raise ProbeFailure("websocket response is not a JSON object")
        if message.get("type") != "capabilities":
            raise ProbeFailure(f"unexpected websocket response type: {message.get('type')!r}")

        capabilities = message.get("capabilities", {})
        if not isinstance(capabilities, dict):
            raise ProbeFailure("websocket capabilities payload is not an object")
        summary = (
            f"variant={capabilities.get('variant', '<unknown>')}, "
            f"model_type={capabilities.get('loaded_model_type', '<unknown>')}"
        )
        return summary, capabilities

    return _timed("engine-websocket", url, _run)


def probe_triton_http(base_url: str, model_name: str, timeout: float) -> ProbeResult:
    def _run():
        root = normalize_http_base(base_url)
        live = requests.get(f"{root}/v2/health/live", timeout=timeout)
        if live.status_code != 200:
            raise ProbeFailure(f"live check failed: HTTP {live.status_code}")

        ready = requests.get(f"{root}/v2/health/ready", timeout=timeout)
        if ready.status_code != 200:
            raise ProbeFailure(f"ready check failed: HTTP {ready.status_code}")

        model_ready = None
        if model_name:
            model_resp = requests.get(f"{root}/v2/models/{model_name}/ready", timeout=timeout)
            if model_resp.status_code != 200:
                raise ProbeFailure(
                    f"model-ready check failed for {model_name}: HTTP {model_resp.status_code}"
                )
            model_ready = True

        details = {
            "server_live": True,
            "server_ready": True,
            "model_name": model_name,
            "model_ready": model_ready,
        }
        summary = "server_live=true, server_ready=true"
        if model_name:
            summary += f", model_ready[{model_name}]=true"
        return summary, details

    return _timed("triton-http", base_url, _run)


def probe_triton_grpc(endpoint: str, model_name: str, timeout: float) -> ProbeResult:
    def _run():
        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient(url=endpoint)
        server_live = bool(client.is_server_live())
        server_ready = bool(client.is_server_ready())
        if not server_live:
            raise ProbeFailure("gRPC server_live=false")
        if not server_ready:
            raise ProbeFailure("gRPC server_ready=false")

        model_ready = None
        if model_name:
            model_ready = bool(client.is_model_ready(model_name))
            if not model_ready:
                raise ProbeFailure(f"gRPC model_ready[{model_name}]=false")

        details = {
            "server_live": server_live,
            "server_ready": server_ready,
            "model_name": model_name,
            "model_ready": model_ready,
        }
        summary = "server_live=true, server_ready=true"
        if model_name:
            summary += f", model_ready[{model_name}]=true"
        return summary, details

    return _timed("triton-grpc", endpoint, _run)


def _selected_probes(only: list[str] | None) -> list[str]:
    if not only:
        return list(DEFAULT_PROBE_TARGETS)
    return dedupe_keep_order(only)


def _render_text(results: list[ProbeResult]) -> str:
    lines = []
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        latency = f"{result.latency_ms:.1f}ms" if result.latency_ms is not None else "-"
        if result.ok:
            lines.append(
                f"[{status}] {result.name:16} {result.endpoint:30} {latency:>8}  {result.summary}"
            )
        else:
            lines.append(
                f"[{status}] {result.name:16} {result.endpoint:30} {latency:>8}  {result.error}"
            )
    ok_count = sum(1 for result in results if result.ok)
    lines.append(f"\nSummary: {ok_count}/{len(results)} probes passed")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified availability probe for standalone engine and Triton interfaces."
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=["engine-grpc", "engine-websocket", "triton-http", "triton-grpc"],
        help="Run only the selected probe(s). Repeatable.",
    )
    parser.add_argument("--engine-grpc", default=DEFAULT_ENGINE_GRPC)
    parser.add_argument("--engine-websocket", default=DEFAULT_ENGINE_WS)
    parser.add_argument("--triton-http", default=DEFAULT_TRITON_HTTP)
    parser.add_argument("--triton-grpc", default=DEFAULT_TRITON_GRPC)
    parser.add_argument("--triton-model", default=DEFAULT_TRITON_MODEL)
    parser.add_argument("--triton-http-model", default=DEFAULT_TRITON_HTTP_MODEL)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = _selected_probes(args.only)

    runners = {
        "engine-grpc": lambda: probe_engine_grpc(args.engine_grpc, args.timeout),
        "engine-websocket": lambda: probe_engine_websocket(args.engine_websocket, args.timeout),
        "triton-http": lambda: probe_triton_http(
            args.triton_http, args.triton_http_model, args.timeout,
        ),
        "triton-grpc": lambda: probe_triton_grpc(args.triton_grpc, args.triton_model, args.timeout),
    }

    results = [runners[name]() for name in selected]
    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        print(_render_text(results))
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
