"""OpenAI Realtime WebSocket sidecar backed by Triton streaming gRPC."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import threading
from pathlib import Path
from typing import Any

from ..config import resolve_model_package_paths
from ..session import SessionService
from .openai_realtime import (
    OPENAI_REALTIME_PATH,
    OPENAI_REALTIME_PROTOCOL,
    QWEN_REALTIME_EXTENSION_PROTOCOL,
    QWEN_TEXT_BUFFER_EXTENSION,
    OpenAIRealtimeGateway,
)
from .triton_realtime_backend import TritonRealtimeBackend
from .session_backend import TritonSessionBackend
from .native_session_gateway import (
    NATIVE_WEBSOCKET_PATH,
    NATIVE_WEBSOCKET_PROTOCOL,
    NativeSessionGateway,
)

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - deployment dependency
    web = None


logger = logging.getLogger(__name__)


class JsonlUsageRecorder:
    """Append one complete billing record per line to a local ledger."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    async def __call__(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def create_app(
    backend: TritonRealtimeBackend,
    *,
    usage_recorder: Any = None,
):
    """Create the testable sidecar application without binding a socket."""

    if web is None:
        raise RuntimeError("Triton Realtime gateway requires aiohttp")
    # The sidecar uses the same typed logical-session contract as standalone.
    # Triton-specific streaming frames stop at TritonSessionBackend.
    service = SessionService(TritonSessionBackend(backend))
    gateway = OpenAIRealtimeGateway(
        None,
        session_service=service,
        usage_recorder=usage_recorder,
    )
    def capabilities_payload() -> dict[str, Any]:
        return {
            # Keep the historical flat field stable for one compatibility
            # release.  New clients use the endpoint-scoped ``protocols`` map
            # below, which now advertises both first-class wire adapters.
            "supported_api_protocols": [OPENAI_REALTIME_PROTOCOL],
            "openai_realtime_path": OPENAI_REALTIME_PATH,
            "native_websocket_path": NATIVE_WEBSOCKET_PATH,
            "supported_realtime_extensions": [
                QWEN_TEXT_BUFFER_EXTENSION,
                "qwen.text_progress.v1",
            ],
            "supported_progress_features": [
                "text_progress_anchor_v1",
                "playback_progress_v1",
                "qwen.text_progress.v1",
            ],
            "backend": "triton-grpc",
            "model": backend.model_name,
            "model_version": backend.model_version,
            "usage": {
                "response_field": "response.done.response.usage",
                "input": "model_tokenizer",
                "output_audio_token_ms": 50,
            },
            "protocols": {
                "native_websocket": {
                    "path": NATIVE_WEBSOCKET_PATH,
                    "current": NATIVE_WEBSOCKET_PROTOCOL,
                    "supported": [NATIVE_WEBSOCKET_PROTOCOL],
                    "features": ["persistent_sessions_v1"],
                    "audio_formats": ["pcm_f32", "pcm_s16le"],
                },
                "openai_realtime": {
                    "path": OPENAI_REALTIME_PATH,
                    "base": OPENAI_REALTIME_PROTOCOL,
                    "supported_extensions": [
                        QWEN_TEXT_BUFFER_EXTENSION,
                        "qwen.text_progress.v1",
                    ],
                    "extension_protocol": QWEN_REALTIME_EXTENSION_PROTOCOL,
                    "features": [
                        "base64_pcm16",
                        "full_duplex",
                        "serial_responses",
                    ],
                    "audio_formats": ["pcm_s16le"],
                },
            },
        }

    native_gateway = NativeSessionGateway(
        service,
        capabilities=capabilities_payload,
    )
    app = web.Application()

    async def health(_request):
        ready = await backend.is_ready()
        return web.json_response(
            {
                "status": "ready" if ready else "unavailable",
                "running": ready,
                "backend": "triton",
                "model": backend.model_name,
            },
            status=200 if ready else 503,
        )

    async def capabilities(_request):
        return web.json_response(capabilities_payload())

    async def cleanup(_app):
        await gateway.close()

    app.router.add_get(OPENAI_REALTIME_PATH, gateway.handle_websocket)
    app.router.add_get(NATIVE_WEBSOCKET_PATH, native_gateway.handle_websocket)
    app.router.add_get("/v1/capabilities", capabilities)
    app.router.add_get("/health", health)
    app.on_cleanup.append(cleanup)
    return app


def _parser() -> argparse.ArgumentParser:
    model_version = os.environ.get("TRITON_MODEL_VERSION", "1")
    package_dir = os.environ.get(
        "TRITON_MODEL_PACKAGE_DIR", f"/models/tts_orchestrator/{model_version}"
    )
    parser = argparse.ArgumentParser(
        description="OpenAI Realtime sidecar for the Triton TTS orchestrator"
    )
    parser.add_argument(
        "--host", default=os.environ.get("TRITON_REALTIME_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("TRITON_REALTIME_PORT", "50052"))
    )
    parser.add_argument(
        "--triton-endpoint",
        default=os.environ.get("TRITON_GRPC_ENDPOINT", "localhost:8001"),
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("TRITON_MODEL_NAME", "tts_orchestrator"),
    )
    parser.add_argument("--model-version", default=model_version)
    parser.add_argument("--model-package-dir", default=package_dir)
    parser.add_argument(
        "--tokenizer-dir", default=os.environ.get("TRITON_TOKENIZER_DIR", "")
    )
    parser.add_argument(
        "--usage-log", default=os.environ.get("TRITON_REALTIME_USAGE_LOG", "")
    )
    return parser


def _triton_headers() -> dict[str, str]:
    token = os.environ.get("TRITON_GRPC_AUTH_TOKEN", "").strip()
    if not token:
        return {}
    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"authorization": token}


async def _serve(args: argparse.Namespace) -> None:
    package_paths = resolve_model_package_paths(args.model_package_dir)
    tokenizer_dir = args.tokenizer_dir or package_paths.tokenizer_dir
    backend = TritonRealtimeBackend(
        args.triton_endpoint,
        model_name=args.model_name,
        model_version=args.model_version,
        tokenizer_dir=tokenizer_dir,
        headers=_triton_headers(),
    )
    usage_recorder = JsonlUsageRecorder(args.usage_log) if args.usage_log else None
    app = create_app(backend, usage_recorder=usage_recorder)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except NotImplementedError:  # pragma: no cover - Windows
            pass
    try:
        await site.start()
        logger.info(
            "OpenAI Realtime Triton sidecar listening on %s:%d%s -> %s/%s:%s",
            args.host,
            args.port,
            OPENAI_REALTIME_PATH,
            args.triton_endpoint,
            args.model_name,
            args.model_version or "latest",
        )
        await stop_event.wait()
    finally:
        await runner.cleanup()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parser().parse_args(argv)
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


__all__ = ["JsonlUsageRecorder", "create_app", "main"]
