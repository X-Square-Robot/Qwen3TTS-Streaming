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

from ..config import load_model_manifest, resolve_model_package_paths
from ..runtime.release_gate import ReleaseCapability, ReleaseGate, evaluate_release_gate
from ..core.speech_state_bundle import validate_speech_state_bundle
from ..distribution.sdk import mount_sdk_routes
from ..distribution.site import mount_demo_config_route
from ..session import ResumableSessionRegistry, SessionService
from .capabilities import RuntimeType, build_gateway_capabilities
from .openai_realtime import (
    OPENAI_REALTIME_PATH,
    OpenAIRealtimeGateway,
)
from .triton_realtime_backend import TritonRealtimeBackend
from .session_backend import TritonSessionBackend
from .native_session_gateway import (
    NATIVE_WEBSOCKET_PATH,
    NativeSessionGateway,
)

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - deployment dependency
    web = None


logger = logging.getLogger(__name__)


def _runtime_capabilities_from_package(
    package_paths, tokenizer_dir: str
) -> dict[str, Any]:
    """Describe only features proven by the mounted model package."""

    arch = load_model_manifest(package_paths.engine_dir, tokenizer_dir=tokenizer_dir)
    manifest: dict[str, Any] = {}
    try:
        loaded_manifest = json.loads(
            Path(package_paths.manifest_path).read_text(encoding="utf-8")
        )
        if isinstance(loaded_manifest, dict):
            manifest = loaded_manifest
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read runtime manifest for release gating: %s", exc)
    evidence = None
    bundle_root = Path(package_paths.package_dir)
    for evidence_path in (
        bundle_root / "capability_evidence.json",
        Path(package_paths.engine_dir) / "capability_evidence.json",
    ):
        if not evidence_path.is_file():
            continue
        try:
            loaded_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_evidence = None
        if isinstance(loaded_evidence, dict):
            evidence = loaded_evidence
            break
    release_gate = evaluate_release_gate(manifest, evidence)
    bundle_validation = validate_speech_state_bundle(
        manifest,
        bundle_root=bundle_root,
        runtime_artifact_path=Path(package_paths.runtime_artifact_path),
    )
    weights_config: dict[str, Any] = {}
    weights_config_path = Path(package_paths.weights_dir) / "config.json"
    try:
        loaded_config = json.loads(weights_config_path.read_text(encoding="utf-8"))
        if isinstance(loaded_config, dict):
            weights_config = loaded_config
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not read capability metadata from %s: %s",
            weights_config_path,
            exc,
        )
    speaker_ids = weights_config.get("spk_id")
    language_ids = weights_config.get("codec_language_id")
    supported_speakers = (
        sorted(str(name).strip().lower() for name in speaker_ids if str(name).strip())
        if isinstance(speaker_ids, dict)
        else []
    )
    supported_languages = (
        [
            "auto",
            *sorted(
                str(name).strip().lower()
                for name in language_ids
                if str(name).strip() and str(name).strip().lower() != "auto"
            ),
        ]
        if isinstance(language_ids, dict)
        else []
    )
    loaded_model_type = str(arch.tts_model_type or "").strip()
    if not loaded_model_type or loaded_model_type == "unknown":
        tasks = [task for task in arch.supported_task_types if task]
        loaded_model_type = tasks[0] if len(tasks) == 1 else "unknown"
    profile = arch.engine_profile
    native_cursor = dict(getattr(arch, "native_cursor", {}) or {})
    if native_cursor.get("enabled") is not True:
        native_cursor["enabled"] = False
        native_cursor["progress_available"] = False
        native_cursor["reason"] = "malformed_native_cursor_capability"
    elif native_cursor.get("progress_available") is not True:
        if "progress_available" in native_cursor and not isinstance(
            native_cursor["progress_available"], bool
        ):
            native_cursor["reason"] = "malformed_native_cursor_capability"
        native_cursor["progress_available"] = False
    elif not release_gate.verified(ReleaseCapability.NATIVE_CURSOR):
        native_cursor["progress_available"] = False
        native_cursor["reason"] = release_gate.reason(ReleaseCapability.NATIVE_CURSOR)
    result = {
        "variant": arch.variant,
        "loaded_model_type": loaded_model_type,
        "native_cursor": native_cursor,
        "engine_version": os.environ.get("ENGINE_VERSION", "").strip(),
        "declared_supported_task_types": list(arch.supported_task_types),
        "supported_input_modes": ["token", "clause", "long_segment", "full_text"],
        "supported_group_policies": ["none", "auto"],
        "supported_speakers": supported_speakers,
        "supported_languages": supported_languages,
        "supported_vad_strategies": ["disabled", "energy"],
        "supported_audio_formats": [
            {"encoding": "pcm_f32", "sample_rate": 24000, "channels": 1},
            {"encoding": "pcm_f32", "sample_rate": 16000, "channels": 1},
            {"encoding": "pcm_s16le", "sample_rate": 24000, "channels": 1},
            {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        ],
        # Reference support depends on model parameters and optional runtime
        # components that the sidecar cannot inspect safely. Keep it disabled
        # until Triton publishes an affirmative capability contract.
        "ref_audio_available": False,
        "ref_audio_reason": "Triton model did not advertise reference support",
        "engine_profile": {
            "max_batch_size": profile.max_batch_size,
            "max_input_len": profile.max_input_len,
            "max_seq_len": profile.max_seq_len,
            "engine_dtype": profile.engine_dtype,
            "triton_io_float_dtype": profile.triton_io_float_dtype,
        },
    }
    if not isinstance(manifest.get("speech_state"), dict):
        speech_state_reason = bundle_validation.reason
    elif not bundle_validation.verified:
        speech_state_reason = bundle_validation.reason
    else:
        speech_state_reason = release_gate.reason(ReleaseCapability.SPEECH_STATE)
    result["speech_state"] = {
        "supported": (
            bundle_validation.verified
            and release_gate.verified(ReleaseCapability.SPEECH_STATE)
        ),
        "reason": speech_state_reason,
    }
    return result


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
    sdk_dir: str | os.PathLike[str] | None = None,
    runtime_capabilities: dict[str, Any] | None = None,
):
    """Create the testable sidecar application without binding a socket."""

    if web is None:
        raise RuntimeError("Triton Realtime gateway requires aiohttp")
    # The sidecar uses the same typed logical-session contract as standalone.
    # Triton-specific streaming frames stop at TritonSessionBackend.
    service = SessionService(TritonSessionBackend(backend))
    resume_registry = ResumableSessionRegistry(service)
    gateway = OpenAIRealtimeGateway(
        None,
        session_service=service,
        resume_registry=resume_registry,
        usage_recorder=usage_recorder,
    )

    def capabilities_payload() -> dict[str, Any]:
        capabilities = build_gateway_capabilities(
            runtime_capabilities,
            runtime_type=RuntimeType.TRITON,
            backend="triton-grpc",
            native_path=NATIVE_WEBSOCKET_PATH,
            resume_grace_ms=int(resume_registry.grace_seconds * 1000),
            resume_max_buffer_bytes=resume_registry.max_buffer_bytes,
            model_name=backend.model_name,
            model_version=backend.model_version,
        )
        capabilities["usage"] = {
            "response_field": "response.done.response.usage",
            "input": "model_tokenizer",
            "output_audio_token_ms": 50,
        }
        return capabilities

    native_gateway = NativeSessionGateway(
        service,
        capabilities=capabilities_payload,
        resume_registry=resume_registry,
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
        await native_gateway.close()
        await gateway.close()

    app.router.add_get(OPENAI_REALTIME_PATH, gateway.handle_websocket)
    app.router.add_get(NATIVE_WEBSOCKET_PATH, native_gateway.handle_websocket)
    app.router.add_get("/v1/capabilities", capabilities)
    app.router.add_get("/health", health)
    sdk_distribution = mount_sdk_routes(app, sdk_dir=sdk_dir)
    mount_demo_config_route(
        app,
        runtime_type=RuntimeType.TRITON.value,
        capabilities_provider=capabilities_payload,
        sdk_distribution=sdk_distribution,
    )
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
    runtime_capabilities = _runtime_capabilities_from_package(
        package_paths, tokenizer_dir
    )
    app = create_app(
        backend,
        usage_recorder=usage_recorder,
        runtime_capabilities=runtime_capabilities,
    )
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
