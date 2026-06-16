from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from typing import Any

from aiohttp import WSMsgType, web

from .audio_assets import attach_audio_to_result
from .audio_store import AudioStore
from . import llm_pk
from .jobs import ConcurrencyJobManager
from qwen3_tts_protocol.schemas import TraceEvent
from .trace_store import TraceStore
from qwen3_tts_protocol.triton_types import TtsRequest
from .triton_client import probe_ready, stream_once


DEFAULT_TEXT = "你好，今天天气不错，我们来聊聊最近你看过的书，有没有什么推荐的？"
DEFAULT_SPEAKER = os.environ.get("QWEN_DEMO_DEFAULT_SPEAKER", "Serena")
DEFAULT_LANGUAGE = os.environ.get("QWEN_DEMO_DEFAULT_LANGUAGE", "auto")
DEFAULT_MS_PER_TOKEN = float(os.environ.get("QWEN_DEMO_DEFAULT_MS_PER_TOKEN", "30"))
TRITON_GRPC = os.environ.get("QWEN_DEMO_TRITON_GRPC", "localhost:8001")
TRITON_MODEL = os.environ.get("QWEN_DEMO_TRITON_MODEL", "tts_orchestrator")
TRITON_MAX_BATCH_SLOTS = int(os.environ.get("QWEN_DEMO_TRITON_MAX_BATCH_SLOTS", os.environ.get("TRITON_MAX_BATCH_SLOTS", "128")))
TRITON_MAX_SESSIONS = int(os.environ.get("QWEN_DEMO_TRITON_MAX_SESSIONS", os.environ.get("TRITON_MAX_SESSIONS", "128")))

RELEASE_METADATA = {
    "stage": "engineering_preview",
    "positioning": "Engineering preview: demonstrates the Qwen3-TTS TensorRT/token-streaming optimization path and does not promise production stability.",
    "recommended_variant": "custom-1.7b",
    "stable_paths": ["custom_voice"],
    "experimental_paths": ["voice_design"],
    "planned_paths": ["base_voice_clone", "icl_voice_clone"],
}

STREAMING_LIMITATIONS = [
    "The current stable open-source scope is focused on the custom-1.7b/custom_voice path.",
    "Streaming mode may still hallucinate, repeat, skip, or insert text that was not provided, especially on longer inputs.",
    "The 13ms TTFT figure is only the lowest observed value under specific hardware, a warm engine, a cache hit, and a single request.",
    "The upstream token rate in LLM PK is client-side simulation, intended only to demonstrate the perceived difference between streaming and offline TTS.",
]


@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = os.environ.get("QWEN_DEMO_CORS_ORIGIN", "*")
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "content-type"
    return response


def create_app() -> web.Application:
    store = TraceStore()
    audio_store = AudioStore()
    jobs = ConcurrencyJobManager(
        enable_live=os.environ.get("QWEN_DEMO_ENABLE_LIVE_CONCURRENCY", "1").lower()
        in {"1", "true", "yes", "on"},
        audio_store=audio_store,
        live_slot_limit=TRITON_MAX_BATCH_SLOTS,
    )
    app = web.Application(middlewares=[cors_middleware])
    app["trace_store"] = store
    app["jobs"] = jobs
    app["audio_store"] = audio_store
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/api/v1/capabilities", handle_capabilities)
    app.router.add_get("/api/v1/audio/{audio_id}", handle_audio)
    app.router.add_post("/api/v1/llm-pk", handle_llm_pk)
    app.router.add_get("/api/v1/trt-live", handle_trt_live)
    app.router.add_post("/api/v1/concurrency", handle_concurrency_start)
    app.router.add_get("/api/v1/concurrency/{job_id}", handle_concurrency_ws)
    return app


async def handle_healthz(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def handle_capabilities(request: web.Request) -> web.Response:
    triton_ready = False
    triton_error = ""
    try:
        triton_ready = await asyncio.to_thread(probe_ready, TRITON_GRPC, TRITON_MODEL)
    except Exception as exc:
        triton_error = str(exc)

    store: TraceStore = request.app["trace_store"]
    jobs: ConcurrencyJobManager = request.app["jobs"]
    default_request = store.load_default_request()
    return web.json_response(
        {
            "default_request": {
                "text": default_request.get("text") or DEFAULT_TEXT,
                "speaker": default_request.get("speaker") or DEFAULT_SPEAKER,
                "language": default_request.get("language") or DEFAULT_LANGUAGE,
                "ms_per_token": float(default_request.get("ms_per_token") or DEFAULT_MS_PER_TOKEN),
            },
            "backends": [
                {
                    "id": "triton_streaming",
                    "label": "Triton Streaming TTS (token-by-token)",
                    "streaming": True,
                    "live_available": triton_ready,
                    "endpoint": TRITON_GRPC,
                    "model": TRITON_MODEL,
                },
                {
                    "id": "triton_offline",
                    "label": "Triton Offline TTS (wait for full text)",
                    "streaming": False,
                    "live_available": triton_ready,
                    "endpoint": TRITON_GRPC,
                    "model": TRITON_MODEL,
                },
                {
                    "id": "triton_trt_streaming",
                    "label": "Triton TRT Streaming",
                    "streaming": True,
                    "live_available": triton_ready,
                    "endpoint": TRITON_GRPC,
                    "model": TRITON_MODEL,
                    "error": triton_error,
                    "runtime": {
                        "max_batch_slots": TRITON_MAX_BATCH_SLOTS,
                        "max_sessions": TRITON_MAX_SESSIONS,
                    },
                },
            ],
            "concurrency": {
                "live_enabled": jobs.enable_live,
                "triton_active_slot_limit": TRITON_MAX_BATCH_SLOTS,
                "triton_max_sessions": TRITON_MAX_SESSIONS,
            },
            "release": RELEASE_METADATA,
            "limitations": STREAMING_LIMITATIONS,
            "headline": {
                "single_stream_cache_hit_ttft_ms": 13,
                "concurrent_128_avg_ttft_ms": 180,
            },
        }
    )


async def handle_llm_pk(request: web.Request) -> web.Response:
    body = await _read_json(request)
    audio_store: AudioStore = request.app["audio_store"]

    text = str(body.get("text") or DEFAULT_TEXT)
    speaker = str(body.get("speaker") or DEFAULT_SPEAKER)
    language = str(body.get("language") or DEFAULT_LANGUAGE)
    ms_per_token = float(body.get("ms_per_token") or DEFAULT_MS_PER_TOKEN)
    timeout_sec = float(body.get("timeout_sec") or 120.0)

    request_payload = {
        "text": text,
        "speaker": speaker,
        "language": language,
        "ms_per_token": ms_per_token,
    }

    results: list[dict[str, Any]] = []
    warnings: list[str] = []

    for mode_label, runner in (
        ("streaming", llm_pk.run_streaming),
        ("offline", llm_pk.run_offline),
    ):
        try:
            run_result = await runner(
                text=text,
                ms_per_token=ms_per_token,
                speaker=speaker,
                language=language,
                endpoint=TRITON_GRPC,
                model_name=TRITON_MODEL,
                timeout_sec=timeout_sec,
            )
            attach_audio_to_result(run_result, audio_store)
            results.append(run_result.to_dict())
            warnings.extend(run_result.warnings)
        except llm_pk.LlmPkError as exc:
            warnings.append(f"{mode_label} run unavailable: {exc}")

    return web.json_response(
        {
            "type": "llm_pk_result",
            "request": request_payload,
            "results": results,
            "warnings": warnings,
            "release": RELEASE_METADATA,
            "limitations": STREAMING_LIMITATIONS,
        }
    )


async def handle_audio(request: web.Request) -> web.StreamResponse:
    audio_store: AudioStore = request.app["audio_store"]
    path = audio_store.path_for(request.match_info["audio_id"])
    if path is None:
        raise web.HTTPNotFound(text="audio not found")
    return web.FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})


async def handle_trt_live(request: web.Request) -> web.StreamResponse:
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    async for message in ws:
        if message.type == WSMsgType.TEXT:
            try:
                payload = json.loads(message.data)
            except json.JSONDecodeError as exc:
                await ws.send_json({"type": "error", "message": f"invalid JSON: {exc}"})
                continue
            if payload.get("type") != "speak":
                await ws.send_json({"type": "error", "message": "expected message type 'speak'"})
                continue
            await _stream_trt_live(ws, payload)
        elif message.type == WSMsgType.ERROR:
            break
    return ws


async def _stream_trt_live(ws: web.WebSocketResponse, payload: dict[str, Any]) -> None:
    run_id = f"trt-live-{uuid.uuid4().hex[:10]}"
    started = time.perf_counter()
    await ws.send_json(
        {
            "type": "event",
            "event": TraceEvent(
                run_id=run_id,
                backend="triton_trt_streaming",
                type="request_started",
                t_ms=0.0,
                meta={"source": "live_triton"},
            ).to_dict(),
        }
    )
    try:
        async for item in stream_once(
            _tts_request_from_body(payload),
            run_id=run_id,
            endpoint=TRITON_GRPC,
            model_name=TRITON_MODEL,
            timeout_sec=float(payload.get("timeout_sec") or 60.0),
        ):
            if item["kind"] == "event":
                await ws.send_json({"type": "event", "event": item["event"].to_dict()})
            elif item["kind"] == "audio":
                await ws.send_bytes(item["audio"])
        return
    except Exception as exc:
        await ws.send_json(
            {
                "type": "event",
                "event": TraceEvent(
                    run_id=run_id,
                    backend="triton_trt_streaming",
                    type="warning",
                    t_ms=(time.perf_counter() - started) * 1000.0,
                    meta={
                        "source": "live_triton",
                        "message": f"live TRT stream unavailable; no synthetic audio emitted: {exc}",
                    },
                ).to_dict(),
            }
        )
        await ws.send_json({"type": "error", "message": str(exc)})


async def handle_concurrency_start(request: web.Request) -> web.Response:
    body = await _read_json(request)
    jobs: ConcurrencyJobManager = request.app["jobs"]
    job = await jobs.create_job(
        {
            "text": body.get("text") or DEFAULT_TEXT,
            "speaker": body.get("speaker") or DEFAULT_SPEAKER,
            "language": body.get("language") or DEFAULT_LANGUAGE,
            "cache_mode": body.get("cache_mode") or "hit",
            "concurrency": int(body.get("concurrency") or 128),
            "live": bool(body.get("live")),
            "timeout_sec": float(body.get("timeout_sec") or 60.0),
        }
    )
    return web.json_response({"job_id": job.job_id})


async def handle_concurrency_ws(request: web.Request) -> web.StreamResponse:
    jobs: ConcurrencyJobManager = request.app["jobs"]
    job = jobs.get(request.match_info["job_id"])
    if job is None:
        return web.json_response({"error": "unknown job_id"}, status=404)
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    queue = await jobs.subscribe(job)
    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "heartbeat"})
                continue
            await ws.send_json(message)
            if message.get("type") == "summary":
                break
    finally:
        jobs.unsubscribe(job, queue)
        await ws.close()
    return ws


async def _read_json(request: web.Request) -> dict[str, Any]:
    if request.content_length == 0:
        return {}
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="JSON body must be an object")
    return payload


def _tts_request_from_body(body: dict[str, Any]) -> TtsRequest:
    return TtsRequest(
        text=str(body.get("text") or DEFAULT_TEXT),
        speaker=str(body.get("speaker") or DEFAULT_SPEAKER),
        language=str(body.get("language") or DEFAULT_LANGUAGE),
        cache_mode=str(body.get("cache_mode") or "hit"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS Triton WebUI demo API")
    parser.add_argument("--host", default=os.environ.get("QWEN_DEMO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QWEN_DEMO_PORT", "7860")))
    args = parser.parse_args()
    web.run_app(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
