from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from qwen3tts_protocol.schemas import RunMetrics, RunResult, TraceEvent
from qwen3tts_protocol.triton_types import TtsRequest, build_action_payload
from qwen3tts_protocol.audio import decode_obj
from .triton_client import (
    DEFAULT_TRITON_GRPC,
    DEFAULT_TRITON_MODEL,
    TritonUnavailable,
    _load_triton_client,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_DIR = os.environ.get(
    "QWEN_DEMO_TOKENIZER_DIR",
    str(REPO_ROOT / "workspace" / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
)


class LlmPkError(RuntimeError):
    pass


_TOKENIZER_CACHE: Any | None = None


def _get_tokenizer() -> Any:
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is not None:
        return _TOKENIZER_CACHE
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer

    _TOKENIZER_CACHE = LightQwen3TTSTokenizer(DEFAULT_TOKENIZER_DIR)
    return _TOKENIZER_CACHE


def _tokenize_for_simulation(text: str) -> list[tuple[int, str]]:
    tokenizer = _get_tokenizer()
    ids, fragments = tokenizer.encode_with_text(text, add_special_tokens=False)
    return list(zip(ids, fragments))


async def run_streaming(
    *,
    text: str,
    ms_per_token: float,
    speaker: str,
    language: str,
    endpoint: str = DEFAULT_TRITON_GRPC,
    model_name: str = DEFAULT_TRITON_MODEL,
    timeout_sec: float = 120.0,
) -> RunResult:
    return await _run_pk(
        mode="streaming",
        text=text,
        ms_per_token=ms_per_token,
        speaker=speaker,
        language=language,
        endpoint=endpoint,
        model_name=model_name,
        timeout_sec=timeout_sec,
    )


async def run_offline(
    *,
    text: str,
    ms_per_token: float,
    speaker: str,
    language: str,
    endpoint: str = DEFAULT_TRITON_GRPC,
    model_name: str = DEFAULT_TRITON_MODEL,
    timeout_sec: float = 120.0,
) -> RunResult:
    return await _run_pk(
        mode="offline",
        text=text,
        ms_per_token=ms_per_token,
        speaker=speaker,
        language=language,
        endpoint=endpoint,
        model_name=model_name,
        timeout_sec=timeout_sec,
    )


async def _run_pk(
    *,
    mode: str,
    text: str,
    ms_per_token: float,
    speaker: str,
    language: str,
    endpoint: str,
    model_name: str,
    timeout_sec: float,
) -> RunResult:
    if mode not in {"streaming", "offline"}:
        raise ValueError(f"unsupported mode: {mode!r}")
    if not text.strip():
        raise LlmPkError("text must be non-empty")
    if ms_per_token <= 0:
        raise LlmPkError("ms_per_token must be positive")

    tokens = _tokenize_for_simulation(text)
    if not tokens:
        raise LlmPkError("tokenizer produced zero tokens for input")

    backend = "triton_streaming" if mode == "streaming" else "triton_offline"
    label = (
        "Triton Streaming TTS (token-by-token)"
        if mode == "streaming"
        else "Triton Offline TTS (wait for full text)"
    )
    description = (
        "Triton orchestrator init(token mode) + append_text + text_complete; engine starts on first token"
        if mode == "streaming"
        else "Triton orchestrator synthesize; engine waits for full text before generating"
    )

    run_id = f"{backend}-{uuid.uuid4().hex[:10]}"
    session_id = run_id
    interval = ms_per_token / 1000.0

    tts_request = TtsRequest(
        text=text,
        speaker=speaker,
        language=language,
        input_mode="token" if mode == "streaming" else None,
        group_policy="none" if mode == "streaming" else None,
    )

    grpcclient = _load_triton_client()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    started = time.perf_counter()
    audio_format: dict[str, Any] = {
        "encoding": tts_request.audio_encoding,
        "sample_rate": tts_request.sample_rate,
        "channels": 1,
    }
    first_audio_ms: float | None = None
    triton_adapter_ttft_ms: float | None = None
    audio_parts: list[bytes] = []
    chunks = 0

    events: list[TraceEvent] = [
        TraceEvent(
            run_id=run_id,
            backend=backend,
            type="request_started",
            t_ms=0.0,
            meta={"mode": mode, "ms_per_token": ms_per_token, "tokens": len(tokens)},
        )
    ]

    def emit(item: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def callback(result, error) -> None:  # pragma: no cover - exercised against live Triton
        now_ms = (time.perf_counter() - started) * 1000.0
        if error:
            emit({"kind": "error", "message": str(error), "t_ms": now_ms})
            return
        try:
            event_type = ""
            payload: dict[str, Any] = {}
            audio_bytes_value: bytes = b""
            is_final = False

            event_type_arr = result.as_numpy("event_type")
            event_json_arr = result.as_numpy("event_json")
            audio_arr = result.as_numpy("audio_chunk")
            is_final_arr = result.as_numpy("is_final")

            if event_type_arr is not None and event_type_arr.size:
                event_type = decode_obj(event_type_arr.flatten()[0])
            if event_json_arr is not None and event_json_arr.size:
                raw = decode_obj(event_json_arr.flatten()[0])
                if raw:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        payload = parsed
            if audio_arr is not None and audio_arr.size:
                raw_audio = audio_arr.flatten()[0]
                if isinstance(raw_audio, str):
                    raw_audio = raw_audio.encode("utf-8")
                audio_bytes_value = bytes(raw_audio) if raw_audio else b""
            if is_final_arr is not None and is_final_arr.size:
                is_final = bool(is_final_arr.flatten()[0])

            # ack-only response from append_text / text_complete:
            # empty event_type, no audio, no payload, but is_final=True
            if not event_type and not audio_bytes_value and not payload:
                if is_final:
                    emit({"kind": "ack", "t_ms": now_ms})
                return

            emit(
                {
                    "kind": "raw",
                    "event_type": event_type,
                    "payload": payload,
                    "audio": audio_bytes_value,
                    "is_final": is_final,
                    "t_ms": now_ms,
                }
            )
        except Exception as exc:
            emit({"kind": "error", "message": str(exc), "t_ms": now_ms})

    client = grpcclient.InferenceServerClient(url=endpoint)
    outputs = [
        grpcclient.InferRequestedOutput("audio_chunk"),
        grpcclient.InferRequestedOutput("event_type"),
        grpcclient.InferRequestedOutput("event_json"),
        grpcclient.InferRequestedOutput("is_final"),
    ]

    def send_action(action: str, *, text_chunk: str = "") -> None:
        body = build_action_payload(
            action,
            session_id,
            text=text_chunk,
            request=tts_request if action in ("init", "synthesize") else None,
        )
        body_json = json.dumps(body, ensure_ascii=False)
        req_input = grpcclient.InferInput("request", [1], "BYTES")
        req_input.set_data_from_numpy(np.array([body_json], dtype=object))
        client.async_stream_infer(model_name=model_name, inputs=[req_input], outputs=outputs)

    deadline = time.perf_counter() + timeout_sec

    async def consume_until(stop_pred) -> None:
        nonlocal first_audio_ms, triton_adapter_ttft_ms, chunks
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise LlmPkError(f"Triton stream timed out after {timeout_sec:.1f}s")
            item = await asyncio.wait_for(queue.get(), timeout=remaining)
            kind = item["kind"]
            if kind == "error":
                raise LlmPkError(str(item["message"]))
            if kind == "ack":
                # ack from append_text / text_complete; never a stop signal
                continue
            if kind != "raw":
                continue
            event_type = item["event_type"]
            payload = item["payload"]
            audio_bytes_value = item["audio"]
            is_final = item["is_final"]
            now_ms = item["t_ms"]

            if event_type == "start":
                audio_format.update(payload.get("audio_format", {}) or {})
                events.append(
                    TraceEvent(
                        run_id=run_id,
                        backend=backend,
                        type="session_started",
                        t_ms=now_ms,
                        meta={"audio_format": dict(audio_format)},
                    )
                )
            elif event_type == "audio" and audio_bytes_value:
                chunks += 1
                audio_parts.append(audio_bytes_value)
                if first_audio_ms is None:
                    first_audio_ms = now_ms
                    meta = dict(payload.get("meta", {}) or {})
                    raw_adapter = meta.get("triton_adapter_ttft_ms")
                    if raw_adapter is not None:
                        try:
                            triton_adapter_ttft_ms = float(raw_adapter)
                        except (TypeError, ValueError):
                            triton_adapter_ttft_ms = None
                    events.append(
                        TraceEvent(
                            run_id=run_id,
                            backend=backend,
                            type="first_audio_chunk",
                            t_ms=now_ms,
                            meta={
                                "bytes": len(audio_bytes_value),
                                "audio_format": dict(audio_format),
                                **meta,
                            },
                        )
                    )
                else:
                    events.append(
                        TraceEvent(
                            run_id=run_id,
                            backend=backend,
                            type="audio_chunk",
                            t_ms=now_ms,
                            meta={"bytes": len(audio_bytes_value)},
                        )
                    )
            elif event_type in {"warning", "text_token", "text_boundary_commit", "segment_end"}:
                events.append(
                    TraceEvent(
                        run_id=run_id,
                        backend=backend,
                        type=event_type,
                        t_ms=now_ms,
                        text=str(payload.get("text") or ""),
                        meta=payload,
                    )
                )

            if is_final and event_type == "error":
                raise LlmPkError(str(payload.get("message") or "Triton error"))
            if is_final and event_type == "end":
                events.append(
                    TraceEvent(
                        run_id=run_id,
                        backend=backend,
                        type="done",
                        t_ms=now_ms,
                        meta={"audio_format": dict(audio_format), "source_event": event_type},
                    )
                )

            if stop_pred(item):
                return

    client.start_stream(callback=callback)
    try:
        if mode == "streaming":
            send_action("init")
            await consume_until(lambda item: item.get("event_type") == "start")
            for index, (_token_id, fragment) in enumerate(tokens):
                await asyncio.sleep(interval)
                send_ms = (time.perf_counter() - started) * 1000.0
                events.append(
                    TraceEvent(
                        run_id=run_id,
                        backend=backend,
                        type="llm_token",
                        t_ms=send_ms,
                        text=fragment,
                        meta={"index": index},
                    )
                )
                send_action("append_text", text_chunk=fragment)
            simulated_complete_ms = (time.perf_counter() - started) * 1000.0
            events.append(
                TraceEvent(
                    run_id=run_id,
                    backend=backend,
                    type="simulated_llm_complete",
                    t_ms=simulated_complete_ms,
                    meta={"tokens": len(tokens)},
                )
            )
            send_action("text_complete")
            await consume_until(
                lambda item: item.get("event_type") == "end" and item.get("is_final"),
            )
        else:
            for index, (_token_id, fragment) in enumerate(tokens):
                await asyncio.sleep(interval)
                sim_ms = (time.perf_counter() - started) * 1000.0
                events.append(
                    TraceEvent(
                        run_id=run_id,
                        backend=backend,
                        type="llm_token",
                        t_ms=sim_ms,
                        text=fragment,
                        meta={"index": index},
                    )
                )
            simulated_complete_ms = (time.perf_counter() - started) * 1000.0
            events.append(
                TraceEvent(
                    run_id=run_id,
                    backend=backend,
                    type="simulated_llm_complete",
                    t_ms=simulated_complete_ms,
                    meta={"tokens": len(tokens)},
                )
            )
            full_text = "".join(fragment for _, fragment in tokens)
            tts_request = TtsRequest(text=full_text, speaker=speaker, language=language)
            send_action("synthesize")
            events.append(
                TraceEvent(
                    run_id=run_id,
                    backend=backend,
                    type="oneshot_sent",
                    t_ms=simulated_complete_ms,
                    meta={"chars": len(full_text)},
                )
            )
            await consume_until(
                lambda item: item.get("event_type") == "end" and item.get("is_final"),
            )
    except LlmPkError:
        raise
    except TritonUnavailable as exc:
        raise LlmPkError(str(exc)) from exc
    except Exception as exc:
        raise LlmPkError(str(exc)) from exc
    finally:
        try:
            client.stop_stream()
        except Exception:
            pass

    if first_audio_ms is None:
        raise LlmPkError("Triton stream completed without audio")

    total_ms = max(event.t_ms for event in events)
    raw_audio = b"".join(audio_parts)
    audio_duration_ms: float | None = None
    if raw_audio and audio_format.get("encoding") == "pcm_f32":
        audio_duration_ms = (
            len(raw_audio) / 4.0 / float(audio_format.get("sample_rate", 24000)) * 1000.0
        )

    return RunResult(
        run_id=run_id,
        backend=backend,
        label=label,
        mode=description,
        source="live_triton",
        metrics=RunMetrics(
            client_ttfb_ms=first_audio_ms,
            first_playable_ms=first_audio_ms,
            triton_adapter_ttft_ms=triton_adapter_ttft_ms,
            total_ms=total_ms,
            chunks=chunks,
            audio_duration_ms=audio_duration_ms,
            simulated_llm_complete_ms=simulated_complete_ms,
        ),
        events=events,
        audio_format=audio_format,
        raw_audio=raw_audio,
    )
