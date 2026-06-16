"""Shared Triton streaming helpers for tests/tools."""

from __future__ import annotations

import json
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SAMPLE_RATE = 24000
REQUEST_MODEL_NAME = "tts_orchestrator"


@dataclass
class StreamResult:
    text: str
    session_id: str = ""
    first_chunk_ms: float | None = None
    total_ms: float = 0.0
    num_chunks: int = 0
    total_samples: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    audio: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return self.total_samples / SAMPLE_RATE if self.total_samples > 0 else 0.0

    @property
    def rtf(self) -> float:
        if self.duration_sec <= 0 or self.total_ms <= 0:
            return 0.0
        return (self.total_ms / 1000) / self.duration_sec


def decode_obj(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_audio_bytes(raw: bytes, audio_format: dict[str, Any]) -> np.ndarray:
    if (audio_format.get("encoding") or "pcm_f32") == "pcm_s16le":
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return np.frombuffer(raw, dtype=np.float32)


def save_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return out_path


def build_request_payload(
    *,
    text: str,
    task_type: str = "",
    language: str = "auto",
    speaker: str = "",
    instruct: str = "",
    action: str = "",
    session_id: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "language": language,
    }
    if task_type:
        payload["task_type"] = task_type
    if speaker:
        payload["speaker"] = speaker
    if instruct:
        payload["instruct"] = instruct
    if action:
        payload["action"] = action
    if session_id:
        payload["session_id"] = session_id
    if extra:
        payload.update(extra)
    return payload


def build_variant_request_payload(
    *,
    variant: str,
    text: str,
    language: str = "auto",
    speaker: str = "",
    instruct: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = variant.lower()
    if normalized.startswith("design"):
        return build_request_payload(
            text=text,
            task_type="voice_design",
            language=language,
            instruct=instruct,
            extra=extra,
        )
    if normalized.startswith("custom"):
        return build_request_payload(
            text=text,
            task_type="custom_voice",
            language=language,
            speaker=speaker,
            instruct=instruct,
            extra=extra,
        )
    raise ValueError(
        f"Variant {variant!r} is not supported by this helper; use custom-* or design-*."
    )


def build_stream_request(
    grpcclient,
    req_dict: dict[str, Any],
):
    req_json = json.dumps(req_dict)
    req_input = grpcclient.InferInput("request", [1], "BYTES")
    req_input.set_data_from_numpy(np.array([req_json], dtype=object))
    return req_input


def build_stream_outputs(grpcclient) -> list[Any]:
    return [
        grpcclient.InferRequestedOutput("audio_chunk"),
        grpcclient.InferRequestedOutput("event_type"),
        grpcclient.InferRequestedOutput("event_json"),
        grpcclient.InferRequestedOutput("is_final"),
    ]


def build_text_stream_requests(
    init_req: dict[str, Any],
    text_chunks: Sequence[str],
) -> list[dict[str, Any]]:
    session_id = init_req.get("session_id")
    if not session_id:
        raise ValueError("init_req must include a session_id for streaming text input")

    requests = [dict(init_req)]
    requests.extend(
        {
            "action": "append_text",
            "session_id": session_id,
            "text": chunk_text,
        }
        for chunk_text in text_chunks
    )
    requests.append(
        {
            "action": "text_complete",
            "session_id": session_id,
        }
    )
    return requests


def infer_stream_sequence(
    client,
    grpcclient,
    requests: Sequence[dict[str, Any]],
    *,
    timeout: float = 120.0,
    delay_between_requests_ms: float = 0.0,
    result_text: str | None = None,
    session_id: str | None = None,
) -> StreamResult:
    if not requests:
        raise ValueError("at least one Triton request is required")

    first_request = requests[0]
    resolved_session_id = session_id or first_request.get("session_id", "")
    result = StreamResult(
        text=result_text if result_text is not None else first_request.get("text", ""),
        session_id=str(resolved_session_id) if resolved_session_id else "",
    )
    chunks: list[np.ndarray] = []
    errors: list[str] = []
    done = threading.Event()
    first_ts = [None]
    audio_format: dict[str, Any] = {"encoding": "pcm_f32", "sample_rate": SAMPLE_RATE}

    def callback(result_obj=None, error=None):
        if error:
            err_str = str(error)
            if "CAPABILITIES:" not in err_str:
                errors.append(err_str)
                done.set()
            return
        if result_obj is None:
            return
        event_type = result_obj.as_numpy("event_type")
        event_json = result_obj.as_numpy("event_json")
        audio = result_obj.as_numpy("audio_chunk")
        is_final = result_obj.as_numpy("is_final")
        et = decode_obj(event_type.flatten()[0]) if event_type is not None and event_type.size else ""
        payload = {}
        if event_json is not None and event_json.size:
            raw_json = decode_obj(event_json.flatten()[0])
            if raw_json:
                payload = json.loads(raw_json)
        if et == "start":
            audio_format.update(payload.get("audio_format", {}) or {})
            result.metadata["audio_format"] = dict(audio_format)
        elif et == "warning":
            result.warnings.append(payload.get("message", ""))
        elif et == "audio" and audio is not None and audio.size:
            if first_ts[0] is None:
                first_ts[0] = time.perf_counter()
            chunks.append(decode_audio_bytes(audio.flatten()[0], audio_format))
        elif et == "error":
            errors.append(payload.get("message", "unknown error"))
            done.set()
            return
        final = bool(is_final.flatten()[0]) if is_final is not None and is_final.size else False
        if final:
            done.set()

    t0 = time.perf_counter()
    client.start_stream(callback=callback)
    try:
        for idx, req_dict in enumerate(requests):
            if idx > 0 and delay_between_requests_ms > 0:
                time.sleep(delay_between_requests_ms / 1000.0)
            client.async_stream_infer(
                model_name=REQUEST_MODEL_NAME,
                inputs=[build_stream_request(grpcclient, req_dict)],
                outputs=build_stream_outputs(grpcclient),
            )
        done.wait(timeout=timeout)
    finally:
        client.stop_stream()
    elapsed = time.perf_counter() - t0

    if errors:
        result.error = errors[0]
    result.total_ms = elapsed * 1000
    result.first_chunk_ms = (first_ts[0] - t0) * 1000 if first_ts[0] else None
    result.num_chunks = len(chunks)
    if chunks:
        result.audio = np.concatenate(chunks)
        result.total_samples = result.audio.size
    return result


def infer_stream(
    client,
    grpcclient,
    req_dict: dict[str, Any],
    *,
    timeout: float = 120.0,
) -> StreamResult:
    return infer_stream_sequence(client, grpcclient, [req_dict], timeout=timeout)


def infer_text_stream(
    triton_url: str,
    grpcclient,
    init_req: dict[str, Any],
    text_chunks: Sequence[str],
    *,
    chunk_delay_ms: float = 50.0,
    timeout: float = 120.0,
) -> StreamResult:
    chunks = list(text_chunks)
    session_id = init_req.get("session_id", "stream-unknown")
    client = grpcclient.InferenceServerClient(url=triton_url)
    return infer_stream_sequence(
        client,
        grpcclient,
        build_text_stream_requests(init_req, chunks),
        timeout=timeout,
        delay_between_requests_ms=chunk_delay_ms,
        result_text=" ".join(chunks),
        session_id=str(session_id),
    )
