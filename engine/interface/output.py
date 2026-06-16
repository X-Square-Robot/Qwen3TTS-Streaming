from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core.types import AudioConfig, AudioEncoding
from .protocol import PROTOCOL_VERSION, output_policy_json, timing_context_json
from .types import AudioFrame, SessionStartRequest, StreamEvent

ENGINE_SAMPLE_RATE = 24000
TIMING_CONTRACT = "server_monotonic_v1"


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if audio.size == 0 or src_sr == dst_sr:
        return audio
    duration = audio.shape[0] / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


class OutputPipeline:
    """Canonical output/timing pipeline shared by transport adapters."""

    def __init__(
        self,
        start_request: SessionStartRequest,
        *,
        native_sample_rate: int = ENGINE_SAMPLE_RATE,
        request_received_monotonic: float | None = None,
        request_received_epoch_ms: int | None = None,
    ) -> None:
        self._start_request = start_request
        self._audio = start_request.config.audio
        self._native_sample_rate = int(native_sample_rate or ENGINE_SAMPLE_RATE)
        self._request_received_monotonic = (
            float(request_received_monotonic)
            if request_received_monotonic is not None
            else time.monotonic()
        )
        self._request_received_epoch_ms = (
            int(request_received_epoch_ms)
            if request_received_epoch_ms is not None
            else int(round(time.time() * 1000.0))
        )
        self._first_audio_epoch_ms: int | None = None
        self._chunk_index = 0

    @property
    def start_request(self) -> SessionStartRequest:
        return self._start_request

    @property
    def request_received_epoch_ms(self) -> int:
        return self._request_received_epoch_ms

    @property
    def chunk_count(self) -> int:
        return self._chunk_index

    def convert_audio_chunk(self, pcm_bytes: bytes) -> AudioFrame:
        audio = np.frombuffer(pcm_bytes, dtype=np.float32)
        if self._audio.sample_rate != self._native_sample_rate:
            audio = _resample_linear(audio, self._native_sample_rate, self._audio.sample_rate)
        if self._audio.encoding == AudioEncoding.PCM_S16LE:
            audio = np.clip(audio, -1.0, 1.0)
            payload = (audio * 32767.0).astype(np.int16).tobytes()
        else:
            payload = audio.astype(np.float32, copy=False).tobytes()

        chunk_index = self._chunk_index
        self._chunk_index += 1
        first_chunk = chunk_index == 0
        now_epoch_ms = int(round(time.time() * 1000.0))
        meta = {
            "chunk_index": str(chunk_index),
            "timing_contract": TIMING_CONTRACT,
        }
        if first_chunk:
            self._first_audio_epoch_ms = now_epoch_ms
            meta["first_audio_chunk"] = "true"
            meta["server_ttft_ms"] = f"{(time.monotonic() - self._request_received_monotonic) * 1000.0:.3f}"
            meta["server_first_audio_epoch_ms"] = str(now_epoch_ms)

        return AudioFrame(
            pcm_bytes=payload,
            audio=self._audio,
            chunk_index=chunk_index,
            first_chunk=first_chunk,
            final_chunk=False,
            meta=meta,
        )

    def done_meta(self, metrics: dict[str, Any] | None = None) -> dict[str, str]:
        meta = {
            "timing_contract": TIMING_CONTRACT,
            "server_request_received_epoch_ms": str(self._request_received_epoch_ms),
            "server_done_epoch_ms": str(int(round(time.time() * 1000.0))),
            "server_total_latency_ms": f"{(time.monotonic() - self._request_received_monotonic) * 1000.0:.3f}",
            "audio_chunk_count": str(self._chunk_index),
        }
        timing = self._start_request.timing
        if timing.request_id:
            meta["request_id"] = timing.request_id
        if timing.turn_id:
            meta["turn_id"] = timing.turn_id
        if timing.client_request_ts_ms > 0:
            meta["client_request_ts_ms"] = str(timing.client_request_ts_ms)
        if timing.client_text_ts_ms > 0:
            meta["client_text_ts_ms"] = str(timing.client_text_ts_ms)
        if timing.client_end_ts_ms > 0:
            meta["client_end_ts_ms"] = str(timing.client_end_ts_ms)
        for key, value in dict(timing.extra or {}).items():
            meta[str(key)] = str(value)
        if self._first_audio_epoch_ms is not None:
            meta["server_first_audio_epoch_ms"] = str(self._first_audio_epoch_ms)
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key == "error":
                    continue
                meta[str(key)] = str(value)
        return meta


def _base_meta(start_request: SessionStartRequest) -> dict[str, str]:
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "timing_contract": TIMING_CONTRACT,
        "input_mode": start_request.config.input_mode.value,
        "group_policy": start_request.config.group_policy.value,
        "task_type": start_request.config.task_type or "",
        "output_policy_json": output_policy_json(start_request.output_policy),
        "timing_context_json": timing_context_json(start_request.timing),
        "vad_enabled": "true" if start_request.output_policy.vad.enabled else "false",
        "vad_strategy": str(start_request.output_policy.vad.strategy or "disabled"),
    }
    if start_request.timing.request_id:
        meta["request_id"] = start_request.timing.request_id
    if start_request.timing.turn_id:
        meta["turn_id"] = start_request.timing.turn_id
    if start_request.timing.client_request_ts_ms > 0:
        meta["client_request_ts_ms"] = str(start_request.timing.client_request_ts_ms)
    return meta


def build_start_event(session_id: str, start_request: SessionStartRequest) -> StreamEvent:
    return StreamEvent(
        type="start",
        session_id=session_id,
        audio=start_request.config.audio,
        meta=_base_meta(start_request),
    )


def build_forward_event(
    session_id: str,
    event: dict[str, Any],
    start_request: SessionStartRequest,
) -> StreamEvent:
    meta = {str(k): str(v) for k, v in (event.get("meta", {}) or {}).items()}
    if start_request.timing.request_id and "request_id" not in meta:
        meta["request_id"] = start_request.timing.request_id
    if start_request.timing.turn_id and "turn_id" not in meta:
        meta["turn_id"] = start_request.timing.turn_id
    return StreamEvent(
        type=str(event.get("type", "") or ""),
        session_id=session_id,
        segment_id=int(event.get("segment_idx", -1)),
        text=str(event.get("text", "") or ""),
        message=str(event.get("message", "") or ""),
        audio=start_request.config.audio if event.get("type") == "start" else None,
        meta=meta,
    )


def build_done_event(
    session_id: str,
    metrics: dict[str, Any] | None,
    pipeline: OutputPipeline,
) -> StreamEvent:
    error = metrics.get("error") if isinstance(metrics, dict) else None
    return StreamEvent(
        type="error" if error else "done",
        session_id=session_id,
        message=str(error or ""),
        meta=pipeline.done_meta(metrics),
    )
