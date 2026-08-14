"""TTS streaming transport and audio collection for one sweep trial."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from qwen3tts_protocol import (
    AudioChunk,
    AudioFormat,
    SessionStartRequest,
    StreamEvent,
    SynthesisConfig,
)

from .models import DEFAULT_SAMPLE_RATE, SynthesisResult, TextPacket, TrialStatus


def _event_record(event: StreamEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "segment_id": event.segment_id,
        "text": event.text,
        "message": event.message,
        "meta": dict(event.meta or {}),
    }


def _audio_array(chunk: AudioChunk) -> np.ndarray:
    encoding = (chunk.audio.encoding or "pcm_f32").lower()
    if encoding == "pcm_s16le":
        return (
            np.frombuffer(chunk.pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        )
    if encoding == "pcm_f32":
        return np.frombuffer(chunk.pcm_bytes, dtype=np.float32).copy()
    raise ValueError(f"unsupported audio encoding: {encoding!r}")


def synthesize_once(
    client: Any,
    packets: Sequence[TextPacket],
    *,
    speaker: str,
    session_id: str,
    input_mode: str = "token",
    group_policy: str = "auto",
    timeout: float = 300.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> SynthesisResult:
    """Run one serial streaming trial and retain exact packet/event evidence."""

    config = SynthesisConfig(
        task_type="custom_voice",
        language="auto",
        speaker=speaker,
        input_mode=input_mode,
        group_policy=group_policy,
        audio=AudioFormat(
            encoding="pcm_f32", sample_rate=DEFAULT_SAMPLE_RATE, channels=1
        ),
    )
    request = SessionStartRequest(
        session_id=session_id,
        config=config,
        output_policy=config.output_policy,
        timing=config.timing_context,
    )
    arrays: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    sample_rate: int | None = None
    first_audio_at: float | None = None
    terminal_event: str | None = None
    eos_reason: str | None = None
    error: str | None = None
    session: Any = None
    started_at = time.perf_counter()
    try:
        session = client.open_stream(request)
        for packet in packets:
            session.send_text(packet.text)
            if packet.delay_after_s > 0:
                sleep_fn(packet.delay_after_s)
        session.end()
        for message in session.iter_messages(post_send_idle_timeout=timeout):
            if isinstance(message, AudioChunk):
                current_rate = int(message.audio.sample_rate or DEFAULT_SAMPLE_RATE)
                if sample_rate is not None and current_rate != sample_rate:
                    raise ValueError(
                        f"sample rate changed from {sample_rate} to {current_rate}"
                    )
                sample_rate = current_rate
                samples = _audio_array(message)
                if samples.size:
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter()
                    arrays.append(samples)
            elif isinstance(message, StreamEvent):
                events.append(_event_record(message))
                if message.type in {"done", "error"}:
                    terminal_event = message.type
                    eos_reason = (
                        (message.meta or {}).get("reason")
                        or (message.meta or {}).get("eos_reason")
                        or (message.message if message.type == "done" else None)
                    )
                if message.type == "error":
                    error = message.message or "engine error event"
    except Exception as exc:  # noqa: BLE001 - retain transport failures as data
        error = f"{type(exc).__name__}: {exc}"
        if session is not None:
            try:
                session.close(reason="sweep trial failed")
            except Exception as close_exc:  # noqa: BLE001 - preserve root failure
                error += f"; close failed: {type(close_exc).__name__}: {close_exc}"

    finished_at = time.perf_counter()
    samples = (
        np.concatenate(arrays).astype(np.float32, copy=False)
        if arrays
        else np.empty(0, dtype=np.float32)
    )
    resolved_rate = sample_rate or DEFAULT_SAMPLE_RATE
    status = (
        TrialStatus.ERROR
        if error
        else TrialStatus.OK
        if samples.size
        else TrialStatus.NO_AUDIO
    )
    return SynthesisResult(
        status=status,
        samples=samples,
        sample_rate=resolved_rate,
        duration_s=samples.size / resolved_rate,
        ttft_ms=(
            round((first_audio_at - started_at) * 1000.0)
            if first_audio_at is not None
            else None
        ),
        total_ms=round((finished_at - started_at) * 1000.0),
        chunks=len(arrays),
        terminal_event=terminal_event,
        eos_reason=eos_reason,
        events=events,
        error=error,
    )
