"""SDK streaming collectors for the standalone and frozen Triton arms."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from qwen3tts_protocol import (
    AudioChunk,
    AudioFormat,
    OutputPolicy,
    SessionStartRequest,
    StreamEvent,
    SynthesisConfig,
)

from .arm_types import (
    AudioChunkRecord,
    CollectedRun,
    DEFAULT_LANGUAGE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEAKER,
    finish_run,
)
from .models import ArmKind, RunStatus


ClientFactory = Callable[..., Any]


def _connect_tts_client(endpoint: str, **kwargs: Any) -> Any:
    # Transport extras are needed only when an endpoint arm actually connects.
    from qwen3tts import TTSClient

    return TTSClient.connect(endpoint, **kwargs)


def _event_record(event: StreamEvent) -> dict[str, Any]:
    audio = event.audio
    return {
        "type": event.type,
        "session_id": event.session_id,
        "segment_id": event.segment_id,
        "text": event.text,
        "message": event.message,
        "audio": (
            {
                "encoding": audio.encoding,
                "sample_rate": int(audio.sample_rate),
                "channels": int(audio.channels),
            }
            if audio is not None
            else None
        ),
        "meta": dict(event.meta or {}),
    }


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decode_audio_chunk(chunk: AudioChunk) -> np.ndarray:
    encoding = str(chunk.audio.encoding or "pcm_f32").lower()
    if encoding == "pcm_f32":
        if len(chunk.pcm_bytes) % np.dtype("<f4").itemsize:
            raise ValueError("pcm_f32 AudioChunk byte length is not divisible by 4")
        return np.frombuffer(chunk.pcm_bytes, dtype="<f4").astype(
            np.float32, copy=True
        )
    if encoding == "pcm_s16le":
        if len(chunk.pcm_bytes) % np.dtype("<i2").itemsize:
            raise ValueError("pcm_s16le AudioChunk byte length is not divisible by 2")
        return (
            np.frombuffer(chunk.pcm_bytes, dtype="<i2").astype(np.float32)
            / 32767.0
        )
    raise ValueError(f"unsupported audio encoding: {encoding!r}")


def _validated_audio_format(message: AudioChunk) -> tuple[int, int]:
    sample_rate = int(message.audio.sample_rate or DEFAULT_SAMPLE_RATE)
    channels = int(message.audio.channels or 1)
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(
            f"arm returned {sample_rate} Hz; expected {DEFAULT_SAMPLE_RATE} Hz"
        )
    if channels != 1:
        raise ValueError(f"arm returned {channels} channels; expected mono")
    return sample_rate, channels


def _chunk_record(
    message: AudioChunk,
    decoded: np.ndarray,
    *,
    sequence_index: int,
    sample_start: int,
    sample_rate: int,
    channels: int,
) -> AudioChunkRecord:
    audio_format = message.audio
    meta = {
        str(key): str(value) for key, value in dict(message.meta or {}).items()
    }
    sample_end = sample_start + int(decoded.size)
    return AudioChunkRecord(
        sequence_index=sequence_index,
        chunk_index=int(getattr(message, "chunk_index", sequence_index)),
        sample_start=sample_start,
        sample_end=sample_end,
        sample_count=int(decoded.size),
        sample_rate=sample_rate,
        channels=channels,
        encoding=str(audio_format.encoding or "pcm_f32"),
        first_chunk=bool(getattr(message, "first_chunk", False)),
        final_chunk=bool(getattr(message, "final_chunk", False)),
        output_sample_start=(
            getattr(message, "output_sample_start", None)
            if getattr(message, "output_sample_start", None) is not None
            else _optional_int(meta.get("output_sample_start"))
        ),
        output_sample_end=(
            getattr(message, "output_sample_end", None)
            if getattr(message, "output_sample_end", None) is not None
            else _optional_int(meta.get("output_sample_end"))
        ),
        meta=meta,
        pcm_sha256=hashlib.sha256(message.pcm_bytes).hexdigest(),
    )


def _join_audio(arrays: list[np.ndarray]) -> np.ndarray:
    return (
        np.concatenate(arrays).astype(np.float32, copy=False)
        if arrays
        else np.empty(0, dtype=np.float32)
    )


def _ttft_ms(first_audio_at: float | None, started_at: float) -> int | None:
    return (
        round((first_audio_at - started_at) * 1000.0)
        if first_audio_at is not None
        else None
    )


class EndpointArmAdapter:
    """Common SDK collector for standalone-engine and Triton gRPC arms."""

    def __init__(
        self,
        endpoint: str,
        *,
        arm: ArmKind,
        transport: str,
        timeout: float = 600.0,
        speaker: str = DEFAULT_SPEAKER,
        language: str = DEFAULT_LANGUAGE,
        model_name: str | None = None,
        model_version: str | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.endpoint = endpoint
        self.arm = ArmKind(arm)
        self.transport = transport
        self.timeout = float(timeout)
        self.speaker = speaker
        self.language = language
        self.model_name = model_name
        self.model_version = model_version
        self._client_factory = client_factory or _connect_tts_client
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            connect_kwargs: dict[str, Any] = {
                "transport": self.transport,
                "timeout": self.timeout,
            }
            if self.model_name is not None:
                connect_kwargs["model_name"] = self.model_name
            if self.model_version is not None:
                connect_kwargs["model_version"] = self.model_version
            self._client = self._client_factory(self.endpoint, **connect_kwargs)
        return self._client

    def close(self) -> None:
        client, self._client = self._client, None
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> EndpointArmAdapter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def collect(
        self,
        text: str,
        *,
        session_id: str,
        seed: int = 0,
    ) -> CollectedRun:
        """Send *text* once, byte-for-byte unchanged, and retain every output."""

        if not isinstance(text, str):
            raise TypeError("text must be str")
        if not session_id:
            raise ValueError("session_id must not be empty")

        return self._collect_packets(
            (text,),
            session_id=session_id,
            seed=seed,
            input_mode="full_text",
            group_policy="auto",
            output_policy=None,
        )

    def collect_packets(
        self,
        text_packets: Sequence[str],
        session_id: str,
        seed: int = 0,
        *,
        input_mode: str = "long_segment",
        group_policy: str = "none",
        output_policy: OutputPolicy | None = None,
    ) -> CollectedRun:
        """Replay frozen segments, in order, through one logical stream.

        The public SID is forwarded unchanged so the engine derives each
        segment's sampling identity from its real segment index.  No per-packet
        session or seed rewriting occurs.
        """

        if isinstance(text_packets, (str, bytes)):
            raise TypeError("text_packets must be a sequence of strings")
        packets = tuple(text_packets)
        if not packets:
            raise ValueError("text_packets must not be empty")
        if not all(isinstance(packet, str) for packet in packets):
            raise TypeError("every text packet must be str")
        if not session_id:
            raise ValueError("session_id must not be empty")
        return self._collect_packets(
            packets,
            session_id=session_id,
            seed=seed,
            input_mode=input_mode,
            group_policy=group_policy,
            output_policy=output_policy,
        )

    def _collect_packets(
        self,
        text_packets: tuple[str, ...],
        *,
        session_id: str,
        seed: int,
        input_mode: str,
        group_policy: str,
        output_policy: OutputPolicy | None,
    ) -> CollectedRun:
        config = SynthesisConfig(
            task_type="custom_voice",
            language=self.language,
            speaker=self.speaker,
            input_mode=input_mode,
            group_policy=group_policy,
            audio=AudioFormat(
                encoding="pcm_f32",
                sample_rate=DEFAULT_SAMPLE_RATE,
                channels=1,
            ),
        )
        if output_policy is not None:
            config.output_policy = output_policy
        request = SessionStartRequest(
            session_id=session_id,
            config=config,
            output_policy=config.output_policy,
            timing=config.timing_context,
        )
        started_at = time.perf_counter()
        first_audio_at: float | None = None
        session: Any = None
        arrays: list[np.ndarray] = []
        events: list[dict[str, Any]] = []
        audio_chunks: list[AudioChunkRecord] = []
        sample_cursor = 0
        terminal_event: str | None = None
        eos_reason: str | None = None
        tts_error: str | None = None
        try:
            session = self._get_client().open_stream(request)
            # Deliberately no strip(), normalization, or packet rewriting here.
            for packet in text_packets:
                session.send_text(packet)
            session.end()
            for message in session.iter_messages(
                post_send_idle_timeout=self.timeout
            ):
                if isinstance(message, AudioChunk):
                    sample_rate, channels = _validated_audio_format(message)
                    decoded = _decode_audio_chunk(message)
                    if first_audio_at is None and decoded.size:
                        first_audio_at = time.perf_counter()
                    record = _chunk_record(
                        message,
                        decoded,
                        sequence_index=len(audio_chunks),
                        sample_start=sample_cursor,
                        sample_rate=sample_rate,
                        channels=channels,
                    )
                    audio_chunks.append(record)
                    arrays.append(decoded)
                    sample_cursor = record.sample_end
                    continue

                if isinstance(message, StreamEvent):
                    events.append(_event_record(message))
                    if message.type in {"done", "error"}:
                        terminal_event = message.type
                        meta: Mapping[str, Any] = message.meta or {}
                        eos_reason = str(
                            meta.get("reason")
                            or meta.get("eos_reason")
                            or (message.message if message.type == "done" else "")
                            or ""
                        ) or None
                    if message.type == "error":
                        tts_error = message.message or "TTS stream emitted error"
        except Exception as exc:  # noqa: BLE001 - failure is experiment evidence
            if session is not None:
                try:
                    session.close(reason="long-form collection failed")
                except Exception:
                    pass
            return finish_run(
                self.arm,
                session_id=session_id,
                seed=seed,
                status=RunStatus.ERROR,
                started_at=started_at,
                error=f"{type(exc).__name__}: {exc}",
                events=events,
                audio_chunks=audio_chunks,
                samples=_join_audio(arrays),
                ttft_ms=_ttft_ms(first_audio_at, started_at),
                terminal_event=terminal_event,
                eos_reason=eos_reason,
            )

        samples = _join_audio(arrays)
        failure = tts_error
        if failure is None and terminal_event != "done":
            failure = "TTS stream ended without a done event"
        if failure is None and samples.size == 0:
            failure = "TTS stream produced no audio"
        return finish_run(
            self.arm,
            session_id=session_id,
            seed=seed,
            status=RunStatus.TTS_FAILED if failure else RunStatus.OK,
            started_at=started_at,
            error=failure,
            events=events,
            audio_chunks=audio_chunks,
            samples=samples,
            ttft_ms=_ttft_ms(first_audio_at, started_at),
            terminal_event=terminal_event,
            eos_reason=eos_reason,
        )


class EngineGrpcArmAdapter(EndpointArmAdapter):
    """Current-HEAD standalone engine reached through the SDK gRPC adapter."""

    def __init__(self, endpoint: str, **kwargs: Any) -> None:
        super().__init__(
            endpoint,
            arm=ArmKind.CURRENT_HEAD,
            transport="engine-grpc",
            **kwargs,
        )


class TritonGrpcArmAdapter(EndpointArmAdapter):
    """Frozen 0818 Triton prototype reached through the SDK gRPC adapter."""

    def __init__(self, endpoint: str, **kwargs: Any) -> None:
        super().__init__(
            endpoint,
            arm=ArmKind.TRITON_0818,
            transport="triton-grpc",
            **kwargs,
        )


EngineGrpcArm = EngineGrpcArmAdapter
TritonGrpcArm = TritonGrpcArmAdapter


__all__ = [
    "EndpointArmAdapter",
    "EngineGrpcArm",
    "EngineGrpcArmAdapter",
    "TritonGrpcArm",
    "TritonGrpcArmAdapter",
]
