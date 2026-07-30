"""Qwen3-TTS shared protocol types.

This package defines the wire-format types shared between the server and
client SDKs.  It intentionally has **no** dependency on the engine, numpy,
tritonclient, or any other heavy runtime — only stdlib and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Audio format
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioFormat:
    """PCM audio format descriptor carried in every audio-bearing message."""

    encoding: str = "pcm_f32"
    sample_rate: int = 24000
    channels: int = 1


# ---------------------------------------------------------------------------
# Synthesis config
# ---------------------------------------------------------------------------


@dataclass
class VADPolicy:
    enabled: bool = False
    strategy: str = "disabled"  # "disabled" | "energy" | "tenvad"
    implementation: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    # Direct VAD parameters
    chunk_ms: int = 16
    begin_threshold: float = 0.6
    begin_count: int = 5
    end_threshold: float = 0.35
    end_count: int = 31
    start_margin_ms: int = 20


@dataclass
class OutputPolicy:
    vad: VADPolicy = field(default_factory=VADPolicy)
    chunk_ms: int = 0
    packet_format: str = "raw_pcm"
    emit_text_events: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingContext:
    request_id: str = ""
    turn_id: str = ""
    client_request_ts_ms: int = 0
    client_text_ts_ms: int = 0
    client_end_ts_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisConfig:
    """Full TTS request configuration."""

    task_type: str = "custom_voice"
    language: str = "auto"
    speaker: str = ""
    instruct: str = ""
    ref_audio: bytes = b""
    ref_text: str = ""
    x_vector_only: bool = False
    input_mode: str | None = None
    group_policy: str | None = None
    audio: AudioFormat = field(default_factory=AudioFormat)
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    timing_context: TimingContext = field(default_factory=TimingContext)
    protocol_version: str | None = None


# ---------------------------------------------------------------------------
# Stream messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamEvent:
    """A non-audio event in a TTS stream (text token, boundary, done, …)."""

    type: str = ""
    session_id: str = ""
    segment_id: int = -1
    text: str = ""
    message: str = ""
    audio: AudioFormat | None = None
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioChunk:
    """A single PCM audio chunk in a TTS stream."""

    pcm_bytes: bytes = b""
    audio: AudioFormat = field(default_factory=AudioFormat)
    chunk_index: int = 0
    first_chunk: bool = False
    final_chunk: bool = False
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamTextChunk:
    """A text chunk to send into an open stream."""

    text: str = ""
    seq_no: int | None = None
    client_timestamp_ms: int | None = None


# ---------------------------------------------------------------------------
# Session lifecycle requests
# ---------------------------------------------------------------------------


@dataclass
class SessionStartRequest:
    session_id: str = ""
    config: SynthesisConfig = field(default_factory=SynthesisConfig)
    output_policy: OutputPolicy | None = None
    timing: TimingContext | None = None


@dataclass
class SessionEndRequest:
    session_id: str = ""


@dataclass
class StreamCancelRequest:
    session_id: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BytesResult:
    """Non-streaming result: concatenated audio bytes + metadata."""

    audio_bytes: bytes = b""
    audio_format: AudioFormat = field(default_factory=AudioFormat)
    session_id: str = ""
    transport: str = ""
    events: list[StreamEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArrayResult:
    """Non-streaming result: numpy float32 array + metadata."""

    audio_array: Any = None  # numpy.ndarray, lazy-imported
    audio_format: AudioFormat = field(default_factory=AudioFormat)
    session_id: str = ""
    transport: str = ""
    events: list[StreamEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@dataclass
class Capabilities:
    variant: str = ""
    loaded_model_type: str = ""
    default_speaker: str = ""
    fallback_speaker: str = ""
    speakers: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    max_batch_size: int = 0
    max_input_len: int = 0
    max_seq_len: int = 0
    ref_audio_max_duration_sec: float = 0.0
    protocol_version: str = ""
    # Engine release stamp (git tag) for SDK<->engine wheel pairing; distinct
    # from protocol_version (the wire-protocol generation). Empty on
    # pre-versioning / source-tree engine builds.
    engine_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedTransport:
    """Result of transport auto-detection."""

    transport: str = ""
    requested_endpoint: str = ""
    resolved_endpoint: str = ""
    model_name: str = ""
    model_version: str = ""
    probe_report: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def capabilities_from_mapping(payload: dict[str, Any]) -> Capabilities:
    """Build a Capabilities from a raw JSON-like dict."""
    return Capabilities(
        variant=str(payload.get("variant", "") or ""),
        loaded_model_type=str(payload.get("loaded_model_type", "") or ""),
        default_speaker=str(payload.get("default_speaker", "") or ""),
        fallback_speaker=str(payload.get("fallback_speaker", "") or ""),
        speakers=list(payload.get("speakers") or []),
        languages=list(payload.get("languages") or []),
        max_batch_size=int(payload.get("max_batch_size", 0) or 0),
        max_input_len=int(payload.get("max_input_len", 0) or 0),
        max_seq_len=int(payload.get("max_seq_len", 0) or 0),
        ref_audio_max_duration_sec=float(
            payload.get("ref_audio_max_duration_sec", 0) or 0
        ),
        protocol_version=str(payload.get("protocol_version", "") or ""),
        engine_version=str(payload.get("engine_version", "") or ""),
        extra={
            k: v
            for k, v in payload.items()
            if k
            not in {
                "variant",
                "loaded_model_type",
                "default_speaker",
                "fallback_speaker",
                "speakers",
                "languages",
                "max_batch_size",
                "max_input_len",
                "max_seq_len",
                "ref_audio_max_duration_sec",
                "protocol_version",
                "engine_version",
            }
        },
    )


def serialize_output_policy(policy: OutputPolicy) -> dict[str, Any]:
    vad_dict: dict[str, Any] = {
        "enabled": bool(policy.vad.enabled),
        "strategy": str(policy.vad.strategy),
        "implementation": str(policy.vad.implementation),
        "config": dict(policy.vad.config),
    }
    if policy.vad.strategy not in ("disabled", ""):
        vad_dict.update(
            {
                "chunk_ms": int(policy.vad.chunk_ms),
                "begin_threshold": float(policy.vad.begin_threshold),
                "begin_count": int(policy.vad.begin_count),
                "end_threshold": float(policy.vad.end_threshold),
                "end_count": int(policy.vad.end_count),
                "start_margin_ms": int(policy.vad.start_margin_ms),
            }
        )
    return {
        "vad_policy": vad_dict,
        "chunk_ms": int(policy.chunk_ms),
        "packet_format": str(policy.packet_format),
        "emit_text_events": bool(policy.emit_text_events),
        "config": dict(policy.config),
    }


def parse_output_policy(raw: Any) -> OutputPolicy:
    if not isinstance(raw, dict):
        return OutputPolicy()
    vad_raw = raw.get("vad_policy") or raw.get("vad") or {}

    # ``or default`` is incorrect for numeric policy fields because zero is a
    # meaningful value (for example ``start_margin_ms=0`` disables lookback
    # and ``end_threshold=0.0`` is a valid threshold).  Treat only an omitted,
    # explicit-null, or empty-string wire value as absent.
    def _numeric_value(mapping: dict[str, Any], key: str, default: Any) -> Any:
        value = mapping.get(key)
        return default if value is None or value == "" else value

    return OutputPolicy(
        vad=VADPolicy(
            enabled=bool(vad_raw.get("enabled", False)),
            strategy=str(vad_raw.get("strategy", "disabled")),
            implementation=str(vad_raw.get("implementation", "")),
            config=dict(vad_raw.get("config") or {}),
            chunk_ms=int(_numeric_value(vad_raw, "chunk_ms", 16)),
            begin_threshold=float(_numeric_value(vad_raw, "begin_threshold", 0.6)),
            begin_count=int(_numeric_value(vad_raw, "begin_count", 5)),
            end_threshold=float(_numeric_value(vad_raw, "end_threshold", 0.35)),
            end_count=int(_numeric_value(vad_raw, "end_count", 31)),
            start_margin_ms=int(_numeric_value(vad_raw, "start_margin_ms", 20)),
        ),
        chunk_ms=int(_numeric_value(raw, "chunk_ms", 0)),
        packet_format=str(raw.get("packet_format", "raw_pcm")),
        emit_text_events=bool(raw.get("emit_text_events", True)),
        config=dict(raw.get("config") or {}),
    )


def serialize_timing_context(timing: TimingContext) -> dict[str, Any]:
    return {
        "request_id": timing.request_id,
        "turn_id": timing.turn_id,
        "client_request_ts_ms": int(timing.client_request_ts_ms),
        "client_text_ts_ms": int(timing.client_text_ts_ms),
        "client_end_ts_ms": int(timing.client_end_ts_ms),
        "extra": dict(timing.extra),
    }


def parse_timing_context(raw: Any) -> TimingContext:
    if not isinstance(raw, dict):
        return TimingContext()
    return TimingContext(
        request_id=str(raw.get("request_id", "") or ""),
        turn_id=str(raw.get("turn_id", "") or ""),
        client_request_ts_ms=int(raw.get("client_request_ts_ms", 0) or 0),
        client_text_ts_ms=int(raw.get("client_text_ts_ms", 0) or 0),
        client_end_ts_ms=int(raw.get("client_end_ts_ms", 0) or 0),
        extra=dict(raw.get("extra") or {}),
    )


__all__ = [
    "ArrayResult",
    "AudioChunk",
    "AudioFormat",
    "BytesResult",
    "Capabilities",
    "DetectedTransport",
    "OutputPolicy",
    "SessionEndRequest",
    "SessionStartRequest",
    "StreamCancelRequest",
    "StreamEvent",
    "StreamTextChunk",
    "SynthesisConfig",
    "TimingContext",
    "VADPolicy",
    "capabilities_from_mapping",
    "parse_output_policy",
    "parse_timing_context",
    "serialize_output_policy",
    "serialize_timing_context",
]
