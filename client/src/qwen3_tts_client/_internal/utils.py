from __future__ import annotations

import base64
import json
import socket
from typing import Any
from urllib.parse import urlparse

from qwen3_tts_protocol import (
    AudioChunk,
    AudioFormat,
    BytesResult,
    Capabilities,
    SynthesisConfig,
    StreamEvent,
    StreamTextChunk,
    capabilities_from_mapping,
    parse_output_policy,
    parse_timing_context,
    serialize_output_policy,
    serialize_timing_context,
)

from ..constants import DEFAULT_ENGINE_WS_PATH
from ..exceptions import ProtocolError


def parse_host_port(endpoint: str, *, default_port: int) -> tuple[str, int]:
    value = endpoint.strip()
    if not value:
        raise ValueError("endpoint must not be empty")
    if value.startswith("[") and "]" in value:
        host_part, _, remainder = value.partition("]")
        host = f"{host_part}]"
        if remainder.startswith(":"):
            try:
                return host, int(remainder[1:])
            except ValueError as exc:
                raise ValueError(f"invalid port in endpoint: {endpoint!r}") from exc
        return host, default_port
    host, sep, port_text = value.rpartition(":")
    if not sep:
        return value, default_port
    if not host:
        raise ValueError(f"invalid endpoint: {endpoint!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"invalid port in endpoint: {endpoint!r}") from exc
    return host, port


def maybe_base64(value: bytes | None) -> str | None:
    if value is None:
        return None
    return base64.b64encode(value).decode("ascii")


def normalize_http_base(url: str) -> str:
    return url.rstrip("/")


def ensure_ws_url(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme in {"ws", "wss"}:
        return endpoint
    host, port = parse_host_port(endpoint, default_port=80)
    return f"ws://{host}:{port}{DEFAULT_ENGINE_WS_PATH}"


def synthesis_config_to_mapping(config: SynthesisConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_type": config.task_type,
        "language": config.language,
        "speaker": config.speaker or "",
        "instruct": config.instruct or "",
        "ref_audio": maybe_base64(config.ref_audio),
        "ref_text": config.ref_text or "",
        "x_vector_only": bool(config.x_vector_only),
        "input_mode": config.input_mode,
        "group_policy": config.group_policy,
        "audio": {
            "encoding": config.audio.encoding,
            "sample_rate": int(config.audio.sample_rate),
            "channels": int(config.audio.channels),
        },
        "output_policy": serialize_output_policy(config.output_policy),
        "timing": serialize_timing_context(config.timing_context),
        "protocol_version": config.protocol_version or "",
    }
    return {key: value for key, value in payload.items() if value not in (None, "")}


def stream_text_chunk_to_mapping(chunk: StreamTextChunk) -> dict[str, Any]:
    payload = {
        "text": chunk.text,
        "seq_no": int(chunk.seq_no or 0),
        "client_timestamp_ms": int(chunk.client_timestamp_ms or 0),
    }
    return {key: value for key, value in payload.items() if value not in ("", 0)}


def decode_audio_chunk(payload: dict[str, Any], pcm_bytes: bytes) -> AudioChunk:
    audio = payload.get("audio") or payload.get("audio_format") or {}
    return AudioChunk(
        pcm_bytes=pcm_bytes,
        audio=AudioFormat(
            encoding=str(audio.get("encoding", "pcm_f32")),
            sample_rate=int(audio.get("sample_rate", 24000)),
            channels=int(audio.get("channels", 1)),
        ),
        chunk_index=int((payload.get("meta") or {}).get("chunk_index", 0) or 0),
        first_chunk=str((payload.get("meta") or {}).get("first_audio_chunk", "")).lower() == "true",
        final_chunk=bool(payload.get("final_chunk", False)),
        meta={str(k): str(v) for k, v in dict(payload.get("meta") or {}).items()},
    )


def decode_stream_event(payload: dict[str, Any]) -> StreamEvent:
    if not isinstance(payload, dict):
        raise ProtocolError("stream event payload must be a JSON object")
    audio = payload.get("audio")
    return StreamEvent(
        type=str(payload.get("type", "") or ""),
        session_id=str(payload.get("session_id", "") or ""),
        segment_id=int(payload.get("segment_id", -1) or -1),
        text=str(payload.get("text", "") or ""),
        message=str(payload.get("message", "") or ""),
        audio=(
            AudioFormat(
                encoding=str(audio.get("encoding", "pcm_f32")),
                sample_rate=int(audio.get("sample_rate", 24000)),
                channels=int(audio.get("channels", 1)),
            )
            if isinstance(audio, dict)
            else None
        ),
        meta={str(k): str(v) for k, v in dict(payload.get("meta") or {}).items()},
    )


def build_bytes_result(
    *,
    audio_bytes: bytes,
    audio_format: AudioFormat,
    session_id: str,
    transport: str,
    events: list[StreamEvent],
    warnings: list[str],
    details: dict[str, Any],
) -> BytesResult:
    return BytesResult(
        audio_bytes=audio_bytes,
        audio_format=audio_format,
        session_id=session_id,
        transport=transport,
        events=events,
        warnings=warnings,
        details=details,
    )


def capabilities_from_payload(payload: Any) -> Capabilities:
    if not isinstance(payload, dict):
        raise ProtocolError("capabilities payload must be a JSON object")
    return capabilities_from_mapping(payload)


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def json_loads(payload: str | bytes) -> Any:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def sock_set_timeout(sock: socket.socket, timeout: float) -> None:
    sock.settimeout(timeout)


def decode_triton_http_audio_bytes(raw: bytes, *, meta: dict[str, Any]) -> bytes:
    encoding = str(meta.get("audio_chunk_encoding", "") or "").strip().lower()
    if encoding == "base64":
        return base64.b64decode(raw)
    return raw


def merge_start_contract_into_config(
    config: SynthesisConfig,
    *,
    output_policy: Any = None,
    timing: Any = None,
) -> SynthesisConfig:
    if output_policy is not None:
        config.output_policy = parse_output_policy(output_policy)
    if timing is not None:
        config.timing_context = parse_timing_context(timing)
    return config
