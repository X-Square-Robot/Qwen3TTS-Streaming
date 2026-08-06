from __future__ import annotations

import base64
import json
import os
import re
import socket
import warnings
from typing import Any, Mapping
from urllib.parse import urlparse

from qwen3tts_protocol.protocol import PROTOCOL_VERSION
from qwen3tts_protocol import (
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
from ..exceptions import (
    EngineVersionMismatchError,
    ProtocolError,
    ProtocolVersionMismatchError,
)


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
        first_chunk=str(
            (payload.get("meta") or {}).get("first_audio_chunk", "")
        ).lower()
        == "true",
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
        segment_id=int(payload.get("segment_id", -1)),
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


def check_protocol_version(server_version: Any) -> None:
    """Runtime pairing guard: server protocol generation must match this SDK's.

    A missing/empty server value is tolerated (older builds that predate the
    handshake). On mismatch this raises with a pointer to the matching wheel;
    ``QWEN3TTS_SKIP_PROTOCOL_CHECK=1`` downgrades it to a warning for
    deliberate cross-version experiments.
    """
    server = str(server_version or "").strip()
    if not server or server == PROTOCOL_VERSION:
        return
    message = (
        f"Server protocol version {server!r} does not match this SDK's "
        f"{PROTOCOL_VERSION!r}. Engine and client are version-paired: install "
        "the wheel this engine serves at GET /sdk/ on its health port, or the "
        "matching GitHub/GitLab Release or Package Registry wheel (the engine reports "
        "its release as capabilities.engine_version). "
        "Set QWEN3TTS_SKIP_PROTOCOL_CHECK=1 to proceed anyway."
    )
    if os.environ.get("QWEN3TTS_SKIP_PROTOCOL_CHECK", "") == "1":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    raise ProtocolVersionMismatchError(message)


# A "clean release" is a bare PEP 440 release, optionally an a/b/rc pre-release
# (the repo tags betas as vX.Y.Zb1). Only when BOTH the engine and the SDK
# report such a form does a difference mean a genuine mispairing worth raising
# on. git-describe distance/dirty suffixes (v0.2.0-5-gabc123), hatch-vcs dev
# builds (0.2.1.dev5+gabc123), and the 0.0.0 source-tree fallback are all
# "unversioned" and only ever warn.
_RELEASE_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")


def _normalize_release(version: Any) -> str:
    """Strip a leading ``v`` so the engine's ``git describe`` stamp (``v0.2.0``)
    and the SDK's hatch-vcs version (``0.2.0``) compare equal on a matched tag."""
    value = str(version or "").strip()
    if value[:1] in ("v", "V"):
        value = value[1:]
    return value


def _is_clean_release(version: Any) -> bool:
    normalized = _normalize_release(version)
    if normalized.startswith("0.0.0"):
        return False  # SDK source-tree fallback; never a real release
    return bool(_RELEASE_RE.match(normalized))


def check_engine_version(server_version: Any) -> None:
    """Connect-time SDK<->engine *release* pairing guard, read from capabilities.

    Complements :func:`check_protocol_version` (the wire-protocol generation):
    the engine image and the client wheel are cut 1:1 from the same git tag, so
    a divergence between two *release* versions is a mispaired install. A
    missing/empty engine value (pre-versioning build) is tolerated; a mismatch
    where either side is a dev/dirty/source-tree build only warns.
    ``QWEN3TTS_SKIP_PROTOCOL_CHECK=1`` downgrades a hard mismatch to a warning.
    """
    from qwen3tts import __version__  # deferred: the package imports this module

    server = _normalize_release(server_version)
    if not server or server == _normalize_release(__version__):
        return
    message = (
        f"SDK/engine release mismatch: client qwen3-tts-client {__version__!r} "
        f"vs engine {str(server_version).strip()!r}. The engine image and client "
        f"wheel are released 1:1 from the same git tag — install the wheel this "
        f"engine serves at GET /sdk/ (on its health port), or the matching "
        f"GitHub/GitLab Release or Package Registry wheel. Set "
        f"QWEN3TTS_SKIP_PROTOCOL_CHECK=1 to proceed anyway."
    )
    if (
        os.environ.get("QWEN3TTS_SKIP_PROTOCOL_CHECK", "") == "1"
        or not _is_clean_release(server_version)
        or not _is_clean_release(__version__)
    ):
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    raise EngineVersionMismatchError(message)


def check_capabilities_pairing(caps: Mapping[str, Any]) -> None:
    """Run both connect-time pairing guards over a capabilities mapping:
    protocol generation (:func:`check_protocol_version`) and engine release
    (:func:`check_engine_version`). This is the single funnel every transport's
    capability exchange flows through."""
    getter = caps.get if isinstance(caps, Mapping) else (lambda _k: "")
    check_protocol_version(getter("protocol_version"))
    check_engine_version(getter("engine_version"))


def capabilities_from_payload(payload: Any) -> Capabilities:
    if not isinstance(payload, dict):
        raise ProtocolError("capabilities payload must be a JSON object")
    check_capabilities_pairing(payload)
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
