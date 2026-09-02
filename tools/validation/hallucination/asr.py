"""Optional FunASR client loading and per-WAV transcription."""

from __future__ import annotations

import asyncio
import inspect
import math
import sys
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import AsrStatus


REQUIRED_FUNASR_VERSION = "0.2.0a6"
_REQUIRED_SERVICE_CAPABILITIES = {
    "delivery_modes": "offline",
    "partial_modes": "off",
    "vad_backends": "fsmn",
}
_REQUIRED_SERVICE_FEATURES = frozenset(
    {
        "typed-stream-control-v1",
        "delivery-mode-v1",
    }
)
_MIN_TRANSCRIPTION_TIMEOUT_S = 120.0
_TRANSCRIPTION_TIMEOUT_MULTIPLIER = 2.0
_TRANSCRIPTION_TIMEOUT_OVERHEAD_S = 60.0


def validate_funasr_sdk_version(
    installed_version: str,
    *,
    required_version: str = REQUIRED_FUNASR_VERSION,
) -> str:
    """Fail fast unless the dedicated ASR environment has the pinned SDK."""

    if not isinstance(installed_version, str) or not installed_version.strip():
        raise RuntimeError("FunASR SDK version is missing")
    if installed_version != required_version:
        raise RuntimeError(
            "FunASR SDK version mismatch: "
            f"required {required_version!r}, found {installed_version!r}"
        )
    return installed_version


def validate_funasr_service_capabilities(
    capabilities: Mapping[str, Any],
    *,
    required_version: str = REQUIRED_FUNASR_VERSION,
    strict_protocol: bool = False,
) -> str:
    """Validate a caller-fetched capabilities payload without doing network I/O."""

    if not isinstance(capabilities, Mapping):
        raise RuntimeError("FunASR service capabilities must be a mapping")
    service_version = capabilities.get("version")
    if not isinstance(service_version, str) or not service_version.strip():
        raise RuntimeError("FunASR service capabilities.version is missing")
    if service_version != required_version:
        raise RuntimeError(
            "FunASR service version mismatch: "
            f"required {required_version!r}, found {service_version!r}"
        )
    if strict_protocol:
        protocol_version = capabilities.get("protocol_version")
        if not isinstance(protocol_version, str) or not protocol_version.strip():
            raise RuntimeError("FunASR service protocol_version is missing")
        for field, required_value in _REQUIRED_SERVICE_CAPABILITIES.items():
            advertised = capabilities.get(field)
            if (
                not isinstance(advertised, list)
                or required_value not in advertised
            ):
                raise RuntimeError(
                    "FunASR service lacks required capability: "
                    f"{field}={required_value!r}"
                )
        features = capabilities.get("features")
        if not isinstance(features, list) or not _REQUIRED_SERVICE_FEATURES.issubset(
            set(features)
        ):
            missing = sorted(
                _REQUIRED_SERVICE_FEATURES
                - set(features if isinstance(features, list) else ())
            )
            raise RuntimeError(
                f"FunASR service lacks required protocol features: {missing}"
            )
    return service_version


def import_funasr_client(client_src: Path | None) -> Any:
    if client_src is not None:
        if not client_src.is_dir():
            raise FileNotFoundError(f"FunASR client src does not exist: {client_src}")
        if str(client_src) not in sys.path:
            sys.path.insert(0, str(client_src))
    try:
        from funasrnano import FunASRClient
    except ImportError as exc:
        raise RuntimeError(
            "FunASR SDK is unavailable; install it or pass --funasr-client-src"
        ) from exc
    return FunASRClient


def _supports_keyword(callable_: Any, keyword: str) -> bool:
    """Return whether a client factory advertises a constructor keyword."""

    try:
        parameters = inspect.signature(callable_).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def validate_funasr_client_contract(client_class: Any) -> None:
    """Fail fast unless the pinned SDK exposes the formal long-form controls."""

    missing_constructor = [
        keyword
        for keyword in ("version_check", "vad_params")
        if not _supports_keyword(client_class, keyword)
    ]
    transcribe_file = getattr(client_class, "transcribe_file", None)
    missing_transcribe = [
        keyword
        for keyword in ("chunk_ms", "delivery_mode", "pacing")
        if transcribe_file is None or not _supports_keyword(transcribe_file, keyword)
    ]
    if missing_constructor or missing_transcribe:
        raise RuntimeError(
            "FunASR SDK lacks the formal long-form protocol contract: "
            f"constructor={missing_constructor}, transcribe_file={missing_transcribe}"
        )


def _wav_duration_s(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            raise ValueError("WAV frame rate must be positive")
        return wav_file.getnframes() / frame_rate


def _transcription_deadline_s(
    wav_path: Path,
    *,
    timeout_s: float | None,
    duration_s: float | None,
) -> float:
    if timeout_s is not None:
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number or None")
        return float(timeout_s)

    if duration_s is None:
        duration_s = _wav_duration_s(wav_path)
    if (
        isinstance(duration_s, bool)
        or not isinstance(duration_s, (int, float))
        or not math.isfinite(duration_s)
        or duration_s < 0
    ):
        raise ValueError("duration_s must be a finite non-negative number or None")
    return max(
        _MIN_TRANSCRIPTION_TIMEOUT_S,
        _TRANSCRIPTION_TIMEOUT_MULTIPLIER * float(duration_s)
        + _TRANSCRIPTION_TIMEOUT_OVERHEAD_S,
    )


async def _transcribe_wav_once(
    client_class: Any,
    wav_path: Path,
    *,
    uri: str,
    language: str,
    chunk_ms: int,
    strict_sdk_contract: bool,
) -> dict[str, Any]:
    """Run one protocol-complete transcription on one fresh connection."""

    segments: list[dict[str, Any]] = []
    done: dict[str, Any] | None = None
    client_options: dict[str, Any] = {
        "hotwords": [],
        "language": language,
        "partial_mode": "off",
        "hotword_config": None,
    }
    if strict_sdk_contract:
        validate_funasr_client_contract(client_class)
    if _supports_keyword(client_class, "version_check"):
        client_options["version_check"] = True
    if _supports_keyword(client_class, "vad_params"):
        client_options["vad_params"] = {"type": "fsmn"}

    async with client_class(uri, **client_options) as client:
        async for event in client.transcribe_file(
            str(wav_path),
            chunk_ms=chunk_ms,
            delivery_mode="offline",
            pacing="none",
        ):
            if not isinstance(event, Mapping):
                raise RuntimeError("FunASR emitted a non-mapping event")
            event_type = event.get("type")
            if done is not None:
                if event_type == "stream_done":
                    raise RuntimeError("FunASR emitted duplicate stream_done")
                raise RuntimeError(
                    f"FunASR emitted {event_type!r} after stream_done"
                )
            if event_type == "segment_final":
                segment = event.get("segment")
                if isinstance(segment, Mapping):
                    segments.append(dict(segment))
            elif event_type == "stream_done":
                done = dict(event)
            elif event_type == "error":
                raise RuntimeError(
                    f"{event.get('code', 'asr_error')}: {event.get('message', '')}"
                )

    if done is None:
        raise RuntimeError("FunASR stream ended without stream_done")
    transcript = "".join(str(segment.get("text", "")) for segment in segments)
    return {
        "status": AsrStatus.OK.value,
        "transcript": transcript,
        "segments": segments,
        "stream_done": done,
    }


async def transcribe_wav(
    client_class: Any,
    wav_path: Path,
    *,
    uri: str,
    language: str,
    chunk_ms: int,
    timeout_s: float | None = None,
    duration_s: float | None = None,
    strict_sdk_contract: bool = False,
) -> dict[str, Any]:
    """Transcribe one WAV in a fresh connection under an outer deadline."""

    try:
        deadline_s = _transcription_deadline_s(
            wav_path,
            timeout_s=timeout_s,
            duration_s=duration_s,
        )
        return await asyncio.wait_for(
            _transcribe_wav_once(
                client_class,
                wav_path,
                uri=uri,
                language=language,
                chunk_ms=chunk_ms,
                strict_sdk_contract=strict_sdk_contract,
            ),
            timeout=deadline_s,
        )
    except TimeoutError:
        return {
            "status": AsrStatus.ERROR.value,
            "error": f"TimeoutError: ASR transcription exceeded {deadline_s:g}s",
        }
    except Exception as exc:  # noqa: BLE001 - ASR failure is per-trial evidence
        return {
            "status": AsrStatus.ERROR.value,
            "error": f"{type(exc).__name__}: {exc}",
        }
