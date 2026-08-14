"""Optional FunASR client loading and per-WAV transcription."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import AsrStatus


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


async def transcribe_wav(
    client_class: Any,
    wav_path: Path,
    *,
    uri: str,
    language: str,
    chunk_ms: int,
) -> dict[str, Any]:
    """Transcribe one WAV in a fresh FunASR connection/stream."""

    segments: list[dict[str, Any]] = []
    done: dict[str, Any] | None = None
    try:
        async with client_class(
            uri,
            hotwords=[],
            language=language,
            partial_mode="off",
            hotword_config=None,
        ) as client:
            async for event in client.transcribe_file(
                str(wav_path),
                chunk_ms=chunk_ms,
                delivery_mode="offline",
                pacing="none",
            ):
                event_type = event.get("type")
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
    except Exception as exc:  # noqa: BLE001 - ASR failure is per-trial evidence
        return {
            "status": AsrStatus.ERROR.value,
            "error": f"{type(exc).__name__}: {exc}",
        }
