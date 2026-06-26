from __future__ import annotations

from .exceptions import DependencyMissingError


def import_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise DependencyMissingError(
            "synthesize_array requires the 'audio' extra. Install qwen3-tts-client[audio]."
        ) from exc
    return np


def decode_audio_bytes_to_array(audio_bytes: bytes, *, encoding: str):
    np = import_numpy()
    normalized = str(encoding or "pcm_f32").strip().lower()
    if normalized == "pcm_s16le":
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if normalized == "pcm_f32":
        if len(audio_bytes) % 4 != 0:
            raise ValueError("pcm_f32 payload length is not a multiple of 4")
        return np.frombuffer(audio_bytes, dtype=np.float32)
    raise ValueError(f"unsupported audio encoding: {encoding!r}")
