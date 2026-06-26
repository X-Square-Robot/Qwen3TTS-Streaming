"""Shared audio utilities: WAV writing, PCM conversion, audio decoding.

This module is the single source of truth for audio I/O helpers used
across client, demo_api, scripts, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import wave


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# PCM conversion
# ---------------------------------------------------------------------------

def pcm16_from_float_audio(audio: Any) -> Any:
    """Convert float32 audio samples to int16 PCM.

    Accepts any array-like that numpy can interpret as float32.
    Returns a numpy int16 array.
    """
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16)


# ---------------------------------------------------------------------------
# WAV writing
# ---------------------------------------------------------------------------

def save_wav(
    audio: Any,
    path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    ensure_parent: bool = True,
) -> Path:
    """Write a float32 numpy array to a 16-bit PCM WAV file.

    Parameters
    ----------
    audio:
        Float32 audio samples in [-1.0, 1.0].
    path:
        Output WAV file path.
    sample_rate:
        Sample rate in Hz (default 24000).
    ensure_parent:
        Create parent directories if they don't exist.

    Returns
    -------
    Path
        The written file path.
    """
    out_path = Path(path)
    if ensure_parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    pcm16 = pcm16_from_float_audio(audio)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return out_path


# ---------------------------------------------------------------------------
# Audio byte decoding
# ---------------------------------------------------------------------------

def decode_obj(value: Any) -> str:
    """Decode bytes or str to str — helper for Triton tensor values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_audio_bytes(raw: bytes, audio_format: dict[str, Any] | None = None) -> Any:
    """Decode raw audio bytes to a float32 numpy array.

    Parameters
    ----------
    raw:
        Raw audio bytes from a Triton response.
    audio_format:
        Dict with ``encoding`` key (``"pcm_f32"`` or ``"pcm_s16le"``).

    Returns
    -------
    numpy.ndarray
        Float32 audio samples.
    """
    import numpy as np

    encoding = (audio_format or {}).get("encoding", "pcm_f32")
    if str(encoding) == "pcm_s16le":
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return np.frombuffer(raw, dtype=np.float32)


# ---------------------------------------------------------------------------
# Stream result (lightweight)
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    """Lightweight result holder for a single Triton streaming inference.

    This is kept in the protocol layer because tools/validation and the old
    ``tests.support.triton_streaming`` module both need it.
    """

    text: str
    session_id: str = ""
    first_chunk_ms: float | None = None
    total_ms: float = 0.0
    num_chunks: int = 0
    total_samples: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    audio: Any = None  # numpy.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return self.total_samples / DEFAULT_SAMPLE_RATE if self.total_samples > 0 else 0.0

    @property
    def rtf(self) -> float:
        if self.duration_sec <= 0 or self.total_ms <= 0:
            return 0.0
        return (self.total_ms / 1000) / self.duration_sec


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "StreamResult",
    "decode_audio_bytes",
    "decode_obj",
    "pcm16_from_float_audio",
    "save_wav",
]
