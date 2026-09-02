"""Minimal mono-WAV I/O shared by ASR scoring and blind packaging."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV as contiguous float32 and reject implicit channel mixing."""

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - validation preflight owns this
        raise RuntimeError("long-form scoring requires soundfile") from exc
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.shape[1] != 1:
        raise ValueError(f"expected mono WAV, got {samples.shape[1]} channels: {path}")
    return np.ascontiguousarray(samples[:, 0]), int(sample_rate)


def write_mono_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write review audio as deterministic PCM-16 WAV."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("long-form scoring requires soundfile") from exc
    sf.write(str(path), audio, sample_rate, subtype="PCM_16", format="WAV")


def slice_ms(
    samples: np.ndarray,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
) -> np.ndarray:
    """Return a bounded, half-open millisecond slice."""

    if sample_rate <= 0 or start_ms < 0 or end_ms <= start_ms:
        raise ValueError("invalid audio slice")
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    start = min(audio.size, round(start_ms * sample_rate / 1000.0))
    end = min(audio.size, round(end_ms * sample_rate / 1000.0))
    if end <= start:
        end = min(audio.size, start + 1)
    return np.ascontiguousarray(audio[start:end])


__all__ = ["load_mono_wav", "slice_ms", "write_mono_wav"]
