from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


def pcm16_from_float_audio(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16)


def save_wav(
    audio: np.ndarray,
    path: str | Path,
    *,
    sample_rate: int,
    ensure_parent: bool = True,
) -> Path:
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
