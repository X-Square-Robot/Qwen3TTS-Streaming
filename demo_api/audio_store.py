from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = Path(os.environ.get("QWEN_DEMO_AUDIO_DIR", str(REPO_ROOT / "workspace" / "demo_audio")))


class AudioStore:
    def __init__(self, root: Path = AUDIO_DIR) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_pcm_f32_wav(self, pcm_f32: bytes, *, sample_rate: int = 24000, name_hint: str = "audio") -> dict:
        digest = hashlib.sha1(pcm_f32[:65536] + name_hint.encode("utf-8")).hexdigest()[:16]
        filename = f"{safe_name(name_hint)}-{digest}.wav"
        path = self.root / filename
        if not path.exists():
            path.write_bytes(wav_from_pcm_f32(pcm_f32, sample_rate=sample_rate))
        return {
            "id": filename,
            "url": f"/api/v1/audio/{filename}",
            "encoding": "wav",
            "sample_rate": sample_rate,
        }

    def path_for(self, audio_id: str) -> Path | None:
        if "/" in audio_id or "\\" in audio_id or audio_id.startswith("."):
            return None
        path = self.root / audio_id
        if not path.exists() or not path.is_file():
            return None
        return path


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value.lower())
    return cleaned.strip("-")[:64] or "audio"


def wav_from_pcm_f32(pcm_f32: bytes, *, sample_rate: int) -> bytes:
    sample_count = len(pcm_f32) // 4
    data_size = sample_count * 2
    out = bytearray()
    out += b"RIFF"
    out += struct.pack("<I", 36 + data_size)
    out += b"WAVEfmt "
    out += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    out += b"data"
    out += struct.pack("<I", data_size)
    # Vectorized float32 -> int16 (truncation toward zero matches int()); the
    # old per-sample Python loop blocked the event loop (~20ms per 4s of audio,
    # x128 concurrent encodes). np.frombuffer needs a 4-aligned length.
    # float64 intermediate matches the old `int(float(sample) * 32767.0)` path
    # exactly (float32 math would drift by +-1 LSB near truncation boundaries).
    samples = np.frombuffer(pcm_f32[: sample_count * 4], dtype="<f4").astype(np.float64)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    out += pcm16.tobytes()
    return bytes(out)
