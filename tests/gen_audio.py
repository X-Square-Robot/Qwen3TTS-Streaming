"""Generate WAV audio samples via Triton TTS for listening evaluation.

Usage:
    python tests/gen_audio.py

Requires:
    - Triton server running: bash scripts/bash/deploy.sh run --gateway triton
    - tritonclient[grpc]: pip install tritonclient[grpc]

Outputs WAV files to workspace/audio_samples/
"""
import json
import struct
from pathlib import Path

import numpy as np
from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports("repo", "scripts_python")

from tests.support.triton_streaming import build_request_payload, infer_stream

OUTPUT_DIR = REPO_ROOT / "workspace" / "audio_samples"
SAMPLE_RATE = 24000

GRPC_HOST = "localhost"
GRPC_PORT = 8001

SAMPLES = [
    {
        "name": "01_greeting",
        "text": "你好，欢迎使用Qwen3-TTS语音合成系统。",
        "speaker": "zhitian",
    },
    {
        "name": "02_weather",
        "text": "今天天气晴朗，万里无云，非常适合外出活动。",
        "speaker": "zhitian",
    },
    {
        "name": "03_story",
        "text": "从前有座山，山上有座庙，庙里有个老和尚在给小和尚讲故事。",
        "speaker": "zhitian",
    },
    {
        "name": "04_english",
        "text": "Hello, this is a test of the Qwen3 text to speech system. How does it sound?",
        "speaker": "zhitian",
    },
    {
        "name": "05_mixed",
        "text": "深度学习领域的Transformer架构，自2017年提出以来，已经彻底改变了自然语言处理的格局。",
        "speaker": "zhitian",
    },
]


def make_wav(samples_f32: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Convert float32 samples to 16-bit PCM WAV bytes."""
    pcm16 = np.clip(samples_f32 * 32767, -32768, 32767).astype(np.int16)
    n = pcm16.size
    buf = bytearray()
    buf += b"RIFF"
    buf += struct.pack("<I", 36 + n * 2)
    buf += b"WAVEfmt "
    buf += struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    buf += b"data"
    buf += struct.pack("<I", n * 2)
    buf += pcm16.tobytes()
    return bytes(buf)


def stream_tts(client, text: str, speaker: str, timeout: float = 60.0):
    """Send TTS request and collect all audio chunks."""
    grpcclient = sys.modules["tritonclient.grpc"]
    req_dict = build_request_payload(
        text=text,
        task_type="custom_voice",
        speaker=speaker,
    )
    stream = infer_stream(client, grpcclient, req_dict, timeout=timeout)
    chunks = [stream.audio] if stream.audio is not None and stream.audio.size else []
    first_sec = (stream.first_chunk_ms / 1000.0) if stream.first_chunk_ms is not None else None
    total = stream.total_ms / 1000.0
    return chunks, first_sec, total, stream.error


def main():
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print("ERROR: tritonclient[grpc] not installed.")
        print("  pip install tritonclient[grpc]")
        sys.exit(1)

    client = grpcclient.InferenceServerClient(url=f"{GRPC_HOST}:{GRPC_PORT}")
    if not client.is_server_ready():
        print(f"ERROR: Triton server not ready at {GRPC_HOST}:{GRPC_PORT}")
        print("  bash scripts/bash/deploy.sh run --gateway triton")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(SAMPLES)} audio samples → {OUTPUT_DIR}/\n")

    for sample in SAMPLES:
        name = sample["name"]
        text = sample["text"]
        speaker = sample["speaker"]

        print(f"  [{name}] \"{text[:40]}...\"")
        chunks, first_sec, total_sec, err = stream_tts(client, text, speaker)

        if err:
            print(f"    ERROR: {err}")
            continue

        if not chunks:
            print(f"    WARNING: no audio chunks received")
            continue

        audio = np.concatenate(chunks)
        duration = audio.size / SAMPLE_RATE
        wav_path = OUTPUT_DIR / f"{name}.wav"
        wav_path.write_bytes(make_wav(audio))

        print(f"    → {wav_path.name}  "
              f"duration={duration:.2f}s  "
              f"first_chunk={first_sec*1000:.0f}ms  "
              f"total={total_sec*1000:.0f}ms  "
              f"chunks={len(chunks)}")

    print(f"\nDone. Files in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
