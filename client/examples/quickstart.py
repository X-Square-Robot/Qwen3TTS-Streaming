"""Quickstart — one-shot synthesis, saved to a WAV file.

Prerequisite: a running Qwen3-TTS endpoint (standalone engine or Triton).
See the repo README for how to start one.

    pip install "qwen3-tts-client @ git+https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git@<tag>#subdirectory=client"
    python quickstart.py [endpoint]

Use the tag matching your engine (its /health reports "version"); the engine
also serves the matching wheel at GET /sdk/ on the health port.

Default endpoint: ws://localhost:50052/v1/ws  (standalone engine WebSocket)
Other examples: localhost:50051 (engine gRPC), http://localhost:8000 (Triton).
"""

from __future__ import annotations

import struct
import sys
import wave

from qwen3tts import TTSClient, SynthesisConfig

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:50052/v1/ws"
TEXT = "你好，欢迎使用 Qwen3-TTS。"
OUT = "quickstart.wav"


def main() -> None:
    # transport defaults to "auto": the SDK probes the endpoint and picks the
    # right adapter (engine-websocket / engine-grpc / triton-grpc / triton-http).
    client = TTSClient.connect(ENDPOINT)
    result = client.synthesize_bytes(
        TEXT,
        request=SynthesisConfig(task_type="custom_voice", speaker="serena"),
    )

    fmt = result.audio_format
    print(
        f"transport={result.transport}  encoding={fmt.encoding}  "
        f"sample_rate={fmt.sample_rate}  bytes={len(result.audio_bytes)}"
    )

    _save_wav(OUT, result.audio_bytes, fmt.encoding, fmt.sample_rate)
    print(f"saved {OUT}")


def _save_wav(path: str, pcm: bytes, encoding: str, sample_rate: int) -> None:
    """Write engine PCM (pcm_f32 or pcm_s16le) to a 16-bit WAV."""
    if encoding == "pcm_f32":
        import array

        floats = array.array("f")
        floats.frombytes(pcm)
        pcm16 = b"".join(
            struct.pack("<h", max(-32768, min(32767, int(s * 32767)))) for s in floats
        )
    else:  # pcm_s16le
        pcm16 = pcm
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)


if __name__ == "__main__":
    main()
