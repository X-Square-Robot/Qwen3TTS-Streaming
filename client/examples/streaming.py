"""Streaming session — feed text incrementally, collect audio chunks.

This mirrors the production path: text is pushed token/clause at a time and the
engine streams audio back. Run a Qwen3-TTS endpoint first, then:

    python streaming.py [endpoint]

Default endpoint: ws://localhost:50052/v1/ws
"""

from __future__ import annotations

import struct
import sys
import wave

from qwen3tts import (
    AudioChunk,
    SessionStartRequest,
    StreamEvent,
    SynthesisConfig,
    TTSClient,
)

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:50052/v1/ws"
OUT = "streaming.wav"


def main() -> None:
    client = TTSClient.connect(ENDPOINT)

    config = SynthesisConfig(task_type="custom_voice", speaker="serena")
    session = client.open_stream(
        SessionStartRequest(session_id="streaming-demo", config=config)
    )

    # Push text incrementally (e.g. as an upstream LLM emits it), then close.
    session.send_text("你好，")
    session.send_text("这是流式语音合成示例。")
    session.end()

    pcm = bytearray()
    sample_rate = config.audio.sample_rate
    encoding = config.audio.encoding
    for message in session.iter_messages():
        if isinstance(message, AudioChunk):
            pcm += message.pcm_bytes
            if message.audio and message.audio.sample_rate:
                sample_rate = int(message.audio.sample_rate)
                encoding = message.audio.encoding or encoding
        elif isinstance(message, StreamEvent):
            print(
                f"event: {message.type}"
                + (f" — {message.message}" if message.message else "")
            )
            if message.type == "error":
                return

    print(f"received {len(pcm)} bytes  encoding={encoding}  sample_rate={sample_rate}")
    _save_wav(OUT, bytes(pcm), encoding, sample_rate)
    print(f"saved {OUT}")


def _save_wav(path: str, pcm: bytes, encoding: str, sample_rate: int) -> None:
    if encoding == "pcm_f32":
        import array

        floats = array.array("f")
        floats.frombytes(pcm)
        pcm = b"".join(
            struct.pack("<h", max(-32768, min(32767, int(s * 32767)))) for s in floats
        )
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


if __name__ == "__main__":
    main()
