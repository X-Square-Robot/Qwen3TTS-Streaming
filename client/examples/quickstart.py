"""Quickstart — one-shot synthesis, saved to a WAV file.

Prerequisite: a running Qwen3-TTS endpoint (standalone engine or Triton).
See the repo README for how to start one.

    pip install "qwen3-tts-client @ <matching-github-or-gitlab-release-wheel-url>"
    python quickstart.py [endpoint]
    python quickstart.py wss://localhost:50052/v1/ws --tls-ca-file cert.local.pem
    python quickstart.py wss://localhost:50052/v1/ws --insecure  # local only

Use the tag in the engine's ``capabilities.engine_version``; the deployed
service also serves the matching wheel at its public GET /sdk/ endpoint.

Default endpoint: ws://localhost:50052/v1/ws  (native WebSocket)
Other examples: ws://localhost:50053/v1/ws (Triton sidecar native WebSocket),
ws://localhost:50052/v1/realtime (OpenAI Realtime compatibility),
localhost:50051 (legacy engine gRPC), http://localhost:8000 (legacy Triton HTTP).
"""

from __future__ import annotations

import argparse
import struct
import wave

from qwen3tts import TTSClient, SynthesisConfig

DEFAULT_ENDPOINT = "ws://localhost:50052/v1/ws"
TEXT = "你好，欢迎使用 Qwen3-TTS。"
OUT = "quickstart.wav"


def main() -> None:
    args = _parse_args()
    tls_verify = False if args.insecure else (args.tls_ca_file or True)
    # transport defaults to "auto": the SDK probes the endpoint and picks the
    # Native WebSocket first, with Realtime and older transports as fallbacks.
    client = TTSClient.connect(args.endpoint, tls_verify=tls_verify)
    result = client.synthesize_bytes(
        TEXT,
        request=SynthesisConfig(task_type="custom_voice", speaker="serena"),
    )

    fmt = result.audio_format
    print(
        f"transport={result.transport}  encoding={fmt.encoding}  "
        f"sample_rate={fmt.sample_rate}  bytes={len(result.audio_bytes)}"
    )
    print(f"usage={result.details.get('usage', {})}")

    _save_wav(OUT, result.audio_bytes, fmt.encoding, fmt.sample_rate)
    print(f"saved {OUT}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", nargs="?", default=DEFAULT_ENDPOINT)
    tls = parser.add_mutually_exclusive_group()
    tls.add_argument(
        "--tls-ca-file",
        help="CA certificate bundle for HTTPS/WSS verification",
    )
    tls.add_argument(
        "--insecure",
        action="store_true",
        help="disable HTTPS/WSS certificate verification (local debugging only)",
    )
    return parser.parse_args()


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
