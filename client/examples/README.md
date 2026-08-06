**English** | [中文](README.zh-CN.md)

# Examples

Runnable examples for the `qwen3-tts-client` SDK. Each takes an optional
endpoint argument (default: `ws://localhost:50052/v1/ws`).

First start a Qwen3-TTS endpoint (see the repo README), then:

```bash
# Use the wheel from the engine tag's GitHub or GitLab Release.
pip install "qwen3-tts-client[all] @ https://github.com/X-Square-Robot/Qwen3TTS-Streaming/releases/download/v0.1.0/qwen3_tts_client-0.1.0-py3-none-any.whl"
python quickstart.py                  # one-shot synthesis -> quickstart.wav
python streaming.py                   # incremental text -> streaming.wav
python realtime.py                    # wall-clock aligned frames (WebRTC/playback)
```

Point at a different endpoint:

```bash
python quickstart.py localhost:50051            # engine gRPC
python quickstart.py http://localhost:8000      # Triton HTTP
```

| Example | Shows |
|---------|-------|
| `quickstart.py` | `TTSClient.connect` + `synthesize_bytes`, save WAV |
| `streaming.py` | `open_stream` + `send_text`/`end` + `iter_messages` |
| `realtime.py` | `RealtimeAudioStream` for isochronous playback frames |

> The `speaker` in these examples (`serena`) is model-dependent; use a speaker
> your deployed model supports (custom_voice variants expose named speakers).
