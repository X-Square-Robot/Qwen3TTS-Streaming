# Qwen3-TTS Python Client SDK

`qwen3-tts-client` is a lightweight Python SDK for connecting to Qwen3-TTS deployments through one unified API.

Supported transports:

- `engine-websocket`
- `engine-grpc`
- `triton-grpc`
- `triton-http`

Default behavior uses `transport="auto"` and probes the endpoint before binding to a concrete adaptor.

## Install

Core package:

```bash
pip install qwen3-tts-client
```

With extras:

```bash
pip install qwen3-tts-client[grpc]
pip install qwen3-tts-client[triton]
pip install qwen3-tts-client[audio]
pip install qwen3-tts-client[all]
```

## Quick Start

```python
from qwen3_tts_client import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice"),
)
print(result.audio_format)
print(len(result.audio_bytes))
```

Streaming session:

```python
from qwen3_tts_client import TTSClient, SessionStartRequest, SynthesisConfig

client = TTSClient.connect("localhost")
session = client.open_stream(
    SessionStartRequest(
        session_id="demo-session",
        config=SynthesisConfig(task_type="custom_voice"),
    )
)
session.send_text("你好，")
session.send_text("这是流式输入。")
session.end()
for message in session.iter_messages():
    print(type(message).__name__, getattr(message, "meta", {}))
```

More details:

- project manual: `docs/zh/client_sdk.md`
- public API docs live in `qwen3_tts_client.__init__`

## Protocol Layer

The SDK includes `qwen3_tts_protocol`, a shared protocol package that defines
wire-format types (AudioFormat, SynthesisConfig, StreamEvent, …) and
Triton-specific types (TtsRequest, build_payload, TraceEvent, RunResult, …).
Both the client SDK and other project components (demo_api, tests/tools) import
from this single source of truth.

```python
from qwen3_tts_protocol import AudioFormat, SynthesisConfig
from qwen3_tts_protocol.schemas import TraceEvent, RunResult
from qwen3_tts_protocol.triton_types import TtsRequest, build_payload
from qwen3_tts_protocol.audio import save_wav, StreamResult
```

## Realtime Audio Stream

When consuming audio for real-time playback (e.g. feeding a WebRTC media
track or a local audio device), the engine's irregular output rhythm can
cause underruns.  `RealtimeAudioStream` wraps a streaming session and
produces an isochronous (wall-clock aligned) audio flow, automatically
inserting silence frames to cover gaps:

```python
from qwen3_tts_client import TTSClient, RealtimeAudioStream, SessionStartRequest, SynthesisConfig

client = TTSClient.connect("localhost")
session = client.open_stream(
    SessionStartRequest(
        session_id="webrtc-feed",
        config=SynthesisConfig(task_type="custom_voice"),
    )
)
session.send_text("你好，欢迎使用实时语音合成。")
session.end()

# 20 ms frames, silence-filled — ready for WebRTC / local playback
for frame in RealtimeAudioStream(session):
    if frame.is_silence:
        continue  # or handle silence explicitly
    webrtc_track.write(frame.data)
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fill_silence` | `True` | Insert silence when the engine is late; `False` = passthrough |
| `chunk_s` | `0.02` | Output granularity in seconds (20 ms = WebRTC Opus frame) |
| `sample_rate` | `24000` | Audio sample rate in Hz |
