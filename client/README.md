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
