**English** | [中文](README.zh-CN.md)

# Qwen3-TTS Python Client

`qwen3-tts-client` is a lightweight Python SDK for talking to a Qwen3-TTS
deployment. One import root, one API, four transports — point it at an endpoint
and synthesize speech.

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
result = client.synthesize_bytes("你好，欢迎使用 Qwen3-TTS。",
                                 request=SynthesisConfig(task_type="custom_voice"))
print(result.audio_format, len(result.audio_bytes))
```

> Everything you need is under the single `qwen3tts` package — you never import
> anything else for normal use.

## Features

- **One API, four transports** — `engine-websocket`, `engine-grpc`,
  `triton-grpc`, `triton-http`, all behind the same `TTSClient`.
- **Auto-detection** — `transport="auto"` (the default) probes the endpoint and
  binds the right adapter, so you usually just pass a URL.
- **One-shot, streaming, and realtime** modes.
- **Sync and async** clients (`TTSClient` / `AsyncTTSClient`).
- **Slim dependencies** — the core install only needs `requests`; gRPC / Triton /
  numpy are opt-in extras.

## Install

```bash
pip install qwen3-tts-client          # core (WebSocket + HTTP transports)
```

Extras, by what you connect to / need:

| Extra | Install | Pulls in | Use when |
|-------|---------|----------|----------|
| `grpc` | `pip install "qwen3-tts-client[grpc]"` | `grpcio`, `protobuf` | engine-grpc transport |
| `triton` | `pip install "qwen3-tts-client[triton]"` | `tritonclient` | triton-grpc transport |
| `audio` | `pip install "qwen3-tts-client[audio]"` | `numpy` | `synthesize_array()` (ndarray output) |
| `all` | `pip install "qwen3-tts-client[all]"` | everything above | not sure / want it all |

Requires Python 3.10+.

## Quick start

### One-shot

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice", speaker="serena"),
)
# result.audio_bytes is raw PCM; result.audio_format tells you encoding + rate.
print(result.transport, result.audio_format.encoding, result.audio_format.sample_rate)
```

Need a numpy array instead of bytes (requires the `audio` extra)?

```python
arr = client.synthesize_array("你好。", request=SynthesisConfig(task_type="custom_voice"))
print(arr.audio_array.shape)
```

### Streaming

Feed text incrementally (e.g. as an upstream LLM emits it) and consume audio as
it arrives:

```python
from qwen3tts import TTSClient, SessionStartRequest, SynthesisConfig, AudioChunk, StreamEvent

client = TTSClient.connect("ws://localhost:50052/v1/ws")
session = client.open_stream(
    SessionStartRequest(session_id="demo", config=SynthesisConfig(task_type="custom_voice"))
)
session.send_text("你好，")
session.send_text("这是流式输入。")
session.end()

for message in session.iter_messages():
    if isinstance(message, AudioChunk):
        ...  # message.pcm_bytes
    elif isinstance(message, StreamEvent):
        print("event:", message.type)
```

### Realtime playback (WebRTC / audio device)

The engine emits audio in an irregular rhythm. `RealtimeAudioStream` wraps a
session and yields fixed-size frames on a wall-clock cadence, inserting silence
to cover gaps so a playback device / WebRTC track never underruns:

```python
from qwen3tts import TTSClient, RealtimeAudioStream, SessionStartRequest, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
session = client.open_stream(
    SessionStartRequest(session_id="webrtc", config=SynthesisConfig(task_type="custom_voice"))
)
session.send_text("你好，欢迎使用实时语音合成。")
session.end()

# 20 ms frames, silence-filled — ready for WebRTC / local playback
for frame in RealtimeAudioStream(session, chunk_s=0.02, fill_silence=True):
    if frame.is_silence:
        continue
    webrtc_track.write(frame.data)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fill_silence` | `True` | Insert silence when the engine is late; `False` = passthrough |
| `chunk_s` | `0.02` | Frame size in seconds (20 ms = WebRTC Opus frame) |
| `sample_rate` | `24000` | Audio sample rate in Hz |

### Async

`AsyncTTSClient` mirrors the sync API with `await`:

```python
from qwen3tts import AsyncTTSClient, SynthesisConfig

client = await AsyncTTSClient.connect("ws://localhost:50052/v1/ws")
result = await client.synthesize_bytes("你好。", request=SynthesisConfig(task_type="custom_voice"))
# streaming: session = await client.aopen_stream(SessionStartRequest(...))
```

## Transports

`TTSClient.connect(endpoint, transport="auto")` accepts a URL or `host:port` and
auto-detects the backend. To pin it explicitly, pass `transport=`:

| Endpoint example | Detected transport |
|------------------|--------------------|
| `ws://localhost:50052/v1/ws` | `engine-websocket` |
| `localhost:50051` | `engine-grpc` |
| `http://localhost:8000` | `triton-http` / `triton-grpc` |

## Examples

Runnable scripts in [`examples/`](examples/) — start an endpoint first, then:

```bash
python examples/quickstart.py                  # one-shot     -> quickstart.wav
python examples/streaming.py                   # incremental  -> streaming.wav
python examples/realtime.py                    # wall-clock aligned frames
python examples/quickstart.py localhost:50051  # point at engine gRPC
```

## API reference

The full public API is whatever `qwen3tts` exports — `import qwen3tts;
help(qwen3tts)` or read `qwen3tts.__all__`. Headline names:

- **Clients:** `TTSClient`, `AsyncTTSClient`
- **Requests / config:** `SynthesisConfig`, `SessionStartRequest`, `AudioFormat`,
  `VADPolicy`, `OutputPolicy`
- **Stream messages:** `AudioChunk`, `StreamEvent`, `StreamTextChunk`
- **Results:** `BytesResult`, `ArrayResult`, `Capabilities`
- **Realtime:** `RealtimeAudioStream`, `TimedAudio`
- **Exceptions:** `TTSClientError` and subclasses (`TransportNotSupportedError`,
  `TransportProbeError`, `ProtocolError`, `DependencyMissingError`,
  `StreamClosedError`)

Optional latency/timing diagnostics live in a separate submodule and are not
needed for normal use:

```python
from qwen3tts.diagnostics import LatencyAnalyzer, ServerTimingReport
```

Project manual: `docs/user/client_sdk.md`.

---

### For contributors

This SDK is built on top of `qwen3tts_protocol`, a dependency-free package that
holds the wire-format types and is the single source of truth shared by the
client, the engine, and the demo server. **Client users do not need it** — every
type it defines is re-exported from `qwen3tts`. Only touch `qwen3tts_protocol`
directly when working on the protocol itself or on server-side components.
