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

Engine and SDK are **version-paired**: both are released from the same git
tag, and the wheel version is derived from that tag. First ask your engine
which version it is:

```bash
curl http://<engine-host>:<health-port>/health    # → {"version": "v0.1.0", ...}
```

**Channel 1 — straight from Git**, at the engine's tag (not on PyPI):

```bash
pip install "qwen3-tts-client @ git+https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git@v0.1.0#subdirectory=client"
```

Over SSH, swap `https://github.com/` for `ssh://git@github.com/` (keep the
`@<tag>#subdirectory=client` suffix).

**Channel 2 — a delivered wheel.** Every engine serves its own matching wheel
at `GET /sdk/` on the health port — whatever engine you reach, the wheel it
hands out fits it:

```bash
curl http://<engine-host>:<health-port>/sdk/      # list the .whl
pip install http://<engine-host>:<health-port>/sdk/qwen3_tts_client-0.1.0-py3-none-any.whl
```

Release wheels are built on a tag with
`bash scripts/bash/release_client_wheel.sh` → `client/dist/`.

From a local checkout: `pip install ./client` (repo root). A mispaired
install fails fast at connect with `ProtocolVersionMismatchError`
(set `QWEN3TTS_SKIP_PROTOCOL_CHECK=1` to downgrade it to a warning).

Extras, by what you connect to / need — add them in the brackets, e.g.
`"qwen3-tts-client[grpc] @ git+https://...#subdirectory=client"` or
`pip install "./client[grpc]"`:

| Extra | Pulls in | Use when |
|-------|----------|----------|
| `grpc` | `grpcio`, `protobuf` | engine-grpc transport |
| `triton` | `tritonclient` | triton-grpc transport |
| `audio` | `numpy` | `synthesize_array()` (ndarray output) |
| `all` | everything above | not sure / want it all |

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

### Authentication and persistent WebSockets

Pass `key` when a PaaS gateway requires Bearer authentication. Its default is
`None`, which injects no credentials:

```python
client = TTSClient.connect(
    "wss://tts.example/v1/ws",
    key="your-key",
    connect_timeout=5.0,
    max_connections=32,
    max_idle_connections=8,
    max_pending_acquires=256,
    acquire_timeout=30.0,
)

# Per-call capabilities timeout, then fill the pool to four idle sockets.
caps = client.get_capabilities(timeout=5.0)
idle_connections = client.prewarm(connections=4, timeout=5.0)
```

`engine-websocket` retains completed physical connections and reuses them for
later logical sessions; concurrent sessions use separate pooled connections.
The `SessionStartRequest.session_id` is correlation data only. The gateway
creates a private engine execution ID for every `start`, so equal client IDs on
different requests cannot replace or cancel each other.
`max_connections` bounds connecting, leased, keepalive-probed, and idle sockets.
Once it is reached, callers enter a bounded FIFO queue. A full queue raises
`PoolSaturatedError`; waiting longer than `acquire_timeout` raises
`PoolAcquireTimeoutError`.
The session boundary is its `done`/`error` event, not socket closure. Against a
legacy gateway without the persistent-protocol marker, the SDK safely discards
the socket instead of pooling it. Engine-error connections are also discarded;
only a successful/cancelled `done` explicitly marked reusable enters the pool.
Idle connections are kept alive every 15 seconds; when keepalive is disabled,
they are probed synchronously before reuse. `connect_timeout` bounds connection
handshakes and these idle-socket probes, independently of the stream receive
timeout. `prewarm()` fills only the missing pool capacity in parallel, caps its
target at `max_idle_connections` and `max_connections`, and returns the actual
idle count on success; its optional `timeout` bounds each capabilities
round-trip. `idle_ttl` and `max_lifetime` are disabled by default (`None` or
`0`); an over-age active connection is retired only after its session ends.
`keepalive_jitter=0.2` spreads maintenance traffic across workers.
Tune this with `reconnect_attempts`, `max_connections`,
`max_idle_connections`, `max_pending_acquires`, `acquire_timeout`, and the
connection-lifecycle settings. Automatic reconnects cover idle connections and new
session setup only. A mid-stream disconnect is surfaced as an error and is never
automatically replayed because that could duplicate audio. Call `client.close()`
when finished, or use `TTSClient` as a context manager.

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
