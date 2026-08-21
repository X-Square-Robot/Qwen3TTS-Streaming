**English** | [中文](README.zh-CN.md)

# Qwen3-TTS Python Client

`qwen3-tts-client` is a lightweight Python SDK for talking to a Qwen3-TTS
deployment. Native WebSocket is the primary protocol; OpenAI Realtime is a
compatibility endpoint, and three older direct transports remain available as
migration fallbacks behind the same API.

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

- **Native WebSocket first** — `engine-websocket` is the primary transport;
  `openai-realtime` is the OpenAI compatibility endpoint, while `engine-grpc`,
  `triton-grpc`, and `triton-http` are older direct fallbacks behind the same
  `TTSClient`.
- **Auto-detection** — `transport="auto"` (the default) probes the endpoint and
  prefers native WebSocket when the server advertises or accepts it.
- **One-shot, streaming, and realtime** modes.
- **Sync and async** clients (`TTSClient` / `AsyncTTSClient`).
- **Slim dependencies** — the core install only needs `requests` and
  `websocket-client`; gRPC / Triton / numpy are opt-in extras.

## Install

SDK compatibility is determined by the protocol family and major advertised in
capabilities. `engine_version` is diagnostic: release skew emits a warning but
does not reject the connection. First ask the engine which release it is running:

```bash
curl http://<engine-host>:<ws-port>/v1/capabilities
# → {"engine_version": "v0.1.0", ...}
```

**Channel 1 — GitHub/GitLab Release.** Install the wheel attached to the
matching [GitHub Release](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/releases)
or GitLab Release (a private project requires credentials). The deployed
instance's `/demo/#/sdk` page supplies the exact command without a stale
version or filename in this README.

This fetches the wheel, not a Git checkout. GitLab additionally exposes its
wheel through the project PyPI index using the version reported above.

**Channel 2 — from the deployed service.** Every release image embeds the
already published wheel and serves it at the public `GET /sdk/` endpoint:

```bash
curl https://<public-service-base>/sdk/      # list the .whl
pip install "https://<public-service-base>/sdk/<wheel-filename>"
```

From a local checkout: `pip install ./client` (repo root). An incompatible
protocol family or major fails fast at connect with
`ProtocolVersionMismatchError`; revisions within one major are compatible and
engine/SDK release skew only emits `RuntimeWarning`. Set
`QWEN3TTS_SKIP_PROTOCOL_CHECK=1` to downgrade protocol incompatibility to a
warning.

Extras are selected by what you connect to / need. Add them to the Release
direct reference (`qwen3-tts-client[grpc] @ https://...whl`) or a local install
(`pip install "./client[grpc]"`):

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
print(result.details["usage"])  # terminal input/output token usage for billing
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

# Available after response.done, including partial usage for cancellation/failure.
print(session.usage, session.response_id, session.response_status)
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
| `ws://localhost:50052/v1/ws` | `engine-websocket` (native primary) |
| `ws://localhost:50052/v1/realtime` | `openai-realtime` (standalone) |
| `ws://localhost:50053/v1/realtime` | `openai-realtime` (Triton sidecar) |
| `localhost:50051` | `engine-grpc` |
| `http://localhost:8000` | `triton-http` / `triton-grpc` |

### Realtime compatibility and usage

Complete-text `synthesize_bytes()` uses standard Realtime
`conversation.item.create` and `response.create` events. Incremental
`open_stream()` uses the advertised `qwen.input_text_buffer.v1` extension for
append/commit while audio is received concurrently. A server that does not
advertise that extension is still usable for one-shot synthesis, but the SDK
rejects incremental streaming during session setup.

`response.done.response.usage` is exposed as `result.details["usage"]` for
one-shot calls and as `session.usage` after the terminal event for streaming.
A configured server-side billing ledger remains authoritative when a client
disconnects before receiving that event.

The OpenAI Realtime compatibility transport does not emit a deprecation
warning. The three older direct transports—`engine-grpc`, `triton-grpc`, and
`triton-http`—emit one `FutureWarning` per process and transport.
`QWEN3TTS_SUPPRESS_LEGACY_TRANSPORT_WARNING=1` can temporarily silence those
warnings. Native WebSocket advertises `stream_resume_v1`; Realtime advertises
`qwen.response_resume.v1`. A gateway that declares neither fails explicitly
rather than silently re-synthesizing audio.

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

Local containers use `http://` / `ws://` by default, so ordinary local
development needs no TLS option. When direct self-signed HTTPS/WSS is enabled,
pass the certificate path to retain strict verification, or disable checking
only for temporary local debugging:

```python
# Recommended: trust this self-signed certificate or private CA
client = TTSClient.connect(
    "wss://localhost:50052/v1/ws",
    tls_verify="/path/to/cert.local.pem",
)

# Local debugging only; never use in production
client = TTSClient.connect(
    "wss://localhost:50052/v1/ws",
    tls_verify=False,
)
```

The default `tls_verify=True` uses the system trust store. One policy covers
HTTPS discovery, WSS dials, prewarming, and reconnects.
`verify_protocol=False` only skips protocol compatibility checking; it is
unrelated to TLS. The old `verify` keyword remains an alias for that protocol
option.

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
connection-lifecycle settings. `reconnect_attempts` covers initial connection
setup; active streams have a separate recovery budget. By default,
`active_stream_resume=True`, `stream_resume_attempts=2`,
`stream_resume_timeout=10.0`, and `stream_resume_ack_interval=8`.

On a resume-capable native WebSocket or OpenAI Realtime gateway, a transient
disconnect keeps the same logical engine session alive. Text is de-duplicated
with cumulative sequence ACKs and output resumes from an acknowledged
delivery/sample cursor, so the SDK does not re-synthesize from the beginning or
enqueue audio twice. Native WebSocket advertises `stream_resume_v1`; Realtime
advertises `qwen.response_resume.v1`. Recovery is bounded by the gateway's
grace period and replay window. Expired tokens, exhausted retries, protocol
gaps, server restarts, and routing to a different replica fail explicitly. A
legacy gateway retains the historical fail-fast behavior. Call `client.close()`
when finished, or use `TTSClient` as a context manager.

## Examples

Runnable scripts in [`examples/`](examples/) — start an endpoint first, then:

```bash
python examples/quickstart.py                  # one-shot     -> quickstart.wav
python examples/streaming.py                   # incremental  -> streaming.wav
python examples/realtime.py                    # wall-clock aligned frames
python examples/quickstart.py localhost:50051  # point at engine gRPC
python examples/quickstart.py wss://localhost:50052/v1/ws \
  --tls-ca-file /path/to/cert.local.pem
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

Start with the [5-minute setup](../docs/user/quickstart.md). Continue to
[Advanced configuration](../docs/user/advanced_configuration.md) only when you need VAD, delivery,
or incremental-input controls.
