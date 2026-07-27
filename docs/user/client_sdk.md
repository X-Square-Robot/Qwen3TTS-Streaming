**English** | [中文](client_sdk.zh-CN.md)

# Python Client SDK

## Goal

This SDK targets external callers, providing a unified, lightweight Python client that can be installed directly via `pip install`.

Supported service entry points:

- `engine-websocket`
- `engine-grpc`
- `triton-grpc`
- `triton-http`

The default behavior is `transport="auto"`: the client first probes, then binds to a specific adaptor.

## Layout and Publishing

The SDK lives as an independent subproject under the repository's [`client/`](../../client) directory:

- Packaging config: [`client/pyproject.toml`](../../client/pyproject.toml)
- Source entry point: [`client/src/qwen3tts`](../../client/src/qwen3tts)
- Shared protocol layer: [`client/src/qwen3tts_protocol`](../../client/src/qwen3tts_protocol)
- SDK unit tests: [`client/tests`](../../client/tests)

The purpose of this is to avoid packing heavy server-side dependencies and deployment logic into the client wheel.

## Installation

### Version pairing

Engine and SDK are released **from the same git tag**: the engine image bakes
the tag in at build time (`git describe`), and the wheel version is derived
from the same tag (hatch-vcs). The engine advertises its release on the
versioned **capabilities** surface as `engine_version` (`/health` is a pure
liveness probe and carries no version). To find out what a running engine is:

```bash
curl http://<engine-host>:<ws-port>/v1/capabilities
# → {"loaded_model_type": "...", "engine_version": "v0.1.0", "protocol_version": "...", ...}
```

Install the SDK at that same version. `connect()` reads the server's
capabilities and fails fast on a mismatch — `ProtocolVersionMismatchError` (the
wire-protocol generation) or `EngineVersionMismatchError` (the engine/SDK
release). Set `QWEN3TTS_SKIP_PROTOCOL_CHECK=1` to downgrade either to a warning
for deliberate cross-version experiments, or pass `verify=False` to `connect()`
to skip the connect-time capabilities check entirely.

### Channel 1 — install from Git (developers with repo access)

Pin the engine's tag in the URL (the package is not published to PyPI):

```bash
pip install "qwen3-tts-client @ git+https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git@v0.1.0#subdirectory=client"
```

Over SSH, swap `https://github.com/` for `ssh://git@github.com/`. Optional
extras go inside the brackets (`[grpc]` / `[triton]` / `[audio]` / `[all]`):

```bash
pip install "qwen3-tts-client[all] @ git+https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git@v0.1.0#subdirectory=client"
```

### Channel 2 — delivered wheel (no repo access needed)

Every engine deployment serves the wheel built from its own checkout at
`GET /sdk/` on the health port — fetching from the engine you talk to makes a
version mismatch impossible:

```bash
curl http://<engine-host>:<health-port>/sdk/      # list available wheels
pip install http://<engine-host>:<health-port>/sdk/qwen3_tts_client-0.1.0-py3-none-any.whl
```

Standalone release wheels are built on a tag:

```bash
git tag v0.2.0
bash scripts/bash/release_client_wheel.sh         # → client/dist/*.whl
```

The script refuses a dirty tree or an untagged commit, so a delivered wheel's
version always names the exact source it was built from.

### Local checkout (development)

```bash
cd client
pip install .          # extras likewise: pip install ".[grpc]"
```

Dependency strategy:

- `core`: `requests` + `websocket-client` (pure-Python)
- `grpc`: runtime required for standalone gRPC
- `triton`: runtime required for Triton gRPC / HTTP
- `audio`: `numpy`, used by `synthesize_array`

Notes on the websocket transport (since the migration to `websocket-client`):

- The `/sdk/` wheel channel serves the client wheel only; `pip install` still
  fetches `requests` / `websocket-client` from your package index. On
  restricted networks, pre-install them or point pip at a local mirror.
- `ws://` / `wss://` connections now honor the standard `http_proxy` /
  `https_proxy` / `no_proxy` environment variables (same as `requests`).
  Previously the websocket path always connected directly. If your deployment
  sets a corporate proxy, make sure `no_proxy` covers the engine host, or the
  connection will be tunneled through (and possibly rejected by) the proxy.

## Quick Start

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("localhost")
caps = client.get_capabilities()
print(caps.loaded_model_type)

result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice"),
)
print(result.audio_format)
print(len(result.audio_bytes))
```

For the `engine-websocket` transport, `timeout` is the receive-idle budget for
an established request. Set `connect_timeout` separately when a failed network
handshake should release the calling thread sooner; if omitted, it defaults to
`timeout` for backward compatibility:

```python
client = TTSClient.connect(
    "ws://localhost:50052/v1/ws",
    timeout=120.0,
    connect_timeout=5.0,
)
```

### PaaS authentication and connection reuse

When a PaaS gateway requires a Bearer header, pass `key`. Its default of
`None` means that the SDK injects no credentials; the engine itself does not
validate this header:

```python
client = TTSClient.connect(
    "wss://tts.example/v1/ws",
    key="your-key",  # Authorization: Bearer your-key
)
```

One physical `engine-websocket` connection carries multiple logical sessions
serially; concurrent sessions lease separate pooled connections. Defaults are:

- `reconnect_attempts=1`: retry a new physical connection or failed initial
  `start` write once;
- `max_idle_connections=8`: retain at most eight idle connections;
- `keepalive_interval=15.0`: probe idle connections every 15 seconds; use `0`
  to disable it.

The logical boundary is the terminal `done`/`error` event, not WebSocket
closure. A persistent gateway marks a safely reusable successful/cancelled
`done` with `websocket_connection_reusable=true`. Engine errors are closed and
reconnected; if an older gateway omits the marker, the SDK likewise discards
the socket and safely falls back to reconnecting.

If background keepalive (or the synchronous probe used when keepalive is
disabled) finds that a gateway reaped an idle socket, the SDK opens a
replacement automatically. A disconnect after text or audio has entered an
active session produces an explicit `error` and is not replayed: the current
protocol has no text/audio acknowledgement or resume token, and blind recovery
could duplicate audio. Call `client.close()` when finished, preferably through
the context manager.

## Unified Streaming Interface

```python
from qwen3tts import SessionStartRequest, SynthesisConfig, TTSClient

client = TTSClient.connect("ws://localhost:50052/v1/ws")
session = client.open_stream(
    SessionStartRequest(
        session_id="demo-session",
        config=SynthesisConfig(task_type="custom_voice"),
    )
)

session.send_text("你好，")
session.send_text("这是统一流式协议。")
session.stop()  # same as compatibility API session.end(): stop input and drain audio

for message in session.iter_messages():
    print(type(message).__name__, getattr(message, "meta", {}))
```

Relays can bound a remote stream that accepted `end()` or `cancel()` but never
returned a terminal event with
`session.iter_messages(post_send_idle_timeout=30.0)`. Silence before the send
side closes is not counted, and each received message resets the idle budget.
Use `session.close(reason="worker shutdown")` for a hard local stop; it sends a
best-effort cancel, closes the transport when supported, and immediately
unblocks message consumers. Async sessions expose the matching `aclose()` and
`aiter_messages(post_send_idle_timeout=...)` methods.

## Auto-Probing Rules

When `transport=` is set explicitly, no probing is done and the specified adaptor is used directly.

When `transport="auto"`:

- `ws://` / `wss://`: directly resolved as `engine-websocket`
- `http://` / `https://`: first probe standalone `GET /v1/capabilities`, then on failure probe Triton HTTP `/v2/health/live`, `/v2/health/ready`, `/v2/models/<model>/ready`
- Bare `host:port`: probe standalone first by port rules, then Triton, then fall back to HTTP
- Bare `host`: automatically expand to the default candidate ports `50052`, `50051`, `8001`, `8000`

The probing results are exposed at:

- `client.resolved_transport`
- `client.probe_report`
- `client.detected_transport`

## Triton HTTP Streaming Semantics

Triton HTTP itself does not support true decoupled streaming infer.

Therefore the SDK's unified strategy is:

- `synthesize_bytes` / `synthesize_array`: go directly through a single HTTP infer
- `open_stream(...)`: buffer `start/text/end` locally
- Only after calling `end()` is a single HTTP infer triggered
- The returned session is explicitly marked `degraded_to_oneshot=True`

This means:

- It does not pretend to be server-side incremental synthesis
- But upper-layer calling code can still reuse the same session API

## Current Status

The current release has completed these structural goals:

- The SDK code is consolidated into the `client/` subproject
- A unified sync / async façade is provided
- The shared protocol layer `qwen3tts_protocol` is introduced
- An auto-detect skeleton and four adaptor entry points are provided
- Explicit streaming-degradation semantics are provided for `triton-http`

It is still recommended to treat it as v1 alpha:

- The transport adaptation is in place, but more complete integration validation still needs to be added
- The `engine-websocket` / `engine-grpc` paths are closest to the real contract of the existing server
- For the Triton-related paths, it is recommended to add real-environment smoke tests later
