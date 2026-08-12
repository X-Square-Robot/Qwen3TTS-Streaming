**English** | [中文](client_sdk.zh-CN.md)

# Python Client SDK

## Goal

This SDK targets external callers, providing a unified, lightweight Python client that can be installed directly via `pip install`.

Supported service entry points:

- `openai-realtime` (primary)
- `engine-websocket`
- `engine-grpc`
- `triton-grpc`
- `triton-http`

The default behavior is `transport="auto"`: the client probes Realtime first,
then binds a legacy adaptor only when the primary protocol is unavailable.

## Layout and Publishing

The SDK lives as an independent subproject under the repository's [`client/`](../../client) directory:

- Packaging config: [`client/pyproject.toml`](../../client/pyproject.toml)
- Source entry point: [`client/src/qwen3tts`](../../client/src/qwen3tts)
- Shared protocol layer: [`client/src/qwen3tts_protocol`](../../client/src/qwen3tts_protocol)
- SDK unit tests: [`client/tests`](../../client/tests)

The purpose of this is to avoid packing heavy server-side dependencies and deployment logic into the client wheel.

## Installation

### Version compatibility

Engine and SDK are still released from the same git tag: the engine image bakes
the tag in at build time (`git describe`), and the wheel version is derived
from the same tag (hatch-vcs). The engine advertises its release on the
versioned **capabilities** surface as `engine_version` (`/health` is a pure
liveness probe and carries no version). To find out what a running engine is:

```bash
curl http://<engine-host>:<ws-port>/v1/capabilities
# → {"loaded_model_type": "...", "engine_version": "v0.1.0", "protocol_version": "...", ...}
```

Installing that SDK version is recommended for an exactly reproducible release.
`connect()` reads the server's capabilities and raises
`ProtocolVersionMismatchError` only when the wire-protocol family or major is
incompatible. Protocol revisions within one major are compatible. A differing
`engine_version` only emits `RuntimeWarning` and does not reject the connection;
optional behavior is selected from capabilities. Set
`QWEN3TTS_SKIP_PROTOCOL_CHECK=1` to downgrade protocol incompatibility to a
warning for deliberate experiments, or pass `verify=False` to `connect()` to
skip the connect-time capabilities check entirely.

### Channel 1 — GitHub/GitLab Release and GitLab Package Registry

Each version tag creates a Release wheel on both forges. Install from the
matching Release; `pip` downloads the wheel instead of cloning this monorepo:

```bash
# GitHub
pip install "qwen3-tts-client[all] @ https://github.com/X-Square-Robot/Qwen3TTS-Streaming/releases/download/v0.1.0/qwen3_tts_client-0.1.0-py3-none-any.whl"

# GitLab (the Release's client-sdk link)
pip install "qwen3-tts-client[all] @ https://<gitlab-project>/-/releases/v0.1.0/downloads/client-sdk/qwen3_tts_client-0.1.0-py3-none-any.whl"
```

The identical file is exposed through the project's GitLab PyPI registry:

```bash
pip install \
  --index-url "https://<gitlab-host>/api/v4/projects/<project-id>/packages/pypi/simple" \
  "qwen3-tts-client[all]==0.1.0"
```

For a private project, prefer the PyPI registry and configure a personal or
read-only deploy token in `.netrc`; do not put credentials in a committed
requirements file:

```text
machine <gitlab-host>
login <deploy-token-username>
password <deploy-token>
```

A private Release direct link instead requires a personal access token via
GitLab's documented query parameter or HTTP header, so it is usually simpler
to download the file first and install it locally. If your GitLab administrator
has disabled PyPI package forwarding, configure a trusted dependency index or
pre-install the wheel's third-party dependencies as well.

Optional extras are `[grpc]`, `[triton]`, `[audio]`, and `[all]`. The SDK is one
universal wheel; `pip` still resolves its declared third-party dependencies
from the configured package index.

### Channel 2 — the same wheel from a running engine

Every release engine image embeds the wheel already published by its tag
pipeline and serves it at `GET /sdk/` on the health port:

```bash
curl http://<engine-host>:<health-port>/sdk/      # list available wheels
pip install http://<engine-host>:<health-port>/sdk/qwen3_tts_client-0.1.0-py3-none-any.whl
```

### Release invariant

Pushing (or mirroring) a `vX.Y.Z`, `vX.Y.ZaN`, `vX.Y.ZbN`, or `vX.Y.ZrcN` tag
to a forge starts that forge's `.gitlab-ci.yml` or
`.github/workflows/release.yml`.
Each pipeline independently enforces the same build-once contract:

1. CI checks out only the main repository (`GIT_SUBMODULE_STRATEGY=none` on
   GitLab and `submodules: false` on GitHub).
2. `release_client_wheel.sh` builds exactly one wheel and smoke-tests it.
3. GitLab publishes that wheel to its PyPI Package Registry; GitHub uploads its
   wheel to a draft GitHub Release. This durable object becomes the canonical
   input for the rest of that pipeline.
4. Each image job downloads its forge's canonical wheel, verifies SHA256, and
   embeds those exact bytes under `/app/sdk/` before pushing the engine image
   to `cr.x2robot.cn/audio/qwen3tt-streaming` or GHCR.
5. After the image succeeds, GitLab creates or updates an idempotent Release
   link to its Registry object and GitHub publishes the verified draft Release.
   Neither Release points at an expiring job artifact.

The engine image job needs a Docker-capable runner with ample free disk (at
least 50 GB is recommended for the NVIDIA PyTorch runtime image and build
cache). The base already provides a matched CUDA, PyTorch, and TensorRT stack;
the project installs only its application-level Python dependencies on top.
GitLab's Docker-in-Docker runner must be privileged. GitHub defaults this job
to `self-hosted`; set the repository variable `RELEASE_IMAGE_RUNNER` to the
label of a suitably sized runner when needed; a self-hosted runner must provide
Docker and GitHub CLI (`gh`). Set the protected `NGC_API_KEY` secret as well if
the selected NGC base image requires authenticated access. The GitLab image job
declares a three-hour timeout; any maximum timeout configured on its Runner must
be at least as large.
GitLab also requires masked `X2ROBOT_REGISTRY_USER` and
`X2ROBOT_REGISTRY_PASSWORD` CI/CD variables. If those variables are protected,
the `v*` tags that trigger releases must be protected too. GitLab release images
use the established
`cr.x2robot.cn/audio/qwen3tt-streaming:trt25.10_580_cu13_<git-tag>` naming and
the corresponding NVIDIA PyTorch 25.10 runtime (CUDA 13.0, TensorRT 10.13,
Driver 580 channel).
Protect the `v*` tag namespace and release environment so only release
maintainers can trigger jobs with publishing credentials. GHCR packages are
private by default; make the package public explicitly if anonymous image pulls
are part of the release contract.

Both pipelines default runner downloads to configurable mainland-China
endpoints: BFSU for general Python packages, the NJU mirror for CPU-only
PyTorch used by GitHub unit tests, BFSU for GitLab's Debian/Alpine packages,
and DaoCloud for GitLab job images and the `nvcr.io` NVIDIA PyTorch base image.
Override these names as repository variables on GitHub or CI/CD variables on
GitLab:

| Variable | Purpose |
| --- | --- |
| `PIP_INDEX_URL` | Wheel builds, smoke tests, and general Python image dependencies |
| `ENGINE_BASE_IMAGE` | NVIDIA PyTorch base containing matched CUDA/PyTorch/TensorRT; preferably a digest-pinned company Harbor/ACR copy |
| `X2ROBOT_REGISTRY`, `X2ROBOT_IMAGE`, `X2ROBOT_IMAGE_TAG_PREFIX` | GitLab engine image destination and compatibility tag channel |
| `X2ROBOT_REGISTRY_USER`, `X2ROBOT_REGISTRY_PASSWORD` | Masked GitLab CI/CD credentials for pushing the engine image |
| `PYTORCH_CPU_INDEX` | CPU-only PyTorch index used by GitHub unit tests |
| `RUNNER_*_IMAGE`, `DEBIAN_*_MIRROR`, `ALPINE_MIRROR` | GitLab job/service images and OS package mirrors |

The public proxies are useful for getting a constrained runner working. For a
stable release path, pre-sync the NGC and job images into an internal Registry
and override `ENGINE_BASE_IMAGE` and the `RUNNER_*_IMAGE` values. These package
mirrors do not proxy GitHub Actions, the Release API, GHCR pushes, or GitLab
publication traffic, so the runner must still reach its forge. Both release
pipelines publish inline Docker cache metadata under a mutable `buildcache`
image tag (runtime-prefixed on GitLab); the immutable release tag is still the
deployment artifact. The first build must pull the large NGC base, while later
tag builds can reuse its verified dependency layers through the Registry.

`client/dist/` remains an ignored local/CI staging directory. Wheel binaries
are deliberately not committed to Git, and neither tag pipeline invokes the
local `compose.sh` path that would rebuild them.

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
print(result.details["usage"])
```

On the standalone deployment, `localhost` resolves to
`ws://localhost:50052/v1/realtime`. The Triton compose deployment exposes the
same public protocol through its sidecar at
`ws://localhost:50053/v1/realtime`. `client.resolved_transport` reports the
selected adaptor.

## OpenAI Realtime and Legacy Migration

Pin the primary transport when endpoint selection must be deterministic:

```python
client = TTSClient.connect(
    "wss://tts.example/v1/realtime",
    transport="openai-realtime",
    key="your-key",
)
```

One-shot synthesis uses standard `conversation.item.create` and
`response.create` events. Incremental `open_stream()` uses the
`qwen.input_text_buffer.v1` append/commit extension, which must be advertised by
the server. The transport remains full duplex: text append and
`response.cancel` can be sent while audio deltas are arriving.

Terminal `response.done.response.usage` is available at
`result.details["usage"]` for one-shot calls and `session.usage` for streaming.
`session.response_id` and `session.response_status` expose the billing
correlation and terminal state. A configured server ledger remains
authoritative if a client disconnects before receiving the terminal event.

The four older transports remain available during migration and emit one
`FutureWarning` per process and transport. The temporary environment switch
`QWEN3TTS_SUPPRESS_LEGACY_TRANSPORT_WARNING=1` suppresses that warning. Active
stream resume is not yet implemented for Realtime: an interrupted stream fails
explicitly. The resumable connection-pool behavior documented below applies
only to the compatibility `engine-websocket` transport.

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
    connect_timeout=5.0,
    max_connections=32,
    max_idle_connections=8,
    max_pending_acquires=256,
    acquire_timeout=30.0,
)

# Override the capabilities wait for this call only.
caps = client.get_capabilities(timeout=5.0)

# Fill the pool to four reusable idle sockets before serving traffic.
idle_connections = client.prewarm(connections=4, timeout=5.0)
```

One physical `engine-websocket` connection carries multiple logical sessions
serially; concurrent sessions lease separate pooled connections. Defaults are:

`SessionStartRequest.session_id` is a client correlation ID, not an engine
registry key. Every WebSocket/gRPC start receives a fresh private execution ID,
so concurrent requests that reuse the same public ID remain isolated. A
long-lived business call can be correlated without pinning a connection by
putting its `call_id` in `TimingContext.extra`.

- `reconnect_attempts=1`: retry a new physical connection or failed initial
  `start` write once;
- `active_stream_resume=True`: request safe in-process recovery for active
  WebSocket streams when the gateway supports it;
- `stream_resume_attempts=2` / `stream_resume_timeout=10.0`: use a separate,
  bounded retry count and total deadline for an interrupted active stream;
- `stream_resume_ack_interval=8`: cumulatively acknowledge every eight output
  deliveries (terminal output is acknowledged immediately);
- `max_connections=32`: hard limit across connecting, leased, keepalive-probed,
  and idle sockets;
- `max_idle_connections=8`: retain at most eight idle connections;
- `max_pending_acquires=256`: bound the FIFO lease-wait queue; a full queue
  fails immediately with `PoolSaturatedError`;
- `acquire_timeout=30.0`: fail a queued lease with
  `PoolAcquireTimeoutError` after 30 seconds;
- `idle_ttl=None` / `max_lifetime=None`: disable age-based retirement; `0` is
  equivalent, and an over-age active connection is retired only after return;
- `keepalive_interval=15.0`: probe idle connections every 15 seconds; use `0`
  to disable it;
- `keepalive_jitter=0.2`: randomize each maintenance interval by ±20%.

`TTSClient.prewarm(connections, timeout=...)` treats `connections` as the
desired total idle capacity, not the number to add. It opens only the missing
sockets in parallel, caps the target at `max_idle_connections` and
`max_connections`, and returns the actual idle count on success. Its optional timeout bounds each capabilities
round-trip; connection handshakes still use `connect_timeout`.
`TTSClient.get_capabilities(timeout=...)` applies the same kind of per-call
override when fetching capabilities without changing the client's configured
stream timeout.

The logical boundary is the terminal `done`/`error` event, not WebSocket
closure. A persistent gateway marks a safely reusable successful/cancelled
`done` with `websocket_connection_reusable=true`. Engine errors are closed and
reconnected; if an older gateway omits the marker, the SDK likewise discards
the socket and safely falls back to reconnecting.

If background keepalive (or the synchronous probe used when keepalive is
disabled) finds that a gateway reaped an idle socket, the SDK discards it and
the next session opens a replacement automatically. Those idle-socket probes
use `connect_timeout`, not the potentially much longer stream receive timeout.

For a resume-capable gateway, a mid-stream transport failure keeps the same
engine execution alive during a bounded grace period. The SDK reconnects from
its pool without calling `open_stream()` again, replays only text above the
server's cumulative ACK, and resumes exact output records after its last
delivery/sample cursor. A JSON `audio_header` and the immediately following raw
PCM binary frame form one replayable delivery; the SDK advances the cursor only
after the complete binary frame has entered its local message queue. It never
falls back to re-synthesizing from the beginning. If the token/window expires,
the server process restarted, a different replica receives the reconnect, or
the retry budget is exhausted, the session emits one explicit `error`. Legacy
gateways that do not negotiate the feature retain fail-fast behavior. Call
`client.close()` when finished, preferably through the context manager.

## Unified Streaming Interface

```python
from qwen3tts import SessionStartRequest, SynthesisConfig, TTSClient

client = TTSClient.connect("ws://localhost:50052/v1/realtime")
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

print(session.response_id, session.response_status, session.usage)
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

- `ws://` / `wss://`: `/v1/realtime` is probed as Realtime; a root URL tries
  `/v1/realtime` before `/v1/ws`; an explicit `/v1/ws` remains legacy.
- `http://` / `https://`: probe `GET /v1/capabilities`; prefer its advertised
  `openai-realtime-v1`, otherwise use the legacy standalone or Triton HTTP checks.
- Bare `host:port`: ports `50052` and `50053` probe Realtime first; compatible
  legacy and Triton probes remain fallbacks.
- Bare `host`: expand candidates in priority order `50052` (standalone
  Realtime), `50053` (Triton Realtime sidecar), `50051`, `8001`, `8000`.

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
- OpenAI Realtime is the preferred auto-detected adaptor
- Four legacy adaptor entry points remain available with deprecation warnings
- Explicit streaming-degradation semantics are provided for `triton-http`

It is still recommended to treat it as v1 alpha:

- SDK-to-gateway Realtime integration is covered without requiring a model/GPU
- Real Triton sidecar and GPU smoke tests should still be run in deployment CI
- Durable usage, auth, quotas, and Realtime reconnect acceptance remain gates
  before announcing the legacy removal release
