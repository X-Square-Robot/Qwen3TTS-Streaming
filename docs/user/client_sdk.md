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
   to the GitLab Container Registry or GHCR.
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
| `PYTORCH_CPU_INDEX` | CPU-only PyTorch index used by GitHub unit tests |
| `RUNNER_*_IMAGE`, `DEBIAN_*_MIRROR`, `ALPINE_MIRROR` | GitLab job/service images and OS package mirrors |

The public proxies are useful for getting a constrained runner working. For a
stable release path, pre-sync the NGC and job images into an internal Registry
and override `ENGINE_BASE_IMAGE` and the `RUNNER_*_IMAGE` values. These package
mirrors do not proxy GitHub Actions, the Release API, GHCR pushes, or GitLab
publication traffic, so the runner must still reach its forge. Both release
pipelines publish inline Docker cache metadata under the mutable `buildcache`
image tag; the immutable release tag is still the deployment artifact. The
first build must pull the large NGC base, while later tag builds can reuse its
verified dependency layers through the Registry.

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
