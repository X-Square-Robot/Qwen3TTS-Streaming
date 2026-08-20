**English** | [中文](deployment.zh-CN.md)

# Deployment Guide

This document describes the currently recommended deployment modes, profile parameters, and the container strategy used during development.

## Recommended Path

For v0.1, start with:

```bash
bash scripts/bash/autorun.sh all -m custom-1.7b
```

For the first deployment, running the phases separately makes troubleshooting easier:

```bash
bash scripts/bash/autorun.sh setup -m custom-1.7b
bash scripts/bash/autorun.sh build -m custom-1.7b
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway standalone
```

## Unified Entry Point and Control Parameters

The unified lifecycle entry point is `bash scripts/bash/autorun.sh`, which supports two invocation styles:

- Interactive: `bash scripts/bash/autorun.sh`
- One-shot: `bash scripts/bash/autorun.sh <command> -m <variant> [options]`

All key controls are accessible through autorun: export GPU, TensorRT build GPU, engine profile, runtime limits, deployment mode, and ports.

Phase C is split into two explicit commands:

```bash
# Only assemble deployment artifacts, do not start the service; use this for cross-host / packaging-host scenarios
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway engine-docker

# Only start the service on the current machine; do not run this if the current machine is not the production serving host
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway engine-docker
```

`autorun.sh all` is the "full local pipeline" and runs `setup → build → package → deploy`. If you are preparing artifacts on an export/packaging host for a cloud production container, you should normally run `package` after importing the engine artifact returned from the cloud, then hand the image and model package off to the production deployment system; do not run `deploy` on the packaging host.

Configuration precedence is: command-line arguments > exported environment variables > manifest/default. Common environment variables include `MODEL_VERSION`, `EXPORT_DEVICE`, `BUILD_GPU_DEVICE`, `RUNTIME_GPU_DEVICE`, `MAX_BATCH_SIZE`, `MAX_INPUT_LEN`, `MAX_SEQ_LEN`, `RUNTIME_MAX_BATCH_SIZE`, `RUNTIME_MAX_SEQ_LEN`; but for day-to-day use it is recommended to go through autorun parameters for reproducibility.

When `autorun.sh` starts without a command, its **Release and package metadata** TUI section accepts the model release, engine build identity, packager, and package date directly. The same fields have non-interactive options:

```bash
bash scripts/bash/autorun.sh package -m custom-1.7b \
  --model-release-version 'zehan@20260818' \
  --engine-build-version 'rime@20260820_580_5090_v1' \
  --packager rime \
  --package-date 2026-08-20
```

`--model-version 2` remains the Triton numeric directory; `--model-release-version` is the value stored in the package's `MODEL_VERSION`. Combined variants `all` and `all-1.7b` reject one shared model release value so distinct models cannot be relabeled accidentally.

### Phase B Parameters

```text
Phase B:
  --max-batch-size <N>          TensorRT profile max batch
  --max-input-len <N>           prefill/input max token length
  --max-seq-len <N>             KV cache max sequence length
  --dtype bf16|fp16|fp32|fp8    TensorRT build precision (base default for each submodule)
  --cp-precision <T>            Code Predictor precision (default: follow --dtype, i.e. bf16)
  --code2wav-precision <T>      Code2Wav precision (default: follow --dtype/bf16; fp16 opt-in, low-concurrency only)
  --triton-io-float-dtype <T>   TensorRT/Triton float I/O dtype, defaults to --dtype
  --target-driver <ver>         select NGC image by deployment host NVIDIA driver
  --build-device <dev>          trtexec build GPU
```

### Phase C Parameters

```text
Phase C:
  --gateway standalone|triton|engine-docker
  --engine-mode trt|onnx
  --runtime-max-batch-size <N>  runtime scheduler batch limit
  --runtime-max-seq-len <N>     runtime scheduler seq limit
  --runtime-device <dev>        runtime service GPU
  --realtime-port <N>           Triton OpenAI Realtime host port (default: 50053)
```

### Model Version Number

By default, the Triton model version directory `1` is assembled. If you need to generate a different version directory, specify it with `--model-version <N>`; this places the shared model package under `workspace/model_repository/tts_orchestrator/<N>` and makes standalone, engine Docker, and Triton all read from `/models/tts_orchestrator/<N>`.

```bash
bash scripts/bash/autorun.sh deploy -m custom-1.7b \
  --gateway triton \
  --model-version 2
```

This version number is the Triton model repository version directory, not a Hugging Face / ModelScope weights revision. HTTP clients that explicitly pass a version must request `/v2/models/tts_orchestrator/versions/<N>/infer`; when no version is passed explicitly, Triton selects an available version based on the repository state.

The model's own release identity is separate and travels with the model. A source model can carry a read-only `MODEL_VERSION`, or the value can be entered directly in the autorun TUI and written as a read-only export sidecar. It contains one version line, for example:

```text
zehan@20260601
```

Phase A copies it to `workspace/exported/<variant>/MODEL_VERSION`; an explicit TUI value takes precedence and is written to the same location. Phase C copies it again to the model-package root at `tts_orchestrator/<N>/MODEL_VERSION`; both files use mode `0444`. Export, repository validation, and engine startup reject a missing or empty file. The engine build identity is managed independently and can be entered in the TUI or supplied with `--engine-build-version`; autorun writes it to read-only `ENGINE_BUILD_VERSION`, Phase C carries it into the model package, and runtime reads it directly from that package.

Model-package provenance is stored separately from the model release identity. Phase C creates a read-only `PACKAGE_INFO.json` in the same model-package root:

```json
{
  "package_info_schema_version": 1,
  "packager": "rime",
  "packaged_on": "2026-08-20"
}
```

`packaged_on` has day precision only and must use strict ISO `YYYY-MM-DD` format, without a time or time zone. By default, the packager is the current system user and the package date is the local calendar date. CI and reproducible builds can set them explicitly with `QWEN3_TTS_PACKAGER` and `QWEN3_TTS_PACKAGE_DATE`. Phase C sets mode `0444`; repository validation rejects missing, writable, or malformed package metadata.

## GPU Selection

The default is `--device auto`: the script selects the GPU with the most free memory currently available. You can also explicitly assign the same card to all phases:

```bash
bash scripts/bash/autorun.sh all -m custom-1.7b --device 1
```

You can also assign devices per phase:

```bash
bash scripts/bash/autorun.sh all -m custom-1.7b \
  --export-device auto \
  --build-device 1 \
  --runtime-device 1
```

Parameter meanings:

```text
--device <dev>          applies to export, build, and runtime phases; dev can be auto, 0, 1, cuda:1
--export-device <dev>   Phase A model export only; also supports cpu
--build-device <dev>    Phase B trtexec engine build only; supports auto, all, 0, 1, cuda:1
--runtime-device <dev>  Phase C service runtime only; supports auto, 0, 1, cuda:1
```

Phase B restricts the build GPU at the Docker layer; for example, `--build-device 1` runs `trtexec` in a way similar to `docker run --gpus device=1 ...`. As a result, the `trtexec` log may show `Selected Device ID: 0` inside the container, but the UUID will correspond to physical GPU 1.

## Phase A: Export

Phase A downloads the model, installs dependencies, and exports ONNX/weights/manifest.

```bash
bash scripts/bash/autorun.sh setup -m custom-1.7b
```

Common parameters:

```text
--source auto|hf|modelscope
--skip-models
--skip-deps
--skip-export
--target-driver <driver>
```

## Phase B: Build the TensorRT Engine

Phase B runs trtexec inside the NGC container and writes the actual profile into the manifest:

```bash
bash scripts/bash/autorun.sh build -m custom-1.7b \
  --max-batch-size 64 \
  --max-input-len 128 \
  --max-seq-len 512 \
  --dtype bf16
```

Write location:

```text
workspace/exported/custom-1.7b/triton_manifest.json
```

Key fields:

```json
{
  "engine_profile": {
    "engine_mode": "trt",
    "engine_dtype": "bf16",
    "triton_io_float_dtype": "bf16",
    "max_batch_size": 64,
    "max_input_len": 128,
    "max_seq_len": 512,
    "builder_image": "nvcr.io/nvidia/tritonserver:26.02-py3"
  }
}
```

The runtime batch/seq cannot exceed the profile recorded here. When you need a larger batch or longer text, rerun Phase B.

### Detailed Engine Profile Calculation Logic

If you do not explicitly pass `--max-batch-size`, `--max-input-len`, or `--max-seq-len`, Phase B first reads
`workspace/exported/<variant>/triton_manifest.json` and the exported weights/engine/ONNX files to estimate:

- Fixed footprint: TRT/ONNX main model, runtime embedding weights, and the required reference preprocessing engines
- Per-stream persistent state: Talker KV pool, Code2Wav KV, conv/transconv double buffers, token_counts
- Per-step peak: batched talker KV inputs, C2W KV/state inputs, TRT output buffers, and a small amount of sampling/attention scratch

It then combines this with the total GPU memory from the target machine's `target_profile.json` and rounds down to a supported profile tier:

```text
16 / 32 / 64 / 128
```

Therefore, for cross-host builds, the profile should be estimated from the production host's `target_profile.json` and the export artifacts, not from the packaging host's memory. For example, `custom-1.7b` on a 48G target card can, if the fixed footprint, TRT reserve, and KV/cache estimates still leave enough headroom, default to `max_batch_size=128`.

Only if the export manifest does not exist does it fall back to coarse memory tiers:

```text
~24 GB GPU:  max_batch=16   max_input_len=96   max_seq_len=384
~32 GB GPU:  max_batch=32   max_input_len=128  max_seq_len=512
~48 GB GPU:  max_batch=64   max_input_len=128  max_seq_len=512
~80 GB GPU:  max_batch=128  max_input_len=128  max_seq_len=512
```

These are only default suggestions, not limits; explicit parameters still take the highest precedence. For instance, on a 24G machine you can try building a larger profile for a 48G deployment host:

```bash
bash scripts/bash/autorun.sh build -m custom-1.7b \
  --build-device 1 \
  --max-batch-size 64 \
  --max-input-len 128 \
  --max-seq-len 512
```

However, TensorRT compilation itself also requires GPU memory. If the build machine has insufficient memory, `trtexec` may still OOM; in that case you need to switch to a larger build card, free memory, or lower the profile.

These values are written into the `engine_profile` field of `workspace/exported/<variant>/triton_manifest.json`. When the runtime starts, if the requested batch/seq exceeds the profile, it fails immediately; if the prefill length exceeds `max_input_len`, it also raises a clear error, avoiding a silent clamp or exposing TensorRT shape issues only at runtime.

Examples:

```bash
# Build a smaller profile for validation on low-memory machines
bash scripts/bash/autorun.sh build -m custom-1.7b \
  --max-batch-size 16 --max-input-len 96 --max-seq-len 384

# Runtime usage must not exceed the profile recorded in the manifest
bash scripts/bash/autorun.sh deploy -m custom-1.7b \
  --gateway standalone \
  --runtime-max-batch-size 16 \
  --runtime-max-seq-len 384

# The Triton gateway uses the same runtime limits and GPU entry point
bash scripts/bash/autorun.sh deploy -m custom-1.7b \
  --gateway triton \
  --runtime-device 1 \
  --runtime-max-batch-size 16 \
  --runtime-max-seq-len 384
```

## Phase C: Start the Service

### Standalone

```bash
bash scripts/bash/autorun.sh deploy \
  --gateway standalone \
  -m custom-1.7b \
  --max-batch 32 \
  --max-seq-len 512
```

Endpoints:

- gRPC: `localhost:50051`
- OpenAI Realtime: `ws://localhost:50052/v1/realtime`
- compatibility WebSocket: `ws://localhost:50052/v1/ws`
- capabilities: `http://localhost:50052/v1/capabilities`
- health: `http://localhost:8080/health`

#### Health endpoint and platform probes

The health port binds at process start (before the model loads), so probes always
get an HTTP answer instead of connection-refused. Routes:

| Path | Semantics |
|------|-----------|
| `/health` | `503` + `{"status": "loading", ...}` until the engine is ready (model loaded, warmup done, gateways bound), then `200` + full stats. Point unified liveness/readiness/startup probes here. |
| `/readyz` | Same ready gating as `/health`, unaffected by the probe-mode knob. |
| `/livez` | `200` whenever the process is up (pure liveness). |
| `/metrics` | Always `200`; the loading state is reported in the body, not as a scrape error. |

If the engine loop thread dies after startup, `/health` and `/readyz` drop back to
`503` (`"status": "engine_loop_dead"`) so a platform liveness probe restarts the
container — intended self-healing for a stateless engine.

The same four routes are also served on the WebSocket port (default `50052`),
sharing one readiness state, for platforms that can only probe the service port.
Two differences versus the health port: the WebSocket port binds only after the
model load (probes get connection-refused during the load window, so the startup
grace must cover the cold start), and it answers from the gateway event loop, so
it also verifies the actual serving path is responsive. Prefer the dedicated
health port when your platform lets you choose.

#### Single-port Kubernetes deployment

On platforms that expose only one container port, set `PORT=8000` and
`HEALTH_PORT=0`. This disables the dedicated health listener, not health checks:
`/health`, `/readyz`, `/livez`, Demo, SDK, capabilities, and Realtime WebSocket
are all served by port `8000`. Internal gRPC may continue listening on its default
`50051` without appearing in the Service or Ingress.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3-tts
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qwen3-tts
  template:
    metadata:
      labels:
        app: qwen3-tts
    spec:
      containers:
        - name: engine
          image: qwen3-engine:<tag>
          env:
            - name: PORT
              value: "8000"
            - name: HEALTH_PORT
              value: "0"
            - name: DEMO_ENABLED
              value: "true"
          ports:
            - name: public
              containerPort: 8000
          startupProbe:
            httpGet:
              path: /health
              port: public
            periodSeconds: 10
            failureThreshold: 30
          readinessProbe:
            httpGet:
              path: /readyz
              port: public
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /livez
              port: public
            periodSeconds: 10
          resources:
            limits:
              nvidia.com/gpu: 1
          volumeMounts:
            - name: models
              mountPath: /models
              readOnly: true
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: <model-pvc>
---
apiVersion: v1
kind: Service
metadata:
  name: qwen3-tts
spec:
  selector:
    app: qwen3-tts
  ports:
    - name: public
      port: 8000
      targetPort: public
```

The same Service address now provides:

- `http://<service>:8000/demo/`
- `ws://<service>:8000/v1/realtime`
- `http://<service>:8000/sdk/`
- `http://<service>:8000/health`

#### Recommended: one HTTPS domain for Demo, WebSocket, and SDK

Point the custom domain at an Ingress/Gateway and terminate TLS there with a
trusted CA certificate. The container and Service remain plain HTTP/WS on
`8000`; certificates do not need to enter the model container. This example
uses ingress-nginx and cert-manager; replace the `ClusterIssuer` name with the
one installed in the cluster:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: qwen3-tts
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
spec:
  ingressClassName: nginx
  tls:
    - hosts: [tts.example.com]
      secretName: qwen3-tts-tls
  rules:
    - host: tts.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: qwen3-tts
                port:
                  number: 8000
```

The Ingress must preserve WebSocket upgrades; mainstream Kubernetes Ingress
controllers handle upgrades on the same HTTP route. The long read/write
timeouts keep persistent connections from being closed prematurely. One public
`443` listener now serves:

- Demo: `https://tts.example.com/demo/`
- Python SDK: `TTSClient.connect("https://tts.example.com")`
- Realtime: `wss://tts.example.com/v1/realtime`
- SDK downloads: `https://tts.example.com/sdk/`

A public-CA certificate is trusted by browsers and Python by default, so no
`tls_verify=False`, certificate path, or extra environment variable is needed.
Do not create a second Service or port for Demo. When set together,
`ENGINE_SERVER_WEBSOCKET_PORT` /
`ENGINE_SERVER_HEALTH_PORT` take precedence over `PORT` / `HEALTH_PORT`.

When a development LAN does not need microphone capture or output-device
selection, it does not need to duplicate this TLS setup. Public traffic can
continue through Ingress as `https://` / `wss://`, while the LAN simultaneously
uses `http://<ddns-host>:8000/demo/` and
`ws://<ddns-host>:8000/v1/realtime` against the same Service or container. The
player falls back automatically on HTTP and still plays through the system
default speaker. Python can use
`TTSClient.connect("http://<ddns-host>:8000")`. A service without a certificate
cannot be addressed as `https://`.

Platform probe checklist:

- Size the startup grace to cover the cold start (TRT deserialize + warmup is
  roughly 20–40 s depending on GPU and batch profile; measure once and add margin,
  e.g. `period 10s × failureThreshold 30`). Too little grace means the probe kills
  the container mid-load and it never comes up.
- Probe `127.0.0.1:8080`, not `localhost` — the server binds IPv4 `0.0.0.0` only.
- If your platform's liveness grace cannot be configured to cover the load, set
  `ENGINE_SERVER_HEALTH_PROBE_MODE=alive` (or `server.health_probe_mode: alive` in
  `engine.yaml`): `/health` then returns `200` as soon as the port is up, and you
  lose ready-gating on that path (`/readyz` keeps it). The short alias
  `ENGINE_HEALTH_PROBE_MODE` works only in compose / engine-docker deployments,
  where the entrypoint maps it.
- For Triton's native readiness, probe
  `http://<host>:8000/v2/health/ready`. For the public Realtime serving path,
  probe the sidecar at `http://<host>:50053/health`; it stays `503` until both
  the sidecar and `tts_orchestrator` are ready.

In the standalone TRT path, `base` / `icl` reference preprocessing is executed serially by
`speaker_encoder.engine`, `speech_tokenizer_codec_fused.engine`, and the optional
`code2wav_decoder.engine`, without batching for now. `spliter.max_concurrent_segments`
only affects subsequent text segmentation and EngineLoop slot concurrency; it does not control the speech encoder. The maximum reference
audio duration is governed by `ref_audio_max_duration_sec` in capabilities; the current
TRT build defaults to 8 seconds. `reference_cache` caches ref-audio preprocessing
features, and `prefix_cache` caches Talker ICL prefix KV; the two are configured independently. When
`reference_cache` is enabled, the standalone engine warms up
`references.default` and registry entries before loading the main `model.plan`, and releases the
ref TRT engine after each preprocessing stage, reducing the probability that reloading the speaker/codec engine
on the request path triggers an OOM on 24GB-class memory.

### Engine Docker

```bash
bash scripts/bash/autorun.sh deploy \
  --gateway engine-docker \
  -m custom-1.7b
```

Compose publishes gRPC `50051` and the public HTTP/WebSocket gateway `50052`.
The early-start `8080` health listener is container-internal and feeds the
Docker healthcheck; it is not bound on the host. Public `/health`, `/demo/`,
`/sdk/`, `/v1/realtime`, and `/v1/ws` are all served on `50052`.

This mode uses the engine image as a fixed application layer: the image contains the TensorRT/Python
runtime, the `/app/engine` engine code, a default `/app/engine.yaml`, and startup scripts;
at runtime it read-only mounts the same model package as Triton,
`workspace/model_repository/tts_orchestrator/1`. The regular image already contains the
`/app/engine` code, so after a code update you need to rebuild and publish the image.

The local standalone gateway also uses the same model package, only served by the local Python
process running `engine.server`. In other words, the model artifacts for standalone, engine-docker, and Triton
are all `model_repository/tts_orchestrator/1`; the only difference is the runtime process and container layer.
Phase C assemble syncs the repository's `resources/` into
`model_repository/tts_orchestrator/<version>/resources/`. The Base/ICL
reference registry can directly use model-package-relative paths, for example
`resources/speakers/<alias>/ref.wav` and `resources/speakers/<alias>/ref.txt`
(configured via `ref_text_path`).

For production, it is recommended to split "application image, model artifacts, deployment config" into three layers:

- Application image: `qwen3-engine:<tag>`, containing dependencies and engine code, without mounting a source directory.
- Model artifacts: a unified Triton-compatible `model_repository`. Both engine and Triton
  read `runtime/`, `weights/`, `tokenizer/`, and the manifest from `/models/tts_orchestrator/1`. engine-docker currently requires a `trt`
  package with `runtime/model.plan`.
- Deployment config: ports, batch, seq_len, session, speaker defaults, etc., pointed to via
  `ENGINE_CONFIG` referencing a read-only mounted YAML, or overridden via `ENGINE_*` environment variables.

Example:

```bash
ENGINE_CONFIG=/etc/qwen3-tts/engine.yaml \
ENGINE_CONFIG_FILE=/srv/qwen3/config/engine.yaml \
MODEL_REPO_DIR=/srv/qwen3/model_repository \
bash scripts/bash/autorun.sh deploy --gateway engine-docker -m custom-1.7b
```

If you write compose volumes directly, you can mount:

```yaml
volumes:
  - /srv/qwen3/model_repository:/models:ro
  - /srv/qwen3/config/engine.yaml:/etc/qwen3-tts/engine.yaml:ro
environment:
  ENGINE_CONFIG: /etc/qwen3-tts/engine.yaml
```

During development, it is recommended to use instead:

```bash
bash scripts/bash/compose.sh up --gateway engine --variant custom-1.7b --dev
```

or:

```bash
bash scripts/bash/compose.sh watch --gateway engine --variant custom-1.7b
```

This separates the environment layer from the code layer and avoids rebuilding the dependency image every time you change Python code.

### Triton

```bash
bash scripts/bash/autorun.sh package \
  --gateway triton \
  -m custom-1.7b \
  --engine-mode trt

bash scripts/bash/autorun.sh deploy \
  --gateway triton \
  -m custom-1.7b
```

Triton default ports:

- HTTP: `localhost:8000`
- gRPC: `localhost:8001`
- Metrics: `localhost:8002`
- OpenAI Realtime sidecar: `ws://localhost:50053/v1/realtime`
- Realtime capabilities / health: `http://localhost:50053/v1/capabilities` and
  `http://localhost:50053/health`

The compose wrapper starts Triton and the Realtime sidecar together. Use
`--realtime-port <N>` to change the sidecar's host port. Billing usage is always
returned in `response.done.response.usage`; the default deployment also appends
complete, cancelled, and failed response records to
`workspace/realtime_usage/realtime_usage.jsonl`.

### Logs inside the container

Engine Docker and Triton keep rotating logs inside the container, independently
of the Docker logging driver:

- Engine Docker: `/var/log/qwen3tts/engine.log`
- Triton: `/var/log/qwen3tts/triton.log`
- Triton Realtime sidecar: `/var/log/qwen3tts/realtime-gateway.log`

By default, each log rotates at 50 MiB and keeps 10 backup files. Configure the
policy with these container environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `QWEN_LOG_DIR` | `/var/log/qwen3tts` | Directory for log files. |
| `QWEN_LOG_MAX_BYTES` | `52428800` | Maximum size of one log file in bytes. |
| `QWEN_LOG_BACKUP_COUNT` | `10` | Number of rotated backup files to keep. |
| `QWEN_LOG_STDOUT` | `1` | Also mirror the combined service output to container stdout; set to `0` to disable. |

The stdout mirror is best-effort so a stalled logging driver or attached client
cannot block inference. Use the files above as the complete retained history.

After opening a shell in the container, inspect or package the logs with:

```bash
ls -lh "${QWEN_LOG_DIR:-/var/log/qwen3tts}"
tail -n 1000 "${QWEN_LOG_DIR:-/var/log/qwen3tts}/engine.log"
tail -n 1000 "${QWEN_LOG_DIR:-/var/log/qwen3tts}/triton.log"
tar -C "${QWEN_LOG_DIR:-/var/log/qwen3tts}" \
  -czf /tmp/qwen3tts-logs.tar.gz .
```

Only the log file for the selected gateway is present. Container-local logs are
lost when the container is deleted; mount `QWEN_LOG_DIR` on persistent storage
if logs must survive container replacement.

## Built-in Demo and documentation

The release image already contains the version-matched product Demo, Browser
SDK, Python wheel index, and selected Markdown documentation. It is enabled by
default on the same public endpoint as Realtime; no separate Demo API or Node
process is used. The shared endpoint is HTTP/WS by default; the presence of
mounted local certificate files does not switch protocols automatically.
CI/CD builds the Browser SDK npm tarball once and embeds the
same bytes under `/demo/downloads/`; the SDK page generates an `npm install
"https://...tgz"` command for the current instance, without requiring a source
checkout. GitLab tag pipelines also publish that tarball to the project npm
Registry as a second installation channel. Set `DEMO_ENABLED=false` at startup
to disable it:

```bash
bash scripts/bash/compose.sh up --build \
  --gateway engine --variant custom-1.7b
# Open http://localhost:50052/demo/
```

#### Direct HTTPS/WSS on a development host

Like FunASR Nano, a standalone development host without an Ingress can
terminate TLS directly on the public WebSocket port. Mount the certificate and
private key read-only and configure both together:

```bash
SAN_EXTRA_DNS=demo.example.test ./tools/generate_demo_local_cert.sh
TLS_AUTO_ENABLE=true \
  bash scripts/bash/compose.sh up --build --gateway engine --variant custom-1.7b
```

For a CA-issued certificate, mount it explicitly:

```bash
TLS_HOST_DIR=/host/path/to/certificate \
TLS_CERT_FILE=/app/tls/fullchain.pem \
TLS_KEY_FILE=/app/tls/privkey.pem \
bash scripts/bash/compose.sh up --build --gateway engine --variant custom-1.7b
```

The same port then serves `https://<host>:50052/demo/`,
`wss://<host>:50052/v1/realtime`, `https://<host>:50052/sdk/`, and
`https://<host>:50052/health`. With `TLS_AUTO_ENABLE=true`, the entrypoint
discovers `/app/tls/cert.local.pem` and `/app/tls/key.local.pem` for a
development certificate after it has been explicitly trusted by the test
browser. Auto-discovery is off by default, so merely mounting a certificate directory never changes
`http://` / `ws://` into `https://` / `wss://`. The certificate must cover the
actual hostname. A missing file, partial pair, or certificate/key mismatch
fails before the GPU model is loaded.

For Python SDK testing against that self-signed endpoint, explicitly trust the
generated certificate:

```python
client = TTSClient.connect(
    "wss://localhost:50052/v1/realtime",
    tls_verify="workspace/tls/cert.local.pem",
)
```

`tls_verify=False` is available for a temporary TLS-path check but must never
be used in production. Accepting a certificate in a browser does not change
Python's trust store, and the Browser SDK cannot disable browser certificate
verification from JavaScript. When TLS itself is not under test, use the
default local `http://` / `ws://` endpoint instead.

Production Kubernetes normally leaves `TLS_CERT_FILE` / `TLS_KEY_FILE` unset
and terminates trusted TLS at its Ingress or Gateway before proxying to the
Pod's HTTP port. Both modes still expose only one public service port.

For Triton, replace `--gateway engine` with `--gateway triton` and open
`http://localhost:50053/demo/`. The portal uses relative URLs, so a deployment
below `/infer/<instance>` keeps that prefix for Demo assets, `/sdk/`,
`/v1/capabilities`, and `/v1/realtime`. Set `DEMO_ENABLED=false` when the portal
must not be public; `/demo/` then returns 404.

`demo_api` remains an optional engineering backend for detailed traces; the
repository no longer carries a second WebUI. LLM PK, concurrency, and traces
are entered through the single `/demo/#/lab` portal. Enable Compose profile
`demo` and publish its URL through `DEMO_LAB_URL` only when those tools are needed.

### Public gateway security boundary

The portal does not implement a second login system and never asks the browser
to enter or persist a long-lived API key. A public deployment must protect
`/demo`, `/sdk`, and `/v1/*` behind the same reverse proxy and must:

- preserve the complete instance path prefix and WebSocket upgrade headers;
- enforce an explicit same-origin/allowlist check on WebSocket `Origin`;
- authenticate the user/tenant before upgrade and enforce tenant concurrency,
  request-rate, and usage quotas;
- bound text, reference-audio, and WebSocket message sizes, never exceeding the
  reference limit advertised by `/v1/capabilities`;
- configure handshake, idle, per-response, and connection-lifetime timeouts and
  cap total connections; and
- avoid caching `config.json` or tenant-bearing responses, and never log text,
  reference audio, or credentials.

For a cross-origin `DEMO_LAB_URL`, restrict `QWEN_DEMO_CORS_ORIGIN` to the exact
portal Origin; do not retain the default `*` in production.

## FAQ

### runtime max_seq_len exceeds profile

The error looks like:

```text
runtime max_seq_len=1024 exceeds engine profile max_seq_len=512
```

Solutions:

- Lower `--max-seq-len` / `ENGINE_SCHEDULER_MAX_SEQ_LEN`.
- Or rebuild the engine: `bash scripts/bash/autorun.sh build -m custom-1.7b --max-seq-len 1024`.

### dtype mismatch

If Triton reports errors like `TYPE_FP32` / `TYPE_BF16`, make sure:

- Phase B's `--dtype` and `--triton-io-float-dtype` match the target.
- `triton_manifest.json` has been updated by Phase B.
- Re-assemble model_repository.

### TensorRT plan fails to deserialize

The TensorRT plan is tightly bound to the runtime version. After switching the TensorRT/NGC image, you need to rebuild the engine.

### The built-in portal does not show the Lab entry

The portal shows Lab only when `lab_available=true` and `demo_api /healthz` is reachable. Check:

- Whether the Triton gRPC port is `localhost:8001`.
- Whether the demo API's `QWEN_DEMO_TRITON_GRPC` is correct.
- Whether the runtime publishes a browser-reachable `DEMO_LAB_URL`.
