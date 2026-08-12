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

## WebUI

Run locally:

```bash
python -m demo_api --host 0.0.0.0 --port 7860

cd webui
npm install
npm run dev
```

Run with Compose:

```bash
bash scripts/bash/autorun.sh deploy --gateway triton -m custom-1.7b
docker compose --profile demo up --build demo-api webui
```

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

### WebUI shows fixture fallback

This indicates that live Triton or live engine is currently unreachable. Check:

- Whether the Triton gRPC port is `localhost:8001`.
- Whether the demo API's `QWEN_DEMO_TRITON_GRPC` is correct.
- Whether the standalone engine WebSocket is `localhost:50052`.
