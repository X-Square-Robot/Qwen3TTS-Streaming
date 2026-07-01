**English** | [中文](README.zh-CN.md)

# Qwen3TTS-Streaming

*Let's play text the way we play audio!*

## Introduction

Qwen3TTS-Streaming is an **engineering preview** project: it exports the official Qwen3-TTS PyTorch weights into an ONNX/TensorRT runtime and builds token-level streaming TTS around a Triton/standalone engine, together with model fusion, frontend segmentation, prefix cache, continuous batching, and a WebUI performance showcase. The project opens up a highly optimized, reproducible, and continuously verifiable engineering pipeline, inviting the community to polish it together into a reliable open-source inference system.

> ⚠️ **Status: v0.1 engineering preview, not production-ready.** Streaming mode may still exhibit **hallucination, repetition, and dropped reading** (roughly 10–18% on the current checkpoint, rooted in the model and sampling; see [Known Limitations](docs/user/known_limitations.md)). **The currently recommended stable scope is the `custom-1.7b` / `custom_voice` path.** `design-1.7b`, `base-1.7b` / x-vector voice cloning, and `icl` voice cloning are experimental; the `0.6b` variants are not part of the v0.1 mainline. Do not use it directly for production content generation.

## Performance Claims

The low-latency numbers mentioned in this project are conditional results, not general guarantees:

- `13ms TTFT`: the lowest observed value, dependent on the specified hardware, a warm engine, prefix/cache hits, single-request load, and a local link.
- Standalone `engine-grpc` TTFT is measured by default over a ready/reused gRPC channel and, like WebSocket, does not count the client connection setup cost toward first-packet latency; a cold/lazy channel adds roughly 10ms.
- `180ms 128-stream avg TTFT`: a concurrency stress-test measure that requires specifying the hardware, cache, input, profile, sampling parameters, and client-side measurement method.
- The WebUI only represents replayable real-time synthesized audio when the result source is marked `live_triton` or `live_engine_websocket` and carries an `audio` field.

For detailed benchmark methodology, see [Benchmark Methodology](docs/user/benchmark_methodology.md).

## Capability Status

| Path | Current status | Open-source scope |
| --- | --- | --- |
| `custom-1.7b` / `custom_voice` | Prioritized/stable | The v0.1 recommended path; the WebUI and demo showcase it by default |
| `design-1.7b` / `voice_design` | Experimental | Code and export entry points can be kept, but must be marked as not fully validated |
| `base-1.7b` / x-vector voice clone | Experimental | Standalone already wires up ref audio → speaker embedding; needs the base export artifacts and real end-to-end validation |
| `icl` voice clone | Experimental | Standalone already wires up ref audio + ref text → ref codec/code injection; needs the TRT ref-audio engine and real end-to-end validation |
| `0.6b` variants | Not part of the v0.1 mainline | Export/download entry points can be kept, but need separate validation before release |

## Prerequisites

- **GPU**: an NVIDIA GPU, ≥16GB VRAM recommended (1.7B + KV pool + TensorRT runtime); a matching NVIDIA driver is required.
- **CUDA / TensorRT**: provided via NVIDIA NGC containers (`nvcr.io/nvidia/tensorrt`, `nvcr.io/nvidia/tritonserver`); see `scripts/bash/ngc_matrix.conf` for the version matrix. **Pulling an NGC image constitutes acceptance of the NVIDIA EULA.**
- **Docker**: used to orchestrate the engine/Triton containers (with the NVIDIA Container Toolkit to enable `--gpus`).
- **Disk**: roughly 20–40GB for the model plus export/build artifacts.
- **Model weights**: on first run, download from ModelScope / Hugging Face (see the flow below); this repository does not distribute weights.

> The first end-to-end run includes "download weights → export ONNX → build TensorRT," whose duration depends on your GPU; afterward you can reuse the artifacts or import them across hosts.

## Quick Start

```bash
git clone --recursive https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git
cd Qwen3TTS-Streaming

# Interactive mode
bash scripts/bash/autorun.sh

# Run the full local pipeline in one shot (custom-1.7b + standalone + TensorRT)
bash scripts/bash/autorun.sh all -m custom-1.7b
```

Three phases: **Phase A** `setup_env.sh` (download the model, install the environment, export ONNX/weights/manifest) → **Phase B** `build_engines.sh` (build the TensorRT engine with trtexec) → **Phase C** `package` + `deploy` (assemble the model package/image and start the service).

You can also run the phases separately, which is convenient for troubleshooting or reusing already-exported artifacts:

```bash
bash scripts/bash/autorun.sh setup   -m custom-1.7b          # Phase A
bash scripts/bash/autorun.sh build   -m custom-1.7b          # Phase B
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway standalone --engine-mode trt  # Phase C1
bash scripts/bash/autorun.sh deploy  -m custom-1.7b --gateway standalone --engine-mode trt  # Phase C2
```

## Deployment Options

For detailed parameters (the unified entry-point control parameters, Engine Profile computation logic, GPU selection, and model version number), see the [Deployment Guide](docs/user/deployment.md).

### Standalone

Run `engine.server` in local Python, suitable for debugging the engine, protocol, and WebSocket/gRPC. Before startup it assembles the `workspace/model_repository/tts_orchestrator/<model-version>` model package, then reads `runtime/`, `weights/`, `tokenizer/`, and the manifest via `--model-package-dir`.

```bash
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway standalone --engine-mode trt
```

Default ports: gRPC `50051`, WebSocket `ws://localhost:50052/v1/ws`, HTTP capabilities `http://localhost:50052/v1/capabilities`, health `http://localhost:8080/health`.

### Engine Docker

A standalone engine container that uses the same model package; the image contains the runtime and the `/app/engine` code.

```bash
# Assemble artifacts + rebuild the image
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway engine-docker --build
# Start the service
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway engine-docker --engine-mode trt
```

During development you can use bind mount or watch mode to avoid frequently rebuilding the image:

```bash
bash scripts/bash/compose.sh up --gateway engine --variant custom-1.7b --dev
bash scripts/bash/compose.sh watch --gateway engine --variant custom-1.7b
```

engine-docker currently requires the model package to be `--engine-mode trt`, because `engine.server` consumes `runtime/model.plan`; Triton can still run `trt` or `onnx` with the same package structure.

### Triton

Assemble `workspace/model_repository` and start Triton:

```bash
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway triton --engine-mode trt
```

For advanced debugging you can use compose directly:

```bash
bash scripts/bash/compose.sh prepare --gateway triton --variant custom-1.7b --engine-mode trt
bash scripts/bash/compose.sh up --gateway triton --variant custom-1.7b
```

### Base / ICL Experimental Paths

When deploying the `base-1.7b` / `icl` experimental paths, you need to prepare a default reference audio and a reference registry:

```bash
mkdir -p workspace/default_refs
# Put in a 3-10 second wav at 24k (or resampleable):
# workspace/default_refs/base_ref.wav

ENGINE_DEFAULT_BASE_REF_AUDIO_PATH=workspace/default_refs/base_ref.wav \
ENGINE_DEFAULT_BASE_REF_TEXT="text corresponding to the reference audio" \
bash scripts/bash/autorun.sh all -m base-1.7b --gateway standalone --engine-mode trt
```

You can also configure the reference library and reference cache in `engine.yaml`; for detailed field semantics and ICL preprocessing requirements, see the [Deployment Guide](docs/user/deployment.md).

## Client SDK

A standalone Python SDK package that provides unified access to the engine and Triton endpoints, supporting four transports: engine-websocket / engine-grpc / triton-grpc / triton-http. The default `transport="auto"` auto-detects the endpoint.

```bash
pip install qwen3-tts-client           # Core package
pip install qwen3-tts-client[grpc]     # + gRPC transport
pip install qwen3-tts-client[triton]   # + Triton transport
pip install qwen3-tts-client[all]      # All transports + audio
```

Quick usage:

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice"),
)
```

Streaming session:

```python
from qwen3tts import TTSClient, SessionStartRequest, SynthesisConfig

client = TTSClient.connect("localhost")
session = client.open_stream(
    SessionStartRequest(session_id="demo", config=SynthesisConfig(task_type="custom_voice"))
)
session.send_text("你好，")
session.send_text("这是流式输入。")
session.end()
for message in session.iter_messages():
    print(type(message).__name__, getattr(message, "meta", {}))
```

For detailed documentation, see [Client SDK](docs/user/client_sdk.md) and the [`client/`](client) subproject.

## Testing and Acceptance

Test entry points are unified under `tests/`; for a detailed map, see [tests/README.md](tests/README.md).

```bash
# Unit + integration tests
pytest tests/unit tests/integration -q

# Main entry point for serving acceptance and benchmarks
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py --targets engine-grpc
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py --targets triton-grpc,triton-http
```

Validate the base/icl reference resolver and the ICL prefix cache:

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py \
  --targets engine-grpc \
  --reference-tests \
  --reference-alias vivian \
  --ref-audio-path workspace/default_refs/vivian.wav \
  --ref-text "这是一段与 vivian 参考音频完全一致的文本。"
```

## WebUI Demo

The WebUI contains three panels: **Text Player** (plays text by engine decode step — text tokens in the first half, PAD steps shown after flush, and the slider seeks the actual WAV audio once synthesis completes), **LLM PK** (simulates an upstream LLM emitting tokens one by one, comparing streaming vs. non-streaming on the same timeline), and **Concurrency** (the TTFT distribution and throughput of multi-stream synthesis, requesting live Triton by default and saving the real audio).

**Text Player**

![Text Player demo](docs/images/文本播放器.gif)

**LLM PK**

![Streaming vs. non-streaming comparison demo](docs/images/流式非流式对比.gif)

**Concurrency**

![Multi-stream synthesis demo](docs/images/多路合成.gif)

Full screen recording: [演示视频.mp4](docs/videos/演示视频.mp4)

One-click startup (the WebUI dev server, Demo API, and Triton are all started/reused by the launcher):

```bash
bash scripts/demo/start_webui_demo.sh --variant custom-1.7b
```

You can also start it manually in separate steps:

```bash
python -m demo_api --host 0.0.0.0 --port 7860   # Terminal 1
cd webui && npm install && npm run dev             # Terminal 2
```

Open `http://localhost:5173` in your browser. If the live backend is unavailable, the WebUI shows a warning; the audio button is enabled only when real waveform bytes are captured, and no beep placeholder is used.

Docker Compose demo profile:

```bash
bash scripts/bash/compose.sh up --gateway triton --variant custom-1.7b
docker compose --profile demo -f infra/docker/compose.yaml up --build demo-api webui
```

## Streaming Protocol

The standalone engine supports both gRPC and WebSocket. Example WebSocket control frames:

```json
{"type":"start","session_id":"demo","config":{"task_type":"custom_voice","speaker":"Serena"}}
{"type":"text","text":"你好，世界。"}
{"type":"end"}
```

The server returns JSON event frames (protocol events, text tokens, boundaries, completion) and binary frames (PCM audio chunks, whose format is declared by the start/event metadata).

## Project Structure

```text
Qwen3TTS-Streaming/
├── engine/                     # Inference engine: frontend/backend/gateway/core
├── client/                     # Standalone Python SDK package (pip install qwen3-tts-client)
│   ├── src/qwen3tts/           #   Client implementation and transport adapters
│   └── src/qwen3tts_protocol/  #   Shared protocol layer (single source of truth)
├── demo_api/                   # WebUI Demo API (depends on the client package)
├── webui/                      # Vite/React WebUI
├── proto/                      # Single source of the protocol definition (tts.proto + generated code)
├── model_repository/           # Triton Python BLS model definitions
├── infra/
│   └── docker/                 # Dockerfile + compose configuration
├── scripts/
│   ├── bash/                   # autorun/setup/build/deploy lifecycle
│   ├── compose/                # Container entry-point scripts
│   ├── demo/                   # Demo startup scripts (start_webui_demo.sh)
│   ├── export/                 # PyTorch → ONNX/manifest export
│   └── python/                 # Config/manifest/audit tools
├── tests/
│   ├── unit/                   # pytest unit tests
│   ├── integration/            # pytest integration tests
│   ├── e2e/                    # pytest end-to-end tests
│   └── support/                # Shared test code
├── tools/
│   ├── validation/             # Manual validation and benchmarks
│   ├── repro/                  # Frozen bug reproduction cases
│   └── data/                   # Tool data
├── docs/
│   ├── user/                   # User documentation (deployment, SDK, benchmark, limitations)
│   ├── dev/                    # Developer documentation (architecture, design, investigation, operations)
│   └── process/                # Process/historical documentation (archived)
├── resources/                  # Static resources (synthetic reference audio, etc.)
├── third_party/                # git submodule (Qwen3-TTS upstream, Apache-2.0)
└── workspace/                  # Runtime artifacts (gitignored)
```

## Documentation Navigation

- 📖 [User Documentation](docs/user/README.md) — deployment, SDK, benchmark, known limitations
- 📖 [Developer Documentation](docs/dev/README.md) — architecture, design, investigation, operations

## Contributing

This project is a **v0.1 engineering preview**, and streaming quality is still being polished; you are welcome to participate via issues, discussions, and PRs.

- 🤝 [Contributing Guide](CONTRIBUTING.md) — development environment, testing, proto workflow, code style
- 💬 [Support Channels](SUPPORT.md) — how questions / bug reports / suggestions are routed
- 🔒 [Security Policy](SECURITY.md) — the private vulnerability reporting process (please do not file public issues)
- 📜 [Code of Conduct](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1
- 📝 [Changelog](CHANGELOG.md) — record of version changes

## License

- **This project's own code** (`engine/`, `client/`, `demo_api/`, `webui/`, `scripts/`, etc.) is released under the [MIT](LICENSE) license, copyright XSquareRobot.
- **Upstream [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** (the `third_party/` submodule) is Apache 2.0, which is compatible with MIT.
- **Model weights** are released by Qwen/Alibaba; their license is governed by the respective [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) / [Hugging Face](https://huggingface.co/Qwen) model cards; this repository does not distribute any weights.
- **TensorRT / Triton Inference Server** (NVIDIA NGC images) are NVIDIA proprietary software, not bundled in this repository; using them constitutes acceptance of the NVIDIA EULA.
- **[TEN VAD](https://github.com/TEN-framework/ten-vad)** is an **optional** dependency, used only by the experimental `tenvad` VAD mode (disabled by default; installed by the user, not bundled). It is licensed under **Apache 2.0 with additional conditions** (non-compete, single-applicant use) — **not** a standard permissive license; review its terms before enabling that mode.
- The reference audio under `resources/speakers/` is **synthetic audio** with **fictional** speaker names, corresponding to no real individuals.

For full third-party attribution, see [NOTICE](NOTICE).
