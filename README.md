**English** | [中文](README.zh-CN.md)

<div align="center">

# Qwen3TTS-Streaming

*Let's play text the way we play audio!*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/actions/workflows/ci.yml/badge.svg)](https://github.com/X-Square-Robot/Qwen3TTS-Streaming/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-v0.2%20stability%20release-blue.svg)](#capability-status)
[![GitHub stars](https://img.shields.io/github/stars/X-Square-Robot/Qwen3TTS-Streaming?style=social)](https://github.com/X-Square-Robot/Qwen3TTS-Streaming)

<img src="docs/images/文本播放器.gif" width="720" alt="Historical pre-consolidation Text Player screenshot; open the current unified portal at /demo/">

<small><em>Historical pre-consolidation capture; the current browser experience is the unified `/demo/` portal.</em></small>

*Text tokens go in, audio chunks come out — in real time. See why that matters in [Token-Level Streaming](#token-level-streaming), then try it in the [built-in Demo](#built-in-demo-and-documentation).*

</div>

## Introduction

Qwen3TTS-Streaming is a **v0.2 stability-focused open-source inference system**: it exports the official Qwen3-TTS PyTorch weights into an ONNX/TensorRT runtime and builds **token-level streaming TTS** around a Triton/standalone engine, together with model fusion, frontend segmentation, prefix cache, continuous batching, and a built-in product Demo. The project provides a highly optimized, reproducible, and continuously verifiable engineering pipeline for real-time speech generation.

> **Status: v0.2 stability release.** The validated `custom-1.7b` / `custom_voice` checkpoint and runtime safeguards substantially suppress streaming hallucination, repetition, and dropped reading. The recorded deterministic validation is **0/100 runaway cases** on the validated full-bf16 engine, compared with the historical 0601 checkpoint's roughly 10–18% runaway rate. Residual behavior remains checkpoint-, input-, and sampling-dependent, so production deployments still need workload-specific validation. `design-1.7b`, `base-1.7b` / x-vector voice cloning, and `icl` voice cloning remain experimental; the `0.6b` variants are outside the current v0.2 stable scope. See [Known Limitations](docs/user/known_limitations.md).

## News

- **2026-09-10: v0.2 substantially improves streaming stability.** The validated retrained
  checkpoint and runtime defenses suppress the historical runaway-hallucination failure mode;
  the recorded full-bf16 validation is 0/100 on the deterministic probe set. This is a
  strong reduction, not a universal semantic-correctness guarantee for arbitrary checkpoints.
- **2026-09-10: Streaming TN and native text progress are now wired into the current engine.**
  The primary streaming TN layer owns raw Unicode, mutable tails, and monotonic
  `TextCommit` records. WeText and mixed-language routing decide the spoken form on CPU, while
  already committed spoken prefixes remain append-only across transport packetization.
- **2026-09-10: `custom-1.7b` supports the cursor-enabled TRT progress route.** The native cursor
  observes the fused graph's sampled codec codebook-0 and projects its continuous label position
  back to normalized/raw text coordinates through stable owner spans. The public extension is
  `qwen.text_progress.v1`, with `native`, `ema`, and `disabled` progress routes.
- **The capability boundary remains explicit.** Native cursor is enabled only for a matched and
  validated `custom-1.7b` cursor artifact. Other models use standard TRT plus EMA/disabled
  progress; the runtime never silently attaches an incompatible 2M cursor head. `SOFT_DRAIN`
  training and low-seam state rollover are not released as default runtime capabilities.

## Features

### Token-Level Streaming

Most TTS pipelines wait for a full sentence — or the whole LLM response — before synthesis even starts. Qwen3TTS-Streaming synthesizes as text tokens arrive, so audio starts while the sentence is still being written:

```text
Traditional (sentence-level) TTS
  LLM  "Hello, how are you today?"  ──(wait for the full sentence)──▶  TTS  ──▶  🔊
                                                                             one long wait, then playback

Qwen3TTS-Streaming (token-level)
  LLM   "Hello" ─ "," ─ " how" ─ " are" ─ " you" ─ " today?" ──▶
           │        │       │       │        │         │
           ▼        ▼       ▼       ▼        ▼         ▼
         chunk    chunk   chunk   chunk    chunk     chunk   ──▶  🔊
                                                                    current c1 chunk ~23ms
```

The historical low-latency baseline was **14.9 ± 0.3ms** server TTFT (n=50, min 14.4ms), with
242–275ms client TTFT under a 128-stream burst. The current 2026-09-10 cursor/TN refresh uses
an SDK-client basis and exposes a c128 latency regression signal versus that historical baseline; see
[Performance Claims](#performance-claims) and the
[current benchmark refresh](docs/dev/investigation/serving_performance_benchmark.md#current-cursortn-refresh-2026-09-10).

### Streaming TN and Native Text Progress

Streaming input cannot safely send every transport delta directly to the tokenizer: `99%`, dates,
URLs, model identifiers, and mixed-language spans can change their spoken form when later
characters arrive. The engine now places a primary streaming TN layer before tokenization and
splitting:

```text
raw Unicode delta
  -> mutable tail / open-span detection
  -> WeText + domain resolution
  -> monotonic TextCommit
  -> tokenizer / Spliter / fused TRT
```

On a `custom-1.7b` cursor-enabled artifact, the fused TRT graph consumes the cursor label plan and
the codec0 sampled inside the graph; CPU owns TN, owner provenance, tail rewrite/reanchor, and
protocol projection. Public `raw_codepoint_end` and `normalized_codepoint_end` values are
conservative integer high-water marks. `display_*_position` values are display-only interpolation
for highlighting and must not be used for billing, resume, or audio-release decisions.

Clients should read `/v1/capabilities` first. Select native progress only when
`native_cursor.progress_available=true`; otherwise use EMA or disable text progress. Progress is
published through `qwen.text_progress.v1` / `qwen.text_progress`. See
[Realtime endpoints and events](docs/user/realtime_api.md) for field and playback-ack semantics.

### A Scheduler Built for Autoregressive Streaming

Triton's built-in `dynamic_batching` assumes stateless requests with a fixed sequence length — it has no concept of "this request is mid-decode, waiting on more text tokens, and holding live KV state." Token-level TTS needs exactly that: every session's KV grows unevenly, and decode must pause (`WAIT_TEXT`) without losing state whenever the upstream LLM stalls.

So the engine drops Triton as the scheduler and runs its own **iteration-level continuous batching** underneath: padded KV alignment with masking, MLFQ-style priority (new sessions protect first-audio latency, long-running ones get demoted instead of starved), and a three-stage decode loop — `prefill → stream-while-waiting → flush` — that suspends and resumes per session. Triton is still a supported front door (the `tts_orchestrator` BLS model is a thin protocol adapter over the same engine); this scheduler is what runs underneath either way.

See the [Engine Design Panorama](docs/dev/architecture/engine_overview.md) for the full constraint-to-design trace, including which trade-offs are hard model constraints and which are still open to improvement.

### Compile Once, Deploy Anywhere

TensorRT engines are pinned to a specific GPU/driver/TensorRT combination — a `.plan` built on one machine won't reliably run on another. Building directly on every target means shipping the full NGC toolchain (and a GPU) to each one, which production/edge/air-gapped hosts often don't have.

Qwen3TTS-Streaming separates *where you build* from *where you deploy*:

```bash
bash scripts/bash/autorun.sh probe-target --out target_profile.json                              # 1. fingerprint the target machine
bash scripts/bash/autorun.sh make-bundle  -m custom-1.7b --target-profile target_profile.json     # 2. build a matching engine bundle
bash scripts/bash/autorun.sh import-artifact workspace/engine_artifact_bundle.tar.zst              # 3. import it on the target — no trtexec needed there

# or fingerprint + build over SSH in one shot:
bash scripts/bash/autorun.sh remote-build -m custom-1.7b --target-profile target_profile.json --remote-host user@host
```

See the [Deployment Guide](docs/user/deployment.md) for the full cross-machine build workflow.

### One Engine, Not Four

A single decode step in this pipeline touches four distinct stages: the talker backbone (prefill/decode), the Code Predictor, codec-embedding summation, and the code2wav vocoder. Exporting each as its own ONNX/TensorRT engine would mean four Python dispatches and four host↔device round-trips — every step, for the life of the stream.

```text
Without fusion — 4 engines per decode step
  talker  ──▶  code predictor  ──▶  codec_sum  ──▶  code2wav  ──▶  🔊
    4 Python dispatches, 4 host↔device round-trips, every single step

Qwen3TTS-Streaming — 1 fused engine per decode step
  talker + code predictor + codec_sum + code2wav  ──▶  🔊
    1 ONNX graph, 1 TensorRT engine, 1 dispatch
```

This isn't TensorRT's automatic kernel fusion — TensorRT doesn't merge across model boundaries on its own. The project's own export code does the model surgery: `TalkerCode2WavFusedONNX` in [`export_09_talker_code2wav_fused.py`](scripts/export/export_09_talker_code2wav_fused.py) chains all four stages into one forward pass and exports them as a single graph, compiled into one `talker_code2wav_fused.engine`. A second fusion, [`export_04_speech_tokenizer_codec_fused.py`](scripts/export/export_04_speech_tokenizer_codec_fused.py), merges the speech tokenizer with codec-embedding summation for reference-audio paths.

The whole path is in this repo, not the `third_party/` submodule — export code (`scripts/export/`), TensorRT profile/IO-format helpers (`scripts/python/trt_fused_*.py`), and tests (`tests/integration/test_trt_fused_io_formats.py`, `tests/unit/engine_core/test_executor_trt_engine.py`).

## Highlights

- ⚡ **Token-level streaming, not sentence-level** — token-level first-chunk delivery; current performance evidence is tracked separately from the historical baseline
- 🧩 **A scheduler built for autoregressive decode**, not Triton's stateless `dynamic_batching` — continuous batching + `WAIT_TEXT` pause/resume
- 🌐 **Compile once, deploy anywhere** — fingerprint a target, build a matching bundle, import it with no GPU toolchain on-site
- 🧵 **One TensorRT engine per decode step, not four** — talker + Code Predictor + codec-embedding sum + code2wav fused into a single exported graph, export-to-test code all in this repo
- 🧠 **Prefix KV cache** — a 16-entry LRU skips prefill on repeat system prompts, saving 10–50ms (see [Engine Design Panorama](docs/dev/architecture/engine_overview.md))
- 🧮 **Code predictor unrolled into one static TRT graph** — no per-step KV, higher GPU utilization than step-by-step decode (see [Engine Design Panorama](docs/dev/architecture/engine_overview.md))
- 🖥️ **Built-in product Demo** — no-code playback, parameter tuning, SDK download, unified docs, and an optional Lab

The first four points above are unpacked in [Features](#features); the last two are covered in the [Engine Design Panorama](docs/dev/architecture/engine_overview.md).

## Table of Contents

- [Features](#features)
  - [Token-Level Streaming](#token-level-streaming)
  - [A Scheduler Built for Autoregressive Streaming](#a-scheduler-built-for-autoregressive-streaming)
  - [Compile Once, Deploy Anywhere](#compile-once-deploy-anywhere)
  - [One Engine, Not Four](#one-engine-not-four)
- [Performance Claims](#performance-claims)
- [News](#news)
- [Streaming TN and Native Text Progress](#streaming-tn-and-native-text-progress)
- [Capability Status](#capability-status)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
- [Client SDK](#client-sdk)
- [Testing and Acceptance](#testing-and-acceptance)
- [Built-in Demo and documentation](#built-in-demo-and-documentation)
- [Streaming Protocol](#streaming-protocol)
- [Project Structure](#project-structure)
- [Documentation Navigation](#documentation-navigation)
- [Contributing](#contributing)
- [License](#license)

## Performance Claims

The low-latency numbers mentioned in this project are conditional results, not general guarantees:

| Scenario | TTFT | Conditions |
| --- | --- | --- |
| **Current 2026-09-10 engine-only refresh, single request** | **23.290ms WebSocket / 24.593ms gRPC** SDK-client TTFT avg | RTX 5090, `qwen3-engine:25.10`, cursor-enabled `custom-1.7b`, bf16, fixed text/speaker, cache hit |
| **Current 2026-09-10 engine-only refresh, 128-stream burst** | **1,273.570ms gRPC / 1,114.293ms WebSocket** SDK-client TTFT avg | SDK WebSocket pool explicitly set to 128; both transports reached max batch 128; 0/11,674 failures overall; Triton not running |
| Historical 2026-07-08 single request | **14.9 ± 0.3ms** server-side (min 14.4, p99 15.9, n=50); ~16.1ms p50 client-side over a reused local gRPC channel | RTX 5090, warm engine, prefix-cache hit, local link, all-bf16 `custom-1.7b`, batch=128 profile |
| Historical 2026-07-08 128 concurrent streams | **242–275ms** by transport (engine-websocket 242 / engine-grpc 275; p99 341–494ms); Triton was not re-benchmarked after those engine optimizations | same stack, single service per run, all 128 admitted and decoded in one batch; historical baseline only |

> ⚠️ **The current 2026-09-10 refresh is not a production throughput claim.** Its c128 latency is
> substantially slower than the historical baseline even after correcting the SDK WebSocket pool
> configuration. Investigate and remeasure before sizing production concurrency. The historical
> 128-stream baseline remains available in the [serving performance benchmark](docs/dev/investigation/serving_performance_benchmark.md).

- Standalone `engine-grpc` TTFT is measured by default over a ready/reused gRPC channel and, like WebSocket, does not count the client connection setup cost toward first-packet latency; a cold/lazy channel adds roughly 13ms.
- A one-off browser metric in the Demo Lab uses a different measurement window and load shape from the table above and is not directly comparable; public claims must use benchmark data with complete conditions.
- The product Demo calls only the current instance's public `/v1/realtime`; when a live backend is unavailable it fails explicitly and never falls back to fixtures or simulated audio.

For detailed benchmark methodology, see [Benchmark Methodology](docs/user/benchmark_methodology.md).

## Capability Status

| Path | Current status | Open-source scope |
| --- | --- | --- |
| `custom-1.7b` / `custom_voice` | 🟢 Prioritized/stable | The v0.2 recommended path; the product Demo showcases it by default |
| streaming TN / monotonic commitment | 🟢 Integrated | The primary TN owns spoken-form truth; open semantic tails may wait, then resolve or use an explicit fallback |
| native text progress | 🟢 Limited scope | Only a matched `custom-1.7b` cursor-enabled TRT artifact; other models fall back to EMA/disabled |
| `SOFT_DRAIN` / state rollover | ⚪ Not released | Design and acceptance contracts exist; current runtime continues to use `WAIT_TEXT` and explicit hard finalization |
| `design-1.7b` / `voice_design` | 🟡 Experimental | Code and export entry points can be kept, but must be marked as not fully validated |
| `base-1.7b` / x-vector voice clone | 🟡 Experimental | Standalone already wires up ref audio → speaker embedding; needs the base export artifacts and real end-to-end validation |
| `icl` voice clone | 🟡 Experimental | Standalone already wires up ref audio + ref text → ref codec/code injection; needs the TRT ref-audio engine and real end-to-end validation |
| `0.6b` variants | ⚪ Outside the v0.2 stable scope | Export/download entry points can be kept, but need separate validation before release |

## Prerequisites

- **GPU**: an NVIDIA GPU, ≥16GB VRAM recommended (1.7B + KV pool + TensorRT runtime); a matching NVIDIA driver is required.
- **CUDA / TensorRT**: provided via NVIDIA NGC containers (`nvcr.io/nvidia/tensorrt`, `nvcr.io/nvidia/tritonserver`); see `scripts/bash/ngc_matrix.conf` for the version matrix. **Pulling an NGC image constitutes acceptance of the NVIDIA EULA.**
- **Docker**: used to orchestrate the engine/Triton containers (with the NVIDIA Container Toolkit to enable `--gpus`).
- **Python environment**: Phase A manages the host Python environment via conda; if none is present, `setup_env.sh` downloads and installs [Miniforge](https://github.com/conda-forge/miniforge) (BSD-3-Clause) and creates the `qwen3-tts` conda environment. You may instead activate your own conda env or venv beforehand.
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

Default ports: gRPC `50051`, native WebSocket `ws://localhost:50052/v1/ws`, OpenAI Realtime compatibility endpoint `ws://localhost:50052/v1/realtime`, HTTP capabilities `http://localhost:50052/v1/capabilities`, health `http://localhost:8080/health` (binds at process start; returns `503` while the model loads, `200` once ready — see [deployment](docs/user/deployment.md) for probe details).

### Engine Docker

A standalone engine container that uses the same model package; the image contains the runtime and the `/app/engine` code.

```bash
# Assemble artifacts + rebuild the image
bash scripts/bash/autorun.sh package -m custom-1.7b --gateway engine-docker --build
# Start the service
bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway engine-docker --engine-mode trt
```

Engine Docker publishes gRPC `50051` and the public gateway `50052`. Its
early-start health listener remains container-internal and is used by Docker's
healthcheck; public `/health`, `/demo/`, `/sdk/`, and both WebSocket protocols
all share `50052`. It therefore does not reserve a host `8080` port.

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

The Triton deployment starts both Triton and a public WebSocket sidecar. Its
default public endpoints are native WebSocket `ws://localhost:50053/v1/ws`,
OpenAI Realtime compatibility `ws://localhost:50053/v1/realtime`, capabilities
`http://localhost:50053/v1/capabilities`, and health
`http://localhost:50053/health`; Triton's native HTTP/gRPC/metrics ports remain
`8000/8001/8002`. Set `--realtime-port` on `compose.sh` to change the host port.
Completed and partial-response usage is returned on the wire and appended to
`workspace/realtime_usage/realtime_usage.jsonl` for billing ingestion.

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

The Python SDK prefers native `engine-websocket` when `transport="auto"` can
discover it. OpenAI Realtime remains available as a compatibility transport
without a deprecation warning; engine-grpc, triton-grpc, and triton-http are
older direct transports that remain available with deprecation warnings.

SDK compatibility is determined by the wire-protocol family and major reported
by `GET /v1/capabilities`. Release skew in `engine_version` is diagnostic and
only produces a warning; installing the engine's wheel is still the simplest
way to reproduce an exactly matched environment:

```bash
curl https://<public-service-base>/v1/capabilities
# → {"engine_version": "v0.2.0", ...}

# For an exact, copyable command, open the current instance's /demo/#/sdk page.
# Forge users can instead select the wheel attached to the matching release:
# https://github.com/X-Square-Robot/Qwen3TTS-Streaming/releases

# The public service serves the exact same published wheel. The index and its
# links stay relative, including behind an /infer/<instance> proxy prefix.
curl https://<public-service-base>/sdk/    # list, then use the returned filename:
pip install "https://<public-service-base>/sdk/<wheel-filename>"

# Or from a local checkout
pip install "./client[all]"
```

Quick usage:

```python
from qwen3tts import TTSClient, SynthesisConfig

client = TTSClient.connect("ws://localhost:50052/v1/ws")
result = client.synthesize_bytes(
    "你好，欢迎使用 Qwen3-TTS。",
    request=SynthesisConfig(task_type="custom_voice"),
)
print(result.details["usage"])
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
print(session.response_id, session.response_status, session.usage)
```

For detailed documentation, see [Client SDK](docs/user/client_sdk.md) and the [`client/`](client) subproject.

## Testing and Acceptance

Test entry points are unified under `tests/`; for a detailed map, see [tests/README.md](tests/README.md).

```bash
# Unit + integration tests
mamba run -n qwen3-tts pytest tests/unit tests/integration -q

# Main entry point for serving acceptance and benchmarks
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py --targets engine-grpc
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py --targets triton-grpc,triton-http

# Current SDK performance matrix (add Triton only when its service is running)
mamba run -n qwen3-tts bash -c \
  'TARGETS="engine-grpc,engine-websocket" LEVELS="1,8,16,32,64,128" \
   CONCURRENCY_SAMPLES=20 CONCURRENCY_WARMUP=3 CONN_SAMPLES=50 CONN_WARMUP=5 \
   MAX_CONNECTIONS=128 \
   bash tools/validation/run_perf_matrix.sh'
mamba run -n qwen3-tts python tools/validation/summarize_perf_matrix.py \
  workspace/perf_matrix/<run_id>
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

## Built-in Demo and documentation

Every release runtime image contains one version-matched portal at `/demo/`.
It is enabled by default on the same public port as `/v1/ws`, `/v1/realtime`, and `/sdk/`;
set `DEMO_ENABLED=false` at startup to disable it. The portal discovers the
current instance and synthesizes through the
Browser SDK, plays PCM through the system speaker, exposes capability-gated VAD
and delivery controls, downloads WAV, and renders this repository's Markdown.
No separate Demo API or Node process is required for the normal experience.

Contributors working on the portal should start with the
[`web/packages/demo` development guide](web/packages/demo/README.md).

The former standalone `webui/` feature showcase has been consolidated into
`web/packages/demo`; this is now the only browser frontend and portal entry
point. Use `/demo/#/lab` for the unified engineering Lab. Its basic LLM
PK and concurrency experiments use the instance's public `/v1/realtime` path;
trace, LLM PK, concurrency, and capability inspection all use the same public
Realtime and capabilities endpoints, and never connect to Triton directly.

CI packages the Browser SDK once as an npm tarball and embeds those exact bytes
under `/demo/downloads/`. The SDK page generates an `npm install
"https://<instance>/demo/downloads/<package>.tgz"` command, so consumers do not
need a repository checkout. GitLab releases additionally publish the same
archive to the project npm Registry.

The built-in Lab provides LLM PK, concurrency, and event trace over public
Realtime. Results describe only the current browser-to-instance run; the page
never substitutes fixtures or hard-coded performance numbers for a live backend.

The following media are retained as historical captures from the former
standalone WebUI. They illustrate the experiments, but their layout and any
numbers shown in them are not current UI or benchmark claims.

**Historical LLM PK demo asset** — streaming vs. non-streaming, same timeline

![Streaming vs. non-streaming comparison demo](docs/images/流式非流式对比.gif)

**Historical concurrency demo asset** — multi-stream distribution and throughput

![Multi-stream synthesis demo](docs/images/多路合成.gif)

Full screen recording: [演示视频.mp4](docs/videos/演示视频.mp4)

Standalone startup:

```bash
bash scripts/bash/compose.sh up --build --gateway engine --variant custom-1.7b
```

Open `http://localhost:50052/demo/`. For the Triton deployment, use
`--gateway triton` and open `http://localhost:50053/demo/`. Reverse proxies may
mount the service below `/infer/<instance>`; all portal, SDK, WebSocket and asset
links remain relative to that prefix.
No Vite dev server or second WebUI process is needed for this runtime path.

HTTP/WS is the default local protocol; mounting certificate files does not
enable HTTPS automatically. For direct self-signed WSS testing, the Python SDK
can strictly trust one certificate with
`tls_verify="/path/to/cert.local.pem"`, or use `tls_verify=False` only during
temporary local debugging. Browser and Python trust stores are independent.

When Kubernetes permits only one public port, set
`PORT=8000 HEALTH_PORT=0` on the engine container and expose
only `8000` in the Service. `/demo/`, `/sdk/`, `/health`, `/v1/ws`, and `/v1/realtime` then
share that port. See the [deployment guide](docs/user/deployment.md#single-port-kubernetes-deployment)
for complete probe and Service examples.

A development host without an Ingress can also set `TLS_CERT_FILE` and
`TLS_KEY_FILE`, or set `TLS_AUTO_ENABLE=true` for the bundled local certificate,
as in FunASR Nano, to serve HTTPS/WSS directly from that same public port. See
[direct HTTPS/WSS](docs/user/deployment.md#direct-httpswss-on-a-development-host)
for the certificate mount contract.

The built-in **Lab** tab runs LLM PK, concurrency, and event trace experiments
through the same public Realtime endpoint:

```bash
bash scripts/bash/compose.sh up --build --gateway engine --variant custom-1.7b
```

## Streaming Protocol

The official Python SDK uses native `/v1/ws` as its primary protocol; `/v1/realtime` is the OpenAI Realtime compatibility endpoint. The compatibility endpoint is full duplex: complete text uses `conversation.item.create` plus `response.create`, and token-level input uses the `qwen.input_text_buffer.append/commit` extension. See [OpenAI Realtime TTS Protocol and Triton Boundary](docs/dev/architecture/openai_realtime.md).

Native `/v1/ws` uses these control frames:

```json
{"type":"start","session_id":"demo","config":{"task_type":"custom_voice","speaker":"Serena"}}
{"type":"text","text":"你好，世界。"}
{"type":"stop"}
```

`stop` gracefully ends input and drains audio (`end` remains an alias); `cancel`
aborts the active session. The server returns JSON event frames and binary PCM
frames. A `done`/`error` event—not socket closure—is the logical session
boundary. After a successful or cancelled `done`, the same WebSocket can accept
another `start`; an engine `error` closes it so the next session reconnects.

Active WebSocket streams also support bounded in-process resume. The SDK sends
sequenced text and acknowledges exact output deliveries; after a transient
network/proxy disconnect it leases a replacement socket and continues the same
engine execution from its last complete audio sample. It never restarts the
synthesis and guesses at de-duplication. Resume state is process-local and
expires, so engine restarts fail explicitly and multi-replica deployments need
sticky or token-consistent routing.

## Project Structure

```text
Qwen3TTS-Streaming/
├── engine/                     # Inference engine: frontend/backend/gateway/core
├── client/                     # Standalone Python SDK package (qwen3-tts-client, released as a wheel)
│   ├── src/qwen3tts/           #   Client implementation and transport adapters
│   └── src/qwen3tts_protocol/  #   Shared protocol layer (single source of truth)
├── web/                        # Browser SDK and the single React/Vite product portal
│   ├── packages/browser-sdk/   # Browser SDK package
│   └── packages/demo/          # Unified /demo/ portal (experience/SDK/docs/Lab)
├── proto/                      # Single source of the protocol definition (tts.proto + generated code)
├── model_repository/           # Triton Python BLS model definitions
├── infra/
│   └── docker/                 # Dockerfile + compose configuration
├── scripts/
│   ├── bash/                   # autorun/setup/build/deploy lifecycle
│   ├── compose/                # Container entry-point scripts
│   ├── demo/                   # Demo / engineering-lab launchers
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

This project is a **v0.2 stability-focused release**. The validated checkpoint substantially
reduces runaway hallucination, while broader checkpoint and workload validation remains valuable;
you are welcome to participate via issues, discussions, and PRs.

- 🤝 [Contributing Guide](CONTRIBUTING.md) — development environment, testing, proto workflow, code style
- 💬 [Support Channels](SUPPORT.md) — how questions / bug reports / suggestions are routed
- 🔒 [Security Policy](SECURITY.md) — the private vulnerability reporting process (please do not file public issues)
- 📜 [Code of Conduct](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1
- 📝 [Changelog](CHANGELOG.md) — record of version changes

## License

- **This project's own code** (`engine/`, `client/`, `web/`, `scripts/`, etc.) is released under the [MIT](LICENSE) license, copyright XSquareRobot.
- **Upstream [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** (the `third_party/` submodule) is Apache 2.0, which is compatible with MIT.
- **Model weights** are released by Qwen/Alibaba; their license is governed by the respective [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) / [Hugging Face](https://huggingface.co/Qwen) model cards; this repository does not distribute any weights.
- **TensorRT / Triton Inference Server** (NVIDIA NGC images) are NVIDIA proprietary software, not bundled in this repository; using them constitutes acceptance of the NVIDIA EULA.
- **[TEN VAD](https://github.com/TEN-framework/ten-vad)** is an **optional** dependency, used only by the experimental `tenvad` VAD mode (disabled by default; installed by the user, not bundled). It is licensed under **Apache 2.0 with additional conditions** (non-compete, single-applicant use) — **not** a standard permissive license; review its terms before enabling that mode.
- The reference audio under `resources/speakers/` is **synthetic audio** with **fictional** speaker names, corresponding to no real individuals.

For full third-party attribution, see [NOTICE](NOTICE).
