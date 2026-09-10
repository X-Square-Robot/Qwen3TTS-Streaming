**English** | [中文](benchmark_methodology.zh-CN.md)

# Benchmark Methodology

This document defines the benchmark conventions used by the product Demo, README, and release traces. All published performance numbers must carry these conditions.

The unified `/demo/#/lab` page is an exploratory browser client: its basic LLM
PK and concurrency runs use the instance's public `/v1/realtime` endpoint and
report browser-side timings. The optional `demo_api` service exposes separate
server-side Triton/trace routes. Record which source produced a result and do
not compare browser-Lab or `demo_api` measurements directly with the published
engine benchmark table.

## Metric Definitions

| Metric | Meaning |
| --- | --- |
| TTFT | Time To First Token/Frame. The time for the server to produce the first sendable audio chunk. |
| TTFB | Time To First Byte. The time for the client to receive the first segment of audio bytes. |
| first playable | The time at which the client can start playback. In offline mode this usually equals the time the entire audio finishes generating. |
| first audible | The time at which the player actually hears non-silent audio, which may be later than first playable. |
| total latency | The time to finish all audio generation or to end the stream. |
| audio duration | The duration of the output audio, used to compute the real-time rate. |

Metric meanings cannot be mixed across different backends:

- Official PyTorch Offline: primarily look at `first_playable_ms`, because playback can only start after the entire audio is complete.
- Official PyTorch Streaming: primarily look at the server's first packet and the client's first byte.
- Bare Engine Streaming: primarily look at the standalone engine's WebSocket/gRPC first packet.
- Triton TRT Streaming: primarily look at the TensorRT/Triton streaming first packet.

## Required Test Conditions

Public benchmarks must record:

- GPU model, memory, and machine specs.
- NVIDIA driver, CUDA, TensorRT, NGC image tag.
- Model variant, e.g. `custom-1.7b`.
- engine profile: `max_batch_size`, `max_input_len`, `max_seq_len`, dtype, I/O dtype.
- Deployment mode: standalone, engine-docker, Triton.
- Input text, language, speaker, cache mode.
- Warmup strategy: whether warmup is excluded.
- Concurrency, request count, failure count.
- For SDK WebSocket runs, `max_connections` must be at least the largest
  measured concurrency; otherwise the client pool serializes lanes before the
  engine sees them.
- Measurement location: server, adapter, client.
- Experiment source: built-in `/demo/#/lab` over public Realtime, optional
  `demo_api`, or a direct engine/Triton client.
- Whether a fixture trace is used.

## Streaming TN and Native Cursor Reporting

Streaming TN and text progress are correctness and protocol capabilities; they cannot be inferred
from TTFT or throughput alone. A new benchmark report should also record:

- `loaded_model_type`, `native_cursor.graph_enabled`,
  `native_cursor.progress_available`, and `supported_progress_modes` from
  `/v1/capabilities`;
- model/cursor-head/label-vocabulary/TN-rules fingerprints, plus `right_context`,
  `max_labels`, codebook count, and frame duration;
- the actual progress route: `native`, `ema`, or `disabled`, and the event
  `progress_basis`;
- TN replay or contract-test results. `raw_codepoint_end` and
  `normalized_codepoint_end` are monotonic integer high-water marks and must
  not be replaced by display interpolation;
- whether `WAIT_TEXT`, EOS, tail rewrite, concurrent slot isolation, and the
  no-cursor-head fallback were verified.

Keep performance matrices separate from text-progress correctness. Do not average standard
TRT/EMA, cursor-enabled TRT/native, or Triton results from different artifacts into one number.

## Current Matrix Entry Point

The performance matrix uses the real SDK client and requires the target services to be running.
Run it inside the project virtual environment:

```bash
mamba run -n qwen3-tts bash -c \
  'TARGETS="engine-grpc,engine-websocket" LEVELS="1,8,16,32,64,128" \
   CONCURRENCY_SAMPLES=20 CONCURRENCY_WARMUP=3 CONN_SAMPLES=50 CONN_WARMUP=5 \
   MAX_CONNECTIONS=128 \
   bash tools/validation/run_perf_matrix.sh'
mamba run -n qwen3-tts python tools/validation/summarize_perf_matrix.py \
  workspace/perf_matrix/<run_id>
```

Add Triton to `TARGETS` only when the corresponding Triton service is running. Do not represent an
unavailable Triton run with empty results or historical data. Preserve raw JSON, environment
snapshots, and summary CSVs by run id, and state whether the refresh is engine-only.

## The Single-Stream TTFT Convention

Single-stream TTFT can only be described as a measured distribution under specific conditions, not as default performance or a stable guarantee. The current measured reference (2026-07-08, RTX 5090, all-bf16 `custom-1.7b`, batch=128 profile) is a **server TTFT of 14.9 ± 0.3ms (min 14.4, p99 15.9, n=50)** — see the [serving performance benchmark](../dev/investigation/serving_performance_benchmark.md) for the full breakdown and raw data. The historical "13ms" figure was the lowest observed value on an older engine build and should no longer be quoted. Any such number requires all of the following to hold simultaneously:

- Cache hit.
- The engine is already warm.
- Single-stream request.
- Fixed input text and speaker.
- Fixed engine profile.
- Local deployment or a near-local link.

Recommended public wording:

```text
Under the specified hardware, a warm engine, a cache hit, a single-stream request, and the custom-1.7b profile, the measured server TTFT is 14.9 ± 0.3ms (min 14.4ms, n=50).
```

Discouraged wording:

```text
TTFT 15ms.
```

## The 128-Stream Concurrency Convention

Concurrency benchmarks must report at least:

- concurrency.
- avg / p50 / p90 / p99 / max TTFT.
- failed streams.
- total elapsed time.
- throughput audio sec/sec.
- whether live concurrency is enabled.

The built-in Lab's browser-side concurrency is not the 128-stream benchmark;
it is intended for an interactive, instance-local comparison. For an optional
server-side concurrency job, start `demo_api` and record that source explicitly.
The Compose `demo` profile defaults live concurrency to `0`; direct module
execution defaults to `1`, so set the variable deliberately:

```bash
QWEN_DEMO_ENABLE_LIVE_CONCURRENCY=1 python -m demo_api --port 7860
```

## Collecting a Release Trace

Example command:

```bash
python scripts/demo/collect_demo_traces.py \
  --live-triton \
  --triton-grpc localhost:8001 \
  --output workspace/demo_traces/race_default.json
```

After collection, check:

- Whether `source` is `live_triton` / `live_engine` / `live_official_pytorch`.
- Whether `benchmark_conditions` is complete.
- Whether `warnings` is empty or explained.
- Whether the audio is playable and free of obvious misreadings, repetition, or hallucination.

## Release Report Template

```text
Model: custom-1.7b
Gateway: triton
GPU:
Driver:
NGC image:
TensorRT:
Engine profile:
Precision:
Input:
Cache:
Warmup:
Concurrency:
Requests:
Failures:

TTFT avg/p50/p90/p99/max:
TTFB avg/p50/p90/p99/max:
Throughput:
Known quality issues:
```
