**English** | [中文](e2e_test_summary.zh-CN.md)

# End-to-End Testing and Acceptance Entry Points

## 1. Entry-point Layering

The E2E-related entry points are now separated by responsibility:

| Entry point | Type | Description |
| --- | --- | --- |
| `tests/e2e/test_e2e.py` | pytest | End-to-end assertions for the Triton gRPC `tts_orchestrator`, covering basic synthesis, error handling, and a first-chunk latency smoke test. |
| `tests/e2e/test_engine_standalone.py` | pytest | End-to-end assertions for the bare standalone engine gRPC, auto-skipped when the service is unreachable. |
| `tools/validation/serving_endpoints.py` | Manual acceptance / benchmark | Unified serving tool covering `engine-grpc`, `engine-websocket`, `triton-grpc`, and `triton-http`. |
| `tools/validation/*.py` | Manual tools | Audio generation, full-path listening, ONNX/TRT comparison, long-text investigation, export verification, etc. |

See [`tests/README.md`](../../../tests/README.md) for the full test map.

## 2. Triton Pytest E2E

After starting Triton, run:

```bash
bash scripts/bash/deploy.sh run --gateway triton
pytest tests/e2e/test_e2e.py -v -s
```

Coverage:

| Case | Description |
| --- | --- |
| `test_e2e_voice_design_or_custom` | Basic `custom_voice`/`voice_design` streaming audio. |
| `test_e2e_custom_voice` | The `speaker + instruct` custom voice path. |
| `test_e2e_error_*` | Empty text, invalid task type, and voice clone with missing or corrupted ref audio. |
| `test_e2e_first_chunk_latency` | Smoke test for the Triton orchestrator's first audio chunk latency. |

The `voice_clone`-related pytest cases are skipped by default, because they require real ref audio and the full ref audio path.

## 3. Standalone Engine Pytest E2E

After starting the bare engine, run:

```bash
python -m engine.server --config engine.yaml
pytest tests/e2e/test_engine_standalone.py -v -s
```

Coverage:

| Case group | Description |
| --- | --- |
| `TestEngineSmokeAndStreaming` | capabilities, single-shot synthesis, English text, streaming text. |
| `TestEngineCustomVoiceInstruct` | custom voice instruct, auto-skipped for unsupported models. |
| `TestEngineLongText` | medium/very long text rollover. |
| `TestEngineBadCases` | empty text, whitespace text, single character, cancel. |
| `TestEnginePerformance` | first-chunk latency smoke test, with a CI-friendly loose threshold assertion. |

## 4. Full Serving Acceptance Tool

`tools/validation/serving_endpoints.py` is the recommended manual acceptance entry point:

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py --targets engine-grpc,engine-websocket
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py --targets triton-grpc,triton-http
```

Default matrix:

- standalone engine gRPC: close to the full bare-engine pytest suite.
- standalone engine WebSocket: the same suite over the WebSocket transport.
- Triton gRPC: health + a real synthesis request + optional long text.
- Triton HTTP: health + model metadata/config + a real synthesis request + optional long text.

Common quick acceptance:

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py \
  --targets engine-grpc \
  --skip-long --skip-badcase
```

## 5. Bare-engine TTFT Distribution Benchmark

If you want to measure "a somewhat more accurate bare-engine TTFT", use the unified serving tool's TTFT distribution mode:

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py \
  --targets engine-grpc \
  --skip-single --skip-streaming --skip-custom-instruct \
  --skip-concurrent --skip-long --skip-badcase \
  --ttft-warmup 3 \
  --ttft-samples 30 \
  --ttft-text "今天天气真好。"
```

The output includes:

- `mean_ms`
- `variance_ms2` (sample variance)
- `population_variance_ms2`
- `stdev_ms`
- `coefficient_of_variation`
- `min/p50/p90/p95/max/range`
- a fluctuation bar for each sample relative to the mean
- structured details under `--json`

TTFT here is defined as the wall-clock time from when the client issues the standalone engine `SynthesizeOnce` request to when it receives the first audio chunk. It covers the entire bare-engine path — client gRPC, local scheduling, prefill/decode/code2wav through to first-chunk emission — and does not include the Triton orchestrator.

## 6. Metric Definitions

| Metric | Meaning |
| --- | --- |
| `first_chunk_ms` / `ttft_ms` | Wall-clock time from request start to the first playable audio chunk arriving at the client. |
| `total_ms` | Request start to the termination event / final response completion. |
| `chunks` | Number of audio chunks received. |
| `samples` / `duration_sec` | Number of generated audio samples and the corresponding duration. |
| `rtf` | Total wall-clock time / audio duration. |
| `decode_step_*` | Statistics of the inter-arrival intervals of consecutive audio chunks, used to observe streaming stability. |

Benchmark numbers must always carry the full conditions: hardware, driver, image/environment, engine profile, input text, warmup, sample count, target endpoint, sampling parameters, and failure rate.
