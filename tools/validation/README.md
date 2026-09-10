**English** | [中文](README.zh-CN.md)

# Manual Tools

`tools/validation/` contains scripts for validation, benchmarking, audio inspection, and debugging.
These are **not** pytest tests — they are standalone CLI tools.

Shared types and schemas live in `qwen3tts_protocol`.  Repo-internal helpers
(path constants, bootstrap) live in `scripts/python/`.

## Primary Tool

**`serving_endpoints.py`** — canonical full serving acceptance and benchmark entry point.

```bash
python tools/validation/serving_endpoints.py --targets engine-grpc
python tools/validation/serving_endpoints.py --targets triton-grpc,triton-http
python tools/validation/serving_endpoints.py --targets engine-grpc --ttft-warmup 3 --ttft-samples 30
```

### Current Engine Performance Matrix

Use the SDK-based matrix from the `qwen3-tts` virtual environment when refreshing streaming
TN/native-cursor performance. Target services must already be running. Without Triton, run only
the standalone engine targets and label the report as an engine-only refresh:

```bash
mamba run -n qwen3-tts bash -c \
  'TARGETS="engine-grpc,engine-websocket" LEVELS="1,8,16,32,64,128" \
   CONCURRENCY_SAMPLES=20 CONCURRENCY_WARMUP=3 CONN_SAMPLES=50 CONN_WARMUP=5 \
   MAX_CONNECTIONS=128 \
   bash tools/validation/run_perf_matrix.sh'
mamba run -n qwen3-tts python tools/validation/summarize_perf_matrix.py \
  workspace/perf_matrix/<run_id>
```

`run_perf_matrix.sh` is the current cross-protocol matrix entry point;
`summarize_perf_matrix.py` writes `raw_requests.csv` and `summary.csv`.
`benchmark.py` and older direct-client scripts remain compatibility/investigation tools and do not
replace the current environment snapshot or connection-mode comparison.

## Unified Tools (replaces many legacy scripts)

### `compare_audio.py` — Audio generation and comparison

Replaces: `gen_engine_audio.py`, `gen_reference_audio.py`,
`compare_official_vs_triton_audio.py`, `compare_official_vs_fused_onnx.py`,
`full_chain_audio_listen.py`, `generate_audio_compare.py`,
`fused_onnx_audio.py`, `long_streaming_listen_ab.py`

```bash
python tools/validation/compare_audio.py gen-triton       # Generate WAVs via Triton
python tools/validation/compare_audio.py gen-engine       # Generate WAVs via engine gRPC
python tools/validation/compare_audio.py gen-reference    # Generate reference audio
python tools/validation/compare_audio.py compare-triton   # Official vs Triton compare
python tools/validation/compare_audio.py compare-ort      # Official vs ORT fused compare
python tools/validation/compare_audio.py compare-full-chain  # Full chain compare
python tools/validation/compare_audio.py fused-onnx       # Run fused ONNX to WAV
python tools/validation/compare_audio.py long-ab           # Long text A/B compare
```

### `benchmark.py` — Performance benchmarks

Replaces: `engine_standalone_benchmark.py`, `triton_concurrent_tts.py`

```bash
python tools/validation/benchmark.py --target engine-standalone   # Engine gRPC benchmark
python tools/validation/benchmark.py --target triton-concurrent   # Triton concurrent test
```

### `prefill_compare.py` — Prefill path comparison

Replaces: `official_prefill.py`, `compare_live_vs_exported_prefill.py`,
`compare_prefill_paths.py`, `official_vs_manual_rollout.py`, `cp_sampled_parity.py`

```bash
python tools/validation/prefill_compare.py --mode official            # Official prefill builder
python tools/validation/prefill_compare.py --mode live-vs-exported    # Live vs exported prefill
python tools/validation/prefill_compare.py --mode compare-paths       # Prefill path comparison
python tools/validation/prefill_compare.py --mode manual-rollout      # Manual decode rollout
python tools/validation/prefill_compare.py --mode cp-parity           # CP sampled parity
```

### `fused_onnx_trt_parity.py` — Frozen-input ORT/TRT comparison

Captures prefill/decode inputs from a real fused Executor, then feeds the same tensors to ORT
and one or more TRT plans. It reports per-step differences for `full_codec`, hidden/logits,
C2W, PCM, and cursor outputs. Pass `--torch-model` to also run the exporter-owned PyTorch
fused wrapper on the same inputs. This is a graph precision diagnostic, not full serving
acceptance.

```bash
ENGINE_CUDA_GRAPH_DECODE=0 python tools/validation/fused_onnx_trt_parity.py \
  --capture-artifact /path/to/cursor-artifact \
  --onnx-artifact /path/to/cursor-artifact \
  --torch-model /path/to/X2Streaming-TTS-1.7B \
  --trt-artifact /path/to/bf16-artifact \
  --trt-artifact /path/to/fp32-artifact \
  --steps 4 --random-prefill --include-prefill \
  --output /tmp/fused-parity.json
```

Cursor plans default to the validated TRT profile 0; use `--trt-profile N` to override it.
The capture artifact dtype must match its manifest, and a cursor package must include the
matching model-owned cursor head. Otherwise the tool should expose the package/engine contract
failure rather than produce evidence.

## Standalone Tools

| Tool | Purpose |
|------|---------|
| `gen_audio.py` | Generate WAV samples via Triton TTS |
| `serving_endpoints.py` | Full serving acceptance + TTFT benchmark |
| `pad_tolerance_experiment.py` | Pad token insertion study |
| `vad_verification.py` | VAD verification |
| `trt_direct.py` | Direct TRT engine decode-to-WAV |

## Helper Modules (not standalone CLIs)

| Module | Used by |
|--------|---------|
| `_bootstrap.py` | Path bootstrap for tools |
| `triton_tts_client.py` | compare_audio, benchmark |

## Maintenance

Before adding a new tool here, check whether an existing unified tool already covers
the use case (e.g. `compare_audio.py --mode` instead of a new compare script).
Shared logic belongs in `qwen3tts_protocol` or `scripts/python/`.
