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

## Standalone Tools

| Tool | Purpose |
|------|---------|
| `gen_audio.py` | Generate WAV samples via Triton TTS |
| `serving_endpoints.py` | Full serving acceptance + TTFT benchmark |
| `suggest_engine_profile.py` | Suggest TensorRT/runtime profile limits |
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
