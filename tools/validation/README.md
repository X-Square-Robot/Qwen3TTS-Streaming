# Manual Tools

`tools/validation/` contains scripts for validation, benchmarking, audio inspection, and debugging.
These are **not** pytest tests — they are standalone CLI tools.

Shared types and schemas live in `qwen3_tts_protocol`.  Repo-internal helpers
(path constants, bootstrap) live in `scripts/python/qwen3tts_tools/`.

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

### `verify_engine.py` — Engine-level verification

Replaces: `verify_e2e.py`, `verify_e2e_trt.py`, `verify_e2e_trt_ref.py`,
`verify_fused_triton_backend.py`, `verify_onnx_autoregressive.py`,
`verify_code_predictor_trt.py`, `verify_trt_talker.py`, `verify_multi_variant.py`

```bash
python tools/validation/verify_engine.py --phase e2e              # Full decomposed pipeline
python tools/validation/verify_engine.py --phase e2e-trt          # TRT engine verification
python tools/validation/verify_engine.py --phase e2e-trt-ref      # Generate TRT reference
python tools/validation/verify_engine.py --phase fused-triton     # ORT vs Triton TRT parity
python tools/validation/verify_engine.py --phase code-predictor   # CP TRT parity
python tools/validation/verify_engine.py --phase trt-talker       # Talker TRT verification
python tools/validation/verify_engine.py --phase multi-variant    # Multi-variant prefill
```

### `verify_components.py` — Component-level verification

Replaces: `verify_code2wav_streaming.py`, `verify_speech_tokenizer_encoder.py`,
`verify_precision_ort.py`, `verify_prototype_parity.py`

```bash
python tools/validation/verify_components.py --component code2wav           # Code2wav streaming
python tools/validation/verify_components.py --component speech-tokenizer    # Speech tokenizer
python tools/validation/verify_components.py --component precision-ort      # FP32 precision
python tools/validation/verify_components.py --component prototype-parity   # Prototype parity
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

## Helper Modules (not standalone CLIs)

| Module | Used by |
|--------|---------|
| `_bootstrap.py` | Path bootstrap for tools |
| `triton_tts_client.py` | compare_audio, benchmark |
| `trt_direct.py` | verify_engine |

## Maintenance

Before adding a new tool here, check whether an existing unified tool already covers
the use case (e.g. `compare_audio.py --mode` instead of a new compare script).
Shared logic belongs in `qwen3_tts_protocol` or `scripts/python/qwen3tts_tools/`.
