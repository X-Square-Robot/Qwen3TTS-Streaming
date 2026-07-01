[English](README.md) | **中文**

# 手动工具

`tools/validation/` 包含用于验证、基准、音频检查和调试的脚本。
它们**不是** pytest 测试——而是独立的 CLI 工具。

共享的类型与 schema 位于 `qwen3tts_protocol`。仓库内部辅助
（路径常量、bootstrap）位于 `scripts/python/`。

## 主要工具

**`serving_endpoints.py`** —— 规范的完整 serving 验收与基准入口。

```bash
python tools/validation/serving_endpoints.py --targets engine-grpc
python tools/validation/serving_endpoints.py --targets triton-grpc,triton-http
python tools/validation/serving_endpoints.py --targets engine-grpc --ttft-warmup 3 --ttft-samples 30
```

## 统一工具（取代大量遗留脚本）

### `compare_audio.py` —— 音频生成与对比

取代：`gen_engine_audio.py`、`gen_reference_audio.py`、
`compare_official_vs_triton_audio.py`、`compare_official_vs_fused_onnx.py`、
`full_chain_audio_listen.py`、`generate_audio_compare.py`、
`fused_onnx_audio.py`、`long_streaming_listen_ab.py`

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

### `benchmark.py` —— 性能基准

取代：`engine_standalone_benchmark.py`、`triton_concurrent_tts.py`

```bash
python tools/validation/benchmark.py --target engine-standalone   # Engine gRPC benchmark
python tools/validation/benchmark.py --target triton-concurrent   # Triton concurrent test
```

### `prefill_compare.py` —— Prefill 路径对比

取代：`official_prefill.py`、`compare_live_vs_exported_prefill.py`、
`compare_prefill_paths.py`、`official_vs_manual_rollout.py`、`cp_sampled_parity.py`

```bash
python tools/validation/prefill_compare.py --mode official            # Official prefill builder
python tools/validation/prefill_compare.py --mode live-vs-exported    # Live vs exported prefill
python tools/validation/prefill_compare.py --mode compare-paths       # Prefill path comparison
python tools/validation/prefill_compare.py --mode manual-rollout      # Manual decode rollout
python tools/validation/prefill_compare.py --mode cp-parity           # CP sampled parity
```

## 独立工具

| Tool | Purpose |
|------|---------|
| `gen_audio.py` | 通过 Triton TTS 生成 WAV 样本 |
| `serving_endpoints.py` | 完整 serving 验收 + TTFT 基准 |
| `suggest_engine_profile.py` | 建议 TensorRT/运行时 profile 上限 |
| `pad_tolerance_experiment.py` | Pad token 插入研究 |
| `vad_verification.py` | VAD 验证 |
| `trt_direct.py` | 直接 TRT 引擎解码到 WAV |

## 辅助模块（非独立 CLI）

| Module | Used by |
|--------|---------|
| `_bootstrap.py` | 工具的路径 bootstrap |
| `triton_tts_client.py` | compare_audio, benchmark |

## 维护

在此处添加新工具之前，请检查是否已有统一工具覆盖了该用例
（例如用 `compare_audio.py --mode` 而不是新的 compare 脚本）。
共享逻辑属于 `qwen3tts_protocol` 或 `scripts/python/`。
