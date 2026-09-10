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

### 当前引擎性能矩阵

涉及 streaming TN/native cursor 的性能重测统一使用 `qwen3-tts` 虚拟环境中的 SDK
矩阵脚本；目标服务必须已启动。没有 Triton 服务时只跑 standalone engine，并在报告中
明确这是 engine-only refresh：

```bash
mamba run -n qwen3-tts bash -c \
  'TARGETS="engine-grpc,engine-websocket" LEVELS="1,8,16,32,64,128" \
   CONCURRENCY_SAMPLES=20 CONCURRENCY_WARMUP=3 CONN_SAMPLES=50 CONN_WARMUP=5 \
   MAX_CONNECTIONS=128 \
   bash tools/validation/run_perf_matrix.sh'
mamba run -n qwen3-tts python tools/validation/summarize_perf_matrix.py \
  workspace/perf_matrix/<run_id>
```

`run_perf_matrix.sh` 是当前跨协议矩阵入口；`summarize_perf_matrix.py` 生成
`raw_requests.csv` 和 `summary.csv`。`benchmark.py` 以及早期直连脚本只保留兼容/调查
用途，不能替代当前矩阵的环境快照和 connection-mode 对照。

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

### `fused_onnx_trt_parity.py` —— 冻结输入的 ORT/TRT 对照

从真实 fused Executor 捕获 prefill/decode 输入，再把同一组输入送入 ORT 和一个或多个
TRT plan，报告 `full_codec`、hidden、logits、C2W、PCM 与 cursor 输出的逐步误差。适合
定位 BF16/FP32 与 graph 精度问题，不等价于完整 serving 验收。

传入 `--torch-model /path/to/source-checkpoint` 时，还会用导出器的同一 PyTorch fused
wrapper 对冻结输入运行参考，并额外输出 `torch_vs_onnx` 与 `torch_vs_trt`。这要求源模型
目录包含完整 Talker、speech tokenizer，以及 cursor-enabled plan 所需的模型-owned
`qwen3_tts_12hz_la1_seed0.pt`；它仍是冻结 graph-input 对照，不替代端到端服务验收。

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

cursor plan 默认固定使用已验证的 TRT profile 0；也可用 `--trt-profile N` 显式指定。
报告应使用与 capture artifact 相同的 manifest dtype，且 cursor 包必须携带匹配的
model-owned head；否则工具会暴露 package/engine 输入合同不一致，而不是生成可用证据。

## 独立工具

| Tool | Purpose |
|------|---------|
| `gen_audio.py` | 通过 Triton TTS 生成 WAV 样本 |
| `serving_endpoints.py` | 完整 serving 验收 + TTFT 基准 |
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
