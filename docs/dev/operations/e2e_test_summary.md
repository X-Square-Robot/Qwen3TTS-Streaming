# 端到端测试与验收入口

## 1. 入口分层

E2E 相关入口现在按职责分开：

| 入口 | 类型 | 说明 |
| --- | --- | --- |
| `tests/e2e/test_e2e.py` | pytest | Triton gRPC `tts_orchestrator` 端到端断言，覆盖基础合成、错误处理、首包延迟 smoke。 |
| `tests/e2e/test_engine_standalone.py` | pytest | 裸 standalone engine gRPC 端到端断言，服务不可达时自动 skip。 |
| `tools/validation/serving_endpoints.py` | 手动验收/benchmark | 统一 serving 工具，覆盖 `engine-grpc`、`engine-websocket`、`triton-grpc`、`triton-http`。 |
| `tools/validation/*.py` | 手动工具 | 音频生成、全链路试听、ONNX/TRT 对比、长文本调查、导出验证等。 |

完整测试地图见 [`tests/README.md`](../../../tests/README.md)。

## 2. Triton Pytest E2E

启动 Triton 后运行：

```bash
bash scripts/bash/build_triton.sh run
pytest tests/e2e/test_e2e.py -v -s
```

覆盖范围：

| 用例 | 说明 |
| --- | --- |
| `test_e2e_voice_design_or_custom` | `custom_voice`/`voice_design` 基础流式音频。 |
| `test_e2e_custom_voice` | `speaker + instruct` custom voice 路径。 |
| `test_e2e_error_*` | 空文本、非法 task type、voice clone 缺少或损坏 ref audio。 |
| `test_e2e_first_chunk_latency` | Triton orchestrator 首个 audio chunk 延迟 smoke。 |

`voice_clone` 相关 pytest 默认 skip，因为需要真实 ref audio 和完整 ref audio 链路。

## 3. Standalone Engine Pytest E2E

启动裸 engine 后运行：

```bash
python -m engine.server --config engine.yaml
pytest tests/e2e/test_engine_standalone.py -v -s
```

覆盖范围：

| 用例组 | 说明 |
| --- | --- |
| `TestEngineSmokeAndStreaming` | capabilities、单次合成、英文文本、流式文本。 |
| `TestEngineCustomVoiceInstruct` | custom voice instruct，非支持模型自动 skip。 |
| `TestEngineLongText` | medium/very long 文本 rollover。 |
| `TestEngineBadCases` | 空文本、空白文本、单字、cancel。 |
| `TestEnginePerformance` | 首包延迟 smoke，阈值为 CI 友好的宽松断言。 |

## 4. 完整 Serving 验收工具

`tools/validation/serving_endpoints.py` 是推荐的人工验收入口：

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py --targets engine-grpc,engine-websocket
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py --targets triton-grpc,triton-http
```

默认矩阵：

- standalone engine gRPC：接近裸 engine pytest 的完整 suite。
- standalone engine WebSocket：同一 suite 的 WebSocket transport。
- Triton gRPC：health + 真实合成请求 + 可选长文本。
- Triton HTTP：health + model metadata/config + 真实合成请求 + 可选长文本。

常用快速验收：

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py \
  --targets engine-grpc \
  --skip-long --skip-badcase
```

## 5. 裸 Engine TTFT 分布 Benchmark

如果要测“裸引擎更准确一点的 TTFT”，使用统一 serving 工具的 TTFT 分布模式：

```bash
mamba run -n qwen3-tts python tools/validation/serving_endpoints.py \
  --targets engine-grpc \
  --skip-single --skip-streaming --skip-custom-instruct \
  --skip-concurrent --skip-long --skip-badcase \
  --ttft-warmup 3 \
  --ttft-samples 30 \
  --ttft-text "今天天气真好。"
```

输出包含：

- `mean_ms`
- `variance_ms2`（样本方差）
- `population_variance_ms2`
- `stdev_ms`
- `coefficient_of_variation`
- `min/p50/p90/p95/max/range`
- 每次采样相对均值的 fluctuation bar
- `--json` 下的结构化明细

TTFT 这里定义为客户端发出 standalone engine `SynthesizeOnce` 请求到收到第一个 audio chunk 的 wall-clock 时间。它包含客户端 gRPC、本地调度、prefill/decode/code2wav 到首包产出的整条裸 engine 路径，不包含 Triton orchestrator。

## 6. 指标口径

| 指标 | 含义 |
| --- | --- |
| `first_chunk_ms` / `ttft_ms` | 请求开始到第一个可播放 audio chunk 到达客户端的 wall-clock 时间。 |
| `total_ms` | 请求开始到终止事件/最终响应完成。 |
| `chunks` | 收到的音频 chunk 数。 |
| `samples` / `duration_sec` | 生成音频采样点数和换算时长。 |
| `rtf` | wall-clock 总耗时 / 音频时长。 |
| `decode_step_*` | 连续 audio chunk 到达间隔统计，用于观察流式稳定性。 |

benchmark 数字必须带完整条件：硬件、driver、镜像/环境、engine profile、输入文本、warmup、样本数、目标 endpoint、采样参数、失败率。
