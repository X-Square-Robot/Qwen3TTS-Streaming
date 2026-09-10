[English](benchmark_methodology.md) | **中文**

# Benchmark 方法

本文档定义产品 Demo、README 和 release trace 使用的 benchmark 口径。所有公开性能数字都必须带上这些条件。

统一的 `/demo/#/lab` 页面是交互式浏览器实验工具：基础 LLM PK 和并发实验通过实例
公共 `/v1/realtime` 执行，并报告浏览器侧耗时。可选 `demo_api` 提供独立的服务端
Triton/trace 接口。发布结果时必须记录来源，不能把浏览器实验或 `demo_api` 测量值直接
与下方正式 engine benchmark 表比较。

## 指标定义

| 指标 | 含义 |
| --- | --- |
| TTFT | Time To First Token/Frame。服务端生成第一个可发送音频 chunk 的时间。 |
| TTFB | Time To First Byte。客户端收到第一段音频字节的时间。 |
| first playable | 客户端可以开始播放的时间。离线模式通常等于整段音频生成完成时间。 |
| first audible | 播放器真正听到非静音音频的时间，可能晚于 first playable。 |
| total latency | 完成全部音频生成或流式结束的时间。 |
| audio duration | 输出音频时长，用来计算实时率。 |

不同 backend 的指标含义不能混用：

- Official PyTorch Offline：主要看 `first_playable_ms`，因为整段音频完成后才能播放。
- Official PyTorch Streaming：主要看服务端首包和客户端首字节。
- Bare Engine Streaming：主要看 standalone engine WebSocket/gRPC 首包。
- Triton TRT Streaming：主要看 TensorRT/Triton streaming 首包。

## 必填测试条件

公开 benchmark 必须记录：

- GPU 型号、显存、机器规格。
- NVIDIA driver、CUDA、TensorRT、NGC image tag。
- 模型 variant，例如 `custom-1.7b`。
- engine profile：`max_batch_size`、`max_input_len`、`max_seq_len`、dtype、I/O dtype。
- 部署模式：standalone、engine-docker、Triton。
- 输入文本、语言、speaker、cache mode。
- warmup 策略：是否排除 warmup。
- 并发数、请求数、失败数。
- SDK WebSocket 测试的 `max_connections` 必须不小于最大并发档位；否则客户端连接池会
  在请求到达引擎前串行化 lane。
- 测量位置：服务端、adapter、客户端。
- 实验来源：内置 `/demo/#/lab` 的公共 Realtime、可选 `demo_api`，或直接 engine/Triton 客户端。
- 是否使用 fixture trace。

## 流式 TN 与原生游标的记录要求

流式 TN/文本进度属于正确性与协议能力，不应只用 TTFT 或吞吐量推断。每次发布新
benchmark 时还应记录：

- `/v1/capabilities` 中的 `loaded_model_type`、`native_cursor.graph_enabled`、
  `native_cursor.progress_available` 和 `supported_progress_modes`；
- `native_cursor` 使用的 model/cursor-head/label-vocab/TN rules fingerprint，以及
  `right_context`、`max_labels`、codebook 数和 frame 时长；
- 实际 progress route：`native`、`ema` 或 `disabled`，以及事件中的
  `progress_basis`；
- TN 回放或合同测试的结果。`raw_codepoint_end`、
  `normalized_codepoint_end` 是单调整数 high-water，不应被展示插值替代；
- 是否验证了 `WAIT_TEXT`、EOS、tail rewrite、并发 slot 隔离和无游标头 fallback。

性能矩阵与文本进度正确性必须分开报告。不能把 standard TRT/EMA、cursor-enabled TRT/
native，或不同 artifact 的 Triton 数字合并成一个平均值。

## 当前矩阵入口

性能矩阵使用真实 SDK 客户端，要求目标服务已经启动。必须在项目虚拟环境中运行：

```bash
mamba run -n qwen3-tts bash -c \
  'TARGETS="engine-grpc,engine-websocket" LEVELS="1,8,16,32,64,128" \
   CONCURRENCY_SAMPLES=20 CONCURRENCY_WARMUP=3 CONN_SAMPLES=50 CONN_WARMUP=5 \
   MAX_CONNECTIONS=128 \
   bash tools/validation/run_perf_matrix.sh'
mamba run -n qwen3-tts python tools/validation/summarize_perf_matrix.py \
  workspace/perf_matrix/<run_id>
```

Triton 只有在对应 Triton 服务正在运行时才加入 `TARGETS`；没有运行 Triton 时不要用
空结果或历史数据冒充本次重测。原始 JSON、环境快照和汇总 CSV 应按 run id 保存，并在
报告中注明服务版本、artifact 和是否为 engine-only refresh。

## 单路 TTFT 的口径

单路 TTFT 只能描述为"特定条件下的实测分布"，不能描述为默认性能或稳定承诺。当前实测参考值（2026-07-08，RTX 5090，全 bf16 `custom-1.7b`，batch=128 profile）为 **server TTFT 14.9 ± 0.3ms（min 14.4，p99 15.9，n=50）**——完整拆解与原始数据见[服务性能压测报告](../dev/investigation/serving_performance_benchmark.zh-CN.md)。历史上的"13ms"是旧版引擎构建的最低观测值，不应再引用。任何此类数字都需要同时满足：

- cache 命中。
- engine 已 warm。
- 单路请求。
- 固定输入文本和 speaker。
- 固定 engine profile。
- 本地部署或近似本地链路。

建议公开写法：

```text
在指定硬件、warm engine、cache hit、单路请求、custom-1.7b profile 下，实测 server TTFT 为 14.9 ± 0.3ms（min 14.4ms，n=50）。
```

不建议写法：

```text
TTFT 15ms。
```

## 128 路并发口径

并发 benchmark 至少报告：

- concurrency。
- avg / p50 / p90 / p99 / max TTFT。
- failed streams。
- 总耗时。
- throughput audio sec/sec。
- 是否启用 live concurrency。

内置“实验”页的浏览器并发不是 128 路正式 benchmark，而是面向当前实例的交互式对照。
如需服务端并发任务，请启动 `demo_api` 并明确记录来源。Compose 的 `demo` profile 默认将
live concurrency 设为 `0`；直接运行模块时默认是 `1`，因此应显式设置该变量：

```bash
QWEN_DEMO_ENABLE_LIVE_CONCURRENCY=1 python -m demo_api --port 7860
```

## 采集 release trace

示例命令：

```bash
python scripts/demo/collect_demo_traces.py \
  --live-triton \
  --triton-grpc localhost:8001 \
  --output workspace/demo_traces/race_default.json
```

采集完成后检查：

- `source` 是否是 `live_triton` / `live_engine` / `live_official_pytorch`。
- `benchmark_conditions` 是否完整。
- `warnings` 是否为空或已解释。
- 音频是否可播放，且没有明显错读、重复、幻觉。

## 发布报告模板

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
