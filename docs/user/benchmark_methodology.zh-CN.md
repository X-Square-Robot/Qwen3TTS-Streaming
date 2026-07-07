[English](benchmark_methodology.md) | **中文**

# Benchmark 方法

本文档定义 WebUI、README 和 release trace 使用的 benchmark 口径。所有公开性能数字都必须带上这些条件。

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
- 测量位置：服务端、adapter、客户端。
- 是否使用 fixture trace。

## 单路 TTFT 的口径

单路 TTFT 只能描述为"特定条件下的实测分布"，不能描述为默认性能或稳定承诺。当前实测参考值（2026-07-07，RTX 5090，全 bf16 `custom-1.7b`，batch=128 profile）为 **server TTFT 14.9 ± 0.3ms（min 14.5，p99 15.7，n=50）**——完整拆解与原始数据见[服务性能压测报告](../dev/investigation/serving_performance_benchmark.zh-CN.md)。历史上的"13ms"是旧版引擎构建的最低观测值，不应再引用。任何此类数字都需要同时满足：

- cache 命中。
- engine 已 warm。
- 单路请求。
- 固定输入文本和 speaker。
- 固定 engine profile。
- 本地部署或近似本地链路。

建议公开写法：

```text
在指定硬件、warm engine、cache hit、单路请求、custom-1.7b profile 下，实测 server TTFT 为 14.9 ± 0.3ms（min 14.5ms，n=50）。
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

WebUI 的 live concurrency 默认关闭：

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
