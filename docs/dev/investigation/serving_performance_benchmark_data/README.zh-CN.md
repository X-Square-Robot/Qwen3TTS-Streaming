[English](README.md) | **中文**

# 服务性能压测——原始数据

支撑 [`../serving_performance_benchmark.zh-CN.md`](../serving_performance_benchmark.zh-CN.md) 的原始数据。

**数据出处(混合数据集):** engine 侧行(`engine-grpc`/`engine-websocket`)采集于
**2026-07-07**,基于优化后引擎构建(含 `b80cf45`/`2a9578b`/`a8e7274`);`triton-grpc`
行**沿用 2026-07-06 数据集**,其引擎核心早于上述 commit——这些数字高估了当前 Triton
延迟,仅作最近一次实测参考保留。

- `conditions.json` — 硬件/软件/模型/调度器配置、被测引擎包含的优化 commit、数据出处、`prefill_ms` 语义变化说明、实验设计(压测报告必须记录哪些条件见 [`docs/user/benchmark_methodology.zh-CN.md`](../../../user/benchmark_methodology.zh-CN.md))。
- `raw_requests.csv` — 一轮代表性 engine 侧(engine-01,2026-07-07)+ 一轮代表性 Triton 侧(triton-01,2026-07-06)的逐请求数据(17,327 条)。跨轮一致性见 `summary_by_trial.csv`。
- `summary.csv` — 按 (协议, 并发档位) 和 (协议, 连接模式) 分组的 avg/p50/p90/p99/max,汇总**全部 6 轮**(engine 侧 3 轮 2026-07-07 + Triton 侧 3 轮 2026-07-06;共 51,981 条请求,0 失败)。
- `summary_by_trial.csv` — 同样的分组统计按轮次(6 次独立运行)并列展示,证明数字跨轮稳定复现。

## 复现方法

需要一个运行中的 engine(`engine-grpc`/`engine-websocket`)**或** Triton(`triton-grpc`)——在 32 GiB 卡上 `max_batch_size=128` 时两者无法同时运行;测一侧时停掉另一侧(见报告"隔离"行)。部署命令见仓库根目录 [CLAUDE.md](../../../../CLAUDE.md);`qwen3tts` client SDK 需在 `PYTHONPATH`(`client/src`)。

```bash
# engine 侧(Triton 停止):
TARGETS="engine-grpc,engine-websocket" bash tools/validation/run_perf_matrix.sh
# Triton 侧(engine 停止):
TARGETS="triton-grpc" bash tools/validation/run_perf_matrix.sh
# 写入 workspace/perf_matrix/<UTC 时间戳>/*.json

# 拍平成 raw_requests.csv + summary.csv:
python tools/validation/summarize_perf_matrix.py workspace/perf_matrix/<run_id>
```

环境变量覆盖项(`TARGETS`、`LEVELS`、`CONCURRENCY_SAMPLES`、`CONCURRENCY_WARMUP`、`CONN_SAMPLES`、`CONN_WARMUP`、各 endpoint)记录在 `run_perf_matrix.sh` 文件头。本数据集 engine 侧用 `LEVELS=1,8,16,32,64,128 CONCURRENCY_SAMPLES=20 CONCURRENCY_WARMUP=3 CONN_SAMPLES=50 CONN_WARMUP=5` 跑 3 次(Triton 侧为相同采样,档位 1,16,32,64,128)。

## 数据解读

- **两侧的 `prefill_ms` 语义不同。** engine 行(2026-07-07):数值覆盖整个批量准入 pass(见报告"测试条件"的说明框),因此天然随并发增长。Triton 行(2026-07-06):旧的单 session 语义。负载下不要跨侧对比该列。
- 测量行的 `cache_hit` 100% 为 `True`——符合预期,因为所有请求共用同一个 prefix-cache key(固定文本/说话人)。因此所有 `prefill_ms` 都是缓存命中(仅后缀)的 prefill;冷 prefix 的 prefill 约贵 30 倍(见报告"已知局限")。
- `kind=concurrency` 行来自并发扫描;`kind=connection` 行来自单流连接模式隔离测试(`conn_mode` = `reuse`/`cold`)。
- `phase=warmup` 行有记录且**计入汇总统计**(与 2026-07-06 数据集同口径,保证跨数据集对比一致;各档影响 ≤1%)。要剔除时按 `phase=measure` 过滤 `raw_requests.csv`。
