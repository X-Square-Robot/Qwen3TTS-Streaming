[English](serving_performance_benchmark.md) | **中文**

# 服务性能压测:连接/排队/推理耗时拆解,跨协议与并发对比

## 范围

本报告在开源前回答四个问题:

1. 连接建立要多久(WebSocket vs gRPC vs Triton)?
2. 引擎在排队和真正推理上各花多少时间?
3. 服务路径的选择(`engine-grpc` / `engine-websocket` / `triton-grpc`)对延迟有什么影响?
4. 并发从 1 涨到 128 路时,上述各项如何变化——真正的瓶颈是什么?

支撑本报告每一个数字的逐请求原始数据和汇总 CSV 都在
[`serving_performance_benchmark_data/`](serving_performance_benchmark_data/)
(复现方法见其 `README.zh-CN.md`)。指标名称遵循
[`docs/dev/operations/timing_metrics.zh-CN.md`](../operations/timing_metrics.zh-CN.md);
报告格式遵循 [`docs/user/benchmark_methodology.zh-CN.md`](../../user/benchmark_methodology.zh-CN.md)。

## 测试条件

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090,32 GB,驱动 580.126.20 |
| 镜像 | `qwen3-engine:25.10`、`qwen3tts-streaming:25.10`(NGC tag 25.10),2026-07-06 重建 |
| TensorRT | 10.13.3.9 |
| 模型 | `custom-1.7b`、`custom_voice`、engine_mode=trt,**全 bf16**(backbone/cp/code2wav) |
| 引擎优化 | CP 展开图内保留 KV(decode 计算量 ~3×↓)、CUDA-graph decode 回放、批 KV gather arena 化——commit 哈希见 `conditions.json` |
| 引擎 profile | **`max_batch_size=128`**、`max_input_len=128`、`max_seq_len=512`——由 export-aware 选型自动选出(本卡原始容量 ~141 路);**batch profile 是测试变量,不是硬件天花板**(见下文) |
| 调度器 | `max_sessions=128`、`max_queue_size=256`,MLFQ 开启 |
| 请求 | 每个请求固定文本/说话人(`Serena`)/`task_type=custom_voice`/`language=auto`——隔离协议/并发效应,同时把全部测量轮次钉在 **~100% prefix-cache 命中**状态(已验证:测量行 `cache_hit=True` 占 100%);下文所有 `prefill_ms` 都是缓存命中(仅后缀)的 prefill |
| 隔离 | **每次只跑一个服务**:测 engine 时停 Triton,测 Triton 时停 engine(batch=128 下引擎单独就占 ~29 GiB,这张卡上两者无法共存) |
| Warmup | 并发扫描 3 轮 / 连接隔离 5 轮,不计入统计 |
| 轮次 | 每个服务侧 3 轮独立完整重跑(可复现性验证) |
| 请求总数 | 50,877,**0 失败** |
| 压测工具 | [`tools/validation/perf_matrix_sdk.py`](../../../tools/validation/perf_matrix_sdk.py),基于 `qwen3tts` client SDK |

## 测量前修复的问题

搭建本压测过程中发现四个服务端 bug,全部修复并落库后才采集本数据集(commit `a41f35d`、`d5741b6`、`3758ad7`):

1. **`engine_queue_wait_ms` 是死代码**——计时累加器查找用了一个从不携带 `session_config` 的请求对象,该指标自加入起对任何协议都没被真正写入过。
2. **Triton 从来没算过计时指标**——BLS 编排器不像两个原生网关那样创建 `ServerTimingAccumulator`/`OutputPipeline`,`queue_wait`/`prefill`/`cache_hit`/`total_latency` 对 Triton 会话根本没被计算过。修复的连带变更:Triton 终止事件类型从 `"end"` 统一为 `"done"`。
3. **`cache_hit` 永远上报 `False`**——字符串字面量错配(检查了从未被赋值的 `"prefix_cache_hit"`);缓存快路径本身一直正常。本数据集在修复之后采集:全部测量行 `cache_hit_rate` 为 1.0,与固定 cache key 的负载预期一致。
4. **export-aware 引擎选型从未生效**——helper 路径失效导致每次构建都静默回退到过度保守的显存分档表(把这张 32 GiB 卡压到 batch=32);重新校准后的估算器现在给本卡选型 batch=128。

## 连接建立耗时

单流,每模式 50 个有效轮次,汇总 3 轮;`cold` = 每轮全新 `TTSClient.connect()`,`reuse` = 一个 client 贯穿全部轮次:

| 传输方式 | cold TTFT 均值 | reuse TTFT 均值 | Δ(连接开销) |
|---|---|---|---|
| engine-grpc | 26.5 ms | 16.2 ms | **~10.3 ms** |
| engine-websocket | 16.7 ms | 16.6 ms | ~0.1 ms(噪声) |
| triton-grpc | 21.4 ms | 21.7 ms | ~0(噪声) |

只有 `engine-grpc` 有可测量的连接建立开销(~10ms,gRPC channel + HTTP/2 握手),这也是 client SDK 默认跨 session 保持 gRPC channel 常驻的原因。WebSocket 握手和 Triton 的 channel 建立在本机链路上都在噪声水平;跨真实网络时以上开销都随 RTT 缩放。

## 单请求生命周期拆解

单流、warm engine、prefix-cache 命中——标准低负载画像:

| 阶段 | 数值 |
|---|---|
| `queue_wait_ms`(软件层 inbox 排队) | 0.43 ms 均值 |
| `prefill_ms`(prefix-cache 命中,仅后缀) | 0.97 ms 均值 |
| decode step(平均间隔) | 12.7 ms → 每路 RTF ~0.16 |
| **服务端 TTFT**(session 创建 → 首帧原始音频;与历史"13ms"同口径) | **14.9 ± 0.2 ms**(min 14.5,p50 14.9,p99 15.3,n=50) |
| 客户端 TTFT(本机,复用连接,engine-grpc) | 16.0 ± 0.3 ms(min 15.6,p99 16.9,n=150) |
| total(完整 ~10s 语句) | ~1.28 s |

冷启动、缓存未命中的 prefill 远贵于这里的 0.97ms(抽查实测 ~30ms);见"已知局限"。

## 三协议对比

各协议 × 并发档位的 TTFT 均值(ms),各汇总 3 轮:

| 并发 | engine-grpc | engine-websocket | triton-grpc |
|---|---|---|---|
| 1 | 16.7 | 16.5 | 22.4 |
| 16 | 62.5 | **51.1** | 71.3 |
| 32 | 97.8 | **76.4** | 112.6 |
| 64 | 174.8 | **145.0** | 185.7 |
| 128 | 333.8 | **265.5** | 309.3 |

`engine-websocket` 在负载下稳定最快(128 路时 TTFT 比 engine-grpc 低 ~20%)。`triton-grpc` 高并发的均值介于两者之间,但尾部最重(128 路 p99:672ms,对比 grpc/ws 的 476/389ms)。三者共享引擎核心,扩展*形态*一致;差异来自传输/网关开销。

## 并发扩展性:时间究竟花在哪

`engine-grpc`,汇总均值(3 轮 × 每档 20 轮):

| 并发 | TTFT | `queue_wait_ms` | decode step(均值) | 每路 RTF | `batch_size_seen` |
|---|---|---|---|---|---|
| 1 | 16.7 ms | 0.43 ms | 12.7 ms | 0.16 | 1 |
| 16 | 62.5 ms | 12.4 ms | 17.7 ms | 0.22 | 16 |
| 32 | 97.8 ms | 21.7 ms | 23.6 ms | 0.29 | 32 |
| 64 | 174.8 ms | 34.4 ms | 36.8 ms | 0.46 | 64 |
| 128 | 333.8 ms | 71.9 ms | 67.5 ms | **0.84** | **128** |

(每个 decode step 产出 80ms 音频;每路 RTF = step 耗时 / 80ms)

三个观察:

1. **128 路真正同批解码**——`batch_size_seen` 到达 128,TTFT 平滑增长(并发每翻倍约 ~2×),无悬崖。实测佐证:128 路突发期间轮询 `/health`,`active_sessions=128` 一次性全部接纳。
2. **这张卡的真实天花板是每路 RTF,不是 TTFT。** decode step 耗时随 batch 宽度增长;128 宽时达到 67.5ms/80ms 音频——RTF 0.84,仍可实时但余量只剩 ~16%。这是本 GPU/模型的诚实容量表述:**128 路并发实时流,接近饱和。**
3. **batch profile 开满时,排队是次要项。** 128 路下 `queue_wait_ms` 也只有 ≤72ms(对比 profile 卡在 32 时数秒级的槽位等待——见下节)。

`prefill_ms` 均值在每档都是 1.0–1.5ms——但记住这是缓存命中、仅后缀的 prefill(见"测试条件")。

## 突发 TTFT 花在哪:爬坡串行化,不是 GPU

128 并发下数字看似矛盾:TTFT ~334ms,但 `queue_wait` 只有 ~72ms、每路 RTF 0.84(GPU 跟得上)。对一次真实 128 突发做逐 session 生命周期拆解(`workspace/ttft_ramp_decompose.py`,128/128 全采集,时间戳取自 done 事件的服务端 epoch 字段)显示剩余时间的去向:

| 阶段 | avg | p50 | p90 | max |
|---|---|---|---|---|
| 请求接收 → session 创建 | ~0 | 0 | 0 | 0 |
| 创建 → 文本入队 | 19.7 ms | 8.0 | 70.3 | 78 |
| 入队 → 出队(`queue_wait`) | 93.6 ms | 103 | 168 | 181 |
| **出队 → prefill 开始(调度间隙)** | **78.7 ms** | 66.5 | 130 | 200 |
| prefill 计算 | 1.4 ms | 1.0 | 1.0 | 19 |
| **prefill 完成 → 首帧音频(等首个 decode 步)** | **89.2 ms** | 86 | 102 | 183 |
| 首帧 → 客户端收到 | 0.1 ms | 0.1 | 0.5 | 0.6 |
| 客户端 TTFT(本次运行) | 381.7 ms | 417.5 | 422.6 | 423.1 |

机制:引擎循环交替执行*排空 inbox → 调度 prefill → 给整个活跃 batch 跑一个 decode 步*。突发时,prefill 准入(首个 session 6ms 进入,最后一个 376ms;中段速率 ~54 路/100ms)被**不断变宽的** batch 的 decode 步隔开——步耗时随宽度从 ~13ms 涨到 ~67ms,后到 session 的 prefill 排在这些越来越贵的步后面;即便 prefill 完成,还要再等约一步(中位 ~86ms)才轮到自己的首个 decode 槽。每次准入的真实 GPU 计算量微不足道(prefill 1.4ms)。

因此 RTF 0.84 与 334ms TTFT 描述的是不同阶段:RTF 是稳态解码吞吐;突发 TTFT 是 0→128 路灌进单一准入循环的瞬态成本。**优化线索**:(a) 批量 prefill——把等待中的 session 合并成一次 GPU 调用准入,而不是在 decode 步之间逐个做,可把 ~370ms 的准入斜坡压向 ~100ms 的理论下界(批量 prefill + 一个满宽步);(b) 爬坡感知调度——突发期先集中做 prefill,少跑那些之后反正要重复的部分宽度 decode 步。

## Batch profile 是变量(不是天花板)

TRT 引擎的 `max_batch_size` **编译进 plan 文件**,决定一次前向传播最多带多少会话。超出的会话会被接纳(`max_sessions`)但要等空闲执行槽位——等的是前面会话的*整段话*,不是一步。

同一张卡上早前采集的 **batch=32** 数据集(优化前引擎代码、cp=fp32、双服务共存——条件差异不止 profile 一项,所以只作机制演示、不作受控对比)完整展示了那种失效模式:128 路并发时会话按 32 一波执行,`active_sessions` 阶梯式 128→96→64→32 下降,decode step 在上限处平台化,**TTFT 均值 ~6.5s**——比本报告同并发的 ~334ms 差 ~20 倍。而 batch=32 这个 profile 本身是选型 bug 的产物(上文修复 #4),不是硬件极限。

选型指引现在内置于 `scripts/python/suggest_engine_profile.py`(export-aware;本模型每路 ~160 MiB + 固定 ~5 GiB):32 GiB 卡容纳 batch=128;~19 GiB 容纳 64;~13.5 GiB 容纳 32。按预期峰值并发选 profile——batch 宽度超卖正是 TTFT 悬崖的来源。

## 优化效果(开发期参考)

开发期间在 batch=128 下对 CP-KV + CUDA-graph + arena 优化前后的测量(优化前原始数据未保留,仅汇总数字):

| | 优化前 | 优化后 |
|---|---|---|
| decode step @128 | 119.8 ms(RTF 1.5——**不能**实时) | 67.5 ms(RTF 0.84) |
| decode step @64 | 63.6 ms | 36.8 ms |
| decode step @1 | 11.3 ms | 12.7 ms(+1.4ms;CP-KV/CUDA-graph 路径的每步固定开销) |
| TTFT 均值 @128 | ~371 ms | ~334 ms |

这组优化用单流每步 ~1.4ms 的代价换来满 batch 宽度下 ~44% 的每步耗时下降——正是它把 128 路服务从低于实时拉回到实时。

## 已知局限 / 后续引擎优化线索

- **固定文本/说话人——所有 prefill 数字都是缓存命中的 prefill。** 冷 prefix(缓存未命中)的 prefill 抽查实测 ~30ms vs ~1ms;缓存未命中在并发负载下的表现未覆盖。文本长度、说话人多样性、缓存未命中扫描是自然的后续方向。
- **突发到达是最坏情况。** 并发测试同时发起全部 N 路;真实世界错峰到达时,同样稳态并发下单请求 TTFT 会更低。
- **仅本机链路。** 网络 RTT 直接叠加到连接开销和 TTFT 上;~10ms 的 gRPC 连接差值随 RTT 缩放。
- **批处理优化的单流回退。** batch=1 时每步 +1.4ms 的代价是真实的;未来可以做自适应路径,在 batch 宽度低于阈值时跳过 arena/graph 机制。
- **`serving_endpoints.py` 仍未使用 `qwen3tts` client SDK**——迁移作为独立后续工作跟踪;本压测工具(`perf_matrix_sdk.py`)是基于 SDK 的。

## 复现方法

```bash
bash tools/validation/run_perf_matrix.sh
python tools/validation/summarize_perf_matrix.py workspace/perf_matrix/<run_id>
```

环境变量覆盖项和完整的原始/汇总 CSV 见
[`serving_performance_benchmark_data/README.zh-CN.md`](serving_performance_benchmark_data/README.zh-CN.md)。
