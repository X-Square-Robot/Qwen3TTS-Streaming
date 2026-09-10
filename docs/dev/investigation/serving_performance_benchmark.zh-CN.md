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

## 当前 cursor/TN 刷新：2026-09-10

当前 cursor-enabled artifact 已通过 SDK 矩阵在 `qwen3-engine:25.10` 上完成测量。
本轮是 engine-only：没有运行 Triton，因此下方历史 Triton 行没有被本轮更新。完整原始
数据、条件和汇总见
[`serving_performance_benchmark_data/refresh_20260910/`](serving_performance_benchmark_data/refresh_20260910/)。
修正后的 run id 为 `20260910T120301Z`，SDK WebSocket 连接池的
`max_connections` 显式设为 128。

| 协议 | 并发 | TTFT 均值 | decode step 均值 | 最大观测 batch |
| --- | ---: | ---: | ---: | ---: |
| engine-grpc | 1 | 24.593 ms | 13.243 ms | 1 |
| engine-grpc | 8 | 96.829 ms | 15.682 ms | 8 |
| engine-grpc | 16 | 175.358 ms | 18.151 ms | 16 |
| engine-grpc | 32 | 327.079 ms | 24.068 ms | 32 |
| engine-grpc | 64 | 640.013 ms | 37.106 ms | 64 |
| engine-grpc | 128 | 1,273.570 ms | 72.465 ms | 128 |
| engine-websocket | 1 | 23.290 ms | 13.230 ms | 1 |
| engine-websocket | 8 | 93.928 ms | 15.243 ms | 8 |
| engine-websocket | 16 | 150.909 ms | 17.464 ms | 16 |
| engine-websocket | 32 | 280.803 ms | 22.614 ms | 32 |
| engine-websocket | 64 | 547.713 ms | 34.218 ms | 64 |
| engine-websocket | 128 | 1,114.293 ms | 61.747 ms | 128 |

这些是当前矩阵的 SDK 客户端 TTFT，不是 2026-07 报告使用的历史服务端 TTFT 口径。
本轮 11,674 条记录全部成功，固定 workload 的 cache-hit rate 为 100%。SDK WebSocket
连接池已显式提高到 128，因此 c64/c128 测量的是真实并发 lane，两个 transport 都达到
请求的 batch 宽度。当前 c128 延迟仍显著慢于历史基线，在此之前不能发布新的生产吞吐
声明。原生游标正确性与这条性能矩阵分开跟踪。

> **数据集修订(2026-07-08 刷新)。** 下文 engine 侧数字于 2026-07-08 在**全 bf16 重建**
> (repo HEAD `7db42d7`;自 `b80c38d` 基线以来的两个 commit `9ad7cb0`/`7db42d7` 只改构建期
> 精度默认与文档、不动引擎 runtime,故 runtime 与基线一致)上以 3 trial 重新采集,复现
> `ca6ddb9` 数据集、噪声范围内一致。`triton-grpc` 数字**未重测**:沿用 2026-07-06 数据集,
> 其引擎核心不含第三轮优化的全部 commit。由于 Triton BLS 路径包装的是同一个引擎核心,这些沿用
> 数字**高估**了当前 Triton 延迟;仅作最近一次实测参考,下文各表以"(07-06)"标注。

## 测试条件

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090,32 GB,驱动 580.126.20 |
| 镜像 | `qwen3-engine:25.10`(NGC 25.10 base,TensorRT 10.13.3.9),2026-07-08 全 bf16 重建;`qwen3tts-streaming:25.10`(Triton 数据集为 2026-07-06) |
| TensorRT | 10.13.3.9 |
| 模型 | `custom-1.7b`、`custom_voice`、engine_mode=trt,**全 bf16**(backbone/cp/code2wav) |
| 引擎优化 | CP 展开图内保留 KV(decode 计算量 ~3×↓)、CUDA-graph decode 回放、批 KV gather arena 化、突发批量准入 + Phase-1 逐 slot 税消除(`b80cf45`)、c2w KV 入池 + 服务热路径瘦身(`2a9578b`)、VAD 禁用路径旁路(`a8e7274`)、**复盘审计修复批(`cc58c6a`)+ 批量化 `p3_launch` decode 步(`8596996`)+ session open 线程卸载精简(`b80c38d`)**——commit 哈希见 `conditions.json` |
| 引擎 profile | **`max_batch_size=128`**、`max_input_len=128`、`max_seq_len=512`——由 export-aware 选型自动选出(本卡原始容量 ~141 路);**batch profile 是测试变量,不是硬件天花板**(见下文) |
| 调度器 | `max_sessions=128`、`max_queue_size=256`,MLFQ 开启 |
| 请求 | 每个请求固定文本/说话人(`Serena`)/`task_type=custom_voice`/`language=auto`——隔离协议/并发效应,同时把全部测量轮次钉在 **~100% prefix-cache 命中**状态(已验证:测量行 `cache_hit=True` 占 100%);下文所有 `prefill_ms` 都是缓存命中(仅后缀)的 prefill |
| 隔离 | **每次只跑一个服务**:测 engine 时停 Triton,测 Triton 时停 engine(batch=128 下引擎单独就占 ~29 GiB,这张卡上两者无法共存) |
| Warmup | 并发扫描 3 轮 / 连接隔离 5 轮;有记录,并与 2026-07-06 数据集同口径地计入汇总(各档影响 ≤1%) |
| 轮次 | engine 侧 3 轮在 2026-07-08 全 bf16 重建上重跑(档位 1–128 含 8)+ Triton 侧 3 轮沿用 2026-07-06(跨轮一致性见 `summary_by_trial.csv`) |
| 请求总数 | 51,981(engine 侧新采 35,022 + Triton 侧沿用 16,959),**0 失败** |
| 幻觉门禁 | 采集前在本构建上确定性探针重跑 **0/100**(时长 min 9.36s / 中位 10.08s / max 10.88s) |
| 压测工具 | [`tools/validation/perf_matrix_sdk.py`](../../../tools/validation/perf_matrix_sdk.py),基于 `qwen3tts` client SDK |

> **注记。** 本数据集就是**出厂默认:全 bf16**（backbone/cp/code2wav 均 bf16）。`code2wav=fp16` 变体于 2026-07-08 评估过——单流 decode 更快（c1 −21%），但 **b128 decode step 反而 +8%（交叉 ~c32）**,因为 c2w 流式状态每步都要做随 batch 放大的 bf16↔fp16 reformat。因此它是低并发 opt-in，**不是**默认;下方数字仍是 batch-128 的参考基准（见 [engine_overview §1.5](../architecture/engine_overview.zh-CN.md)）。

### ⚠️ 本数据集中 `prefill_ms` 语义已变

自批量突发准入(`b80cf45`)起,`server_prefill_started/completed` 括起的是**整个批量准入
pass**——slot 分配 + 批量 prefix-cache 恢复 + 该 pass 内*所有* session 的批量后缀
embed——而不再是单个 session 自己的后缀计算。因此突发负载下每 session 的 `prefill_ms`
会读出几十 ms(128 并发时均值 60.4ms),尽管单 session 的 GPU 工作量没变、总准入耗时
反而*下降*了。并发 1 时一个 pass 只含一个 session,数值与旧语义一致(~1ms)。沿用的
Triton 行早于该变更,仍是旧的单 session 语义——**负载下两个数据集的 `prefill_ms`
不可互相对比。**

## 测量前修复的问题

搭建本压测过程中发现四个服务端 bug,全部修复并落库后才采集 2026-07-06 数据集(commit `a41f35d`、`d5741b6`、`3758ad7`):

1. **`engine_queue_wait_ms` 是死代码**——计时累加器查找用了一个从不携带 `session_config` 的请求对象,该指标自加入起对任何协议都没被真正写入过。
2. **Triton 从来没算过计时指标**——BLS 编排器不像两个原生网关那样创建 `ServerTimingAccumulator`/`OutputPipeline`,`queue_wait`/`prefill`/`cache_hit`/`total_latency` 对 Triton 会话根本没被计算过。修复的连带变更:Triton 终止事件类型从 `"end"` 统一为 `"done"`。
3. **`cache_hit` 永远上报 `False`**——字符串字面量错配(检查了从未被赋值的 `"prefix_cache_hit"`);缓存快路径本身一直正常。本数据集在修复之后采集:全部测量行 `cache_hit_rate` 为 1.0,与固定 cache key 的负载预期一致。
4. **export-aware 引擎选型从未生效**——helper 路径失效导致每次构建都静默回退到过度保守的显存分档表(把这张 32 GiB 卡压到 batch=32);重新校准后的估算器现在给本卡选型 batch=128。

## 连接建立耗时

单流,每模式 50 个有效轮次,汇总 3 轮;`cold` = 每轮全新 `TTSClient.connect()`,`reuse` = 一个 client 贯穿全部轮次:

| 传输方式 | cold TTFT 均值 | reuse TTFT 均值 | Δ(连接开销) |
|---|---|---|---|
| engine-grpc | 34.4 ms | 16.5 ms | **~17.9 ms** |
| engine-websocket | 15.7 ms | 15.7 ms | ~0(噪声) |
| triton-grpc(07-06) | 21.4 ms | 21.7 ms | ~0(噪声) |

只有 `engine-grpc` 有可测量的连接建立开销(本轮 ~18ms,gRPC channel + HTTP/2 握手;该项对系统链路状态较敏感,跨轮 33–36ms 稳定),这也是 client SDK 默认跨 session 保持 gRPC channel 常驻的原因。WebSocket 握手和 Triton 的 channel 建立在本机链路上都在噪声水平;跨真实网络时以上开销都随 RTT 缩放。

## 单请求生命周期拆解

单流、warm engine、prefix-cache 命中——标准低负载画像:

| 阶段 | 数值 |
|---|---|
| `queue_wait_ms`(软件层 inbox 排队) | 0.6 ms 均值 |
| `prefill_ms`(prefix-cache 命中,仅后缀;单 session 准入 pass) | 0.9 ms 均值 |
| decode step(平均间隔) | 12.6 ms → 每路 RTF ~0.16 |
| **服务端 TTFT**(session 创建 → 首帧原始音频;与历史"13ms"同口径) | **14.9 ± 0.3 ms**(min 14.4,p50 14.8,p99 15.9,n=50,2026-07-08 测量;与 07-07 的 avg 14.96 ± 0.28 持平) |
| 客户端 TTFT(本机,复用连接) | engine-grpc p50 16.1 ms / engine-websocket p50 15.6 ms(各 n=165) |
| total(完整 ~10s 语句) | ~1.15 s |

冷启动、缓存未命中的 prefill 远贵于这里的 0.9ms(抽查实测 ~30ms);见"已知局限"。

## 三协议对比

各协议 × 并发档位的 TTFT 均值(ms),各汇总 3 轮:

| 并发 | engine-grpc | engine-websocket | triton-grpc(07-06) |
|---|---|---|---|
| 1 | 17.1 | 15.6 | 22.4 |
| 8 | 41.6 | **34.4** | — |
| 16 | 55.3 | **49.9** | 71.3 |
| 32 | 88.6 | **76.0** | 112.6 |
| 64 | 142.4 | **128.6** | 185.7 |
| 128 | 275.2 | **241.6** | 309.3 |

`engine-websocket` 在所有档位都最快;128 路时对 gRPC 的领先在均值上约 12%。尾部差距更大(128 路 p99:grpc 494ms vs ws 341ms)。Triton 列为沿用的 2026-07-06 数据集(优化前引擎核心;其 128 路 p99 为 672ms)——不可直接对比,仅作最近实测参考。三者共享引擎核心,扩展*形态*一致;差异来自传输/网关开销。

## 并发扩展性:时间究竟花在哪

`engine-grpc`,汇总均值(3 轮 × 每档 20 轮):

| 并发 | TTFT | `queue_wait_ms` | decode step(均值) | 每路 RTF | `batch_size_seen` |
|---|---|---|---|---|---|
| 1 | 17.1 ms | 0.86 ms | 12.6 ms | 0.16 | 1 |
| 8 | 41.6 ms | 5.6 ms | 13.9 ms | 0.17 | 8 |
| 16 | 55.3 ms | 10.9 ms | 15.1 ms | 0.19 | 16 |
| 32 | 88.6 ms | 25.9 ms | 17.7 ms | 0.22 | 32 |
| 64 | 142.4 ms | 28.0 ms | 24.5 ms | 0.31 | 64 |
| 128 | 275.2 ms | 61.1 ms | 42.1 ms | **0.53** | **128** |

(每个 decode step 产出 80ms 音频;每路 RTF = step 耗时 / 80ms)

三个观察:

1. **128 路真正同批解码**——`batch_size_seen` 到达 128,TTFT 平滑增长,无悬崖。实测佐证:128 路突发期间轮询 `/health`,`active_sessions=128` 一次性全部接纳。
2. **满宽解码现在有了真实余量。** 2026-07-06 数据集在 128 宽时测得 RTF 0.84("仍可实时,余量 ~16%,接近饱和");2026-07-07 那轮降到 0.59。第三轮批量化 `p3_launch`(`8596996`)后,128 宽的 step 为 42.1ms/80ms 音频——**RTF 0.53,余量 ~47%**。本 GPU/模型的容量表述更新为:**128 路并发实时流,且有余量**;128 路的约束项已是编译进 plan 的 batch profile(和 ~141 路的显存容量),不再是解码吞吐。
3. **排队仍是次要项。** 128 路下 `queue_wait_ms` 均值也只有 ≤61ms(对比 profile 卡在 32 时数秒级的槽位等待——见下节)。

本数据集中 `prefill_ms` 随并发增长(均值从档位 1 的 0.8ms 到 128 的 60.4ms)——这是**语义变化**(数值现在覆盖整个批量准入 pass,见"测试条件"),不是 prefill 回退;单 session 的后缀 prefill 工作量没变,总准入斜坡反而更快了(下节)。

## 突发 TTFT 花在哪:准入已批量化,到达散布成为主项

2026-07-06 数据集显示 128 突发 TTFT(均值 ~334ms)的主项是**爬坡串行化**:session 在不断变宽的 batch 的 decode 步之间逐个 prefill(最后一个 ~376ms 才准入;调度间隙均值 78.7ms;每 session 首步等待均值 89.2ms)。`b80cf45`(批量准入)消除的正是这个机制。用同一工具重跑逐 session 生命周期拆解(`workspace/ttft_ramp_decompose.py`,128/128 全采集,时间戳取自 done 事件的服务端 epoch 字段;两次运行,客户端 TTFT 均值 285.8/287.9ms;**2026-07-07 在第二轮构建上测量**——突发 TTFT 总量此后稳定在 grpc ~275ms / ws ~242ms,但下表的分阶段机制不变):

| 阶段 | avg | p50 | p90 | max |
|---|---|---|---|---|
| 请求接收 → session 创建 | ~0 | 0 | 0 | 0 |
| 创建 → 文本入队 | 13.2 ms | 7.0 | 25.3 | 27 |
| 入队 → 出队(`queue_wait`) | 80.3 ms | 87.5 | 150.3 | 163 |
| 出队 → prefill 开始(调度间隙) | **13.1 ms**(原 78.7) | 16.0 | 19.0 | 54 |
| prefill(批量准入 pass,新语义) | 51.6 ms | 23.0 | 107.0 | 108 |
| prefill 完成 → 首帧音频(等首个 decode 步) | **58.9 ms**(原 89.2) | 53.0 | 74.0 | 75 |
| 首帧 → 客户端收到 | 5.8 ms | 5.0 | 6.2 | 101.1 |
| 客户端 TTFT(本次运行) | 285.8 ms | 302.7 | 316.5 | 317.9 |

准入现在以少数几个宽 pass 落地,而不是 128 次串行:首个 session 在突发开始后 ~5ms 进入 prefill,*最后*一个准入 pass 在 ~251ms 启动(串行时代为 ~376ms),单个 pass 可用 ~35ms 的引擎时间准入 ~120 个 session。突发 TTFT 剩下的部分是:(a) **客户端/网关侧的到达散布**——"同时"发起的 128 突发,其请求在共享 channel 上要 ~163ms 才全部到达服务端,晚到者机械地继承这个偏移;(b) 引擎循环迭代之间的排队消化节奏(`queue_wait` p50 87.5ms);(c) 一个批量 pass + 首步等待。GPU 计算仍然不是瓶颈。

**剩余优化线索**:到达/入队散布在客户端与网关侧(开发期实测多 channel 客户端的 TTFT 明显更低);引擎侧则是爬坡中段更激进地排空 inbox、压缩首步等待——这是幸存的两个最大项。

## Batch profile 是变量(不是天花板)

TRT 引擎的 `max_batch_size` **编译进 plan 文件**,决定一次前向传播最多带多少会话。超出的会话会被接纳(`max_sessions`)但要等空闲执行槽位——等的是前面会话的*整段话*,不是一步。

同一张卡上早前采集的 **batch=32** 数据集(优化前引擎代码、cp=fp32、双服务共存——条件差异不止 profile 一项,所以只作机制演示、不作受控对比)完整展示了那种失效模式:128 路并发时会话按 32 一波执行,`active_sessions` 阶梯式 128→96→64→32 下降,decode step 在上限处平台化,**TTFT 均值 ~6.5s**——比本报告同并发的 ~275ms 差 ~24 倍。而 batch=32 这个 profile 本身是选型 bug 的产物(上文修复 #4),不是硬件极限。

选型指引现在内置于 `scripts/python/suggest_engine_profile.py`(export-aware;本模型每路 ~160 MiB + 固定 ~5 GiB):32 GiB 卡容纳 batch=128;~19 GiB 容纳 64;~13.5 GiB 容纳 32。按预期峰值并发选 profile——batch 宽度超卖正是 TTFT 悬崖的来源。

## 优化效果(开发期参考)

同一张卡 batch=128 下的三轮优化(原始数据:"第 1 轮"对应 2026-07-06 数据集,"第 2 轮"对应 2026-07-07 数据集,"第 3 轮"对应本数据集;优化前原始数据未保留,仅汇总数字):

| | 优化前 | 第 1 轮(07-06:CP-KV + CUDA-graph + KV arena) | 第 2 轮(07-07:批量准入 + Phase-1 税 + 热路径 + VAD 旁路) | 第 3 轮(07-08:审计修复批 + 批量化 p3_launch) |
|---|---|---|---|---|
| decode step @128 | 119.8 ms(RTF 1.5——**不能**实时) | 67.5 ms(RTF 0.84) | 47.4 ms(RTF 0.59) | **42.1 ms(RTF 0.53)** |
| decode step @64 | 63.6 ms | 36.8 ms | 27.1 ms | **24.5 ms** |
| decode step @1 | 11.3 ms | 12.7 ms | 12.8 ms | 12.6 ms(较优化前 +1.3ms;CP-KV/CUDA-graph 路径的每步固定开销) |
| TTFT 均值 @128(engine-grpc) | ~371 ms | ~334 ms | ~275 ms | **~275 ms** |
| 服务端 total latency p50 @128 | — | 6333 ms | 4453 ms(−30%) | **3988 ms**(较第 2 轮 −11%,较第 1 轮 −37%) |

第 1 轮用单流每步 ~1.4ms 的代价换来满 batch 宽度下 ~44% 的每步耗时下降——把 128 路服务从低于实时拉回实时。第 2 轮在满宽下再降 ~30% 每步耗时(Phase-1 逐 slot CPU 税 + c2w 入池)、消除了串行准入斜坡(TTFT −18%),并瘦身了服务热路径(异步日志、orjson、gRPC chunk 合并、VAD 路径旁路)。第 2 轮的纯数据搬运改动经逐字节验证输出音频不变(8 个确定性种子,sha256)。第 3 轮为复盘审计修复批(`cc58c6a`)+ 位级一致的批量化 `p3_launch`(`8596996`),满宽下每 decode 步再降 ~11%,`server_total` 同幅传导;相对 2026-07-06 数据集累计为 128 宽下 **TTFT −18%、decode step −38%、server total −37%**。每轮采集数据前幻觉探针均保持 0/100。

## 已知局限 / 后续引擎优化线索

- **固定文本/说话人——所有 prefill 数字都是缓存命中的 prefill。** 冷 prefix(缓存未命中)的 prefill 抽查实测 ~30ms vs ~1ms;缓存未命中在并发负载下的表现未覆盖。文本长度、说话人多样性、缓存未命中扫描是自然的后续方向。
- **Triton 数字已过期(2026-07-06)。** Triton BLS 路径包装同一个引擎核心,引擎侧收益(`b80cf45` 至 `8596996`)会传导过去,但未重新压测;其网关侧在新准入模式下的表现未测量。
- **突发到达是最坏情况。** 并发测试同时发起全部 N 路;真实世界错峰到达时,同样稳态并发下单请求 TTFT 会更低。此外,实测突发 TTFT 还包含共享 channel 上 ~163ms 的客户端侧到达散布(见爬坡拆解)。
- **仅本机链路。** 网络 RTT 直接叠加到连接开销和 TTFT 上;~18ms 的 gRPC 连接差值随 RTT 缩放。
- **批处理优化的单流回退。** batch=1 时每步 +1.3ms(11.3 → 12.6ms,第 2、3 轮间持平)的代价是真实的;未来可以做自适应路径,在 batch 宽度低于阈值时跳过 arena/graph 机制。
- **`serving_endpoints.py` 仍未使用 `qwen3tts` client SDK**——迁移作为独立后续工作跟踪;本压测工具(`perf_matrix_sdk.py`)是基于 SDK 的。

## 复现方法

```bash
bash tools/validation/run_perf_matrix.sh
python tools/validation/summarize_perf_matrix.py workspace/perf_matrix/<run_id>
```

环境变量覆盖项和完整的原始/汇总 CSV 见
[`serving_performance_benchmark_data/README.zh-CN.md`](serving_performance_benchmark_data/README.zh-CN.md)。
