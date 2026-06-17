# 引擎架构决策：脱离 Triton，自建 TTS 推理引擎

> 日期：2026-04-03  
> 状态：已决策 — 采用路线 1（自建引擎），保留未来 Triton 集成选项

---

## 背景

项目原计划以 Triton Inference Server 为推理框架，通过 Python BLS
编排子模型（TRT/ORT backend）完成 TTS 流式推理。在实际开发中发现
Triton 对有状态自回归模型的价值有限，需要重新评估技术路线。

## 评估的两条路线

### 路线 1：自建推理引擎（已采用）

完全脱离 Triton，自己实现 gRPC 服务、GPU 调度、continuous batching。

| 维度 | 评估 |
|------|------|
| 性能控制 | **最优** — 直接管理 CUDA stream，CPU-GPU 流水线，无框架中间层 |
| 调度精细度 | **最优** — 可实现首包优先、segment 级并行、长文本限流 |
| 可扩展性 | 中 — 需自建模型加载、健康检查、指标等生产组件 |
| 工程量 | 中 — 核心骨架已搭建完成（`engine/`） |
| 可靠性 | 需积累 — 异常恢复、内存泄漏等需要持续打磨 |

### 路线 2：重构 BLS + Triton Dynamic Batching（已否决）

开 64 个 BLS 实例各管一个 session，利用 Triton 的 dynamic batcher
收集 decode 请求统一推理。

| 维度 | 评估 |
|------|------|
| 性能控制 | **差** — KV cache 需 seq_len 维 padding，计算/显存浪费严重 |
| 调度精细度 | **差** — Triton dynamic batcher 按到达时间聚合，无法表达优先级 |
| GIL 瓶颈 | **严重** — 64 个 Python BLS 实例共享 GIL，CPU 准备工作串行化 |
| BLS→TRT 开销 | **不可忽略** — pb_utils + dlpack 路径在 ~4ms 单步预算内占比可观 |
| 首包优先级 | **无法实现** — 长文本并发合成会抢占首 token 紧急请求的 GPU 资源 |

**否决核心原因：** Triton dynamic batching 为无状态请求设计（batch 维
concat），不适用于 KV cache 长度各异的自回归 decode。所有主流 LLM
推理引擎（vLLM、TRT-LLM、SGLang）均自建 continuous batching
而非使用 Triton dynamic batcher。

#### Triton Dynamic Batching 的前提条件

Dynamic batching 生效需要同时满足以下条件，说明了它的适用边界：

1. **模型输入第一维必须是 batch 维度** — Triton 沿 dim-0 拼接请求
2. **`max_batch_size > 0`** — 设为 0 表示模型不支持 batching
3. **`config.pbtxt` 显式开启** — `dynamic_batching { preferred_batch_size: [...] }`
4. **batch 内所有请求的非 batch 维度形状必须一致** — 否则需要 padding

对于自回归 decode，条件 4 天然不满足：每个请求的 KV Cache `seq_len`
维度不同，Triton 不会自动 pad — 它只做 batch 维度的 concat。这意味着
要么所有请求恰好同长（不现实），要么由 BLS 手动 pad 后以 batch=1 的
假象送入（绕过了 dynamic batcher 的意义）。

## Triton Server 的真正价值域

Triton 在以下场景有不可替代的价值：

- **无状态模型**：图像分类、embedding、reranking、推荐 CTR —
  dynamic batching 可将 GPU 利用率从 10% 拉到 90%
- **模型动物园**：同一集群托管数十种异构模型，统一运维
- **企业基础设施**：K8s 集成、模型版本管理、A/B 测试、Prometheus 指标

对于**单模型、有状态、自回归、需要精细调度**的 TTS 场景，Triton
退化为 gRPC 代理 + 进程容器。在 TRT-LLM 的 Triton backend 中，
Triton 也只是一个壳，所有调度逻辑在 TRT-LLM Executor 内部。

## Batching 策略选型

### Dynamic Batching vs Continuous Batching

两者的核心区别在于**调度粒度**：

- **Dynamic Batching**（请求级调度）：多个请求拼成 batch 一起执行，
  **整个 batch 必须全部完成才能返回**。短请求跑完后空等长请求，GPU
  后期利用率下降。适合无状态、单次前向的模型（分类、embedding）。

- **Continuous Batching**（迭代级调度）：以每个 decode step 为调度
  单位，**每步都可以增减请求**。完成的请求立即释放资源，新请求下一步
  即可加入，GPU 始终在做有效计算。

### Continuous Batching 如何处理不同 shape

continuous batching 能工作的核心机制：

1. **Decode 阶段天然对齐** — 每个请求的"当前输入"都是 1 个 token
   （shape `[1, hidden]`），QKV 线性层可以直接 batch，无需 padding。

2. **Paged KV Cache** — KV Cache 按固定大小的 page 分配（类似 OS
   虚拟内存），每个请求通过 page table 索引不连续的物理显存。请求完成
   后 page 立即回收，新请求按需分配，不需要预留 max_len 的连续显存。

3. **特殊 Attention kernel** — 标准 Attention 要求 batch 内 KV 长度
   一致。Continuous batching 使用：
   - **PagedAttention**：kernel 接受 `block_tables[batch, max_pages]`
     和 `context_lens[batch]`，按 page table 到不连续地址取 KV
   - **FlashAttention varlen**：QKV 打平为 `[total_tokens, H, D]`，
     通过 `cu_seqlens`（cumulative sequence lengths）划分请求边界

4. **Chunked Prefill** — 长 prefill 拆成固定大小 chunk 与 decode
   请求混合执行，防止长文本编码阻塞在途 decode 请求。

### 为什么本项目不能用标准 Continuous Batching

上述机制依赖两个前提，本项目均不满足：

1. **PagedAttention / varlen kernel** — 需要自定义 CUDA kernel。
   本项目的 Attention 已编译进 TRT engine（ONNX → trtexec → .plan），
   计算图固定，无法运行时替换 Attention kernel 或注入 `cu_seqlens`
   参数。

2. **灵活的 KV Cache 内存管理** — TRT engine 的 KV Cache 输入 shape
   是 `[batch, num_heads, seq_len, head_dim]`，要求 batch 内所有请求
   的 `seq_len` 维度一致。不支持 page table 寻址或打平模式。

**根本约束**：TRT engine 是固定计算图。要实现标准 continuous batching
需要改造 Attention 图并嵌入 PagedAttention kernel — 等价于从 ONNX
导出层重写，或迁移到 TRT-LLM。

### 采用的方案：Padded Iteration-Level Batching

调度粒度为 iteration-level（每个 decode step 可增减请求），但 KV Cache
对齐方式为 padding（受 TRT engine 限制）：

- 请求可在任意 step 加入或完成，无需等整个 batch 结束
- 每步将活跃请求的 KV Cache pad 到 batch 内最大长度
- 完成的请求立即释放 KV Cache slot，新请求下一步即可加入

这是 "continuous scheduling + padded execution" 的混合策略 — 在 TRT
固定图约束下的最优折中。

### 演进路线

| 阶段 | 策略 | 说明 |
|------|------|------|
| 早期验证 | batch_size=1 + Multi-Instance | 见下方说明 |
| 生产 V1 | Padded iteration-level batching | Dispatcher 管理 slot 池，pad 开销可控 |
| 未来可选 | 自定义 PagedAttention kernel | 消除 padding 浪费，需 CUDA 开发 |
| 未来可选 | TRT-LLM 迁移 | 内置 inflight batching，但需重构模型构建流程 |

#### 早期验证：Multi-Instance 策略

不做 batch，而是跑多个模型实例，每个实例 batch_size=1 独立 decode：

- **零代码改动** — 纯配置：Triton 下设 `instance_group[{count:N}]`，
  自建引擎下起多个 executor 线程
- **无 shape 对齐问题** — 每个实例独立管理 KV Cache，互不干扰
- **CUDA Stream 并行** — 多实例可分配到不同 stream，GPU 计算核心有
  一定并行性（视算力余量）

**代价**：显存 × N（每个实例一份 engine context + KV Cache）。对于
talker ~1.7B 模型，单 GPU（24GB+）开 2-4 实例可行，是性价比最高的
初始并发方案。

## 采用的架构

```
┌──────────────────────────────────────────────────┐
│  Gateway 层 (可插拔)                               │
│                                                   │
│  A) Standalone gRPC Server      B) Triton Backend │
│     engine/gateway/                (未来可选)      │
│     - 开发/生产均可用              - 大规模部署    │
│     - 零外部依赖                  - 模型热更新     │
└──────────────┬────────────────────┬───────────────┘
               │                    │
               ▼                    ▼
┌──────────────────────────────────────────────────┐
│  TTSEngine (核心引擎，不感知外层 serving 框架)     │
│                                                   │
│  公开 API:                                        │
│    synthesize_stream()                            │
│    feed_text() / text_complete() / cancel()       │
│    start() / stop()                               │
└──────────────────────┬───────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Dispatcher     EngineLoop      KVCachePool
   (asyncio)    (GPU thread)     (预分配 slot)
```

### 分层职责

| 层 | 目录 | 职责 |
|----|------|------|
| Gateway | `engine/gateway/` | gRPC 双向流、协议转换、连接管理 |
| Frontend | `engine/frontend/` | 会话管理、文本分句（Spliter）、segment 调度 |
| Core | `engine/core/` | 纯数据结构（EngineRequest/Result/Session），无 GPU 依赖 |
| Backend | `engine/backend/` | GPU 线程、KV cache 池、prefill、batch decode、executor |

### 关键设计约束

1. **Engine 生命周期自包含** — `start()`/`stop()` 可被外部调用者
   在任意线程调用，不假设自己拥有进程（为未来 Triton 集成做准备）

2. **请求/响应用纯数据结构** — `EngineRequest`/`EngineResult` 是
   纯 dataclass，不绑定 gRPC stub 也不绑定 Triton `pb_utils`

3. **CPU-GPU 流水线** — GPU 执行 step N 的同时，CPU 处理 step N-1
   结果并准备 step N+1 输入

4. **优先级调度** — prefill 优先级 `FIRST_SEGMENT > CONTINUATION
   > PREFETCHED`；decode batch 可按首包紧迫性排序

### 长文本 Offline 预切分语义

离线场景下，`Spliter` 中的 `presplit` 与 `driver` 不是同一层概念：

- **presplit = 分组（group）**：利用全文视野把长文本切成多个较自然、长度接近的组，
  目标是提高离线合成时的并行度，并尽量降低单组触碰 `max_seq_len` 的风险
- **driver = 组内分句（segment）**：每个 group 内仍由 driver 按 L1/L2/L3/d
  阈值和后端状态协同决定真正的 flush 时机，driver 是最终裁决者
- **backend segment**：真正提交到 engine 的执行单元；一个 group 可以产出多个
  backend segment

因此，offline 路径的真实层次是：

`全文 -> presplit groups -> driver flush -> backend segments`

而不是“presplit 直接决定最终 segment 边界”。

### 超长 presplit 队列的执行模型

当长文本被 `presplit` 切成远多于 `max_batch_size` 的 group 时，系统仍可正常执行，
因为 frontend 与 backend 都做了分层限流：

- **frontend 限流**：offline group 不会一次性全部提交，只会启动到
  `max_concurrent_segments` 为止；其余 group 留在队列中等待前面的 segment 完成
- **backend 限流**：每个 decode iteration 只会从活跃 segment 中选择最多
  `max_batch_size` 个进入本轮 batch
- **顺序保证**：音频下发顺序按 `group_idx + local_idx` 进行层级重排，避免
  “前面 group 的后续句子被后面 group 抢先播放”

这意味着“presplit 很长”带来的主要问题是**排队和尾延迟**，而不是 correctness
失效或 engine 被一次性灌爆。

### EMA 与 Offline 长文本

对于超长 offline 请求，未来 group 不能一直复用最初的切分阈值。随着前面 segment
完成，audio/text ratio 的 EMA 会持续更新，因此：

- active driver 的阈值会随 EMA 刷新
- 新启动的 offline group 也必须基于**最新 EMA**重新计算阈值

否则，后半段文本会长期使用过时阈值，导致分组/分句策略逐渐偏离真实 decode 行为。

### 何时引入 Triton 壳

当遇到以下场景时考虑：

- 多模型共存（TTS + ASR + LLM 同 GPU）
- K8s 大规模部署需要 Triton 的 readiness/liveness probe
- 运维团队已有 Triton 集群管理经验
- 需要模型版本管理做 A/B 测试

集成方式：将 `TTSEngine` 封装为 `TritonPythonModel`，Triton 只做
收发请求，所有调度逻辑仍在 engine 内部。预计工作量：几百行适配代码。

## 自建引擎需要补齐的生产组件

| 组件 | 工作量 | 优先级 | 备注 |
|------|--------|--------|------|
| gRPC 服务 + 健康检查 | 低 | P0 | `grpc.aio` 成熟 |
| TRT engine 加载 | 低 | P0 | `tensorrt` Python API |
| Prometheus 指标 | 低 | P1 | `prometheus_client` |
| 优雅关停 / 请求排空 | 低 | P1 | signal handler + drain |
| 异常恢复 / session 超时 | 中 | P1 | watchdog + 超时回收 |
| CUDA Graph 优化 | 中 | P2 | 固定 batch size 场景 |
| 热更新 / 灰度发布 | 中 | P3 | 可后期做或交给 Triton |
| 多 GPU 调度 | 高 | P3 | 1.7B 单卡绑定，暂不需要 |
