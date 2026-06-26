# Qwen3-TTS 引擎可观测性目标

## 背景

近期的延迟排查暴露了当前独立引擎和远程 worker 架构中的一个顽疾：
当端到端请求变慢时，我们通常需要结合多个日志源，加上手工对齐时间戳，
才能还原出"到底发生了什么"。

典型场景包括：

- 客户端文本很早就发出了，但引擎很晚才收到
- 引擎 Session 创建很早，但第一个 `START_TOKENS` 很晚才到
- 远程音频按时到达，但在本地被前缀裁剪延迟
- TTFT 看起来很慢，但不同组件使用了不同的起始时间点
- 多轮对话中，后续轮次的延迟来源完全无法区分

这导致系统难以运维、难以排查、难以在我们同时重构传输/会话协议时安全演进。

本文档的目标是定义一套系统化的可观测性规范，使得：

1. **仅凭引擎侧日志**即可解释一次请求的完整过程
2. **仅凭客户端收发的协议包**即可重建引擎的动作时间线
3. **计时指标**使用显式的、稳定的语义，而非隐式推断

本文档是产品和架构目标文档，不是实现补丁说明。

---

## 问题陈述

当前可观测性存在以下弱点：

### W1: 计时指标起始点不一致

- 示例：引擎 `first_audio` 当前语义是 `session_created → first_audio_chunk_consumed`，
  但多数人理解为 `first_text_arrived_at_engine → first_audio`
- 不同组件对"首音频"的定义不同，但没有显式声明

### W2: 延迟成分混合在一起

- 一个数字混合了传输延迟、排队延迟、推理延迟、本地音频门控延迟
- 无法区分"慢在推理"还是"慢在排队"还是"慢在前缀裁剪"

### W3: 关键状态转换没有作为规范事件记录

- Session 从 PENDING → PREFILL → DECODING 的转换时间点不可见
- 文本从入队到出队的时间差不可见
- 前缀缓存命中/缺失不可见

### W4: 日志必须跨进程交叉比对才能还原真相

- 引擎线程日志、前端接口日志、网关日志分散在不同位置
- 没有统一的 request_id / session_id 贯穿所有日志

### W5: 客户端协议元数据不足，无法离线重建时间线

- 客户端发送了 `client_request_ts_ms` / `client_text_ts_ms` / `client_end_ts_ms`
- 但服务端**没有回传**服务端生命周期时间戳
- 客户端只能看到"音频什么时候到的"，看不到"引擎内部发生了什么"

### W6: 前缀裁剪/VAD/本地门控影响感知 TTFT，但协议没有显式区分

- VAD 策略已在协议中定义（prefix_trim / tail_guard / hybrid），但实现尚未落地
- 一旦落地，`first_raw_audio` 和 `first_effective_audio` 之间的差距将成为重要的延迟来源
- 目前没有任何机制让客户端知道"音频被裁剪了多少"

### W7: 文本路径没有可观测性

- 我们无法知道：文本什么时候到达引擎、什么时候被引擎线程取出、
  文本是否因为进度保护被延迟、文本是否被合并

### W8: 多 Segment / 多轮场景缺乏分段可观测性

- 当前只有 Session 级别的 summary
- 每个 Segment 的 prefill 时间、decode 步数、音频产出量不可见
- 多轮对话中，后续轮次是命中前缀缓存还是全量 prefill 不可见

### W9: 错误路径可观测性几乎为零

- 超时、驱逐、PREFILL 失败等错误事件没有结构化记录
- 客户端只收到一个 error 事件，无法区分错误发生在哪个阶段

---

## 核心设计目标

对于每一次合成请求，我们要实现两个完整且一致的可观测视图：

### 服务端可观测视图

> 一个运维人员只需要 grep 一个 request_id / session_id，
> 就能看到完整的生命周期，包含稳定的事件名称和计时分解。

### 客户端协议视图

> 一个客户端仅凭收发的协议包，就能计算出主要的延迟分段，
> 并解释请求为什么慢。

两个视图必须使用相同的语义阶段来描述同一条时间线。

---

## 可观测性原则

### 原则 1：每个计时指标必须有显式的语义起点和终点

我们不再使用未定义来源的 "ttft" 这样的非正式名称。
每个指标必须定义：

| 属性 | 含义 |
|------|------|
| 起始事件 | 时间计算的起点是什么事件 |
| 终止事件 | 时间计算的终点是什么事件 |
| 时钟域 | 单调钟（monotonic）还是挂钟（epoch） |
| 强弱分类 | 强指标还是上下文指标 |
| 原始/派生 | 是原始时间戳还是派生时长 |

示例：

```
server_session_create_to_first_raw_audio_ms
  起始: session.created (monotonic)
  终止: engine.audio.first_raw_chunk (monotonic)
  时钟域: 服务端单调钟
  分类: 强指标
  类型: 派生

server_first_text_enqueue_to_first_raw_audio_ms
  起始: text.first_enqueued (monotonic)
  终止: engine.audio.first_raw_chunk (monotonic)
  时钟域: 服务端单调钟
  分类: 强指标
  类型: 派生
```

### 原则 2：强指标和上下文指标必须分离

**强指标：**

- 完全在一个权威时钟域内生成
- 适合告警和回归追踪
- 示例：
  - 引擎线程出队到首段原始音频
  - 前端创建 Session 到首个音频块
  - 客户端本地接收到本地渲染

**上下文指标：**

- 涉及多个时钟域或可选的客户端时间戳
- 适用于诊断但不适合 SLO
- 示例：
  - 客户端请求发送到服务端请求接收
  - VAD 模型结束时刻到客户端首音频接收

### 原则 3：原始事件和派生指标必须同时存在

我们不应该只记录已经计算好的时长。我们还需要原始阶段时间戳，因为：

- 下游系统可以重新派生指标
- 计时契约变更不会破坏可回放性
- 可疑的算术可以后续重新计算

### 原则 4："原始音频"和"有效音频"必须是不同的概念

当前困惑的主要来源是前缀裁剪/输出门控可能使"首个返回音频"不同于"首个远程接收音频"。

需要区分的概念：

| 概念 | 含义 |
|------|------|
| 原始音频 (raw audio) | 引擎推理产出的、未经任何门控处理的音频帧/块 |
| 有效音频 (effective audio) | 经前缀裁剪/VAD门控/输出策略后实际发送给客户端的音频帧/块 |
| 发布音频 (published audio) | 写入引擎事件总线的音频 |
| 渲染音频 (rendered audio) | 客户端实际播放的音频 |

### 原则 5：文本路径可观测性必须与音频路径对等

我们需要知道：

- Session 什么时候开始
- 首段文本什么时候被网关/worker 接收
- 首段文本什么时候进入引擎队列
- 首段文本什么时候被引擎线程取出
- 实际到达合成阶段的文本是什么
- 文本是否因 batching/coalescing/进度保护 被延迟

这必须在日志和协议元数据中都可见。

### 原则 6：错误路径必须有结构化的可观测性

每个错误事件必须包含：

- 发生在哪个阶段
- 错误类型和错误码
- 此时的 Session 状态
- 已完成的步骤数（已合成的 segment 数、已产出的音频步数等）
- 超时类错误需要包含等待时长

---

## 范围

### 包含

- 独立引擎（engine/）
- 网关适配器（engine/gateway/）
- 前端接口（engine/frontend/）
- 输出管线（engine/interface/output.py）
- 远程 Worker（workspace/qwen3_tts_remote.py）
- 协议元数据（tts.proto / 协议层）
- 客户端 SDK（client/）
- 人类可读日志

### 不包含（本阶段）

- 分布式追踪基础设施（OpenTelemetry 等）
- 外部指标后端（Prometheus 等）
- UI 仪表盘
- 全局时钟同步保证

这些可以在语义稳定后再加。

---

## 生命周期模型

我们标准化一个规范的请求生命周期。

### 规范阶段定义

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        请求生命周期                                  │
 │                                                                     │
 │  1. request.accepted           网关接收到请求                        │
 │  2. session.config.validated   配置校验通过                          │
 │  3. session.created            Session 对象创建完成                   │
 │  4. session.registered         Session 注册到后端                     │
 │  5. text.first_received        首段文本到达网关/worker               │
 │  6. text.first_sent            首段文本发送向引擎                     │
 │  7. text.first_enqueued        首段文本入队到引擎 inbox              │
 │  8. text.first_dequeued        首段文本被引擎线程取出                 │
 │  9. engine.prefill.started     Prefill 开始                          │
 │ 10. engine.prefill.completed   Prefill 完成                          │
 │ 11. engine.decode.first_step   首个 decode 步骤启动                  │
 │ 12. engine.audio.first_raw     首段原始音频由引擎产出                 │
 │ 13. output.audio.first_effective 首段有效音频发布到输出流            │
 │ 14. session.completed          会话完成                              │
 │                                                                     │
 │  ── 多 Segment 场景（每个 segment 重复 9-13）──                     │
 │  9s. engine.segment.N.prefill.started                              │
 │ 10s. engine.segment.N.prefill.completed                            │
 │ 11s. engine.segment.N.decode.first_step                            │
 │ 12s. engine.segment.N.audio.first_raw                              │
 │ 13s. engine.segment.N.audio.first_effective                        │
 │                                                                     │
 │  ── 异常路径 ──                                                     │
 │ E1. session.timeout           会话超时                               │
 │ E2. session.evicted           会话被驱逐                             │
 │ E3. engine.prefill.failed     Prefill 失败                          │
 │ E4. session.cancelled         会话被取消                             │
 │ E5. session.error             通用错误                               │
 └─────────────────────────────────────────────────────────────────────┘
```

不是每个传输层都需要每个阶段，但所有阶段应有稳定的名称。

---

## 期望成果

### A. 日志应直接回答"发生了什么"

给定一个 request_id / session_id，仅凭日志就能回答：

| 问题 | 对应阶段 |
|------|----------|
| Session 创建花了多久？ | 1→3 |
| 首段文本多久到达引擎？ | 5→8 |
| 文本是否因排队/合并/进度保护被延迟？ | 7→8 间距 + 额外标记 |
| Prefill 什么时候开始？ | 9 |
| Decode step 0 什么时候开始？ | 11 |
| 首段原始音频什么时候存在？ | 12 |
| 首音频是否被前缀裁剪/VAD门控延迟？ | 12→13 间距 |
| 实际合成了什么文本？ | text observability 字段 |
| 请求是否使用了前缀缓存/全量 prefill/热启动？ | 9 的附带信息 |
| 激活的协议版本和输出策略是什么？ | 2 的附带信息 |
| 每个 Segment 各花了多久？ | 9s→13s 的分段计时 |
| 请求为什么会失败？ | E1-E5 的结构化错误信息 |

### B. 客户端协议包应能回答"引擎做了什么"（无需服务端日志）

仅凭客户端收发的包，客户端应能重建：

| 信息 | 来源 |
|------|------|
| 服务端何时接受了请求 | 协议元数据时间戳 |
| 服务端何时接受了首段文本 | 协议元数据时间戳 |
| 合成实际何时开始 | 协议元数据时间戳 |
| 首段原始音频何时存在于服务端 | 协议元数据时间戳 |
| 首段有效音频何时发送到传输层 | 协议元数据时间戳 |
| 输出是否被前缀裁剪/尾部保护/混合策略门控 | 协议元数据标记 |
| 请求何时完成 | 协议元数据时间戳 |
| 每个 Segment 的音频量和文本量 | 协议元数据统计 |
| 请求失败发生在哪个阶段 | 错误事件的阶段标记 |

### C. 指标应同时支持诊断和回归追踪

我们应该能够追踪：

- 协议层延迟
- 纯引擎延迟
- 纯传输延迟
- 客户端可感知延迟
- 门控引入的延迟

而不需要每次排查 bug 时重新定义指标。

---

## 规范可观测性表面

### 1. 规范请求事件日志

引入结构化的请求生命周期日志，使用稳定的事件名称。

每条生命周期日志至少包含：

| 字段 | 说明 |
|------|------|
| `session_id` | 会话唯一标识 |
| `request_id` | 请求唯一标识（如有时） |
| `turn_id` | 轮次标识（如有时） |
| `segment_id` | 段标识（多段时） |
| `phase` | 生命周期阶段名称 |
| `monotonic_ts` | 单调钟时间戳 |
| `epoch_ts` | 挂钟时间戳（安全时） |
| 阶段关键字段 | 该阶段的特定信息 |

日志格式应该是机器可解析的（推荐 JSON 或 key=value 结构化格式），且跨版本稳定。

#### 事件定义表

| 阶段名称 | 所属组件 | 关键附带字段 | 强/上下文 |
|----------|----------|-------------|-----------|
| `request.accepted` | 网关 | transport, client_request_ts | 上下文 |
| `session.config.validated` | 前端接口 | input_mode, output_policy, vad_strategy, protocol_version | 强 |
| `session.created` | 前端接口 | speaker_id, ref_audio_mode | 强 |
| `session.registered` | 后端 | kv_pool_slot | 强 |
| `text.first_received` | 网关 | text_preview, client_text_ts | 上下文 |
| `text.first_sent` | 前端接口 | text_length, normalized_preview | 强 |
| `text.first_enqueued` | 前端接口 | queue_depth | 强 |
| `text.first_dequeued` | 引擎线程 | wait_ms, queue_depth_at_dequeue | 强 |
| `engine.prefill.started` | 引擎线程 | segment_id, cache_hit, cache_tokens | 强 |
| `engine.prefill.completed` | 引擎线程 | segment_id, prefill_tokens, duration_ms | 强 |
| `engine.decode.first_step` | 引擎线程 | segment_id | 强 |
| `engine.audio.first_raw` | 引擎线程 | segment_id, audio_shape | 强 |
| `output.audio.first_effective` | 输出管线 | segment_id, trim_applied, trimmed_ms | 强 |
| `session.completed` | 前端接口 | total_segments, total_audio_ms | 强 |
| `session.timeout` | 引擎线程 | waited_ms, last_active_phase | 强 |
| `session.evicted` | 引擎线程 | reason, active_steps | 强 |
| `engine.prefill.failed` | 引擎线程 | segment_id, error_type, error_msg | 强 |
| `session.cancelled` | 前端接口 | completed_segments | 强 |
| `session.error` | 任意 | error_type, error_msg, current_phase | 强 |

### 2. Session 汇总日志

在请求完成时，发出一条结构化汇总，聚合：

| 类别 | 字段 |
|------|------|
| 文本统计 | total_text_chars, total_segments, text_coalesced, progress_protected |
| 音频统计 | total_audio_chunks, total_audio_ms, total_audio_steps |
| 缓存路径 | prefix_cache_hit, cache_tokens_reused, full_prefill_count |
| VAD/裁剪路径 | vad_policy, prefix_trim_applied, prefix_trimmed_ms, first_raw_to_effective_ms |
| 强指标 | session_create_to_first_raw_audio_ms, first_text_enqueue_to_first_raw_audio_ms, first_text_dequeue_to_first_raw_audio_ms, first_text_dequeue_to_first_effective_audio_ms, total_latency_ms |
| 上下文指标 | client_request_to_server_first_audio_ms (需客户端时间戳) |

此汇总应替代当前含糊的单字段汇总如：

```
first_audio=231.6ms
```

改为显式的多字段汇总：

```json
{
  "session_id": "abc123",
  "session_create_to_first_raw_audio_ms": 231.6,
  "first_text_enqueue_to_first_raw_audio_ms": 45.2,
  "first_text_dequeue_to_first_raw_audio_ms": 12.1,
  "engine_prefill_ms": 28.3,
  "first_raw_to_first_effective_ms": 8.5,
  "prefix_trim_applied": true,
  "prefix_trimmed_ms": 8.5,
  "total_latency_ms": 1520.3,
  "total_segments": 3,
  "prefix_cache_hit": true,
  "cache_tokens_reused": 512
}
```

### 3. 客户端协议元数据

#### 3.1 当前已有的元数据

当前协议已暴露：

- `protocol_version`
- `timing_contract`
- `client_request_ts_ms`
- `client_text_ts_ms`
- `client_end_ts_ms`
- `server_first_audio_epoch_ms`

#### 3.2 需要新增的服务端生命周期时间戳

在响应元数据中新增稳定的服务端生命周期标记：

| 字段名 | 类型 | 语义 | 强/上下文 |
|--------|------|------|-----------|
| `server_request_received_epoch_ms` | int64 | 网关接收到请求的挂钟时间 | 上下文 |
| `server_session_created_epoch_ms` | int64 | Session 创建完成的挂钟时间 | 强 |
| `server_first_text_received_epoch_ms` | int64 | 首段文本到达网关的挂钟时间 | 上下文 |
| `server_first_text_enqueued_epoch_ms` | int64 | 首段文本入队的挂钟时间 | 强 |
| `server_first_text_dequeued_epoch_ms` | int64 | 首段文本被引擎取出的挂钟时间 | 强 |
| `server_prefill_started_epoch_ms` | int64 | Prefill 开始的挂钟时间 | 强 |
| `server_prefill_completed_epoch_ms` | int64 | Prefill 完成的挂钟时间 | 强 |
| `server_first_raw_audio_epoch_ms` | int64 | 首段原始音频产出的挂钟时间 | 强 |
| `server_first_effective_audio_epoch_ms` | int64 | 首段有效音频发出的挂钟时间 | 强 |
| `server_done_epoch_ms` | int64 | 会话完成的挂钟时间 | 强 |

#### 3.3 需要新增的派生时长字段

| 字段名 | 语义 | 计算 |
|--------|------|------|
| `server_session_create_to_first_raw_audio_ms` | Session创建到首段原始音频 | created → first_raw |
| `server_session_create_to_first_effective_audio_ms` | Session创建到首段有效音频 | created → first_effective |
| `server_first_text_enqueue_to_first_raw_audio_ms` | 文本入队到首段原始音频 | enqueued → first_raw |
| `server_first_text_enqueue_to_first_effective_audio_ms` | 文本入队到首段有效音频 | enqueued → first_effective |
| `server_first_text_dequeue_to_first_raw_audio_ms` | 文本出队到首段原始音频 | dequeued → first_raw |
| `server_first_text_dequeue_to_first_effective_audio_ms` | 文本出队到首段有效音频 | dequeued → first_effective |
| `server_first_raw_to_first_effective_audio_ms` | 原始音频到有效音频（门控延迟） | first_raw → first_effective |
| `server_total_latency_ms` | 请求总延迟 | request_received → done |

#### 3.4 需要新增的策略/结果字段

| 字段名 | 类型 | 语义 |
|--------|------|------|
| `server_prefix_trim_applied` | bool | 是否应用了前缀裁剪 |
| `server_prefix_trimmed_ms` | float | 前缀裁剪掉的毫秒数 |
| `server_vad_policy` | string | VAD 策略名称 |
| `server_output_gating_mode` | string | 输出门控模式 |
| `server_first_audio_kind` | enum | `raw` / `effective` — 首音频类型 |
| `server_cache_hit` | bool | 是否命中前缀缓存 |
| `server_cache_tokens_reused` | int32 | 复用的 KV 缓存 token 数 |

#### 3.5 Segment 级别元数据

每个 Segment 完成时（`segment_end` 事件），应携带：

| 字段名 | 语义 |
|--------|------|
| `segment_id` | 段编号 |
| `segment_text_preview` | 该段文本预览 |
| `segment_prefill_ms` | 该段 prefill 耗时 |
| `segment_decode_steps` | 该段 decode 步数 |
| `segment_audio_ms` | 该段产出的音频时长 |
| `segment_cache_hit` | 该段是否命中缓存 |

#### 3.6 错误事件元数据

错误事件应携带：

| 字段名 | 语义 |
|--------|------|
| `error_phase` | 错误发生的阶段 |
| `error_type` | 错误类型（timeout / eviction / prefill_failed / cancelled / internal） |
| `error_message` | 人类可读的错误信息 |
| `segments_completed` | 已完成的段数 |
| `audio_produced_ms` | 已产出的音频时长 |

### 4. 文本可观测性字段

我们需要暴露合成了什么文本以及它如何被转换的。

| 字段 | 级别 | 语义 |
|------|------|------|
| `raw_first_text_preview` | Session | 原始首段文本预览 |
| `normalized_first_text_preview` | Session | 标准化后首段文本预览 |
| `final_synthesized_text` | Session | 最终合成的完整文本 |
| `per_segment_text` | Segment | 每段文本 |
| `text_coalesced` | Session | 文本是否被合并 |
| `text_progress_protected` | Session | 文本是否被进度保护延迟 |
| `text_input_mode` | Session | 输入模式（TOKEN/CLAUSE/LONG_SEGMENT/FULL_TEXT） |

对于隐私敏感的部署，这必须是可配置的：

| 级别 | 行为 |
|------|------|
| `disabled` | 不记录任何文本 |
| `preview` | 仅记录前 N 个字符 |
| `hashed` | 仅记录哈希 |
| `full` | 记录完整文本 |

### 5. 音频门控可观测性

我们需要对输出整形（output shaping）的显式可观测性：

| 字段 | 语义 |
|------|------|
| `prefix_trim_enabled` | 前缀裁剪是否启用 |
| `prefix_trim_implementation` | 裁剪实现方式 |
| `prefix_trim_dropped_samples` | 裁剪丢弃的采样数 |
| `prefix_trimmed_ms` | 裁剪丢弃的毫秒数 |
| `prefix_trim_trigger_sample` | 触发裁剪的采样位置 |
| `first_raw_audio_arrival_time` | 首段原始音频到达时间 |
| `first_effective_audio_publish_time` | 首段有效音频发布时间 |

这非常关键，因为感知 TTFT 可能被门控延迟主导，即使推理很快。

---

## 从近期事件中发现的具体缺口

### 缺口 1：引擎 `first_audio` 命名有误导性

当前引擎汇总报告：

- `session created → first audio chunk consumed`

但很多读者理解为：

- `first text arrived at engine → first audio`

这种歧义在 TTFT 分析中反复造成困惑。

**修复**：使用 `session_create_to_first_raw_audio_ms` 和 `first_text_dequeue_to_first_raw_audio_ms` 替代笼统的 `first_audio`。

### 缺口 2：首段文本路径在引擎内部不可见

我们可以推断：

- Session 创建时间
- 后端 Prefill 时间

但无法直接看到：

- 首段文本何时入队到后端
- 首段文本何时被引擎线程取出

这恰好是区分排队延迟和推理延迟所需的时间戳。

**修复**：在 `EngineRequest` 入队和出队时记录时间戳。

### 缺口 3：客户端可见的首音频语义混合

当前 Worker 计时：

- 记录前缀裁剪后的首个有效音频
- 但日志中把它当作"首个返回音频"

这隐藏了"远端引擎产出音频"和"本地 Worker 决定发送音频"的区别。

**修复**：在日志和协议中分别记录 `first_raw_audio` 和 `first_effective_audio`。

### 缺口 4：协议元数据仍然太薄，无法离线重建

客户端目前无法仅从协议包重建完整时间线。

**修复**：按 3.2-3.6 节定义新增协议字段。

### 缺口 5：文本内容和段映射不是一等请求事实

我们经常需要知道：

- 什么文本实际到达了 segment 0
- 请求是否只有一个 segment
- end 是否在首音频之前/之后到达

这些不应该需要深度日志考古。

**修复**：按第 4 节定义新增文本可观测性字段。

### 缺口 6：错误路径没有结构化信息

超时、驱逐、Prefill 失败等只产生一条普通日志，客户端只收到一个 error 事件。

**修复**：按 3.6 节定义新增错误事件元数据。

---

## 与现有代码的映射

以下是当前代码中需要改造的关键位置：

| 组件 | 文件 | 当前状态 | 需要的改造 |
|------|------|----------|-----------|
| Session 计时 | `engine/core/session.py` | 仅有 `created_at` + `first_audio_at` | 新增 `first_text_enqueued_at`, `first_text_dequeued_at`, `first_raw_audio_at`, `first_effective_audio_at` 等 |
| 引擎请求 | `engine/core/types.py` | `EngineRequest` 无时间戳 | 新增 `enqueued_at`, `dequeued_at` 字段 |
| 引擎循环 | `engine/backend/engine_loop.py` | 仅有 `_total_steps` 等全局计数 | 在每个 EngineRequest 处理时记录生命周期事件 |
| 前端接口 | `engine/frontend/interface.py` | Session 创建和文本处理有基本日志 | 改为结构化日志 + 生命周期事件 |
| 输出管线 | `engine/interface/output.py` | 仅日志记录 VAD 状态 | 区分 raw/effective 音频时间戳 |
| gRPC 网关 | `engine/gateway/grpc_server.py` | Session 级日志 | 新增 `request.accepted` 事件，在响应 meta 中注入时间戳 |
| WebSocket 网关 | `engine/gateway/websocket_server.py` | Session 级日志 | 同 gRPC |
| Proto 定义 | `engine/gateway/tts.proto` | `TimingContext` 只有客户端时间戳 | 新增 `ServerTiming` 消息类型 |
| 客户端 SDK | `client/src/qwen3tts/` | 不解析服务端时间戳 | 新增 `TimingReport` 解析和计算 |

---

## 目标层级

### 目标 0：指标卫生

重命名含糊的指标并文档化精确语义。

**成功标准：**

- 每个暴露的指标都有文档化的起止定义
- 旧的含糊名称被废弃或明确别名化
- `first_audio` 不再作为独立指标使用

### 目标 1：引擎内部生命周期可见性

使服务端日志足以解释一次请求。

**成功标准：**

- 一个 request_id / session_id grep 显示完整生命周期阶段
- 引擎排队延迟 vs 推理延迟可见
- 汇总日志包含显式的阶段时长
- 错误路径有结构化信息

### 目标 2：客户端可重建生命周期

使协议元数据足以进行仅包分析。

**成功标准：**

- 客户端可以从返回的元数据计算强服务端阶段
- 原始音频 vs 有效音频区分可见
- 协议暴露输出门控信息
- 错误事件包含阶段信息

### 目标 3：请求回放/调试友好性

允许后续回放和诊断而无需猜测语义。

**成功标准：**

- 原始时间戳与派生时长并存
- 每 Segment 文本/音频事实可记录或序列化
- 客户端 SDK 提供 `TimingReport` 工具类

---

## 非目标

- 本阶段不需要全面采用 OpenTelemetry
- 不需要保证客户端和服务端的全局时钟同步
- 不需要暴露每个内部 tensor 或调度器细节
- 不需要在语义稳定前打造精美的 UI

---

## 建议的分阶段推行

### 第一阶段：语义清理

- 定义规范生命周期阶段名称
- 文档化指标语义
- 重命名或替换含糊的汇总日志
- **预期产出**：本文档作为契约，所有后续实现以此为参照

### 第二阶段：引擎/服务端埋点

- 在 `EngineRequest` 新增入队/出队时间戳
- 在 `Session` 新增关键阶段时间戳
- 在引擎循环中添加首段原始音频时间戳
- 添加生命周期汇总日志
- **预期产出**：一个 session_id grep 可见完整生命周期

### 第三阶段：输出管线埋点

- 在输出管线中区分 raw-audio 和 effective-audio 计时
- 显式暴露前缀裁剪/门控效果
- **预期产出**：门控延迟可量化

### 第四阶段：协议丰富化

- 在 `tts.proto` 新增 `ServerTiming` 消息
- 在响应元数据中添加生命周期时间戳和派生指标
- 向客户端暴露门控和段事实
- **预期产出**：客户端可从协议包重建时间线

### 第五阶段：客户端 SDK 和工具

- 客户端 SDK 新增 `TimingReport` 工具类
- 添加从日志重建生命周期的小型分析器
- 添加仅包分析器用于协议元数据
- **预期产出**：开箱即用的可观测性工具

---

## 验收标准

本工作在以下所有条件满足时应视为成功：

1. **一个慢请求可以从一条结构化的服务端汇总中解释清楚**，
   无需手工跨文件对齐时间戳做减法。

2. **一个客户端可以仅从协议包解释一次请求**，
   包括延迟是来自文本路径、引擎路径还是输出门控。

3. **系统能区分以下延迟成分：**

| 延迟成分 | 计算方式 |
|----------|----------|
| Session 创建延迟 | request.accepted → session.created |
| 文本接入延迟 | text.first_received → text.first_enqueued |
| 引擎排队延迟 | text.first_enqueued → text.first_dequeued |
| 推理延迟 | text.first_dequeued → engine.audio.first_raw |
| 门控延迟 | engine.audio.first_raw → output.audio.first_effective |
| 传输延迟 | 客户端时间戳 - 服务端时间戳（上下文） |

4. **计时字段名称足够稳定，可以作为协议契约的一部分。**

5. **错误事件包含足够的信息定位问题阶段和原因。**

---

## 立即的下一步

下一步设计应该是将本文档转化为具体的字段矩阵：

| 阶段/事件名 | 所属组件 | 日志字段名 | 协议字段名 | 强/上下文 | 原始/派生 |
|-------------|----------|-----------|-----------|----------|----------|

该矩阵可以驱动引擎、网关和远程 Worker 的实现，而不会引入新一轮的含糊计时名称。

---

## 附录：术语表

| 术语 | 定义 |
|------|------|
| 强指标 (strong metric) | 完全在一个时钟域内生成的指标，适合 SLO 和告警 |
| 上下文指标 (contextual metric) | 涉及多个时钟域的指标，适合诊断但不适合 SLO |
| 原始音频 (raw audio) | 引擎推理产出的未经门控处理的音频 |
| 有效音频 (effective audio) | 经前缀裁剪/VAD/输出策略处理后发送给客户端的音频 |
| 门控延迟 (gating latency) | raw audio → effective audio 之间的延迟 |
| 前缀裁剪 (prefix trim) | 移除音频开头无声/噪声采样的策略 |
| 进度保护 (progress protection) | 当引擎正在处理前一段时，延迟新文本入队的机制 |
| 单调钟 (monotonic clock) | 不受系统时间调整影响的时钟，适合测量时长 |
| 挂钟 (epoch clock) | 系统挂钟时间，适合跨进程时间对齐 |
