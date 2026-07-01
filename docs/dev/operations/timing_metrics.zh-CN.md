[English](timing_metrics.md) | **中文**

# Qwen3-TTS 时序指标参考

本文档列出了 Qwen3-TTS 引擎暴露的所有时序指标，
及其精确语义定义。

> 从 `engine/core/timing_semantics.py` 自动生成。

## 标准生命周期阶段

| # | 阶段名称 | 组件 | 描述 |
|---|-----------|-----------|-------------|
| 1 | `request.accepted` | Gateway | Gateway 接收请求 |
| 2 | `session.config.validated` | Frontend | 配置验证完成 |
| 3 | `session.created` | Frontend | Session 对象创建 |
| 4 | `session.registered` | Backend | Session 注册到后端 |
| 5 | `text.first_received` | Gateway | 首个文本到达 gateway/worker |
| 6 | `text.first_sent` | Frontend | 首个文本发往引擎 |
| 7 | `text.first_enqueued` | Frontend | 首个文本入队到引擎收件箱 |
| 8 | `text.first_dequeued` | Engine thread | 首个文本被引擎线程出队 |
| 9 | `engine.prefill.started` | Engine thread | Prefill 开始 |
| 10 | `engine.prefill.completed` | Engine thread | Prefill 完成 |
| 11 | `engine.decode.first_step` | Engine thread | 首个 decode 步骤开始 |
| 12 | `engine.audio.first_raw` | Engine thread | 首个原始音频产生 |
| 13 | `output.audio.first_effective` | Output pipeline | 首个有效音频发布 |
| 14 | `session.completed` | Frontend | Session 完成 |

## 原始时间戳指标（服务器单调时钟）

| 指标名称 | 阶段 | 分类 |
|-------------|-------|----------------|
| `server_session_created_monotonic` | session.created | strong |
| `server_first_text_enqueued_monotonic` | text.first_enqueued | strong |
| `server_first_text_dequeued_monotonic` | text.first_dequeued | strong |
| `server_prefill_started_monotonic` | engine.prefill.started | strong |
| `server_prefill_completed_monotonic` | engine.prefill.completed | strong |
| `server_first_raw_audio_monotonic` | engine.audio.first_raw | strong |
| `server_first_effective_audio_monotonic` | output.audio.first_effective | strong |

## Epoch 时间戳指标（协议层）

| 指标名称 | 阶段 | 分类 |
|-------------|-------|----------------|
| `server_request_received_epoch_ms` | request.accepted | contextual |
| `server_session_created_epoch_ms` | session.created | strong |
| `server_first_text_received_epoch_ms` | text.first_received | contextual |
| `server_first_text_enqueued_epoch_ms` | text.first_enqueued | strong |
| `server_first_text_dequeued_epoch_ms` | text.first_dequeued | strong |
| `server_prefill_started_epoch_ms` | engine.prefill.started | strong |
| `server_prefill_completed_epoch_ms` | engine.prefill.completed | strong |
| `server_first_raw_audio_epoch_ms` | engine.audio.first_raw | strong |
| `server_first_effective_audio_epoch_ms` | output.audio.first_effective | strong |
| `server_done_epoch_ms` | session.completed | strong |

## 派生时长指标

| 指标名称 | 起始事件 | 结束事件 | 分类 | 已弃用别名 |
|-------------|-------------|-----------|----------------|------------------|
| `session_create_to_first_raw_audio_ms` | session.created | engine.audio.first_raw | strong | `first_audio_latency_ms` |
| `session_create_to_first_effective_audio_ms` | session.created | output.audio.first_effective | strong | |
| `first_text_enqueue_to_first_raw_audio_ms` | text.first_enqueued | engine.audio.first_raw | strong | |
| `first_text_enqueue_to_first_effective_audio_ms` | text.first_enqueued | output.audio.first_effective | strong | |
| `first_text_dequeue_to_first_raw_audio_ms` | text.first_dequeued | engine.audio.first_raw | strong | |
| `first_text_dequeue_to_first_effective_audio_ms` | text.first_dequeued | output.audio.first_effective | strong | |
| `engine_queue_wait_ms` | text.first_enqueued | text.first_dequeued | strong | |
| `engine_prefill_ms` | engine.prefill.started | engine.prefill.completed | strong | |
| `first_raw_to_first_effective_audio_ms` | engine.audio.first_raw | output.audio.first_effective | strong | |
| `total_latency_ms` | request.accepted | session.completed | strong | `server_total_latency_ms` |

## 上下文（跨域）指标

| 指标名称 | 描述 |
|-------------|-------------|
| `client_request_to_server_first_audio_ms` | 客户端请求 → 服务器首个有效音频（跨时钟） |
| `client_request_to_server_first_raw_audio_ms` | 客户端请求 → 服务器首个原始音频（跨时钟） |

## 延迟分解

给定一个慢请求，可以计算以下延迟组成：

| 组件 | 计算方式 |
|-----------|-------------|
| Session 创建延迟 | request.accepted → session.created |
| 文本入站延迟 | text.first_received → text.first_enqueued |
| 引擎队列等待 | text.first_enqueued → text.first_dequeued |
| 推理延迟 | text.first_dequeued → engine.audio.first_raw |
| 门控延迟 | engine.audio.first_raw → output.audio.first_effective |
| 传输延迟 | 客户端时间戳 − 服务器时间戳（上下文相关） |

## 已弃用名称

| 旧名称 | 新名称 | 说明 |
|----------|----------|-------|
| `first_audio_latency_ms` | `session_create_to_first_raw_audio_ms` | 起始点不明确 |
| `server_ttft_ms` | `server_ttft_effective_ms` / `server_ttft_raw_ms` | 混淆原始/有效音频 |
| `server_first_audio_epoch_ms` | `server_first_effective_audio_epoch_ms` | 混淆原始/有效音频 |
| `server_total_latency_ms` | `total_latency_ms` | 命名一致性 |
