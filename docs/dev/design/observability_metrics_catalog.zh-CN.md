[English](observability_metrics_catalog.md) | **中文**

# 引擎可观测性指标目录（实现前·单一真相源）

> 编写日期：2026-06-30
> 状态：**实现前契约 / 实现检查单**——本文是分层可观测性的**完整指标集**，实现时照此逐项落地，
> 避免"边写边发现缺指标再补"。新增指标先进本目录，再写代码。
> 关联：[[observability_tiers]]（四层模型与控制面）· [[observability_goals]]（计时语义契约）· [[engine-design-overview]]

---

## 0. 本目录怎么来的 / 怎么读

通过对全引擎三路穷举扫描（后端 / 前端文本路径 / 输出·网关·VAD），抽出约 **400 个原始可观测字段**。
原始字段直接列清单既冗长又无法导航，故本目录把它们**折叠成约 45 个"观测点"**——每个观测点是一个
**事件 / 记录 / 汇总**，打包一组相关字段，在同一时机一次性发出。实现按观测点为单位埋点。

每个观测点标注：

| 标注 | 含义 |
|---|---|
| **触发** | 何时发出（生命周期时机 / 每段 / 每切点 / 周期） |
| **字段** | 打包的字段集 |
| **来源** | `file:function`（已校验路径） |
| **现状** | ✅已有 · 🟡部分(零散/仅DEBUG/仅meta) · 🆕新增 |
| **协议** | 是否进客户端 meta（决定 L0 能否自分析） |

层级语义见 [[observability_tiers]] §2：**L1 常开低量** / **L2 决策"为什么"** / **L3 原始举证**，且 `L3⊃L2⊃L1`。
事件/字段命名沿用 [[observability_goals]]，协议字段统一 `server_*` 前缀。

> **核心 vs 穷举尾**：标 ⭐ 的是该层**核心指标**（实现第一优先、构成验收）；未标的是兜底细节，
> 按观测点已分组，需要时随该观测点一起落地，不会变成"又发现缺一个字段"。

---

## A. L1 日常日志（常开、量少而精）

目标体量：稳态每请求 = 规范生命周期事件（结构化）+ 每段一行 + 一条 `session.summary` + 周期性 health。
全部经 `LifecycleLogger.emit`（`engine/core/lifecycle.py`），机读 JSON + 人读伴随行。

### A1. 规范生命周期事件（18 phase，补齐到全集）

沿用 [[observability_goals]] §事件定义表。现已发 7 个（✅），其余 🆕 补齐。均 L1·强。

| ⭐ | phase | 触发 | 关键字段 | 来源 | 现状 |
|---|---|---|---|---|---|
| ⭐ | `request.accepted` | 网关收到请求 | transport, request_id, turn_id, client_request_ts_ms | grpc_server.py / websocket_server.py | 🆕 |
| ⭐ | `session.config.validated` | 配置校验通过 | input_mode, group_policy, task_type, vad_strategy, output_policy, protocol_version, **obs_level**, obs_level_clamped? | frontend/interface.py | 🟡 |
| ⭐ | `session.created` | Session 对象建好 | speaker, ref_audio_mode, session_state | frontend/interface.py | ✅ |
| | `session.registered` | 注册到后端 | kv_pool_slot, free_slots_at_register | backend/engine_loop.py | 🆕 |
| | `text.first_received` | 首文到网关 | text_preview(钳), client_text_ts_ms, raw_len | gateway | 🆕 |
| | `text.first_sent` | 首文发向引擎 | text_length, normalized_preview(钳) | frontend/interface.py | 🆕 |
| ⭐ | `text.first_enqueued` | 首文入 inbox | queue_depth | frontend/dispatcher.py | ✅ |
| ⭐ | `text.first_dequeued` | 引擎线程取出 | queue_wait_ms, queue_depth_at_dequeue | backend/engine_loop.py | ✅ |
| ⭐ | `engine.prefill.started` | prefill 开始 | segment_id, cache_hit, cache_tokens_reused | backend/engine_loop.py | 🆕 |
| ⭐ | `engine.prefill.completed` | prefill 完成 | segment_id, prefill_ms, prefill_source, prefill_tokens | backend/engine_loop.py | ✅ |
| | `engine.decode.first_step` | 首 decode step | segment_id | backend/engine_loop.py | 🆕 |
| ⭐ | `engine.audio.first_raw` | 首原始音频产出 | segment_id, audio_shape, first_raw_audio_epoch_ms | backend/engine_loop.py | 🟡 |
| ⭐ | `output.audio.first_effective` | 首有效音频发出 | segment_id, prefix_trim_applied, prefix_trimmed_ms, gating_delay_ms | interface/output.py | 🟡 |
| ⭐ | `session.completed` | 会话完成 | total_segments, total_audio_ms | frontend/interface.py | 🟡 |

### A2. 段级收尾事件 `engine.segment.eos`（每段一条）⭐

> 答日常"每段花了多久、合成了什么、拼了batch没、为啥这段短"——日常排查命中率最高的一条。

- **触发**：每段 EOS（`_handle_segment_eos`）
- **字段**：`segment_id` · `segment_text_preview`(钳) · `audio_steps` · `text_tokens` · `audio_text_ratio` ·
  `eos_reason`(codec_eos/silence_abort/kv_overflow/loop_abort) · `overflow` · `cache_hit` · `segment_prefill_ms` · `batched`(bool) · `batch_size_at_prefill`
- **来源**：`backend/engine_loop.py:_handle_segment_eos`
- **现状**：🟡（segment_end meta 已有部分，缺 `eos_reason`/`batched`/`ratio`）·**协议**：是（segment_end 事件 meta）

### A3. 会话汇总 `session.summary`（每请求一条）⭐

替代含糊 `first_audio=231.6ms`。聚合五个日常问题 + 吞吐 + 缓存。字段复用 `core/timing.py:ServerTimingAccumulator`
派生指标 + 🆕 `batch_summary`。完整字段见 [[observability_tiers]] §4.3。分组：

| 组 | 字段 | 来源 | 现状 |
|---|---|---|---|
| TTFT⭐ | prefill_ms, dequeue_to_first_raw_ms, create_to_first_raw_ms | timing.py | ✅ |
| 链路六分段⭐ | session_create / text_ingress / engine_queue / inference / gating / transport(ctx) | timing.py | 🟡 汇总 |
| 合成内容⭐ | final_synthesized_text(钳), total_chars, total_segments, coalesced, progress_protected | interface.py | 🟡 |
| **batch 汇总**⭐ | segments, batched, solo, max_batch_size_seen | engine_loop.py | 🆕 |
| VAD 汇总⭐ | policy, prefix_trimmed_ms, tail_trimmed_ms, begin_count, end_count, original/effective_audio_ms | interface/vad.py, gateway | 🟡 提炼 |
| 缓存路径 | prefix_cache_hit, cache_tokens_reused, full_prefill_count | engine_loop.py | ✅ |
| 吞吐 | total_steps, total_prefills, batch_size_avg | engine_loop.py | 🟡 |
| 总延迟⭐ | total_latency_ms | timing.py | ✅ |

人读伴随行：`session=<id> DONE ttft=… infer=… batch=2/3 vad_trim=… cache=HIT segs=3 "首句…"`

### A4. 错误事件（L1·强，结构化）⭐

| phase | 字段 | 来源 | 现状 |
|---|---|---|---|
| `session.timeout` | waited_ms, last_active_phase, session_age | engine_loop.py:~1514 | 🟡 |
| `session.evicted` | reason("idle>{n}s"), active_steps | engine_loop.py:~1095 | 🟡 |
| `engine.prefill.failed` | segment_id, error_type, error_msg | engine_loop.py:~562/1418 | 🟡 |
| `session.cancelled` | completed_segments | engine_loop.py:~490 | 🟡 |
| `session.error`(backpressure等) | error_type, error_msg, current_phase, segments_completed, audio_produced_ms | engine_loop.py:~371 | 🟡 |

错误字段集沿用 [[observability_goals]] §3.6。**协议**：是（error 事件 meta，供 L0 定位阶段）。

### A5. 引擎健康周期 gauge `engine.health`（非每请求，周期发）⭐

> 此前枚举未覆盖但属日常运维核心：聚合态健康，按固定间隔或显著变化发一条。

- **字段**：`active_sessions` · `pool_used/free/utilization` · `prefix_cache_hit_rate` · `total_pool_memory_mb` ·
  `queue_depth` · `cum_backpressure_rejected` · `cum_timeouts` · `cum_evictions` · `cum_eos`
- **来源**：`engine_loop.py`（计数器已在）/ `kv_cache_pool.py:stats` / `prefix_cache.py:stats`
- **现状**：🟡（计数器已存在，缺周期性汇总行）·**协议**：否（服务端运维用）

---

## B. L2 debug 日志（开发机/定点、决策"为什么"）

L2 在 L1 之上叠加决策理由记录，结构化、派生、人读、不含 tensor。每类带 `obs:<name>` 标签便于过滤。

### B1. 文本切分（答"子句怎么切、为什么这么切"）⭐ 最大缺口

| ⭐ | obs | 触发 | 字段 | 来源 | 现状 |
|---|---|---|---|---|---|
| ⭐ | `split_decision` | 每个切点 | path(offline/streaming), trigger(l1_punct/force_fallback_l1/force_hard_cut/driver_l1-3), remaining_kv, prefill_len, ema_ratio, thresholds{min_tokens_l1, force_split_at, l1/l2/l3_cap}, token_count_at_split, last_l1_pos, chosen_level, reason, text_preview | spliter.py:pre_split / driver.py | 🆕 |
| | `presplit_group` | 离线预切完成 | groups_created, tokens_per_group[], last_l1_pos, hard_cut_engaged, forced_split_count | spliter.py:_enqueue_presplit_groups | 🆕 |
| | `driver_transition` | 流式 FSM 跳转 | driver_state, token_count, threshold_met{l1,l2,l3,force}, fsm_rule_name, action_type, is_final | driver.py / spliter/core.py | 🆕 |
| | `punct_classify` | 标点 token(可采样) | token_text, punct_level, saw_level3_break, trailing_closer_handled | spliter.py:classify_punct_level | 🆕 |
| | `coalesce` | 合并/进度保护触发 | coalesced_from_tokens, accumulation_pos, progress_protected | spliter.py / interface.py | 🆕 |
| | `ema_update` | SEGMENT_END 更新 EMA | ema_before, ema_after, observed_ratio, alpha_used(normal/overflow), clamped_at | spliter.py:~601 | 🟡(零散 debug) |

### B2. 调度 / 批处理（答"为什么和它拼/没拼、为什么被饿"）⭐

| ⭐ | obs | 触发 | 字段 | 来源 | 现状 |
|---|---|---|---|---|---|
| ⭐ | `batch_compose`(prefill) | 每次 prefill 批 | batch_size, kv_free_before, members[{session,segment,priority}], skipped[{session,segment,reason}] | engine_loop.py:_try_prefill_pending/_one | 🆕 |
| ⭐ | `batch_compose`(decode) | 每次 decode 批(可采样) | batch_size, candidate_count, mlfq_level_dist, boosted_count, members[] | engine_loop.py:_get_active_slots_mlfq | 🆕 |
| | `mlfq_state` | 每段(采样/跳级时) | mlfq_level, decode_steps, steps_since_schedule, global_step, boosted | core/mlfq.py | 🟡 |
| | `queue_dynamics` | inbox 排空(采样) | pending_queue_size, drained_count, concurrent_backpressure_active | engine_loop.py:_drain_inbox / spliter.py | 🟡 |
| | `backpressure` | 拒绝时 | rejected, active_count, limit | engine_loop.py:~362 | ✅ |

### B3. Prefill / 缓存 / 参考音频

| obs | 触发 | 字段 | 来源 | 现状 |
|---|---|---|---|---|
| `prefill_detail` | 每段 prefill | prefill_source, plan_warnings[], ref_warnings[], cache_key, language, speaker, x_vector_only, instruct_preview | engine_loop.py / prefill.py | 🟡 |
| `cache_event` | 查缓存 | cache_hit, cache_tokens_reused, prefix_cache_{hits,misses,evictions,hit_rate,entries} | prefix_cache.py:stats | 🟡 |
| `ref_audio` | 用参考音频 | ref_source, ref_id, ref_audio_sha256, ref_text_hash, ref_preprocess_ms | prefill.py:~121 | 🟡 |

### B4. KV / Slot / Pad-EOS（答"为什么这段被截/补静音"）⭐

| ⭐ | obs | 触发 | 字段 | 来源 | 现状 |
|---|---|---|---|---|---|
| | `slot_state` | 每段(采样/转换时) | slot_id, past_len, frame_idx, text_idx, c2w_kv_length, trailing_count, active_slot_count, idle_seconds | kv_cache_pool.py / engine_loop.py | 🟡 |
| ⭐ | `pad_phase` | 进入/退出 pad | in_pad, pad_start_frame, pad_steps, pad_consecutive_silence, remaining_kv, dynamic_silence_limit, silence_abort_triggered | engine_loop.py:~1178-1244 | 🟡 |
| ⭐ | `kv_overflow` | past_len≥max_seq_len | past_len, max_seq_len, segment_overflow | engine_loop.py:~1036 | 🟡 |

### B5. 段合成 / 异常（答"合成为什么错了"）⭐

| ⭐ | obs | 触发 | 字段 | 来源 | 现状 |
|---|---|---|---|---|---|
| ⭐ | `segment_synthesis` | 每段收尾 | do_sample, temperature, repetition_penalty, sampling_seed, audio_steps, text_tokens, audio_text_ratio, eos_reason, **anomaly[]**(hit_kv_512/ratio_outlier/no_codec_eos), reason | engine_loop.py:_handle_segment_eos + executor.py | 🆕 |

`eos_reason=kv_overflow/silence_abort` + ratio 离群 + 命中 512 = [[engine-design-overview]] 记录的 C4 幻觉/不收尾征兆。

### B6. 输出门控 / 重排

| obs | 触发 | 字段 | 来源 | 现状 |
|---|---|---|---|---|
| `vad_transition` | 每次 begin/end | event, frame_ms, score/prob, threshold, begin_counter/end_counter | interface/vad.py | 🆕 |
| `reorder_state` | 乱序/stall 风险 | group_idx, local_idx, next_emit_segment, buffer_keys_pending, done_keys, stall_risk, reorder_latency_frames | spliter/reorder.py | 🆕 |

### B7. 重放支撑（原始时间戳 + tokenizer 快照）

| obs | 字段 | 来源 | 现状 |
|---|---|---|---|
| `raw_timestamps` | 全部 monotonic 原始戳(session_created/first_text_enqueued/dequeued/prefill_started/completed/first_raw/first_effective) | core/timing.py | ✅(派生已用,原始随 L2 落) |
| `tokenizer_observability` | raw_text, normalized_text, token_ids, spans, offsets, pieces(escaped) | interface.py(guarded DEBUG) + spliter/tokenizer.py | ✅ |

### B8. 错误细节（L1 错误事件的 L2 补充）

`prefill_error_invalid_task_type` · `prefix_cache_invalid` · `replacing_segment_warning` · `invalid_ref_codec_shape` ·
`append_after_done(overflow_token_count)`。来源 `engine_loop.py` / `prefill.py`，现状 🟡。

---

## C. L3 dump 日志（疑难杂症、原始举证）

L3 = **B 全部决策日志** + 逐步 tensor。复用 `engine/backend/debug_dump.py:EngineDebugDumper`（`ENGINE_DUMP_*`）。

| ⭐ | obs | 内容 | 来源 | 现状 |
|---|---|---|---|---|
| ⭐ | `step_tensor_dump` | 逐步 inputs/outputs：next_embed, codec_sum, full_codec, logits, gumbel_noise, cp_gumbel_noise, token_counts, attention_bias, position_ids, c2w_conv/transconv_states, wav | debug_dump.py | ✅ |
| ⭐ | `step_decision`(对齐) | 逐步：frame_idx, codec_id, eos_check(full_codec[:,0]==codec_eos_id), sampling_seed, in_pad, text_idx —— 与 tensor 同 dump_id | executor.py + debug_dump.py | 🆕 |
| | `dump_timeline` | timeline.jsonl/.tsv + slot_rows(past_len/frame_idx/text_idx/cache_position/full_codec_head/eos) | debug_dump.py | ✅ |
| | `split_decisions.jsonl`(对齐) | B1 的 split_decision 快照写入 dump 目录,与 tensor 同 session | spliter.py + debug_dump.py | 🆕 |
| | `model_config_snapshot` | 每 run 一次：num_layers, kv_heads, head_dim, hidden_size, codec_vocab_size, c2w_{layers,kv_heads,sliding_window}, dtype, max_batch_size, max_seq_len, 全部阈值/采样参数 | kv_cache_pool.py / executor.py / config.py | 🆕 |
| | `raw_kv_tensors` | talker_kv, c2w_kv, spk_embedding, cacheable_prefix_embeds, request_prefill_embeds | kv_cache_pool.py / prefill.py | ✅(随 dump) |
| | `dump_enabled` 事件 | dir, limit, sessions, include_wav —— 让日志能定位 dump 产物 | debug_dump.py | 🟡 |

排查闭环：L2 `segment_synthesis` 异常段 → 按 `session_id` 进 dump 目录 → `step_decision`+`step_tensor_dump`
逐步 logits/codec/CP stage + 同步 seed → 决策与原始证据经 `dump_id`/`frame_idx` 对照。

---

## D. L0 客户端可自分析子集（协议暴露）

客户端凭 done/segment_end/error 事件 meta 自答 L1 大部分问题。现协议已暴露 ~90 字段（`interface/output.py`
+ `core/timing.py:to_meta_dict`），客户端 `qwen3tts.diagnostics` 已能解析时间线。**新增进协议**：

| L1 观测点 | 进协议字段 | 现状 |
|---|---|---|
| TTFT/链路分段/总延迟 | `server_*_ms` 系列 | ✅ |
| 合成内容 | `final_synthesized_text`(钳), `segment_text_preview` | 🟡 |
| **batch 汇总** | `server_batch_summary`(json) | 🆕 |
| VAD 汇总 | `server_prefix_trimmed_ms` 等 | 🟡 提炼 |
| 段级 eos_reason | `segment_eos_reason` | 🆕 |
| 错误阶段 | `error_phase`, `error_type` | 🟡 |

L2/L3 不进协议（决策日志/tensor 仅服务端）。客户端 `summary()` 答不出时打印升级指令（见 [[observability_tiers]] §7）。

---

## E. 覆盖统计与实现工作量

| 层 | 观测点数 | 含核心⭐ | 主要 🆕（即实现工作） |
|---|---|---|---|
| L1 | ~14 事件 + summary + health | 11 | request.accepted/prefill.started/decode.first_step 等补齐、`engine.segment.eos` 增 eos_reason/batched、`batch_summary`、health 周期行 |
| L2 | ~22 记录 | 8 | **split_decision**、batch_compose×2、segment_synthesis、vad_transition、reorder_state、driver_transition |
| L3 | ~7 | 2 | step_decision 对齐、split_decisions.jsonl、model_config_snapshot |
| L0 | 协议子集 | — | batch_summary/eos_reason/final_text 进 meta + 客户端 summary() |

原始字段约 400 → 折叠 ~45 观测点。**绝大多数底层数据已存在于代码中（143/177 后端已记），缺的是
"按观测点结构化发出 + 分层 gating + 进协议"**，而非新增测量。这意味着实现以"埋点重构"为主，风险低。

> 实现顺序仍按 [[observability_tiers]] §10 的 P0→P4。本目录是各阶段的逐项检查单：每落一个观测点，
> 在对应行把现状从 🆕/🟡 改为 ✅。
