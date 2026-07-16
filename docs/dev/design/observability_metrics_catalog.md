**English** | [中文](observability_metrics_catalog.zh-CN.md)

# Engine Observability Metrics Catalog (Pre-Implementation · Single Source of Truth)

> Written: 2026-06-30
> Status: **Pre-implementation contract / implementation checklist** — this document is the **complete metric set** for tiered observability; implement it item by item accordingly,
> avoiding "discovering a missing metric mid-implementation and patching it in." New metrics enter this catalog first, then the code is written.
> Related: [[observability_tiers]] (the four-tier model and control plane) · [[observability_goals]] (the timing-semantics contract) · [[engine-design-overview]]

---

## 0. How this catalog came to be / how to read it

Through an exhaustive three-path scan of the whole engine (backend / frontend text path / output·gateway·VAD), about **400 raw observable fields** were extracted.
Listing the raw fields directly would be both verbose and impossible to navigate, so this catalog **folds them into about 45 "observation points"** — each observation point is a single
**event / record / summary** that packages a group of related fields and emits them all at once at the same moment. Implementation instruments per observation point.

Each observation point is annotated with:

| Annotation | Meaning |
|---|---|
| **Trigger** | When it is emitted (lifecycle moment / per segment / per split point / periodic) |
| **Fields** | The packaged field set |
| **Source** | `file:function` (verified path) |
| **State** | ✅ existing · 🟡 partial (scattered/DEBUG-only/meta-only) · 🆕 new |
| **Protocol** | Whether it goes into the client meta (determines whether L0 can self-analyze) |

For the tier semantics, see [[observability_tiers]] §2: **L1 always-on low-volume** / **L2 decision "why"** / **L3 raw evidence**, with `L3⊃L2⊃L1`.
Event/field naming follows [[observability_goals]]; protocol fields uniformly use the `server_*` prefix.

> **Core vs the exhaustive tail**: those marked ⭐ are that tier's **core metrics** (first priority to implement, constitute the acceptance criteria); the unmarked ones are fallback details,
> already grouped under an observation point and landed together with it when needed, so they won't turn into "another missing field discovered."

---

## A. L1 Daily Logs (always on, low-volume and lean)

Target volume: per request in steady state = canonical lifecycle events (structured) + one line per segment + one `session.summary` + a periodic health.
All go through `LifecycleLogger.emit` (`engine/core/lifecycle.py`), machine-readable JSON + a human-readable accompanying line.

### A1. Canonical lifecycle events (18 phases, completed to the full set)

Following [[observability_goals]] §Event definition table. 7 are already emitted (✅); the rest are 🆕 completed. All L1·strong.

| ⭐ | phase | Trigger | Key fields | Source | State |
|---|---|---|---|---|---|
| ⭐ | `request.accepted` | Gateway receives request | transport, request_id, turn_id, client_request_ts_ms | grpc_server.py / websocket_server.py | 🆕 |
| ⭐ | `session.config.validated` | Config validation passed | input_mode, group_policy, task_type, vad_strategy, output_policy, protocol_version, **obs_level**, obs_level_clamped? | frontend/interface.py | 🟡 |
| ⭐ | `session.created` | Session object built | speaker, ref_audio_mode, session_state | frontend/interface.py | ✅ |
| | `session.registered` | Registered to backend | kv_pool_slot, free_slots_at_register | backend/engine_loop.py | 🆕 |
| | `text.first_received` | First text reaches gateway | text_preview(clamped), client_text_ts_ms, raw_len | gateway | 🆕 |
| | `text.first_sent` | First text sent to engine | text_length, normalized_preview(clamped) | frontend/interface.py | 🆕 |
| ⭐ | `text.first_enqueued` | First text enters inbox | queue_depth | frontend/dispatcher.py | ✅ |
| ⭐ | `text.first_dequeued` | Engine thread pulls it out | queue_wait_ms, queue_depth_at_dequeue | backend/engine_loop.py | ✅ |
| ⭐ | `engine.prefill.started` | prefill starts | segment_id, cache_hit, cache_tokens_reused | backend/engine_loop.py | 🆕 |
| ⭐ | `engine.prefill.completed` | prefill completes | segment_id, prefill_ms, prefill_source, prefill_tokens | backend/engine_loop.py | ✅ |
| | `engine.decode.first_step` | First decode step | segment_id | backend/engine_loop.py | 🆕 |
| ⭐ | `engine.audio.first_raw` | First raw audio produced | segment_id, audio_shape, first_raw_audio_epoch_ms | backend/engine_loop.py | 🟡 |
| ⭐ | `output.audio.first_effective` | First effective audio sent | segment_id, prefix_trim_applied, prefix_trimmed_ms, gating_delay_ms | interface/output.py | 🟡 |
| ⭐ | `session.completed` | Session completed | total_segments, total_audio_ms | frontend/interface.py | 🟡 |

### A2. Segment-level close-out event `engine.segment.eos` (one per segment) ⭐

> Answers the daily "how long did each segment take, what was synthesized, was it batched, why is this segment short" — the highest hit-rate line for daily troubleshooting.

- **Trigger**: each segment's EOS (`_handle_segment_eos`)
- **Fields**: `segment_id` · `segment_text_preview`(clamped) · `audio_steps` · `text_tokens` · `audio_text_ratio` ·
  `eos_reason`(codec_eos/silence_abort/kv_overflow/loop_abort) · `overflow` · `cache_hit` · `segment_prefill_ms` · `batched`(bool) · `batch_size_at_prefill`
- **Source**: `backend/engine_loop.py:_handle_segment_eos`
- **State**: 🟡 (segment_end meta already has some, missing `eos_reason`/`batched`/`ratio`) · **Protocol**: yes (segment_end event meta)

### A3. Session summary `session.summary` (one per request) ⭐

Replaces the vague `first_audio=231.6ms`. Aggregates the five daily questions + throughput + cache. Fields reuse `core/timing.py:ServerTimingAccumulator`
derived metrics + 🆕 `batch_summary`. For the full fields, see [[observability_tiers]] §4.3. Grouping:

| Group | Fields | Source | State |
|---|---|---|---|
| TTFT⭐ | prefill_ms, dequeue_to_first_raw_ms, create_to_first_raw_ms | timing.py | ✅ |
| Pipeline six segments⭐ | session_create / text_ingress / engine_queue / inference / gating / transport(ctx) | timing.py | 🟡 aggregate |
| Synthesized content⭐ | final_synthesized_text(clamped), total_chars, total_segments, coalesced, progress_protected | interface.py | 🟡 |
| **batch summary**⭐ | segments, batched, solo, max_batch_size_seen | engine_loop.py | 🆕 |
| VAD summary⭐ | policy, prefix_trimmed_ms, tail_trimmed_ms, begin_count, end_count, original/effective_audio_ms | interface/vad.py, gateway | 🟡 distill |
| Cache path | prefix_cache_hit, cache_tokens_reused, full_prefill_count | engine_loop.py | ✅ |
| Throughput | total_steps, total_prefills, batch_size_avg | engine_loop.py | 🟡 |
| Total latency⭐ | total_latency_ms | timing.py | ✅ |

Accompanying human-readable line: `session=<id> DONE ttft=… infer=… batch=2/3 vad_trim=… cache=HIT segs=3 "first sentence…"`

### A4. Error events (L1·strong, structured) ⭐

| phase | Fields | Source | State |
|---|---|---|---|
| `session.timeout` | waited_ms, last_active_phase, session_age | engine_loop.py:~1514 | 🟡 |
| `session.evicted` | reason("idle>{n}s"), active_steps | engine_loop.py:~1095 | 🟡 |
| `engine.prefill.failed` | segment_id, error_type, error_msg | engine_loop.py:~562/1418 | 🟡 |
| `session.cancelled` | completed_segments | engine_loop.py:~490 | 🟡 |
| `session.error`(backpressure etc.) | error_type, error_msg, current_phase, segments_completed, audio_produced_ms | engine_loop.py:~371 | 🟡 |

The error field set follows [[observability_goals]] §3.6. **Protocol**: yes (error event meta, for L0 to locate the stage).

### A5. Engine health periodic gauge `engine.health` (not per request, emitted periodically) ⭐

> Not covered by the earlier enumeration but a core of daily ops: aggregate-state health, emitted at a fixed interval or on a significant change.

- **Fields**: `active_sessions` · `pool_used/free/utilization` · `prefix_cache_hit_rate` · `total_pool_memory_mb` ·
  `queue_depth` · `cum_backpressure_rejected` · `cum_timeouts` · `cum_evictions` · `cum_eos`
- **Source**: `engine_loop.py` (counters already present) / `kv_cache_pool.py:stats` / `prefix_cache.py:stats`
- **State**: 🟡 (counters already exist, missing a periodic summary line) · **Protocol**: no (server-side ops use)

---

## B. L2 Debug Logs (dev box / targeted, the decision "why")

L2 layers decision-reason records on top of L1: structured, derived, human-readable, no tensors. Each kind carries an `obs:<name>` tag for easy filtering.

### B1. Text splitting (answers "how clauses were split, why they were split this way") ⭐ the biggest gap

| ⭐ | obs | Trigger | Fields | Source | State |
|---|---|---|---|---|---|
| ⭐ | `split_decision` | Each split point | path(offline/streaming), trigger(l1_punct/force_fallback_l1/force_hard_cut/driver_l1-3), remaining_kv, prefill_len, ema_ratio, thresholds{min_tokens_l1, force_split_at, l1/l2/l3_cap}, token_count_at_split, last_l1_pos, chosen_level, reason, text_preview | spliter.py:pre_split / driver.py | 🆕 |
| | `presplit_group` | Offline pre-split done | groups_created, tokens_per_group[], last_l1_pos, hard_cut_engaged, forced_split_count | spliter.py:_enqueue_presplit_groups | 🆕 |
| | `driver_transition` | Streaming FSM transition | driver_state, token_count, threshold_met{l1,l2,l3,force}, fsm_rule_name, action_type, is_final | driver.py / spliter/core.py | 🆕 |
| | `punct_classify` | Punctuation token (sampleable) | token_text, punct_level, saw_level3_break, trailing_closer_handled | spliter.py:classify_punct_level | 🆕 |
| | `coalesce` | Coalesce / progress protection triggered | coalesced_from_tokens, accumulation_pos, progress_protected | spliter.py / interface.py | 🆕 |
| | `ema_update` | SEGMENT_END updates EMA | ema_before, ema_after, observed_ratio, alpha_used(normal/overflow), clamped_at | spliter.py:~601 | 🟡 (scattered debug) |

### B2. Scheduling / batching (answers "why it was/wasn't batched with X, why it was starved") ⭐

| ⭐ | obs | Trigger | Fields | Source | State |
|---|---|---|---|---|---|
| ⭐ | `batch_compose`(prefill) | Each prefill batch | batch_size, kv_free_before, members[{session,segment,priority}], skipped[{session,segment,reason}] | engine_loop.py:_try_prefill_pending/_one | 🆕 |
| ⭐ | `batch_compose`(decode) | Each decode batch (sampleable) | batch_size, candidate_count, mlfq_level_dist, boosted_count, members[] | engine_loop.py:_get_active_slots_mlfq | 🆕 |
| | `mlfq_state` | Per segment (sampled / on level change) | mlfq_level, decode_steps, steps_since_schedule, global_step, boosted | core/mlfq.py | 🟡 |
| | `queue_dynamics` | inbox drain (sampled) | pending_queue_size, drained_count, concurrent_backpressure_active | engine_loop.py:_drain_inbox / spliter.py | 🟡 |
| | `backpressure` | On rejection | rejected, active_count, limit | engine_loop.py:~362 | ✅ |

### B3. Prefill / cache / reference audio

| obs | Trigger | Fields | Source | State |
|---|---|---|---|---|
| `prefill_detail` | Each segment prefill | prefill_source, plan_warnings[], ref_warnings[], cache_key, language, speaker, x_vector_only, instruct_preview | engine_loop.py / prefill.py | 🟡 |
| `cache_event` | Cache lookup | cache_hit, cache_tokens_reused, prefix_cache_{hits,misses,evictions,hit_rate,entries} | prefix_cache.py:stats | 🟡 |
| `ref_audio` | Using reference audio | ref_source, ref_id, ref_audio_sha256, ref_text_hash, ref_preprocess_ms | prefill.py:~121 | 🟡 |

### B4. KV / Slot / Pad-EOS (answers "why this segment was truncated / padded with silence") ⭐

| ⭐ | obs | Trigger | Fields | Source | State |
|---|---|---|---|---|---|
| | `slot_state` | Per segment (sampled / on transition) | slot_id, past_len, frame_idx, text_idx, c2w_kv_length, trailing_count, active_slot_count, idle_seconds | kv_cache_pool.py / engine_loop.py | 🟡 |
| ⭐ | `pad_phase` | Enter/exit pad | in_pad, pad_start_frame, pad_steps, pad_consecutive_silence, remaining_kv, dynamic_silence_limit, silence_abort_triggered | engine_loop.py:~1178-1244 | 🟡 |
| ⭐ | `kv_overflow` | past_len≥max_seq_len | past_len, max_seq_len, segment_overflow | engine_loop.py:~1036 | 🟡 |

### B5. Segment synthesis / anomaly (answers "why synthesis went wrong") ⭐

| ⭐ | obs | Trigger | Fields | Source | State |
|---|---|---|---|---|---|
| ⭐ | `segment_synthesis` | Each segment close-out | do_sample, temperature, repetition_penalty, sampling_seed, audio_steps, text_tokens, audio_text_ratio, eos_reason, **anomaly[]**(hit_kv_512/ratio_outlier/no_codec_eos), reason | engine_loop.py:_handle_segment_eos + executor.py | 🆕 |

`eos_reason=kv_overflow/silence_abort` + an outlier ratio + hitting 512 = the C4 hallucination / non-termination sign recorded in [[engine-design-overview]].

### B6. Output gating / reordering

| obs | Trigger | Fields | Source | State |
|---|---|---|---|---|
| `vad_transition` | Each begin/end | event, frame_ms, score/prob, threshold, begin_counter/end_counter | interface/vad.py | 🆕 |
| `reorder_state` | Out-of-order / stall risk | group_idx, local_idx, next_emit_segment, buffer_keys_pending, done_keys, stall_risk, reorder_latency_frames | spliter/reorder.py | 🆕 |

### B7. Replay support (raw timestamps + tokenizer snapshot)

| obs | Fields | Source | State |
|---|---|---|---|
| `raw_timestamps` | All monotonic raw stamps (session_created/first_text_enqueued/dequeued/prefill_started/completed/first_raw/first_effective) | core/timing.py | ✅ (derivations already used, raw landed with L2) |
| `tokenizer_observability` | raw_text, normalized_text, token_ids, spans, offsets, pieces(escaped) | interface.py (guarded DEBUG) + spliter/tokenizer.py | ✅ |

### B8. Error details (L2 supplement to the L1 error events)

`prefill_error_invalid_task_type` · `prefix_cache_invalid` · `replacing_segment_warning` · `invalid_ref_codec_shape` ·
`append_after_done(overflow_token_count)`. Source `engine_loop.py` / `prefill.py`, state 🟡.

---

## C. L3 Dump Logs (hard cases, raw evidence)

L3 = **all of B's decision logs** + per-step tensors. Reuse `engine/backend/debug_dump.py:EngineDebugDumper` (`ENGINE_DUMP_*`).

| ⭐ | obs | Content | Source | State |
|---|---|---|---|---|
| ⭐ | `step_tensor_dump` | Per-step inputs/outputs: next_embed, codec_sum, full_codec, logits, gumbel_noise, cp_gumbel_noise, token_counts, attention_bias, position_ids, c2w_conv/transconv_states, wav | debug_dump.py | ✅ |
| ⭐ | `step_decision`(aligned) | Per step: frame_idx, codec_id, eos_check(full_codec[:,0]==codec_eos_id), sampling_seed, in_pad, text_idx —— same dump_id as the tensor | executor.py + debug_dump.py | 🆕 |
| | `dump_timeline` | timeline.jsonl/.tsv + slot_rows(past_len/frame_idx/text_idx/cache_position/full_codec_head/eos) | debug_dump.py | ✅ |
| | `split_decisions.jsonl`(aligned) | B1's split_decision snapshots written into the dump directory, same session as the tensors | spliter.py + debug_dump.py | 🆕 |
| | `model_config_snapshot` | Once per run: num_layers, kv_heads, head_dim, hidden_size, codec_vocab_size, c2w_{layers,kv_heads,sliding_window}, dtype, max_batch_size, max_seq_len, all thresholds/sampling params | kv_cache_pool.py / executor.py / config.py | 🆕 |
| | `raw_kv_tensors` | talker_kv, c2w_kv, spk_embedding, cacheable_prefix_embeds, request_prefill_embeds | kv_cache_pool.py / prefill.py | ✅ (with dump) |
| | `dump_enabled` event | dir, limit, sessions, include_wav —— lets the logs locate the dump artifacts | debug_dump.py | 🟡 |

Troubleshooting closed loop: L2 `segment_synthesis` anomaly segment → into the dump directory by `session_id` → `step_decision`+`step_tensor_dump`
per-step logits/codec/CP stage + synchronized seed → decision and raw evidence cross-referenced via `dump_id`/`frame_idx`.

---

## D. L0 Client Self-Analyzable Subset (protocol-exposed)

The client answers most of L1's questions from the done/segment_end/error event meta. The current protocol already exposes ~90 fields (`interface/output.py`
+ `core/timing.py:to_meta_dict`), and the client `qwen3tts.diagnostics` can already parse the timeline. **Added to the protocol**:

| L1 observation point | Field added to protocol | State |
|---|---|---|
| TTFT/pipeline segmentation/total latency | the `server_*_ms` series | ✅ |
| Synthesized content | `final_synthesized_text`(clamped), `segment_text_preview` | 🟡 |
| **batch summary** | `server_batch_summary`(json) | 🆕 |
| VAD summary | `server_prefix_trimmed_ms` etc. | 🟡 distill |
| Segment-level eos_reason | `segment_eos_reason` | 🆕 |
| Error stage | `error_phase`, `error_type` | 🟡 |

L2/L3 do not go into the protocol (decision logs/tensors are server-only). When the client `summary()` can't answer, it prints an escalation instruction (see [[observability_tiers]] §7).

---

## E. Coverage Statistics and Implementation Effort

| Tier | Observation points | Incl. core⭐ | Main 🆕 (i.e. implementation work) |
|---|---|---|---|
| L1 | ~14 events + summary + health | 11 | Complete request.accepted/prefill.started/decode.first_step etc., add eos_reason/batched to `engine.segment.eos`, `batch_summary`, health periodic line |
| L2 | ~22 records | 8 | **split_decision**, batch_compose×2, segment_synthesis, vad_transition, reorder_state, driver_transition |
| L3 | ~7 | 2 | step_decision alignment, split_decisions.jsonl, model_config_snapshot |
| L0 | Protocol subset | — | batch_summary/eos_reason/final_text into meta + client summary() |

Raw fields ~400 → folded into ~45 observation points. **The vast majority of the underlying data already exists in the code (143/177 recorded in the backend); what's missing is
"structured emission per observation point + tiered gating + into the protocol"**, not new measurement. This means implementation is mostly "instrumentation refactoring," low risk.

> The implementation order still follows P0→P4 in [[observability_tiers]] §10. This catalog is the item-by-item checklist for each phase: as each observation point lands,
> change the state on the corresponding row from 🆕/🟡 to ✅.
