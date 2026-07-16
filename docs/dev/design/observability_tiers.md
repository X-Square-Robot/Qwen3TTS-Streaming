**English** | [中文](observability_tiers.zh-CN.md)

# Engine Observability Tiered Design (Four-Tier Model + Escalation Troubleshooting Chain)

> Written: 2026-06-30
> Status: **Design document (pre-implementation contract)** — this document locks down the four-tier model, the control plane, per-tier fields, and the escalation troubleshooting flow,
> so subsequent implementation references it and avoids introducing another round of vague naming.
> Related: [[observability_goals]] (prerequisite; the goals and contract for the timing/lifecycle dimension) · [[engine-design-overview]] ·
> [[frontend_segmentation_pipeline]] · [[vad_design_goals]] · [[decode_fsm]]

---

## 0. This document's relationship to `observability_goals.md`

[[observability_goals]] defines the goals along the **timing/lifecycle dimension**: stable event naming, strong/contextual metrics,
raw/effective audio, and the client reconstructing the timeline from the protocol. It is the contract for "**what to observe**."

This document answers an **orthogonal question: how many tiers to observe at, what goes in each tier, and how to escalate from one tier to the next**. The two do not overlap:

- `observability_goals.md` gives the **event namespace** (`session.created` / `engine.prefill.completed` …) and **field semantics**;
- this document **cuts those events into four tiers by verbosity**, adds the **debug decision logs** and **dump post-hoc evidence** it does not cover,
  and defines an **escalation troubleshooting chain** that strings the four tiers into a troubleshooting flow.

Event names, field names, and the strong/contextual/raw/derived classifications **all follow** `observability_goals.md`; this document only adds the tiering and the debug/dump-specific fields.

---

## 1. Problem and Goals

### 1.1 Current state (the parts are scattered; a unified tiering is missing)

| Existing part | File | Nature |
|---|---|---|
| Structured lifecycle JSON logs | `engine/core/lifecycle.py` (`LifecycleLogger.emit`) | A single event entry point, but only emits ~7 of the canonical 18 phases |
| Cross-thread timing accumulator | `engine/core/timing.py` (`ServerTimingAccumulator`) | TTFT/pipeline segmentation is complete, serialized into protocol meta |
| Client self-analysis | `client/src/qwen3tts/diagnostics.py` etc. | `ServerTimingReport.explain_latency()` etc., strong timeline analysis |
| Full tensor dump | `engine/backend/debug_dump.py` (`EngineDebugDumper`) | Controlled by `ENGINE_DUMP_*` env, step-by-step dump |
| Scattered free-form logs | `logger.info/debug` in various modules | ~40 sites in engine_loop, no unified semantics |

**Structural gap**: the log level is hardcoded to `INFO` (`engine/server.py:1060`), with no runtime/config/per-session control;
dump is a separate env, debug is `isEnabledFor` scattered everywhere — the three are managed independently, with **no unified knob for "escalate from ① to ② to ③."**

### 1.2 The target troubleshooting flow (the main line this document establishes)

```
Client self-analysis ──can't solve──▶  Daily logs (L1) ──can't solve──▶  Debug logs (L2) ──can't solve──▶  Dump logs (L3)
   L0                          always on              dev box / targeted            hard cases
 Self-serve on high-freq       Recurring daily            Rare hard cases       Model post-hoc / compute-decision
 issues in seconds             troubleshooting                                   evidence
```

Let **high-frequency issues be self-served quickly**, and let **hard issues have a deterministic escalation path**. Each escalation gives finer info, higher cost, and a narrower enablement scope.

---

## 2. The Four-Tier Model

| Tier | Name | Audience/scenario | Default | Volume | Core question answered |
|----|------|----------|------|------|---------------|
| **L0** | Client self-analysis | Client SDK users | Always available (carried by the protocol) | 0 server cost | TTFT, pipeline segmentation, synthesized content, batch, VAD — **most daily issues** |
| **L1** | Daily logs (DAILY) | Production ops | **Always on** | A few lines per request | Same as L0, but a fuller server-side view; locating recurring daily troubleshooting |
| **L2** | Debug logs (DEBUG) | Dev box / targeted | opt-in | Tens of lines per request | **How clauses were split, why they were split that way, why synthesis went wrong** (the decision "why") |
| **L3** | Dump logs (DUMP) | Hard cases | Manually enabled per case | Per-step tensor files | **Raw evidence** for model post-hoc and sampling/CP compute decisions |

**Tiers are cumulative**: `L3 ⊃ L2 ⊃ L1`. Enabling L2 automatically includes all of L1; enabling L3 automatically includes all of L2's decision logs **plus** the tensor dump on top
(decision logs and tensors align by the same `dump_id`/`frame_idx`, so when troubleshooting, "escalating to dump" yields a complete side-by-side of decisions + raw evidence).

**Design invariants**:

1. **Same event namespace, refined tier by tier**. A tier only decides the *verbosity* and *which optional events fire*; it does not change event names (following `observability_goals.md`).
2. **Every L1 daily fact must also go into the protocol meta** — this is the precondition for L0 to hold. What the client cannot answer, server L1 answers with the same-named fields, only fuller → the escalation chain connects naturally.
3. **L2 = explanation ("why," derived, human-readable, cheap); L3 = evidence (raw tensors/post-hoc, machine-readable, expensive).** Use L2 for rare issues; escalate to L3 only for hard cases.

---

## 3. Control Plane (global + per-session override)

### 3.1 The level knob

Introduce a first-class concept `observability.level`, taking `daily | debug | dump` (corresponding to L1/L2/L3; L0 is always online and needs no switch).

**Priority (high → low, following existing engine.yaml conventions)**:

```
per-session override  >  ENGINE_OBS_LEVEL env var  >  engine.yaml observability.level  >  default (daily)
```

New section in `engine.yaml`:

```yaml
observability:
  level: daily                 # daily | debug | dump —— global default tier
  max_session_level: daily     # highest tier a client/request may escalate a single session to (production abuse-prevention, see §3.3)
  text_capture: preview        # disabled | preview | hashed | full (text privacy, following observability_goals §4)
  text_preview_chars: 64
  dump:                        # only takes effect at level=dump or when a single session is escalated to dump; equivalent to existing ENGINE_DUMP_*
    dir: workspace/engine_dump
    limit: 0                   # 0=unlimited
    include_wav: true
    sessions: ""               # comma-separated; empty=all
    input_keys: ""
    output_keys: ""
```

`ENGINE_OBS_LEVEL=debug` and similar env vars override the global tier; the existing `ENGINE_DUMP_*` env vars are kept as aliases for the L3 detail knob (backward compatible).

### 3.2 Per-session override (production targeted troubleshooting)

The client carries `output_policy.config["obs_level"]` (or `timing.extra["obs_level"]`) in the request,
which is parsed at the `session.config.validated` stage and pinned onto `SessionConfig.observability_level`,
then threaded through the `LifecycleLogger` calls and dumper gating.

Significance: **targeted debugging of a single bad session in production, without turning the global level to DEBUG and flooding the logs.** This is the key to making the troubleshooting chain runnable in production.

### 3.3 Abuse prevention (important safety constraint)

Per-session escalation is **clamped by the server-side `observability.max_session_level`**:

- Production defaults to `max_session_level: daily` → the client **cannot** escalate to debug/dump on its own (to prevent dump IO amplification into a DoS, and to prevent text leakage).
- Dev/staging environments set `max_session_level: dump` → the client is allowed to escalate in a targeted way.
- When the tier the request asks for > `max_session_level`, **clamp to the ceiling and record an `obs_level_clamped` warning in the `session.config.validated` event**, rather than silently swallowing it.

### 3.4 Runtime level check (implementation convention)

- Expensive payload construction for decisions/dumps must always check the level before constructing (following the existing `if not logger.isEnabledFor(DEBUG): return` guard style), avoiding L2/L3 overhead under L1.
- `LifecycleLogger.emit` gains optional `level`/`min_level` parameters: events below the currently effective tier become a no-op directly.

---

## 4. L1 Daily Log Specification (always on, low-volume and lean)

**Form**: per request = a small number of canonical lifecycle events (structured JSON, machine-readable) + **one human-readable session summary line** (grep and you get it).
Target volume: single-digit INFO lines per request in steady state.

### 4.1 The five daily questions ⇄ field mapping

| Daily question | Carrying event/field | Current state |
|---|---|---|
| **What is the TTFT** | Summary line `server_engine_prefill_ms` / `server_first_text_dequeue_to_first_raw_audio_ms` / `server_session_create_to_first_raw_audio_ms` | timing.py already computes; missing a human-readable summary |
| **How long did the client→pipeline take** | Pipeline segmentation: session creation / text ingress / queueing / inference / gating / transport (the six segments in `observability_goals §Acceptance Criteria`) | timing.py already has the derivations; missing a unified single line |
| **What content was synthesized** | `final_synthesized_text` + per-segment `segment_text_preview` (clamped by `text_capture`) | accumulator/segment_end meta has the preview; missing a session-level aggregate |
| **Whether batching happened** | **New** `engine.prefill.batched` / summary `batch_summary`: whether each segment's prefill was solo or mixed-batch, batch_size, same-batch session/segment | **Missing** — the batching decision is currently only scattered in DEBUG |
| **How VAD handled it** | Summary `vad_summary`: `prefix_trimmed_ms` / `tail_trimmed_ms` / `begin_count` / `end_count` / `original_vs_effective_audio_ms` | `_inject_vad_metrics` exists but only goes into done_meta; missing a daily line |

### 4.2 Canonical lifecycle events (completed to the full set)

Following the 18 phases in `observability_goals.md §Event definition table`. 7 are already emitted; L1 should complete the following (all INFO, structured):

`request.accepted` · `session.registered` · `text.first_received` · `text.first_sent` ·
`engine.prefill.started` · `engine.decode.first_step` · `engine.audio.first_raw` ·
`output.audio.first_effective` · `session.completed` · error-path `session.evicted` / `engine.prefill.failed` / `session.cancelled`.

### 4.3 Session summary line (replacing the vague `first_audio=231.6ms`)

On request completion, emit one structured summary aggregating the five questions of §4.1 + the cache path. Fields directly reuse `observability_goals §2 Session summary log`
and the derived metrics of `timing.py`, **plus** `batch_summary`:

```json
{
  "lifecycle": true, "phase": "session.summary", "session_id": "abc123",
  "ttft": {"prefill_ms": 28.3, "dequeue_to_first_raw_ms": 12.1, "create_to_first_raw_ms": 231.6},
  "pipeline_ms": {"session_create": 5.2, "text_ingress": 8.1, "engine_queue": 3.0,
                  "inference": 190.0, "gating": 8.5, "transport": null},
  "text": {"final_synthesized_text": "…", "total_chars": 142, "total_segments": 3,
           "coalesced": false, "progress_protected": false},
  "batch_summary": {"segments": 3, "batched": 2, "solo": 1, "max_batch_size_seen": 4},
  "vad_summary": {"policy": "tenvad", "prefix_trimmed_ms": 8.5, "tail_trimmed_ms": 0.0,
                  "begin_count": 1, "end_count": 1, "original_audio_ms": 1820, "effective_audio_ms": 1811},
  "cache": {"prefix_cache_hit": true, "cache_tokens_reused": 512, "full_prefill_count": 1},
  "total_latency_ms": 1520.3
}
```

Accompanying human-readable line (grep one line and you get it):

```
INFO engine.lifecycle session=abc123 DONE ttft=231.6ms infer=190.0ms batch=2/3 vad_trim=8.5ms cache=HIT segs=3 "今天天气…"
```

---

## 5. L2 Debug Log Specification (dev box / targeted, explaining "why")

L2 layers **decision-reason records** on top of L1 (structured, derived, human-readable, no tensors). It answers the three questions you listed at its core:
**how clauses were split, why they were split that way, and why synthesis went wrong.** Each kind of record has a stable `obs` tag for easy filtering.

### 5.1 Split decision ("why it was split this way") — the biggest gap

Splitting happens in `engine/frontend/spliter/spliter.py`: thresholds are computed by `_make_thresholds()`→`compute_thresholds(remaining_kv, ema_ratio, …)`;
`pre_split()` splits at an L1 punctuation where `n >= min_tokens_l1`, and falls back to last_l1 or does a hard cut when `n >= force_split_at`;
the streaming path is split reactively by the driver FSM using the L1/L2/L3 thresholds. **These "whys" are currently completely invisible.** L2 records one entry per split point:

```json
{
  "obs": "split_decision", "session_id": "abc123", "segment_id": 2, "group_idx": 1,
  "path": "offline_pre_split",              // offline_pre_split | streaming_driver
  "trigger": "force_fallback_l1",           // l1_punct | force_fallback_l1 | force_hard_cut | driver_l1/l2/l3
  "remaining_kv": 468, "prefill_len": 12, "ema_ratio": 6.0,
  "thresholds": {"min_tokens_l1": 38, "force_split_at": 327,
                 "l1_cap": 327, "l2_cap": 374, "l3_cap": 421},
  "token_count_at_split": 327, "last_l1_pos": 41, "chosen_level": 1,
  "reason": "force_split_at(327) reached without L1 since pos 41; fell back to last L1 at 41",
  "text_preview": "……他说，"
}
```

When troubleshooting "why was this sentence chopped up / why didn't it break where it should have," a single record gives the thresholds, KV watermark, EMA, triggering branch, and human-readable reason.

### 5.2 Batch composition ("why it was/wasn't batched with X")

The prefill batch is in `engine_loop._try_prefill_pending/_try_prefill_one` (by `RequestPriority` + `kv_pool.free_count` + the per-session slot cap);
the decode batch is in `_get_active_slots_mlfq` (MLFQ + overflow/silence eviction). L2 records one entry per batch:

```json
{
  "obs": "batch_compose", "stage": "prefill",          // prefill | decode
  "batch_size": 3, "kv_free_before": 60,
  "members": [{"session_id":"a","segment_id":0,"priority":"FIRST_SEGMENT"},
              {"session_id":"b","segment_id":1,"priority":"CONTINUATION"}],
  "skipped": [{"session_id":"c","segment_id":0,"reason":"max_slots_per_session"}]
}
```

### 5.3 VAD state transitions ("why this segment was trimmed")

L1 only gives the VAD summary; L2 gives each begin/end transition (from `vad_processor.process_chunk`):

```json
{"obs":"vad_transition","session_id":"abc123","event":"begin","frame_ms":120.0,"prob":0.82,"threshold":0.5}
```

### 5.4 Synthesis anomaly / sampling summary ("why synthesis went wrong")

At each segment's close, record one sampling + close-out summary (segment-level, not per-step — per-step belongs to L3), and mark suspicious segments with a **heuristic anomaly flag**:

```json
{
  "obs": "segment_synthesis", "session_id": "abc123", "segment_id": 2,
  "do_sample": true, "temperature": 0.9, "repetition_penalty": 1.1, "sampling_seed": 178412,
  "audio_steps": 511, "text_tokens": 40, "audio_text_ratio": 12.8,
  "eos_reason": "kv_overflow",              // codec_eos | silence_abort | kv_overflow | loop_abort
  "anomaly": ["hit_kv_512", "ratio_outlier_gt_clamp"],   // empty array = normal
  "reason": "ran to KV cap 512 without codec EOS; ratio 12.8 > clamp 10 → likely hallucination tail"
}
```

The combination of `eos_reason=kv_overflow/silence_abort` + an outlier `ratio` + hitting 512 is exactly the
C4 hallucination / non-termination sign recorded in [[engine-design-overview]]. L2 turns "why synthesis went wrong" from log archaeology into a single record carrying a `reason`.

---

## 6. L3 Dump Log Specification (hard cases, raw evidence)

L3 = **all of L2's decision logs** + **`EngineDebugDumper`'s per-step tensor/post-hoc**, aligned.

Reuse the existing `engine/backend/debug_dump.py`: at `level=dump` (or when a single session escalates to dump), automatically set the equivalent `ENGINE_DUMP_DIR`,
inheriting its `.pt` payload + `timeline.jsonl`/`.tsv` index + per-session/key filtering. **New alignment**:

1. Write the §5.1 `split_decision` decision snapshots into the dump directory (`split_decisions.jsonl`), in the same session directory as the tensors.
2. Each line of the dump's `timeline.jsonl` is augmented with `sampling_seed` / `eos_check` (`full_codec[:,0]` vs `codec_eos_id`),
   so that **sampling post-hoc and CP compute decisions** can be cross-referenced with L2's `segment_synthesis` via `dump_id`/`frame_idx`.
3. At the dump entry point, emit one `obs:"dump_enabled"` lifecycle event (with dir/limit/sessions), so the logs can locate where the dump artifacts are.

When troubleshooting "why the model post-hoc looks like this, which step CP flipped at," go from L2's `segment_synthesis` anomaly segment → into the dump directory by `session_id` →
per-step logits/codec/CP stage output + the synchronized sampling seed, forming a decision→evidence closed loop.

---

## 7. L0 Client Self-Analysis Specification (self-serve on high-frequency issues)

The client can answer most of L1's questions from the returned protocol **without pulling backend logs**. The existing `qwen3tts.diagnostics`
(`ServerTimingReport.explain_latency()` / `SegmentTimingReport` / `ErrorTimingReport` / `LatencyAnalyzer` / `TimelineReconstructor`)
already covers the timeline; the **gap** is stringing the five questions of §4.1 into a single "look at this first" entry point.

Add `SessionDiagnostics.summary()`, aligned with the §4.3 server summary, answering the same five questions + giving an **escalation suggestion**:

| What the client can self-answer | Source protocol field | Escalation hint when it can't answer |
|---|---|---|
| TTFT, which pipeline segment is slow | done_meta derived durations | "Inference segment is a large share → check server L1 `session.summary`" |
| What was synthesized | `final_synthesized_text` / `segment_text_preview` | "Segment text doesn't match expectations → escalate to L2 and check `split_decision`" |
| Whether batching happened | done_meta `batch_summary` (added to the protocol in §4) | —— |
| How much VAD trimmed | done_meta `vad_summary` | "Abnormal trimming → escalate to L2 and check `vad_transition`" |
| Which stage a failure occurred in | error event `error_phase` | "After locating the stage → escalate to L2/L3" |

**Key**: when `summary()` cannot answer, it **directly prints the next-tier troubleshooting instruction** ("grep `session=<id>` on the server to see L1" or "replay with `obs_level=debug`"),
making the escalation chain a client-visible product path rather than something passed on by word of mouth.

---

## 8. Field Matrix (event ⇄ log name ⇄ protocol name ⇄ tier)

> **The full metrics catalog is in [[observability_metrics_catalog]]** — the ~400 raw fields obtained from an exhaustive three-path scan of the whole engine,
> folded into a single source of truth of ~45 observation points, serving as the item-by-item checklist for implementation. The table below keeps only backbone examples to avoid maintaining duplicates against the catalog.

This is the matrix required by `observability_goals.md §Immediate next step`, completed per this document's four tiers. The `State` column: ✅ existing / 🟡 partial / 🆕 new.

| Event/field | Component | Log field name | Protocol field name | Tier | Strong/contextual | State |
|---|---|---|---|---|---|---|
| Canonical lifecycle 7 phases | lifecycle | each phase | done_meta epoch timestamps | L1 | strong | ✅ |
| Canonical lifecycle remaining 11 phases | lifecycle | each phase | — | L1 | strong | 🟡 complete |
| TTFT three measures | timing | `server_*_first_raw_audio_ms` | same name | L1 | strong | ✅ |
| Pipeline six segments | timing | `pipeline_ms.*` | each derived duration | L1 | strong/contextual | 🟡 aggregate |
| Synthesized content | interface | `final_synthesized_text` | same name (clamped by text_capture) | L1 | contextual | 🟡 |
| **batch summary** | engine_loop | `batch_summary` | `server_batch_summary` | L1 | strong | 🆕 |
| VAD summary | gateway | `vad_summary` | done_meta existing trim fields | L1 | strong | 🟡 distill |
| **split decision** | spliter | `obs:split_decision` | — (server-only) | L2 | —— | 🆕 |
| **batch composition** | engine_loop | `obs:batch_compose` | — | L2 | —— | 🆕 |
| **VAD transition** | gateway/vad | `obs:vad_transition` | — | L2 | —— | 🆕 |
| **segment synthesis / anomaly** | engine_loop | `obs:segment_synthesis` | — | L2 | —— | 🆕 |
| Per-step tensor/logits | debug_dump | `.pt` + `timeline.jsonl` | — | L3 | —— | ✅ |
| **dump↔decision alignment** | debug_dump | `split_decisions.jsonl` + seed/eos fields | — | L3 | —— | 🆕 |
| Client unified summary | client | — | reuses the protocol fields listed above | L0 | —— | 🆕 |

---

## 9. Mapping to Existing Code (rework points)

| Concern | File | Rework |
|---|---|---|
| Level knob | `engine/config.py` + `engine.yaml` | Add the `observability` section, `ENGINE_OBS_LEVEL` parsing, priority merge |
| Log initialization | `engine/server.py:1060` | The `basicConfig` level is driven by `observability.level`, no longer hardcoded to INFO |
| Level gating | `engine/core/lifecycle.py` | `emit` gains `min_level`, no-op below the effective tier |
| Per-session tier | `engine/core/types.py` (`SessionConfig`) | Add `observability_level`, §3.2 parsing + §3.3 clamping |
| L1 summary | `engine/frontend/interface.py` | Aggregate and emit `session.summary`/`session.completed` |
| batch observation | `engine/backend/engine_loop.py` | `_try_prefill_*` / `_get_active_slots_mlfq` record the L1 summary + L2 `batch_compose` |
| split decision | `engine/frontend/spliter/spliter.py` | Record L2 `split_decision` at the `pre_split` / driver split points (thresholds are already in `_make_thresholds`) |
| segment synthesis | `engine/backend/engine_loop.py` | `_handle_segment_eos` records L2 `segment_synthesis` + the anomaly heuristic |
| VAD observation | `engine/gateway/grpc_server.py` | `_inject_vad_metrics` distills the L1 `vad_summary`; `process_chunk` records L2 `vad_transition` |
| dump alignment | `engine/backend/debug_dump.py` | level=dump auto-enables; timeline gains seed/eos; write `split_decisions.jsonl` |
| Client summary | `client/src/qwen3tts/diagnostics.py` | `SessionDiagnostics.summary()` + escalation hints |

---

## 10. Phased Rollout

| Phase | Content | Output |
|---|---|---|
| **P0 Control plane** | `observability` config section + level knob + priority + per-session clamping + lifecycle `min_level` gating | One knob governs all three tiers; production can escalate in a targeted way |
| **P1 L1 daily** | Complete the lifecycle events + `session.summary` summary + `batch_summary` + `vad_summary` distillation + sync into the protocol | Each of the five daily questions has a greppable line; the client gets the same-named fields |
| **P2 L0 client** | `SessionDiagnostics.summary()` + escalation hints | High-frequency issues are self-served by the client; the escalation-chain entry point is visible |
| **P3 L2 debug** | `split_decision` / `batch_compose` / `vad_transition` / `segment_synthesis` | Rare issues get a "why"; the three questions are answerable |
| **P4 L3 dump alignment** | dump auto-enable + decision-log alignment + seed/eos fields | Decision→tensor evidence closed loop |

P0→P1→P2 is the "high-frequency half" of the troubleshooting chain and has the highest priority; P3→P4 is the "hard-case half," wired in after the first two are stable.

---

## 11. Acceptance Criteria

1. **One knob** (config/env/per-session) can switch between daily/debug/dump, with production defaulting to daily and the client unable to escalate beyond its permissions.
2. **With L1 always on**, one `session.summary` + one human-readable summary line answers the five questions of TTFT/pipeline/synthesized content/batch/VAD, without cross-file timestamp alignment.
3. **The L0 client** answers most of the above five questions from the protocol alone, and prints an explicit escalation instruction when it can't.
4. **Under L2**, "why the clause was split this way" is given by a single `split_decision` with thresholds + watermark + reason; "why synthesis went wrong" is given by `segment_synthesis`'s `eos_reason`+`anomaly`+`reason`.
5. **Under L3**, L2 decision records and the tensor dump can be cross-referenced via `session_id`/`dump_id`/`frame_idx`, forming a decision→evidence closed loop.
6. Field names are stable and can be part of the protocol contract (following the `observability_goals.md` naming).

---

## 12. Non-Goals

- Do not introduce OpenTelemetry / Prometheus (to be revisited once the semantics are stable, following `observability_goals §Non-Goals`).
- Do not guarantee global clock synchronization between client and server.
- L3 does not aim for real time — it is an offline evidence tool, large in volume and enabled per case.
- No UI dashboard.
