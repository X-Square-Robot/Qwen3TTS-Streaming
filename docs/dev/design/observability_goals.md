**English** | [中文](observability_goals.zh-CN.md)

# Qwen3-TTS Engine Observability Goals

> This document defines the goals and field semantics along the **timing/lifecycle dimension** ("what to observe").
> For its tiered rollout (the four-tier daily/debug/dump model, the control plane, debug decision logs, dump evidence, and the escalation troubleshooting chain), see [[observability_tiers]];
> for the full metrics catalog, see [[observability_metrics_catalog]].

## Background

Recent latency investigations have exposed a chronic problem in the current standalone-engine and remote-worker architecture:
when an end-to-end request slows down, we typically have to combine several log sources plus manually aligned timestamps
to reconstruct "what actually happened."

Typical scenarios include:

- The client emits text early, but the engine receives it much later
- The engine Session is created early, but the first `START_TOKENS` arrives much later
- Remote audio arrives on time but is delayed locally by prefix trimming
- TTFT looks slow, but different components use different start points
- In multi-turn conversations, the latency source of later turns is completely indistinguishable

This makes the system hard to operate, hard to troubleshoot, and hard to evolve safely while we simultaneously refactor the transport/session protocol.

The goal of this document is to define a systematic observability specification such that:

1. **The engine-side logs alone** can explain the full course of a request
2. **The protocol packets the client sends and receives alone** can reconstruct the engine's action timeline
3. **Timing metrics** use explicit, stable semantics rather than implicit inference

This document is a product and architecture goals document, not an implementation-patch note.

---

## Problem Statement

The current observability has the following weaknesses:

### W1: Timing metrics have inconsistent start points

- Example: the engine's `first_audio` currently means `session_created → first_audio_chunk_consumed`,
  but most people read it as `first_text_arrived_at_engine → first_audio`
- Different components define "first audio" differently, but never state it explicitly

### W2: Latency components are mixed together

- A single number blends transport latency, queueing latency, inference latency, and local audio-gating latency
- There is no way to tell whether the slowness is "in inference," "in queueing," or "in prefix trimming"

### W3: Key state transitions are not recorded as canonical events

- The transition times of a Session from PENDING → PREFILL → DECODING are invisible
- The time gap between text being enqueued and dequeued is invisible
- Prefix cache hits/misses are invisible

### W4: Logs must be cross-correlated across processes to reconstruct the truth

- Engine-thread logs, frontend-interface logs, and gateway logs are scattered in different places
- There is no unified request_id / session_id threading through all logs

### W5: Client protocol metadata is insufficient to reconstruct the timeline offline

- The client sends `client_request_ts_ms` / `client_text_ts_ms` / `client_end_ts_ms`
- But the server **does not return** the server-side lifecycle timestamps
- The client can only see "when audio arrived," not "what happened inside the engine"

### W6: Prefix trimming / VAD / local gating affect perceived TTFT, but the protocol does not distinguish them explicitly

- VAD policies are already defined in the protocol (prefix_trim / tail_guard / hybrid), but the implementation has not landed yet
- Once it lands, the gap between `first_raw_audio` and `first_effective_audio` will become an important latency source
- There is currently no mechanism for the client to know "how much audio was trimmed"

### W7: The text path has no observability

- We cannot know: when text arrived at the engine, when it was pulled out by the engine thread,
  whether the text was delayed by progress protection, or whether the text was coalesced

### W8: Multi-segment / multi-turn scenarios lack per-segment observability

- Currently there is only a Session-level summary
- Each Segment's prefill time, decode steps, and audio output volume are invisible
- In multi-turn conversations, whether later turns hit the prefix cache or did a full prefill is invisible

### W9: Error-path observability is almost zero

- Error events such as timeout, eviction, and PREFILL failure have no structured record
- The client only receives a single error event and cannot tell which stage the error occurred in

---

## Core Design Goals

For every synthesis request, we want to achieve two complete and consistent observable views:

### Server-side observable view

> An operator only needs to grep a single request_id / session_id
> to see the full lifecycle, complete with stable event names and a timing breakdown.

### Client protocol view

> A client, from the protocol packets it sends and receives alone, can compute the major latency segments
> and explain why the request was slow.

Both views must use the same semantic stages to describe the same timeline.

---

## Observability Principles

### Principle 1: Every timing metric must have an explicit semantic start and end

We no longer use informal names of undefined origin such as "ttft."
Every metric must define:

| Attribute | Meaning |
|------|------|
| Start event | Which event the time measurement starts from |
| End event | Which event the time measurement ends at |
| Clock domain | Monotonic clock or epoch (wall) clock |
| Strong/weak class | Strong metric or contextual metric |
| Raw/derived | A raw timestamp or a derived duration |

Example:

```
server_session_create_to_first_raw_audio_ms
  start: session.created (monotonic)
  end:   engine.audio.first_raw_chunk (monotonic)
  clock domain: server monotonic clock
  class: strong metric
  type: derived

server_first_text_enqueue_to_first_raw_audio_ms
  start: text.first_enqueued (monotonic)
  end:   engine.audio.first_raw_chunk (monotonic)
  clock domain: server monotonic clock
  class: strong metric
  type: derived
```

### Principle 2: Strong metrics and contextual metrics must be separated

**Strong metrics:**

- Generated entirely within a single authoritative clock domain
- Suitable for alerting and regression tracking
- Examples:
  - Engine thread dequeue to first raw audio segment
  - Frontend Session creation to first audio chunk
  - Client local receive to local render

**Contextual metrics:**

- Span multiple clock domains or rely on optional client timestamps
- Useful for diagnosis but unsuitable for SLOs
- Examples:
  - Client request send to server request receive
  - VAD model end instant to client first-audio receive

### Principle 3: Raw events and derived metrics must both exist

We should not only record already-computed durations. We also need the raw stage timestamps, because:

- Downstream systems can re-derive metrics
- Changes to the timing contract do not break replayability
- Suspicious arithmetic can be recomputed later

### Principle 4: "Raw audio" and "effective audio" must be distinct concepts

The main source of current confusion is that prefix trimming / output gating can make "the first returned audio" differ from "the first remotely received audio."

Concepts to distinguish:

| Concept | Meaning |
|------|------|
| Raw audio | Audio frames/chunks produced by engine inference, before any gating |
| Effective audio | Audio frames/chunks actually sent to the client after prefix trimming / VAD gating / output policy |
| Published audio | Audio written to the engine event bus |
| Rendered audio | Audio the client actually plays |

### Principle 5: Text-path observability must be on par with the audio path

We need to know:

- When the Session started
- When the first text segment was received by the gateway/worker
- When the first text segment entered the engine queue
- When the first text segment was pulled out by the engine thread
- What text actually reached the synthesis stage
- Whether the text was delayed by batching/coalescing/progress protection

This must be visible in both the logs and the protocol metadata.

### Principle 6: The error path must have structured observability

Every error event must include:

- Which stage it occurred in
- The error type and error code
- The Session state at that moment
- The number of completed steps (segments synthesized, audio steps produced, etc.)
- Timeout-type errors must include the wait duration

---

## Scope

### In scope

- The standalone engine (engine/)
- The gateway adapters (engine/gateway/)
- The frontend interface (engine/frontend/)
- The output pipeline (engine/interface/output.py)
- The remote worker (workspace/qwen3_tts_remote.py)
- Protocol metadata (tts.proto / the protocol layer)
- The client SDK (client/)
- Human-readable logs

### Out of scope (this phase)

- Distributed-tracing infrastructure (OpenTelemetry, etc.)
- External metrics backends (Prometheus, etc.)
- UI dashboards
- Global clock-synchronization guarantees

These can be added once the semantics are stable.

---

## Lifecycle Model

We standardize a canonical request lifecycle.

### Canonical stage definitions

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        Request lifecycle                            │
 │                                                                     │
 │  1. request.accepted           Gateway receives the request          │
 │  2. session.config.validated   Config validation passed              │
 │  3. session.created            Session object created                │
 │  4. session.registered         Session registered to the backend     │
 │  5. text.first_received        First text segment reaches gateway/worker │
 │  6. text.first_sent            First text segment sent to the engine  │
 │  7. text.first_enqueued        First text segment enqueued to engine inbox │
 │  8. text.first_dequeued        First text segment pulled out by engine thread │
 │  9. engine.prefill.started     Prefill starts                        │
 │ 10. engine.prefill.completed   Prefill completes                     │
 │ 11. engine.decode.first_step   First decode step starts              │
 │ 12. engine.audio.first_raw     First raw audio produced by the engine │
 │ 13. output.audio.first_effective First effective audio published to output stream │
 │ 14. session.completed          Session completed                     │
 │                                                                     │
 │  ── Multi-segment case (repeat 9-13 per segment) ──                │
 │  9s. engine.segment.N.prefill.started                              │
 │ 10s. engine.segment.N.prefill.completed                            │
 │ 11s. engine.segment.N.decode.first_step                            │
 │ 12s. engine.segment.N.audio.first_raw                              │
 │ 13s. engine.segment.N.audio.first_effective                        │
 │                                                                     │
 │  ── Exceptional path ──                                            │
 │ E1. session.timeout           Session timed out                     │
 │ E2. session.evicted           Session evicted                       │
 │ E3. engine.prefill.failed     Prefill failed                        │
 │ E4. session.cancelled         Session cancelled                     │
 │ E5. session.error             Generic error                         │
 └─────────────────────────────────────────────────────────────────────┘
```

Not every transport needs every stage, but all stages should have stable names.

---

## Desired Outcomes

### A. Logs should directly answer "what happened"

Given a request_id / session_id, the logs alone should answer:

| Question | Corresponding stages |
|------|----------|
| How long did Session creation take? | 1→3 |
| How long until the first text segment reached the engine? | 5→8 |
| Was the text delayed by queueing/coalescing/progress protection? | The 7→8 gap + extra flags |
| When did Prefill start? | 9 |
| When did decode step 0 start? | 11 |
| When did the first raw audio exist? | 12 |
| Was the first audio delayed by prefix trimming / VAD gating? | The 12→13 gap |
| What text was actually synthesized? | Text observability fields |
| Did the request use the prefix cache / full prefill / warm start? | Ancillary info on 9 |
| What protocol version and output policy were active? | Ancillary info on 2 |
| How long did each Segment take? | Per-segment timing 9s→13s |
| Why did the request fail? | Structured error info E1-E5 |

### B. Client protocol packets should answer "what the engine did" (without server logs)

From the packets the client sends and receives alone, the client should be able to reconstruct:

| Information | Source |
|------|------|
| When the server accepted the request | Protocol metadata timestamp |
| When the server accepted the first text segment | Protocol metadata timestamp |
| When synthesis actually started | Protocol metadata timestamp |
| When the first raw audio existed on the server | Protocol metadata timestamp |
| When the first effective audio was sent to the transport | Protocol metadata timestamp |
| Whether the output was gated by prefix trim / tail guard / hybrid policy | Protocol metadata flags |
| When the request completed | Protocol metadata timestamp |
| The audio and text volume of each Segment | Protocol metadata stats |
| Which stage a failed request failed in | Stage flag on the error event |

### C. Metrics should support both diagnosis and regression tracking

We should be able to track:

- Protocol-layer latency
- Pure engine latency
- Pure transport latency
- Client-perceivable latency
- Gating-introduced latency

without redefining metrics every time we debug an issue.

---

## Canonical Observability Surface

### 1. Canonical request event log

Introduce structured request-lifecycle logs that use stable event names.

Each lifecycle log entry contains at least:

| Field | Description |
|------|------|
| `session_id` | Unique session identifier |
| `request_id` | Unique request identifier (when available) |
| `turn_id` | Turn identifier (when available) |
| `segment_id` | Segment identifier (when multi-segment) |
| `phase` | Lifecycle stage name |
| `monotonic_ts` | Monotonic-clock timestamp |
| `epoch_ts` | Wall-clock timestamp (when safe) |
| Stage-key fields | Info specific to that stage |

The log format should be machine-parseable (JSON or key=value structured format recommended) and stable across versions.

#### Event definition table

| Stage name | Component | Key ancillary fields | Strong/contextual |
|----------|----------|-------------|-----------|
| `request.accepted` | Gateway | transport, client_request_ts | contextual |
| `session.config.validated` | Frontend interface | input_mode, output_policy, vad_strategy, protocol_version | strong |
| `session.created` | Frontend interface | speaker_id, ref_audio_mode | strong |
| `session.registered` | Backend | kv_pool_slot | strong |
| `text.first_received` | Gateway | text_preview, client_text_ts | contextual |
| `text.first_sent` | Frontend interface | text_length, normalized_preview | strong |
| `text.first_enqueued` | Frontend interface | queue_depth | strong |
| `text.first_dequeued` | Engine thread | wait_ms, queue_depth_at_dequeue | strong |
| `engine.prefill.started` | Engine thread | segment_id, cache_hit, cache_tokens | strong |
| `engine.prefill.completed` | Engine thread | segment_id, prefill_tokens, duration_ms | strong |
| `engine.decode.first_step` | Engine thread | segment_id | strong |
| `engine.audio.first_raw` | Engine thread | segment_id, audio_shape | strong |
| `output.audio.first_effective` | Output pipeline | segment_id, trim_applied, trimmed_ms | strong |
| `session.completed` | Frontend interface | total_segments, total_audio_ms | strong |
| `session.timeout` | Engine thread | waited_ms, last_active_phase | strong |
| `session.evicted` | Engine thread | reason, active_steps | strong |
| `engine.prefill.failed` | Engine thread | segment_id, error_type, error_msg | strong |
| `session.cancelled` | Frontend interface | completed_segments | strong |
| `session.error` | Any | error_type, error_msg, current_phase | strong |

### 2. Session summary log

On request completion, emit one structured summary that aggregates:

| Category | Fields |
|------|------|
| Text stats | total_text_chars, total_segments, text_coalesced, progress_protected |
| Audio stats | total_audio_chunks, total_audio_ms, total_audio_steps |
| Cache path | prefix_cache_hit, cache_tokens_reused, full_prefill_count |
| VAD/trim path | vad_policy, prefix_trim_applied, prefix_trimmed_ms, first_raw_to_effective_ms |
| Strong metrics | session_create_to_first_raw_audio_ms, first_text_enqueue_to_first_raw_audio_ms, first_text_dequeue_to_first_raw_audio_ms, first_text_dequeue_to_first_effective_audio_ms, total_latency_ms |
| Contextual metrics | client_request_to_server_first_audio_ms (requires client timestamps) |

This summary should replace the current vague single-field summary such as:

```
first_audio=231.6ms
```

with an explicit multi-field summary:

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

### 3. Client protocol metadata

#### 3.1 Currently available metadata

The current protocol already exposes:

- `protocol_version`
- `timing_contract`
- `client_request_ts_ms`
- `client_text_ts_ms`
- `client_end_ts_ms`
- `server_first_audio_epoch_ms`

#### 3.2 Server-side lifecycle timestamps to add

Add stable server-side lifecycle markers to the response metadata:

| Field name | Type | Semantics | Strong/contextual |
|--------|------|------|-----------|
| `server_request_received_epoch_ms` | int64 | Wall-clock time the gateway received the request | contextual |
| `server_session_created_epoch_ms` | int64 | Wall-clock time Session creation completed | strong |
| `server_first_text_received_epoch_ms` | int64 | Wall-clock time the first text segment reached the gateway | contextual |
| `server_first_text_enqueued_epoch_ms` | int64 | Wall-clock time the first text segment was enqueued | strong |
| `server_first_text_dequeued_epoch_ms` | int64 | Wall-clock time the first text segment was pulled out by the engine | strong |
| `server_prefill_started_epoch_ms` | int64 | Wall-clock time Prefill started | strong |
| `server_prefill_completed_epoch_ms` | int64 | Wall-clock time Prefill completed | strong |
| `server_first_raw_audio_epoch_ms` | int64 | Wall-clock time the first raw audio was produced | strong |
| `server_first_effective_audio_epoch_ms` | int64 | Wall-clock time the first effective audio was sent | strong |
| `server_done_epoch_ms` | int64 | Wall-clock time the session completed | strong |

#### 3.3 Derived-duration fields to add

| Field name | Semantics | Computation |
|--------|------|------|
| `server_session_create_to_first_raw_audio_ms` | Session creation to first raw audio | created → first_raw |
| `server_session_create_to_first_effective_audio_ms` | Session creation to first effective audio | created → first_effective |
| `server_first_text_enqueue_to_first_raw_audio_ms` | Text enqueue to first raw audio | enqueued → first_raw |
| `server_first_text_enqueue_to_first_effective_audio_ms` | Text enqueue to first effective audio | enqueued → first_effective |
| `server_first_text_dequeue_to_first_raw_audio_ms` | Text dequeue to first raw audio | dequeued → first_raw |
| `server_first_text_dequeue_to_first_effective_audio_ms` | Text dequeue to first effective audio | dequeued → first_effective |
| `server_first_raw_to_first_effective_audio_ms` | Raw audio to effective audio (gating latency) | first_raw → first_effective |
| `server_total_latency_ms` | Total request latency | request_received → done |

#### 3.4 Policy/result fields to add

| Field name | Type | Semantics |
|--------|------|------|
| `server_prefix_trim_applied` | bool | Whether prefix trimming was applied |
| `server_prefix_trimmed_ms` | float | Milliseconds trimmed off the prefix |
| `server_vad_policy` | string | VAD policy name |
| `server_output_gating_mode` | string | Output gating mode |
| `server_first_audio_kind` | enum | `raw` / `effective` — first-audio type |
| `server_cache_hit` | bool | Whether the prefix cache was hit |
| `server_cache_tokens_reused` | int32 | Number of reused KV-cache tokens |

#### 3.5 Segment-level metadata

When each Segment completes (the `segment_end` event), it should carry:

| Field name | Semantics |
|--------|------|
| `segment_id` | Segment number |
| `segment_text_preview` | Preview of this segment's text |
| `segment_prefill_ms` | This segment's prefill time |
| `segment_decode_steps` | This segment's decode steps |
| `segment_audio_ms` | Audio duration produced by this segment |
| `segment_cache_hit` | Whether this segment hit the cache |

#### 3.6 Error event metadata

Error events should carry:

| Field name | Semantics |
|--------|------|
| `error_phase` | The stage the error occurred in |
| `error_type` | Error type (timeout / eviction / prefill_failed / cancelled / internal) |
| `error_message` | Human-readable error message |
| `segments_completed` | Number of completed segments |
| `audio_produced_ms` | Audio duration already produced |

### 4. Text observability fields

We need to expose what text was synthesized and how it was transformed.

| Field | Level | Semantics |
|------|------|------|
| `raw_first_text_preview` | Session | Raw first-segment text preview |
| `normalized_first_text_preview` | Session | Normalized first-segment text preview |
| `final_synthesized_text` | Session | The final complete synthesized text |
| `per_segment_text` | Segment | Per-segment text |
| `text_coalesced` | Session | Whether the text was coalesced |
| `text_progress_protected` | Session | Whether the text was delayed by progress protection |
| `text_input_mode` | Session | Input mode (TOKEN/CLAUSE/LONG_SEGMENT/FULL_TEXT) |

For privacy-sensitive deployments, this must be configurable:

| Level | Behavior |
|------|------|
| `disabled` | Record no text |
| `preview` | Record only the first N characters |
| `hashed` | Record only a hash |
| `full` | Record the complete text |

### 5. Audio gating observability

We need explicit observability into output shaping:

| Field | Semantics |
|------|------|
| `prefix_trim_enabled` | Whether prefix trimming is enabled |
| `prefix_trim_implementation` | The trimming implementation |
| `prefix_trim_dropped_samples` | Number of samples dropped by trimming |
| `prefix_trimmed_ms` | Milliseconds dropped by trimming |
| `prefix_trim_trigger_sample` | The sample position that triggered trimming |
| `first_raw_audio_arrival_time` | First raw-audio arrival time |
| `first_effective_audio_publish_time` | First effective-audio publish time |

This is critical because perceived TTFT can be dominated by gating latency even when inference is fast.

---

## Concrete Gaps Found from Recent Incidents

### Gap 1: The engine's `first_audio` name is misleading

The current engine summary reports:

- `session created → first audio chunk consumed`

But many readers interpret it as:

- `first text arrived at engine → first audio`

This ambiguity has repeatedly caused confusion in TTFT analysis.

**Fix**: Use `session_create_to_first_raw_audio_ms` and `first_text_dequeue_to_first_raw_audio_ms` instead of the blanket `first_audio`.

### Gap 2: The first text-segment path is invisible inside the engine

We can infer:

- Session creation time
- Backend prefill time

But we cannot directly see:

- When the first text segment was enqueued to the backend
- When the first text segment was pulled out by the engine thread

These are exactly the timestamps needed to separate queueing latency from inference latency.

**Fix**: Record timestamps when the `EngineRequest` is enqueued and dequeued.

### Gap 3: The client-visible first-audio semantics are mixed

The current worker timing:

- Records the first effective audio after prefix trimming
- But treats it in the log as the "first returned audio"

This hides the distinction between "the remote engine produced audio" and "the local worker decided to send audio."

**Fix**: Record `first_raw_audio` and `first_effective_audio` separately in both the log and the protocol.

### Gap 4: Protocol metadata is still too thin to reconstruct offline

The client currently cannot reconstruct the full timeline from the protocol packets alone.

**Fix**: Add the protocol fields defined in §3.2-3.6.

### Gap 5: Text content and segment mapping are not first-class request facts

We often need to know:

- What text actually reached segment 0
- Whether the request had only one segment
- Whether the end arrived before/after the first audio

These should not require deep log archaeology.

**Fix**: Add the text observability fields defined in §4.

### Gap 6: The error path has no structured information

Timeout, eviction, prefill failure, and the like produce only an ordinary log, and the client only receives a single error event.

**Fix**: Add the error-event metadata defined in §3.6.

---

## Mapping to Existing Code

The following are the key locations in the current code that need reworking:

| Component | File | Current state | Required rework |
|------|------|----------|-----------|
| Session timing | `engine/core/session.py` | Only `created_at` + `first_audio_at` | Add `first_text_enqueued_at`, `first_text_dequeued_at`, `first_raw_audio_at`, `first_effective_audio_at`, etc. |
| Engine request | `engine/core/types.py` | `EngineRequest` has no timestamps | Add `enqueued_at`, `dequeued_at` fields |
| Engine loop | `engine/backend/engine_loop.py` | Only global counters such as `_total_steps` | Record lifecycle events as each EngineRequest is processed |
| Frontend interface | `engine/frontend/interface.py` | Basic logs for Session creation and text handling | Switch to structured logs + lifecycle events |
| Output pipeline | `engine/interface/output.py` | Only logs VAD state | Distinguish raw/effective audio timestamps |
| gRPC gateway | `engine/gateway/grpc_server.py` | Session-level logs | Add a `request.accepted` event and inject timestamps into response meta |
| WebSocket gateway | `engine/gateway/websocket_server.py` | Session-level logs | Same as gRPC |
| Proto definition | `engine/gateway/tts.proto` | `TimingContext` has only client timestamps | Add a `ServerTiming` message type |
| Client SDK | `client/src/qwen3tts/` | Does not parse server timestamps | Add `TimingReport` parsing and computation |

---

## Goal Hierarchy

### Goal 0: Metric hygiene

Rename vague metrics and document precise semantics.

**Success criteria:**

- Every exposed metric has a documented start/end definition
- Old vague names are deprecated or explicitly aliased
- `first_audio` is no longer used as a standalone metric

### Goal 1: Engine-internal lifecycle visibility

Make the server logs sufficient to explain a request.

**Success criteria:**

- A single request_id / session_id grep shows the full lifecycle stages
- Engine queueing latency vs inference latency is visible
- The summary log includes explicit stage durations
- The error path has structured information

### Goal 2: Client can reconstruct the lifecycle

Make the protocol metadata sufficient for packet-only analysis.

**Success criteria:**

- The client can compute the strong server stages from the returned metadata
- The raw-audio vs effective-audio distinction is visible
- The protocol exposes output-gating information
- Error events include stage information

### Goal 3: Request replay/debugging friendliness

Allow later replay and diagnosis without guessing the semantics.

**Success criteria:**

- Raw timestamps coexist with derived durations
- Per-Segment text/audio facts can be logged or serialized
- The client SDK provides a `TimingReport` utility class

---

## Non-Goals

- Full OpenTelemetry adoption is not required this phase
- Global clock synchronization between client and server is not guaranteed
- Exposing every internal tensor or scheduler detail is not required
- Building a polished UI before the semantics are stable is not required

---

## Suggested Phased Rollout

### Phase 1: Semantic cleanup

- Define canonical lifecycle stage names
- Document metric semantics
- Rename or replace vague summary logs
- **Expected output**: This document as the contract, all subsequent implementation referencing it

### Phase 2: Engine/server instrumentation

- Add enqueue/dequeue timestamps to `EngineRequest`
- Add key stage timestamps to `Session`
- Add the first raw-audio timestamp in the engine loop
- Add the lifecycle summary log
- **Expected output**: A single session_id grep reveals the full lifecycle

### Phase 3: Output-pipeline instrumentation

- Distinguish raw-audio and effective-audio timing in the output pipeline
- Explicitly expose the prefix-trim/gating effect
- **Expected output**: Gating latency becomes quantifiable

### Phase 4: Protocol enrichment

- Add a `ServerTiming` message to `tts.proto`
- Add lifecycle timestamps and derived metrics to the response metadata
- Expose gating and segment facts to the client
- **Expected output**: The client can reconstruct the timeline from protocol packets

### Phase 5: Client SDK and tooling

- Add a `TimingReport` utility class to the client SDK
- Add a small analyzer that reconstructs the lifecycle from logs
- Add a packet-only analyzer for protocol metadata
- **Expected output**: Out-of-the-box observability tools

---

## Acceptance Criteria

This work should be considered successful when all of the following hold:

1. **A slow request can be explained from a single structured server summary**,
   without manually aligning timestamps across files and subtracting.

2. **A client can explain a request from the protocol packets alone**,
   including whether the latency came from the text path, the engine path, or output gating.

3. **The system can distinguish the following latency components:**

| Latency component | Computation |
|----------|----------|
| Session creation latency | request.accepted → session.created |
| Text ingress latency | text.first_received → text.first_enqueued |
| Engine queueing latency | text.first_enqueued → text.first_dequeued |
| Inference latency | text.first_dequeued → engine.audio.first_raw |
| Gating latency | engine.audio.first_raw → output.audio.first_effective |
| Transport latency | Client timestamp - server timestamp (contextual) |

4. **The timing field names are stable enough to be part of the protocol contract.**

5. **Error events contain enough information to locate the problem stage and cause.**

---

## Immediate Next Step

The next design step should be to turn this document into a concrete field matrix:

| Stage/event name | Component | Log field name | Protocol field name | Strong/contextual | Raw/derived |
|-------------|----------|-----------|-----------|----------|----------|

This matrix can drive the implementation of the engine, gateway, and remote worker without introducing another round of vague timing names.

---

## Appendix: Glossary

| Term | Definition |
|------|------|
| strong metric | A metric generated entirely within a single clock domain, suitable for SLOs and alerting |
| contextual metric | A metric spanning multiple clock domains, suitable for diagnosis but not SLOs |
| raw audio | Audio produced by engine inference, before gating |
| effective audio | Audio sent to the client after prefix trimming / VAD / output policy |
| gating latency | The latency between raw audio → effective audio |
| prefix trim | The policy of removing silent/noisy samples at the start of audio |
| progress protection | The mechanism that delays enqueueing new text while the engine is still processing the previous segment |
| monotonic clock | A clock unaffected by system-time adjustments, suitable for measuring durations |
| epoch clock | The system wall-clock time, suitable for cross-process time alignment |
