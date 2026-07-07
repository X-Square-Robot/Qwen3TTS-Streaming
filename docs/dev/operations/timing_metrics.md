**English** | [中文](timing_metrics.zh-CN.md)

# Qwen3-TTS Timing Metrics Reference

This document lists all timing metrics exposed by the Qwen3-TTS engine,
together with their precise semantic definitions.

> Auto-generated from `engine/core/timing_semantics.py`.

## Standard Lifecycle Stages

| # | Stage name | Component | Description |
|---|-----------|-----------|-------------|
| 1 | `request.accepted` | Gateway | The gateway accepts the request |
| 2 | `session.config.validated` | Frontend | Config validation completed |
| 3 | `session.created` | Frontend | The session object is created |
| 4 | `session.registered` | Backend | The session is registered with the backend |
| 5 | `text.first_received` | Gateway | The first text arrives at the gateway/worker |
| 6 | `text.first_sent` | Frontend | The first text is sent to the engine |
| 7 | `text.first_enqueued` | Frontend | The first text is enqueued to the engine inbox |
| 8 | `text.first_dequeued` | Engine thread | The first text is dequeued by the engine thread |
| 9 | `engine.prefill.started` | Engine thread | Prefill starts |
| 10 | `engine.prefill.completed` | Engine thread | Prefill completes |

> **Batched-admission semantics (since `b80cf45`):** when a session is admitted through the
> batched burst-admission path, `engine.prefill.started`/`completed` bracket the **whole batched
> admission pass** (slot allocation + batched prefix-cache restore + batched suffix embed for
> every session in that pass), not the session's own suffix compute. `engine_prefill_ms`
> therefore grows with the number of sessions admitted together (tens of ms under a 128-burst);
> at single-stream load the pass contains one session and the value matches the old semantics.
| 11 | `engine.decode.first_step` | Engine thread | The first decode step starts |
| 12 | `engine.audio.first_raw` | Engine thread | The first raw audio is produced |
| 13 | `output.audio.first_effective` | Output pipeline | The first effective audio is published |
| 14 | `session.completed` | Frontend | The session completes |

## Raw Timestamp Metrics (Server Monotonic Clock)

| Metric name | Stage | Category |
|-------------|-------|----------------|
| `server_session_created_monotonic` | session.created | strong |
| `server_first_text_enqueued_monotonic` | text.first_enqueued | strong |
| `server_first_text_dequeued_monotonic` | text.first_dequeued | strong |
| `server_prefill_started_monotonic` | engine.prefill.started | strong |
| `server_prefill_completed_monotonic` | engine.prefill.completed | strong |
| `server_first_raw_audio_monotonic` | engine.audio.first_raw | strong |
| `server_first_effective_audio_monotonic` | output.audio.first_effective | strong |

## Epoch Timestamp Metrics (Protocol Layer)

| Metric name | Stage | Category |
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

## Derived Duration Metrics

| Metric name | Start event | End event | Category | Deprecated alias |
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

## Contextual (Cross-domain) Metrics

| Metric name | Description |
|-------------|-------------|
| `client_request_to_server_first_audio_ms` | Client request → server first effective audio (cross-clock) |
| `client_request_to_server_first_raw_audio_ms` | Client request → server first raw audio (cross-clock) |

## Latency Breakdown

For a given slow request, the following latency components can be computed:

| Component | Computation |
|-----------|-------------|
| Session creation latency | request.accepted → session.created |
| Text inbound latency | text.first_received → text.first_enqueued |
| Engine queue wait | text.first_enqueued → text.first_dequeued |
| Inference latency | text.first_dequeued → engine.audio.first_raw |
| Gating latency | engine.audio.first_raw → output.audio.first_effective |
| Transport latency | client timestamp − server timestamp (contextual) |

## Deprecated Names

| Old name | New name | Note |
|----------|----------|-------|
| `first_audio_latency_ms` | `session_create_to_first_raw_audio_ms` | The start point was ambiguous |
| `server_ttft_ms` | `server_ttft_effective_ms` / `server_ttft_raw_ms` | Conflated raw/effective audio |
| `server_first_audio_epoch_ms` | `server_first_effective_audio_epoch_ms` | Conflated raw/effective audio |
| `server_total_latency_ms` | `total_latency_ms` | Naming consistency |
