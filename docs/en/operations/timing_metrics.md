# Qwen3-TTS Timing Metrics Reference

This document lists all timing metrics exposed by the Qwen3-TTS engine,
with their precise semantic definitions.

> Auto-generated from `engine/core/timing_semantics.py`.

## Canonical Lifecycle Phases

| # | Phase Name | Component | Description |
|---|-----------|-----------|-------------|
| 1 | `request.accepted` | Gateway | Gateway receives request |
| 2 | `session.config.validated` | Frontend | Configuration validated |
| 3 | `session.created` | Frontend | Session object created |
| 4 | `session.registered` | Backend | Session registered to backend |
| 5 | `text.first_received` | Gateway | First text arrives at gateway/worker |
| 6 | `text.first_sent` | Frontend | First text sent toward engine |
| 7 | `text.first_enqueued` | Frontend | First text enqueued to engine inbox |
| 8 | `text.first_dequeued` | Engine thread | First text dequeued by engine thread |
| 9 | `engine.prefill.started` | Engine thread | Prefill begins |
| 10 | `engine.prefill.completed` | Engine thread | Prefill completes |
| 11 | `engine.decode.first_step` | Engine thread | First decode step starts |
| 12 | `engine.audio.first_raw` | Engine thread | First raw audio produced |
| 13 | `output.audio.first_effective` | Output pipeline | First effective audio published |
| 14 | `session.completed` | Frontend | Session complete |

## Raw Timestamp Metrics (Server Monotonic)

| Metric Name | Phase | Classification |
|-------------|-------|----------------|
| `server_session_created_monotonic` | session.created | strong |
| `server_first_text_enqueued_monotonic` | text.first_enqueued | strong |
| `server_first_text_dequeued_monotonic` | text.first_dequeued | strong |
| `server_prefill_started_monotonic` | engine.prefill.started | strong |
| `server_prefill_completed_monotonic` | engine.prefill.completed | strong |
| `server_first_raw_audio_monotonic` | engine.audio.first_raw | strong |
| `server_first_effective_audio_monotonic` | output.audio.first_effective | strong |

## Epoch Timestamp Metrics (Protocol)

| Metric Name | Phase | Classification |
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

| Metric Name | Start Event | End Event | Classification | Deprecated Alias |
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

## Contextual (Cross-Domain) Metrics

| Metric Name | Description |
|-------------|-------------|
| `client_request_to_server_first_audio_ms` | Client request → server first effective audio (cross-clock) |
| `client_request_to_server_first_raw_audio_ms` | Client request → server first raw audio (cross-clock) |

## Latency Decomposition

Given a slow request, the following latency components can be computed:

| Component | Calculation |
|-----------|-------------|
| Session creation delay | request.accepted → session.created |
| Text ingress delay | text.first_received → text.first_enqueued |
| Engine queue wait | text.first_enqueued → text.first_dequeued |
| Inference latency | text.first_dequeued → engine.audio.first_raw |
| Gating delay | engine.audio.first_raw → output.audio.first_effective |
| Transport delay | client timestamps − server timestamps (contextual) |

## Deprecated Names

| Old Name | New Name | Notes |
|----------|----------|-------|
| `first_audio_latency_ms` | `session_create_to_first_raw_audio_ms` | Ambiguous start point |
| `server_ttft_ms` | `server_ttft_effective_ms` / `server_ttft_raw_ms` | Conflated raw/effective |
| `server_first_audio_epoch_ms` | `server_first_effective_audio_epoch_ms` | Ambiguous raw/effective |
| `server_total_latency_ms` | `total_latency_ms` | Consistent naming |
