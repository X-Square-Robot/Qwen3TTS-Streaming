# Qwen3-TTS Triton — Developer Documentation

> Documentation for contributors, developers, and architects

## Architecture

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | Full architecture document (Chinese, 2333 lines) — the single comprehensive reference |
| [decode_fsm.md](architecture/decode_fsm.md) | Decode FSM design — token-level streaming state machine |
| [engine_decisions.md](architecture/engine_decisions.md) | Engine architecture decisions — why we chose this structure |
| [streaming_protocol.md](architecture/streaming_protocol.md) | Standalone protocol redesign — gRPC/WebSocket streaming |

## Design Goals

| Document | Description |
|----------|-------------|
| [vad_design_goals.md](design/vad_design_goals.md) | VAD (Voice Activity Detection) design goals and rationale |
| [observability_goals.md](design/observability_goals.md) | Observability and metrics design goals |
| [realtime_audio.md](design/realtime_audio.md) | Realtime audio streaming design goals |

## Investigations

| Document | Description |
|----------|-------------|
| [streaming_hallucination.md](investigation/streaming_hallucination.md) | Streaming hallucination investigation — diagnosis and mitigation |
| [code2wav_state_size.md](investigation/code2wav_state_size.md) | Code2wav state size analysis — memory and compute tradeoffs |

## Operations

| Document | Description |
|----------|-------------|
| [cross_host_build.md](operations/cross_host_build.md) | Cross-host build workflow — building TRT engines on remote machines |
| [e2e_test_summary.md](operations/e2e_test_summary.md) | E2E test summary — test entry points and coverage |
| [timing_metrics.md](operations/timing_metrics.md) | Timing metrics reference — metric definitions and measurement methodology |
| [tooling_governance.md](operations/tooling_governance.md) | Tooling governance — placement rules, naming conventions, mental model |

## Process (Historical)

| Document | Description |
|----------|-------------|
| [refactor_goals_v2.md](process/refactor_goals_v2.md) | V2 refactor goals (completed) — tests/tools consolidation, scripts/bash simplification |
| [refactor_goals_v3.md](process/refactor_goals_v3.md) | V3 refactor goals (current) — systematic project governance |
| [progress_2026-03-25.md](process/progress_2026-03-25.md) | Historical progress snapshot from March 2026 |
