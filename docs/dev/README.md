**English** | [中文](README.zh-CN.md)

# Developer Documentation

> For contributors, architects, and deep-dive developers

> **Branching & release rules** (user branches → `dev` → `beta` → `main`,
> hotfix channel, `v`-prefixed version tags only): see
> [CONTRIBUTING — Branching Model & Releases](../../CONTRIBUTING.md#branching-model--releases).

## Architecture

| Document | Description |
|----------|-------------|
| [Architecture Design](architecture.md) | Complete architecture document — inference engine, frontend/backend, protocol, full KV Cache chain |
| [Decode FSM Design](architecture/decode_fsm.md) | Decode-phase finite state machine design |
| [Engine Architecture Decisions](architecture/engine_decisions.md) | Core engine design decisions and rationale |
| [Streaming Protocol Design](architecture/streaming_protocol.md) | Standalone protocol redesign — gRPC/WebSocket streaming |

## Design Goals

| Document | Description |
|----------|-------------|
| [VAD Design Goals](design/vad_design_goals.md) | VAD (Voice Activity Detection) design goals and rationale |
| [Observability Goals](design/observability_goals.md) | Observability and metrics design goals |
| [Real-Time Audio Streaming](design/realtime_audio.md) | Real-time audio streaming design goals |
| [Incremental Text Normalization and Soft Drain](design/incremental_text_normalization_and_soft_drain.md) | Monotonic text commitment, ambiguity handling, WAIT/HOLD, Soft Drain, and state rollover design |

## Investigation Reports

| Document | Description |
|----------|-------------|
| [Streaming Hallucination Investigation](investigation/streaming_hallucination.md) | Diagnosis and mitigation of streaming/sampling hallucination issues |
| [Code2Wav State Size](investigation/code2wav_state_size.md) | Code2Wav state size analysis and memory trade-offs |
| [Serving Performance Benchmark](investigation/serving_performance_benchmark.md) | Connection/queue/inference latency breakdown across protocols and concurrency, with raw data |

## Operations

| Document | Description |
|----------|-------------|
| [Cross-Host Build](operations/cross_host_build.md) | Cross-host build workflow — compiling the TRT engine on a remote machine |
| [E2E Test Summary](operations/e2e_test_summary.md) | E2E test entry points and coverage |
| [Timing Metrics Reference](operations/timing_metrics.md) | Timing metric definitions and measurement methods |
| [Tooling Governance](operations/tooling_governance.md) | Tooling governance — placement rules, naming conventions, mental model |

For regular users, please see the [User Documentation](../user/).
