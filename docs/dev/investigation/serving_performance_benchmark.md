**English** | [中文](serving_performance_benchmark.zh-CN.md)

# Serving Performance Benchmark: Connection / Queue / Inference Breakdown Across Protocols and Concurrency

## Scope

This report answers four questions ahead of open-sourcing the repo:

1. How long does connection establishment take (WebSocket vs gRPC vs Triton)?
2. How long does the engine spend queuing vs. actually running inference?
3. How does choice of serving path (`engine-grpc` / `engine-websocket` / `triton-grpc`) affect latency?
4. How does each of the above change as concurrency scales from 1 to 128 simultaneous streams — and what actually limits it?

Raw per-request data and the aggregated CSVs backing every number here live in
[`serving_performance_benchmark_data/`](serving_performance_benchmark_data/)
(see its `README.md` for how to reproduce). All metric names follow
[`docs/dev/operations/timing_metrics.md`](../operations/timing_metrics.md);
report format follows [`docs/user/benchmark_methodology.md`](../../user/benchmark_methodology.md).

## Test conditions

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB, driver 580.126.20 |
| Images | `qwen3-engine:25.10`, `qwen3tts-streaming:25.10` (NGC tag 25.10), rebuilt 2026-07-06 |
| TensorRT | 10.13.3.9 |
| Model | `custom-1.7b`, `custom_voice`, engine_mode=trt, **full bf16** (backbone/cp/code2wav) |
| Engine optimizations | CP-KV retained in the unrolled graph (~3× decode compute ↓), CUDA-graph decode replay, arena-ized batch KV gather — see `conditions.json` for commit hashes |
| Engine profile | **`max_batch_size=128`**, `max_input_len=128`, `max_seq_len=512` — auto-selected by export-aware sizing (raw capacity ~141 lanes on this card); **the batch profile is a test variable, not a hardware ceiling** (see below) |
| Scheduler | `max_sessions=128`, `max_queue_size=256`, MLFQ enabled |
| Request | fixed text/speaker(`Serena`)/`task_type=custom_voice`/`language=auto` for every request — isolates protocol/concurrency effects, and pins all measured rounds to a **~100% prefix-cache-hit regime** (verified: `cache_hit=True` on 100% of measured rows); every `prefill_ms` below is cache-hit (suffix-only) prefill |
| Isolation | **single service per run**: Triton stopped during engine trials, engine stopped during Triton trials (at batch=128 the engine alone holds ~29 GiB; the two cannot coexist on this card) |
| Warmup | 3 rounds (concurrency sweep) / 5 rounds (connection isolation), excluded |
| Trials | 3 independent full runs per serving side (reproducibility check) |
| Requests | 50,877 total, **0 failures** |
| Harness | [`tools/validation/perf_matrix_sdk.py`](../../../tools/validation/perf_matrix_sdk.py), built on the `qwen3tts` client SDK |

## Fixes applied before measuring

Building this benchmark surfaced four server-side bugs, all fixed and landed before this dataset was collected (commits `a41f35d`, `d5741b6`, `3758ad7`):

1. **`engine_queue_wait_ms` was dead code** — the timing-accumulator lookup used a request object that never carries `session_config`, so the metric was never populated for any transport since it was added.
2. **Triton never computed timing metrics at all** — the BLS orchestrator never created a `ServerTimingAccumulator`/`OutputPipeline`, unlike the two native gateways; `queue_wait`/`prefill`/`cache_hit`/`total_latency` were never computed for Triton sessions. As a side effect of the fix, Triton's terminal event type was unified from `"end"` to `"done"`.
3. **`cache_hit` always reported `False`** — a string-literal mismatch (`"prefix_cache_hit"` was checked but never assigned); the cache fast path itself always worked. This dataset postdates the fix: `cache_hit_rate` is 1.0 across all measured rows, as expected for a fixed-key workload.
4. **Export-aware engine sizing never ran** — a stale helper path made every build silently fall back to a wildly conservative memory-tier table (which capped this 32 GiB card at batch=32); the recalibrated estimator now sizes this card at batch=128.

## Connection establishment time

Single-stream, 50 measured rounds per mode, pooled across 3 trials; `cold` = fresh `TTSClient.connect()` per round, `reuse` = one client across all rounds:

| Transport | cold TTFT avg | reuse TTFT avg | Δ (connection cost) |
|---|---|---|---|
| engine-grpc | 26.5 ms | 16.2 ms | **~10.3 ms** |
| engine-websocket | 16.7 ms | 16.6 ms | ~0.1 ms (noise) |
| triton-grpc | 21.4 ms | 21.7 ms | ~0 (noise) |

Only `engine-grpc` shows a measurable connection-setup cost (~10 ms, gRPC channel + HTTP/2 handshake), which is why the client SDK keeps gRPC channels warm across sessions by default. Both WebSocket handshake and Triton's channel setup are noise-level on a local link; over a real network all of these scale with RTT.

## Single-request lifecycle breakdown

Single stream, warm engine, prefix-cache hit — the standard low-load profile:

| Stage | value |
|---|---|
| `queue_wait_ms` (software inbox wait) | 0.43 ms avg |
| `prefill_ms` (prefix-cache hit, suffix-only) | 0.97 ms avg |
| decode step (mean interval) | 12.7 ms → per-stream RTF ~0.16 |
| **server TTFT** (session create → first raw audio; same basis as the historical "13ms" claim) | **14.9 ± 0.2 ms** (min 14.5, p50 14.9, p99 15.3, n=50) |
| client TTFT (localhost, reused channel, engine-grpc) | 16.0 ± 0.3 ms (min 15.6, p99 16.9, n=150) |
| total (full ~10 s utterance) | ~1.28 s |

A cold, cache-miss prefill costs far more than the 0.97 ms shown here (~30 ms measured in spot checks); see Known limitations.

## Protocol comparison

TTFT avg (ms) by protocol × concurrency, pooled across 3 trials each:

| Concurrency | engine-grpc | engine-websocket | triton-grpc |
|---|---|---|---|
| 1 | 16.7 | 16.5 | 22.4 |
| 16 | 62.5 | **51.1** | 71.3 |
| 32 | 97.8 | **76.4** | 112.6 |
| 64 | 174.8 | **145.0** | 185.7 |
| 128 | 333.8 | **265.5** | 309.3 |

`engine-websocket` is consistently the fastest under load (~20% lower TTFT than engine-grpc at 128). `triton-grpc` sits between the two on averages at high concurrency but carries the heaviest tail (p99 at 128: 672 ms vs 476/389 for grpc/ws). All three share the engine core, so scaling *shape* is identical; the differences are transport/gateway overhead.

## Concurrency scaling: where the time actually goes

`engine-grpc`, pooled avg values (3 trials, 20 rounds/level):

| Concurrency | TTFT | `queue_wait_ms` | decode step (mean) | per-stream RTF | `batch_size_seen` |
|---|---|---|---|---|---|
| 1 | 16.7 ms | 0.43 ms | 12.7 ms | 0.16 | 1 |
| 16 | 62.5 ms | 12.4 ms | 17.7 ms | 0.22 | 16 |
| 32 | 97.8 ms | 21.7 ms | 23.6 ms | 0.29 | 32 |
| 64 | 174.8 ms | 34.4 ms | 36.8 ms | 0.46 | 64 |
| 128 | 333.8 ms | 71.9 ms | 67.5 ms | **0.84** | **128** |

(each decode step yields 80 ms of audio; per-stream RTF = step time / 80 ms)

Three observations:

1. **All 128 streams genuinely decode together** — `batch_size_seen` reaches 128, and TTFT grows smoothly (~2× per concurrency doubling) with no cliff. Verified live: polling `/health` during a 128-burst shows `active_sessions=128` with all sessions admitted at once.
2. **The real ceiling on this card is per-stream RTF, not TTFT.** Decode-step time grows with batch width; at 128-wide it reaches 67.5 ms per 80 ms of audio — RTF 0.84, still real-time but with only ~16% headroom. This is the honest capacity statement for this GPU/model: **128 concurrent real-time streams, near saturation.**
3. **Under a full-width batch profile, queueing is a minor term.** `queue_wait_ms` stays ≤72 ms even at 128 (vs. multi-second slot-waits when the profile was capped at 32 — next section).

`prefill_ms` averages stay at 1.0–1.5 ms at every level — but remember this is cache-hit, suffix-only prefill (see Test conditions).

## Where burst TTFT goes: ramp serialization, not GPU

At 128 concurrency the numbers look paradoxical: TTFT ~334 ms, yet `queue_wait` is only ~72 ms and per-stream RTF is 0.84 (the GPU keeps up). A per-session lifecycle decomposition of a live 128-burst (`workspace/ttft_ramp_decompose.py`, all 128 sessions captured, server epoch timestamps from the done-event meta) shows where the rest goes:

| Stage | avg | p50 | p90 | max |
|---|---|---|---|---|
| request received → session created | ~0 | 0 | 0 | 0 |
| created → text enqueued | 19.7 ms | 8.0 | 70.3 | 78 |
| enqueued → dequeued (`queue_wait`) | 93.6 ms | 103 | 168 | 181 |
| **dequeued → prefill start (scheduling gap)** | **78.7 ms** | 66.5 | 130 | 200 |
| prefill compute | 1.4 ms | 1.0 | 1.0 | 19 |
| **prefill end → first raw audio (first-step wait)** | **89.2 ms** | 86 | 102 | 183 |
| first raw → client receive | 0.1 ms | 0.1 | 0.5 | 0.6 |
| client TTFT (this run) | 381.7 ms | 417.5 | 422.6 | 423.1 |

The mechanism: the engine loop alternates *drain inbox → schedule prefills → run one decode step for the whole active batch*. Under a burst, prefill admission (first session enters at 6 ms, the last at 376 ms; ~54 sessions/100 ms mid-ramp) is interleaved with decode steps of the **growing** batch, whose cost rises from ~13 ms to ~67 ms as the batch widens — later sessions' prefills wait behind those increasingly expensive steps, and even a prefilled session then waits ~one step (~86 ms median) for its first decode slot. Actual GPU compute per admission is trivial (prefill 1.4 ms).

RTF 0.84 and 334 ms TTFT therefore describe different phases: RTF is steady-state decode throughput; burst TTFT is the transient cost of ramping 0→128 sessions through a single admission loop. **Optimization leads**: (a) batched prefill — admitting waiting sessions in one GPU call instead of serially between decode steps could cut the ~370 ms admission ramp toward its ~100 ms theoretical floor (batched prefills + one full-width step); (b) ramp-aware scheduling — front-loading prefills during a burst instead of running partial-width decode steps that will be repeated anyway.

## Batch profile as a variable (not a ceiling)

The TRT engine's `max_batch_size` is **compiled into the plan** and bounds how many sessions can execute in one forward pass. Sessions beyond it are admitted (`max_sessions`) but wait for a free execution slot — for their predecessors' *entire utterances*, not just one step.

An earlier dataset collected on this same card with **batch=32** (pre-optimization engine code, cp=fp32, dual-resident services — conditions differ in more than the profile, so treat this as an illustration of the mechanism, not a controlled comparison) showed exactly that failure mode: at 128 concurrent requests, sessions executed in waves of 32, `active_sessions` stepped down 128→96→64→32 as each wave finished, decode step plateaued at the cap, and **TTFT averaged ~6.5 s** — ~20× worse than the ~334 ms measured here at the same concurrency. The batch=32 profile itself was an artifact of the broken auto-sizing (fix #4 above), not a hardware limit.

Sizing guidance now ships in `scripts/python/suggest_engine_profile.py` (export-aware; ~160 MiB per lane + ~5 GiB fixed for this model): a 32 GiB card fits batch=128; ~19 GiB fits 64; ~13.5 GiB fits 32. Pick the profile to cover your expected peak concurrency — oversubscribing the batch width is what creates the TTFT cliff.

## Optimization impact (dev-time reference)

Measured during development at batch=128 before/after the CP-KV + CUDA-graph + arena optimizations (pre-optimization raw data not retained; summary numbers only):

| | pre-opt | post-opt |
|---|---|---|
| decode step @128 | 119.8 ms (RTF 1.5 — **not** real-time) | 67.5 ms (RTF 0.84) |
| decode step @64 | 63.6 ms | 36.8 ms |
| decode step @1 | 11.3 ms | 12.7 ms (+1.4 ms; fixed per-step overhead of the CP-KV/CUDA-graph path) |
| TTFT avg @128 | ~371 ms | ~334 ms |

The optimizations trade ~1.4 ms/step of single-stream latency for a ~44% per-step reduction at full batch width — which is what moved 128-stream serving from below-real-time to real-time on this card.

## Known limitations / leads for future engine work

- **Fixed text/speaker — every prefill number is cache-hit prefill.** Cold-prefix (cache-miss) prefill measured ~30 ms vs ~1 ms in spot checks; cache-miss behavior under concurrent load is uncharacterized. Text-length, speaker-variety, and cache-miss sweeps are the natural follow-up.
- **Burst arrival is the worst case.** The concurrency test fires all N sessions simultaneously; staggered real-world arrivals would see lower per-request TTFT at the same steady-state concurrency.
- **Localhost only.** Network RTT adds directly to connection cost and TTFT; the ~10 ms gRPC connection delta scales with RTT.
- **Single-stream regression from batching optimizations.** The +1.4 ms/step cost at batch=1 is real; a future adaptive path could skip the arena/graph machinery below a batch-width threshold.
- **`serving_endpoints.py` still doesn't use the `qwen3tts` client SDK** — migrating it is tracked as separate follow-up work; this benchmark's harness (`perf_matrix_sdk.py`) is SDK-based.

## Reproducing

```bash
bash tools/validation/run_perf_matrix.sh
python tools/validation/summarize_perf_matrix.py workspace/perf_matrix/<run_id>
```

See [`serving_performance_benchmark_data/README.md`](serving_performance_benchmark_data/README.md) for environment variable overrides and the full raw/summary CSVs.
