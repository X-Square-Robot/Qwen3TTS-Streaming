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

> **Dataset revision (2026-07-08 refresh).** The engine-side numbers below were re-collected on
> 2026-07-08 on a **full-bf16 rebuild** (repo HEAD `7db42d7`; the two commits since the `b80c38d`
> baseline, `9ad7cb0`/`7db42d7`, change only build-time precision defaults and docs, not engine
> runtime, so the runtime matches the baseline) — 3 trials reproducing the `ca6ddb9` dataset within
> noise. The `triton-grpc` numbers were **not re-run**: they are retained from the 2026-07-06
> dataset, whose engine core predates all of the round-3 commits. Since the Triton BLS path wraps
> the same engine core, its retained numbers **overstate** current Triton latency; treat them as
> the last measured reference, marked "(07-06)" in every table below.

## Test conditions

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB, driver 580.126.20 |
| Images | `qwen3-engine:25.10` (NGC 25.10 base, TensorRT 10.13.3.9), rebuilt 2026-07-08 (full-bf16); `qwen3tts-streaming:25.10` (Triton dataset from 2026-07-06) |
| TensorRT | 10.13.3.9 |
| Model | `custom-1.7b`, `custom_voice`, engine_mode=trt, **full bf16** (backbone/cp/code2wav) |
| Engine optimizations | CP-KV retained in the unrolled graph (~3× decode compute ↓), CUDA-graph decode replay, arena-ized batch KV gather, batched burst admission + per-slot Phase-1 tax elimination (`b80cf45`), c2w KV slot pooling + serving hot-path slimming (`2a9578b`), VAD-disabled chunk-path bypass (`a8e7274`), **post-review audit fix batch (`cc58c6a`) + batched `p3_launch` decode step (`8596996`) + session-open thread-offload trim (`b80c38d`)** — see `conditions.json` for commit hashes |
| Engine profile | **`max_batch_size=128`**, `max_input_len=128`, `max_seq_len=512` — auto-selected by export-aware sizing (raw capacity ~141 lanes on this card); **the batch profile is a test variable, not a hardware ceiling** (see below) |
| Scheduler | `max_sessions=128`, `max_queue_size=256`, MLFQ enabled |
| Request | fixed text/speaker(`Serena`)/`task_type=custom_voice`/`language=auto` for every request — isolates protocol/concurrency effects, and pins all measured rounds to a **~100% prefix-cache-hit regime** (verified: `cache_hit=True` on 100% of measured rows); every `prefill_ms` below is cache-hit (suffix-only) prefill |
| Isolation | **single service per run**: Triton stopped during engine trials, engine stopped during Triton trials (at batch=128 the engine alone holds ~29 GiB; the two cannot coexist on this card) |
| Warmup | 3 rounds (concurrency sweep) / 5 rounds (connection isolation); recorded, and pooled into the aggregates exactly as in the 2026-07-06 dataset (effect ≤1% at every level) |
| Trials | 3 independent full engine-side runs on the 2026-07-08 full-bf16 rebuild (levels 1–128 incl. level 8) + 3 Triton-side runs retained from 2026-07-06 (reproducibility check: `summary_by_trial.csv`) |
| Requests | 51,981 total (35,022 engine-side new + 16,959 Triton-side retained), **0 failures** |
| Hallucination gate | deterministic probe re-run **0/100** on this exact build immediately before data collection (duration min 9.36 s / median 10.08 s / max 10.88 s) |
| Harness | [`tools/validation/perf_matrix_sdk.py`](../../../tools/validation/perf_matrix_sdk.py), built on the `qwen3tts` client SDK |

> **Note.** This dataset is the **shipped default: full-bf16** (backbone/cp/code2wav all bf16). A `code2wav=fp16` variant was evaluated on 2026-07-08 — it speeds up single-stream decode (−21% at c1) but **regresses the b128 decode step (+8%, crossover ~c32)** because the c2w streaming states round-trip through a per-step bf16↔fp16 reformat that scales with batch. It is therefore a low-concurrency opt-in, **not** the default; the numbers below stand as the batch-128 reference (see [engine_overview §1.5](../architecture/engine_overview.md)).

### ⚠️ `prefill_ms` semantics changed in this dataset

As of the batched burst admission (`b80cf45`), `server_prefill_started/completed` bracket the
**whole batched admission pass** — slot allocation + batched prefix-cache restore + batched
suffix embed for *every* session admitted in that pass — not one session's own suffix compute.
Under burst load `prefill_ms` therefore reads as tens of ms per session (avg 60.4 ms at
concurrency 128) even though per-session GPU work is unchanged and total admission time went
*down*. At concurrency 1 the pass contains a single session and the value matches the old
semantics (~1 ms). The retained Triton rows predate the change and use the old per-session
semantics — **do not compare `prefill_ms` across the two datasets at load.**

## Fixes applied before measuring

Building this benchmark surfaced four server-side bugs, all fixed and landed before the 2026-07-06 dataset was collected (commits `a41f35d`, `d5741b6`, `3758ad7`):

1. **`engine_queue_wait_ms` was dead code** — the timing-accumulator lookup used a request object that never carries `session_config`, so the metric was never populated for any transport since it was added.
2. **Triton never computed timing metrics at all** — the BLS orchestrator never created a `ServerTimingAccumulator`/`OutputPipeline`, unlike the two native gateways; `queue_wait`/`prefill`/`cache_hit`/`total_latency` were never computed for Triton sessions. As a side effect of the fix, Triton's terminal event type was unified from `"end"` to `"done"`.
3. **`cache_hit` always reported `False`** — a string-literal mismatch (`"prefix_cache_hit"` was checked but never assigned); the cache fast path itself always worked. This dataset postdates the fix: `cache_hit_rate` is 1.0 across all measured rows, as expected for a fixed-key workload.
4. **Export-aware engine sizing never ran** — a stale helper path made every build silently fall back to a wildly conservative memory-tier table (which capped this 32 GiB card at batch=32); the recalibrated estimator now sizes this card at batch=128.

## Connection establishment time

Single-stream, 50 measured rounds per mode, pooled across 3 trials; `cold` = fresh `TTSClient.connect()` per round, `reuse` = one client across all rounds:

| Transport | cold TTFT avg | reuse TTFT avg | Δ (connection cost) |
|---|---|---|---|
| engine-grpc | 34.4 ms | 16.5 ms | **~17.9 ms** |
| engine-websocket | 15.7 ms | 15.7 ms | ~0 (noise) |
| triton-grpc (07-06) | 21.4 ms | 21.7 ms | ~0 (noise) |

Only `engine-grpc` shows a measurable connection-setup cost (~18 ms this run, gRPC channel + HTTP/2 handshake; this term is sensitive to host link state — a stable 33–36 ms cold across trials), which is why the client SDK keeps gRPC channels warm across sessions by default. Both WebSocket handshake and Triton's channel setup are noise-level on a local link; over a real network all of these scale with RTT.

## Single-request lifecycle breakdown

Single stream, warm engine, prefix-cache hit — the standard low-load profile:

| Stage | value |
|---|---|
| `queue_wait_ms` (software inbox wait) | 0.6 ms avg |
| `prefill_ms` (prefix-cache hit, suffix-only; single-session admission pass) | 0.9 ms avg |
| decode step (mean interval) | 12.6 ms → per-stream RTF ~0.16 |
| **server TTFT** (session create → first raw audio; same basis as the historical "13ms" claim) | **14.9 ± 0.3 ms** (min 14.4, p50 14.8, p99 15.9, n=50, measured 2026-07-08; matches the 07-07 avg 14.96 ± 0.28 within noise) |
| client TTFT (localhost, reused channel) | 16.1 ms p50 engine-grpc / 15.6 ms p50 engine-websocket (n=165 each) |
| total (full ~10 s utterance) | ~1.15 s |

A cold, cache-miss prefill costs far more than the 0.9 ms shown here (~30 ms measured in spot checks); see Known limitations.

## Protocol comparison

TTFT avg (ms) by protocol × concurrency, pooled across 3 trials each:

| Concurrency | engine-grpc | engine-websocket | triton-grpc (07-06) |
|---|---|---|---|
| 1 | 17.1 | 15.6 | 22.4 |
| 8 | 41.6 | **34.4** | — |
| 16 | 55.3 | **49.9** | 71.3 |
| 32 | 88.6 | **76.0** | 112.6 |
| 64 | 142.4 | **128.6** | 185.7 |
| 128 | 275.2 | **241.6** | 309.3 |

`engine-websocket` is the fastest under load at every level; the gRPC gap at 128 is ~12% on the
average. The tail gap is larger
still (p99 at 128: 494 ms grpc vs 341 ms ws). The Triton column is the retained 2026-07-06 dataset
(pre-optimization engine core; its p99 at 128 was 672 ms) — not directly comparable, kept as the
last measured reference. All three share the engine core, so scaling *shape* is identical; the
differences are transport/gateway overhead.

## Concurrency scaling: where the time actually goes

`engine-grpc`, pooled avg values (3 trials, 20 rounds/level):

| Concurrency | TTFT | `queue_wait_ms` | decode step (mean) | per-stream RTF | `batch_size_seen` |
|---|---|---|---|---|---|
| 1 | 17.1 ms | 0.86 ms | 12.6 ms | 0.16 | 1 |
| 8 | 41.6 ms | 5.6 ms | 13.9 ms | 0.17 | 8 |
| 16 | 55.3 ms | 10.9 ms | 15.1 ms | 0.19 | 16 |
| 32 | 88.6 ms | 25.9 ms | 17.7 ms | 0.22 | 32 |
| 64 | 142.4 ms | 28.0 ms | 24.5 ms | 0.31 | 64 |
| 128 | 275.2 ms | 61.1 ms | 42.1 ms | **0.53** | **128** |

(each decode step yields 80 ms of audio; per-stream RTF = step time / 80 ms)

Three observations:

1. **All 128 streams genuinely decode together** — `batch_size_seen` reaches 128, and TTFT grows smoothly with no cliff. Verified live: polling `/health` during a 128-burst shows `active_sessions=128` with all sessions admitted at once.
2. **Full-width decode now has real headroom.** The 2026-07-06 dataset measured RTF 0.84 at 128-wide ("still real-time, ~16% headroom, near saturation"); the 2026-07-07 round brought it to 0.59. After the round-3 batched `p3_launch` (`8596996`), the 128-wide step is 42.1 ms per 80 ms of audio — **RTF 0.53, ~47% headroom**. The capacity statement for this GPU/model is now: **128 concurrent real-time streams with margin**; the binding limit at 128 is the compiled batch profile (and the ~141-lane memory capacity), no longer decode throughput.
3. **Queueing stays a minor term.** `queue_wait_ms` is ≤61 ms avg even at 128 (vs. multi-second slot-waits when the profile was capped at 32 — next section).

`prefill_ms` grows with concurrency in this dataset (0.8 → 60.4 ms avg from level 1 to 128) — this is the **semantics change** (the value now spans the whole batched admission pass, see Test conditions), not a prefill regression; per-session suffix prefill work is unchanged and the total admission ramp got faster (next section).

## Where burst TTFT goes: admission is now batched; arrival spread dominates

The 2026-07-06 dataset showed 128-burst TTFT (~334 ms avg) was dominated by **ramp serialization**: sessions were prefilled one at a time between decode steps of the growing batch (last admission at ~376 ms; scheduling gap avg 78.7 ms; per-session first-step wait avg 89.2 ms). That mechanism is what `b80cf45` (batched admission) removed. Re-running the same per-session lifecycle decomposition (`workspace/ttft_ramp_decompose.py`, 128/128 sessions captured, server epoch timestamps from the done-event meta; two runs, client TTFT avg 285.8/287.9 ms; **measured 2026-07-07 on the round-2 build** — headline burst TTFT has since settled at ~275 ms grpc / ~242 ms ws, but the stage-level mechanism below is unchanged) shows:

| Stage | avg | p50 | p90 | max |
|---|---|---|---|---|
| request received → session created | ~0 | 0 | 0 | 0 |
| created → text enqueued | 13.2 ms | 7.0 | 25.3 | 27 |
| enqueued → dequeued (`queue_wait`) | 80.3 ms | 87.5 | 150.3 | 163 |
| dequeued → prefill start (scheduling gap) | **13.1 ms** (was 78.7) | 16.0 | 19.0 | 54 |
| prefill (batched admission pass, new semantics) | 51.6 ms | 23.0 | 107.0 | 108 |
| prefill end → first raw audio (first-step wait) | **58.9 ms** (was 89.2) | 53.0 | 74.0 | 75 |
| first raw → client receive | 5.8 ms | 5.0 | 6.2 | 101.1 |
| client TTFT (this run) | 285.8 ms | 302.7 | 316.5 | 317.9 |

Admission now lands in a few wide passes instead of 128 serial ones: the first session starts prefill at ~5 ms after burst start and the *last* admission pass starts at ~251 ms (was ~376 ms serial), with a single pass able to admit ~120 sessions in ~35 ms of engine time. What remains of burst TTFT is (a) **client/gateway arrival spread** — requests of a "simultaneous" 128-burst arrive at the server over ~163 ms on a shared channel, so late arrivals mechanically inherit that offset; (b) the queue-drain cadence between engine loop iterations (`queue_wait` p50 87.5 ms); and (c) one batched pass + first-step wait. GPU compute is still not the limiter.

**Remaining optimization leads**: the arrival/enqueue spread is client- and gateway-side (multi-channel clients measured materially lower TTFT in dev runs); on the engine side, draining the inbox more aggressively mid-ramp and shaving the first-step wait would attack the two biggest surviving terms.

## Batch profile as a variable (not a ceiling)

The TRT engine's `max_batch_size` is **compiled into the plan** and bounds how many sessions can execute in one forward pass. Sessions beyond it are admitted (`max_sessions`) but wait for a free execution slot — for their predecessors' *entire utterances*, not just one step.

An earlier dataset collected on this same card with **batch=32** (pre-optimization engine code, cp=fp32, dual-resident services — conditions differ in more than the profile, so treat this as an illustration of the mechanism, not a controlled comparison) showed exactly that failure mode: at 128 concurrent requests, sessions executed in waves of 32, `active_sessions` stepped down 128→96→64→32 as each wave finished, decode step plateaued at the cap, and **TTFT averaged ~6.5 s** — ~24× worse than the ~275 ms measured here at the same concurrency. The batch=32 profile itself was an artifact of the broken auto-sizing (fix #4 above), not a hardware limit.

Sizing guidance now ships in `scripts/python/suggest_engine_profile.py` (export-aware; ~160 MiB per lane + ~5 GiB fixed for this model): a 32 GiB card fits batch=128; ~19 GiB fits 64; ~13.5 GiB fits 32. Pick the profile to cover your expected peak concurrency — oversubscribing the batch width is what creates the TTFT cliff.

## Optimization impact (dev-time reference)

Three optimization rounds measured at batch=128 on this card (raw data: 2026-07-06 dataset for "round 1", the 2026-07-07 dataset for "round 2", this dataset for "round 3"; pre-opt raw data not retained, summary numbers only):

| | pre-opt | round 1 (07-06: CP-KV + CUDA-graph + KV arena) | round 2 (07-07: batched admission + Phase-1 tax + hot path + VAD bypass) | round 3 (07-08: audit fix batch + batched p3_launch) |
|---|---|---|---|---|
| decode step @128 | 119.8 ms (RTF 1.5 — **not** real-time) | 67.5 ms (RTF 0.84) | 47.4 ms (RTF 0.59) | **42.1 ms (RTF 0.53)** |
| decode step @64 | 63.6 ms | 36.8 ms | 27.1 ms | **24.5 ms** |
| decode step @1 | 11.3 ms | 12.7 ms | 12.8 ms | 12.6 ms (+1.3 ms vs pre-opt; fixed per-step overhead of the CP-KV/CUDA-graph path) |
| TTFT avg @128 (engine-grpc) | ~371 ms | ~334 ms | ~275 ms | **~275 ms** |
| server total latency p50 @128 | — | 6333 ms | 4453 ms (−30%) | **3988 ms** (−11% vs round 2, −37% vs round 1) |

Round 1 traded ~1.4 ms/step of single-stream latency for a ~44% per-step cut at full width — moving 128-stream serving from below-real-time to real-time. Round 2 cut another ~30% per step at width (Phase-1 per-slot CPU tax + c2w pooling), removed the serial admission ramp (TTFT −18%), and slimmed the serving hot path (async logging, orjson, gRPC chunk coalescing, VAD-path bypass). Audio output was verified byte-identical across round 2's data-movement changes (8 deterministic seeds, sha256). Round 3 is the post-review audit fix batch (`cc58c6a`) plus the bit-exact batched `p3_launch` (`8596996`), cutting another ~11% per decode step at width and carrying `server_total` down with it; cumulatively vs the 2026-07-06 dataset that is **TTFT −18%, decode step −38%, server total −37%** at 128-wide. The hallucination probe stayed 0/100 before every round's data collection.

## Known limitations / leads for future engine work

- **Fixed text/speaker — every prefill number is cache-hit prefill.** Cold-prefix (cache-miss) prefill measured ~30 ms vs ~1 ms in spot checks; cache-miss behavior under concurrent load is uncharacterized. Text-length, speaker-variety, and cache-miss sweeps are the natural follow-up.
- **Triton numbers are stale (2026-07-06).** The Triton BLS path wraps the same engine core, so the engine-side gains (`b80cf45` through `8596996`) would carry over, but it was not re-benchmarked; its gateway-side behavior under the new admission pattern is unmeasured.
- **Burst arrival is the worst case.** The concurrency test fires all N sessions simultaneously; staggered real-world arrivals would see lower per-request TTFT at the same steady-state concurrency. On top of that, the measured burst TTFT includes ~163 ms of client-side arrival spread on a shared channel (see the ramp decomposition).
- **Localhost only.** Network RTT adds directly to connection cost and TTFT; the ~18 ms gRPC connection delta scales with RTT.
- **Single-stream regression from batching optimizations.** The +1.3 ms/step cost at batch=1 (11.3 → 12.6 ms, unchanged across rounds 2–3) is real; a future adaptive path could skip the arena/graph machinery below a batch-width threshold.
- **`serving_endpoints.py` still doesn't use the `qwen3tts` client SDK** — migrating it is tracked as separate follow-up work; this benchmark's harness (`perf_matrix_sdk.py`) is SDK-based.

## Reproducing

```bash
bash tools/validation/run_perf_matrix.sh
python tools/validation/summarize_perf_matrix.py workspace/perf_matrix/<run_id>
```

See [`serving_performance_benchmark_data/README.md`](serving_performance_benchmark_data/README.md) for environment variable overrides and the full raw/summary CSVs.
