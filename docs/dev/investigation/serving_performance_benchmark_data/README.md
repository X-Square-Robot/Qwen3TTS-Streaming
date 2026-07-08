**English** | [中文](README.zh-CN.md)

# Serving performance benchmark — raw data

Raw data backing [`../serving_performance_benchmark.md`](../serving_performance_benchmark.md).

**Provenance (mixed dataset):** engine-side rows (`engine-grpc`/`engine-websocket`) were
**re-collected 2026-07-08 on a full-bf16 rebuild** (repo HEAD `7db42d7`, runtime identical to the
`b80c38d` baseline) — 3 trials that reproduce the `ca6ddb9` dataset within noise; `triton-grpc` rows
are **retained from the 2026-07-06 dataset**, whose engine core predates all of the round-3 commits —
they overstate current Triton latency and are kept as the last measured reference.

- `conditions.json` — hardware/software/model/scheduler configuration, optimization commits included in the engine under test, dataset provenance, the `prefill_ms` semantics change, and trial design (see [`docs/user/benchmark_methodology.md`](../../../user/benchmark_methodology.md) for what a benchmark report must record).
- `raw_requests.csv` — one row per synthesis request from one representative engine trial (bf16refresh-engine-01, 2026-07-08) + one representative Triton trial (triton-01, 2026-07-06); 17,327 requests. Cross-trial consistency is shown in `summary_by_trial.csv`.
- `summary.csv` — avg/p50/p90/p99/max per (protocol, concurrency level) and (protocol, connection mode), pooled across **all 6 trials** (3 engine-side 2026-07-08 + 3 Triton-side 2026-07-06; 51,981 requests total, 0 failures).
- `summary_by_trial.csv` — the same per-trial summaries side by side (6 independent runs), to show the numbers are stable run over run rather than a one-off measurement.

## Reproducing

Requires a running engine (`engine-grpc`/`engine-websocket`) **or** Triton (`triton-grpc`) — at `max_batch_size=128` on a 32 GiB card the two cannot run simultaneously; benchmark one side at a time with the other stopped (see the report's Isolation row). Deployment commands are in the repo root [CLAUDE.md](../../../../CLAUDE.md); the `qwen3tts` client SDK must be on `PYTHONPATH` (`client/src`).

```bash
# Engine side (Triton stopped):
TARGETS="engine-grpc,engine-websocket" bash tools/validation/run_perf_matrix.sh
# Triton side (engine stopped):
TARGETS="triton-grpc" bash tools/validation/run_perf_matrix.sh
# writes workspace/perf_matrix/<UTC timestamp>/*.json

# Flatten into raw_requests.csv + summary.csv:
python tools/validation/summarize_perf_matrix.py workspace/perf_matrix/<run_id>
```

Env overrides (`TARGETS`, `LEVELS`, `CONCURRENCY_SAMPLES`, `CONCURRENCY_WARMUP`, `CONN_SAMPLES`, `CONN_WARMUP`, endpoints) are documented in the header of `run_perf_matrix.sh`. The engine side of this dataset used `LEVELS=1,8,16,32,64,128 CONCURRENCY_SAMPLES=20 CONCURRENCY_WARMUP=3 CONN_SAMPLES=50 CONN_WARMUP=5`, run 3 times (the Triton side used the same sampling at levels 1,16,32,64,128).

## Reading the data

- **`prefill_ms` semantics differ between the two sides.** Engine rows (2026-07-08): the value spans the whole batched admission pass (see the report's Test conditions callout), so it grows with concurrency by construction. Triton rows (2026-07-06): old per-session semantics. Do not compare the column across sides at load.
- `cache_hit` is `True` on 100% of measured rows — expected, since every request shares one prefix-cache key (fixed text/speaker). All `prefill_ms` values are therefore cache-hit (suffix-only) prefill; cold-prefix prefill is ~30× more expensive (see the report's Known limitations).
- `kind=concurrency` rows come from the concurrency sweep; `kind=connection` rows from the single-stream connection-mode isolation (`conn_mode` = `reuse`/`cold`).
- `phase=warmup` rows are recorded and **pooled into the summary aggregates** (same aggregation as the 2026-07-06 dataset, keeping cross-dataset deltas apples-to-apples; the effect is ≤1% at every level). Filter `raw_requests.csv` on `phase=measure` to exclude them.
