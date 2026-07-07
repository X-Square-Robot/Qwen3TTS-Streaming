**English** | [中文](README.zh-CN.md)

# Serving performance benchmark — raw data

Raw data backing [`../serving_performance_benchmark.md`](../serving_performance_benchmark.md).

- `conditions.json` — hardware/software/model/scheduler configuration, optimization commits included in the engine under test, and trial design (see [`docs/user/benchmark_methodology.md`](../../../user/benchmark_methodology.md) for what a benchmark report must record).
- `raw_requests.csv` — one row per synthesis request from one representative engine trial + one representative Triton trial (16,959 requests). Cross-trial consistency is shown in `summary_by_trial.csv`.
- `summary.csv` — avg/p50/p90/p99/max per (protocol, concurrency level) and (protocol, connection mode), pooled across **all 6 trials** (3 engine-side + 3 Triton-side; 50,877 requests total, 0 failures).
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

Env overrides (`TARGETS`, `LEVELS`, `CONCURRENCY_SAMPLES`, `CONCURRENCY_WARMUP`, `CONN_SAMPLES`, `CONN_WARMUP`, endpoints) are documented in the header of `run_perf_matrix.sh`. This dataset used `CONCURRENCY_SAMPLES=20 CONCURRENCY_WARMUP=3 CONN_SAMPLES=50 CONN_WARMUP=5`, run 3 times per serving side.

## Reading the data

- `cache_hit` is `True` on 100% of measured rows — expected, since every request shares one prefix-cache key (fixed text/speaker). All `prefill_ms` values are therefore cache-hit (suffix-only) prefill; cold-prefix prefill is ~30× more expensive (see the report's Known limitations).
- `kind=concurrency` rows come from the concurrency sweep; `kind=connection` rows from the single-stream connection-mode isolation (`conn_mode` = `reuse`/`cold`).
- `phase=warmup` rows are recorded but excluded from every aggregate.
