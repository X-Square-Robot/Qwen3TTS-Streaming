#!/usr/bin/env python3
"""Flatten run_perf_matrix.sh raw JSON output into two CSVs.

Reads every ``*.json`` file produced by ``run_perf_matrix.sh`` in a run
directory (each file is the flat JSON array emitted by one
``perf_matrix_sdk.py`` invocation -- one record per synthesis request) and
writes:

  raw_requests.csv  -- one row per individual synthesis request (the
                        per-request "raw data" for the perf report)
  summary.csv       -- avg/p50/p90/p99/max + failures, grouped by
                        (protocol, concurrency level) and
                        (protocol, connection mode)

Usage:
    python tools/validation/summarize_perf_matrix.py workspace/perf_matrix/<run_id>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

CONCURRENCY_RE = re.compile(r"^(?P<target>[a-z-]+)_c(?P<level>\d+)\.json$")
CONN_MODE_RE = re.compile(r"^(?P<target>[a-z-]+)_conn-(?P<mode>reuse|cold)\.json$")

RAW_FIELDS = [
    "kind",
    "protocol",
    "level",
    "conn_mode",
    "phase",
    "run",
    "lane",
    "session_id",
    "ok",
    "error",
    "ttft_ms",
    "queue_wait_ms",
    "prefill_ms",
    "decode_step_mean_ms",
    "total_ms",
    "server_total_latency_ms",
    "cache_hit",
    "batch_size_seen",
    "batched",
    "num_chunks",
    "audio_bytes",
]

# Metrics summarized with avg/p50/p90/p99/max in summary.csv.
SUMMARY_METRICS = (
    "ttft_ms",
    "queue_wait_ms",
    "prefill_ms",
    "decode_step_mean_ms",
    "total_ms",
    "server_total_latency_ms",
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warning: skipping unreadable {path.name}: {exc}")
        return []
    if not isinstance(data, list):
        print(f"warning: unexpected JSON shape in {path.name}, skipped")
        return []
    return data


def _blank_row() -> dict[str, Any]:
    return {field: None for field in RAW_FIELDS}


def _rows_from_file(
    path: Path, target: str, *, level: int | None, conn_mode: str | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(path):
        row = _blank_row()
        row["kind"] = "connection" if conn_mode else "concurrency"
        row["protocol"] = target
        row["level"] = level if level is not None else record.get("level")
        row["conn_mode"] = conn_mode
        for key in (
            "phase",
            "run",
            "lane",
            "session_id",
            "ok",
            "error",
            "ttft_ms",
            "queue_wait_ms",
            "prefill_ms",
            "decode_step_mean_ms",
            "total_ms",
            "server_total_latency_ms",
            "cache_hit",
            "batch_size_seen",
            "batched",
            "num_chunks",
            "audio_bytes",
        ):
            row[key] = record.get(key)
        rows.append(row)
    return rows


def _percentile(sorted_values: list[float], pct: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = rank - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def _summarize(
    rows: list[dict[str, Any]], group_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        group_rows = groups[key]
        ok_rows = [r for r in group_rows if r.get("ok")]
        entry: dict[str, Any] = dict(zip(group_keys, key))
        entry["requests"] = len(group_rows)
        entry["failed"] = len(group_rows) - len(ok_rows)

        for metric in SUMMARY_METRICS:
            values = sorted(
                r[metric] for r in ok_rows if r.get(metric) is not None
            )
            if not values:
                continue
            entry[f"{metric}_avg"] = round(statistics.mean(values), 3)
            entry[f"{metric}_p50"] = round(_percentile(values, 0.50), 3)
            entry[f"{metric}_p90"] = round(_percentile(values, 0.90), 3)
            entry[f"{metric}_p99"] = round(_percentile(values, 0.99), 3)
            entry[f"{metric}_max"] = round(max(values), 3)

        batch_values = [
            r["batch_size_seen"] for r in ok_rows if r.get("batch_size_seen") is not None
        ]
        if batch_values:
            entry["batch_size_seen_mean"] = round(statistics.mean(batch_values), 2)
            entry["batch_size_seen_max"] = max(batch_values)

        cache_values = [r["cache_hit"] for r in ok_rows if r.get("cache_hit") is not None]
        if cache_values:
            entry["cache_hit_rate"] = round(
                sum(1 for v in cache_values if v) / len(cache_values), 3
            )

        out.append(entry)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    raw_rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        name = path.name
        m = CONCURRENCY_RE.match(name)
        if m:
            raw_rows.extend(
                _rows_from_file(
                    path, m["target"], level=int(m["level"]), conn_mode=None
                )
            )
            continue
        m = CONN_MODE_RE.match(name)
        if m:
            raw_rows.extend(
                _rows_from_file(path, m["target"], level=None, conn_mode=m["mode"])
            )
            continue
        print(f"note: unrecognized file, skipped: {name}")

    raw_path = run_dir / "raw_requests.csv"
    _write_csv(raw_path, raw_rows, RAW_FIELDS)
    print(f"wrote {len(raw_rows)} raw rows -> {raw_path}")

    concurrency_rows = [r for r in raw_rows if r["kind"] == "concurrency"]
    connection_rows = [r for r in raw_rows if r["kind"] == "connection"]
    concurrency_summary = _summarize(concurrency_rows, ("protocol", "level"))
    connection_summary = _summarize(connection_rows, ("protocol", "conn_mode"))

    identity_cols = ["protocol", "level", "conn_mode", "requests", "failed"]
    metric_cols = sorted(
        {
            key
            for row in (concurrency_summary + connection_summary)
            for key in row
            if key not in identity_cols
        }
    )
    summary_path = run_dir / "summary.csv"
    _write_csv(
        summary_path,
        concurrency_summary + connection_summary,
        identity_cols + metric_cols,
    )
    print(
        f"wrote {len(concurrency_summary) + len(connection_summary)} summary rows "
        f"-> {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
