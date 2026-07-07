#!/usr/bin/env python3
"""SDK-based serving performance benchmark.

Unlike serving_endpoints.py (which implements its own gRPC/WebSocket/Triton
protocol clients from scratch), this drives requests through the actual
``qwen3tts`` client SDK -- the same package real integrators install -- and
parses server timing via ``qwen3tts.diagnostics`` instead of hand-rolled
event parsing. This is the intended foundation for perf benchmarking: it
measures what SDK users actually experience, and it can't drift from the
SDK's own event/meta parsing since it *is* the SDK's parsing.

Runs one (transport, concurrency level) cell: `--warmup-rounds` unmeasured
rounds followed by `--rounds` measured rounds, each round firing
`--concurrency` concurrent streaming sessions. Emits a flat JSON array of
per-request raw records (one row per synthesis request), consumed by
tools/validation/summarize_perf_matrix.py.

Usage:
    PYTHONPATH=client/src python tools/validation/perf_matrix_sdk.py \\
        --endpoint localhost:50051 --transport engine-grpc \\
        --concurrency 16 --rounds 20 --warmup-rounds 3 --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_validation_dir = str(Path(__file__).resolve().parent)
if _validation_dir not in sys.path:
    sys.path.insert(0, _validation_dir)
from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()

from qwen3tts import SessionStartRequest, SynthesisConfig, TTSClient  # noqa: E402
from qwen3tts.diagnostics import SessionDiagnostics  # noqa: E402
from qwen3tts_protocol import AudioChunk, StreamEvent  # noqa: E402

TEST_TEXT = "你好，欢迎使用 Qwen3-TTS 语音合成系统，这是一段用于压测的示例文本。"


def _run_one(
    client: TTSClient, spec: SynthesisConfig, text: str, session_id: str
) -> dict[str, Any]:
    t0 = time.perf_counter()
    error: str | None = None
    chunk_ts: list[float] = []
    messages: list[Any] = []
    try:
        session = client.open_stream(
            SessionStartRequest(session_id=session_id, config=spec)
        )
        session.send_text(text)
        session.end()
        for message in session.iter_messages():
            messages.append(message)
            if isinstance(message, AudioChunk):
                chunk_ts.append(time.perf_counter())
            if isinstance(message, StreamEvent) and message.type == "error":
                error = message.message or "stream error"
    except Exception as exc:  # noqa: BLE001 -- one lane's failure must not abort the round
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()

    diag = SessionDiagnostics.from_messages(messages)
    timing = diag.timing
    batch = diag.summary().get("batch") or {}
    audio_bytes = sum(len(m.pcm_bytes) for m in messages if isinstance(m, AudioChunk))
    decode_intervals_ms = [
        (chunk_ts[i] - chunk_ts[i - 1]) * 1000.0 for i in range(1, len(chunk_ts))
    ]

    return {
        "session_id": session_id,
        "ok": error is None,
        "error": error,
        "ttft_ms": (chunk_ts[0] - t0) * 1000.0 if chunk_ts else None,
        "total_ms": (t_end - t0) * 1000.0,
        "queue_wait_ms": timing.server_engine_queue_wait_ms,
        "prefill_ms": timing.server_engine_prefill_ms,
        "server_total_latency_ms": timing.server_total_latency_ms,
        "cache_hit": timing.server_cache_hit,
        "batch_size_seen": batch.get("max_batch_size_seen"),
        "batched": bool(batch.get("batched")) if batch else None,
        "audio_bytes": audio_bytes,
        "num_chunks": len(chunk_ts),
        "decode_step_mean_ms": (
            statistics.mean(decode_intervals_ms) if decode_intervals_ms else None
        ),
    }


def _run_round(
    client: TTSClient,
    spec: SynthesisConfig,
    text: str,
    *,
    level: int,
    phase: str,
    run_idx: int,
    transport: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=level) as executor:
        futures = {
            executor.submit(
                _run_one,
                client,
                spec,
                text,
                f"{transport}-c{level}-{phase}-{run_idx}-{idx}",
            ): idx
            for idx in range(level)
        }
        for future in as_completed(futures):
            idx = futures[future]
            record = future.result()
            record.update(
                {"level": level, "phase": phase, "run": run_idx, "lane": idx}
            )
            records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--transport",
        required=True,
        choices=["engine-grpc", "engine-websocket", "triton-grpc", "triton-http"],
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--task-type", default="custom_voice")
    parser.add_argument("--speaker", default="Serena")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--text", default=TEST_TEXT)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--connection-mode",
        choices=["reuse", "cold"],
        default="reuse",
        help=(
            "reuse: one TTSClient shared across every round/lane (steady-state). "
            "cold: a fresh TTSClient.connect() per round, sequential lanes only "
            "-- meant for --concurrency 1 connection-time isolation runs."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.connection_mode == "cold" and args.concurrency != 1:
        raise SystemExit(
            "--connection-mode cold only supports --concurrency 1 "
            "(each round needs a dedicated fresh connection to measure cold cost)"
        )

    spec = SynthesisConfig(
        task_type=args.task_type, speaker=args.speaker, language=args.language
    )

    all_records: list[dict[str, Any]] = []
    shared_client: TTSClient | None = None
    if args.connection_mode == "reuse":
        shared_client = TTSClient.connect(
            args.endpoint, transport=args.transport, timeout=args.timeout
        )

    total_rounds = args.warmup_rounds + args.rounds
    try:
        for run_idx in range(total_rounds):
            phase = "warmup" if run_idx < args.warmup_rounds else "measure"
            phase_idx = run_idx if phase == "warmup" else run_idx - args.warmup_rounds

            if args.connection_mode == "cold":
                client = TTSClient.connect(
                    args.endpoint, transport=args.transport, timeout=args.timeout
                )
            else:
                client = shared_client

            records = _run_round(
                client,
                spec,
                args.text,
                level=args.concurrency,
                phase=phase,
                run_idx=phase_idx,
                transport=args.transport,
            )
            all_records.extend(records)

            if args.connection_mode == "cold":
                client.close()
    finally:
        if shared_client is not None:
            shared_client.close()

    if args.json:
        print(json.dumps(all_records, ensure_ascii=False, indent=2))
    else:
        ok = sum(1 for r in all_records if r["ok"])
        print(f"{args.transport} c{args.concurrency}: {ok}/{len(all_records)} ok")

    return 0 if all(r["ok"] for r in all_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
