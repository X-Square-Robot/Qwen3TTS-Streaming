#!/usr/bin/env python3
"""Manual E2E benchmark for the standalone TTS engine.

This is the CLI counterpart to ``tests/e2e/test_engine_standalone.py``.
The pytest file contains assertions; this tool generates WAV samples and
prints benchmark summaries.

Usage:
  python -m engine.server --config engine.yaml
  python tests/tools/engine_standalone_benchmark.py --host localhost --port 50051
  python tests/tools/engine_standalone_benchmark.py --concurrency 1,2,4,8
  python tests/tools/engine_standalone_benchmark.py --stress-concurrency 8 --stress-rounds 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

REPO_ROOT = bootstrap_tool_imports()

from tests.support.engine_standalone import (
    GRPC_HOST,
    GRPC_PORT,
    OUTPUT_DIR,
    TTSResult,
    _check_server,
    _get_capabilities,
    run_badcases,
    run_concurrent,
    run_custom_voice_instruct,
    run_long_text,
    run_single_smoke,
    run_streaming_text,
    stress_concurrent,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual E2E benchmark for standalone TTS engine (gRPC)"
    )
    parser.add_argument("--host", default=GRPC_HOST)
    parser.add_argument("--port", type=int, default=GRPC_PORT)
    parser.add_argument(
        "--concurrency",
        default="1,2,4",
        help="Concurrency levels, comma-separated (default: 1,2,4)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory for WAV files",
    )
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument(
        "--skip-custom-instruct",
        action="store_true",
        help="Skip CustomVoice + instruct tests",
    )
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-concurrent", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--skip-badcase", action="store_true")
    parser.add_argument(
        "--stress-concurrency",
        type=int,
        default=0,
        help="Run repeated concurrent stress rounds at this concurrency; 0 disables",
    )
    parser.add_argument(
        "--stress-rounds",
        type=int,
        default=0,
        help="Measured stress rounds to run when --stress-concurrency > 0",
    )
    parser.add_argument(
        "--stress-warmup-rounds",
        type=int,
        default=1,
        help="Warmup rounds before measured stress rounds",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]

    print("=" * 60)
    print("  TTS Engine Standalone E2E Benchmark")
    print("=" * 60)
    print(f"  Engine:      {args.host}:{args.port}")
    print(f"  Concurrency: {concurrency_levels}")
    if args.stress_concurrency > 0:
        print(
            f"  Stress:      concurrency={args.stress_concurrency}, "
            f"warmup={args.stress_warmup_rounds}, rounds={args.stress_rounds}"
        )
    print(f"  Output:      {output_dir}")

    if not _check_server(args.host, args.port):
        print(f"\nERROR: Engine not reachable at {args.host}:{args.port}")
        print("Start the engine first:")
        print("  python -m engine.server --config engine.yaml")
        return 1
    print("  Server:      READY")

    cap = _get_capabilities(args.host, args.port)
    print(f"  Variant:     {cap['variant'] or 'unknown'}")
    print(f"  Model Type:  {cap['loaded_model_type'] or 'unknown'}")
    print(f"  Ref Audio:   {cap['ref_audio_available']}")
    if cap["ref_audio_reason"]:
        print(f"  Ref Reason:  {cap['ref_audio_reason']}")

    all_results: dict[str, list[TTSResult]] = {}

    if not args.skip_single:
        result = run_single_smoke(args.host, args.port, output_dir)
        all_results["single"] = [result]

    if not args.skip_streaming:
        result = run_streaming_text(args.host, args.port, output_dir)
        all_results["streaming"] = [result]

    if not args.skip_custom_instruct:
        instruct_results = run_custom_voice_instruct(args.host, args.port, output_dir)
        if instruct_results:
            all_results["custom_instruct"] = instruct_results

    if not args.skip_concurrent:
        for level in concurrency_levels:
            results = run_concurrent(args.host, args.port, level, output_dir)
            all_results[f"concurrent_x{level}"] = results

    if not args.skip_long:
        long_results = run_long_text(args.host, args.port, output_dir)
        all_results["long_text"] = long_results

    if not args.skip_badcase:
        run_badcases(args.host, args.port, output_dir)

    if args.stress_concurrency > 0 and args.stress_rounds > 0:
        stress_rounds = stress_concurrent(
            args.host,
            args.port,
            concurrency=args.stress_concurrency,
            rounds=args.stress_rounds,
            warmup_rounds=args.stress_warmup_rounds,
        )
        measured = stress_rounds[args.stress_warmup_rounds:]
        all_results[f"stress_x{args.stress_concurrency}"] = [
            result for round_results in measured for result in round_results
        ]

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    for name, results in all_results.items():
        ok = sum(1 for result in results if result.error is None)
        fail = sum(1 for result in results if result.error is not None)
        first_chunks = [result.first_chunk_ms for result in results if result.first_chunk_ms is not None]
        avg_first = statistics.mean(first_chunks) if first_chunks else 0
        print(f"  {name:20s}  OK={ok}  FAIL={fail}  avg_first_chunk={avg_first:.0f}ms")

    total_ok = sum(1 for results in all_results.values() for result in results if result.error is None)
    total_fail = sum(1 for results in all_results.values() for result in results if result.error is not None)
    print(f"\n  TOTAL: {total_ok} OK, {total_fail} FAILED")
    print(f"  Output: {output_dir.resolve()}")
    smoke_wav = output_dir / "test1_single_smoke.wav"
    if smoke_wav.exists():
        print(f"\n  Play audio: aplay {smoke_wav}")
        print(f"         or:  ffplay -autoexit {smoke_wav}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
