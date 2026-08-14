"""Command-line parsing for deterministic leading-prefix regression sweeps."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .models import (
    DEFAULT_BODY_TEXT,
    DEFAULT_LEADING_PREFIX,
    ChunkPattern,
    SuspectThresholds,
    SweepConfig,
    build_text_packets,
)
from .runner import run_sweep

CLI_DESCRIPTION = """Run a deterministic leading-prefix TTS regression sweep.

The default case is the exact problematic input described in the v0.1.2a8
investigation: one U+0020 followed by
``好啦,我已经把可乐从厨房的冰箱里递送到当前房间``. The three packet
patterns intentionally exercise different contracts:

* ``whole-leading`` sends the prefix and body in one ``send_text`` call;
* ``split-leading`` sends the prefix alone, waits, then sends the body;
* ``clean`` sends only the body.

Session IDs are deterministic and do not contain a timestamp. Use the same
``--sid-prefix`` and ``--start`` for before/after arms so the engine derives
the same sampling seeds. WAVs and JSON sidecars are written for every trial.

Optional FunASR scoring is an automatic screen, not human ground truth. This
tool therefore reports only an "ASR-supported suspect" rate. Blind listening
is still required before calling a sample a hallucination.
"""


def parse_metadata(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or not key:
            raise ValueError(f"metadata must be KEY=VALUE, got {raw!r}")
        parsed[key] = value
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument("--endpoint", default="localhost:50051")
    parser.add_argument("--transport", default="engine-grpc")
    parser.add_argument("--speaker", default="serena")
    parser.add_argument("--input-mode", default="token")
    parser.add_argument("--group-policy", default="auto")
    parser.add_argument(
        "--pattern",
        choices=[item.value for item in ChunkPattern],
        default=ChunkPattern.WHOLE_LEADING.value,
    )
    parser.add_argument(
        "--text", default=DEFAULT_BODY_TEXT, help="spoken body; never stripped"
    )
    parser.add_argument(
        "--leading-prefix",
        default=DEFAULT_LEADING_PREFIX,
        help="prefix prepended/split by leading patterns; never stripped",
    )
    parser.add_argument("--split-delay-ms", type=float, default=50.0)
    parser.add_argument("-n", "--num-trials", type=int, default=10)
    parser.add_argument(
        "--start", type=int, default=1, help="first deterministic SID index"
    )
    parser.add_argument("--sid-prefix", default="leading-prefix")
    parser.add_argument("--arm", default="unspecified")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--outdir", "--output-dir", type=Path, default=None)
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--asr-url", default="", help="optional FunASR typed-v1 WebSocket URL"
    )
    parser.add_argument("--funasr-client-src", type=Path, default=None)
    parser.add_argument("--asr-language", default="中文")
    parser.add_argument("--asr-chunk-ms", type=int, default=960)
    parser.add_argument("--asr-max-duration-s", type=float, default=30.0)
    parser.add_argument("--duration-threshold-s", type=float, default=30.0)
    parser.add_argument("--cer-threshold", type=float, default=0.30)
    parser.add_argument("--insertion-threshold", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def _config_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> SweepConfig:
    if args.num_trials <= 0:
        parser.error("--num-trials must be positive")
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.timeout <= 0 or args.asr_chunk_ms <= 0:
        parser.error("timeouts and ASR chunk size must be positive")
    if args.duration_threshold_s <= 0 or args.asr_max_duration_s <= 0:
        parser.error("duration thresholds must be positive")
    if args.cer_threshold < 0 or args.insertion_threshold < 0:
        parser.error("CER/insertion thresholds must be non-negative")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between zero and one")

    pattern = ChunkPattern(args.pattern)
    try:
        build_text_packets(
            args.text,
            leading_prefix=args.leading_prefix,
            pattern=pattern,
            split_delay_ms=args.split_delay_ms,
        )
        metadata = parse_metadata(args.metadata)
    except ValueError as exc:
        parser.error(str(exc))

    return SweepConfig(
        endpoint=args.endpoint,
        transport=args.transport,
        speaker=args.speaker,
        input_mode=args.input_mode,
        group_policy=args.group_policy,
        pattern=pattern,
        body_text=args.text,
        leading_prefix=args.leading_prefix,
        split_delay_ms=args.split_delay_ms,
        num_trials=args.num_trials,
        start=args.start,
        sid_prefix=args.sid_prefix,
        arm=args.arm,
        timeout=args.timeout,
        output_dir=args.outdir,
        metadata=metadata,
        asr_url=args.asr_url,
        funasr_client_src=args.funasr_client_src,
        asr_language=args.asr_language,
        asr_chunk_ms=args.asr_chunk_ms,
        asr_max_duration_s=args.asr_max_duration_s,
        thresholds=SuspectThresholds(
            duration_s=args.duration_threshold_s,
            cer=args.cer_threshold,
            insertions=args.insertion_threshold,
        ),
        confidence=args.confidence,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    config = _config_from_args(parser, parser.parse_args(argv))
    return run_sweep(config)
