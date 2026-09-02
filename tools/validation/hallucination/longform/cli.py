"""Argument parsing for the independent 0818 long-form comparison workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .collection import DEFAULT_SEEDS
from .commands import (
    command_collect_endpoint,
    command_collect_matched_endpoint,
    command_collect_matched_official,
    command_collect_official,
    command_diagnostic_report,
    command_finalize_labels,
    command_init,
    command_greedy_hashes,
    command_prepare_adjudication,
    command_prepare_review,
    command_prepare_second,
    command_report,
    command_rootcause_replay_endpoint,
    command_rootcause_replay_official,
    command_rootcause_score_asr,
    command_score,
    command_status,
)
from .models import ArmKind


DEFAULT_TEXT = Path("resources/dataset/badcase/verylong.txt")
DEFAULT_PACKAGE = Path("workspace/model_repository/tts_orchestrator/2")
DEFAULT_CHECKPOINT = Path("workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEFAULT_ASR_URL = "wss://infer.x2robot.com/infer/inf-dddq5qn77jrws5eu/v1/ws"
DEFAULT_FUNASR_WHEEL_SHA256 = (
    "da6ea124ee062b26336ffbbee0f358b9888588cd9091ebf036f11878916d1227"
)


def _output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Long-form TTS hallucination comparison and root-cause diagnostics. "
            "The ASR/telemetry replay path is independent of optional review "
            "workflows and of the short leading-prefix sweep."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="freeze text/model/code/ASR identities")
    _output_argument(init)
    init.add_argument("--repo-root", type=Path, default=Path.cwd())
    init.add_argument("--text-file", type=Path, default=DEFAULT_TEXT)
    init.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    init.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    init.add_argument("--asr-url", default=DEFAULT_ASR_URL)
    init.add_argument("--funasr-wheel", required=True, type=Path)
    init.add_argument("--funasr-wheel-sha256", default=DEFAULT_FUNASR_WHEEL_SHA256)
    init.add_argument("--seed", action="append", type=int, default=None)
    init.add_argument(
        "--config-file",
        action="append",
        type=Path,
        default=[
            Path("engine.yaml"),
            Path("workspace/model_repository/tts_orchestrator/config.pbtxt"),
            Path("infra/docker/Dockerfile.triton"),
            Path("infra/docker/Dockerfile.longform-eval"),
        ],
    )
    init.set_defaults(handler=command_init)

    endpoint = sub.add_parser(
        "collect-endpoint", help="collect current or frozen Triton arm"
    )
    _output_argument(endpoint)
    endpoint.add_argument(
        "--arm",
        required=True,
        choices=[ArmKind.CURRENT_HEAD.value, ArmKind.TRITON_0818.value],
    )
    endpoint.add_argument("--endpoint", required=True)
    endpoint.add_argument("--speaker", default="001")
    endpoint.add_argument("--language", default="auto")
    endpoint.add_argument("--timeout", type=float, default=1800.0)
    endpoint.add_argument("--model-name", default="tts_orchestrator")
    endpoint.add_argument("--model-version", default="2")
    endpoint.add_argument("--metadata", action="append", default=[])
    endpoint.set_defaults(handler=command_collect_endpoint)

    official = sub.add_parser(
        "collect-official", help="collect official PyTorch baseline"
    )
    _output_argument(official)
    official.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    official.add_argument("--speaker", default="001")
    official.add_argument("--language", default="Auto")
    official.add_argument("--device", default="cuda:0")
    official.add_argument("--dtype", default="bfloat16")
    official.add_argument("--attn-implementation", default="eager")
    official.add_argument("--sampling-base-seed", type=int, default=0)
    official.add_argument("--metadata", action="append", default=[])
    official.set_defaults(handler=command_collect_official)

    matched_endpoint = sub.add_parser(
        "collect-matched-endpoint",
        help="gate-protected replay of dynamically frozen Triton commit groups",
    )
    _output_argument(matched_endpoint)
    matched_endpoint.add_argument(
        "--mode", choices=("sample", "greedy"), default="sample"
    )
    matched_endpoint.add_argument(
        "--arm",
        required=True,
        choices=[ArmKind.CURRENT_HEAD.value, ArmKind.TRITON_0818.value],
    )
    matched_endpoint.add_argument("--endpoint", required=True)
    matched_endpoint.add_argument("--speaker", default="001")
    matched_endpoint.add_argument("--language", default="auto")
    matched_endpoint.add_argument("--timeout", type=float, default=1800.0)
    matched_endpoint.add_argument("--model-name", default="tts_orchestrator")
    matched_endpoint.add_argument("--model-version", default="2")
    matched_endpoint.add_argument(
        "--matched-runtime-confirmation",
        required=True,
        help="operator evidence string identifying the unified sample/greedy runtime config",
    )
    matched_endpoint.add_argument("--metadata", action="append", default=[])
    matched_endpoint.set_defaults(handler=command_collect_matched_endpoint)

    matched_official = sub.add_parser(
        "collect-matched-official",
        help="gate-protected official per-segment replay with true segment seeds",
    )
    _output_argument(matched_official)
    matched_official.add_argument(
        "--mode", choices=("sample", "greedy"), default="sample"
    )
    matched_official.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    matched_official.add_argument("--speaker", default="001")
    matched_official.add_argument("--language", default="Auto")
    matched_official.add_argument("--device", default="cuda:0")
    matched_official.add_argument("--dtype", default="bfloat16")
    matched_official.add_argument("--attn-implementation", default="eager")
    matched_official.set_defaults(handler=command_collect_matched_official)

    hashes = sub.add_parser(
        "greedy-hashes", help="compare current/Triton greedy PCM hashes"
    )
    _output_argument(hashes)
    hashes.set_defaults(handler=command_greedy_hashes)

    rootcause_endpoint = sub.add_parser(
        "rootcause-replay-endpoint",
        help="ungated exact-group endpoint replay for ASR/telemetry diagnosis",
    )
    _output_argument(rootcause_endpoint)
    rootcause_endpoint.add_argument("--text-file", type=Path, default=DEFAULT_TEXT)
    rootcause_endpoint.add_argument("--groups-file", required=True, type=Path)
    rootcause_endpoint.add_argument(
        "--arm",
        required=True,
        choices=[ArmKind.CURRENT_HEAD.value, ArmKind.TRITON_0818.value],
    )
    rootcause_endpoint.add_argument("--endpoint", required=True)
    rootcause_endpoint.add_argument(
        "--mode", choices=("sample", "greedy"), required=True
    )
    rootcause_endpoint.add_argument("--seed", action="append", type=int, required=True)
    rootcause_endpoint.add_argument("--speaker", default="001")
    rootcause_endpoint.add_argument("--language", default="auto")
    rootcause_endpoint.add_argument("--timeout", type=float, default=1800.0)
    rootcause_endpoint.add_argument("--model-name", default="tts_orchestrator")
    rootcause_endpoint.add_argument("--model-version", default="2")
    rootcause_endpoint.set_defaults(handler=command_rootcause_replay_endpoint)

    rootcause_official = sub.add_parser(
        "rootcause-replay-official",
        help="ungated exact-group official PyTorch replay",
    )
    _output_argument(rootcause_official)
    rootcause_official.add_argument("--text-file", type=Path, default=DEFAULT_TEXT)
    rootcause_official.add_argument("--groups-file", required=True, type=Path)
    rootcause_official.add_argument(
        "--mode", choices=("sample", "greedy"), required=True
    )
    rootcause_official.add_argument("--seed", action="append", type=int, required=True)
    rootcause_official.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    rootcause_official.add_argument("--speaker", default="001")
    rootcause_official.add_argument("--language", default="Auto")
    rootcause_official.add_argument("--device", default="cuda:0")
    rootcause_official.add_argument("--dtype", default="bfloat16")
    rootcause_official.add_argument("--attn-implementation", default="eager")
    rootcause_official.set_defaults(handler=command_rootcause_replay_official)

    rootcause_asr = sub.add_parser(
        "rootcause-score-asr",
        help="ASR-score ungated root-cause replay WAVs with one connection each",
    )
    _output_argument(rootcause_asr)
    rootcause_asr.add_argument("--text-file", type=Path, default=DEFAULT_TEXT)
    rootcause_asr.add_argument("--asr-url", default=DEFAULT_ASR_URL)
    rootcause_asr.add_argument("--funasr-client-src", type=Path, default=None)
    rootcause_asr.add_argument("--language", default="中文")
    rootcause_asr.add_argument("--chunk-ms", type=int, default=960)
    rootcause_asr.set_defaults(handler=command_rootcause_score_asr)

    score = sub.add_parser(
        "score", help="ASR-screen all full and delivered-segment WAVs"
    )
    _output_argument(score)
    score.add_argument("--asr-url", default=DEFAULT_ASR_URL)
    score.add_argument("--funasr-wheel", required=True, type=Path)
    score.add_argument("--funasr-client-src", type=Path, default=None)
    score.add_argument("--language", default="中文")
    score.add_argument("--chunk-ms", type=int, default=960)
    score.set_defaults(handler=command_score)

    diagnostic = sub.add_parser(
        "diagnostic-report",
        help="emit fail-closed pre-review ASR/acoustic/telemetry diagnostics only",
    )
    _output_argument(diagnostic)
    diagnostic.set_defaults(handler=command_diagnostic_report)

    review = sub.add_parser(
        "prepare-review", help="create randomized arm-blind round-1 package"
    )
    _output_argument(review)
    review.add_argument("--review-seed", type=int, default=8182026)
    review.set_defaults(handler=command_prepare_review)

    second = sub.add_parser("prepare-second-review")
    _output_argument(second)
    second.add_argument("--round1-csv", required=True, type=Path)
    second.add_argument("--reviewer-1-id", required=True)
    second.add_argument("--reviewer-2-id", required=True)
    second.add_argument("--selection-seed", type=int, default=8182027)
    second.set_defaults(handler=command_prepare_second)

    adjudicate = sub.add_parser("prepare-adjudication")
    _output_argument(adjudicate)
    adjudicate.add_argument("--round1-csv", required=True, type=Path)
    adjudicate.add_argument("--round2-csv", required=True, type=Path)
    adjudicate.set_defaults(handler=command_prepare_adjudication)

    finalize = sub.add_parser("finalize-labels")
    _output_argument(finalize)
    finalize.add_argument("--round1-csv", required=True, type=Path)
    finalize.add_argument("--round2-csv", required=True, type=Path)
    finalize.add_argument("--adjudication-csv", type=Path, default=None)
    finalize.add_argument("--bootstrap-iterations", type=int, default=10_000)
    finalize.add_argument("--bootstrap-seed", type=int, default=8182028)
    finalize.set_defaults(handler=command_finalize_labels)

    report = sub.add_parser(
        "report", help="regenerate JSON/CSV/Markdown from final labels"
    )
    _output_argument(report)
    report.add_argument("--bootstrap-iterations", type=int, default=10_000)
    report.add_argument("--bootstrap-seed", type=int, default=8182028)
    report.set_defaults(handler=command_report)

    status = sub.add_parser("status")
    _output_argument(status)
    status.set_defaults(handler=command_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        args.seed = tuple(args.seed or DEFAULT_SEEDS)
    if hasattr(args, "timeout") and args.timeout <= 0:
        parser.error("--timeout must be positive")
    if hasattr(args, "chunk_ms") and args.chunk_ms <= 0:
        parser.error("--chunk-ms must be positive")
    if hasattr(args, "bootstrap_iterations") and args.bootstrap_iterations <= 0:
        parser.error("--bootstrap-iterations must be positive")
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


__all__ = ["build_parser", "main"]
