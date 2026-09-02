"""Command handlers for the long-form comparison entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.validation.hallucination.asr import (
    REQUIRED_FUNASR_VERSION,
    import_funasr_client,
    validate_funasr_sdk_version,
    validate_funasr_service_capabilities,
)

from .arms import EngineGrpcArmAdapter, OfficialPyTorchArmAdapter, TritonGrpcArmAdapter
from .artifacts import discover_run_records, read_json, write_json
from .collection import collect_arm_runs, initialize_experiment, record_phase
from .models import ArmKind
from .matched import (
    collect_matched_endpoint,
    collect_matched_official,
    greedy_hash_comparison,
)
from .preflight import (
    assert_read_only_package,
    checkpoint_identity,
    fetch_asr_capabilities,
    git_identity,
    installed_distribution_version,
    model_package_identity,
    runtime_identity,
    sha256_file,
    source_tree_identity,
    tokenizer_text_identity,
    wheel_identity,
)
from .pre_review_diagnostics import generate_pre_review_diagnostic_report
from .review_package import (
    build_adjudication,
    build_review_package,
    build_second_review,
    finalize_reviews,
)
from .reporting import generate_report
from .rootcause_replay import replay_endpoint, replay_official, score_replay_cases
from .scoring import score_all_runs, scoring_completion_summary


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _metadata(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or not key:
            raise ValueError(f"metadata must be KEY=VALUE, got {raw!r}")
        result[key] = value
    return result


def command_init(args: Any) -> int:
    assert_read_only_package(args.package_dir)
    capabilities = fetch_asr_capabilities(args.asr_url)
    validate_funasr_service_capabilities(capabilities)
    identities = {
        "git": git_identity(args.repo_root),
        "official_source_git": git_identity(args.repo_root / "third_party/Qwen3-TTS"),
        "current_engine_and_evaluator_code": source_tree_identity(
            args.repo_root,
            [
                Path("engine"),
                Path("client/src/qwen3tts"),
                Path("client/src/qwen3tts_protocol"),
                Path("tools/validation/hallucination"),
                Path("scripts/bash/validation"),
                Path("scripts/python/run_engine_dump.py"),
            ],
        ),
        "runtime": runtime_identity(),
        "frozen_model_package": model_package_identity(args.package_dir),
        "frozen_text_tokenization": tokenizer_text_identity(
            args.package_dir, args.text_file
        ),
        "official_checkpoint": checkpoint_identity(args.checkpoint),
        "asr": {
            "websocket_url": args.asr_url,
            "capabilities": capabilities,
            "wheel": wheel_identity(
                args.funasr_wheel, expected_sha256=args.funasr_wheel_sha256
            ),
            "required_sdk_version": REQUIRED_FUNASR_VERSION,
        },
        "configuration_files": {
            str(path.resolve()): sha256_file(path) for path in args.config_file
        },
    }
    manifest = initialize_experiment(
        args.output_dir,
        text_file=args.text_file,
        seeds=args.seed,
        identities=identities,
    )
    _print(
        {
            "output_dir": str(args.output_dir.resolve()),
            "text_sha256": manifest["text"]["sha256"],
            "seeds": manifest["seeds"],
            "model_version": identities["frozen_model_package"]["model_version"],
            "trt_plan_sha256": identities["frozen_model_package"]["plan"]["sha256"],
        }
    )
    return 0


def command_collect_endpoint(args: Any) -> int:
    arm = ArmKind(args.arm)
    common = {
        "timeout": args.timeout,
        "speaker": args.speaker,
        "language": args.language,
    }
    if arm is ArmKind.CURRENT_HEAD:
        adapter = EngineGrpcArmAdapter(args.endpoint, **common)
    elif arm is ArmKind.TRITON_0818:
        adapter = TritonGrpcArmAdapter(
            args.endpoint,
            model_name=args.model_name,
            model_version=args.model_version,
            **common,
        )
    else:
        raise ValueError("collect-endpoint supports only current_head or triton_0818")
    metadata = {
        "endpoint": args.endpoint,
        "speaker": args.speaker,
        "language": args.language,
        "defaults_preserved": True,
        **_metadata(args.metadata),
    }
    with adapter:
        records = collect_arm_runs(args.output_dir, adapter, arm_metadata=metadata)
    _print({"arm": arm.value, "runs": records})
    return int(any(record["status"] != "ok" for record in records))


def command_collect_official(args: Any) -> int:
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "speaker": args.speaker,
        "language": args.language,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "generation_defaults_preserved": True,
        **_metadata(args.metadata),
    }
    adapter = OfficialPyTorchArmAdapter(
        args.checkpoint,
        speaker=args.speaker,
        language=args.language,
        device_map=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        sampling_base_seed=args.sampling_base_seed,
    )
    with adapter:
        records = collect_arm_runs(args.output_dir, adapter, arm_metadata=metadata)
    _print({"arm": ArmKind.PYTORCH_0818.value, "runs": records})
    return int(any(record["status"] != "ok" for record in records))


def command_collect_matched_endpoint(args: Any) -> int:
    arm = ArmKind(args.arm)
    common = {
        "timeout": args.timeout,
        "speaker": args.speaker,
        "language": args.language,
    }
    if arm is ArmKind.CURRENT_HEAD:
        adapter = EngineGrpcArmAdapter(args.endpoint, **common)
    elif arm is ArmKind.TRITON_0818:
        adapter = TritonGrpcArmAdapter(
            args.endpoint,
            model_name=args.model_name,
            model_version=args.model_version,
            **common,
        )
    else:
        raise ValueError("matched endpoint supports current_head or triton_0818")
    with adapter:
        records = collect_matched_endpoint(
            args.output_dir,
            adapter,
            mode=args.mode,
            runtime_metadata={
                "endpoint": args.endpoint,
                "operator_confirmation": args.matched_runtime_confirmation,
                **_metadata(args.metadata),
            },
        )
    _print({"mode": args.mode, "arm": arm.value, "runs": records})
    return int(any(record["status"] != "ok" for record in records))


def command_collect_matched_official(args: Any) -> int:
    adapter = OfficialPyTorchArmAdapter(
        args.checkpoint,
        speaker=args.speaker,
        language=args.language,
        device_map=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        sampling_base_seed=0,
    )
    with adapter:
        records = collect_matched_official(args.output_dir, adapter, mode=args.mode)
    _print({"mode": args.mode, "arm": ArmKind.PYTORCH_0818.value, "runs": records})
    return int(any(record["status"] != "ok" for record in records))


def command_greedy_hashes(args: Any) -> int:
    result = greedy_hash_comparison(args.output_dir)
    _print(result)
    return 0


def command_rootcause_replay_endpoint(args: Any) -> int:
    arm = ArmKind(args.arm)
    common = {
        "timeout": args.timeout,
        "speaker": args.speaker,
        "language": args.language,
    }
    if arm is ArmKind.CURRENT_HEAD:
        adapter = EngineGrpcArmAdapter(args.endpoint, **common)
    elif arm is ArmKind.TRITON_0818:
        adapter = TritonGrpcArmAdapter(
            args.endpoint,
            model_name=args.model_name,
            model_version=args.model_version,
            **common,
        )
    else:
        raise ValueError("root-cause endpoint replay supports only endpoint arms")
    with adapter:
        records = replay_endpoint(
            adapter,
            text_file=args.text_file,
            groups_file=args.groups_file,
            output_dir=args.output_dir,
            seeds=args.seed,
            runtime_mode=args.mode,
        )
    summaries = _rootcause_replay_summaries(records)
    _print({"arm": arm.value, "mode": args.mode, "runs": summaries})
    return int(any(record["status"] != "ok" for record in summaries))


def command_rootcause_replay_official(args: Any) -> int:
    adapter = OfficialPyTorchArmAdapter(
        args.checkpoint,
        speaker=args.speaker,
        language=args.language,
        device_map=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        sampling_base_seed=0,
    )
    with adapter:
        records = replay_official(
            adapter,
            text_file=args.text_file,
            groups_file=args.groups_file,
            output_dir=args.output_dir,
            seeds=args.seed,
            mode=args.mode,
        )
    summaries = _rootcause_replay_summaries(records)
    _print(
        {
            "arm": ArmKind.PYTORCH_0818.value,
            "mode": args.mode,
            "runs": summaries,
        }
    )
    return int(any(record["status"] != "ok" for record in summaries))


def _rootcause_replay_summaries(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for record in records:
        raw_status = record.get("status")
        status = getattr(raw_status, "value", str(raw_status))
        summaries.append(
            {
                "seed": int(record["seed"]),
                "status": status,
                "duration_s": float(record["duration_s"]),
                "frozen_boundaries_preserved": bool(
                    record["frozen_boundaries_preserved"]
                ),
                "wav": record["artifacts"]["wav"],
                "error": record.get("error"),
            }
        )
    return summaries


def command_rootcause_score_asr(args: Any) -> int:
    version = installed_distribution_version("funasrnano")
    validate_funasr_sdk_version(version)
    capabilities = fetch_asr_capabilities(args.asr_url)
    validate_funasr_service_capabilities(capabilities, strict_protocol=True)
    client_class = import_funasr_client(args.funasr_client_src)
    summaries = score_replay_cases(
        args.output_dir,
        text_file=args.text_file,
        client_class=client_class,
        asr_url=args.asr_url,
        language=args.language,
        chunk_ms=args.chunk_ms,
    )
    _print({"sdk_version": version, "runs": summaries})
    return int(any(record["status"] != "ok" for record in summaries))


def _freeze_scoring_evaluator_identity(
    args: Any,
    manifest: dict[str, Any],
    *,
    sdk_version: str,
    wheel_sha256: str,
    capabilities: dict[str, Any],
) -> Path:
    """Freeze the post-synthesis scorer revision before any ASR request."""

    repo_root = Path(__file__).resolve().parents[4]
    preflight_code = dict(
        manifest.get("identities", {}).get("current_engine_and_evaluator_code") or {}
    )
    preflight_tree_sha256 = preflight_code.get("tree_sha256")
    if (manifest.get("text") or manifest.get("seeds")) and not preflight_tree_sha256:
        raise RuntimeError("formal manifest lacks its preflight evaluator identity")
    payload = {
        "schema_version": 1,
        "preflight_engine_and_evaluator_tree_sha256": preflight_tree_sha256,
        "scoring_evaluator_code": source_tree_identity(
            repo_root,
            [Path("tools/validation/hallucination")],
        ),
        "asr_request": {
            "websocket_url": args.asr_url,
            "language": args.language,
            "chunk_ms": int(args.chunk_ms),
            "delivery_mode": "offline",
            "pacing": "none",
            "partial_mode": "off",
            "hotwords": [],
            "vad": "fsmn",
        },
        "asr_runtime": {
            "sdk_version": sdk_version,
            "wheel_sha256": wheel_sha256,
            "service_version": capabilities.get("version"),
            "protocol_version": capabilities.get("protocol_version"),
        },
    }
    identity_path = args.output_dir / "runtime" / "scoring_evaluator_identity.json"
    if identity_path.is_file():
        if read_json(identity_path) != payload:
            raise RuntimeError(
                "scoring evaluator/config changed after its formal identity was frozen"
            )
    else:
        write_json(identity_path, payload)
    return identity_path


def command_score(args: Any) -> int:
    manifest = read_json(args.output_dir / "manifest.json")
    asr_identity = dict(manifest["identities"]["asr"])
    if asr_identity["websocket_url"] != args.asr_url:
        raise RuntimeError("ASR URL differs from the frozen preflight identity")
    wheel = wheel_identity(args.funasr_wheel)
    if wheel["sha256"] != asr_identity["wheel"]["sha256"]:
        raise RuntimeError("ASR wheel differs from the frozen preflight identity")
    version = installed_distribution_version("funasrnano")
    validate_funasr_sdk_version(version)
    capabilities = fetch_asr_capabilities(args.asr_url)
    validate_funasr_service_capabilities(capabilities)
    scoring_identity_path = _freeze_scoring_evaluator_identity(
        args,
        manifest,
        sdk_version=version,
        wheel_sha256=wheel["sha256"],
        capabilities=capabilities,
    )
    client_class = import_funasr_client(args.funasr_client_src)
    observations = score_all_runs(
        args.output_dir,
        client_class=client_class,
        asr_url=args.asr_url,
        language=args.language,
        chunk_ms=args.chunk_ms,
    )
    summary = scoring_completion_summary(args.output_dir)
    if not summary["ready_for_review"]:
        _print(
            {
                "observation_count": len(observations),
                "status": "scoring_incomplete",
                "summary": summary,
            }
        )
        return 1
    record_phase(
        args.output_dir,
        "score-asr",
        details={
            "sdk_version": version,
            "wheel_sha256": wheel["sha256"],
            "capabilities": capabilities,
            "observation_count": len(observations),
            "concurrency": 1,
            "fresh_connection_per_wav": True,
            "asr_wav_count": summary["asr_wav_count"],
            "fresh_connection_origin_wav_count": summary[
                "fresh_connection_origin_wav_count"
            ],
            "scoring_summary": summary,
            "scoring_evaluator_identity": str(scoring_identity_path),
            "scoring_evaluator_identity_sha256": sha256_file(scoring_identity_path),
        },
    )
    _print(
        {
            "observation_count": len(observations),
            "status": "review_pending",
            "summary": summary,
        }
    )
    return 0


def command_prepare_review(args: Any) -> int:
    result = build_review_package(args.output_dir, review_seed=args.review_seed)
    record_phase(args.output_dir, "prepare-review", details=result)
    _print(result)
    return 0


def command_diagnostic_report(args: Any) -> int:
    report = generate_pre_review_diagnostic_report(args.output_dir)
    _print(
        {
            "status": "diagnostic_only_pre_review",
            "grid": report["grid"],
            "report": "diagnostics/pre_review.md",
            "root_cause_gate_status": report["policy"]["root_cause_gate_status"],
        }
    )
    return 0


def command_prepare_second(args: Any) -> int:
    result = build_second_review(
        args.output_dir,
        args.round1_csv,
        reviewer1_id=args.reviewer_1_id,
        reviewer2_id=args.reviewer_2_id,
        selection_seed=args.selection_seed,
    )
    _print(result)
    return 0


def command_prepare_adjudication(args: Any) -> int:
    result = build_adjudication(args.output_dir, args.round1_csv, args.round2_csv)
    _print(result)
    return 0


def command_finalize_labels(args: Any) -> int:
    rows = finalize_reviews(
        args.output_dir,
        args.round1_csv,
        args.round2_csv,
        args.adjudication_csv,
    )
    record_phase(args.output_dir, "finalize-labels", details={"label_count": len(rows)})
    record_phase(
        args.output_dir,
        "report",
        details={
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
        },
    )
    report = generate_report(
        args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    _print(
        {
            "label_count": len(rows),
            "labels": "review/private/final_labels.json",
            "report": "report/report.md",
            "conclusion": report["conclusion"],
        }
    )
    return 0


def command_report(args: Any) -> int:
    report = generate_report(
        args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    _print({"report": "report/report.md", "conclusion": report["conclusion"]})
    return 0


def command_status(args: Any) -> int:
    records = discover_run_records(args.output_dir)
    by_arm: dict[str, dict[str, int]] = {}
    for record in records:
        arm = str(record["arm"])
        status = str(record["status"])
        by_arm.setdefault(arm, {})[status] = (
            by_arm.setdefault(arm, {}).get(status, 0) + 1
        )
    observations = 0
    path = args.output_dir / "sentence_observations.json"
    if path.is_file():
        observations = len(read_json(path).get("observations") or [])
    _print(
        {
            "output_dir": str(args.output_dir.resolve()),
            "runs": len(records),
            "runs_by_arm_and_status": by_arm,
            "sentence_observations": observations,
            "review_package": (args.output_dir / "review" / "public").is_dir(),
            "final_labels": (
                args.output_dir / "review" / "private" / "final_labels.json"
            ).is_file(),
        }
    )
    return 0


__all__ = [name for name in globals() if name.startswith("command_")]
