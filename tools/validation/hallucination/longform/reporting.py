"""Machine-readable and Markdown reports from adjudicated blind truth."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .artifacts import discover_run_records, read_json, write_json
from .evidence_inventory import artifact_identity, runtime_evidence_inventory
from .metrics import (
    evaluate_comparison_gate,
    evaluate_invalidity,
    paired_bootstrap_comparison,
    wilson_interval,
)
from .models import (
    ArmKind,
    ReferenceSentence,
    ReviewLabel,
    RunStatus,
    parse_reference_sentences,
)


_ARMS = (
    ArmKind.CURRENT_HEAD,
    ArmKind.TRITON_0818,
    ArmKind.PYTORCH_0818,
)
_COMPARISONS = (
    (ArmKind.TRITON_0818, ArmKind.CURRENT_HEAD),
    (ArmKind.PYTORCH_0818, ArmKind.CURRENT_HEAD),
    (ArmKind.PYTORCH_0818, ArmKind.TRITON_0818),
)
REPORT_PROVENANCE_PATH = Path("report/report_provenance.json")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report_provenance(
    output_root: Path,
    *,
    report_path: Path,
    manifest_path: Path,
    reference_groups_path: Path,
    final_labels_path: Path,
) -> dict[str, Any]:
    """Bind the primary report to every human-truth input by content hash."""

    payload = {
        "schema_version": 1,
        "artifact_role": "primary_arm_blind_truth_report",
        "truth_source": "arm-blind human review",
        "artifacts": {
            "experiment_manifest": artifact_identity(output_root, manifest_path),
            "reference_groups": artifact_identity(output_root, reference_groups_path),
            "final_labels": artifact_identity(output_root, final_labels_path),
            "report": artifact_identity(output_root, report_path),
        },
    }
    write_json(output_root / REPORT_PROVENANCE_PATH, payload)
    return payload


def _aggregate_unit_rate(
    rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    values: list[bool | None] = []
    for group in grouped.values():
        if any(row.get("severe_hallucination") is True for row in group):
            values.append(True)
        elif all(row.get("valid_for_rate") is True for row in group):
            values.append(False)
        else:
            values.append(None)
    valid = [value for value in values if isinstance(value, bool)]
    positives = sum(value is True for value in valid)
    result = wilson_interval(positives, len(valid))
    result.update({"expected": len(values), "invalid": len(values) - len(valid)})
    return result


def _diagnostics_by_arm(output_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    records = discover_run_records(output_root)
    for arm in _ARMS:
        selected = [record for record in records if record.get("arm") == arm.value]
        cers: list[float] = []
        duration_ratios: list[float] = []
        substitutions = deletions = insertions = 0
        retries = loop_recoveries = aborts = guards = 0
        for record in selected:
            scoring_path = output_root / str(record["_record_path"])
            scoring_path = scoring_path.parent / "scoring.json"
            if scoring_path.is_file():
                scoring = read_json(scoring_path)
                errors = dict(scoring.get("full_character_errors") or {})
                if isinstance(errors.get("cer"), (int, float)):
                    cers.append(float(errors["cer"]))
                substitutions += int(errors.get("substitutions", 0) or 0)
                deletions += int(errors.get("deletions", 0) or 0)
                insertions += int(errors.get("insertions", 0) or 0)
                telemetry = dict(scoring.get("engine_telemetry") or {})
                maxima = dict(telemetry.get("metric_maxima") or {})
                explicit_retry_event = any(
                    "retry" in str(item.get("type", "")).lower()
                    for item in telemetry.get("retry_events", [])
                    if isinstance(item, Mapping)
                )
                retries += int(
                    explicit_retry_event
                    or int(maxima.get("retry_idx", 0) or 0) > 0
                    or int(maxima.get("retry_count", 0) or 0) > 0
                )
                loop_recoveries += int(
                    int(maxima.get("loop_recovery_count", 0) or 0) > 0
                )
                aborts += int(
                    any(
                        str(reason).endswith("_abort")
                        for reason in telemetry.get("eos_reasons", [])
                    )
                    or int(maxima.get("loop_abort_count", 0) or 0) > 0
                )
                guards += int(bool(telemetry.get("guard_modes")))
            characters = max(1, len(str(record.get("source_text", ""))))
            duration_ratios.append(float(record.get("duration_s", 0.0)) / characters)
        total = len(selected)
        result[arm.value] = {
            "run_count": total,
            "tts_failure_count": sum(
                record.get("status") != "ok" for record in selected
            ),
            "mean_cer": mean(cers) if cers else None,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "mean_audio_seconds_per_source_codepoint": (
                mean(duration_ratios) if duration_ratios else None
            ),
            "guard_run_rate": guards / total if total else None,
            "retry_run_rate": retries / total if total else None,
            "loop_recovery_run_rate": (loop_recoveries / total if total else None),
            "abort_run_rate": aborts / total if total else None,
            "diagnostic_only": True,
        }
    return result


def _expected_observations(
    manifest: Mapping[str, Any],
) -> tuple[list[int], tuple[ReferenceSentence, ...]]:
    seeds = list(manifest.get("seeds") or [])
    if (
        len(seeds) != 3
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        or len(set(seeds)) != 3
    ):
        raise ValueError("experiment manifest must contain three unique integer seeds")
    text_record = manifest.get("text")
    source_text = text_record.get("text") if isinstance(text_record, Mapping) else None
    if not isinstance(source_text, str):
        raise ValueError("experiment manifest must contain the exact source text")
    sentences = parse_reference_sentences(source_text)
    if not sentences:
        raise ValueError("experiment manifest source text contains no sentences")
    return [int(seed) for seed in seeds], sentences


def _complete_observation_grid(
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    sentences: Sequence[ReferenceSentence],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fill absent expected positions as invalid and quarantine duplicate rows."""

    sentence_by_ordinal = {sentence.ordinal: sentence for sentence in sentences}
    expected_positions = {
        (int(seed), sentence.ordinal) for seed in seeds for sentence in sentences
    }
    completed: list[dict[str, Any]] = []
    completeness: dict[str, Any] = {
        "expected_per_arm": len(expected_positions),
        "arms": {},
        "unexpected_arm_rows": sum(
            row.get("arm") not in {arm.value for arm in _ARMS} for row in rows
        ),
    }
    for arm in _ARMS:
        selected = [row for row in rows if row.get("arm") == arm.value]
        by_position: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
        unexpected_rows = 0
        for row in selected:
            seed = row.get("seed")
            ordinal = row.get("sentence_ordinal")
            if (
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not isinstance(ordinal, int)
                or isinstance(ordinal, bool)
                or (seed, ordinal) not in expected_positions
            ):
                unexpected_rows += 1
                continue
            by_position[(seed, ordinal)].append(row)

        missing_positions = expected_positions - by_position.keys()
        duplicate_positions = {
            position: candidates
            for position, candidates in by_position.items()
            if len(candidates) > 1
        }
        for seed, ordinal in sorted(expected_positions):
            candidates = by_position.get((seed, ordinal), [])
            if len(candidates) == 1:
                completed.append(
                    {
                        **dict(candidates[0]),
                        "arm": arm.value,
                        "seed": seed,
                        "sentence_ordinal": ordinal,
                        "sentence_id": sentence_by_ordinal[ordinal].sentence_id,
                    }
                )
                continue
            sentence = sentence_by_ordinal[ordinal]
            completed.append(
                {
                    "arm": arm.value,
                    "seed": seed,
                    "sentence_ordinal": ordinal,
                    "sentence_id": sentence.sentence_id,
                    "review_label": None,
                    "severe_hallucination": None,
                    "valid_for_rate": False,
                    "status": RunStatus.INVALID.value,
                    "report_invalid_reason": (
                        "missing_observation"
                        if not candidates
                        else "duplicate_observation"
                    ),
                }
            )
        arm_completeness = {
            "expected": len(expected_positions),
            "observed_rows": len(selected),
            "observed_expected_positions": len(by_position),
            "missing_observations": len(missing_positions),
            "duplicate_positions": len(duplicate_positions),
            "duplicate_extra_rows": sum(
                len(candidates) - 1 for candidates in duplicate_positions.values()
            ),
            "unexpected_rows": unexpected_rows,
        }
        arm_completeness["all_expected_positions_present"] = (
            arm_completeness["missing_observations"] == 0
        )
        arm_completeness["integrity_complete"] = all(
            arm_completeness[field] == 0
            for field in (
                "missing_observations",
                "duplicate_positions",
                "unexpected_rows",
            )
        )
        completeness["arms"][arm.value] = arm_completeness
    completeness["total_expected"] = len(expected_positions) * len(_ARMS)
    completeness["integrity_complete"] = completeness[
        "unexpected_arm_rows"
    ] == 0 and all(arm["integrity_complete"] for arm in completeness["arms"].values())
    return completed, completeness


def _apply_grid_integrity(
    invalidity: dict[str, Any], completeness: Mapping[str, Any]
) -> None:
    """Make structural grid corruption a fail-closed validity condition."""

    integrity_reasons: list[str] = []
    if int(completeness.get("unexpected_arm_rows", 0) or 0):
        integrity_reasons.append("observation_grid:unexpected_arm_rows")
    for arm, values in dict(completeness.get("arms") or {}).items():
        for field in (
            "missing_observations",
            "duplicate_positions",
            "unexpected_rows",
        ):
            if int(values.get(field, 0) or 0):
                integrity_reasons.append(f"{arm}:{field}")
    invalidity["observation_grid_integrity"] = {
        "complete": not integrity_reasons,
        "reasons": integrity_reasons,
    }
    if integrity_reasons:
        invalidity["reasons"] = list(invalidity.get("reasons") or []) + [
            reason
            for reason in integrity_reasons
            if reason not in invalidity.get("reasons", [])
        ]
        invalidity["insufficient"] = True
        invalidity["conclusion_sufficient"] = False


def _arm_summary(
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Mapping[str, Any]],
    completeness: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    group_for_sentence = {
        int(ordinal): int(group["group_index"])
        for group in groups
        for ordinal in group["sentence_ordinals"]
    }
    summary: dict[str, dict[str, Any]] = {}
    for arm in _ARMS:
        selected = [row for row in rows if row.get("arm") == arm.value]
        enriched = [
            dict(row, reference_group=group_for_sentence[int(row["sentence_ordinal"])])
            for row in selected
        ]
        valid = [row for row in enriched if row.get("valid_for_rate") is True]
        positive = [row for row in valid if row.get("severe_hallucination") is True]
        labels = Counter(str(row.get("review_label", "")) for row in valid)
        sentence_rate = wilson_interval(len(positive), len(valid))
        arm_completeness = dict(completeness.get(arm.value) or {})
        sentence_rate.update(
            {
                "expected": len(enriched),
                "observed": arm_completeness.get("observed_expected_positions"),
                "missing": arm_completeness.get("missing_observations"),
                "invalid": len(enriched) - len(valid),
            }
        )
        summary[arm.value] = {
            "sentence_rate": sentence_rate,
            "document_rate": _aggregate_unit_rate(enriched, ("seed",)),
            "reference_group_rate": _aggregate_unit_rate(
                enriched, ("seed", "reference_group")
            ),
            "loop_count": labels[ReviewLabel.SINGLE_UNIT_LOOP.value],
            "noise_count": labels[ReviewLabel.ABNORMAL_NOISE.value],
            "unsupported_speech_count": labels[ReviewLabel.UNSUPPORTED_SPEECH.value],
            "omission_count": labels[ReviewLabel.OMISSION.value],
            "mispronunciation_count": labels[ReviewLabel.MISPRONUNCIATION.value],
            "labels": dict(sorted(labels.items())),
            "completeness": arm_completeness,
            "diagnostics": dict(diagnostics.get(arm.value) or {}),
        }
    return summary


def generate_report(
    output_root: Path,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 8182028,
) -> dict[str, Any]:
    """Apply the predeclared rate/invalidity gates and persist all report forms."""

    raw_rows = list(
        read_json(output_root / "review" / "private" / "final_labels.json").get("rows")
        or []
    )
    groups = list(read_json(output_root / "reference_groups.json").get("groups") or [])
    diagnostics = _diagnostics_by_arm(output_root)
    manifest_path = output_root / "manifest.json"
    reference_groups_path = output_root / "reference_groups.json"
    final_labels_path = output_root / "review" / "private" / "final_labels.json"
    manifest = read_json(manifest_path)
    seeds, sentences = _expected_observations(manifest)
    rows, completeness = _complete_observation_grid(raw_rows, seeds, sentences)
    counts = {
        arm.value: {
            "total": sum(row.get("arm") == arm.value for row in rows),
            "invalid": sum(
                row.get("arm") == arm.value and row.get("valid_for_rate") is not True
                for row in rows
            ),
        }
        for arm in _ARMS
    }
    invalidity = evaluate_invalidity(counts)
    _apply_grid_integrity(invalidity, completeness)
    arm_summary = _arm_summary(rows, groups, diagnostics, completeness["arms"])
    comparisons: list[dict[str, Any]] = []
    for baseline, comparison in _COMPARISONS:
        try:
            bootstrap = paired_bootstrap_comparison(
                rows,
                baseline,
                comparison,
                iterations=bootstrap_iterations,
                random_seed=bootstrap_seed,
            )
            gate = evaluate_comparison_gate(bootstrap, invalidity=invalidity)
            comparisons.append({"bootstrap": bootstrap, "gate": gate})
        except ValueError as exc:
            comparisons.append(
                {
                    "baseline_arm": baseline.value,
                    "comparison_arm": comparison.value,
                    "error": str(exc),
                    "gate": {"passes": False, "reason": "insufficient paired truth"},
                }
            )

    current_vs_triton = next(
        item
        for item in comparisons
        if (item.get("bootstrap") or item).get("baseline_arm")
        == ArmKind.TRITON_0818.value
        and (item.get("bootstrap") or item).get("comparison_arm")
        == ArmKind.CURRENT_HEAD.value
    )
    any_large_gap = any(item.get("gate", {}).get("passes") for item in comparisons)
    if invalidity["insufficient"]:
        conclusion = "INSUFFICIENT_INVALID_SAMPLES"
        next_step = "补齐 TTS 失败/UNSCORABLE 样本后再判定；缺失不得记为 clean。"
    elif current_vs_triton.get("gate", {}).get("passes"):
        conclusion = "DEFAULT_GAP_REQUIRES_MATCHED_REPLAY"
        next_step = (
            f"默认差异很大；先做统一采样、冻结的 {len(groups)} 个 Triton commit "
            "分组同参重放，不能直接判 TRT 核心错误。"
        )
    elif any_large_gap:
        conclusion = "LARGE_IMPLEMENTATION_GAP_REQUIRES_MATCHED_REPLAY"
        next_step = "至少一组实现差异很大；按最小失败段执行同参重放与首分歧层诊断。"
    else:
        conclusion = "NO_PREDECLARED_LARGE_RATE_GAP"
        next_step = "未通过比例门槛；不因单个坏例进入引擎根因分支。"

    report = {
        "schema_version": 1,
        "truth_source": "arm-blind human review",
        "asr_and_acoustics_are_diagnostic_only": True,
        "report_inputs": {
            "experiment_manifest": artifact_identity(output_root, manifest_path),
            "reference_groups": artifact_identity(output_root, reference_groups_path),
            "final_labels": artifact_identity(output_root, final_labels_path),
        },
        "runtime_evidence": runtime_evidence_inventory(output_root),
        "reference_group_count": len(groups),
        "expected_sentences_per_arm": len(seeds) * len(sentences),
        "expected_sentence_observations_total": completeness["total_expected"],
        "observation_completeness": completeness,
        "invalidity": invalidity,
        "arms": arm_summary,
        "comparisons": comparisons,
        "conclusion": conclusion,
        "next_step": next_step,
        "root_cause_confirmed": False,
        "report_provenance": str(REPORT_PROVENANCE_PATH),
    }
    report_dir = output_root / "report"
    report_path = report_dir / "report.json"
    write_json(report_path, report)
    _write_report_provenance(
        output_root,
        report_path=report_path,
        manifest_path=manifest_path,
        reference_groups_path=reference_groups_path,
        final_labels_path=final_labels_path,
    )
    _write_csv(
        report_dir / "sentences.csv",
        [
            {
                "arm": row.get("arm"),
                "seed": row.get("seed"),
                "sentence_ordinal": row.get("sentence_ordinal"),
                "sentence_id": row.get("sentence_id"),
                "blind_id": row.get("blind_id"),
                "review_label": row.get("review_label"),
                "severe_hallucination": row.get("severe_hallucination"),
                "valid_for_rate": row.get("valid_for_rate"),
                "status": row.get("status"),
                "report_invalid_reason": row.get("report_invalid_reason"),
                "asr_cer": (row.get("character_errors") or {}).get("cer"),
                "asr_substitutions": (row.get("character_errors") or {}).get(
                    "substitutions"
                ),
                "asr_deletions": (row.get("character_errors") or {}).get("deletions"),
                "asr_insertions": (row.get("character_errors") or {}).get("insertions"),
                "clip": row.get("clip"),
            }
            for row in rows
        ],
    )
    _write_csv(
        report_dir / "arms.csv",
        [
            {
                "arm": arm,
                "severe": values["sentence_rate"]["successes"],
                "valid": values["sentence_rate"]["total"],
                "invalid": values["sentence_rate"]["invalid"],
                "observed": values["sentence_rate"]["observed"],
                "missing": values["sentence_rate"]["missing"],
                "sentence_rate": values["sentence_rate"]["rate"],
                "wilson_low": values["sentence_rate"]["low"],
                "wilson_high": values["sentence_rate"]["high"],
                "document_rate": values["document_rate"]["rate"],
                "reference_group_rate": values["reference_group_rate"]["rate"],
                "loop_count": values["loop_count"],
                "noise_count": values["noise_count"],
                "unsupported_speech_count": values["unsupported_speech_count"],
                "cer": values["diagnostics"].get("mean_cer"),
            }
            for arm, values in arm_summary.items()
        ],
    )
    comparison_rows: list[dict[str, Any]] = []
    for item in comparisons:
        bootstrap = dict(item.get("bootstrap") or {})
        comparison_rows.append(
            {
                "baseline_arm": bootstrap.get("baseline_arm", item.get("baseline_arm")),
                "comparison_arm": bootstrap.get(
                    "comparison_arm", item.get("comparison_arm")
                ),
                "risk_difference": bootstrap.get("risk_difference"),
                "risk_difference_ci_low": (
                    bootstrap.get("risk_difference_ci") or [None, None]
                )[0],
                "risk_difference_ci_high": (
                    bootstrap.get("risk_difference_ci") or [None, None]
                )[1],
                "risk_ratio": bootstrap.get("risk_ratio"),
                "gate_passes": item.get("gate", {}).get("passes"),
                "error": item.get("error"),
            }
        )
    _write_csv(report_dir / "comparisons.csv", comparison_rows)
    (report_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 0818 长文本幻觉率三臂对比",
        "",
        f"结论：`{report['conclusion']}`",
        "",
        str(report["next_step"]),
        "",
        f"冻结 Triton commit 分组数：`{report['reference_group_count']}`。",
        "",
        "> 最终真值来自 arm-blind 人工审阅；ASR、CER、声学特征和引擎遥测只用于定位。",
        "",
        "## 主指标",
        "",
        "| Arm | 严重/有效 | 句率 (Wilson 95%) | 文档率 | 参考组率 | Loop | Noise | 插入语音 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in report["arms"].items():
        rate = values["sentence_rate"]
        interval = f"{_percent(rate['rate'])} ({_percent(rate['low'])}–{_percent(rate['high'])})"
        lines.append(
            f"| {arm} | {rate['successes']}/{rate['total']} | {interval} | "
            f"{_percent(values['document_rate']['rate'])} | "
            f"{_percent(values['reference_group_rate']['rate'])} | "
            f"{values['loop_count']} | {values['noise_count']} | {values['unsupported_speech_count']} |"
        )
    lines.extend(
        [
            "",
            "## 配对比较",
            "",
            "| Baseline → Comparison | RD (95% CI) | RR | 门槛 |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in report["comparisons"]:
        bootstrap = item.get("bootstrap") or {}
        if not bootstrap:
            lines.append(
                f"| {item.get('baseline_arm')} → {item.get('comparison_arm')} | — | — | 不充分 |"
            )
            continue
        ci = bootstrap["risk_difference_ci"]
        lines.append(
            f"| {bootstrap['baseline_arm']} → {bootstrap['comparison_arm']} | "
            f"{_percent(bootstrap['risk_difference'])} ({_percent(ci[0])}–{_percent(ci[1])}) | "
            f"{bootstrap['risk_ratio']:.3g} | {'通过' if item['gate']['passes'] else '未通过'} |"
        )
    invalidity = report["invalidity"]
    completeness = report["observation_completeness"]
    evidence = report["runtime_evidence"]
    report_inputs = report["report_inputs"]
    lines.extend(
        [
            "",
            "## 观测完整性",
            "",
            f"期望句级观察共 {completeness['total_expected']} 个；"
            f"网格完整：`{completeness['integrity_complete']}`。",
            "",
            "| Arm | 期望 | 已观测位置 | 缺失 | 重复位置 | 网格外行 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, values in completeness["arms"].items():
        lines.append(
            f"| {arm} | {values['expected']} | "
            f"{values['observed_expected_positions']} | "
            f"{values['missing_observations']} | {values['duplicate_positions']} | "
            f"{values['unexpected_rows']} |"
        )
    lines.extend(
        [
            "",
            "## 可追溯性",
            "",
            f"实验预检清单 SHA-256：`{report_inputs['experiment_manifest']['sha256']}`。",
            "",
            f"盲审终值 SHA-256：`{report_inputs['final_labels']['sha256']}`。",
            "",
            f"参考分组 SHA-256：`{report_inputs['reference_groups']['sha256']}`。",
            "",
            f"运行时证据清单：audit-ready=`{evidence['audit_ready']}`，"
            f"文件 {evidence['file_count']} 个，符号链接 {evidence['symlink_count']} 个，"
            f"tree SHA-256=`{evidence['tree_sha256']}`。",
            "",
            "| 运行时证据 | 类型 | 字节 | SHA-256 | 哈希范围 |",
            "|---|---|---:|---|---|",
        ]
    )
    if evidence["entries"]:
        for entry in evidence["entries"]:
            lines.append(
                f"| `{entry['path']}` | {entry['kind']} | {entry['bytes']} | "
                f"`{entry['sha256']}` | {entry['hash_scope']} |"
            )
    else:
        lines.append("| — | — | — | — | 未找到 `output/runtime` 证据 |")
    lines.extend(
        [
            "",
            "## 有效性",
            "",
            f"结论不充分：`{invalidity['insufficient']}`；原因：{', '.join(invalidity['reasons']) or '无'}。",
            "",
            "机器可读明细见 `report.json`、`arms.csv`、`comparisons.csv` 和 `sentences.csv`。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["REPORT_PROVENANCE_PATH", "generate_report"]
