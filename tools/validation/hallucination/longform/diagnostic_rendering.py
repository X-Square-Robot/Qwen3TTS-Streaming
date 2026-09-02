"""CSV and Markdown rendering for pre-review diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import write_json


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(str(field))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(
            {field: _csv_value(value) for field, value in row.items()} for row in rows
        )


def _cer(value: Any) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def _duration(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}s"


def _count(value: Any) -> str:
    return "N/A" if value is None else str(value)


def render_pre_review_markdown(report: Mapping[str, Any]) -> str:
    grid = report["grid"]
    lines = [
        "# 0818 长文本盲审前诊断报告",
        "",
        "> 本报告中的 ASR、CER、声学特征与引擎遥测仅用于定位候选问题。",
        "> 在 arm-blind 人工审阅完成前，不能输出确认严重幻觉率，也不能评估或触发根因 gate。",
        "",
        "## 正式网格",
        "",
        f"已严格校验 {grid['observed_run_count']}/{grid['expected_run_count']} 个文档运行、"
        f"{grid['observed_observation_count']}/{grid['expected_observation_count']} 个句级观察、"
        f"{grid['reference_group_count']} 个冻结参考组；grid_complete=`{grid['complete']}`。",
        f"ASR 证据覆盖 {grid['fresh_connection_origin_wav_count']}/"
        f"{grid['asr_wav_count']} 个 WAV，全部来源于各自独立建立的连接。",
        "FunASR 0.2.0a6 SDK 在首个 `stream_done` 后结束事件迭代；因此唯一终态校验"
        "覆盖 SDK 暴露的事件流，无法独立观察迭代关闭后服务端可能发送的重复终态帧。",
        "",
        "## Arm 诊断聚合",
        "",
        "| Arm | Runs | 时长均值 | 加权 CER | S/D/I | Segments | Loop abort/recovery | Guard runs | Actual retry runs | PCM 唯一哈希 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in report["arms"].items():
        lines.append(
            f"| {arm} | {values['run_count']} | "
            f"{_duration(values['duration_mean_s'])} | "
            f"{_cer(values['weighted_cer'])} | {values['substitutions']}/"
            f"{values['deletions']}/{values['insertions']} | "
            f"{values['delivered_segment_count']} | {values['loop_abort_count']}/"
            f"{values['loop_recovery_count']} | {values['guard_run_count']} | "
            f"{values['actual_retry_run_count']} | "
            f"{values['unique_pcm_f32le_hash_count']} |"
        )
    lines.extend(
        [
            "",
            "## 文档级诊断",
            "",
            "| Arm | Seed | Duration | PCM f32 SHA-256 | CER | S/D/I | Segments | EOS | Loop max | Guard | Actual retry |",
            "|---|---:|---:|---|---:|---:|---:|---|---:|---|---:|",
        ]
    )
    for row in report["documents"]:
        eos = (
            ", ".join(
                f"{name}:{count}" for name, count in row["eos_reason_counts"].items()
            )
            or "—"
        )
        lines.append(
            f"| {row['arm']} | {row['seed']} | {_duration(row['duration_s'])} | "
            f"`{row['pcm_f32le_sha256']}` | {_cer(row['cer'])} | "
            f"{_count(row['substitutions'])}/{_count(row['deletions'])}/"
            f"{_count(row['insertions'])} | {row['delivered_segment_count']} | "
            f"{eos} | {row['loop_max_run']} | "
            f"{', '.join(row['guard_modes']) or '—'} | "
            f"{row['actual_retry_event_count']} |"
        )
    candidates = [
        row
        for row in report["sentences"]
        if row["insertion_candidate_count"] or row["repeat_candidate_count"]
    ]
    candidates.sort(
        key=lambda row: (
            -int(row["insertion_candidate_count"]),
            -int(row["repeat_candidate_count"]),
            -(float(row["cer"]) if row["cer"] is not None else -1.0),
        )
    )
    lines.extend(
        [
            "",
            "## 句级 ASR 候选（非真值）",
            "",
            "| Arm | Seed | 句位 | 参考组 | CER | Insert candidates | Repeat candidates |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidates[:30]:
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['sentence_ordinal']} | "
            f"{row['reference_group']} | {_cer(row['cer'])} | "
            f"{row['insertion_candidate_count']} | "
            f"{row['repeat_candidate_count']} |"
        )
    if not candidates:
        lines.append("| — | — | — | — | — | 0 | 0 |")
    punctuation_segments = sum(
        row["reference_kind"] == "punctuation_only" for row in report["segments"]
    )
    lines.extend(
        [
            "",
            "## 输入身份",
            "",
            f"诊断评测器身份 SHA-256：`{report['evaluator_identity']['sha256']}`。",
            "",
            "| 输入文件 | Bytes | SHA-256 |",
            "|---|---:|---|",
        ]
    )
    for artifact in report["input_artifacts"]:
        lines.append(
            f"| `{artifact['path']}` | {artifact['bytes']} | `{artifact['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## 明细文件",
            "",
            f"参考组明细共 {len(report['reference_groups'])} 行"
            "（3 arms × 3 seeds × 9 groups）；"
            f"segment 明细共 {len(report['segments'])} 行，其中 "
            f"{punctuation_segments} 个纯标点 segment 的 CER 明确记为 N/A。",
            "",
            "- `documents.csv`：文档级时长、哈希、ASR S/D/I/CER 与引擎事件。",
            "- `reference_groups.csv`：每个 arm/seed/参考组的句级 ASR 聚合。",
            "- `sentences.csv`：342 个句位及 insertion/repeat 候选。",
            "- `segments.csv`：交付 segment、文本/音频边界 provenance 及 CER 可用性。",
            "",
            "真实比例与根因 gate 必须等待盲审标签，以上候选不得自动转成阳性。",
            "",
        ]
    )
    return "\n".join(lines)


def write_pre_review_outputs(output_root: Path, report: Mapping[str, Any]) -> None:
    diagnostic_root = output_root / "diagnostics"
    _write_csv(diagnostic_root / "documents.csv", report["documents"])
    _write_csv(diagnostic_root / "reference_groups.csv", report["reference_groups"])
    _write_csv(diagnostic_root / "sentences.csv", report["sentences"])
    _write_csv(diagnostic_root / "segments.csv", report["segments"])
    write_json(diagnostic_root / "pre_review.json", report)
    (diagnostic_root / "pre_review.md").write_text(
        render_pre_review_markdown(report), encoding="utf-8"
    )


__all__ = ["render_pre_review_markdown", "write_pre_review_outputs"]
