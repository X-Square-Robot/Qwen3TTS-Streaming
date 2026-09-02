from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.validation.hallucination.longform.artifacts import read_json, write_json
from tools.validation.hallucination.longform.cli import build_parser
from tools.validation.hallucination.longform.diagnostic_metrics import (
    is_actual_retry_event,
)
from tools.validation.hallucination.longform.models import (
    ArmKind,
    RunStatus,
    parse_reference_sentences,
)
from tools.validation.hallucination.longform.pre_review_diagnostics import (
    generate_pre_review_diagnostic_report,
)
from tools.validation.hallucination.longform.text_normalization import (
    normalize_transcript,
)


SEEDS = (11, 22, 33)
ARMS = (
    ArmKind.CURRENT_HEAD,
    ArmKind.TRITON_0818,
    ArmKind.PYTORCH_0818,
)
GROUP_SIZES = (5, 4, 4, 5, 5, 4, 5, 5, 1)


def _metrics(reference: str, *, ordinal: int | None = None) -> dict[str, Any]:
    normalized = normalize_transcript(reference)
    insertions = 1 if ordinal == 1 else 0
    insertion_spans = (
        [
            {
                "hypothesis_start": len(normalized),
                "hypothesis_end": len(normalized) + 1,
                "length": 1,
                "text": "啊",
                "repetition_spans": [],
            }
        ]
        if insertions
        else []
    )
    repetitions = (
        [
            {
                "start": 0,
                "end": 3,
                "length": 3,
                "unit": "哈",
                "unit_length": 1,
                "repetitions": 3,
                "text": "哈哈哈",
            }
        ]
        if ordinal == 2
        else []
    )
    return {
        "reference_characters": len(normalized),
        "hypothesis_characters": len(normalized) + insertions,
        "distance": insertions,
        "substitutions": 0,
        "deletions": 0,
        "insertions": insertions,
        "cer": insertions / len(normalized),
        "insertion_spans": insertion_spans,
        "repetition_spans": repetitions,
    }


def _build_fixture(output_root: Path) -> None:
    source_text = "".join(f"这是第{ordinal}个测试句子。" for ordinal in range(1, 39))
    source_bytes = source_text.encode("utf-8")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    references = parse_reference_sentences(source_text)
    assert len(references) == 38
    write_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "text": {"text": source_text, "sha256": source_sha256},
            "seeds": list(SEEDS),
        },
    )
    groups: list[dict[str, Any]] = []
    sentence_cursor = 0
    for group_index, size in enumerate(GROUP_SIZES, start=1):
        selected = references[sentence_cursor : sentence_cursor + size]
        raw_start = selected[0].start
        raw_end = selected[-1].end
        groups.append(
            {
                "group_index": group_index,
                "segment_id": group_index - 1,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "text": source_text[raw_start:raw_end],
                "event_text": source_text[raw_start:raw_end],
                "provenance": "derived_from_exact_commit_text",
                "sentence_ordinals": [sentence.ordinal for sentence in selected],
            }
        )
        sentence_cursor += size
    write_json(
        output_root / "reference_groups.json",
        {
            "schema_version": 1,
            "source_arm": ArmKind.TRITON_0818.value,
            "seed_count": 3,
            "group_count": 9,
            "groups": groups,
        },
    )
    write_json(
        output_root / "scoring_summary.json",
        {
            "schema_version": 1,
            "arm_run_counts": {arm.value: 3 for arm in ARMS},
            "manifest_seed_grid_valid": True,
            "run_grid_complete": True,
            "run_count": 9,
            "expected_run_count": 9,
            "observation_count": 342,
            "expected_observation_count": 342,
            "review_ready_observation_count": 342,
            "blocked_observation_count": 0,
            "tts_failed_run_count": 0,
            "unscored_run_count": 0,
            "full_asr_failure_count": 0,
            "segment_asr_failure_count": 0,
            "asr_wav_count": 64,
            "fresh_connection_origin_wav_count": 64,
            "invalid_asr_provenance_wav_count": 0,
            "reasons": [],
            "missing_run_positions": [],
            "unexpected_run_positions": [],
            "duplicate_run_positions": [],
            "tts_failed_runs": [],
            "unscored_runs": [],
            "full_asr_failed_runs": [],
            "segment_asr_failed_runs": [],
            "invalid_asr_provenance_wavs": [],
            "ready_for_review": True,
        },
    )

    all_observations: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = output_root / "arms" / arm.value / f"seed_{seed:04d}"
            events: list[dict[str, Any]] = [
                {
                    "type": "text_boundary_commit",
                    "segment_id": 0,
                    "text": source_text,
                    "meta": {
                        "raw_codepoint_start": "0",
                        "raw_codepoint_end": "0",
                    },
                },
                {
                    "type": "segment_end",
                    "segment_id": 0,
                    "text": source_text,
                    "meta": {
                        "eos_reason": "codec_eos",
                        "loop_max_run": "2",
                        "loop_suspect_count": "0",
                        "loop_recovery_count": "0",
                        "retry_idx": "0",
                    },
                },
            ]
            if arm is ArmKind.CURRENT_HEAD and seed == SEEDS[0]:
                events.append(
                    {
                        "type": "segment_retry",
                        "segment_id": 0,
                        "meta": {"retry_idx": "1", "guard_mode": "progress"},
                    }
                )
            record = {
                "schema_version": 1,
                "arm": arm.value,
                "seed": seed,
                "session_id": f"fixture-{seed}",
                "status": RunStatus.OK.value,
                "sample_rate": 24000,
                "sample_count": 240000,
                "duration_s": 10.0,
                "source_text": source_text,
                "source_text_sha256": source_sha256,
                "events": events,
                "terminal_event": "done",
                "pcm_f32le_sha256": f"f32-{arm.value}-{seed}",
                "pcm_s16le_sha256": f"s16-{arm.value}-{seed}",
                "artifacts": {"wav": f"arms/{arm.value}/seed_{seed:04d}/full.wav"},
            }
            write_json(run_dir / "run.json", record)
            document_metrics = _metrics(source_text)
            delivered_segments: list[dict[str, Any]] = [
                {
                    "sequence_index": 0,
                    "segment_id": 0,
                    "text": source_text,
                    "duration_s": 10.0,
                    "output_sample_start": 0,
                    "output_sample_end": 240000,
                    "progress_meta": {
                        "delivery_boundary_source": "final_progress_output_end"
                    },
                    "asr": {"status": "ok", "transcript": source_text},
                    "character_errors": document_metrics,
                    "character_errors_unavailable_reason": None,
                }
            ]
            if arm is ArmKind.CURRENT_HEAD and seed == SEEDS[0]:
                delivered_segments.append(
                    {
                        "sequence_index": 1,
                        "segment_id": 1,
                        "text": "。",
                        "duration_s": 0.1,
                        "output_sample_start": 240000,
                        "output_sample_end": 242400,
                        "progress_meta": {
                            "delivery_boundary_source": "final_progress_output_end"
                        },
                        "asr": {"status": "ok", "transcript": ""},
                        "character_errors": None,
                        "character_errors_unavailable_reason": (
                            "empty_normalized_reference"
                        ),
                    }
                )
            write_json(
                run_dir / "scoring.json",
                {
                    "status": RunStatus.OK.value,
                    "full_asr": {"status": "ok", "transcript": source_text},
                    "full_asr_provenance": {"wav_sha256": f"wav-{arm.value}-{seed}"},
                    "full_character_errors": document_metrics,
                    "delivered_segments": delivered_segments,
                    "asr_failures": {"full_wav": 0, "delivered_segments": 0},
                    "acoustic": {"diagnostic_only": True, "rms": 0.1},
                },
            )
            run_observations: list[dict[str, Any]] = []
            for sentence in references:
                metrics = _metrics(sentence.text, ordinal=sentence.ordinal)
                observation = {
                    "arm": arm.value,
                    "seed": seed,
                    "session_id": f"fixture-{seed}",
                    "sentence_ordinal": sentence.ordinal,
                    "sentence_id": sentence.sentence_id,
                    "reference_text": sentence.text,
                    "status": RunStatus.REVIEW_PENDING.value,
                    "tts_run_status": RunStatus.OK.value,
                    "tts_complete": True,
                    "asr_status": "ok",
                    "valid_for_review": True,
                    "asr_transcript": sentence.text,
                    "character_errors": metrics,
                    "clip": f"sentence_{sentence.ordinal:03d}.wav",
                    "context_clip": f"sentence_{sentence.ordinal:03d}_context.wav",
                    "timing": {"start_ms": sentence.ordinal * 100},
                    "acoustic": {"diagnostic_only": True, "rms": 0.1},
                }
                run_observations.append(observation)
                all_observations.append(observation)
            write_json(
                run_dir / "sentence_observations.json",
                {"observations": run_observations},
            )
    write_json(
        output_root / "sentence_observations.json",
        {"observations": all_observations},
    )


def test_pre_review_diagnostic_report_is_traceable_diagnostic_only_and_complete(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    _build_fixture(output_root)

    report = generate_pre_review_diagnostic_report(output_root)

    assert report["grid"] == {
        "expected_run_count": 9,
        "observed_run_count": 9,
        "expected_observation_count": 342,
        "observed_observation_count": 342,
        "reference_sentence_count": 38,
        "reference_group_count": 9,
        "asr_wav_count": 64,
        "fresh_connection_origin_wav_count": 64,
        "complete": True,
    }
    assert report["policy"]["diagnostic_only"] is True
    assert report["policy"]["root_cause_gate_status"] == (
        "not_evaluated_and_cannot_be_triggered"
    )
    assert "severe_hallucination_rate" not in json.dumps(report)
    assert len(report["documents"]) == 9
    assert len(report["reference_groups"]) == 81
    assert len(report["sentences"]) == 342
    current_first = next(
        row
        for row in report["documents"]
        if row["arm"] == ArmKind.CURRENT_HEAD.value and row["seed"] == SEEDS[0]
    )
    triton_first = next(
        row
        for row in report["documents"]
        if row["arm"] == ArmKind.TRITON_0818.value and row["seed"] == SEEDS[0]
    )
    assert current_first["actual_retry_event_count"] == 1
    assert triton_first["actual_retry_event_count"] == 0
    assert report["arms"][ArmKind.CURRENT_HEAD.value]["actual_retry_run_count"] == 1
    sentence_one = next(
        row
        for row in report["sentences"]
        if row["arm"] == ArmKind.CURRENT_HEAD.value
        and row["seed"] == SEEDS[0]
        and row["sentence_ordinal"] == 1
    )
    sentence_two = next(
        row
        for row in report["sentences"]
        if row["arm"] == ArmKind.CURRENT_HEAD.value
        and row["seed"] == SEEDS[0]
        and row["sentence_ordinal"] == 2
    )
    assert sentence_one["insertion_candidate_count"] == 1
    assert sentence_two["repeat_candidate_count"] == 1
    punctuation = next(
        row for row in report["segments"] if row["reference_kind"] == "punctuation_only"
    )
    assert punctuation["cer"] is None
    assert punctuation["cer_display"] == "N/A"
    assert punctuation["cer_availability"] == "not_applicable_punctuation_only"
    assert punctuation["metrics_available"] is False

    diagnostic_root = output_root / "diagnostics"
    expected_files = {
        "evaluator_identity.json",
        "pre_review.json",
        "documents.csv",
        "reference_groups.csv",
        "sentences.csv",
        "segments.csv",
        "pre_review.md",
    }
    assert {path.name for path in diagnostic_root.iterdir()} == expected_files
    identity = read_json(diagnostic_root / "evaluator_identity.json")
    assert len(identity["source"]["tree_sha256"]) == 64
    assert set(identity["frozen_core_inputs"]) == {
        "manifest.json",
        "scoring_summary.json",
        "reference_groups.json",
    }
    markdown = (diagnostic_root / "pre_review.md").read_text(encoding="utf-8")
    assert "不能输出确认严重幻觉率" in markdown
    assert "不能评估或触发根因 gate" in markdown
    assert report["evaluator_identity"]["sha256"] in markdown
    assert "scoring_summary.json" in markdown
    assert "64/64" in markdown
    assert "FunASR 0.2.0a6 SDK" in markdown
    with (diagnostic_root / "documents.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 9
    with (diagnostic_root / "reference_groups.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 81
    with (diagnostic_root / "sentences.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 342


def test_pre_review_diagnostic_report_fails_before_output_on_missing_observation(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    _build_fixture(output_root)
    path = (
        output_root
        / "arms"
        / ArmKind.CURRENT_HEAD.value
        / "seed_0011"
        / "sentence_observations.json"
    )
    payload = read_json(path)
    write_json(path, {"observations": payload["observations"][:-1]})

    with pytest.raises(RuntimeError, match="sentence observation grid incomplete"):
        generate_pre_review_diagnostic_report(output_root)

    assert not (output_root / "diagnostics").exists()


def test_pre_review_diagnostic_report_rejects_unready_summary_and_sentence(
    tmp_path: Path,
) -> None:
    summary_root = tmp_path / "summary-failure"
    _build_fixture(summary_root)
    summary_path = summary_root / "scoring_summary.json"
    summary = read_json(summary_path)
    summary["ready_for_review"] = False
    summary["full_asr_failure_count"] = 1
    summary["reasons"] = ["full_asr_failed_runs"]
    write_json(summary_path, summary)

    with pytest.raises(RuntimeError, match="scoring summary is not review-ready"):
        generate_pre_review_diagnostic_report(summary_root)
    assert not (summary_root / "diagnostics").exists()

    sentence_root = tmp_path / "sentence-failure"
    _build_fixture(sentence_root)
    sentence_path = (
        sentence_root
        / "arms"
        / ArmKind.CURRENT_HEAD.value
        / "seed_0011"
        / "sentence_observations.json"
    )
    sentence_payload = read_json(sentence_path)
    sentence_payload["observations"][0]["valid_for_review"] = False
    write_json(sentence_path, sentence_payload)

    with pytest.raises(RuntimeError, match="sentence observation is not review-ready"):
        generate_pre_review_diagnostic_report(sentence_root)
    assert not (sentence_root / "diagnostics").exists()


def test_retry_zero_is_not_retry_and_cli_help_exposes_diagnostic_report(
    tmp_path: Path,
) -> None:
    assert (
        is_actual_retry_event(
            {"type": "segment_end", "meta": {"retry_idx": "0", "retry_count": "0"}}
        )
        is False
    )
    assert (
        is_actual_retry_event({"type": "segment_end", "meta": {"retry_idx": "1"}})
        is True
    )
    assert is_actual_retry_event({"type": "segment_retry", "meta": {}}) is True

    parser = build_parser()
    assert "diagnostic-report" in parser.format_help()
    args = parser.parse_args(
        ["diagnostic-report", "--output-dir", str(tmp_path / "experiment")]
    )
    assert args.command == "diagnostic-report"
    assert args.handler.__name__ == "command_diagnostic_report"
