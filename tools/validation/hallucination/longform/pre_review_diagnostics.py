"""Assembly entry point for fail-closed pre-review diagnostic reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .diagnostic_grid import load_diagnostic_inputs
from .diagnostic_identity import freeze_diagnostic_evaluator_identity
from .diagnostic_rendering import write_pre_review_outputs
from .diagnostic_rows import (
    aggregate_arms,
    build_document_rows,
    build_reference_group_rows,
    build_segment_rows,
    build_sentence_rows,
)


def generate_pre_review_diagnostic_report(output_root: Path) -> dict[str, Any]:
    """Generate diagnostics that cannot substitute for blind-listening truth."""

    inputs = load_diagnostic_inputs(output_root)
    documents = build_document_rows(inputs)
    sentences = build_sentence_rows(inputs)
    reference_groups = build_reference_group_rows(inputs, sentences)
    segments = build_segment_rows(inputs)
    evaluator_identity = freeze_diagnostic_evaluator_identity(inputs)
    report = {
        "schema_version": 1,
        "report_kind": "pre_review_diagnostics",
        "policy": {
            "diagnostic_only": True,
            "human_blind_review_is_final_truth": True,
            "truth_rate_status": "unavailable_before_blind_review",
            "root_cause_gate_status": "not_evaluated_and_cannot_be_triggered",
            "asr_acoustic_telemetry_must_not_create_positive_labels": True,
            "asr_terminal_validation_scope": "funasrnano_sdk_event_surface_only",
        },
        "grid": {
            "expected_run_count": inputs.expected_run_count,
            "observed_run_count": len(inputs.records),
            "expected_observation_count": inputs.expected_observation_count,
            "observed_observation_count": len(sentences),
            "reference_sentence_count": len(inputs.sentences),
            "reference_group_count": len(inputs.reference_groups),
            "asr_wav_count": int(inputs.scoring_summary["asr_wav_count"]),
            "fresh_connection_origin_wav_count": int(
                inputs.scoring_summary["fresh_connection_origin_wav_count"]
            ),
            "complete": True,
        },
        "evaluator_identity": evaluator_identity,
        "input_artifacts": list(inputs.input_artifacts),
        "arms": aggregate_arms(documents),
        "documents": documents,
        "reference_groups": reference_groups,
        "sentences": sentences,
        "segments": segments,
        "outputs": {
            "evaluator_identity": "diagnostics/evaluator_identity.json",
            "json": "diagnostics/pre_review.json",
            "markdown": "diagnostics/pre_review.md",
            "documents_csv": "diagnostics/documents.csv",
            "reference_groups_csv": "diagnostics/reference_groups.csv",
            "sentences_csv": "diagnostics/sentences.csv",
            "segments_csv": "diagnostics/segments.csv",
        },
    }
    write_pre_review_outputs(output_root, report)
    return report


__all__ = ["generate_pre_review_diagnostic_report"]
