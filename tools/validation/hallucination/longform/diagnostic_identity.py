"""Frozen evaluator identity for reproducible pre-review diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import read_json, write_json
from .diagnostic_grid import DiagnosticInputs
from .preflight import sha256_file, source_tree_identity


def freeze_diagnostic_evaluator_identity(
    inputs: DiagnosticInputs,
) -> dict[str, Any]:
    """Write once and reject a changed evaluator or frozen core input identity."""

    repo_root = Path(__file__).resolve().parents[4]
    required_inputs = {
        artifact["path"]: artifact
        for artifact in inputs.input_artifacts
        if artifact["path"]
        in {"manifest.json", "scoring_summary.json", "reference_groups.json"}
    }
    if set(required_inputs) != {
        "manifest.json",
        "scoring_summary.json",
        "reference_groups.json",
    }:
        raise RuntimeError(
            "diagnostic evaluator identity is missing a frozen core input"
        )
    payload = {
        "schema_version": 1,
        "evaluator": "longform_pre_review_diagnostics",
        "diagnostic_only": True,
        "source": source_tree_identity(
            repo_root,
            [Path("tools/validation/hallucination/longform")],
        ),
        "frozen_core_inputs": required_inputs,
        "policy": {
            "human_blind_review_is_final_truth": True,
            "truth_rate_status": "unavailable_before_blind_review",
            "root_cause_gate_status": "not_evaluated_and_cannot_be_triggered",
        },
    }
    path = inputs.output_root / "diagnostics" / "evaluator_identity.json"
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError(
                "pre-review diagnostic evaluator/input identity changed after freeze"
            )
    else:
        write_json(path, payload)
    return {
        "path": str(path.relative_to(inputs.output_root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_tree_sha256": payload["source"]["tree_sha256"],
    }


__all__ = ["freeze_diagnostic_evaluator_identity"]
