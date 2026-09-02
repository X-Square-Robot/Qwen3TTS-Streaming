"""Fail-closed admission gate for matched long-form replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import read_json
from .evidence_inventory import artifact_identity
from .formal_statistics import (
    FORMAL_BOOTSTRAP_CONFIDENCE,
    require_formal_bootstrap_protocol,
)
from .models import ArmKind, ReviewLabel, RunStatus, parse_reference_sentences
from .rate_statistics import (
    evaluate_comparison_gate,
    evaluate_invalidity,
    paired_bootstrap_comparison,
    wilson_interval,
)
from .reporting import REPORT_PROVENANCE_PATH


_PRIMARY_COMPARISONS = {
    (ArmKind.TRITON_0818.value, ArmKind.CURRENT_HEAD.value),
    (ArmKind.PYTORCH_0818.value, ArmKind.CURRENT_HEAD.value),
    (ArmKind.PYTORCH_0818.value, ArmKind.TRITON_0818.value),
}
_PRIMARY_ARMS = tuple(arm.value for arm in ArmKind)
_SEVERE_REVIEW_LABELS = frozenset(
    {
        ReviewLabel.SINGLE_UNIT_LOOP,
        ReviewLabel.ABNORMAL_NOISE,
        ReviewLabel.UNSUPPORTED_SPEECH,
    }
)


def _validated_final_truth(
    primary_root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Require the exact frozen 3 x 3 x 38 adjudicated sentence grid."""

    seeds = manifest.get("seeds")
    text_payload = manifest.get("text")
    source_text = (
        text_payload.get("text") if isinstance(text_payload, Mapping) else None
    )
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        or len(set(seeds)) != 3
        or not isinstance(source_text, str)
    ):
        raise RuntimeError("primary manifest does not define the frozen truth grid")
    sentences = parse_reference_sentences(source_text)
    expected = {
        (arm, seed, sentence.ordinal)
        for arm in _PRIMARY_ARMS
        for seed in seeds
        for sentence in sentences
    }
    if len(sentences) != 38 or len(expected) != 342:
        raise RuntimeError("primary truth grid must contain exactly 342 positions")

    payload = read_json(primary_root / "review" / "private" / "final_labels.json")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 342:
        raise RuntimeError("primary final labels must contain exactly 342 rows")
    sentence_by_ordinal = {sentence.ordinal: sentence for sentence in sentences}
    rows: list[dict[str, Any]] = []
    positions: set[tuple[str, int, int]] = set()
    blind_ids: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError("primary final labels contain a non-object row")
        row = dict(raw)
        arm = row.get("arm")
        seed = row.get("seed")
        ordinal = row.get("sentence_ordinal")
        if (
            not isinstance(arm, str)
            or arm not in _PRIMARY_ARMS
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed not in seeds
            or not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
        ):
            raise RuntimeError("final labels contain an invalid grid coordinate")
        position = (arm, seed, ordinal)
        if position not in expected or position in positions:
            raise RuntimeError(f"invalid or duplicate final-label position: {position}")
        positions.add(position)
        blind_id = row.get("blind_id")
        if not isinstance(blind_id, str) or not blind_id or blind_id in blind_ids:
            raise RuntimeError("final labels require unique non-empty blind IDs")
        blind_ids.add(blind_id)
        sentence = sentence_by_ordinal[ordinal]
        if (
            row.get("sentence_id") != sentence.sentence_id
            or row.get("reference_text") != sentence.text
        ):
            raise RuntimeError(f"final-label sentence identity mismatch: {position}")
        raw_label = row.get("review_label")
        try:
            label = ReviewLabel(raw_label)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid final review label: {position}") from exc
        if raw_label != label.value:
            raise RuntimeError(f"non-canonical final review label: {position}")
        expected_severe = (
            None if label is ReviewLabel.UNSCORABLE else label in _SEVERE_REVIEW_LABELS
        )
        valid = row.get("valid_for_rate")
        severe = row.get("severe_hallucination")
        if valid is True:
            if (
                not isinstance(severe, bool)
                or severe is not expected_severe
                or row.get("status") != RunStatus.REVIEWED.value
            ):
                raise RuntimeError(f"invalid reviewed truth row: {position}")
        elif valid is False:
            if (
                expected_severe is not None
                or severe is not None
                or row.get("status") != RunStatus.INVALID.value
            ):
                raise RuntimeError(f"invalid excluded truth row: {position}")
        else:
            raise RuntimeError(f"final truth row lacks valid_for_rate: {position}")
        rows.append(row)
    if positions != expected:
        raise RuntimeError("primary final-label grid is incomplete")
    return rows


def _validate_report_hash_binding(
    primary_root: Path,
    report: Mapping[str, Any],
) -> None:
    """Verify the report/input/provenance content-addressed evidence chain."""

    paths = {
        "experiment_manifest": primary_root / "manifest.json",
        "reference_groups": primary_root / "reference_groups.json",
        "final_labels": primary_root / "review" / "private" / "final_labels.json",
        "report": primary_root / "report" / "report.json",
    }
    actual = {
        name: artifact_identity(primary_root, path) for name, path in paths.items()
    }
    report_inputs = report.get("report_inputs")
    if not isinstance(report_inputs, Mapping) or dict(report_inputs) != {
        name: actual[name]
        for name in ("experiment_manifest", "reference_groups", "final_labels")
    }:
        raise RuntimeError("primary report input hashes do not match current artifacts")
    if report.get("report_provenance") != str(REPORT_PROVENANCE_PATH):
        raise RuntimeError("primary report does not name its frozen provenance")
    provenance = read_json(primary_root / REPORT_PROVENANCE_PATH)
    if (
        provenance.get("schema_version") != 1
        or provenance.get("artifact_role") != "primary_arm_blind_truth_report"
        or provenance.get("truth_source") != "arm-blind human review"
        or provenance.get("artifacts") != actual
    ):
        raise RuntimeError("primary report provenance/hash binding is invalid")


def _validate_completeness(report: Mapping[str, Any]) -> None:
    completeness = report.get("observation_completeness")
    if not isinstance(completeness, Mapping):
        raise RuntimeError("primary report observation completeness is invalid")
    completeness_arms = completeness.get("arms")
    if not isinstance(completeness_arms, Mapping) or set(completeness_arms) != set(
        _PRIMARY_ARMS
    ):
        raise RuntimeError("primary report completeness arm structure is invalid")
    expected_values = {
        "expected": 114,
        "observed_rows": 114,
        "observed_expected_positions": 114,
        "missing_observations": 0,
        "duplicate_positions": 0,
        "duplicate_extra_rows": 0,
        "unexpected_rows": 0,
        "all_expected_positions_present": True,
        "integrity_complete": True,
    }
    for arm in _PRIMARY_ARMS:
        values = completeness_arms[arm]
        if not isinstance(values, Mapping) or any(
            values.get(field) != expected for field, expected in expected_values.items()
        ):
            raise RuntimeError(f"primary report completeness is invalid for arm {arm}")


def _validate_primary_metrics(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Recompute every primary rate and comparison instead of trusting flags."""

    counts = {
        arm: {
            "total": 114,
            "invalid": sum(
                row.get("arm") == arm and row.get("valid_for_rate") is not True
                for row in rows
            ),
        }
        for arm in _PRIMARY_ARMS
    }
    recomputed_invalidity = evaluate_invalidity(counts)
    reported_invalidity = report.get("invalidity")
    if not isinstance(reported_invalidity, Mapping):
        raise RuntimeError("primary report is missing invalidity results")
    for field, value in recomputed_invalidity.items():
        if reported_invalidity.get(field) != value:
            raise RuntimeError("primary report invalidity disagrees with final labels")
    grid_integrity = reported_invalidity.get("observation_grid_integrity")
    if (
        not isinstance(grid_integrity, Mapping)
        or grid_integrity.get("complete") is not True
    ):
        raise RuntimeError("primary report observation grid is not complete")
    _validate_completeness(report)

    arm_summaries = report.get("arms")
    if not isinstance(arm_summaries, Mapping) or set(arm_summaries) != set(
        _PRIMARY_ARMS
    ):
        raise RuntimeError("primary report arm summary structure is invalid")
    for arm in _PRIMARY_ARMS:
        selected = [row for row in rows if row.get("arm") == arm]
        valid = [row for row in selected if row.get("valid_for_rate") is True]
        positives = sum(row.get("severe_hallucination") is True for row in valid)
        expected_rate = wilson_interval(
            positives,
            len(valid),
            confidence=FORMAL_BOOTSTRAP_CONFIDENCE,
        )
        arm_summary = arm_summaries[arm]
        if not isinstance(arm_summary, Mapping):
            raise RuntimeError(f"primary arm summary is not an object: {arm}")
        rate = arm_summary.get("sentence_rate")
        if not isinstance(rate, Mapping) or any(
            rate.get(field) != value for field, value in expected_rate.items()
        ):
            raise RuntimeError(f"primary sentence rate disagrees for arm {arm}")
        if (
            rate.get("expected") != 114
            or rate.get("observed") != 114
            or rate.get("missing") != 0
            or rate.get("invalid") != 114 - len(valid)
        ):
            raise RuntimeError(f"primary sentence-rate grid is invalid for arm {arm}")

    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 3:
        raise RuntimeError("primary report must contain three paired comparisons")
    observed_pairs: set[tuple[str, str]] = set()
    pair_gate_results: dict[tuple[str, str], bool] = {}
    any_gate_passed = False
    for item in comparisons:
        if not isinstance(item, Mapping):
            raise RuntimeError("primary report contains a non-object comparison")
        bootstrap = item.get("bootstrap")
        if not isinstance(bootstrap, Mapping):
            raise RuntimeError("primary comparison lacks paired bootstrap evidence")
        baseline_arm = bootstrap.get("baseline_arm")
        comparison_arm = bootstrap.get("comparison_arm")
        if not isinstance(baseline_arm, str) or not isinstance(comparison_arm, str):
            raise RuntimeError("primary comparison arms must be strings")
        pair = (baseline_arm, comparison_arm)
        if pair not in _PRIMARY_COMPARISONS or pair in observed_pairs:
            raise RuntimeError(f"unexpected or duplicate primary comparison: {pair}")
        observed_pairs.add(pair)
        try:
            require_formal_bootstrap_protocol(bootstrap)
            recomputed = paired_bootstrap_comparison(
                rows,
                pair[0],
                pair[1],
                iterations=bootstrap["iterations"],
                confidence=bootstrap["confidence"],
                random_seed=bootstrap["random_seed"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid primary bootstrap configuration: {pair}"
            ) from exc
        if dict(bootstrap) != recomputed:
            raise RuntimeError(f"primary bootstrap disagrees with final labels: {pair}")
        gate = item.get("gate")
        recomputed_gate = evaluate_comparison_gate(
            recomputed, invalidity=recomputed_invalidity
        )
        if not isinstance(gate, Mapping) or dict(gate) != recomputed_gate:
            raise RuntimeError(f"primary comparison gate is not reproducible: {pair}")
        pair_gate_results[pair] = recomputed_gate["passes"]
        any_gate_passed = any_gate_passed or recomputed_gate["passes"]
    if observed_pairs != _PRIMARY_COMPARISONS:
        raise RuntimeError("primary comparison matrix is incomplete")
    if recomputed_invalidity["insufficient"]:
        raise RuntimeError(
            "primary result is insufficient under the invalid-sample rule"
        )
    current_vs_triton_passed = pair_gate_results[
        (ArmKind.TRITON_0818.value, ArmKind.CURRENT_HEAD.value)
    ]
    expected_conclusion = (
        "DEFAULT_GAP_REQUIRES_MATCHED_REPLAY"
        if current_vs_triton_passed
        else "LARGE_IMPLEMENTATION_GAP_REQUIRES_MATCHED_REPLAY"
        if any_gate_passed
        else "NO_PREDECLARED_LARGE_RATE_GAP"
    )
    if report.get("conclusion") != expected_conclusion:
        raise RuntimeError("primary conclusion disagrees with recomputed metrics")
    return any_gate_passed


def require_primary_truth_gate(primary_root: Path) -> dict[str, Any]:
    """Return the frozen report only when all primary truth evidence validates."""

    report_path = primary_root / "report" / "report.json"
    if not report_path.is_file():
        raise RuntimeError(
            "primary blind-truth report must exist before matched replay"
        )
    report = read_json(report_path)
    if (
        report.get("schema_version") != 1
        or report.get("truth_source") != "arm-blind human review"
        or report.get("asr_and_acoustics_are_diagnostic_only") is not True
        or report.get("expected_sentences_per_arm") != 114
        or report.get("expected_sentence_observations_total") != 342
    ):
        raise RuntimeError("primary report is not an arm-blind 342-row truth report")
    completeness = report.get("observation_completeness")
    if (
        not isinstance(completeness, Mapping)
        or completeness.get("total_expected") != 342
        or completeness.get("integrity_complete") is not True
    ):
        raise RuntimeError("primary report does not contain a complete 342-row grid")
    manifest = read_json(primary_root / "manifest.json")
    rows = _validated_final_truth(primary_root, manifest)
    _validate_report_hash_binding(primary_root, report)
    if not _validate_primary_metrics(report, rows):
        raise RuntimeError("primary rate comparison did not pass the root-cause gate")
    return report


__all__ = ["require_primary_truth_gate"]
