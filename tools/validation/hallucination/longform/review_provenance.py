"""Final blind-review truth reconstruction and evidence hashing."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .artifacts import read_json, write_json
from .models import ReviewLabel, RunStatus
from .preflight import sha256_file
from .review import derive_severe_hallucination
from .review_selection import (
    read_review_csv,
    validate_frozen_round2_selection,
    validated_review_rows,
)


PROVENANCE_SCHEMA_VERSION = 1
REVIEW_PROTOCOL = "qwen3tts-longform-blind-review"
REVIEW_PROTOCOL_SCHEMA_VERSION = 3
PUBLIC_CONTROL_FILES = (
    "manifest.json",
    "review_round1.csv",
    "README.md",
    "index.html",
)


class AdjudicationEvidenceStatus(str, Enum):
    """How adjudication participates in one finalized evidence chain."""

    REQUIRED_PRESENT = "required_present"
    NOT_REQUIRED_ABSENT = "not_required_absent"
    NOT_REQUIRED_EMPTY = "not_required_empty"

def verify_schema_v3_public_controls(
    output_root: Path,
    key: Mapping[str, Any],
    *,
    require_schema_v3: bool,
) -> None:
    """Verify the schema-v3 key binding for immutable public control files."""

    schema_version = key.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("private review key schema_version must be an integer")
    if require_schema_v3 and schema_version != REVIEW_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("formal finalization requires a schema-v3 private review key")
    if schema_version < REVIEW_PROTOCOL_SCHEMA_VERSION:
        return
    hashes = key.get("public_control_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(PUBLIC_CONTROL_FILES):
        raise ValueError("private review key has incomplete public control file hashes")
    for relative in PUBLIC_CONTROL_FILES:
        expected = hashes.get(relative)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected.lower())
        ):
            raise ValueError(
                f"private review key has invalid public control file hash: {relative}"
            )
        path = output_root / "review" / "public" / relative
        if not path.is_file():
            raise ValueError(f"blind review control file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected.lower():
            raise ValueError(
                "blind review control file hash mismatch: "
                f"file={relative}, expected={expected.lower()}, actual={actual}"
            )


def _validated_adjudication(
    adjudication_csv: Path | None,
    *,
    disagreements: set[str],
) -> tuple[dict[str, str], AdjudicationEvidenceStatus]:
    if disagreements:
        if adjudication_csv is None:
            raise ValueError(
                "adjudication CSV is required while reviewer labels disagree"
            )
        rows = read_review_csv(adjudication_csv)
        adjudicated: dict[str, str] = {}
        for row in rows:
            blind_id = str(row.get("blind_id", "")).strip()
            if not blind_id or blind_id in adjudicated:
                raise ValueError(
                    f"missing or duplicate adjudication blind_id: {blind_id!r}"
                )
            value = str(row.get("adjudicated_label", "")).strip()
            if not value:
                raise ValueError(f"missing adjudication for {blind_id}")
            adjudicated[blind_id] = ReviewLabel(value).value
        missing = disagreements - adjudicated.keys()
        extra = adjudicated.keys() - disagreements
        if missing or extra:
            raise ValueError(
                "adjudication IDs do not exactly match reviewer disagreements: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return adjudicated, AdjudicationEvidenceStatus.REQUIRED_PRESENT

    if adjudication_csv is None:
        return {}, AdjudicationEvidenceStatus.NOT_REQUIRED_ABSENT
    if read_review_csv(adjudication_csv):
        raise ValueError("adjudication CSV must be empty when reviewers agree")
    return {}, AdjudicationEvidenceStatus.NOT_REQUIRED_EMPTY


def _reconstruct_final_rows(
    output_root: Path,
    round1_csv: Path,
    round2_csv: Path,
    adjudication_csv: Path | None,
    *,
    require_formal: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key_path = output_root / "review" / "private" / "key.json"
    key = read_json(key_path)
    verify_schema_v3_public_controls(
        output_root,
        key,
        require_schema_v3=require_formal,
    )
    round1_rows = validated_review_rows(round1_csv, require_all=True)
    round2_rows = validated_review_rows(round2_csv, require_all=True)
    validated = validate_frozen_round2_selection(
        output_root,
        round1_rows=round1_rows,
        round2_rows=round2_rows,
        require_formal=require_formal,
    )
    round1 = {row["blind_id"]: row for row in round1_rows}
    round2 = {row["blind_id"]: row for row in round2_rows}
    disagreements = {
        blind_id
        for blind_id, row in round2.items()
        if round1[blind_id]["label"] != row["label"]
    }
    adjudicated, adjudication_status = _validated_adjudication(
        adjudication_csv,
        disagreements=disagreements,
    )

    key_rows = key.get("rows")
    if not isinstance(key_rows, list):
        raise ValueError("private review key lacks rows")
    key_ids = [str(item.get("blind_id", "")).strip() for item in key_rows]
    if (
        any(not blind_id for blind_id in key_ids)
        or len(set(key_ids)) != len(key_ids)
        or set(key_ids) != set(round1)
    ):
        raise ValueError("private key and round1 submission IDs do not exactly match")

    final_rows: list[dict[str, Any]] = []
    for item in key_rows:
        blind_id = str(item["blind_id"])
        first = round1[blind_id]
        if blind_id in adjudicated:
            label = ReviewLabel(adjudicated[blind_id])
            source = "adjudication"
        elif blind_id in round2:
            label = ReviewLabel(round2[blind_id]["label"])
            source = "reviewer_agreement"
        else:
            label = ReviewLabel(first["label"])
            source = "reviewer1_unselected_negative"
        observation = dict(item["observation"])
        tts_status = observation.get(
            "tts_run_status", observation.get("status", "error")
        )
        severe = derive_severe_hallucination(label)
        valid_for_rate = isinstance(severe, bool) and tts_status == "ok"
        observation.update(
            {
                "blind_id": blind_id,
                "review_label": label.value,
                "review_resolution": source,
                "severe_hallucination": severe if valid_for_rate else None,
                "valid_for_rate": valid_for_rate,
                "status": (
                    RunStatus.INVALID.value
                    if not valid_for_rate
                    else RunStatus.REVIEWED.value
                ),
            }
        )
        final_rows.append(observation)
    return final_rows, {
        "key": key,
        "selection": validated.selection,
        "reviewer1_id": validated.reviewer1_id,
        "reviewer2_id": validated.reviewer2_id,
        "adjudication_status": adjudication_status,
    }


def _file_evidence(path: Path, *, logical_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"review evidence file is missing: {path}")
    return {
        "status": "present",
        "logical_path": logical_path,
        "source_name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _provenance_payload(
    output_root: Path,
    round1_csv: Path,
    round2_csv: Path,
    adjudication_csv: Path | None,
    *,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    adjudication_status = AdjudicationEvidenceStatus(
        context["adjudication_status"]
    )
    if adjudication_status is AdjudicationEvidenceStatus.NOT_REQUIRED_ABSENT:
        adjudication_evidence: dict[str, Any] = {
            "status": adjudication_status.value,
            "logical_path": "adjudication_submission",
            "source_name": None,
            "bytes": 0,
            "sha256": None,
        }
    else:
        if adjudication_csv is None:
            raise ValueError("adjudication evidence path is unexpectedly absent")
        adjudication_evidence = _file_evidence(
            adjudication_csv,
            logical_path="adjudication_submission",
        )
        adjudication_evidence["status"] = adjudication_status.value

    private_root = output_root / "review" / "private"
    public_root = output_root / "review" / "public"
    selection = context["selection"]
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "protocol": {
            "name": REVIEW_PROTOCOL,
            "schema_version": REVIEW_PROTOCOL_SCHEMA_VERSION,
        },
        "tool": {
            "component": "tools.validation.hallucination.longform.review_provenance",
            "schema_version": PROVENANCE_SCHEMA_VERSION,
        },
        "reviewers": {
            "round1": context["reviewer1_id"],
            "round2": context["reviewer2_id"],
        },
        "selection": {
            "schema_version": selection.get("schema_version", 1),
            "algorithm": selection.get("selection_algorithm"),
            "seed": selection.get("selection_seed"),
        },
        "evidence": {
            "private_key": _file_evidence(
                private_root / "key.json", logical_path="review/private/key.json"
            ),
            "public_manifest": _file_evidence(
                public_root / "manifest.json",
                logical_path="review/public/manifest.json",
            ),
            "round1_submission": _file_evidence(
                round1_csv,
                logical_path="round1_submission",
            ),
            "round2_submission": _file_evidence(
                round2_csv,
                logical_path="round2_submission",
            ),
            "round2_selection": _file_evidence(
                private_root / "round2_selection.json",
                logical_path="review/private/round2_selection.json",
            ),
            "adjudication": adjudication_evidence,
            "final_labels": _file_evidence(
                private_root / "final_labels.json",
                logical_path="review/private/final_labels.json",
            ),
        },
    }


def _assert_exact(expected: Any, actual: Any, *, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise ValueError(f"review evidence field mismatch at {path}")
        for key, value in expected.items():
            _assert_exact(value, actual[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"review evidence field mismatch at {path}")
        for index, value in enumerate(expected):
            _assert_exact(value, actual[index], path=f"{path}[{index}]")
        return
    if expected != actual or type(expected) is not type(actual):
        raise ValueError(
            f"review evidence field mismatch at {path}: "
            f"expected={expected!r}, actual={actual!r}"
        )


def finalize_review_evidence(
    output_root: Path,
    round1_csv: Path,
    round2_csv: Path,
    adjudication_csv: Path | None = None,
    *,
    require_formal: bool,
) -> list[dict[str, Any]]:
    """Reconstruct final truth, write it, and seal every review input hash."""

    final_rows, context = _reconstruct_final_rows(
        output_root,
        round1_csv,
        round2_csv,
        adjudication_csv,
        require_formal=require_formal,
    )
    private_root = output_root / "review" / "private"
    write_json(private_root / "final_labels.json", {"rows": final_rows})
    provenance = _provenance_payload(
        output_root,
        round1_csv,
        round2_csv,
        adjudication_csv,
        context=context,
    )
    write_json(private_root / "review_provenance.json", provenance)
    return final_rows


def verify_review_provenance(
    output_root: Path,
    round1_csv: Path,
    round2_csv: Path,
    adjudication_csv: Path | None = None,
    *,
    require_formal: bool = True,
) -> dict[str, Any]:
    """Strictly rebuild final labels and compare every field and evidence hash."""

    expected_rows, context = _reconstruct_final_rows(
        output_root,
        round1_csv,
        round2_csv,
        adjudication_csv,
        require_formal=require_formal,
    )
    private_root = output_root / "review" / "private"
    stored_final = read_json(private_root / "final_labels.json")
    _assert_exact(expected_rows, stored_final.get("rows"), path="final_labels.rows")
    expected_provenance = _provenance_payload(
        output_root,
        round1_csv,
        round2_csv,
        adjudication_csv,
        context=context,
    )
    stored_provenance = read_json(private_root / "review_provenance.json")
    _assert_exact(
        expected_provenance,
        stored_provenance,
        path="review_provenance",
    )
    return {
        "verified": True,
        "final_label_count": len(expected_rows),
        "reviewer_ids": {
            "round1": context["reviewer1_id"],
            "round2": context["reviewer2_id"],
        },
        "provenance_sha256": sha256_file(
            private_root / "review_provenance.json"
        ),
    }


__all__ = [
    "AdjudicationEvidenceStatus",
    "PROVENANCE_SCHEMA_VERSION",
    "PUBLIC_CONTROL_FILES",
    "REVIEW_PROTOCOL_SCHEMA_VERSION",
    "finalize_review_evidence",
    "verify_review_provenance",
    "verify_schema_v3_public_controls",
]
