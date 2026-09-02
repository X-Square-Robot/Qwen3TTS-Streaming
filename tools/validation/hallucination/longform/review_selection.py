"""Private reviewer identities and deterministic round-2 selection."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import read_json
from .models import ReviewLabel


ROUND2_SELECTION_SCHEMA_VERSION = 2
ROUND2_SELECTION_ALGORITHM = "python-random-shuffle-v1"


@dataclass(frozen=True)
class Round2Selection:
    """Deterministic second-review selection in public-manifest order."""

    mandatory_blind_ids: tuple[str, ...]
    random_negative_blind_ids: tuple[str, ...]
    blind_ids: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedSelection:
    """Frozen selection joined to reviewer identities."""

    selection: Mapping[str, Any]
    expected: Round2Selection
    reviewer1_id: str | None
    reviewer2_id: str | None


def read_review_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8/BOM-compatible review CSV."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def validated_review_rows(
    path: Path,
    *,
    require_all: bool,
) -> list[dict[str, str]]:
    """Validate review IDs and labels without adding private identity columns."""

    rows = read_review_csv(path)
    seen: set[str] = set()
    for row in rows:
        blind_id = str(row.get("blind_id", "")).strip()
        if not blind_id or blind_id in seen:
            raise ValueError(f"missing or duplicate blind_id in {path}: {blind_id!r}")
        seen.add(blind_id)
        value = str(row.get("label", "")).strip()
        if require_all and not value:
            raise ValueError(f"missing label for {blind_id} in {path}")
        if value:
            row["label"] = ReviewLabel(value).value
        row["blind_id"] = blind_id
    return rows


def validate_reviewer_pair(
    reviewer1_id: object,
    reviewer2_id: object,
) -> tuple[str, str]:
    """Return two canonical non-empty and distinct private reviewer IDs."""

    normalized: list[str] = []
    for role, value in (("round1", reviewer1_id), ("round2", reviewer2_id)):
        if not isinstance(value, str):
            raise ValueError(f"{role} reviewer identity is required")
        identity = value.strip()
        if not identity or any(character in identity for character in "\r\n\0"):
            raise ValueError(f"{role} reviewer identity is required and must be one line")
        normalized.append(identity)
    if normalized[0].casefold() == normalized[1].casefold():
        raise ValueError("round1 and round2 reviewer identities must be different")
    return normalized[0], normalized[1]


def _rows_in_public_order(
    public_rows: Sequence[Mapping[str, Any]],
    submitted_rows: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    public_ids = [str(row.get("blind_id", "")).strip() for row in public_rows]
    if not public_ids or any(not blind_id for blind_id in public_ids):
        raise ValueError("public review manifest has missing blind_id")
    if len(set(public_ids)) != len(public_ids):
        raise ValueError("public review manifest has duplicate blind_id")
    submitted = {str(row["blind_id"]): row for row in submitted_rows}
    missing = set(public_ids) - submitted.keys()
    extra = submitted.keys() - set(public_ids)
    if missing or extra or len(submitted) != len(submitted_rows):
        raise ValueError(
            "round1 IDs do not match the sealed public manifest: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return [submitted[blind_id] for blind_id in public_ids]


def compute_round2_selection(
    public_rows: Sequence[Mapping[str, Any]],
    round1_rows: Sequence[Mapping[str, str]],
    *,
    selection_seed: int,
) -> Round2Selection:
    """Recompute exact mandatory and 10% negative IDs deterministically."""

    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise ValueError("selection_seed must be an integer")
    ordered = _rows_in_public_order(public_rows, round1_rows)
    mandatory = [
        str(row["blind_id"])
        for row in ordered
        if ReviewLabel(row["label"]) is not ReviewLabel.OK
    ]
    remaining = [
        str(row["blind_id"])
        for row in ordered
        if ReviewLabel(row["label"]) is ReviewLabel.OK
    ]
    random.Random(selection_seed).shuffle(remaining)
    random_negative = remaining[: math.ceil(len(remaining) * 0.10)]
    selected = [*mandatory, *random_negative]
    random.Random(selection_seed + 1).shuffle(selected)
    return Round2Selection(
        mandatory_blind_ids=tuple(mandatory),
        random_negative_blind_ids=tuple(random_negative),
        blind_ids=tuple(selected),
    )


def selection_payload(
    selection: Round2Selection,
    *,
    selection_seed: int,
    reviewer1_id: object,
    reviewer2_id: object,
) -> dict[str, Any]:
    """Create the private schema-v2 round-2 selection sidecar."""

    first, second = validate_reviewer_pair(reviewer1_id, reviewer2_id)
    return {
        "schema_version": ROUND2_SELECTION_SCHEMA_VERSION,
        "selection_algorithm": ROUND2_SELECTION_ALGORITHM,
        "selection_seed": selection_seed,
        "reviewer_ids": {"round1": first, "round2": second},
        "mandatory_count": len(selection.mandatory_blind_ids),
        "random_negative_count": len(selection.random_negative_blind_ids),
        "mandatory_blind_ids": list(selection.mandatory_blind_ids),
        "random_negative_blind_ids": list(selection.random_negative_blind_ids),
        "blind_ids": list(selection.blind_ids),
    }


def validate_frozen_round2_selection(
    output_root: Path,
    *,
    round1_rows: Sequence[Mapping[str, str]],
    round2_rows: Sequence[Mapping[str, str]],
    require_formal: bool,
) -> ValidatedSelection:
    """Recompute and verify the exact frozen selection and submitted ID set."""

    selection_path = output_root / "review" / "private" / "round2_selection.json"
    if not selection_path.is_file():
        raise ValueError("frozen round2_selection.json is required for finalization")
    selection = read_json(selection_path)
    schema_version = selection.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("round2 selection schema_version must be an integer")
    if require_formal and schema_version != ROUND2_SELECTION_SCHEMA_VERSION:
        raise ValueError(
            "formal finalization requires schema-v2 round2 selection evidence"
        )
    seed = selection.get("selection_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("frozen round2 selection lacks an integer selection_seed")
    public_manifest = read_json(output_root / "review" / "public" / "manifest.json")
    public_rows = list(public_manifest.get("rows") or [])
    expected = compute_round2_selection(
        public_rows,
        round1_rows,
        selection_seed=seed,
    )

    frozen_ids = selection.get("blind_ids")
    if not isinstance(frozen_ids, list) or any(
        not isinstance(value, str) or not value.strip() for value in frozen_ids
    ):
        raise ValueError("frozen round2 selection lacks a valid blind_ids list")
    if list(expected.blind_ids) != frozen_ids:
        missing_mandatory = set(expected.mandatory_blind_ids) - set(frozen_ids)
        if missing_mandatory:
            raise ValueError(
                "frozen round2 selection omits mandatory non-OK/uncertain IDs: "
                f"{sorted(missing_mandatory)}"
            )
        raise ValueError(
            "frozen round2 selection does not match exact seeded 10% selection"
        )
    if schema_version >= ROUND2_SELECTION_SCHEMA_VERSION:
        expected_fields = {
            "selection_algorithm": ROUND2_SELECTION_ALGORITHM,
            "mandatory_count": len(expected.mandatory_blind_ids),
            "random_negative_count": len(expected.random_negative_blind_ids),
            "mandatory_blind_ids": list(expected.mandatory_blind_ids),
            "random_negative_blind_ids": list(expected.random_negative_blind_ids),
        }
        for field, expected_value in expected_fields.items():
            if selection.get(field) != expected_value:
                raise ValueError(
                    f"frozen round2 selection field does not match recomputation: {field}"
                )

    submitted_ids = [str(row["blind_id"]) for row in round2_rows]
    missing = set(expected.blind_ids) - set(submitted_ids)
    extra = set(submitted_ids) - set(expected.blind_ids)
    if missing or extra or len(submitted_ids) != len(set(submitted_ids)):
        raise ValueError(
            "round2 IDs do not match frozen selection: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    raw_reviewers = selection.get("reviewer_ids")
    if isinstance(raw_reviewers, Mapping):
        first, second = validate_reviewer_pair(
            raw_reviewers.get("round1"), raw_reviewers.get("round2")
        )
    elif require_formal or schema_version >= ROUND2_SELECTION_SCHEMA_VERSION:
        raise ValueError("formal finalization requires two private reviewer identities")
    else:
        first, second = None, None
    return ValidatedSelection(
        selection=selection,
        expected=expected,
        reviewer1_id=first,
        reviewer2_id=second,
    )

__all__ = [
    "ROUND2_SELECTION_ALGORITHM",
    "ROUND2_SELECTION_SCHEMA_VERSION",
    "Round2Selection",
    "ValidatedSelection",
    "compute_round2_selection",
    "read_review_csv",
    "selection_payload",
    "validate_frozen_round2_selection",
    "validate_reviewer_pair",
    "validated_review_rows",
]
