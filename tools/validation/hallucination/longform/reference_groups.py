"""Freeze reference groups from three consistent Triton commit streams."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import discover_run_records, read_json, write_json
from .models import ArmKind, parse_reference_sentences


_DERIVED_PROVENANCE = "derived_from_exact_commit_text"
_OFFSET_PROVENANCE = "verified_event_raw_codepoint_offsets"


def _commit_events(
    events: Sequence[Mapping[str, Any]],
    reference_length: int,
) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    seen: set[int] = set()
    for event in events:
        if event.get("type") != "text_boundary_commit":
            continue
        try:
            segment_id = int(event["segment_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("commit has an invalid segment_id") from exc
        if segment_id < 0:
            raise RuntimeError("commit segment_id must be non-negative")
        if segment_id in seen:
            raise RuntimeError(f"duplicate commit segment_id: {segment_id}")
        seen.add(segment_id)
        text = event.get("text")
        if not isinstance(text, str) or not text:
            raise RuntimeError(f"commit {segment_id} has empty or non-text content")

        meta = dict(event.get("meta") or {})
        has_start = "raw_codepoint_start" in meta
        has_end = "raw_codepoint_end" in meta
        if has_start != has_end:
            raise RuntimeError(f"commit {segment_id} has a partial offset pair")
        raw_start: int | None = None
        raw_end: int | None = None
        offset_valid = False
        if has_start and has_end:
            try:
                candidate_start = int(meta["raw_codepoint_start"])
                candidate_end = int(meta["raw_codepoint_end"])
            except (TypeError, ValueError):
                pass
            else:
                if 0 <= candidate_start < candidate_end <= reference_length:
                    raw_start = candidate_start
                    raw_end = candidate_end
                    offset_valid = True
        commits.append(
            {
                "segment_id": segment_id,
                "event_text": text,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "offset_valid": offset_valid,
            }
        )

    commits.sort(key=lambda commit: commit["segment_id"])
    if not commits:
        raise RuntimeError("Triton run has no text_boundary_commit events")
    segment_ids = [commit["segment_id"] for commit in commits]
    if segment_ids != list(range(len(commits))):
        raise RuntimeError(
            "commit segment_ids must be continuous, unique, and start at zero"
        )
    return commits


def _groups_from_verified_offsets(
    candidates: Sequence[Sequence[Mapping[str, Any]]],
    reference_text: str,
) -> list[dict[str, Any]]:
    canonical: list[list[tuple[int, int, int, str]]] = []
    for commits in candidates:
        cursor = 0
        current: list[tuple[int, int, int, str]] = []
        for commit in commits:
            raw_start = int(commit["raw_start"])
            raw_end = int(commit["raw_end"])
            if raw_start != cursor:
                raise RuntimeError("commit offsets contain a gap, overlap, or reorder")
            expected_text = reference_text[raw_start:raw_end]
            if commit["event_text"] != expected_text:
                raise RuntimeError("commit text disagrees with its raw offsets")
            current.append(
                (int(commit["segment_id"]), raw_start, raw_end, expected_text)
            )
            cursor = raw_end
        if cursor != len(reference_text):
            raise RuntimeError("commit offsets do not cover the complete source text")
        canonical.append(current)
    if any(candidate != canonical[0] for candidate in canonical[1:]):
        raise RuntimeError("prototype commit groups differ between deterministic seeds")
    return [
        {
            "segment_id": segment_id,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "text": text,
            "event_text": text,
            "provenance": _OFFSET_PROVENANCE,
        }
        for segment_id, raw_start, raw_end, text in canonical[0]
    ]


def _groups_from_exact_commit_text(
    candidates: Sequence[Sequence[Mapping[str, Any]]],
    reference_text: str,
) -> list[dict[str, Any]]:
    text_arrays = [
        [str(commit["event_text"]) for commit in commits] for commits in candidates
    ]
    if any(texts != text_arrays[0] for texts in text_arrays[1:]):
        raise RuntimeError("prototype commit texts differ between deterministic seeds")
    if "".join(text_arrays[0]) != reference_text:
        raise RuntimeError("commit texts do not reconstruct the exact source text")

    groups: list[dict[str, Any]] = []
    cursor = 0
    for segment_id, text in enumerate(text_arrays[0]):
        raw_end = cursor + len(text)
        groups.append(
            {
                "segment_id": segment_id,
                "raw_start": cursor,
                "raw_end": raw_end,
                "text": text,
                "event_text": text,
                "provenance": _DERIVED_PROVENANCE,
            }
        )
        cursor = raw_end
    return groups


def validate_reference_groups(
    groups: Sequence[Mapping[str, Any]],
    reference_text: str,
) -> list[dict[str, Any]]:
    """Require a non-empty, continuous group list that exactly covers text."""

    frozen = [dict(group) for group in groups]
    if not frozen:
        raise RuntimeError("frozen reference groups must not be empty")
    try:
        segment_ids = [int(group["segment_id"]) for group in frozen]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("frozen group has an invalid segment_id") from exc
    if segment_ids != list(range(len(frozen))):
        raise RuntimeError(
            "frozen group segment_ids must be continuous, unique, and start at zero"
        )

    cursor = 0
    for group in frozen:
        try:
            raw_start = int(group["raw_start"])
            raw_end = int(group["raw_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("frozen group has invalid raw offsets") from exc
        text = group.get("text")
        if not isinstance(text, str) or not text:
            raise RuntimeError("frozen group has empty or non-text content")
        if raw_start != cursor or not raw_start < raw_end <= len(reference_text):
            raise RuntimeError("frozen groups contain a gap, overlap, or invalid span")
        if text != reference_text[raw_start:raw_end]:
            raise RuntimeError("frozen group text disagrees with source offsets")
        if "event_text" in group and group["event_text"] != text:
            raise RuntimeError("frozen group event_text disagrees with source text")
        cursor = raw_end
    if cursor != len(reference_text):
        raise RuntimeError("frozen groups do not reconstruct the complete source text")
    return frozen


def freeze_reference_groups(output_root: Path) -> list[dict[str, Any]]:
    """Freeze the group count observed identically in all three Triton runs."""

    manifest = read_json(output_root / "manifest.json")
    reference_text = str(manifest["text"]["text"])
    target = output_root / "reference_groups.json"
    expected_seeds = [int(seed) for seed in manifest.get("seeds") or []]
    if len(expected_seeds) != 3 or len(set(expected_seeds)) != 3:
        raise RuntimeError("reference groups require exactly three distinct trial seeds")
    records = [
        record
        for record in discover_run_records(output_root)
        if record.get("arm") == ArmKind.TRITON_0818.value
        and record.get("status") == "ok"
    ]
    by_seed: dict[int, Mapping[str, Any]] = {}
    for record in records:
        seed = int(record["seed"])
        if seed in by_seed:
            raise RuntimeError(f"duplicate successful Triton run for seed {seed}")
        by_seed[seed] = record
    if set(by_seed) != set(expected_seeds):
        raise RuntimeError("successful Triton runs do not match the three frozen seeds")

    candidates = [
        _commit_events(
            list(by_seed[seed].get("events") or []),
            len(reference_text),
        )
        for seed in expected_seeds
    ]
    offset_states = [
        bool(commit["offset_valid"])
        for commits in candidates
        for commit in commits
    ]
    if all(offset_states):
        provenance = _OFFSET_PROVENANCE
        groups = _groups_from_verified_offsets(candidates, reference_text)
    elif any(offset_states):
        raise RuntimeError("commit offsets are only partially valid across Triton runs")
    else:
        provenance = _DERIVED_PROVENANCE
        groups = _groups_from_exact_commit_text(candidates, reference_text)

    groups = validate_reference_groups(groups, reference_text)
    references = parse_reference_sentences(reference_text)
    for group_index, group in enumerate(groups, start=1):
        group["group_index"] = group_index
        group["sentence_ordinals"] = [
            sentence.ordinal
            for sentence in references
            if sentence.end > group["raw_start"] and sentence.start < group["raw_end"]
        ]
    payload = {
        "schema_version": 1,
        "source_arm": ArmKind.TRITON_0818.value,
        "seed_count": len(expected_seeds),
        "seeds": expected_seeds,
        "group_count": len(groups),
        "provenance": provenance,
        "groups": groups,
    }
    if target.is_file():
        frozen_payload = read_json(target)
        frozen_groups = validate_reference_groups(
            frozen_payload.get("groups") or [], reference_text
        )
        if int(frozen_payload.get("group_count", -1)) != len(frozen_groups):
            raise RuntimeError("frozen group_count disagrees with groups")
        if frozen_payload != payload:
            raise RuntimeError("existing frozen groups differ from Triton commit evidence")
        return frozen_groups
    write_json(target, payload)
    return groups


__all__ = ["freeze_reference_groups", "validate_reference_groups"]
