"""Fail-closed input grid for pre-review long-form diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import discover_run_records, read_json
from .models import (
    ArmKind,
    ReferenceSentence,
    RunStatus,
    parse_reference_sentences,
)
from .preflight import sha256_file
from .reference_groups import validate_reference_groups


ARMS = (
    ArmKind.CURRENT_HEAD,
    ArmKind.TRITON_0818,
    ArmKind.PYTORCH_0818,
)


@dataclass(frozen=True)
class DiagnosticInputs:
    """Validated immutable inputs for one diagnostic report generation."""

    output_root: Path
    manifest: dict[str, Any]
    seeds: tuple[int, ...]
    sentences: tuple[ReferenceSentence, ...]
    reference_groups_payload: dict[str, Any]
    reference_groups: tuple[dict[str, Any], ...]
    scoring_summary: Mapping[str, Any]
    records: Mapping[tuple[str, int], dict[str, Any]]
    scoring: Mapping[tuple[str, int], dict[str, Any]]
    observations: Mapping[tuple[str, int], tuple[dict[str, Any], ...]]
    input_artifacts: tuple[dict[str, Any], ...]

    @property
    def expected_run_count(self) -> int:
        return len(ARMS) * len(self.seeds)

    @property
    def expected_observation_count(self) -> int:
        return self.expected_run_count * len(self.sentences)


def _identity(output_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(output_root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _manifest_grid(
    manifest: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[ReferenceSentence, ...]]:
    raw_seeds = list(manifest.get("seeds") or [])
    if (
        len(raw_seeds) != 3
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) for seed in raw_seeds
        )
        or len(set(raw_seeds)) != 3
    ):
        raise RuntimeError(
            "diagnostic report requires exactly three unique integer seeds"
        )
    text_payload = manifest.get("text")
    source_text = (
        text_payload.get("text") if isinstance(text_payload, Mapping) else None
    )
    if not isinstance(source_text, str):
        raise RuntimeError("formal manifest is missing its exact source text")
    sentences = parse_reference_sentences(source_text)
    if len(sentences) != 38:
        raise RuntimeError(
            f"diagnostic report requires the frozen 38-sentence stimulus, found {len(sentences)}"
        )
    return tuple(int(seed) for seed in raw_seeds), sentences


def _validated_groups(
    payload: Mapping[str, Any],
    source_text: str,
    sentences: tuple[ReferenceSentence, ...],
) -> tuple[dict[str, Any], ...]:
    groups = validate_reference_groups(list(payload.get("groups") or []), source_text)
    if len(groups) != 9 or int(payload.get("group_count", -1)) != 9:
        raise RuntimeError(
            f"pre-review diagnostics require nine frozen reference groups, found {len(groups)}"
        )
    expected_ordinals = {sentence.ordinal for sentence in sentences}
    observed_ordinals: list[int] = []
    for group_index, group in enumerate(groups, start=1):
        if int(group.get("group_index", -1)) != group_index:
            raise RuntimeError(
                "reference group indexes must be continuous and one-based"
            )
        expected_for_span = [
            sentence.ordinal
            for sentence in sentences
            if sentence.end > int(group["raw_start"])
            and sentence.start < int(group["raw_end"])
        ]
        raw_ordinals = group.get("sentence_ordinals")
        if not isinstance(raw_ordinals, list):
            raise RuntimeError("reference group is missing sentence_ordinals")
        ordinals = [int(value) for value in raw_ordinals]
        if ordinals != expected_for_span:
            raise RuntimeError(
                "reference group sentence mapping disagrees with raw offsets"
            )
        observed_ordinals.extend(ordinals)
    if (
        len(observed_ordinals) != len(set(observed_ordinals))
        or set(observed_ordinals) != expected_ordinals
    ):
        raise RuntimeError(
            "reference groups must cover each reference sentence exactly once"
        )
    return tuple(groups)


def _observation_map(
    rows: list[Any],
    *,
    arm: str,
    seed: int,
    sentences: tuple[ReferenceSentence, ...],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    expected_ordinals = {sentence.ordinal for sentence in sentences}
    result: dict[tuple[str, int, int], dict[str, Any]] = {}
    sentence_by_ordinal = {sentence.ordinal: sentence for sentence in sentences}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"non-object sentence observation in {arm}/{seed}")
        row = dict(raw)
        if row.get("arm") != arm or row.get("seed") != seed:
            raise RuntimeError(
                f"sentence observation position mismatch in {arm}/{seed}"
            )
        ordinal = row.get("sentence_ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise RuntimeError(f"invalid sentence ordinal in {arm}/{seed}")
        if ordinal not in expected_ordinals:
            raise RuntimeError(
                f"out-of-grid sentence ordinal in {arm}/{seed}: {ordinal}"
            )
        expected = sentence_by_ordinal[ordinal]
        if row.get("sentence_id") != expected.sentence_id:
            raise RuntimeError(f"sentence identity mismatch in {arm}/{seed}/{ordinal}")
        if row.get("reference_text") != expected.text:
            raise RuntimeError(f"sentence reference mismatch in {arm}/{seed}/{ordinal}")
        if (
            row.get("status") != RunStatus.REVIEW_PENDING.value
            or row.get("tts_run_status") != RunStatus.OK.value
            or row.get("tts_complete") is not True
            or row.get("asr_status") != "ok"
            or row.get("valid_for_review") is not True
            or not isinstance(row.get("character_errors"), Mapping)
        ):
            raise RuntimeError(
                f"sentence observation is not review-ready: {arm}/{seed}/{ordinal}"
            )
        key = (arm, seed, ordinal)
        if key in result:
            raise RuntimeError(f"duplicate sentence observation: {key}")
        result[key] = row
    expected_keys = {(arm, seed, ordinal) for ordinal in expected_ordinals}
    if set(result) != expected_keys:
        missing = sorted(expected_keys - result.keys())
        raise RuntimeError(
            f"sentence observation grid incomplete for {arm}/{seed}: {missing}"
        )
    return result


def _validate_scoring_summary(summary: Mapping[str, Any]) -> None:
    expected_arm_counts = {arm.value: 3 for arm in ARMS}
    exact_values = {
        "ready_for_review": True,
        "run_grid_complete": True,
        "manifest_seed_grid_valid": True,
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
    }
    mismatches = {
        key: {"expected": expected, "observed": summary.get(key)}
        for key, expected in exact_values.items()
        if summary.get(key) != expected
    }
    if summary.get("arm_run_counts") != expected_arm_counts:
        mismatches["arm_run_counts"] = {
            "expected": expected_arm_counts,
            "observed": summary.get("arm_run_counts"),
        }
    for field in (
        "reasons",
        "missing_run_positions",
        "unexpected_run_positions",
        "duplicate_run_positions",
        "tts_failed_runs",
        "unscored_runs",
        "full_asr_failed_runs",
        "segment_asr_failed_runs",
        "invalid_asr_provenance_wavs",
    ):
        if summary.get(field) != []:
            mismatches[field] = {"expected": [], "observed": summary.get(field)}
    if mismatches:
        raise RuntimeError(f"formal scoring summary is not review-ready: {mismatches}")


def _validate_run_scoring(
    position: tuple[str, int],
    record: Mapping[str, Any],
    scoring: Mapping[str, Any],
) -> None:
    if record.get("status") != RunStatus.OK.value:
        raise RuntimeError(f"formal TTS run is not successful: {position}")
    full_asr = scoring.get("full_asr")
    segments = list(scoring.get("delivered_segments") or [])
    segment_asr_ready = bool(segments) and all(
        isinstance(segment, Mapping)
        and isinstance(segment.get("asr"), Mapping)
        and segment["asr"].get("status") == "ok"
        for segment in segments
    )
    failures = scoring.get("asr_failures")
    if (
        scoring.get("status") != RunStatus.OK.value
        or not isinstance(full_asr, Mapping)
        or full_asr.get("status") != "ok"
        or not isinstance(scoring.get("full_character_errors"), Mapping)
        or not segment_asr_ready
        or not isinstance(failures, Mapping)
        or failures.get("full_wav") != 0
        or failures.get("delivered_segments") != 0
    ):
        raise RuntimeError(f"formal scoring is not fully successful: {position}")


def load_diagnostic_inputs(output_root: Path) -> DiagnosticInputs:
    """Load all formal artifacts and reject any run/sentence grid ambiguity."""

    manifest_path = output_root / "manifest.json"
    groups_path = output_root / "reference_groups.json"
    scoring_summary_path = output_root / "scoring_summary.json"
    top_observations_path = output_root / "sentence_observations.json"
    manifest = read_json(manifest_path)
    seeds, sentences = _manifest_grid(manifest)
    source_text = str(manifest["text"]["text"])
    source_sha256 = str(manifest["text"]["sha256"])
    groups_payload = read_json(groups_path)
    groups = _validated_groups(groups_payload, source_text, sentences)

    expected_positions = {(arm.value, seed) for arm in ARMS for seed in seeds}
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for record in discover_run_records(output_root):
        arm = str(record.get("arm", ""))
        seed = record.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise RuntimeError("run record has an invalid seed")
        position = (arm, seed)
        if position not in expected_positions:
            raise RuntimeError(f"unexpected formal run position: {position}")
        if position in records:
            raise RuntimeError(f"duplicate formal run position: {position}")
        if record.get("source_text") != source_text:
            raise RuntimeError(f"run source text differs from manifest: {position}")
        if record.get("source_text_sha256") != source_sha256:
            raise RuntimeError(f"run source hash differs from manifest: {position}")
        records[position] = record
    if set(records) != expected_positions:
        missing = sorted(expected_positions - records.keys())
        raise RuntimeError(
            f"formal run grid must contain exactly nine runs; missing {missing}"
        )

    scoring: dict[tuple[str, int], dict[str, Any]] = {}
    observations: dict[tuple[str, int], tuple[dict[str, Any], ...]] = {}
    observation_index: dict[tuple[str, int, int], dict[str, Any]] = {}
    scoring_summary = read_json(scoring_summary_path)
    _validate_scoring_summary(scoring_summary)
    identities = [
        _identity(output_root, manifest_path),
        _identity(output_root, scoring_summary_path),
        _identity(output_root, groups_path),
    ]
    for arm in (item.value for item in ARMS):
        for seed in seeds:
            position = (arm, seed)
            run_path = output_root / str(records[position]["_record_path"])
            scoring_path = run_path.parent / "scoring.json"
            observation_path = run_path.parent / "sentence_observations.json"
            scoring[position] = read_json(scoring_path)
            _validate_run_scoring(position, records[position], scoring[position])
            payload = read_json(observation_path)
            by_key = _observation_map(
                list(payload.get("observations") or []),
                arm=arm,
                seed=seed,
                sentences=sentences,
            )
            ordered = tuple(
                by_key[(arm, seed, sentence.ordinal)] for sentence in sentences
            )
            observations[position] = ordered
            observation_index.update(by_key)
            identities.extend(
                (
                    _identity(output_root, run_path),
                    _identity(output_root, scoring_path),
                    _identity(output_root, observation_path),
                )
            )

    top_payload = read_json(top_observations_path)
    top_index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in list(top_payload.get("observations") or []):
        if not isinstance(raw, Mapping):
            raise RuntimeError("top-level sentence observations contain a non-object")
        row = dict(raw)
        key = (str(row.get("arm", "")), row.get("seed"), row.get("sentence_ordinal"))
        if key in top_index:
            raise RuntimeError(f"duplicate top-level sentence observation: {key}")
        top_index[key] = row
    if top_index != observation_index:
        raise RuntimeError(
            "top-level sentence observations do not exactly match the nine run sidecars"
        )
    if len(top_index) != 342:
        raise RuntimeError(
            f"formal sentence grid must contain exactly 342 observations, found {len(top_index)}"
        )
    identities.append(_identity(output_root, top_observations_path))

    return DiagnosticInputs(
        output_root=output_root,
        manifest=manifest,
        seeds=seeds,
        sentences=sentences,
        reference_groups_payload=groups_payload,
        reference_groups=groups,
        scoring_summary=scoring_summary,
        records=records,
        scoring=scoring,
        observations=observations,
        input_artifacts=tuple(identities),
    )


__all__ = ["ARMS", "DiagnosticInputs", "load_diagnostic_inputs"]
