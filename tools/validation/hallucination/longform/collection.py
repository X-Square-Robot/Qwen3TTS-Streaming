"""Serial collection and immutable experiment-manifest management."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import persist_collected_run, read_json, write_json
from .models import ArmKind
from .preflight import exact_text_record, runtime_identity


SCHEMA_VERSION = 2
DEFAULT_SEEDS = (17041, 28411, 39671)


def _canonical_value(value: Any) -> Any:
    """Return the JSON value used by every immutable experiment binding."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not canonical JSON: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    """Hash a typed JSON-compatible value without formatting ambiguity."""

    encoded = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def immutable_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select the canonical manifest contract, excluding mutable phase history."""

    return {
        "schema_version": int(manifest.get("schema_version", 1)),
        "text": dict(manifest.get("text") or {}),
        "seeds": list(manifest.get("seeds") or []),
        "logical_session_ids": dict(manifest.get("logical_session_ids") or {}),
        "identities": dict(manifest.get("identities") or {}),
    }


def manifest_binding_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash all immutable experiment inputs used to admit persisted runs."""

    return canonical_sha256(immutable_manifest_payload(manifest))


def logical_session_id(seed: int, *, prefix: str = "verylong-0818") -> str:
    """Put the trial seed into the public SID used by every comparison arm."""

    if not prefix or any(character.isspace() for character in prefix):
        raise ValueError("session prefix must be non-empty and contain no whitespace")
    return f"{prefix}-seed-{int(seed):08d}"


def initialize_experiment(
    output_root: Path,
    *,
    text_file: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    identities: Mapping[str, Any],
) -> dict[str, Any]:
    """Create or verify the immutable top-level experiment manifest."""

    normalized_seeds = [int(seed) for seed in seeds]
    if len(normalized_seeds) != 3 or len(set(normalized_seeds)) != 3:
        raise ValueError("the primary comparison requires exactly three unique seeds")
    text_record = exact_text_record(text_file)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if dict(existing.get("text") or {}) != text_record:
            raise RuntimeError(
                "output directory belongs to a different canonical source text"
            )
        if existing.get("seeds") != normalized_seeds:
            raise RuntimeError("output directory belongs to a different seed set")
        expected_sessions = {
            str(seed): logical_session_id(seed) for seed in normalized_seeds
        }
        if existing.get("logical_session_ids") != expected_sessions:
            raise RuntimeError(
                "output directory belongs to a different logical session mapping"
            )
        if dict(existing.get("identities") or {}) != dict(identities):
            raise RuntimeError(
                "output directory belongs to different experiment identities"
            )
        actual_binding = manifest_binding_sha256(existing)
        stored_binding = existing.get("manifest_binding_sha256")
        if stored_binding is not None and stored_binding != actual_binding:
            raise RuntimeError("stored experiment manifest binding is invalid")
        return existing

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "text": text_record,
        "seeds": normalized_seeds,
        "logical_session_ids": {
            str(seed): logical_session_id(seed) for seed in normalized_seeds
        },
        "identities": dict(identities),
        "phase_history": [],
    }
    manifest["manifest_binding_sha256"] = manifest_binding_sha256(manifest)
    write_json(manifest_path, manifest)
    return manifest


def record_phase(
    output_root: Path,
    phase: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    manifest_path = output_root / "manifest.json"
    manifest = read_json(manifest_path)
    history = list(manifest.get("phase_history") or [])
    history.append(
        {
            "phase": phase,
            "recorded_at": datetime.now(UTC).isoformat(),
            "runtime": runtime_identity(),
            "details": dict(details or {}),
        }
    )
    manifest["phase_history"] = history
    write_json(manifest_path, manifest)


def collect_arm_runs(
    output_root: Path,
    adapter: Any,
    *,
    seeds: Sequence[int] | None = None,
    arm_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one arm at a time, batch=1, retaining failed trials as evidence."""

    manifest = read_json(output_root / "manifest.json")
    requested_seeds = [int(seed) for seed in (seeds or manifest["seeds"])]
    manifest_seeds = [int(seed) for seed in manifest["seeds"]]
    if len(requested_seeds) != len(set(requested_seeds)):
        raise ValueError("requested collection seeds must be unique")
    unexpected_seeds = sorted(set(requested_seeds) - set(manifest_seeds))
    if unexpected_seeds:
        raise RuntimeError(
            f"requested seeds are not in the immutable manifest: {unexpected_seeds}"
        )
    source = dict(manifest["text"])
    source_path = Path(source["path"])
    current = exact_text_record(source_path)
    if current["sha256"] != source["sha256"]:
        raise RuntimeError("source text changed after experiment preflight")
    text = current["text"]
    arm = ArmKind(adapter.arm)
    manifest_binding = manifest_binding_sha256(manifest)
    stored_binding = manifest.get("manifest_binding_sha256")
    if stored_binding is not None and stored_binding != manifest_binding:
        raise RuntimeError("stored experiment manifest binding is invalid")
    normalized_metadata = _canonical_value(dict(arm_metadata or {}))
    collection_contract = {
        "arm": arm.value,
        "adapter_type": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "arm_metadata": normalized_metadata,
    }
    collection_binding = canonical_sha256(collection_contract)
    records: list[dict[str, Any]] = []
    for seed in requested_seeds:
        run_path = output_root / "arms" / arm.value / f"seed_{seed:04d}" / "run.json"
        session_id = str(manifest["logical_session_ids"][str(seed)])
        run_metadata = dict(normalized_metadata)
        evidence_factory = getattr(adapter, "run_contract", None)
        if callable(evidence_factory):
            run_metadata["run_contract"] = _canonical_value(
                evidence_factory(session_id=session_id, seed=seed)
            )
        if run_path.is_file():
            existing = read_json(run_path)
            expected_fields = {
                "arm": arm.value,
                "seed": seed,
                "session_id": session_id,
                "source_text": text,
                "source_text_sha256": source["sha256"],
                "experiment_manifest_binding_sha256": manifest_binding,
                "collection_contract_sha256": collection_binding,
                "arm_metadata": run_metadata,
            }
            mismatches = [
                field
                for field, expected in expected_fields.items()
                if existing.get(field) != expected
            ]
            if mismatches:
                raise RuntimeError(
                    f"existing run is not bound to this manifest/config/runtime "
                    f"({', '.join(mismatches)}): {run_path}"
                )
            records.append(existing)
            continue
        collected = adapter.collect(text, session_id=session_id, seed=seed)
        records.append(
            persist_collected_run(
                output_root,
                collected,
                seed=seed,
                source_text=text,
                source_text_sha256=source["sha256"],
                arm_metadata=run_metadata,
                experiment_manifest_binding_sha256=manifest_binding,
                collection_contract_sha256=collection_binding,
            )
        )
    record_phase(
        output_root,
        f"collect:{arm.value}",
        details={
            "arm": arm.value,
            "seeds": requested_seeds,
            "statuses": [record["status"] for record in records],
        },
    )
    return records


__all__ = [
    "DEFAULT_SEEDS",
    "canonical_sha256",
    "collect_arm_runs",
    "immutable_manifest_payload",
    "initialize_experiment",
    "logical_session_id",
    "manifest_binding_sha256",
    "record_phase",
]
