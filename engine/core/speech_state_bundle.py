"""Pure manifest and artifact checks for verified speech-state bundles.

This module does not load tensors or touch CUDA.  It only proves that the
runtime has a self-consistent model contract and that the files named by the
bundle manifest are the files that were exported.  The executor remains the
owner of the actual state payload and stream ordering.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .speech_state_model import (
    SpeechStateCursorPolicy,
    SpeechStateModelContract,
    SpeechStateTalkerCarry,
)
from .native_cursor import (
    CURSOR_RECURRENT_INPUT_BINDINGS,
    CURSOR_RECURRENT_OUTPUT_BINDINGS,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SpeechStateBundleValidation:
    """Result of validating one manifest-owned speech-state bundle."""

    verified: bool
    reason: str
    model_fingerprint: str = ""
    runtime_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "reason": self.reason,
            "model_fingerprint": self.model_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
        }


def validate_speech_state_bundle(
    manifest: Mapping[str, Any] | None,
    *,
    bundle_root: Path | None,
    runtime_artifact_path: Path | None = None,
) -> SpeechStateBundleValidation:
    """Validate the on-disk identity and layout of a stateful model bundle.

    A missing speech-state section is deliberately distinguishable from a
    malformed stateful bundle.  Neither case enables runtime handoff.
    """

    section = manifest.get("speech_state") if isinstance(manifest, Mapping) else None
    if not isinstance(section, Mapping):
        return SpeechStateBundleValidation(False, "missing_speech_state")
    model_fingerprint = _required_text(section, "model_fingerprint")
    runtime_fingerprint = _required_text(section, "runtime_fingerprint")
    if not model_fingerprint or not runtime_fingerprint:
        return SpeechStateBundleValidation(
            False,
            "missing_runtime_fingerprint",
            model_fingerprint,
            runtime_fingerprint,
        )

    try:
        contract = SpeechStateModelContract.from_mapping(section.get("model_contract"))
    except (TypeError, ValueError):
        return SpeechStateBundleValidation(
            False,
            "missing_or_invalid_model_contract",
            model_fingerprint,
            runtime_fingerprint,
        )
    if contract.model_fingerprint != model_fingerprint:
        return SpeechStateBundleValidation(
            False,
            "model_fingerprint_mismatch",
            model_fingerprint,
            runtime_fingerprint,
        )

    bundle = section.get("bundle")
    if not isinstance(bundle, Mapping):
        return SpeechStateBundleValidation(
            False,
            "missing_bundle_descriptor",
            model_fingerprint,
            runtime_fingerprint,
        )
    if type(bundle.get("schema_version")) is not int or bundle.get("schema_version") != 1:
        return SpeechStateBundleValidation(
            False, "unsupported_bundle_schema", model_fingerprint, runtime_fingerprint
        )
    layout_reason = _validate_layout(manifest, bundle, contract)
    if layout_reason:
        return SpeechStateBundleValidation(
            False, layout_reason, model_fingerprint, runtime_fingerprint
        )
    if bundle_root is None:
        return SpeechStateBundleValidation(
            False,
            "bundle_root_unavailable",
            model_fingerprint,
            runtime_fingerprint,
        )
    artifact_reason = _validate_artifacts(
        bundle,
        bundle_root,
        contract,
        runtime_artifact_path=runtime_artifact_path,
    )
    if artifact_reason:
        return SpeechStateBundleValidation(
            False, artifact_reason, model_fingerprint, runtime_fingerprint
        )
    return SpeechStateBundleValidation(
        True, "verified", model_fingerprint, runtime_fingerprint
    )


def _required_text(section: Mapping[str, Any], key: str) -> str:
    value = section.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def _validate_layout(
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any],
    contract: SpeechStateModelContract,
) -> str:
    c2w = manifest.get("code2wav_fused")
    if not isinstance(c2w, Mapping):
        return "missing_code2wav_state_layout"
    if contract.retain_c2w_kv:
        if not c2w.get("c2w_state_input_names") or not c2w.get("c2w_state_output_names"):
            return "missing_code2wav_state_layout"
    if contract.retain_c2w_history and not c2w.get("initial_state_shapes"):
        return "missing_code2wav_history_layout"
    declared_layout_hash = str(bundle.get("code2wav_layout_sha256", "") or "").strip().lower()
    if not _SHA256.fullmatch(declared_layout_hash):
        return "code2wav_layout_hash_invalid"
    if declared_layout_hash != code2wav_layout_fingerprint(c2w):
        return "code2wav_layout_hash_mismatch"

    bridge = bundle.get("hidden_tail_bridge")
    if contract.talker_carry is SpeechStateTalkerCarry.HIDDEN_TAIL_BRIDGE:
        if not isinstance(bridge, Mapping):
            return "missing_hidden_tail_bridge"
        if bridge.get("source") != "talker_hidden_tail":
            return "hidden_tail_bridge_source_mismatch"
        if bridge.get("tail_tokens") != contract.talker_hidden_tail:
            return "hidden_tail_bridge_length_mismatch"
        architecture = manifest.get("architecture")
        hidden_size = architecture.get("hidden_size") if isinstance(architecture, Mapping) else None
        if hidden_size is not None and bridge.get("hidden_size") != hidden_size:
            return "hidden_tail_bridge_shape_mismatch"

    if contract.cursor_policy is SpeechStateCursorPolicy.MIGRATE:
        native_cursor = manifest.get("native_cursor")
        if not isinstance(native_cursor, Mapping) or native_cursor.get("enabled") is not True:
            return "missing_cursor_artifact"
        declared_inputs = _declared_binding_names(native_cursor.get("input_names"))
        declared_outputs = _declared_binding_names(native_cursor.get("output_names"))
        if (
            not CURSOR_RECURRENT_INPUT_BINDINGS.issubset(declared_inputs)
            or not CURSOR_RECURRENT_OUTPUT_BINDINGS.issubset(declared_outputs)
        ):
            return "incomplete_cursor_recurrent_abi"
    return ""


def _declared_binding_names(value: Any) -> set[str]:
    """Normalize malformed manifest binding lists to a fail-closed set."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {name for name in value if isinstance(name, str)}


def _validate_artifacts(
    bundle: Mapping[str, Any],
    bundle_root: Path,
    contract: SpeechStateModelContract,
    *,
    runtime_artifact_path: Path | None,
) -> str:
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return "missing_artifact_descriptor"
    required = ["model_weights", "runtime_plan"]
    if contract.cursor_policy is not SpeechStateCursorPolicy.DISABLE:
        required.append("cursor_head")
    for name in required:
        descriptor = artifacts.get(name)
        if not isinstance(descriptor, Mapping):
            return f"missing_{name}_artifact"
        reason = _validate_artifact(descriptor, bundle_root, require_source=name != "runtime_plan")
        if reason:
            return f"{name}_{reason}"
        if name == "runtime_plan" and runtime_artifact_path is not None:
            declared_path = _resolve_artifact(
                bundle_root,
                descriptor.get("path") if isinstance(descriptor.get("path"), str) else "",
            )
            try:
                actual_path = Path(runtime_artifact_path).resolve()
            except OSError:
                return "runtime_plan_path_mismatch"
            if declared_path is None or actual_path != declared_path:
                return "runtime_plan_path_mismatch"
    return ""


def _validate_artifact(
    descriptor: Mapping[str, Any],
    bundle_root: Path,
    *,
    require_source: bool,
) -> str:
    path_value = str(descriptor.get("path", "") or "").strip()
    expected = str(descriptor.get("sha256", "") or "").strip().lower()
    if not path_value or not _SHA256.fullmatch(expected):
        return "hash_invalid"
    source_expected = str(descriptor.get("source_sha256", "") or "").strip().lower()
    if require_source and not _SHA256.fullmatch(source_expected):
        return "source_hash_invalid"
    if not _artifact_path_is_inside_bundle(bundle_root, path_value):
        return "path_outside_bundle"
    path = _resolve_artifact(bundle_root, path_value)
    if path is None:
        return "not_found"
    if _sha256_file(path) != expected:
        return "hash_mismatch"
    if require_source:
        if source_expected != expected:
            return "source_export_hash_mismatch"
        source_path_value = str(descriptor.get("source_path", "") or "").strip()
        if not source_path_value:
            return "source_not_found"
        if not _artifact_path_is_inside_bundle(bundle_root, source_path_value):
            return "source_path_outside_bundle"
        source_path = _resolve_artifact(bundle_root, source_path_value)
        if source_path is None or _sha256_file(source_path) != source_expected:
            return "source_hash_mismatch"
    return ""


def _artifact_path_is_inside_bundle(root: Path, value: str) -> bool:
    """Reject absolute, traversal, and escaping-symlink artifact paths."""

    raw = Path(value)
    if raw.is_absolute():
        return False
    try:
        bundle_root = Path(root).resolve()
        candidate = (bundle_root / raw).resolve()
    except OSError:
        return False
    return candidate == bundle_root or bundle_root in candidate.parents


def _resolve_artifact(root: Path, value: str) -> Path | None:
    raw = Path(value)
    if raw.is_absolute() or not _artifact_path_is_inside_bundle(root, value):
        return None
    candidate = (Path(root).resolve() / raw).resolve()
    return candidate if candidate.is_file() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code2wav_layout_fingerprint(layout: Mapping[str, Any]) -> str:
    """Return a stable digest for the state layout recorded in a manifest."""
    payload = json.dumps(dict(layout), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "SpeechStateBundleValidation",
    "code2wav_layout_fingerprint",
    "validate_speech_state_bundle",
]
