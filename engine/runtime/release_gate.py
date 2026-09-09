"""Release evidence gate for model capabilities.

Manifest fields describe what an artifact claims.  This module describes the
separate evidence required before a public gateway may advertise a route.
It is intentionally pure: it does not load TRT or model tensors, call ASR,
make network requests, or add an ASR dependency to the online service.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


def _strict_nonempty_text(value: Any) -> str:
    """Return only typed, non-empty identity text."""

    if not isinstance(value, str):
        return ""
    return value.strip()


class ReleaseCapability(str, Enum):
    NATIVE_CURSOR = "native_cursor"
    SPEECH_STATE = "speech_state"


@dataclass(frozen=True, slots=True)
class ReleaseGate:
    native_cursor_verified: bool
    speech_state_verified: bool
    native_cursor_reason: str
    speech_state_reason: str

    @classmethod
    def disabled(cls, reason: str = "release_evidence_missing") -> "ReleaseGate":
        return cls(False, False, reason, reason)

    def verified(self, capability: ReleaseCapability) -> bool:
        return (
            self.native_cursor_verified
            if capability is ReleaseCapability.NATIVE_CURSOR
            else self.speech_state_verified
        )

    def reason(self, capability: ReleaseCapability) -> str:
        return (
            self.native_cursor_reason
            if capability is ReleaseCapability.NATIVE_CURSOR
            else self.speech_state_reason
        )


def evaluate_release_gate(
    manifest: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None,
) -> ReleaseGate:
    """Evaluate versioned release evidence against manifest identities."""

    schema_version = (
        evidence.get("schema_version") if isinstance(evidence, Mapping) else None
    )
    if type(schema_version) is not int or schema_version != 1:
        return ReleaseGate.disabled()
    speech_state = manifest.get("speech_state") if isinstance(manifest, Mapping) else None
    if not isinstance(speech_state, Mapping):
        return ReleaseGate.disabled("release_manifest_identity_missing")

    model = _strict_nonempty_text(speech_state.get("model_fingerprint"))
    runtime = _strict_nonempty_text(speech_state.get("runtime_fingerprint"))
    if not model or not runtime:
        return ReleaseGate.disabled("release_manifest_identity_missing")

    evidence_model = _strict_nonempty_text(evidence.get("model_fingerprint"))
    evidence_runtime = _strict_nonempty_text(evidence.get("runtime_fingerprint"))
    if not evidence_model or not evidence_runtime:
        return ReleaseGate.disabled("release_evidence_identity_missing")
    if model != evidence_model or runtime != evidence_runtime:
        return ReleaseGate(
            False,
            False,
            "release_evidence_fingerprint_mismatch",
            "release_evidence_fingerprint_mismatch",
        )

    performance = evidence.get("performance")
    performance_ok = isinstance(performance, Mapping) and performance.get("verified") is True
    quality = evidence.get("quality")
    asr_ok = (
        isinstance(quality, Mapping)
        and quality.get("offline_asr_verified") is True
    )
    native = evidence.get("native_cursor")
    native_ok = (
        isinstance(native, Mapping)
        and native.get("trt_numeric_verified") is True
        and native.get("progress_e2e_verified") is True
        and native.get("monotonic_verified") is True
        and performance_ok
        and asr_ok
    )
    speech = evidence.get("speech_state")
    speech_ok = (
        isinstance(speech, Mapping)
        and speech.get("bundle_verified") is True
        and speech.get("trt_transfer_verified") is True
        and speech.get("successor_e2e_verified") is True
        and performance_ok
        and asr_ok
    )
    return ReleaseGate(
        native_ok,
        speech_ok,
        "verified" if native_ok else "native_cursor_release_evidence_incomplete",
        "verified" if speech_ok else "speech_state_release_evidence_incomplete",
    )


__all__ = ["ReleaseCapability", "ReleaseGate", "evaluate_release_gate"]
