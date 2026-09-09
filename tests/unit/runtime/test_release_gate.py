import pytest

from engine.runtime.release_gate import (
    ReleaseCapability,
    evaluate_release_gate,
)


def _evidence(**overrides):
    value = {
        "schema_version": 1,
        "model_fingerprint": "model-v1",
        "runtime_fingerprint": "runtime-v1",
        "performance": {"verified": True},
        "quality": {"offline_asr_verified": True},
        "native_cursor": {
            "trt_numeric_verified": True,
            "progress_e2e_verified": True,
            "monotonic_verified": True,
        },
        "speech_state": {
            "bundle_verified": True,
            "trt_transfer_verified": True,
            "successor_e2e_verified": True,
        },
    }
    value.update(overrides)
    return value


def _manifest():
    return {
        "speech_state": {
            "model_fingerprint": "model-v1",
            "runtime_fingerprint": "runtime-v1",
        }
    }


def test_release_gate_is_fail_closed_without_evidence():
    gate = evaluate_release_gate(_manifest(), None)

    assert not gate.verified(ReleaseCapability.NATIVE_CURSOR)
    assert not gate.verified(ReleaseCapability.SPEECH_STATE)
    assert gate.reason(ReleaseCapability.NATIVE_CURSOR) == "release_evidence_missing"


def test_release_gate_requires_matching_identity_and_complete_evidence():
    gate = evaluate_release_gate(_manifest(), _evidence())

    assert gate.verified(ReleaseCapability.NATIVE_CURSOR)
    assert gate.verified(ReleaseCapability.SPEECH_STATE)
    assert gate.reason(ReleaseCapability.SPEECH_STATE) == "verified"


def test_release_gate_can_verify_state_without_cursor_evidence():
    evidence = _evidence()
    evidence.pop("native_cursor")

    gate = evaluate_release_gate(_manifest(), evidence)

    assert not gate.verified(ReleaseCapability.NATIVE_CURSOR)
    assert gate.verified(ReleaseCapability.SPEECH_STATE)


def test_release_gate_requires_asr_quality_evidence():
    evidence = _evidence()
    evidence.pop("quality")

    gate = evaluate_release_gate(_manifest(), evidence)

    assert not gate.verified(ReleaseCapability.NATIVE_CURSOR)
    assert not gate.verified(ReleaseCapability.SPEECH_STATE)


def test_release_gate_rejects_fingerprint_mismatch_for_all_capabilities():
    gate = evaluate_release_gate(
        _manifest(), _evidence(runtime_fingerprint="runtime-other")
    )

    assert not gate.verified(ReleaseCapability.NATIVE_CURSOR)
    assert not gate.verified(ReleaseCapability.SPEECH_STATE)
    assert gate.reason(ReleaseCapability.SPEECH_STATE) == (
        "release_evidence_fingerprint_mismatch"
    )


def test_release_gate_rejects_complete_evidence_without_manifest_identity():
    gate = evaluate_release_gate({}, _evidence())

    assert not gate.verified(ReleaseCapability.NATIVE_CURSOR)
    assert not gate.verified(ReleaseCapability.SPEECH_STATE)
    assert gate.reason(ReleaseCapability.NATIVE_CURSOR) == (
        "release_manifest_identity_missing"
    )


def test_release_gate_rejects_empty_evidence_identity():
    evidence = _evidence(model_fingerprint="", runtime_fingerprint="")

    gate = evaluate_release_gate(_manifest(), evidence)

    assert not gate.verified(ReleaseCapability.NATIVE_CURSOR)
    assert not gate.verified(ReleaseCapability.SPEECH_STATE)
    assert gate.reason(ReleaseCapability.SPEECH_STATE) == (
        "release_evidence_identity_missing"
    )


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_release_gate_rejects_non_integer_schema_version(schema_version):
    gate = evaluate_release_gate(_manifest(), _evidence(schema_version=schema_version))

    assert not gate.verified(ReleaseCapability.NATIVE_CURSOR)
    assert not gate.verified(ReleaseCapability.SPEECH_STATE)
    assert gate.reason(ReleaseCapability.NATIVE_CURSOR) == "release_evidence_missing"


@pytest.mark.parametrize("value", [1, True, [], {}])
def test_release_gate_rejects_non_string_identity(value):
    manifest = _manifest()
    manifest["speech_state"]["model_fingerprint"] = value
    evidence = _evidence(model_fingerprint=value)

    gate = evaluate_release_gate(manifest, evidence)

    assert not gate.verified(ReleaseCapability.NATIVE_CURSOR)
    assert not gate.verified(ReleaseCapability.SPEECH_STATE)
    assert gate.reason(ReleaseCapability.SPEECH_STATE) == (
        "release_manifest_identity_missing"
    )
