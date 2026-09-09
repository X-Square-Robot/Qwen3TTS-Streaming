import hashlib
from pathlib import Path

import pytest

from engine.core.speech_state_bundle import (
    code2wav_layout_fingerprint,
    validate_speech_state_bundle,
)
from engine.core.speech_state_model import (
    SpeechStateBoundaryPhase,
    SpeechStateBoundaryTokenPolicy,
    SpeechStateCheckpointBoundary,
    SpeechStateCursorPolicy,
    SpeechStateModelContract,
    SpeechStateSuccessorStart,
    SpeechStateSuccessorTextEntry,
    SpeechStateTalkerCarry,
)
from engine.core.speech_state import SpeechStateTransfer


def _contract() -> dict:
    return SpeechStateModelContract(
        model_fingerprint="x2-model-v1",
        checkpoint_id="x2-checkpoint-v1",
        boundaries=(SpeechStateCheckpointBoundary.SEGMENT_END,),
        boundary_phase=SpeechStateBoundaryPhase.CODEC_EOS,
        boundary_token_policy=SpeechStateBoundaryTokenPolicy.EXACT_MODEL_TOKEN,
        boundary_token_id=2150,
        successor_start=SpeechStateSuccessorStart.MODEL_PREFILL,
        successor_text_entry=SpeechStateSuccessorTextEntry.PREFILL_SUCCESSOR_TEXT,
        cursor_policy=SpeechStateCursorPolicy.REANCHOR,
        transfer=SpeechStateTransfer.TRAINED_APPROXIMATE,
        retain_talker_kv=False,
        retain_code_predictor_state=False,
        retain_c2w_kv=True,
        retain_c2w_history=True,
        training_scope="x2-clause-continuation-v1",
        talker_carry=SpeechStateTalkerCarry.HIDDEN_TAIL_BRIDGE,
        talker_hidden_tail=4,
    ).to_dict()


def _write_pair(root: Path, relative: str, source_relative: str, data: bytes) -> dict:
    export = root / relative
    source = root / source_relative
    export.parent.mkdir(parents=True, exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)
    export.write_bytes(data)
    source.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    return {
        "path": relative,
        "sha256": digest,
        "source_path": source_relative,
        "source_sha256": digest,
    }


def _manifest(root: Path) -> dict:
    c2w = {
        "c2w_state_input_names": ["c2w_past_kv_0"],
        "c2w_state_output_names": ["c2w_new_past_kv_0"],
        "initial_state_shapes": [[1, 2, 1, 1, 2]],
    }
    bundle = {
        "schema_version": 1,
        "code2wav_layout_sha256": code2wav_layout_fingerprint(c2w),
        "hidden_tail_bridge": {
            "source": "talker_hidden_tail",
            "tail_tokens": 4,
            "hidden_size": 4,
            "bridge_id": "x2-text-acoustic-v1",
        },
        "artifacts": {
            "model_weights": _write_pair(root, "weights/model.safetensors", "source/model.safetensors", b"weights"),
            "runtime_plan": {"path": "runtime/model.plan", "sha256": hashlib.sha256(b"plan").hexdigest()},
            "cursor_head": _write_pair(root, "weights/qwen3_tts_12hz_la1_seed0.pt", "source/qwen3_tts_12hz_la1_seed0.pt", b"cursor"),
        },
    }
    (root / "runtime/model.plan").parent.mkdir(parents=True, exist_ok=True)
    (root / "runtime/model.plan").write_bytes(b"plan")
    return {
        "architecture": {"hidden_size": 4},
        "code2wav_fused": c2w,
        "native_cursor": {"enabled": True},
        "speech_state": {
            "model_fingerprint": "x2-model-v1",
            "runtime_fingerprint": "trt-runtime-v1",
            "model_contract": _contract(),
            "bundle": bundle,
        },
    }


def test_verified_bundle_requires_real_artifacts_and_matching_layout(tmp_path):
    result = validate_speech_state_bundle(_manifest(tmp_path), bundle_root=tmp_path)

    assert result.verified is True
    assert result.reason == "verified"


def test_verified_bundle_requires_loaded_runtime_plan_to_match_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    result = validate_speech_state_bundle(
        manifest,
        bundle_root=tmp_path,
        runtime_artifact_path=tmp_path / "runtime/model.plan",
    )

    assert result.verified is True

    other = tmp_path / "runtime/other.plan"
    other.write_bytes(b"plan")
    result = validate_speech_state_bundle(
        manifest,
        bundle_root=tmp_path,
        runtime_artifact_path=other,
    )

    assert result.verified is False
    assert result.reason == "runtime_plan_path_mismatch"


def test_bundle_rejects_export_hash_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    artifact = manifest["speech_state"]["bundle"]["artifacts"]["model_weights"]
    artifact["sha256"] = "0" * 64

    result = validate_speech_state_bundle(manifest, bundle_root=tmp_path)

    assert result.verified is False
    assert result.reason == "model_weights_hash_mismatch"


@pytest.mark.parametrize("field", ["model_fingerprint", "runtime_fingerprint"])
def test_bundle_rejects_non_string_fingerprint(tmp_path, field):
    manifest = _manifest(tmp_path)
    manifest["speech_state"][field] = 1

    result = validate_speech_state_bundle(manifest, bundle_root=tmp_path)

    assert result.verified is False
    assert result.reason == "missing_runtime_fingerprint"


def test_bundle_rejects_source_export_hash_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    artifact = manifest["speech_state"]["bundle"]["artifacts"]["model_weights"]
    artifact["source_sha256"] = "1" * 64

    result = validate_speech_state_bundle(manifest, bundle_root=tmp_path)

    assert result.verified is False
    assert result.reason == "model_weights_source_export_hash_mismatch"


def test_bundle_rejects_missing_hidden_tail_bridge(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["speech_state"]["bundle"].pop("hidden_tail_bridge")

    result = validate_speech_state_bundle(manifest, bundle_root=tmp_path)

    assert result.verified is False
    assert result.reason == "missing_hidden_tail_bridge"


def test_bundle_rejects_migrate_contract_with_incomplete_cursor_recurrent_abi(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["speech_state"]["model_contract"]["cursor_policy"] = "migrate"
    manifest["native_cursor"]["input_names"] = ["cursor_mu_in"]
    manifest["native_cursor"]["output_names"] = ["cursor_mu"]

    result = validate_speech_state_bundle(manifest, bundle_root=tmp_path)

    assert result.verified is False
    assert result.reason == "incomplete_cursor_recurrent_abi"


def test_bundle_rejects_non_integer_schema_version(tmp_path):
    for schema_version in (True, 1.0, "1"):
        manifest = _manifest(tmp_path / str(schema_version))
        manifest["speech_state"]["bundle"]["schema_version"] = schema_version

        result = validate_speech_state_bundle(
            manifest,
            bundle_root=tmp_path / str(schema_version),
        )

        assert result.verified is False
        assert result.reason == "unsupported_bundle_schema"


@pytest.mark.parametrize("enabled", ["true", 1, [], {"value": True}])
def test_bundle_rejects_non_boolean_cursor_enabled(tmp_path, enabled):
    manifest = _manifest(tmp_path)
    manifest["speech_state"]["model_contract"]["cursor_policy"] = "migrate"
    manifest["native_cursor"]["enabled"] = enabled

    result = validate_speech_state_bundle(manifest, bundle_root=tmp_path)

    assert result.verified is False
    assert result.reason == "missing_cursor_artifact"


def test_bundle_stays_disabled_without_runtime_bundle_root(tmp_path):
    manifest = _manifest(tmp_path)

    result = validate_speech_state_bundle(manifest, bundle_root=None)

    assert result.verified is False
    assert result.reason == "bundle_root_unavailable"


def test_bundle_rejects_artifact_path_outside_bundle_root(tmp_path):
    manifest = _manifest(tmp_path)
    outside = tmp_path.parent / "outside-model.safetensors"
    outside.write_bytes(b"weights")
    artifact = manifest["speech_state"]["bundle"]["artifacts"]["model_weights"]
    artifact["path"] = "../outside-model.safetensors"
    artifact["sha256"] = hashlib.sha256(b"weights").hexdigest()

    result = validate_speech_state_bundle(manifest, bundle_root=tmp_path)

    assert result.verified is False
    assert result.reason == "model_weights_path_outside_bundle"


def test_bundle_rejects_source_symlink_that_escapes_bundle_root(tmp_path):
    manifest = _manifest(tmp_path)
    outside = tmp_path.parent / "outside-source.safetensors"
    outside.write_bytes(b"weights")
    link = tmp_path / "source" / "outside.safetensors"
    link.symlink_to(outside)
    artifact = manifest["speech_state"]["bundle"]["artifacts"]["model_weights"]
    artifact["source_path"] = "source/outside.safetensors"
    artifact["source_sha256"] = hashlib.sha256(b"weights").hexdigest()

    result = validate_speech_state_bundle(manifest, bundle_root=tmp_path)

    assert result.verified is False
    assert result.reason == "model_weights_source_path_outside_bundle"
