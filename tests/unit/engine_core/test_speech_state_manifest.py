from pathlib import Path
import sys

import pytest

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


_SCRIPTS_PYTHON = Path(__file__).parents[3] / "scripts" / "python"
if str(_SCRIPTS_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_PYTHON))

from triton_manifest_io import build_manifest_for_export  # noqa: E402
from engine.core.speech_state_bundle import code2wav_layout_fingerprint


def _contract_dict(*, cursor_policy=SpeechStateCursorPolicy.REANCHOR) -> dict:
    return SpeechStateModelContract(
        model_fingerprint="x2-model-v1",
        checkpoint_id="x2-checkpoint-v1",
        boundaries=(SpeechStateCheckpointBoundary.SEGMENT_END,),
        boundary_phase=SpeechStateBoundaryPhase.CODEC_EOS,
        boundary_token_policy=SpeechStateBoundaryTokenPolicy.EXACT_MODEL_TOKEN,
        boundary_token_id=2150,
        successor_start=SpeechStateSuccessorStart.MODEL_PREFILL,
        successor_text_entry=SpeechStateSuccessorTextEntry.PREFILL_SUCCESSOR_TEXT,
        cursor_policy=cursor_policy,
        transfer=SpeechStateTransfer.TRAINED_APPROXIMATE,
        retain_talker_kv=False,
        retain_code_predictor_state=False,
        retain_c2w_kv=True,
        retain_c2w_history=True,
        training_scope="x2-clause-continuation-v1",
        talker_carry=SpeechStateTalkerCarry.HIDDEN_TAIL_BRIDGE,
        talker_hidden_tail=4,
    ).to_dict()


def _manifest(speech_state: dict) -> dict:
    return build_manifest_for_export(
        "custom-1.7b",
        {
            "talker_num_layers": 1,
            "talker_hidden_size": 4,
            "talker_num_kv_heads": 1,
            "talker_head_dim": 4,
            "talker_vocab_size": 32,
        },
        {
            "num_code2wav_hidden_layers": 1,
            "c2w_state_input_names": [],
            "c2w_state_output_names": [],
            "initial_state_shapes": [],
        },
        speech_state=speech_state,
    )


def test_manifest_preserves_valid_model_contract():
    manifest = _manifest(
        {
            "model_fingerprint": "x2-model-v1",
            "runtime_fingerprint": "trt-runtime-v1",
            "model_contract": _contract_dict(),
        }
    )

    assert manifest["speech_state"]["model_contract"]["talker_hidden_tail"] == 4
    assert manifest["speech_state"]["model_contract"]["model_fingerprint"] == (
        "x2-model-v1"
    )


def test_manifest_rejects_contract_identity_mismatch():
    contract = _contract_dict()
    contract["model_fingerprint"] = "other-model"

    with pytest.raises(ValueError, match="does not match"):
        _manifest(
            {
                "model_fingerprint": "x2-model-v1",
                "runtime_fingerprint": "trt-runtime-v1",
                "model_contract": contract,
            }
        )


def test_manifest_rejects_non_string_runtime_fingerprint():
    with pytest.raises(ValueError, match="requires model_fingerprint"):
        _manifest(
            {
                "model_fingerprint": "x2-model-v1",
                "runtime_fingerprint": 1,
                "model_contract": _contract_dict(),
            }
        )


def test_manifest_rejects_malformed_model_contract():
    with pytest.raises(ValueError, match="model_contract is invalid"):
        _manifest(
            {
                "model_fingerprint": "x2-model-v1",
                "runtime_fingerprint": "trt-runtime-v1",
                "model_contract": {"checkpoint_id": "missing-fields"},
            }
        )


def test_manifest_bundle_fills_export_layout_hash():
    manifest = _manifest(
        {
            "model_fingerprint": "x2-model-v1",
            "runtime_fingerprint": "trt-runtime-v1",
            "model_contract": _contract_dict(),
            "bundle": {
                "schema_version": 1,
                "artifacts": {
                    "model_weights": {
                        "path": "weights/model.safetensors",
                        "sha256": "a" * 64,
                        "source_path": "source/model.safetensors",
                        "source_sha256": "a" * 64,
                    },
                    "runtime_plan": {
                        "path": "runtime/model.plan",
                        "sha256": "b" * 64,
                    },
                    "cursor_head": {
                        "path": "weights/qwen3_tts_12hz_la1_seed0.pt",
                        "sha256": "c" * 64,
                        "source_path": "source/qwen3_tts_12hz_la1_seed0.pt",
                        "source_sha256": "c" * 64,
                    },
                },
            },
        }
    )

    assert manifest["speech_state"]["bundle"]["code2wav_layout_sha256"] == (
        code2wav_layout_fingerprint(manifest["code2wav_fused"])
    )


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_manifest_rejects_non_integer_bundle_schema_version(schema_version):
    with pytest.raises(ValueError, match="schema_version must be 1"):
        _manifest(
            {
                "model_fingerprint": "x2-model-v1",
                "runtime_fingerprint": "trt-runtime-v1",
                "model_contract": _contract_dict(),
                "bundle": {"schema_version": schema_version},
            }
        )


def test_manifest_rejects_migrate_contract_without_complete_cursor_abi():
    contract = _contract_dict(cursor_policy=SpeechStateCursorPolicy.MIGRATE)

    with pytest.raises(ValueError, match="complete native cursor recurrent ABI"):
        _manifest(
            {
                "model_fingerprint": "x2-model-v1",
                "runtime_fingerprint": "trt-runtime-v1",
                "model_contract": contract,
                "bundle": {
                    "schema_version": 1,
                    "artifacts": {
                        "model_weights": {
                            "path": "weights/model.safetensors",
                            "sha256": "a" * 64,
                            "source_path": "source/model.safetensors",
                            "source_sha256": "a" * 64,
                        },
                        "runtime_plan": {
                            "path": "runtime/model.plan",
                            "sha256": "b" * 64,
                        },
                        "cursor_head": {
                            "path": "weights/qwen3_tts_12hz_la1_seed0.pt",
                            "sha256": "c" * 64,
                            "source_path": "source/qwen3_tts_12hz_la1_seed0.pt",
                            "source_sha256": "c" * 64,
                        },
                    },
                },
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", 1),
        ("sha256", "not-a-hash"),
        ("source_path", "../source/model.safetensors"),
        ("source_sha256", True),
    ],
)
def test_manifest_rejects_malformed_artifact_fields(field, value):
    descriptor = {
        "path": "weights/model.safetensors",
        "sha256": "a" * 64,
        "source_path": "source/model.safetensors",
        "source_sha256": "a" * 64,
    }
    descriptor[field] = value
    with pytest.raises(ValueError, match="model_weights.*incomplete"):
        _manifest(
            {
                "model_fingerprint": "x2-model-v1",
                "runtime_fingerprint": "trt-runtime-v1",
                "model_contract": _contract_dict(),
                "bundle": {
                    "schema_version": 1,
                    "artifacts": {
                        "model_weights": descriptor,
                        "runtime_plan": {
                            "path": "runtime/model.plan",
                            "sha256": "b" * 64,
                        },
                        "cursor_head": {
                            "path": "weights/qwen3_tts_12hz_la1_seed0.pt",
                            "sha256": "c" * 64,
                            "source_path": "source/qwen3_tts_12hz_la1_seed0.pt",
                            "source_sha256": "c" * 64,
                        },
                    },
                },
            }
        )
