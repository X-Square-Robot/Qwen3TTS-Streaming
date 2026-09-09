from dataclasses import FrozenInstanceError

import pytest

from engine.core.speech_state import SpeechStateTransfer
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


def _contract(**overrides):
    values = {
        "model_fingerprint": "x2-model-v1",
        "checkpoint_id": "x2-checkpoint-202609",
        "boundaries": (SpeechStateCheckpointBoundary.PRE_CODEC_EOS,),
        "boundary_phase": SpeechStateBoundaryPhase.DECODE_FRAME,
        "boundary_token_policy": SpeechStateBoundaryTokenPolicy.EXACT_MODEL_TOKEN,
        "boundary_token_id": 1024,
        "successor_start": SpeechStateSuccessorStart.READY_TO_DECODE,
        "successor_text_entry": SpeechStateSuccessorTextEntry.APPEND_TOKENS,
        "cursor_policy": SpeechStateCursorPolicy.REANCHOR,
        "transfer": SpeechStateTransfer.EXACT,
        "retain_talker_kv": True,
        "retain_code_predictor_state": False,
        "retain_c2w_kv": True,
        "retain_c2w_history": True,
        "training_scope": "x2-clause-continuation-v1",
        "talker_carry": SpeechStateTalkerCarry.DIRECT_KV,
        "talker_hidden_tail": 0,
    }
    values.update(overrides)
    return SpeechStateModelContract(**values)


def test_model_contract_exposes_explicit_successor_semantics():
    contract = _contract(cursor_policy=SpeechStateCursorPolicy.MIGRATE)

    assert contract.supports_segment_handoff
    assert contract.to_dict() == {
        "model_fingerprint": "x2-model-v1",
        "checkpoint_id": "x2-checkpoint-202609",
        "boundaries": ["pre_codec_eos"],
        "boundary_phase": "decode_frame",
        "boundary_token_policy": "exact_model_token",
        "boundary_token_id": 1024,
        "successor_start": "ready_to_decode",
        "successor_text_entry": "append_tokens",
        "cursor_policy": "migrate",
        "transfer": "exact",
        "retain_talker_kv": True,
        "retain_code_predictor_state": False,
        "retain_c2w_kv": True,
        "retain_c2w_history": True,
        "training_scope": "x2-clause-continuation-v1",
        "talker_carry": "direct_kv",
        "talker_hidden_tail": 0,
    }
    with pytest.raises(FrozenInstanceError):
        contract.model_fingerprint = "other"


def test_model_contract_mapping_round_trip_is_typed():
    contract = _contract()

    restored = SpeechStateModelContract.from_mapping(contract.to_dict())

    assert restored == contract


@pytest.mark.parametrize(
    "value",
    [None, "contract", {"model_fingerprint": "x2"}],
)
def test_model_contract_mapping_requires_complete_object(value):
    with pytest.raises(ValueError):
        SpeechStateModelContract.from_mapping(value)


def test_model_contract_mapping_rejects_string_boundaries():
    values = _contract().to_dict()
    values["boundaries"] = "segment_end"

    with pytest.raises(ValueError, match="boundaries must be a list"):
        SpeechStateModelContract.from_mapping(values)


def test_model_contract_does_not_infer_transfer_from_retained_state():
    contract = _contract(
        retain_talker_kv=False,
        talker_carry=SpeechStateTalkerCarry.NONE,
        retain_c2w_kv=False,
        transfer=SpeechStateTransfer.RECONSTRUCTION_APPROXIMATE,
    )

    assert contract.supports_segment_handoff
    assert contract.transfer is SpeechStateTransfer.RECONSTRUCTION_APPROXIMATE


def test_x2_contract_can_declare_hidden_tail_bridge_without_talker_kv():
    contract = _contract(
        retain_talker_kv=False,
        talker_carry=SpeechStateTalkerCarry.HIDDEN_TAIL_BRIDGE,
        talker_hidden_tail=4,
    )

    assert contract.talker_carry is SpeechStateTalkerCarry.HIDDEN_TAIL_BRIDGE
    assert contract.talker_hidden_tail == 4


def test_ready_decode_cannot_require_successor_prefill():
    with pytest.raises(ValueError, match="cannot require successor prefill"):
        _contract(
            successor_text_entry=SpeechStateSuccessorTextEntry.PREFILL_SUCCESSOR_TEXT
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_fingerprint": 1},
        {"checkpoint_id": 1},
        {"training_scope": 1},
        {"model_fingerprint": ""},
        {"checkpoint_id": ""},
        {"training_scope": ""},
        {"boundaries": ()},
        {
            "boundaries": (
                SpeechStateCheckpointBoundary.SEGMENT_END,
                SpeechStateCheckpointBoundary.SEGMENT_END,
            )
        },
        {"boundaries": ("unknown",)},
        {"boundary_phase": "unknown"},
        {"boundary_token_policy": "unknown"},
        {"successor_start": "unknown"},
        {"successor_text_entry": "unknown"},
        {"cursor_policy": "unknown"},
        {"transfer": "unknown"},
        {"retain_talker_kv": 1},
        {"retain_code_predictor_state": 0},
        {"talker_carry": SpeechStateTalkerCarry.HIDDEN_TAIL_BRIDGE},
        {"talker_hidden_tail": -1},
    ],
)
def test_malformed_model_contract_fails_closed(overrides):
    with pytest.raises(ValueError):
        _contract(**overrides)


@pytest.mark.parametrize("token_id", [None, -1, True])
def test_exact_token_boundary_requires_nonnegative_integer(token_id):
    with pytest.raises(ValueError, match="token boundary"):
        _contract(boundary_token_id=token_id)


def test_no_eos_boundary_does_not_accept_token_id():
    with pytest.raises(ValueError, match="must not declare"):
        _contract(
            boundary_token_policy=SpeechStateBoundaryTokenPolicy.NO_EOS,
            boundary_token_id=1024,
        )


@pytest.mark.parametrize(
    "boundary",
    [
        SpeechStateCheckpointBoundary.PRE_CODEC_EOS,
        SpeechStateCheckpointBoundary.POST_CODEC_EOS,
        SpeechStateCheckpointBoundary.SEGMENT_END,
    ],
)
def test_model_boundaries_can_advertise_successor_handoff(boundary):
    assert _contract(boundaries=(boundary,)).supports_segment_handoff


def test_explicit_pause_hard_boundary_is_not_successor_handoff():
    contract = _contract(
        boundaries=(SpeechStateCheckpointBoundary.EXPLICIT_PAUSE,),
        successor_start=SpeechStateSuccessorStart.HARD_BOUNDARY_PREFILL,
        successor_text_entry=SpeechStateSuccessorTextEntry.PREFILL_SUCCESSOR_TEXT,
        cursor_policy=SpeechStateCursorPolicy.DISABLE,
        transfer=SpeechStateTransfer.RECONSTRUCTION_APPROXIMATE,
        retain_talker_kv=False,
        talker_carry=SpeechStateTalkerCarry.NONE,
        retain_c2w_kv=False,
        retain_c2w_history=False,
        training_scope="pause-only",
        boundary_token_policy=SpeechStateBoundaryTokenPolicy.NO_EOS,
        boundary_token_id=None,
    )

    assert not contract.supports_segment_handoff
