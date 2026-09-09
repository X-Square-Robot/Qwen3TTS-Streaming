from dataclasses import replace
from types import SimpleNamespace

import torch
import pytest

from engine.backend.engine_loop import EngineSegment
from engine.backend.executor import Executor, _select_cuda_graph_profile
from engine.backend.kv_cache_pool import KVCachePool, ModelConfig
from engine.backend.slot_snapshot import SlotOwnedSnapshot, StandaloneSlotSnapshot
from engine.core.speech_state_bundle import SpeechStateBundleValidation
from engine.runtime.release_gate import ReleaseGate
from engine.backend.speech_state import (
    SpeechStateContractError,
    SpeechStateHandleStore,
    capture_segment_runtime_metadata,
)
from engine.backend.state_transfer import MigrationRequest
from engine.core.speech_state import (
    SpeechStateCapability,
    SpeechStateHandleKind,
    SpeechStateOperation,
    SpeechStateTransfer,
)
from engine.core.speech_state_model import (
    SpeechStateBoundaryPhase,
    SpeechStateBoundaryTokenPolicy,
    SpeechStateCheckpointBoundary,
    SpeechStateCursorPolicy,
    SpeechStateSuccessorStart,
    SpeechStateSuccessorTextEntry,
    SpeechStateTalkerCarry,
)


def _metadata(session_id: str, segment_idx: int):
    return capture_segment_runtime_metadata(EngineSegment(session_id, segment_idx))


def _executor(pool):
    executor = Executor.__new__(Executor)
    executor._kv_pool = pool
    return executor


def _runtime_contract_dict():
    return {
        "model_fingerprint": "model-v1",
        "checkpoint_id": "x2-test",
        "boundaries": [SpeechStateCheckpointBoundary.SEGMENT_END.value],
        "boundary_phase": SpeechStateBoundaryPhase.CODEC_EOS.value,
        "boundary_token_policy": SpeechStateBoundaryTokenPolicy.EXACT_MODEL_TOKEN.value,
        "boundary_token_id": 2150,
        "successor_start": SpeechStateSuccessorStart.MODEL_PREFILL.value,
        "successor_text_entry": SpeechStateSuccessorTextEntry.PREFILL_SUCCESSOR_TEXT.value,
        "cursor_policy": SpeechStateCursorPolicy.REANCHOR.value,
        "transfer": SpeechStateTransfer.EXACT.value,
        "retain_talker_kv": False,
        "retain_code_predictor_state": False,
        "retain_c2w_kv": True,
        "retain_c2w_history": True,
        "training_scope": "test",
        "talker_carry": SpeechStateTalkerCarry.HIDDEN_TAIL_BRIDGE.value,
        "talker_hidden_tail": 4,
    }


@pytest.mark.parametrize(
    ("cursor_enabled", "requested", "expected"),
    [
        (True, "", 0),
        (True, "1", 0),
        (True, "bad", 0),
        (False, "", 1),
        (False, "0", 0),
        (False, "99", 1),
    ],
)
def test_cuda_graph_profile_selection_is_parity_safe(
    cursor_enabled, requested, expected
):
    assert (
        _select_cuda_graph_profile(
            2,
            cursor_enabled=cursor_enabled,
            requested=requested,
        )
        == expected
    )


def _supported_executor(manifest):
    executor = Executor.__new__(Executor)
    executor._manifest = manifest
    executor._speech_state_model_fingerprint = ""
    executor._speech_state_runtime_fingerprint = ""
    executor._speech_state_adapter = SimpleNamespace(
        capability=SpeechStateCapability(
            supported=True,
            handle_kind=SpeechStateHandleKind.OPAQUE,
            operations=(SpeechStateOperation.SEGMENT_HANDOFF,),
            transfer=SpeechStateTransfer.EXACT,
        )
    )
    return executor


def test_executor_speech_state_capability_requires_matching_model_contract():
    manifest = {
        "speech_state": {
            "model_fingerprint": "model-v1",
            "runtime_fingerprint": "runtime-v1",
            "model_contract": _runtime_contract_dict(),
        }
    }
    executor = _supported_executor(manifest)

    assert executor.speech_state_model_contract is not None
    assert executor.speech_state_capability.supports_segment_handoff
    assert executor.speech_state_capability_reason == "enabled"

    manifest["speech_state"]["model_contract"]["model_fingerprint"] = "other-model"
    assert executor.speech_state_model_contract is None
    assert executor.speech_state_capability == SpeechStateCapability.disabled()


def test_executor_rejects_cursor_migrate_without_cursor_state_restore_abi():
    manifest = {
        "speech_state": {
            "model_fingerprint": "model-v1",
            "runtime_fingerprint": "runtime-v1",
            "model_contract": {
                **_runtime_contract_dict(),
                "cursor_policy": SpeechStateCursorPolicy.MIGRATE.value,
            },
        }
    }
    executor = _supported_executor(manifest)

    assert executor.speech_state_model_contract is not None
    assert executor.speech_state_capability == SpeechStateCapability.disabled()
    assert executor.speech_state_capability_reason == "cursor_state_handoff_unavailable"


def test_executor_speech_state_capability_stays_disabled_without_contract():
    executor = _supported_executor(
        {
            "speech_state": {
                "model_fingerprint": "model-v1",
                "runtime_fingerprint": "runtime-v1",
            }
        }
    )

    assert executor.speech_state_capability == SpeechStateCapability.disabled()
    assert executor.speech_state_capability_reason == "missing_or_invalid_model_contract"


def test_executor_bundle_verification_gate_stays_fail_closed():
    executor = _supported_executor(
        {
            "speech_state": {
                "model_fingerprint": "model-v1",
                "runtime_fingerprint": "runtime-v1",
                "model_contract": _runtime_contract_dict(),
            }
        }
    )
    executor._speech_state_bundle_validation = SpeechStateBundleValidation(
        False, "model_weights_hash_mismatch", "model-v1", "runtime-v1"
    )

    assert executor.speech_state_capability == SpeechStateCapability.disabled()
    assert executor.speech_state_capability_reason == "model_weights_hash_mismatch"


def test_executor_reports_loaded_bundle_failure_before_default_adapter_reason():
    executor = Executor.__new__(Executor)
    executor._manifest = {}
    executor._speech_state_adapter = SimpleNamespace(
        capability=SpeechStateCapability.disabled()
    )
    executor._speech_state_bundle_validation = SpeechStateBundleValidation(
        False, "missing_speech_state"
    )

    assert executor.speech_state_capability_reason == "missing_speech_state"


def test_executor_release_evidence_gate_closes_native_and_state_advertising():
    executor = _supported_executor(
        {
            "native_cursor": {"enabled": True, "progress_available": True},
            "speech_state": {
                "model_fingerprint": "model-v1",
                "runtime_fingerprint": "runtime-v1",
                "model_contract": _runtime_contract_dict(),
            },
        }
    )
    executor._cursor_enabled = True
    executor._release_gate = ReleaseGate.disabled()
    executor._speech_state_bundle_validation = SpeechStateBundleValidation(
        True, "verified", "model-v1", "runtime-v1"
    )

    assert executor.native_cursor_capability["progress_available"] is False
    assert executor.native_cursor_capability["reason"] == "release_evidence_missing"
    assert executor.speech_state_capability == SpeechStateCapability.disabled()
    assert executor.speech_state_capability_reason == "release_evidence_missing"


def test_executor_verified_bundle_release_reason_precedes_default_adapter():
    executor = _supported_executor(
        {
            "speech_state": {
                "model_fingerprint": "model-v1",
                "runtime_fingerprint": "runtime-v1",
                "model_contract": _runtime_contract_dict(),
            }
        }
    )
    executor._speech_state_adapter = SimpleNamespace(
        capability=SpeechStateCapability.disabled()
    )
    executor._release_gate = ReleaseGate.disabled()
    executor._speech_state_bundle_validation = SpeechStateBundleValidation(
        True, "verified", "model-v1", "runtime-v1"
    )

    assert executor.speech_state_capability_reason == "release_evidence_missing"


def test_executor_native_cursor_capability_rejects_truthy_progress_metadata():
    executor = _supported_executor(
        {
            "native_cursor": {"enabled": True, "progress_available": "true"},
            "speech_state": {
                "model_fingerprint": "model-v1",
                "runtime_fingerprint": "runtime-v1",
                "model_contract": _runtime_contract_dict(),
            },
        }
    )
    executor._cursor_enabled = True
    executor._release_gate = ReleaseGate(
        True,
        True,
        "verified",
        "verified",
    )

    assert executor.native_cursor_capability == {
        "enabled": True,
        "progress_available": False,
        "reason": "malformed_native_cursor_capability",
    }


@pytest.mark.parametrize("value", [1, True, [], {}])
def test_executor_speech_state_fingerprints_reject_non_string_values(value):
    executor = Executor.__new__(Executor)
    executor._manifest = {
        "speech_state": {
            "model_fingerprint": value,
            "runtime_fingerprint": value,
        }
    }
    executor._speech_state_model_fingerprint = ""
    executor._speech_state_runtime_fingerprint = ""

    assert executor.speech_state_fingerprints == ("", "")


def test_executor_constructor_completes_before_speech_state_properties(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: object())

    executor = Executor(
        engine_dir="",
        weights_dir="",
        max_batch_size=1,
        max_seq_len=4,
    )

    assert hasattr(executor, "_compute_stream")
    assert executor.speech_state_fingerprints == ("", "")


def test_state_transfer_fence_settles_compute_and_deferred_copy_streams(monkeypatch):
    calls = []
    executor = Executor.__new__(Executor)
    executor._device = torch.device("cuda")
    executor._compute_stream = SimpleNamespace(synchronize=lambda: calls.append("compute"))
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(synchronize=lambda: calls.append("current")),
    )

    executor.synchronize_state_transfer()

    assert calls == ["compute", "current"]


def test_state_transfer_fence_deduplicates_the_same_cuda_stream(monkeypatch):
    calls = []
    executor = Executor.__new__(Executor)
    executor._device = torch.device("cuda")
    executor._compute_stream = SimpleNamespace(
        cuda_stream=7, synchronize=lambda: calls.append("compute")
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(
            cuda_stream=7, synchronize=lambda: calls.append("current")
        ),
    )

    executor.synchronize_state_transfer()

    assert calls == ["compute"]


def test_executor_fingerprints_are_manifest_owned_and_fail_closed():
    executor = Executor.__new__(Executor)
    executor._manifest = {
        "speech_state": {
            "model_fingerprint": "model-v1",
            "runtime_fingerprint": "runtime-v1",
        }
    }
    assert executor.speech_state_fingerprints == ("model-v1", "runtime-v1")

    executor._manifest = {"speech_state": {"model_fingerprint": "model-v1"}}
    assert executor.speech_state_fingerprints == ("model-v1", "")


def test_capture_assembles_nonpooled_slot_as_standalone_payload():
    config = ModelConfig(
        dtype=torch.float32, num_layers=1, kv_heads=1, head_dim=2,
        n_c2w_layers=1, c2w_kv_heads=1, c2w_head_dim=2,
        max_seq_len=8, c2w_sliding_window=4,
    )
    pool = KVCachePool(2, config, torch.device("cpu"), preallocate=False)
    slot = pool.allocate("session")
    slot.segment_idx = 3
    slot.past_len = 2
    slot.talker_kv = torch.ones(1, 2, 1, 2, 2)
    slot.c2w_len = 1
    slot.c2w_kv = torch.ones(1, 2, 1, 1, 2)
    executor = _executor(pool)

    bundle = executor.capture_speech_state_bundle(
        slot, _metadata("session", 3), max_tensor_bytes=100_000
    )

    assert isinstance(bundle.slot_payload, StandaloneSlotSnapshot)
    assert bundle.pooled_talker_payload is None
    assert bundle.pooled_c2w_payload is None


def test_capture_assembles_pool_rows_and_preserves_source_identity():
    config = ModelConfig(
        dtype=torch.float32, num_layers=1, kv_heads=1, head_dim=2,
        n_c2w_layers=1, c2w_kv_heads=1, c2w_head_dim=2,
        max_seq_len=8, c2w_sliding_window=4,
    )
    pool = KVCachePool(2, config, torch.device("cpu"), preallocate=True)
    slot = pool.allocate("session")
    slot.segment_idx = 4
    slot.past_len = 2
    pool._talker_kv_pool[slot.slot_id, :, :, :2, :].fill_(2)
    slot.c2w_len = pool.write_c2w_right_aligned(
        slot.slot_id, torch.full((1, 2, 1, 1, 2), 3.0)
    )
    slot.c2w_pooled = True
    executor = _executor(pool)

    bundle = executor.capture_speech_state_bundle(
        slot, _metadata("session", 4), max_tensor_bytes=100_000
    )

    assert isinstance(bundle.slot_payload, SlotOwnedSnapshot)
    assert bundle.source_slot_id == slot.slot_id
    assert bundle.source_allocation_epoch == slot.allocation_epoch
    assert bundle.pooled_talker_payload.shape[3] == 2
    assert bundle.pooled_c2w_payload.shape[3] == 1
    assert torch.all(bundle.pooled_talker_payload == 2)
    assert torch.all(bundle.pooled_c2w_payload == 3)


def test_capture_includes_arena_rows_and_parity():
    config = ModelConfig(
        dtype=torch.float32, num_layers=1, kv_heads=1, head_dim=2,
        n_c2w_layers=1, c2w_kv_heads=1, c2w_head_dim=2,
        max_seq_len=8, c2w_sliding_window=4,
    )
    pool = KVCachePool(1, config, torch.device("cpu"), preallocate=True)
    slot = pool.allocate("session")
    slot.segment_idx = 5
    slot.c2w_pooled = True
    slot.c2w_len = 1
    pool.write_c2w_right_aligned(
        slot.slot_id, torch.full((1, 2, 1, 1, 2), 3.0)
    )
    executor = _executor(pool)
    executor._device = torch.device("cpu")
    executor._config = config
    executor._speech_state_model_fingerprint = "model-test"
    executor._speech_state_runtime_fingerprint = "runtime-test"
    executor._c2w_conv_shapes = [(1, 1, 2)]
    executor._c2w_transconv_shapes = [(1, 1, 1)]
    executor._c2w_arena_a = [torch.full((1, 1, 2), 4.0), torch.full((1, 1, 1), 5.0)]
    executor._c2w_arena_b = [torch.full((1, 1, 2), 6.0), torch.full((1, 1, 1), 7.0)]
    slot.c2w_conv_states, slot.c2w_transconv_states = executor._arena_row_views(
        executor._c2w_arena_a, slot.slot_id
    )
    slot._c2w_conv_write, slot._c2w_transconv_write = executor._arena_row_views(
        executor._c2w_arena_b, slot.slot_id
    )
    slot.c2w_arena_backed = True
    slot.c2w_write_in_a = False

    bundle = executor.capture_speech_state_bundle(
        slot, _metadata("session", 5), max_tensor_bytes=100_000
    )

    assert bundle.c2w_arena_payload.source_slot_id == slot.slot_id
    assert bundle.c2w_arena_payload.source_allocation_epoch == slot.allocation_epoch
    assert bundle.c2w_arena_payload.write_in_a is False
    assert torch.all(bundle.c2w_arena_payload.read_conv[0] == 4)
    assert torch.all(bundle.c2w_arena_payload.write_conv[0] == 6)


@pytest.mark.parametrize("metadata", [
    _metadata("other", 1),
    _metadata("session", 2),
])
def test_capture_rejects_segment_owner_mismatch(metadata):
    config = ModelConfig(dtype=torch.float32, max_seq_len=4, c2w_sliding_window=2)
    pool = KVCachePool(1, config, torch.device("cpu"), preallocate=False)
    slot = pool.allocate("session")
    slot.segment_idx = 1
    with pytest.raises(SpeechStateContractError, match="segment ownership"):
        _executor(pool).capture_speech_state_bundle(
            slot, metadata, max_tensor_bytes=100_000
        )


def test_capture_rejects_total_budget_after_payload_assembly():
    config = ModelConfig(
        dtype=torch.float32, num_layers=1, kv_heads=1, head_dim=2,
        n_c2w_layers=1, c2w_kv_heads=1, c2w_head_dim=2,
        max_seq_len=8, c2w_sliding_window=4,
    )
    pool = KVCachePool(1, config, torch.device("cpu"), preallocate=True)
    slot = pool.allocate("session")
    slot.segment_idx = 0
    slot.past_len = 2
    pool._talker_kv_pool[slot.slot_id, :, :, :2, :].fill_(2)
    with pytest.raises(SpeechStateContractError, match="budget"):
        _executor(pool).capture_speech_state_bundle(
            slot, _metadata("session", 0), max_tensor_bytes=1
        )


def test_restore_round_trips_pool_payload_into_a_new_allocation():
    config = ModelConfig(
        dtype=torch.float32, num_layers=1, kv_heads=1, head_dim=2,
        n_c2w_layers=1, c2w_kv_heads=1, c2w_head_dim=2,
        max_seq_len=8, c2w_sliding_window=4,
    )
    pool = KVCachePool(2, config, torch.device("cpu"), preallocate=True)
    source = pool.allocate("session")
    source.segment_idx = 1
    source.past_len = 2
    pool._talker_kv_pool[source.slot_id, :, :, :2, :].fill_(2)
    source.c2w_len = pool.write_c2w_right_aligned(
        source.slot_id, torch.full((1, 2, 1, 1, 2), 3.0)
    )
    source.c2w_pooled = True
    executor = _executor(pool)
    bundle = executor.capture_speech_state_bundle(
        source, _metadata("session", 1), max_tensor_bytes=100_000
    )
    target = pool.allocate("session")
    target.segment_idx = 1
    executor.restore_speech_state_bundle(
        target, bundle, expected_allocation_epoch=target.allocation_epoch
    )

    assert target.past_len == 2
    assert target.c2w_len == 1
    assert target.c2w_pooled is True
    assert torch.all(pool._talker_kv_pool[target.slot_id, :, :, :2, :] == 2)
    assert torch.all(
        pool._c2w_kv_pool[target.slot_id, :, :, -1:, :] == 3
    )


def test_restore_validates_every_payload_before_mutating_target():
    config = ModelConfig(
        dtype=torch.float32, num_layers=1, kv_heads=1, head_dim=2,
        n_c2w_layers=1, c2w_kv_heads=1, c2w_head_dim=2,
        max_seq_len=8, c2w_sliding_window=4,
    )
    pool = KVCachePool(2, config, torch.device("cpu"), preallocate=True)
    source = pool.allocate("session")
    source.segment_idx = 1
    source.past_len = 2
    pool._talker_kv_pool[source.slot_id, :, :, :2, :].fill_(2)
    executor = _executor(pool)
    bundle = executor.capture_speech_state_bundle(
        source, _metadata("session", 1), max_tensor_bytes=100_000
    )
    target = pool.allocate("session")
    target.segment_idx = 1
    before = pool._talker_kv_pool[target.slot_id].clone()
    bad = replace(
        bundle,
        pooled_talker_payload=torch.zeros(1, 1, 1, 1, 1),
    )
    with pytest.raises(ValueError, match="shape"):
        executor.restore_speech_state_bundle(
            target, bad, expected_allocation_epoch=target.allocation_epoch
        )
    assert target.past_len == 0
    assert torch.equal(pool._talker_kv_pool[target.slot_id], before)


def test_restore_round_trips_nonpooled_slot_into_allocated_target():
    config = ModelConfig(
        dtype=torch.float32, num_layers=1, kv_heads=1, head_dim=2,
        n_c2w_layers=1, c2w_kv_heads=1, c2w_head_dim=2,
        max_seq_len=8, c2w_sliding_window=4,
    )
    pool = KVCachePool(2, config, torch.device("cpu"), preallocate=False)
    source = pool.allocate("session")
    source.segment_idx = 2
    source.past_len = 2
    source.talker_kv = torch.full((1, 2, 1, 2, 2), 2.0)
    source.c2w_len = 1
    source.c2w_kv = torch.full((1, 2, 1, 1, 2), 3.0)
    executor = _executor(pool)
    bundle = executor.capture_speech_state_bundle(
        source, _metadata("session", 2), max_tensor_bytes=100_000
    )
    target = pool.allocate("session")
    target.segment_idx = 2
    executor.restore_speech_state_bundle(
        target, bundle, expected_allocation_epoch=target.allocation_epoch
    )

    assert target.past_len == 2
    assert target.c2w_len == 1
    assert torch.all(target.talker_kv == 2)
    assert torch.all(target.c2w_kv == 3)


def test_engine_boundary_migration_swaps_only_after_target_restore():
    from engine.backend.engine_loop import EngineLoop, EngineSegment

    config = ModelConfig(
        dtype=torch.float32, num_layers=1, kv_heads=1, head_dim=2,
        n_c2w_layers=1, c2w_kv_heads=1, c2w_head_dim=2,
        max_seq_len=8, c2w_sliding_window=4,
    )
    pool = KVCachePool(2, config, torch.device("cpu"), preallocate=False)
    source = pool.allocate("s:2")
    source.segment_idx = 2
    source.retry_idx = 0
    source.past_len = 2
    source.talker_kv = torch.full((1, 2, 1, 2, 2), 2.0)
    source.frame_idx = 9
    seg = EngineSegment("s", 2)
    seg.state = "active"
    seg.slot = source
    group = SimpleNamespace(segments={2: seg})
    executor = _executor(pool)
    executor._device = torch.device("cpu")
    executor._config = config
    executor._speech_state_model_fingerprint = "model-test"
    executor._speech_state_runtime_fingerprint = "runtime-test"
    loop = object.__new__(EngineLoop)
    loop._speech_state_capability = SpeechStateCapability(
        supported=True, handle_kind=SpeechStateHandleKind.OPAQUE,
        operations=(SpeechStateOperation.PAUSE_RESUME,),
        transfer=SpeechStateTransfer.EXACT,
    )
    loop._executor = executor
    loop._speech_state_handles = SpeechStateHandleStore()
    loop._speech_state_generation = 0
    loop._groups = {"s": group}
    loop._seg_by_slot = {source.slot_id: seg}

    old_slot_id = source.slot_id
    loop._migrate_speech_state(MigrationRequest("s", 2, 0, source.allocation_epoch, 100_000))

    assert seg.slot is not source
    assert seg.slot.slot_id != old_slot_id
    assert seg.slot.frame_idx == 9
    assert seg.slot.past_len == 2
    assert loop._seg_by_slot[seg.slot.slot_id] is seg
    assert old_slot_id not in loop._seg_by_slot
    assert pool.get(old_slot_id).is_free


def test_engine_boundary_migration_keeps_source_when_target_restore_fails():
    from engine.backend.engine_loop import EngineLoop, EngineSegment

    config = ModelConfig(dtype=torch.float32, max_seq_len=4, c2w_sliding_window=2)
    pool = KVCachePool(2, config, torch.device("cpu"), preallocate=False)
    source = pool.allocate("s:2")
    source.segment_idx = 2
    source.retry_idx = 0
    seg = EngineSegment("s", 2)
    seg.state = "active"
    seg.slot = source

    class FailingExecutor:
        def __init__(self):
            self.kv_pool = pool
            self.speech_state_fingerprints = ("model-test", "runtime-test")

        def synchronize_state_transfer(self):
            return None

        def capture_speech_state_bundle(self, *args, **kwargs):
            return object()

        def restore_speech_state_bundle(self, *args, **kwargs):
            raise RuntimeError("target restore failed")

    loop = object.__new__(EngineLoop)
    loop._speech_state_capability = SpeechStateCapability(
        supported=True, handle_kind=SpeechStateHandleKind.OPAQUE,
        operations=(SpeechStateOperation.PAUSE_RESUME,),
        transfer=SpeechStateTransfer.EXACT,
    )
    loop._executor = FailingExecutor()
    loop._speech_state_handles = SpeechStateHandleStore()
    loop._speech_state_generation = 0
    loop._groups = {"s": SimpleNamespace(segments={2: seg})}
    loop._seg_by_slot = {source.slot_id: seg}

    with pytest.raises(RuntimeError, match="target restore"):
        loop._migrate_speech_state(
            MigrationRequest("s", 2, 0, source.allocation_epoch, 100_000)
        )

    assert seg.slot is source
    assert loop._seg_by_slot[source.slot_id] is seg
    assert pool.free_count == 1
