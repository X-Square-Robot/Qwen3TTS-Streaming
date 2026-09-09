from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from engine.backend.engine_loop import EngineSegment, EngineSessionGroup
from engine.backend.speech_state import (
    SegmentRuntimeMetadata,
    SpeechStateSnapshotBundle,
    SpeechStateHandleStore,
    NullSpeechStateAdapter,
    SpeechStateContext,
    SpeechStateContractError,
    SpeechStatePhase,
    capability_from_adapter,
    coerce_speech_state_adapter,
    validate_restore_context,
    capture_segment_runtime_metadata,
    project_successor_runtime_metadata,
    restore_segment_runtime_metadata,
)
from engine.core.session import Session
from engine.core.speech_state import (
    SPEECH_STATE_PROTOCOL_VERSION,
    SpeechStateCapability,
    SpeechStateHandle,
    SpeechStateHandleKind,
    SpeechStateOperation,
    SpeechStateTransfer,
)
from engine.core.types import EngineRequest, RequestType, SessionConfig, SessionState
from engine.server import TTSEngine


def test_official_default_is_canonical_disabled_capability():
    capability = SpeechStateCapability.disabled()

    assert capability.to_dict() == {
        "supported": False,
        "protocol_version": SPEECH_STATE_PROTOCOL_VERSION,
        "handle_kind": "none",
        "operations": [],
        "transfer": "none",
        "max_handle_bytes": 0,
    }
    assert capability.supports_pause_resume is False
    assert capability.supports_segment_handoff is False
    assert capability.supports_context_rollover is False


def test_supported_capability_uses_typed_operations_and_stable_wire_order():
    capability = SpeechStateCapability.from_mapping(
        {
            "supported": True,
            "handle_kind": "opaque",
            "operations": ["context_rollover", "pause_resume", "pause_resume"],
            "transfer": "exact",
            "max_handle_bytes": "256",
        }
    )

    assert capability.supported is True
    assert capability.handle_kind is SpeechStateHandleKind.OPAQUE
    assert capability.operations == (
        SpeechStateOperation.PAUSE_RESUME,
        SpeechStateOperation.CONTEXT_ROLLOVER,
    )
    assert capability.transfer is SpeechStateTransfer.EXACT
    assert capability.max_handle_bytes == 256
    assert capability.supports_pause_resume is True
    assert capability.supports_segment_handoff is False


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"supported": True, "handle_kind": "public", "operations": ["pause_resume"], "transfer": "exact"},
        {"supported": True, "handle_kind": "opaque", "operations": [], "transfer": "exact"},
        {"supported": True, "handle_kind": "opaque", "operations": ["pause_resume"], "transfer": "none"},
        {"supported": True, "protocol_version": "future-v2", "handle_kind": "opaque", "operations": ["pause_resume"], "transfer": "exact"},
        {"supported": True, "handle_kind": "opaque", "operations": ["unknown"], "transfer": "exact"},
        {"supported": True, "handle_kind": "opaque", "operations": ["pause_resume"], "transfer": "exact", "max_handle_bytes": -1},
    ],
)
def test_untrusted_capability_metadata_fails_closed(payload):
    assert SpeechStateCapability.from_mapping(payload) == SpeechStateCapability.disabled()


def test_disabled_constructor_cannot_leak_stale_details():
    capability = SpeechStateCapability(
        supported=False,
        handle_kind=SpeechStateHandleKind.OPAQUE,
        operations=(SpeechStateOperation.PAUSE_RESUME,),
        transfer=SpeechStateTransfer.EXACT,
        max_handle_bytes=1024,
    )

    assert capability == SpeechStateCapability.disabled()


def test_opaque_handle_is_immutable_and_redacted_by_default():
    handle = SpeechStateHandle(
        "backend-secret-id",
        generation=3,
        owner_session_id="session-1",
        source_segment_idx=2,
        transfer=SpeechStateTransfer.EXACT,
    )

    assert "backend-secret-id" not in repr(handle)
    assert "handle_id" not in handle.metadata()
    assert handle.metadata(include_id=True)["handle_id"] == "backend-secret-id"
    with pytest.raises(FrozenInstanceError):
        handle.generation = 4


def test_null_adapter_never_captures_or_restores_state():
    adapter = NullSpeechStateAdapter()
    handle = SpeechStateHandle("opaque")

    assert adapter.capability == SpeechStateCapability.disabled()
    assert adapter.capture(session_id="s", segment_idx=0) is None
    assert adapter.restore(handle, session_id="s", segment_idx=1) is False
    assert adapter.release(handle) is None


def test_adapter_boundary_retains_only_complete_typed_adapters():
    capability = SpeechStateCapability(
        supported=True,
        handle_kind=SpeechStateHandleKind.OPAQUE,
        operations=(SpeechStateOperation.SEGMENT_HANDOFF,),
        transfer=SpeechStateTransfer.EXACT,
    )

    class _Adapter:
        def __init__(self):
            self.capability = capability

        def capture(self, **kwargs):
            return None

        def restore(self, handle, **kwargs):
            return False

        def release(self, handle):
            return None

    adapter = _Adapter()

    assert coerce_speech_state_adapter(adapter) is adapter
    assert capability_from_adapter(adapter) is capability
    assert isinstance(coerce_speech_state_adapter(object()), NullSpeechStateAdapter)


def test_async_session_only_carries_capability_metadata_and_keeps_positional_abi():
    config = SessionConfig()
    # The third positional argument has historically been ``state``.
    session = Session("s", config, SessionState.DECODING)

    assert session.state is SessionState.DECODING
    assert session.speech_state_capability == SpeechStateCapability.disabled()
    assert not hasattr(session, "speech_state_handle")


def test_engine_thread_records_capability_but_does_not_create_a_handle():
    capability = SpeechStateCapability(
        supported=True,
        handle_kind=SpeechStateHandleKind.OPAQUE,
        operations=(SpeechStateOperation.SEGMENT_HANDOFF,),
        transfer=SpeechStateTransfer.EXACT,
    )
    request = EngineRequest(type=RequestType.NEW_SESSION, session_id="s")
    group = EngineSessionGroup("s", request, capability)
    segment = EngineSegment("s", 0)

    assert group.speech_state_capability is capability
    assert segment.speech_state_handle is None


def test_group_boundary_accepts_json_capability_but_still_fails_closed():
    request = EngineRequest(type=RequestType.NEW_SESSION, session_id="s")
    group = EngineSessionGroup(
        "s",
        request,
        {
            "supported": True,
            "handle_kind": "opaque",
            "operations": ["segment_handoff"],
            "transfer": "exact",
        },
    )

    assert group.speech_state_capability.supports_segment_handoff is True


def test_engine_capabilities_and_pre_start_health_are_fail_closed():
    engine = TTSEngine()

    expected = SpeechStateCapability.disabled().to_dict()
    for public in (
        engine.describe_capabilities()["speech_state"],
        engine.health_stats()["speech_state"],
    ):
        assert {key: public[key] for key in expected} == expected
        assert public["reason"] == "adapter_disabled"


def test_explicit_adapter_capability_is_visible_without_starting_model():
    capability = SpeechStateCapability(
        supported=True,
        handle_kind=SpeechStateHandleKind.OPAQUE,
        operations=(SpeechStateOperation.PAUSE_RESUME,),
        transfer=SpeechStateTransfer.TRAINED_APPROXIMATE,
    )

    class _Adapter:
        def __init__(self):
            self.capability = capability

        def capture(self, **kwargs):
            return None

        def restore(self, handle, **kwargs):
            return False

        def release(self, handle):
            return None

    engine = TTSEngine(speech_state_adapter=_Adapter())

    assert engine.speech_state_capability is capability
    assert engine.describe_capabilities()["speech_state"]["supported"] is True


def test_restore_context_rejects_wrong_owner_attempt_epoch_and_fingerprint():
    handle = SpeechStateHandle(
        "opaque",
        attempt_id=2,
        slot_allocation_epoch=7,
        owner_session_id="s",
        source_segment_idx=1,
        transfer=SpeechStateTransfer.EXACT,
        model_fingerprint="model-a",
        runtime_fingerprint="runtime-a",
    )
    context = SpeechStateContext(
        operation=SpeechStateOperation.SEGMENT_HANDOFF,
        phase=SpeechStatePhase.RESTORE,
        session_id="s",
        segment_idx=2,
        attempt_id=0,
        slot_allocation_epoch=9,
        source_segment_idx=1,
        source_attempt_id=2,
        source_slot_allocation_epoch=7,
        model_fingerprint="model-a",
        runtime_fingerprint="runtime-a",
        source_generation=0,
    )

    validate_restore_context(handle, context)

    for field, value in (
        ("session_id", "other"),
        ("source_attempt_id", 3),
        ("source_slot_allocation_epoch", 8),
        ("model_fingerprint", "model-b"),
        ("runtime_fingerprint", "runtime-b"),
        ("source_generation", 1),
        ("source_generation", None),
        ("model_fingerprint", ""),
        ("runtime_fingerprint", " "),
    ):
        values = context.to_metadata()
        values[field] = value
        values["operation"] = context.operation.value
        values["phase"] = context.phase.value
        with pytest.raises(SpeechStateContractError):
            validate_restore_context(
                handle,
                SpeechStateContext(**values),
            )


def test_restore_context_requires_restore_phase_and_transfer_class():
    handle = SpeechStateHandle(
        "opaque",
        owner_session_id="s",
        source_segment_idx=1,
        transfer=SpeechStateTransfer.EXACT,
    )
    context = SpeechStateContext(
        operation=SpeechStateOperation.SEGMENT_HANDOFF,
        phase=SpeechStatePhase.CAPTURE,
        session_id="s",
        segment_idx=2,
        attempt_id=0,
        slot_allocation_epoch=1,
        source_segment_idx=1,
    )

    with pytest.raises(SpeechStateContractError):
        validate_restore_context(handle, context)

    inactive = SpeechStateHandle(
        "inactive",
        owner_session_id="s",
        source_segment_idx=1,
        transfer=SpeechStateTransfer.NONE,
    )
    restore_context = SpeechStateContext(
        operation=SpeechStateOperation.SEGMENT_HANDOFF,
        phase=SpeechStatePhase.RESTORE,
        session_id="s",
        segment_idx=2,
        attempt_id=0,
        slot_allocation_epoch=1,
        source_segment_idx=1,
    )
    with pytest.raises(SpeechStateContractError):
        validate_restore_context(inactive, restore_context)


@pytest.mark.parametrize("field", ["model_fingerprint", "runtime_fingerprint"])
@pytest.mark.parametrize("missing", ["", "   "])
def test_restore_rejects_missing_handle_fingerprint(field, missing):
    handle = SpeechStateHandle(
        "opaque", owner_session_id="s", source_segment_idx=0,
        transfer=SpeechStateTransfer.EXACT,
        model_fingerprint="model-a", runtime_fingerprint="runtime-a",
    )
    context = SpeechStateContext(
        operation=SpeechStateOperation.PAUSE_RESUME,
        phase=SpeechStatePhase.RESTORE,
        session_id="s", segment_idx=0, attempt_id=0,
        slot_allocation_epoch=10, source_segment_idx=0,
        source_generation=0,
        model_fingerprint="model-a", runtime_fingerprint="runtime-a",
    )
    validate_restore_context(handle, context)
    with pytest.raises(SpeechStateContractError, match="fingerprint"):
        validate_restore_context(replace(handle, **{field: missing}), context)


@pytest.mark.parametrize("generation", [-1, True, 1.5, float("inf")])
def test_context_rejects_invalid_source_generation(generation):
    with pytest.raises(SpeechStateContractError, match="source_generation"):
        SpeechStateContext(
            operation=SpeechStateOperation.PAUSE_RESUME,
            phase=SpeechStatePhase.RESTORE,
            session_id="s", segment_idx=0, attempt_id=0,
            slot_allocation_epoch=10, source_generation=generation,
        )


def test_handle_store_consumes_opaque_payload_once():
    store = SpeechStateHandleStore()
    handle = store.create(
        generation=3, attempt_id=2, slot_allocation_epoch=11,
        owner_session_id="s", source_segment_idx=4,
        transfer=SpeechStateTransfer.EXACT,
        model_fingerprint="model-a", runtime_fingerprint="runtime-a",
        payload={"private": object()},
    )
    context = SpeechStateContext(
        operation=SpeechStateOperation.SEGMENT_HANDOFF,
        phase=SpeechStatePhase.RESTORE, session_id="s", segment_idx=5,
        attempt_id=2, slot_allocation_epoch=12,
        source_segment_idx=4, source_attempt_id=2,
        source_slot_allocation_epoch=11, source_generation=3,
        model_fingerprint="model-a", runtime_fingerprint="runtime-a",
    )
    payload = store.consume(handle, context)
    assert payload["private"] is not None
    assert len(store) == 0
    with pytest.raises(SpeechStateContractError, match="unknown or already consumed"):
        store.consume(handle, context)


def test_handle_store_rejects_forged_metadata_without_destroying_entry():
    store = SpeechStateHandleStore()
    handle = store.create(
        generation=0, attempt_id=0, slot_allocation_epoch=1,
        owner_session_id="s", source_segment_idx=0,
        transfer=SpeechStateTransfer.EXACT,
        model_fingerprint="m", runtime_fingerprint="r",
        payload=object(),
    )

    forged = replace(handle, generation=1)
    with pytest.raises(SpeechStateContractError, match="metadata"):
        store.consume(
            forged,
            SpeechStateContext(
                operation=SpeechStateOperation.PAUSE_RESUME,
                phase=SpeechStatePhase.RESTORE,
                session_id="s", segment_idx=1, attempt_id=0,
                slot_allocation_epoch=2, source_segment_idx=0,
                source_attempt_id=0, source_slot_allocation_epoch=1,
                source_generation=1,
                model_fingerprint="m", runtime_fingerprint="r",
            ),
        )
    assert len(store) == 1

    with pytest.raises(SpeechStateContractError, match="metadata"):
        store.release(forged)
    assert len(store) == 1

    store.release(handle)
    assert len(store) == 0


def test_segment_runtime_metadata_round_trip_is_detached():
    segment = EngineSegment("s", 3)
    segment.state = "active"
    segment.input_complete = True
    segment.pending_token_ids = [11, 12]
    segment.text_tokens_consumed = 8
    segment.loop_token = 11
    segment.loop_run = 2
    segment.retry_idx = 1
    segment.cursor_plan_revision = 6
    metadata = capture_segment_runtime_metadata(segment)
    segment.pending_token_ids.clear()
    segment.loop_run = 0
    restored = EngineSegment("s", 3)
    restore_segment_runtime_metadata(restored, metadata)
    assert restored.pending_token_ids == [11, 12]
    assert restored.loop_run == 2
    assert restored.cursor_plan_revision == 6
    assert metadata.pending_token_ids == (11, 12)


def test_segment_runtime_metadata_rejects_wrong_owner_without_mutation():
    segment = EngineSegment("s", 3)
    metadata = SegmentRuntimeMetadata(
        session_id="other", segment_idx=3, state="active",
        input_complete=False, trailing_idx=0, text_tokens_consumed=0,
        decode_start_frame=0, pending_token_ids=(), eos_trailing_added=False,
        first_raw_audio_sent=False, loop_token=-1, loop_run=0,
        loop_max_run=0, loop_suspect_count=0, loop_recovery_count=0,
        retry_idx=0, audio_frames_seen=0, audible_frame_seen=False,
        cache_hit=False, cache_tokens_reused=0, max_decode_batch=0,
        cursor_plan_revision=-1,
    )
    with pytest.raises(SpeechStateContractError, match="session mismatch"):
        restore_segment_runtime_metadata(segment, metadata)
    assert segment.state == "pending_prefill"


def test_successor_runtime_metadata_resets_source_segment_local_state():
    source = EngineSegment("s", 3)
    source.state = "active"
    source.input_complete = True
    source.trailing_idx = 8
    source.text_tokens_consumed = 21
    source.pending_token_ids = [11, 12]
    source.eos_trailing_added = True
    source.first_raw_audio_sent = True
    source.loop_token = 17
    source.loop_run = 4
    source.loop_max_run = 9
    source.loop_suspect_count = 2
    source.loop_recovery_count = 1
    source.retry_idx = 2
    source.audio_frames_seen = 33
    source.audible_frame_seen = True
    source.cache_hit = True
    source.cache_tokens_reused = 19
    source.max_decode_batch = 6
    source.cursor_plan_revision = 14

    source_metadata = capture_segment_runtime_metadata(source)
    successor = project_successor_runtime_metadata(
        source_metadata,
        target_segment_idx=4,
        pending_token_ids=(101, 102),
        input_complete=True,
        retry_idx=1,
    )

    assert successor.session_id == "s"
    assert successor.segment_idx == 4
    assert successor.state == "pending_prefill"
    assert successor.pending_token_ids == (101, 102)
    assert successor.text_tokens_consumed == 2
    assert successor.input_complete is True
    assert successor.retry_idx == 1
    assert successor.trailing_idx == 0
    assert successor.audio_frames_seen == 0
    assert successor.audible_frame_seen is False
    assert successor.cache_hit is False
    assert successor.max_decode_batch == 0
    assert successor.cursor_plan_revision == -1

    target = EngineSegment("s", 4)
    restore_segment_runtime_metadata(target, successor)
    assert target.pending_token_ids == [101, 102]
    assert target.text_tokens_consumed == 2
    assert target.state == "pending_prefill"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"target_segment_idx": 3}, "follow source"),
        ({"target_segment_idx": 4, "pending_token_ids": [101]}, "token ids"),
        ({"target_segment_idx": 4, "input_complete": 1}, "input_complete"),
    ],
)
def test_successor_runtime_metadata_rejects_ambiguous_projection(kwargs, message):
    source = capture_segment_runtime_metadata(EngineSegment("s", 3))
    with pytest.raises(SpeechStateContractError, match=message):
        project_successor_runtime_metadata(source, **kwargs)


def test_snapshot_bundle_validates_source_identity():
    segment = EngineSegment("s", 3)
    metadata = capture_segment_runtime_metadata(segment)
    bundle = SpeechStateSnapshotBundle(
        source_session_id="s", source_segment_idx=3, source_slot_id=2,
        source_allocation_epoch=1, segment_metadata=metadata,
        slot_payload=object(), pooled_talker_payload=None,
        pooled_c2w_payload=None, c2w_arena_payload=None,
    )
    assert bundle.source_slot_id == 2
    with pytest.raises(SpeechStateContractError, match="session metadata"):
        SpeechStateSnapshotBundle(
            source_session_id="other", source_segment_idx=3, source_slot_id=2,
            source_allocation_epoch=1, segment_metadata=metadata,
            slot_payload=None, pooled_talker_payload=None,
            pooled_c2w_payload=None, c2w_arena_payload=None,
        )


def test_engine_segment_owns_snapshot_bundle_slot():
    segment = EngineSegment("s", 1)
    assert segment.speech_state_bundle is None
    assert segment.speech_state_handle is None


def test_engine_loop_bundle_attach_is_owner_and_capability_gated():
    loop = object.__new__(__import__("engine.backend.engine_loop", fromlist=["EngineLoop"]).EngineLoop)
    loop._speech_state_capability = SpeechStateCapability(
        supported=True, handle_kind=SpeechStateHandleKind.OPAQUE,
        operations=(SpeechStateOperation.SEGMENT_HANDOFF,),
        transfer=SpeechStateTransfer.EXACT,
    )
    segment = EngineSegment("s", 3)
    metadata = capture_segment_runtime_metadata(segment)
    bundle = SpeechStateSnapshotBundle(
        source_session_id="s", source_segment_idx=3, source_slot_id=1,
        source_allocation_epoch=1, segment_metadata=metadata,
        slot_payload=None, pooled_talker_payload=None,
        pooled_c2w_payload=None, c2w_arena_payload=None,
    )
    loop.attach_speech_state_bundle(segment, bundle)
    assert loop.consume_speech_state_bundle(segment) is bundle
    with pytest.raises(SpeechStateContractError, match="no speech state"):
        loop.consume_speech_state_bundle(segment)
    loop.attach_speech_state_bundle(segment, bundle)
    with pytest.raises(RuntimeError):
        loop.restore_speech_state_bundle(segment, lambda _: (_ for _ in ()).throw(RuntimeError("restore failed")))
    assert segment.speech_state_bundle is bundle
    with pytest.raises(SpeechStateContractError, match="rejected"):
        loop.restore_speech_state_bundle(segment, lambda _: False)
    assert segment.speech_state_bundle is bundle
    loop.discard_speech_state_bundle(segment, reason="restore_failed_hard_boundary")
    with pytest.raises(SpeechStateContractError, match="no speech state"):
        loop.discard_speech_state_bundle(segment, reason="duplicate_cleanup")
    loop.attach_speech_state_bundle(segment, bundle)
    assert loop.restore_speech_state_bundle(segment, lambda item: item.source_slot_id) == 1
    assert segment.speech_state_bundle is None
    loop.attach_speech_state_bundle(segment, bundle)
    with pytest.raises(SpeechStateContractError, match="already owns"):
        loop.attach_speech_state_bundle(segment, bundle)


def test_engine_loop_migration_request_is_disabled_fail_closed():
    from engine.backend.engine_loop import EngineLoop

    loop = object.__new__(EngineLoop)
    loop._speech_state_capability = SpeechStateCapability.disabled()
    future = loop.request_speech_state_migration(
        "s", 0, expected_attempt_id=0, expected_allocation_epoch=1,
        max_tensor_bytes=1,
    )
    with pytest.raises(SpeechStateContractError, match="disabled"):
        future.result()


def test_engine_loop_migration_queue_only_exposes_completion():
    from engine.backend.state_transfer import MigrationRequest, SpeechStateMigrationQueue

    queue = SpeechStateMigrationQueue(max_pending=1)
    request = MigrationRequest("s", 2, 0, 7, 4096)
    future = queue.submit(request)
    observed = []
    queue.drain(lambda item: observed.append(item))
    assert future.result() is None
    assert observed == [request]


def test_engine_loop_drains_migration_after_completed_step_output():
    from engine.backend.engine_loop import EngineLoop
    from engine.backend.state_transfer import MigrationRequest, SpeechStateMigrationQueue

    events = []
    request = MigrationRequest("s", 0, 0, 1, 4096)
    migration_queue = SpeechStateMigrationQueue(max_pending=1)

    class _Future:
        def wait(self):
            events.append("future.wait")
            return SimpleNamespace()

    class _Executor:
        def launch_decode_step(self, _slots):
            events.append("decode.launch")
            return _Future()

    loop = object.__new__(EngineLoop)
    loop._running = True
    loop._executor = _Executor()
    loop._speech_state_migrations = migration_queue
    loop._max_batch = 1
    loop._total_steps = 0
    loop._mlfq = SimpleNamespace(tick=lambda _metas: events.append("mlfq.tick"))
    loop._all_active_mlfq_metas = lambda: []
    loop._try_prefill_pending = lambda: None
    loop._try_evict_idle_slots = lambda: None
    loop._try_timeout_sessions = lambda: None
    loop._maybe_emit_health = lambda: None
    loop._note_iteration_timing = lambda *_args: None
    loop._has_work = lambda: False
    loop._get_active_slots_mlfq = lambda: [object()] if "decode.launch" not in events else []
    loop._process_step_output = lambda _output: (
        events.append("step.process"), setattr(loop, "_running", False)
    )
    drain_count = 0
    submitted_future = None

    def drain_inbox():
        nonlocal drain_count, submitted_future
        drain_count += 1
        if drain_count == 1:
            events.append("migration.submit")
            submitted_future = migration_queue.submit(request)

    loop._drain_inbox = drain_inbox
    loop._migrate_speech_state = lambda _request: events.append("migration.execute")

    loop._run_inner()

    assert events.index("step.process") < events.index("migration.execute")
    assert events.index("future.wait") < events.index("migration.execute")
    assert submitted_future is not None
    assert submitted_future.result() is None


def test_engine_loop_migration_queue_cancellation_and_close_are_safe():
    from engine.backend.state_transfer import MigrationRequest, SpeechStateMigrationQueue

    queue = SpeechStateMigrationQueue(max_pending=1)
    cancelled = queue.submit(MigrationRequest("s", 0, 0, 1, 1))
    assert cancelled.cancel()
    queue.drain(lambda _: pytest.fail("cancelled migration must not execute"))
    pending = queue.submit(MigrationRequest("s", 0, 0, 1, 1))
    queue.close("test shutdown")
    with pytest.raises(SpeechStateContractError, match="shutdown"):
        pending.result()
