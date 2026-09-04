from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from engine.backend.engine_loop import EngineSegment, EngineSessionGroup
from engine.backend.speech_state import (
    NullSpeechStateAdapter,
    capability_from_adapter,
    coerce_speech_state_adapter,
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

    assert engine.describe_capabilities()["speech_state"] == (
        SpeechStateCapability.disabled().to_dict()
    )
    assert engine.health_stats()["speech_state"] == (
        SpeechStateCapability.disabled().to_dict()
    )


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
