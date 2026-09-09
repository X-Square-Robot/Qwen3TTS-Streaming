from types import SimpleNamespace

import pytest
import torch

from engine.backend.engine_loop import EngineSessionGroup
from engine.backend.engine_loop import EngineLoop, EngineSegment
from engine.backend.kv_cache_pool import KVCachePool, ModelConfig
from engine.core.extensions import (
    CursorContextOverlay,
    CursorContinuationState,
    EngineExtensions,
    capture_c2w_state,
    invoke_policy,
    overlay_cursor_state,
)
from engine.core.types import EngineRequest, RequestType, SessionConfig


def test_extension_factories_are_session_scoped():
    seen = []
    policy = object()
    extensions = EngineExtensions(
        name="x2",
        version="test",
        continuity_factory=lambda session_id, config: seen.append(
            (session_id, config)
        )
        or policy,
    )
    request = EngineRequest(
        type=RequestType.NEW_SESSION,
        session_id="session-1",
        session_config=SessionConfig(),
    )

    group = EngineSessionGroup(
        "session-1", request, extensions=extensions
    )

    assert group.extension_continuity is policy
    assert seen == [("session-1", request.session_config)]


def test_continuity_factory_failure_fails_closed_per_session():
    request = EngineRequest(
        type=RequestType.NEW_SESSION,
        session_id="session-broken-extension",
        session_config=SessionConfig(),
    )

    def broken_factory(_session_id, _config):
        raise RuntimeError("optional method package is unavailable")

    group = EngineSessionGroup(
        request.session_id,
        request,
        extensions=EngineExtensions(continuity_factory=broken_factory),
    )

    assert group.extension_continuity is None


def test_invoke_policy_invalidates_after_callback_failure():
    calls = []

    class Policy:
        def apply(self):
            raise RuntimeError("broken")

        def invalidate(self, reason):
            calls.append(reason)

    assert invoke_policy(Policy(), "apply", default="fallback") == "fallback"
    assert calls == ["callback_failed:apply"]


def test_invoke_policy_reports_failure_to_engine_owner():
    failures = []

    class Policy:
        def apply(self):
            raise RuntimeError("broken")

    assert invoke_policy(
        Policy(),
        "apply",
        default="fallback",
        on_failure=failures.append,
    ) == "fallback"
    assert len(failures) == 1
    assert str(failures[0]) == "broken"


def test_capture_c2w_state_is_detached_from_nonpooled_slot():
    slot = SimpleNamespace(
        c2w_kv=torch.ones(1, 2, 1, 2, 2),
        c2w_pooled=False,
        c2w_conv_states=[torch.ones(1, 1, 2)],
        c2w_transconv_states=[torch.ones(1, 1, 1)],
        frame_idx=7,
    )

    state = capture_c2w_state(slot)

    assert state is not None
    slot.c2w_kv.zero_()
    slot.c2w_conv_states[0].zero_()
    assert torch.all(state.c2w_kv == 1)
    assert torch.all(state.c2w_conv_states[0] == 1)
    # External X2 policy uses its historical short names; the engine-owned
    # state keeps the explicit *_states names and exposes both contracts.
    assert state.c2w_conv == state.c2w_conv_states
    assert state.c2w_transconv == state.c2w_transconv_states
    assert state.frame_idx == 7
    assert state.cursor_state is None


def test_capture_c2w_state_carries_only_cursor_recurrent_state():
    slot = SimpleNamespace(
        c2w_kv=torch.ones(1, 2, 1, 2, 2),
        c2w_pooled=False,
        c2w_conv_states=[torch.ones(1, 1, 2)],
        c2w_transconv_states=[torch.ones(1, 1, 1)],
        frame_idx=7,
        cursor_mu=torch.ones(1),
        cursor_frames_since_advance=torch.ones(1),
        cursor_delta_history=torch.ones(1, 8),
        cursor_conv_history=torch.ones(1, 6, 4),
        cursor_last_trunk_input=torch.ones(1, 4),
        cursor_seen_frames=torch.ones(1, dtype=torch.int64),
        cursor_label_ids=torch.full((1, 8), 9, dtype=torch.int64),
    )

    state = capture_c2w_state(slot)

    assert state is not None
    assert isinstance(state.cursor_state, CursorContinuationState)
    assert not hasattr(state.cursor_state, "cursor_label_ids")
    slot.cursor_mu.zero_()
    slot.cursor_label_ids.zero_()
    assert torch.all(state.cursor_state.cursor_mu == 1)


def test_cursor_continuation_state_restores_detached_neural_fields_only():
    source = SimpleNamespace(
        cursor_mu=torch.ones(1),
        cursor_frames_since_advance=torch.ones(1),
        cursor_delta_history=torch.ones(1, 8),
        cursor_conv_history=torch.ones(1, 6, 4),
        cursor_last_trunk_input=torch.ones(1, 4),
        cursor_seen_frames=torch.ones(1, dtype=torch.int64),
    )
    state = CursorContinuationState(
        **{
            name: getattr(source, name).clone()
            for name in (
                "cursor_mu",
                "cursor_frames_since_advance",
                "cursor_delta_history",
                "cursor_conv_history",
                "cursor_last_trunk_input",
                "cursor_seen_frames",
            )
        }
    )
    target = SimpleNamespace(
        **{
            name: getattr(source, name).clone().zero_()
            for name in (
                "cursor_mu",
                "cursor_frames_since_advance",
                "cursor_delta_history",
                "cursor_conv_history",
                "cursor_last_trunk_input",
                "cursor_seen_frames",
            )
        }
    )

    state.restore_into(target)

    assert torch.all(target.cursor_mu == 1)
    assert torch.all(target.cursor_conv_history == 1)


def test_cursor_continuation_restore_is_atomic_on_late_abi_failure():
    names = (
        "cursor_mu",
        "cursor_frames_since_advance",
        "cursor_delta_history",
        "cursor_conv_history",
        "cursor_last_trunk_input",
        "cursor_seen_frames",
    )
    state = CursorContinuationState(
        cursor_mu=torch.ones(1),
        cursor_frames_since_advance=torch.ones(1),
        cursor_delta_history=torch.ones(1, 8),
        cursor_conv_history=torch.ones(1, 6, 4),
        cursor_last_trunk_input=torch.ones(1, 4),
        cursor_seen_frames=torch.ones(1, dtype=torch.int64),
    )
    target = SimpleNamespace(
        cursor_mu=torch.full((1,), 7.0),
        cursor_frames_since_advance=torch.full((1,), 7.0),
        cursor_delta_history=torch.full((1, 8), 7.0),
        cursor_conv_history=torch.full((1, 6, 4), 7.0),
        cursor_last_trunk_input=torch.full((1, 4), 7.0),
        # The last field fails only after all earlier fields have been checked.
        cursor_seen_frames=torch.zeros(2, dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="cursor_seen_frames"):
        state.restore_into(target)

    for name in names[:-1]:
        assert torch.all(getattr(target, name) == 7)
    assert torch.equal(target.cursor_seen_frames, torch.zeros(2, dtype=torch.int64))


def test_cursor_context_overlay_does_not_mutate_external_context():
    context = SimpleNamespace(c2w_kv="c2w", c2w_frame_idx=3)
    state = CursorContinuationState(
        cursor_mu=torch.zeros(1),
        cursor_frames_since_advance=torch.zeros(1),
        cursor_delta_history=torch.zeros(1, 8),
        cursor_conv_history=torch.zeros(1, 6, 4),
        cursor_last_trunk_input=torch.zeros(1, 4),
        cursor_seen_frames=torch.zeros(1, dtype=torch.int64),
    )

    overlay = overlay_cursor_state(context, state)

    assert isinstance(overlay, CursorContextOverlay)
    assert overlay.c2w_kv == "c2w"
    assert overlay.c2w_frame_idx == 3
    assert overlay.cursor_state is state
    assert not hasattr(context, "cursor_state")

    stale = SimpleNamespace(c2w_kv="c2w", cursor_state=object())
    refreshed = overlay_cursor_state(stale, state)
    assert refreshed.cursor_state is state


def test_capture_c2w_state_reads_pooled_right_aligned_history():
    config = ModelConfig(
        dtype=torch.float32,
        num_layers=1,
        kv_heads=1,
        head_dim=2,
        n_c2w_layers=1,
        c2w_kv_heads=1,
        c2w_head_dim=2,
        c2w_sliding_window=4,
    )
    pool = KVCachePool(1, config, torch.device("cpu"), preallocate=True)
    slot = pool.allocate("session")
    slot.c2w_pooled = True
    slot.c2w_len = pool.write_c2w_right_aligned(
        slot.slot_id, torch.full((1, 2, 1, 2, 2), 3.0)
    )
    slot.c2w_conv_states = [torch.ones(1, 1, 2)]
    slot.c2w_transconv_states = [torch.ones(1, 1, 1)]

    state = capture_c2w_state(slot, pool)

    assert state is not None
    assert state.c2w_kv.shape[3] == 2
    assert torch.all(state.c2w_kv == 3)


def test_continuity_consumed_text_requires_a_complete_slot_frontier():
    seg = EngineSegment("session", 0)
    seg.pending_token_ids = [11, 12, 13]

    assert EngineLoop._continuity_consumed_text_tokens(seg) == 0

    seg.slot = SimpleNamespace(text_idx=2)
    assert EngineLoop._continuity_consumed_text_tokens(seg) == 0

    seg.slot.text_idx = 3
    assert EngineLoop._continuity_consumed_text_tokens(seg) == 3


def test_failed_admission_policy_falls_back_to_hard_boundary():
    class Policy:
        def ready_to_admit(self, _segment_idx):
            raise RuntimeError("policy unavailable")

        def invalidate(self, _reason):
            return None

    loop = object.__new__(EngineLoop)
    group = SimpleNamespace(
        session_id="session",
        extension_continuity=Policy(),
        extension_continuity_disabled=False,
    )
    seg = SimpleNamespace(segment_idx=1, pending_token_ids=[1], input_complete=False)

    assert EngineLoop._extension_holds_admission(loop, group, seg) is False
    assert group.extension_continuity_disabled is True
