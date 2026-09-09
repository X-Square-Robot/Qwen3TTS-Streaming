"""Full pooled/arena speech-state transfer against the real fused TRT plan.

This is stronger than the cursor-only transfer test: the detached bundle
contains the Talker KV, Code2Wav state, slot auxiliaries, sampling generator,
and fused cursor recurrent state.  The source and restored target must produce
the same next fused decode result.

Run explicitly with::

    RUN_REAL_NATIVE_SPEECH_STATE_TESTS=1 pytest -q \
        tests/integration/test_real_native_cursor_full_state_transfer.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from engine.backend.engine_loop import EngineSegment
from engine.backend.executor import Executor
from engine.backend.kv_cache_pool import ModelConfig
from engine.backend.speech_state import capture_segment_runtime_metadata


def _artifact_dir() -> Path:
    if os.environ.get("RUN_REAL_NATIVE_SPEECH_STATE_TESTS", "").strip() != "1":
        pytest.skip("set RUN_REAL_NATIVE_SPEECH_STATE_TESTS=1 for full state evidence")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for full state evidence")
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pytest.skip("TensorRT is required for full state evidence")
    root = Path(
        os.environ.get(
            "QWEN_REAL_NATIVE_CURSOR_TRT_ARTIFACT_DIR",
            "/home/rime/workspace/models/x2-exported/custom-1.7b",
        )
    ).resolve()
    if not (root / "talker_code2wav_fused.engine").is_file():
        pytest.skip(f"no cursor-enabled fused TRT plan in {root}")
    if not (root / "weights" / "qwen3_tts_12hz_la1_seed0.pt").is_file():
        pytest.skip(f"no cursor head in {root}")
    return root


def _commit_decode_state(executor: Executor, slot, output) -> None:
    """Commit the engine-thread state boundary for a single direct slot."""
    pool = executor.kv_pool
    pool.scatter_talker_kv_delta(
        [slot.slot_id], output.batch_talker_kv, output.original_past_lens
    )
    if output.batch_c2w_kv is not None and not output.eos_flags[0]:
        if slot.c2w_pooled:
            pool.append_c2w_frames([slot.slot_id], output.batch_c2w_kv)
            slot.c2w_len = min(
                slot.c2w_len + 1,
                executor._config.c2w_sliding_window - 1,
            )
        else:
            slot.c2w_kv = output.batch_c2w_kv[:1].clone()
    conv = [tensor[:1] for tensor in (output.batch_c2w_conv or [])]
    transconv = [tensor[:1] for tensor in (output.batch_c2w_transconv or [])]
    if not output.eos_flags[0] and conv and slot.pingpong_ready:
        slot.copy_c2w_and_flip(conv, transconv)
    elif not output.eos_flags[0]:
        slot.c2w_conv_states = [tensor.clone() for tensor in conv]
        slot.c2w_transconv_states = [tensor.clone() for tensor in transconv]
    if output.updated_tc is not None:
        slot.token_counts = output.updated_tc[:1].clone()
    slot.past_len += 1
    if not output.eos_flags[0]:
        slot.frame_idx += 1
    executor.update_cursor_state(slot, output.cursor_outputs)
    if output.codec_sum is not None and not output.eos_flags[0]:
        slot.next_embed = output.codec_sum[:1].clone()
        slot.last_codec_sum = None


def _assert_next_outputs_equal(left, right) -> None:
    assert left.tokens == right.tokens
    assert left.audio_chunks == right.audio_chunks
    for name in ("codec_sum", "hidden", "updated_tc", "batch_c2w_kv"):
        lhs = getattr(left, name)
        rhs = getattr(right, name)
        assert lhs is not None and rhs is not None, name
        assert torch.equal(lhs, rhs), name
    assert left.eos_flags == right.eos_flags
    assert set(left.cursor_outputs) == set(right.cursor_outputs)
    for name in left.cursor_outputs:
        lhs = left.cursor_outputs[name]
        rhs = right.cursor_outputs[name]
        assert lhs is not None and rhs is not None, name
        assert torch.equal(lhs, rhs), name


@pytest.mark.parametrize("use_graph", [False, True])
def test_real_full_speech_state_restore_matches_next_fused_decode(
    monkeypatch, use_graph
) -> None:
    root = _artifact_dir()
    monkeypatch.setenv("ENGINE_CUDA_GRAPH_DECODE", "1" if use_graph else "0")
    if use_graph:
        monkeypatch.setenv("ENGINE_CUDA_GRAPH_MAX_PAST", "128")
    executor = Executor(
        engine_dir=str(root),
        weights_dir=str(root / "weights"),
        max_batch_size=2,
        max_seq_len=512,
        model_config=ModelConfig(dtype=torch.bfloat16),
    )
    executor.load()
    assert executor.native_cursor_enabled is True
    assert executor._cursor_state_handoff_enabled is True
    if use_graph:
        assert executor._graph_decode is not None

    pool = executor.kv_pool
    source = pool.allocate("full-state")
    target = pool.allocate("full-state")
    assert source is not None and target is not None
    source.segment_idx = target.segment_idx = 0
    source.retry_idx = target.retry_idx = 0
    labels = torch.arange(1, 33, dtype=torch.int64)

    try:
        executor.set_cursor_text_plan(source, labels, active=True)
        embeds = torch.zeros(
            (1, 1, executor._config.hidden_size),
            device="cuda",
            dtype=executor._config.dtype,
        )
        executor.prefill(source, embeds)

        for _ in range(3):
            output = executor.launch_decode_step([source]).wait()
            _commit_decode_state(executor, source, output)

        segment = EngineSegment("full-state", 0)
        segment.state = "active"
        segment.slot = source
        segment.input_complete = True
        segment.pending_token_ids = list(range(1, 33))
        segment.text_tokens_consumed = len(segment.pending_token_ids)
        metadata = capture_segment_runtime_metadata(segment)

        executor.synchronize_state_transfer()
        bundle = executor.capture_speech_state_bundle(
            source,
            metadata,
            expected_slot_session_id="full-state",
            max_tensor_bytes=2_000_000_000,
        )
        executor.restore_speech_state_bundle(
            target,
            bundle,
            expected_allocation_epoch=target.allocation_epoch,
        )
        executor.synchronize_state_transfer()

        assert target.past_len == source.past_len
        assert target.c2w_len == source.c2w_len
        assert target.frame_idx == source.frame_idx
        assert target.cursor_seen_frames is not None
        assert torch.equal(target.cursor_seen_frames, source.cursor_seen_frames)

        source_next = executor.launch_decode_step([source]).wait()
        target_next = executor.launch_decode_step([target]).wait()
        _assert_next_outputs_equal(source_next, target_next)
    finally:
        if source is not None and not source.is_free:
            pool.release(source.slot_id, expected_allocation_epoch=source.allocation_epoch)
        if target is not None and not target.is_free:
            pool.release(target.slot_id, expected_allocation_epoch=target.allocation_epoch)
        torch.cuda.synchronize()
