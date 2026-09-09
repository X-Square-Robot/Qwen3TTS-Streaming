"""Black-box recurrent-state transfer evidence for a real cursor TRT plan.

This is narrower than the X2 successor E2E: both slots keep the same Talker
and Code2Wav state, while the test perturbs and restores only the fused
cursor recurrent state.  It proves that cursor state handoff is numerically
observable in the real graph without claiming the full speech-state release
gate.

Run explicitly with::

    RUN_REAL_NATIVE_CURSOR_STATE_TESTS=1 pytest -q \
        tests/integration/test_real_native_cursor_state_transfer.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from engine.backend.executor import Executor
from engine.backend.kv_cache_pool import ModelConfig
from engine.core.extensions import capture_cursor_state


_CURSOR_FIELDS = (
    "cursor_mu",
    "cursor_frames_since_advance",
    "cursor_delta_history",
    "cursor_conv_history",
    "cursor_last_trunk_input",
    "cursor_seen_frames",
)


def _artifact_dir() -> Path:
    if os.environ.get("RUN_REAL_NATIVE_CURSOR_STATE_TESTS", "").strip() != "1":
        pytest.skip("set RUN_REAL_NATIVE_CURSOR_STATE_TESTS=1 for real cursor state evidence")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real cursor state evidence")
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pytest.skip("TensorRT is required for real cursor state evidence")
    root = Path(
        os.environ.get(
            "QWEN_REAL_NATIVE_CURSOR_TRT_ARTIFACT_DIR",
            "/home/rime/workspace/models/x2-exported/custom-1.7b",
        )
    ).resolve()
    if not (root / "talker_code2wav_fused.engine").is_file():
        pytest.skip(f"no cursor-enabled fused TRT plan in {root}")
    return root


def _assert_tensor_dict_rows_equal(left, right) -> None:
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_real_cursor_recurrent_state_restore_matches_next_graph_step(monkeypatch) -> None:
    root = _artifact_dir()
    monkeypatch.setenv("ENGINE_CUDA_GRAPH_DECODE", "0")
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

    pool = executor.kv_pool
    source = pool.allocate("cursor-state")
    target = pool.allocate("cursor-state")
    assert source is not None and target is not None
    source.segment_idx = target.segment_idx = 0
    labels = torch.tensor([1, 2, 3], dtype=torch.int64)
    try:
        executor.set_cursor_text_plan(source, labels, active=True)
        executor.set_cursor_text_plan(target, labels, active=True)
        embeds = torch.zeros(
            (1, 1, executor._config.hidden_size),
            device="cuda",
            dtype=executor._config.dtype,
        )
        executor.prefill(source, embeds)
        executor.prefill(target, embeds)

        # Advance both slots once so the captured state is non-zero and comes
        # from actual fused cursor outputs rather than initialization.
        first = executor.launch_decode_step([source, target]).wait()
        executor.update_cursor_state(source, first.cursor_outputs, row=0)
        executor.update_cursor_state(target, first.cursor_outputs, row=1)
        captured = capture_cursor_state(source)
        assert captured is not None
        assert int(source.cursor_seen_frames.item()) > 0

        # Make the target observably different, then restore only the detached
        # recurrent cursor state.  Labels and all other graph state stay fixed.
        for name in _CURSOR_FIELDS:
            getattr(target, name).zero_()
        executor.restore_cursor_state(target, captured)
        for name in _CURSOR_FIELDS:
            assert torch.equal(getattr(source, name), getattr(target, name)), name

        second = executor.launch_decode_step([source, target]).wait()
        assert second.tokens[0] == second.tokens[1]
        assert second.audio_chunks[0] == second.audio_chunks[1]
        for name, value in second.cursor_outputs.items():
            assert value is not None, name
            assert torch.equal(value[0], value[1]), name
    finally:
        if source is not None and not source.is_free:
            pool.release(source.slot_id, expected_allocation_epoch=source.allocation_epoch)
        if target is not None and not target.is_free:
            pool.release(target.slot_id, expected_allocation_epoch=target.allocation_epoch)
        torch.cuda.synchronize()
