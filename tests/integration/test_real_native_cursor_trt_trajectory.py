"""Frozen-input PyTorch/reference versus real TensorRT cursor trajectory.

The Talker graph supplies the actual sampled ``codec0`` token.  The test then
feeds that same token and the pre-step cursor state to the released PyTorch
cursor head, comparing it with the cursor outputs fused into the real TRT
graph.  Integer routing outputs must match exactly; floating-point outputs use
an explicit bf16 error budget.

Run explicitly with::

    RUN_REAL_NATIVE_CURSOR_TRAJECTORY_TESTS=1 pytest -q \
        tests/integration/test_real_native_cursor_trt_trajectory.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from engine.backend.executor import Executor
from engine.backend.kv_cache_pool import ModelConfig
from scripts.export.native_cursor_modules import (
    CursorStreamingStep,
    build_cursor_head_from_checkpoint,
)


_STATE_FIELDS = (
    "cursor_mu",
    "cursor_frames_since_advance",
    "cursor_delta_history",
    "cursor_conv_history",
    "cursor_last_trunk_input",
    "cursor_seen_frames",
)
_OUTPUT_NAMES = (
    "cursor_valid",
    "cursor_mu",
    "cursor_delta",
    "cursor_confidence",
    "cursor_candidate_label",
    "cursor_frames_since_advance",
    "cursor_delta_history",
    "cursor_conv_history",
    "cursor_last_trunk_input",
    "cursor_seen_frames",
)


def _artifact_dir() -> Path:
    if os.environ.get("RUN_REAL_NATIVE_CURSOR_TRAJECTORY_TESTS", "").strip() != "1":
        pytest.skip("set RUN_REAL_NATIVE_CURSOR_TRAJECTORY_TESTS=1 for real trajectory evidence")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real cursor trajectory evidence")
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pytest.skip("TensorRT is required for real cursor trajectory evidence")
    root = Path(
        os.environ.get(
            "QWEN_REAL_NATIVE_CURSOR_TRT_ARTIFACT_DIR",
            "/home/rime/workspace/models/x2-exported/custom-1.7b",
        )
    ).resolve()
    engine_path = root / "talker_code2wav_fused.engine"
    head_path = root / "weights" / "qwen3_tts_12hz_la1_seed0.pt"
    if not engine_path.is_file() or not head_path.is_file():
        pytest.skip(f"real cursor TRT engine/head are missing in {root}")
    return root


def _reference_args(slot, codec0: int, before: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    return [
        torch.tensor([codec0], dtype=torch.long),
        slot.cursor_label_ids.detach().cpu().clone(),
        slot.cursor_label_count.detach().cpu().clone(),
        slot.cursor_active.detach().cpu().clone(),
        before["cursor_mu"].float(),
        before["cursor_frames_since_advance"].float(),
        before["cursor_delta_history"].float(),
        before["cursor_conv_history"].float(),
        before["cursor_last_trunk_input"].float(),
        before["cursor_seen_frames"].detach().cpu().clone(),
        slot.cursor_text_start_frame.detach().cpu().clone(),
        slot.cursor_override_valid.detach().cpu().clone(),
        slot.cursor_override_mu.detach().cpu().clone(),
    ]


def test_real_cursor_trt_matches_pytorch_reference_for_frozen_codec0_trajectory() -> None:
    root = _artifact_dir()
    head, _metadata = build_cursor_head_from_checkpoint(
        root / "weights" / "qwen3_tts_12hz_la1_seed0.pt"
    )
    head.eval()
    reference = CursorStreamingStep(head).eval()

    executor = Executor(
        engine_dir=str(root),
        weights_dir=str(root / "weights"),
        max_batch_size=1,
        max_seq_len=512,
        model_config=ModelConfig(dtype=torch.bfloat16),
    )
    executor.load()
    assert executor.native_cursor_enabled is True
    assert executor._cursor_state_handoff_enabled is True

    pool = executor.kv_pool
    slot = pool.allocate("cursor-trajectory")
    assert slot is not None
    slot.segment_idx = 0
    try:
        executor.set_cursor_text_plan(
            slot,
            torch.arange(1, 33, dtype=torch.int64),
            active=True,
        )
        embeds = torch.zeros(
            (1, 1, executor._config.hidden_size),
            device="cuda",
            dtype=executor._config.dtype,
        )
        executor.prefill(slot, embeds)

        for frame_idx in range(32):
            before = {
                name: getattr(slot, name).detach().cpu().clone()
                for name in _STATE_FIELDS
            }
            output = executor.launch_decode_step([slot]).wait()
            codec0 = output.tokens[0]
            assert isinstance(codec0, int)
            with torch.no_grad():
                expected = reference(*_reference_args(slot, codec0, before))

            for name, expected_value in zip(_OUTPUT_NAMES, expected):
                actual_value = output.cursor_outputs[name][0].detach().cpu()
                if actual_value.dtype in (torch.int64, torch.int32):
                    assert torch.equal(
                        actual_value.reshape(-1), expected_value.reshape(-1)
                    ), name
                else:
                    # The real plan is bf16; this budget is deliberately
                    # explicit and is not used as a release-quality claim.
                    try:
                        torch.testing.assert_close(
                            actual_value.float().reshape(-1),
                            expected_value.float().reshape(-1),
                            rtol=0.05,
                            atol=0.08,
                        )
                    except AssertionError as exc:
                        max_error = (
                            actual_value.float() - expected_value.float()
                        ).abs().max().item()
                        raise AssertionError(
                            f"{name} frame={frame_idx} max_error={max_error}: {exc}"
                        ) from exc
            executor.update_cursor_state(slot, output.cursor_outputs)
    finally:
        if not slot.is_free:
            pool.release(slot.slot_id, expected_allocation_epoch=slot.allocation_epoch)
        torch.cuda.synchronize()
