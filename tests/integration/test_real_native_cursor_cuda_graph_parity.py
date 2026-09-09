"""Compare CUDA-Graph and eager decode for the real cursor-enabled TRT plan.

The test uses one loaded Executor and fresh slots for each route so the graph
and eager paths share the exact engine weights, prefill inputs and label plan.
It does not claim full PyTorch/ONNX/TRT parity; it isolates the runtime route
switch required by T4.

Run explicitly with::

    RUN_REAL_NATIVE_CURSOR_GRAPH_TESTS=1 pytest -q \
        tests/integration/test_real_native_cursor_cuda_graph_parity.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from engine.backend.executor import Executor
from engine.backend.kv_cache_pool import ModelConfig


def _artifact_dir() -> Path:
    if os.environ.get("RUN_REAL_NATIVE_CURSOR_GRAPH_TESTS", "").strip() != "1":
        pytest.skip("set RUN_REAL_NATIVE_CURSOR_GRAPH_TESTS=1 for CUDA Graph parity")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA Graph parity")
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pytest.skip("TensorRT is required for CUDA Graph parity")
    root = Path(
        os.environ.get(
            "QWEN_REAL_NATIVE_CURSOR_TRT_ARTIFACT_DIR",
            "/home/rime/workspace/models/x2-exported/custom-1.7b",
        )
    ).resolve()
    if not (root / "talker_code2wav_fused.engine").is_file():
        pytest.skip(f"no cursor-enabled TRT plan in {root}")
    if not (root / "weights" / "qwen3_tts_12hz_la1_seed0.pt").is_file():
        pytest.skip(f"no cursor head in {root}")
    return root


def _snapshot(output) -> dict[str, object]:
    return {
        "tokens": tuple(output.tokens),
        "audio": tuple(output.audio_chunks),
        "codec_sum": output.codec_sum.detach().cpu().clone(),
        "hidden": output.hidden.detach().cpu().clone(),
        "updated_tc": output.updated_tc.detach().cpu().clone(),
        "c2w_kv": output.batch_c2w_kv.detach().cpu().clone(),
        "cursor": {
            name: None if value is None else value.detach().cpu().clone()
            for name, value in output.cursor_outputs.items()
        },
    }


def _assert_snapshot_equal(left: dict[str, object], right: dict[str, object]) -> None:
    assert left["tokens"] == right["tokens"]
    for name in ("codec_sum", "hidden", "updated_tc", "c2w_kv"):
        assert torch.equal(left[name], right[name]), name
    left_audio = left["audio"]
    right_audio = right["audio"]
    assert len(left_audio) == len(right_audio)
    for index, (lhs_bytes, rhs_bytes) in enumerate(zip(left_audio, right_audio)):
        assert (lhs_bytes is None) == (rhs_bytes is None), index
        if lhs_bytes is None:
            continue
        lhs = np.frombuffer(lhs_bytes, dtype=np.float32)
        rhs = np.frombuffer(rhs_bytes, dtype=np.float32)
        assert lhs.shape == rhs.shape
        max_error = float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0
        rms_error = (
            float(np.sqrt(np.mean(np.square(lhs - rhs)))) if lhs.size else 0.0
        )
        # Graph and eager execute the same TRT topology with different staging
        # addresses; PCM is therefore validated numerically, not bytewise.
        np.testing.assert_allclose(
            lhs,
            rhs,
            rtol=0.02,
            atol=2e-3,
            err_msg=(
                f"audio chunk {index} max_error={max_error:.6g} "
                f"rms_error={rms_error:.6g}"
            ),
        )
    left_cursor = left["cursor"]
    right_cursor = right["cursor"]
    assert set(left_cursor) == set(right_cursor)
    for name in left_cursor:
        lhs = left_cursor[name]
        rhs = right_cursor[name]
        assert (lhs is None) == (rhs is None), name
        if lhs is not None:
            assert torch.equal(lhs, rhs), name


def _commit_decode_state(executor: Executor, slots, output) -> None:
    """Apply the engine-thread state boundary needed by the next decode."""
    pool = executor.kv_pool
    assert output.batch_talker_kv is not None
    pool.scatter_talker_kv_delta(
        [slot.slot_id for slot in slots],
        output.batch_talker_kv,
        output.original_past_lens,
    )
    for row, slot in enumerate(slots):
        if output.batch_c2w_kv is not None and not output.eos_flags[row]:
            if slot.c2w_pooled:
                pool.append_c2w_frames(
                    [slot.slot_id], output.batch_c2w_kv[row : row + 1]
                )
                slot.c2w_len = min(
                    slot.c2w_len + 1,
                    executor._config.c2w_sliding_window - 1,
                )
            else:
                slot.c2w_kv = output.batch_c2w_kv[row : row + 1].clone()
        conv = [
            tensor[row : row + 1] for tensor in (output.batch_c2w_conv or [])
        ]
        transconv = [
            tensor[row : row + 1]
            for tensor in (output.batch_c2w_transconv or [])
        ]
        if not output.eos_flags[row] and conv and slot.pingpong_ready:
            slot.copy_c2w_and_flip(conv, transconv)
        elif not output.eos_flags[row]:
            slot.c2w_conv_states = [tensor.clone() for tensor in conv]
            slot.c2w_transconv_states = [tensor.clone() for tensor in transconv]
        if output.updated_tc is not None:
            slot.token_counts = output.updated_tc[row : row + 1].clone()
        slot.past_len += 1
        if not output.eos_flags[row]:
            slot.frame_idx += 1
        executor.update_cursor_state(slot, output.cursor_outputs, row=row)
        if output.codec_sum is not None:
            slot.next_embed = output.codec_sum[row : row + 1].clone()
            slot.last_codec_sum = None


def _run_route(
    executor: Executor, *, use_graph: bool, batch_size: int
) -> tuple[list[dict[str, object]], dict[str, torch.Tensor]]:
    pool = executor.kv_pool
    slots = [pool.allocate("graph-parity") for _ in range(batch_size)]
    assert all(slot is not None for slot in slots)
    slots = [slot for slot in slots if slot is not None]
    for slot in slots:
        slot.segment_idx = 0
    try:
        for slot in slots:
            executor.set_cursor_text_plan(
                slot,
                torch.arange(1, 33, dtype=torch.int64),
                active=True,
            )
        embeds = torch.zeros(
            (batch_size, 1, executor._config.hidden_size),
            device="cuda",
            dtype=executor._config.dtype,
        )
        for row, slot in enumerate(slots):
            executor.prefill(slot, embeds[row : row + 1])
        if not use_graph:
            # Compare graph and eager decode under the same decode-only TRT
            # profile.  Comparing cursor profile 1 against the shared
            # prefill/profile-0 context confounds normal BF16 cross-profile
            # sampling variation with a graph parity failure.
            executor._fused_engine.select_optimization_profile(
                1, executor._compute_stream
            )
        if use_graph:
            assert executor._graph_decode is not None
        outputs: list[dict[str, object]] = []
        first_inputs: dict[str, torch.Tensor] = {}
        for _ in range(4):
            future = executor.launch_decode_step(slots)
            if not outputs:
                first_inputs = {
                    name: value.detach().cpu().clone()
                    for name, value in future._input_refs.items()
                }
            output = future.wait()
            outputs.append(_snapshot(output))
            _commit_decode_state(executor, slots, output)
        return outputs, first_inputs
    finally:
        for slot in slots:
            if not slot.is_free:
                pool.release(slot.slot_id, expected_allocation_epoch=slot.allocation_epoch)
        torch.cuda.synchronize()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_real_cursor_cuda_graph_decode_matches_eager(monkeypatch, batch_size) -> None:
    root = _artifact_dir()
    monkeypatch.setenv("ENGINE_CUDA_GRAPH_DECODE", "1")
    executor = Executor(
        engine_dir=str(root),
        weights_dir=str(root / "weights"),
        max_batch_size=2,
        max_seq_len=512,
        model_config=ModelConfig(dtype=torch.bfloat16),
    )
    executor.load()
    assert executor.native_cursor_enabled is True
    assert executor._graph_decode is not None

    graph_outputs, graph_inputs = _run_route(
        executor, use_graph=True, batch_size=batch_size
    )

    # Drop graph-only staging before the eager route.  The engine context and
    # weights stay shared; only the decode implementation is changed.
    executor._graph_decode = None
    executor._graph_decode_enabled = False
    executor._talker_gather_flat = None
    torch.cuda.empty_cache()
    eager_outputs, eager_inputs = _run_route(
        executor, use_graph=False, batch_size=batch_size
    )

    assert set(graph_inputs) == set(eager_inputs)
    input_shape_mismatches = []
    for name in graph_inputs:
        if tuple(graph_inputs[name].shape) != tuple(eager_inputs[name].shape):
            input_shape_mismatches.append(
                f"{name}: shape {tuple(graph_inputs[name].shape)} != "
                f"{tuple(eager_inputs[name].shape)}"
            )
        else:
            assert torch.equal(graph_inputs[name], eager_inputs[name]), name
    assert input_shape_mismatches

    assert len(graph_outputs) == len(eager_outputs)
    for graph, eager in zip(graph_outputs, eager_outputs):
        _assert_snapshot_equal(graph, eager)
