"""Black-box state-transfer evidence against a real standard TRT plan.

This test deliberately exercises the backend snapshot/restore primitives with
the real fused Talker/Code2Wav engine.  It does not advertise speech-state
support and it is not an X2 successor-continuity test: the package has no
cursor bindings or speech-state manifest.  The evidence is limited to the
claim that an identical detached slot state produces an identical next TRT
step after restore.

Run explicitly with::

    RUN_REAL_TRT_STATE_TRANSFER_TESTS=1 \
    QWEN_REAL_TRT_ARTIFACT_DIR=workspace/exported/custom-1.7b \
    pytest -q tests/integration/test_real_trt_state_transfer_contract.py
"""

from __future__ import annotations

import asyncio
import os
import queue
from pathlib import Path

import pytest
import torch

from engine.backend.engine_loop import EngineLoop, EngineSegment
from engine.backend.executor import Executor, StepOutput
from engine.backend.kv_cache_pool import ModelConfig
from engine.backend.speech_state import (
    SpeechStateContractError,
    capture_segment_runtime_metadata,
)
from engine.config import ModelPackagePaths, load_model_manifest
from engine.gateway.capabilities import RuntimeType, build_gateway_capabilities
from engine.gateway.triton_realtime_server import _runtime_capabilities_from_package
from engine.server import TTSEngine


def _artifact_dir() -> Path:
    if os.environ.get("RUN_REAL_TRT_STATE_TRANSFER_TESTS", "").strip() != "1":
        pytest.skip("set RUN_REAL_TRT_STATE_TRANSFER_TESTS=1 for real TRT evidence")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for a real TRT state-transfer check")
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pytest.skip("TensorRT is required for a real TRT state-transfer check")
    value = os.environ.get("QWEN_REAL_TRT_ARTIFACT_DIR", "").strip()
    if not value:
        pytest.skip("set QWEN_REAL_TRT_ARTIFACT_DIR to an exported variant directory")
    root = Path(value)
    if not root.is_dir():
        pytest.skip(f"TRT artifact directory does not exist: {root}")
    if not (root / "talker_code2wav_fused.engine").is_file() and not (
        root / "model.plan"
    ).is_file():
        pytest.skip(f"no fused TRT plan in {root}")
    return root.resolve()


def _cpu_snapshot(output: StepOutput) -> dict[str, object]:
    def clone(value):
        return None if value is None else value.detach().cpu().clone()

    return {
        "tokens": tuple(output.tokens),
        "audio_chunks": tuple(output.audio_chunks),
        "codec_sum": clone(output.codec_sum),
        "updated_tc": clone(output.updated_tc),
        "c2w_conv": tuple(clone(value) for value in (output.batch_c2w_conv or ())),
        "c2w_transconv": tuple(
            clone(value) for value in (output.batch_c2w_transconv or ())
        ),
    }


def _assert_equal_tensor_group(left, right) -> None:
    assert len(left) == len(right)
    for lhs, rhs in zip(left, right):
        assert (lhs is None) == (rhs is None)
        if lhs is not None:
            assert torch.equal(lhs, rhs)


def test_real_standard_trt_next_step_is_identical_after_bundle_restore(
    monkeypatch,
):
    root = _artifact_dir()
    # CUDA graph staging is a separate shape/address path.  This contract is
    # intentionally about the ordinary executor stream and detached state.
    monkeypatch.setenv("ENGINE_CUDA_GRAPH_DECODE", "0")
    config = ModelConfig(dtype=torch.bfloat16)
    executor = Executor(
        engine_dir=str(root),
        weights_dir=str(root / "weights"),
        max_batch_size=2,
        max_seq_len=512,
        model_config=config,
    )
    executor.load()
    assert executor.native_cursor_enabled is False
    assert executor.speech_state_capability.supported is False
    assert executor.speech_state_capability_reason == "missing_speech_state"
    # Keep capability discovery on the same Server path used by standalone
    # gateways.  The engine is intentionally not started: the loaded executor
    # is the only live TRT object and is reused below for state-transfer checks.
    server_engine = TTSEngine(
        model_arch=load_model_manifest(
            str(root), tokenizer_dir=str(root / "tokenizer")
        ),
        max_batch_size=2,
        max_seq_len=512,
    )
    server_engine._executor = executor
    package_paths = ModelPackagePaths(
        package_dir=str(root),
        engine_dir=str(root),
        weights_dir=str(root / "weights"),
        tokenizer_dir=str(root / "tokenizer"),
        manifest_path=str(root / "triton_manifest.json"),
        runtime_artifact_path=str(root / "talker_code2wav_fused.engine"),
    )
    triton_speech_state = _runtime_capabilities_from_package(
        package_paths, str(root / "tokenizer")
    )
    triton_speech_state = triton_speech_state["speech_state"]
    assert triton_speech_state == {
        "supported": executor.speech_state_capability.supported,
        "reason": executor.speech_state_capability_reason,
    }
    server_declared = server_engine.describe_capabilities()
    assert server_declared["native_cursor"] == executor.native_cursor_capability
    assert server_declared["speech_state"]["supported"] is False
    assert (
        server_declared["speech_state"]["reason"]
        == executor.speech_state_capability_reason
    )
    native_public = build_gateway_capabilities(
        server_declared,
        runtime_type=RuntimeType.STANDALONE,
        backend="native",
        native_path="/v1/ws",
        resume_grace_ms=30_000,
        resume_max_buffer_bytes=1024,
    )
    triton_declared = _runtime_capabilities_from_package(
        package_paths, str(root / "tokenizer")
    )
    triton_public = build_gateway_capabilities(
        triton_declared,
        runtime_type=RuntimeType.TRITON,
        backend="triton",
        native_path="/v1/ws",
        resume_grace_ms=30_000,
        resume_max_buffer_bytes=1024,
    )
    assert native_public["native_cursor"] == triton_public["native_cursor"]
    assert native_public["speech_state"] == triton_public["speech_state"]
    async_loop = asyncio.new_event_loop()
    try:
        loop = EngineLoop(queue.Queue(), async_loop, executor, max_batch_size=2)
        loop._running = True
        migration = loop.request_speech_state_migration(
            "session",
            0,
            expected_attempt_id=0,
            expected_allocation_epoch=1,
            max_tensor_bytes=2_000_000_000,
        )
        with pytest.raises(SpeechStateContractError, match="disabled"):
            migration.result()
    finally:
        async_loop.close()

    pool = executor.kv_pool
    source = pool.allocate("session")
    assert source is not None
    source.segment_idx = 0
    segment = EngineSegment("session", 0)
    segment.state = "active"
    metadata = capture_segment_runtime_metadata(segment)

    target = None
    try:
        # A zero embedding is sufficient here: this is a tensor-state ABI
        # check, not a text-quality or model-prompt acceptance test.
        embeds = torch.zeros(
            (1, 1, config.hidden_size), device="cuda", dtype=config.dtype
        )
        executor.prefill(source, embeds)
        executor.synchronize_state_transfer()
        bundle = executor.capture_speech_state_bundle(
            source,
            metadata,
            expected_slot_session_id="session",
            max_tensor_bytes=2_000_000_000,
        )

        target = pool.allocate("session")
        assert target is not None
        target.segment_idx = 0
        executor.restore_speech_state_bundle(
            target,
            bundle,
            expected_allocation_epoch=target.allocation_epoch,
        )
        executor.synchronize_state_transfer()

        source_output = _cpu_snapshot(executor.launch_decode_step([source]).wait())
        target_output = _cpu_snapshot(executor.launch_decode_step([target]).wait())

        assert source_output["tokens"] == target_output["tokens"]
        assert source_output["audio_chunks"] == target_output["audio_chunks"]
        assert torch.equal(source_output["codec_sum"], target_output["codec_sum"])
        assert torch.equal(source_output["updated_tc"], target_output["updated_tc"])
        _assert_equal_tensor_group(source_output["c2w_conv"], target_output["c2w_conv"])
        _assert_equal_tensor_group(
            source_output["c2w_transconv"], target_output["c2w_transconv"]
        )

        # The same restored state must remain isolated when the two slots are
        # submitted together.  Compare each row after cloning the output
        # staging buffers, which are reused by the next TRT call.
        batch_output = _cpu_snapshot(
            executor.launch_decode_step([source, target]).wait()
        )
        assert batch_output["tokens"][0] == batch_output["tokens"][1]
        assert batch_output["audio_chunks"][0] == batch_output["audio_chunks"][1]
        for key in ("codec_sum", "updated_tc"):
            value = batch_output[key]
            assert torch.equal(value[0], value[1])
        for key in ("c2w_conv", "c2w_transconv"):
            for value in batch_output[key]:
                assert torch.equal(value[0], value[1])
    finally:
        if target is not None and not target.is_free:
            pool.release(
                target.slot_id,
                expected_allocation_epoch=target.allocation_epoch,
            )
        if source is not None and not source.is_free:
            pool.release(
                source.slot_id,
                expected_allocation_epoch=source.allocation_epoch,
            )
        torch.cuda.synchronize()
