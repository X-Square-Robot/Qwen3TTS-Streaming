"""Black-box loading evidence for a real cursor-enabled fused TRT artifact.

This test proves the exported engine, manifest, model-owned cursor head and
runtime ABI agree.  It intentionally does not advertise native progress or
speech-state continuity: those still require release evidence and a real
successor trajectory.

Run explicitly with::

    RUN_REAL_NATIVE_CURSOR_TRT_TESTS=1 \\
    QWEN_REAL_NATIVE_CURSOR_TRT_ARTIFACT_DIR=/path/to/custom-1.7b \\
    pytest -q tests/integration/test_real_native_cursor_trt_artifact_contract.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from engine.backend.executor import Executor
from engine.backend.kv_cache_pool import ModelConfig
from engine.core.native_cursor_labelizer import load_native_cursor_labelizer


def _artifact_dir() -> Path:
    if os.environ.get("RUN_REAL_NATIVE_CURSOR_TRT_TESTS", "").strip() != "1":
        pytest.skip("set RUN_REAL_NATIVE_CURSOR_TRT_TESTS=1 for real cursor TRT evidence")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for a real cursor TRT artifact check")
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pytest.skip("TensorRT is required for a real cursor TRT artifact check")
    value = os.environ.get("QWEN_REAL_NATIVE_CURSOR_TRT_ARTIFACT_DIR", "").strip()
    if not value:
        pytest.skip("set QWEN_REAL_NATIVE_CURSOR_TRT_ARTIFACT_DIR")
    root = Path(value).resolve()
    if not (root / "talker_code2wav_fused.engine").is_file():
        pytest.skip(f"no fused TRT plan in {root}")
    return root


def test_real_cursor_trt_manifest_head_and_executor_abi() -> None:
    root = _artifact_dir()
    manifest = json.loads((root / "triton_manifest.json").read_text(encoding="utf-8"))
    cursor = manifest["native_cursor"]
    assert cursor["enabled"] is True
    assert cursor["cursor_head_sha256"] == cursor["head_sha256"]

    labelizer = load_native_cursor_labelizer(
        root / "weights" / "qwen3_tts_12hz_la1_seed0.pt",
        expected_vocab_sha256=cursor["cursor_vocab_sha256"],
    )
    assert labelizer.vocab_size == cursor["vocab_size"]

    executor = Executor(
        engine_dir=str(root),
        weights_dir=str(root / "weights"),
        max_batch_size=8,
        max_seq_len=512,
        model_config=ModelConfig(dtype=torch.bfloat16),
    )
    executor.load()

    assert executor.native_cursor_enabled is True
    assert executor._cursor_state_handoff_enabled is True
    assert len(executor._cursor_input_names) == len(cursor["input_names"])
    assert len(executor._cursor_output_names) == len(cursor["output_names"])
    assert executor._fused_engine._engine.num_io_tensors == 85

