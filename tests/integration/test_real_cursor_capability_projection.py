"""Verify Native and Triton compatibility-layer capability projection symmetry.

Triton is intentionally exercised only as a package/capability compatibility
layer here.  The real TensorRT executor is loaded once; no second TRT backend
is introduced or implied.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from engine.backend.executor import Executor
from engine.backend.kv_cache_pool import ModelConfig
from engine.config import ModelPackagePaths, load_model_manifest
from engine.gateway.capabilities import RuntimeType, build_gateway_capabilities
from engine.gateway.triton_realtime_server import _runtime_capabilities_from_package
from engine.server import TTSEngine


def _inputs() -> tuple[Path, Path]:
    if os.environ.get("RUN_REAL_CURSOR_CAPABILITY_TESTS", "").strip() != "1":
        pytest.skip("set RUN_REAL_CURSOR_CAPABILITY_TESTS=1 for real capability evidence")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real cursor capability evidence")
    root = Path(
        os.environ.get(
            "QWEN_REAL_NATIVE_CURSOR_TRT_ARTIFACT_DIR",
            "/home/rime/workspace/models/x2-exported/custom-1.7b",
        )
    ).resolve()
    tokenizer = Path(
        os.environ.get(
            "QWEN_REAL_X2_TOKENIZER_DIR",
            "/home/rime/workspace/models/X2Streaming-TTS-1.7B",
        )
    ).resolve()
    if not (root / "talker_code2wav_fused.engine").is_file():
        pytest.skip(f"no cursor-enabled TRT plan in {root}")
    if not (root / "triton_manifest.json").is_file():
        pytest.skip(f"no manifest in {root}")
    return root, tokenizer


def test_real_cursor_native_and_triton_projection_are_identical() -> None:
    root, tokenizer = _inputs()
    manifest = json.loads((root / "triton_manifest.json").read_text(encoding="utf-8"))
    assert manifest["native_cursor"]["enabled"] is True

    executor = Executor(
        engine_dir=str(root),
        weights_dir=str(root / "weights"),
        max_batch_size=8,
        max_seq_len=512,
        model_config=ModelConfig(dtype=torch.bfloat16),
    )
    executor.load()
    assert executor.native_cursor_enabled is True

    arch = load_model_manifest(str(root), tokenizer_dir=str(tokenizer))
    native_engine = TTSEngine(
        model_arch=arch,
        max_batch_size=8,
        max_seq_len=512,
    )
    native_engine._executor = executor
    native_declared = native_engine.describe_capabilities()

    package = ModelPackagePaths(
        package_dir=str(root),
        engine_dir=str(root),
        weights_dir=str(root / "weights"),
        tokenizer_dir=str(tokenizer),
        manifest_path=str(root / "triton_manifest.json"),
        runtime_artifact_path=str(root / "talker_code2wav_fused.engine"),
    )
    triton_declared = _runtime_capabilities_from_package(package, str(tokenizer))

    native_public = build_gateway_capabilities(
        native_declared,
        runtime_type=RuntimeType.STANDALONE,
        backend="native",
        native_path="/v1/ws",
        resume_grace_ms=30_000,
        resume_max_buffer_bytes=1024,
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
    assert native_public["native_cursor"]["graph_enabled"] is True
    assert native_public["native_cursor"]["progress_available"] is False
    assert native_public["native_cursor"]["supported_progress_modes"] == [
        "ema",
        "disabled",
    ]
    assert native_public["speech_state"]["supported"] is False
