"""Optional checks against a real exported TensorRT artifact.

Set ``QWEN_REAL_TRT_ARTIFACT_DIR`` to an exported variant directory to run
this test.  The default repository artifact is a standard custom-1.7B plan;
the test intentionally proves that it is not advertised as cursor or speech
state capable merely because it contains the fused Talker/C2W graph.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from engine.runtime.release_gate import ReleaseCapability, evaluate_release_gate
from engine.core.speech_state_bundle import validate_speech_state_bundle


def _artifact_dir() -> Path:
    value = os.environ.get("QWEN_REAL_TRT_ARTIFACT_DIR", "").strip()
    if not value:
        pytest.skip("set QWEN_REAL_TRT_ARTIFACT_DIR for a real TRT artifact check")
    root = Path(value)
    if not root.is_dir():
        pytest.skip(f"TRT artifact directory does not exist: {root}")
    return root


def test_real_standard_trt_artifact_is_fail_closed_for_optional_capabilities():
    root = _artifact_dir()
    manifest = json.loads((root / "triton_manifest.json").read_text(encoding="utf-8"))
    engine_path = next(
        (
            candidate
            for candidate in (
                root / "talker_code2wav_fused.engine",
                root / "model.plan",
            )
            if candidate.is_file()
        ),
        None,
    )
    if engine_path is None:
        pytest.skip(f"no TensorRT plan in {root}")

    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    assert engine is not None

    names = {
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
    }
    assert "full_codec" in names
    assert "c2w_conv_state_0" in names
    assert "c2w_new_conv_state_0" in names
    assert not any(name.startswith("cursor_") for name in names)

    native_cursor = manifest.get("native_cursor") or {}
    assert native_cursor.get("enabled", False) is False
    assert "speech_state" not in manifest

    bundle = validate_speech_state_bundle(manifest, bundle_root=root)
    assert bundle.verified is False
    assert bundle.reason == "missing_speech_state"

    gate = evaluate_release_gate(manifest, None)
    assert gate.verified(ReleaseCapability.NATIVE_CURSOR) is False
    assert gate.verified(ReleaseCapability.SPEECH_STATE) is False
