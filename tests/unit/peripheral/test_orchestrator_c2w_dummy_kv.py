"""Regression tests for fused orchestrator code2wav cold-start KV handling."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCH_DIR = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if str(ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(ORCH_DIR))

sys.modules.setdefault(
    "triton_python_backend_utils",
    types.SimpleNamespace(TRITONSERVER_RESPONSE_COMPLETE_FINAL=1),
)

from model import TritonPythonModel

pytestmark = pytest.mark.skipif(
    not all(
        hasattr(TritonPythonModel, name)
        for name in ("_build_c2w_attention_bias", "_postprocess_c2w_state_output")
    ),
    reason="thin Triton adapter no longer owns code2wav dummy-KV helpers",
)


def _make_model(*, sliding_window: int = 72) -> TritonPythonModel:
    model = TritonPythonModel.__new__(TritonPythonModel)
    model.device = torch.device("cpu")
    model._code2wav_dtype = torch.float32
    model.code2wav_sliding_window = sliding_window
    return model


def test_build_c2w_attention_bias_masks_dummy_slot():
    model = _make_model()

    bias = model._build_c2w_attention_bias(
        batch=2,
        chunk_t=1,
        c2w_past_len=1,
        use_dummy_past_kv=True,
    )

    assert bias.shape == (2, 1, 1, 2)
    assert torch.isneginf(bias[:, :, :, 0]).all()
    assert torch.equal(bias[:, :, :, 1], torch.zeros_like(bias[:, :, :, 1]))


def test_build_c2w_attention_bias_keeps_real_history_unmasked():
    model = _make_model()

    bias = model._build_c2w_attention_bias(
        batch=1,
        chunk_t=1,
        c2w_past_len=3,
        use_dummy_past_kv=False,
    )

    assert bias.shape == (1, 1, 1, 4)
    assert torch.equal(bias, torch.zeros_like(bias))


def test_postprocess_c2w_state_output_drops_dummy_then_clips():
    model = _make_model(sliding_window=4)
    tensor = torch.arange(1 * 2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)

    out = model._postprocess_c2w_state_output(
        "c2w_present_kv_0_k",
        tensor,
        use_dummy_past_kv=True,
    )

    expected = tensor[:, :, 1:, :][:, :, -3:, :].contiguous()
    assert out.shape == (1, 2, 3, 3)
    assert torch.equal(out, expected)


def test_postprocess_c2w_state_output_keeps_non_kv_state():
    model = _make_model(sliding_window=4)
    tensor = torch.arange(1 * 2 * 6, dtype=torch.float32).reshape(1, 2, 6)

    out = model._postprocess_c2w_state_output(
        "c2w_conv_state_0",
        tensor,
        use_dummy_past_kv=True,
    )

    assert torch.equal(out, tensor)
