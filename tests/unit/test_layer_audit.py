"""Unit tests for ONNX layer prefix audit.

Covers:
- classify_layer_name: prefix-based classification
- AuditReport: health check
- LayerAuditError: threshold enforcement
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Add module to path
sys_path = str(Path(__file__).resolve().parents[3] / "scripts" / "python")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)


# ---------------------------------------------------------------------------
#  classify_layer_name tests
# ---------------------------------------------------------------------------

class TestClassifyLayerName:
    """Tests for classify_layer_name function."""

    def test_backbone_talker_unified(self):
        from qwen3tts_tools.layer_audit import classify_layer_name
        assert classify_layer_name("/talker_fused/talker_unified/LayerNorm_0") == "backbone"

    def test_backbone_codec_sum(self):
        from qwen3tts_tools.layer_audit import classify_layer_name
        assert classify_layer_name("/talker_fused/codec_sum/Add_0") == "backbone"

    def test_cp(self):
        from qwen3tts_tools.layer_audit import classify_layer_name
        assert classify_layer_name("/talker_fused/cp/Linear_0") == "cp"

    def test_code2wav(self):
        from qwen3tts_tools.layer_audit import classify_layer_name
        assert classify_layer_name("/code2wav/Conv1d_0") == "code2wav"

    def test_unclassified(self):
        from qwen3tts_tools.layer_audit import classify_layer_name
        assert classify_layer_name("/unknown_module/MatMul_0") == "unclassified"

    def test_empty_name(self):
        from qwen3tts_tools.layer_audit import classify_layer_name
        assert classify_layer_name("") == "unclassified"

    def test_partial_prefix_no_match(self):
        from qwen3tts_tools.layer_audit import classify_layer_name
        # "/talker_fused/" without a submodule prefix should be unclassified
        assert classify_layer_name("/talker_fused/Gather_0") == "unclassified"


# ---------------------------------------------------------------------------
#  AuditReport tests
# ---------------------------------------------------------------------------

class TestAuditReport:
    """Tests for AuditReport health check."""

    def test_healthy_when_below_threshold(self):
        from qwen3tts_tools.layer_audit import AuditReport
        report = AuditReport(
            total_nodes=100,
            categories={"backbone": [0]*80, "cp": [0]*15, "code2wav": [0]*4, "unclassified": [0]},
            unclassified_ratio=0.0,
        )
        assert report.is_healthy

    def test_healthy_at_boundary(self):
        from qwen3tts_tools.layer_audit import AuditReport
        report = AuditReport(
            total_nodes=100,
            categories={"backbone": [0]*80, "cp": [0]*15, "code2wav": [0]*0, "unclassified": [0]*5},
            unclassified_ratio=0.05,
        )
        assert report.is_healthy  # exactly at 5%

    def test_unhealthy_above_threshold(self):
        from qwen3tts_tools.layer_audit import AuditReport
        report = AuditReport(
            total_nodes=100,
            categories={"backbone": [0]*80, "cp": [0]*10, "code2wav": [0]*0, "unclassified": [0]*10},
            unclassified_ratio=0.10,
        )
        assert not report.is_healthy


# ---------------------------------------------------------------------------
#  mixed_precision_builder classification tests
# ---------------------------------------------------------------------------

class TestMixedPrecisionBuilderClassification:
    """Tests for mixed_precision_builder.classify_layer_name."""

    def test_backbone_talker_unified(self):
        from qwen3tts_tools.mixed_precision_builder import classify_layer_name
        assert classify_layer_name("/talker_fused/talker_unified/LayerNorm_0") == "backbone"

    def test_cp(self):
        from qwen3tts_tools.mixed_precision_builder import classify_layer_name
        assert classify_layer_name("/talker_fused/cp/Linear_0") == "cp"

    def test_code2wav(self):
        from qwen3tts_tools.mixed_precision_builder import classify_layer_name
        assert classify_layer_name("/code2wav/Conv1d_0") == "code2wav"

    def test_is_mixed_precision_uniform(self):
        from qwen3tts_tools.mixed_precision_builder import is_mixed_precision
        assert not is_mixed_precision("bf16", "bf16", "bf16")

    def test_is_mixed_precision_mixed(self):
        from qwen3tts_tools.mixed_precision_builder import is_mixed_precision
        assert is_mixed_precision("bf16", "fp32", "bf16")

    def test_is_mixed_precision_all_different(self):
        from qwen3tts_tools.mixed_precision_builder import is_mixed_precision
        assert is_mixed_precision("fp16", "fp32", "bf16")

    def test_normalize_dtype_valid(self):
        from qwen3tts_tools.mixed_precision_builder import _normalize_dtype
        assert _normalize_dtype("bf16") == "bf16"
        assert _normalize_dtype("bfloat16") == "bf16"
        assert _normalize_dtype("fp32") == "fp32"
        assert _normalize_dtype("float32") == "fp32"
        assert _normalize_dtype("fp16") == "fp16"
        assert _normalize_dtype("float16") == "fp16"

    def test_normalize_dtype_invalid(self):
        from qwen3tts_tools.mixed_precision_builder import _normalize_dtype
        with pytest.raises(ValueError):
            _normalize_dtype("invalid")
