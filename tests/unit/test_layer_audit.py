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
        from layer_audit import classify_layer_name
        assert classify_layer_name("/talker_fused/talker_unified/LayerNorm_0") == "backbone"

    def test_backbone_codec_sum(self):
        from layer_audit import classify_layer_name
        assert classify_layer_name("/talker_fused/codec_sum/Add_0") == "backbone"

    def test_cp(self):
        from layer_audit import classify_layer_name
        assert classify_layer_name("/talker_fused/cp/Linear_0") == "cp"

    def test_code2wav(self):
        from layer_audit import classify_layer_name
        assert classify_layer_name("/code2wav/Conv1d_0") == "code2wav"

    def test_unclassified(self):
        """Unknown /-prefixed modules fall back to backbone."""
        from layer_audit import classify_layer_name
        assert classify_layer_name("/unknown_module/MatMul_0") == "backbone"

    def test_empty_name(self):
        """Empty names fall back to backbone."""
        from layer_audit import classify_layer_name
        assert classify_layer_name("") == "backbone"

    def test_partial_prefix_no_match(self):
        """'/talker_fused/' without a known submodule falls back to backbone."""
        from layer_audit import classify_layer_name
        assert classify_layer_name("/talker_fused/Gather_0") == "backbone"


# ---------------------------------------------------------------------------
#  AuditReport tests
# ---------------------------------------------------------------------------

class TestAuditReport:
    """Tests for AuditReport health check."""

    def test_healthy_when_below_threshold(self):
        from layer_audit import AuditReport
        report = AuditReport(
            total_nodes=100,
            categories={"backbone": [0]*80, "cp": [0]*15, "code2wav": [0]*4, "unclassified": [0]},
            unclassified_ratio=0.0,
        )
        assert report.is_healthy

    def test_healthy_at_boundary(self):
        from layer_audit import AuditReport
        report = AuditReport(
            total_nodes=100,
            categories={"backbone": [0]*80, "cp": [0]*15, "code2wav": [0]*0, "unclassified": [0]*5},
            unclassified_ratio=0.05,
        )
        assert report.is_healthy  # exactly at 5%

    def test_unhealthy_above_threshold(self):
        from layer_audit import AuditReport
        report = AuditReport(
            total_nodes=100,
            categories={"backbone": [0]*80, "cp": [0]*10, "code2wav": [0]*0, "unclassified": [0]*10},
            unclassified_ratio=0.10,
        )
        assert not report.is_healthy

