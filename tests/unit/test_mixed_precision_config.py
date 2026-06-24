"""Unit tests for mixed-precision configuration.

Covers:
- triton_manifest_io: build_manifest_for_export with precision fields
- triton_manifest_io: resolve_submodel_precisions
- engine/config: EngineProfileConfig with precision fields
- update_triton_manifest_profile: precision field recording
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
#  triton_manifest_io tests
# ---------------------------------------------------------------------------

class TestBuildManifestForExport:
    """Tests for build_manifest_for_export with mixed-precision fields."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp_path = tmp_path
        # Add scripts/python to sys.path for imports
        import sys
        sys_path = str(Path(__file__).resolve().parents[3] / "scripts" / "python")
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)

    def _minimal_weights_config(self):
        return {
            "talker_hidden_size": 2048,
            "talker_num_heads": 16,
            "talker_num_kv_heads": 8,
            "talker_head_dim": 128,
            "talker_num_layers": 28,
            "talker_vocab_size": 3072,
        }

    def _minimal_code2wav_layout(self):
        return {
            "num_code2wav_hidden_layers": 8,
            "c2w_state_input_names": [f"conv_state_{i}" for i in range(17)],
            "c2w_state_output_names": [f"new_conv_state_{i}" for i in range(17)],
            "initial_state_shapes": [],
            "logits_topk": 50,
            "cp_num_stages": 15,
            "c2w_kv_heads": 16,
            "c2w_head_dim": 64,
            "c2w_sliding_window": 72,
        }

    def test_uniform_precision_no_explicit_fields(self):
        """When no per-submodel precision is specified, all fields default to engine_dtype."""
        from triton_manifest_io import build_manifest_for_export

        manifest = build_manifest_for_export(
            variant="custom-1.7b",
            weights_config=self._minimal_weights_config(),
            code2wav_layout=self._minimal_code2wav_layout(),
            engine_dtype="bf16",
            triton_io_float_dtype="bf16",
        )

        assert manifest["engine_dtype"] == "bf16"
        assert manifest["backbone_precision"] == "bf16"
        assert manifest["cp_precision"] == "bf16"
        assert manifest["code2wav_precision"] == "bf16"

    def test_mixed_precision_cp_fp32(self):
        """When cp_precision=fp32 is specified, only cp gets fp32."""
        from triton_manifest_io import build_manifest_for_export

        manifest = build_manifest_for_export(
            variant="custom-1.7b",
            weights_config=self._minimal_weights_config(),
            code2wav_layout=self._minimal_code2wav_layout(),
            engine_dtype="bf16",
            triton_io_float_dtype="bf16",
            cp_precision="fp32",
        )

        assert manifest["engine_dtype"] == "bf16"
        assert manifest["backbone_precision"] == "bf16"
        assert manifest["cp_precision"] == "fp32"
        assert manifest["code2wav_precision"] == "bf16"

    def test_mixed_precision_all_explicit(self):
        """When all three precision fields are explicitly specified."""
        from triton_manifest_io import build_manifest_for_export

        manifest = build_manifest_for_export(
            variant="custom-1.7b",
            weights_config=self._minimal_weights_config(),
            code2wav_layout=self._minimal_code2wav_layout(),
            engine_dtype="bf16",
            triton_io_float_dtype="bf16",
            backbone_precision="bf16",
            cp_precision="fp32",
            code2wav_precision="bf16",
        )

        assert manifest["backbone_precision"] == "bf16"
        assert manifest["cp_precision"] == "fp32"
        assert manifest["code2wav_precision"] == "bf16"

    def test_engine_profile_records_precision(self):
        """Engine profile section should record per-submodule precision."""
        from triton_manifest_io import build_manifest_for_export

        manifest = build_manifest_for_export(
            variant="custom-1.7b",
            weights_config=self._minimal_weights_config(),
            code2wav_layout=self._minimal_code2wav_layout(),
            engine_dtype="bf16",
            triton_io_float_dtype="bf16",
            cp_precision="fp32",
        )

        profile = manifest["engine_profile"]
        assert profile["backbone_precision"] == "bf16"
        assert profile["cp_precision"] == "fp32"
        assert profile["code2wav_precision"] == "bf16"


class TestResolveSubmodelPrecisions:
    """Tests for resolve_submodel_precisions helper."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import sys
        sys_path = str(Path(__file__).resolve().parents[3] / "scripts" / "python")
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)

    def test_uniform_from_engine_dtype(self):
        """When no per-submodel fields, all default to engine_dtype."""
        from triton_manifest_io import resolve_submodel_precisions

        manifest = {"engine_dtype": "bf16"}
        result = resolve_submodel_precisions(manifest)
        assert result == {"backbone": "bf16", "cp": "bf16", "code2wav": "bf16"}

    def test_mixed_from_explicit_fields(self):
        """When per-submodel fields are set, they take precedence."""
        from triton_manifest_io import resolve_submodel_precisions

        manifest = {
            "engine_dtype": "bf16",
            "backbone_precision": "bf16",
            "cp_precision": "fp32",
            "code2wav_precision": "bf16",
        }
        result = resolve_submodel_precisions(manifest)
        assert result == {"backbone": "bf16", "cp": "fp32", "code2wav": "bf16"}

    def test_partial_override(self):
        """When only some fields are set, others default to engine_dtype."""
        from triton_manifest_io import resolve_submodel_precisions

        manifest = {
            "engine_dtype": "bf16",
            "cp_precision": "fp32",
        }
        result = resolve_submodel_precisions(manifest)
        assert result == {"backbone": "bf16", "cp": "fp32", "code2wav": "bf16"}


# ---------------------------------------------------------------------------
#  engine/config tests
# ---------------------------------------------------------------------------

class TestEngineProfileConfig:
    """Tests for EngineProfileConfig with mixed-precision fields."""

    def test_default_empty(self):
        from engine.config import EngineProfileConfig

        cfg = EngineProfileConfig()
        assert cfg.backbone_precision == ""
        assert cfg.cp_precision == ""
        assert cfg.code2wav_precision == ""

    def test_set_precision_fields(self):
        from engine.config import EngineProfileConfig

        cfg = EngineProfileConfig(
            engine_dtype="bf16",
            backbone_precision="bf16",
            cp_precision="fp32",
            code2wav_precision="bf16",
        )
        assert cfg.cp_precision == "fp32"


# ---------------------------------------------------------------------------
#  update_triton_manifest_profile tests
# ---------------------------------------------------------------------------

class TestUpdateManifestProfile:
    """Tests for update_triton_manifest_profile with mixed-precision fields."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp_path = tmp_path
        import sys
        sys_path = str(Path(__file__).resolve().parents[3] / "scripts" / "python")
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)

    def _write_minimal_manifest(self, path: Path) -> None:
        data = {
            "schema_version": 2,
            "variant": "custom-1.7b",
            "engine_mode": "trt",
            "engine_dtype": "bf16",
            "triton_io_float_dtype": "bf16",
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def test_mixed_precision_fields_written(self):
        """update_manifest writes mixed-precision fields to manifest."""
        import argparse
        from update_triton_manifest_profile import update_manifest

        manifest_path = self.tmp_path / "triton_manifest.json"
        self._write_minimal_manifest(manifest_path)

        args = argparse.Namespace(
            manifest=str(manifest_path),
            engine_mode="trt",
            engine_dtype="bf16",
            triton_io_float_dtype="bf16",
            backbone_precision="bf16",
            cp_precision="fp32",
            code2wav_precision="bf16",
            max_batch_size=64,
            max_input_len=128,
            max_seq_len=512,
            builder="trtexec",
            builder_image="",
            target_driver="",
            ngc_tag="",
            target_profile="",
            gpu_sm="",
            tensorrt_version="",
            skip_built_at=True,
        )
        update_manifest(args)

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["backbone_precision"] == "bf16"
        assert data["cp_precision"] == "fp32"
        assert data["code2wav_precision"] == "bf16"

        # Also check engine_profile section
        profile = data["engine_profile"]
        assert profile["backbone_precision"] == "bf16"
        assert profile["cp_precision"] == "fp32"
        assert profile["code2wav_precision"] == "bf16"

    def test_default_precision_when_empty(self):
        """When precision args are empty, they default to engine_dtype."""
        import argparse
        from update_triton_manifest_profile import update_manifest

        manifest_path = self.tmp_path / "triton_manifest.json"
        self._write_minimal_manifest(manifest_path)

        args = argparse.Namespace(
            manifest=str(manifest_path),
            engine_mode="trt",
            engine_dtype="bf16",
            triton_io_float_dtype="bf16",
            backbone_precision="",
            cp_precision="",
            code2wav_precision="",
            max_batch_size=64,
            max_input_len=128,
            max_seq_len=512,
            builder="trtexec",
            builder_image="",
            target_driver="",
            ngc_tag="",
            target_profile="",
            gpu_sm="",
            tensorrt_version="",
            skip_built_at=True,
        )
        update_manifest(args)

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Empty args should default to engine_dtype
        assert data["backbone_precision"] == "bf16"
        assert data["cp_precision"] == "bf16"
        assert data["code2wav_precision"] == "bf16"
