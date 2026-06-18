"""Tests for engine.config — YAML config loading + model manifest."""

import json
import os
import tempfile

import pytest

from engine.config import (
    EngineConfig,
    ModelArchConfig,
    load_config,
    load_model_manifest,
    resolve_model_package_paths,
    _coerce_value,
    _apply_env_overrides,
    _dict_to_config,
    to_model_config,
)


class TestCoercion:
    def test_bool_true(self):
        assert _coerce_value("true") is True
        assert _coerce_value("yes") is True

    def test_bool_false(self):
        assert _coerce_value("false") is False
        assert _coerce_value("no") is False

    def test_int(self):
        assert _coerce_value("42") == 42

    def test_float(self):
        assert _coerce_value("3.14") == pytest.approx(3.14)

    def test_string(self):
        assert _coerce_value("hello") == "hello"


class TestEnvOverrides:
    def test_applies_engine_prefix(self):
        raw: dict = {"scheduler": {"max_batch_size": 48}}
        env = {"ENGINE_SCHEDULER_MAX_BATCH_SIZE": "32"}
        with _patch_env(env):
            _apply_env_overrides(raw)
        assert raw["scheduler"]["max_batch_size"] == 32

    def test_applies_server_websocket_overrides(self):
        raw: dict = {"server": {"websocket_port": 0, "websocket_path": "/v1/ws"}}
        env = {
            "ENGINE_SERVER_WEBSOCKET_PORT": "50052",
            "ENGINE_SERVER_WEBSOCKET_PATH": "/stream/ws",
        }
        with _patch_env(env):
            _apply_env_overrides(raw)
        assert raw["server"]["websocket_port"] == 50052
        assert raw["server"]["websocket_path"] == "/stream/ws"

    def test_applies_multiword_section_overrides(self):
        raw: dict = {}
        env = {
            "ENGINE_REFERENCE_CACHE_ENABLED": "false",
            "ENGINE_REFERENCE_CACHE_MAX_ENTRIES": "4",
        }
        with _patch_env(env):
            _apply_env_overrides(raw)
        assert raw["reference_cache"]["enabled"] is False
        assert raw["reference_cache"]["max_entries"] == 4

    def test_ignores_non_engine(self):
        raw: dict = {}
        env = {"OTHER_VAR": "123"}
        with _patch_env(env):
            _apply_env_overrides(raw)
        assert "other" not in raw


class TestLoadConfig:
    def test_defaults(self, tmp_path):
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            cfg = load_config()
        finally:
            os.chdir(cwd)
        assert cfg.scheduler.max_batch_size == 48
        assert cfg.prefix_cache.enabled is True
        assert cfg.reference_cache.enabled is True
        assert cfg.reference_cache.max_entries == 16

    def test_yaml_file(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        content = "scheduler:\n  max_batch_size: 16\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()
            cfg = load_config(f.name)
        os.unlink(f.name)
        assert cfg.scheduler.max_batch_size == 16

    def test_no_model_section(self):
        """EngineConfig no longer has a model section."""
        cfg = load_config()
        assert not hasattr(cfg, "model")

    def test_cli_overrides(self):
        cfg = load_config(cli_overrides={"scheduler": {"max_batch_size": 8}})
        assert cfg.scheduler.max_batch_size == 8

    def test_reference_cache_cli_overrides(self):
        cfg = load_config(
            cli_overrides={
                "reference_cache": {
                    "enabled": False,
                    "max_entries": 3,
                }
            }
        )
        assert cfg.reference_cache.enabled is False
        assert cfg.reference_cache.max_entries == 3

    def test_missing_file_uses_defaults(self):
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg.scheduler.max_batch_size == 48

    def test_websocket_defaults_from_cli_overrides(self):
        cfg = load_config(
            cli_overrides={
                "server": {
                    "websocket_port": 50052,
                    "websocket_path": "/stream/ws",
                }
            }
        )
        assert cfg.server.websocket_port == 50052
        assert cfg.server.websocket_path == "/stream/ws"


class TestModelManifest:
    def test_defaults(self):
        arch = load_model_manifest("", None)
        assert arch.num_layers == 28
        assert arch.hidden_size == 2048

    def test_from_manifest_file(self, tmp_path):
        manifest = {
            "schema_version": 2,
            "variant": "custom-1.7b",
            "architecture": {
                "num_layers": 28,
                "hidden_size": 2048,
                "kv_heads": 8,
                "head_dim": 128,
                "codec_vocab_size": 3072,
                "logits_topk": 50,
                "cp_num_stages": 15,
            },
        }
        manifest_file = tmp_path / "triton_manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        arch = load_model_manifest(str(tmp_path), None)
        assert arch.variant == "custom-1.7b"
        assert arch.hidden_size == 2048
        assert arch.head_dim == 128
        assert arch.codec_vocab_size == 3072
        assert arch.cp_num_stages == 15

    def test_legacy_manifest_compat(self, tmp_path):
        """v1 manifest without architecture section still works."""
        manifest = {
            "schema_version": 1,
            "variant": "base-0.6b",
            "talker": {
                "num_layers": 28,
                "hidden_size": 1536,
                "num_kv_heads": 4,
                "head_dim": 64,
                "vocab_size": 2176,
            },
        }
        manifest_file = tmp_path / "triton_manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        arch = load_model_manifest(str(tmp_path), None)
        assert arch.variant == "base-0.6b"
        assert arch.hidden_size == 1536
        assert arch.head_dim == 64

    def test_manifest_priority_over_defaults(self, tmp_path):
        manifest = {
            "variant": "test",
            "architecture": {"hidden_size": 4096, "head_dim": 256},
        }
        (tmp_path / "triton_manifest.json").write_text(json.dumps(manifest))

        arch = load_model_manifest(str(tmp_path), None)
        assert arch.hidden_size == 4096
        assert arch.head_dim == 256
        assert arch.num_layers == 28  # still default

    def test_manifest_reads_orchestrator_task_capabilities(self, tmp_path):
        manifest = {
            "variant": "base-1.7b",
            "orchestrator": {
                "tts_model_type": "base",
                "supported_task_types": ["base", "instruct", "voice_clone"],
            },
        }
        (tmp_path / "triton_manifest.json").write_text(json.dumps(manifest))

        arch = load_model_manifest(str(tmp_path), None)
        assert arch.tts_model_type == "base"
        assert arch.supported_task_types == ("base", "instruct", "voice_clone")

    def test_manifest_reads_engine_profile(self, tmp_path):
        manifest = {
            "variant": "custom-1.7b",
            "engine_profile": {
                "engine_mode": "trt",
                "engine_dtype": "bf16",
                "triton_io_float_dtype": "bf16",
                "max_batch_size": 64,
                "max_input_len": 128,
                "max_seq_len": 512,
                "builder_image": "nvcr.io/nvidia/tritonserver:26.02-py3",
                "target_driver": "575.57",
            },
        }
        (tmp_path / "triton_manifest.json").write_text(json.dumps(manifest))

        arch = load_model_manifest(str(tmp_path), None)

        assert arch.engine_profile.engine_mode == "trt"
        assert arch.engine_profile.engine_dtype == "bf16"
        assert arch.engine_profile.max_batch_size == 64
        assert arch.engine_profile.max_input_len == 128
        assert arch.engine_profile.max_seq_len == 512
        assert arch.engine_profile.builder_image == "nvcr.io/nvidia/tritonserver:26.02-py3"

    def test_manifest_searches_variant_parent_from_engine_subdir(self, tmp_path):
        manifest = {
            "variant": "custom-1.7b",
            "engine_profile": {"max_batch_size": 32, "max_seq_len": 384},
        }
        (tmp_path / "triton_manifest.json").write_text(json.dumps(manifest))
        engine_subdir = tmp_path / "engines" / "talker_code2wav_fused"
        engine_subdir.mkdir(parents=True)

        arch = load_model_manifest(str(engine_subdir), None)

        assert arch.variant == "custom-1.7b"
        assert arch.engine_profile.max_batch_size == 32
        assert arch.engine_profile.max_seq_len == 384


class TestModelPackagePaths:
    def test_defaults_to_triton_model_version_layout(self):
        paths = resolve_model_package_paths("/models/tts_orchestrator/1")
        assert paths.package_dir == "/models/tts_orchestrator/1"
        assert paths.engine_dir == "/models/tts_orchestrator/1/runtime"
        assert paths.weights_dir == "/models/tts_orchestrator/1/weights"
        assert paths.tokenizer_dir == "/models/tts_orchestrator/1/tokenizer"
        assert paths.manifest_path == "/models/tts_orchestrator/1/runtime/triton_manifest.json"
        assert paths.runtime_artifact_path == "/models/tts_orchestrator/1/runtime/model.plan"

    def test_manifest_can_describe_extensible_package_layout(self, tmp_path):
        package = tmp_path / "tts_orchestrator" / "1"
        package.mkdir(parents=True)
        (package / "triton_manifest.json").write_text(
            json.dumps(
                {
                    "engine_mode": "trt",
                    "package": {
                        "runtime_dir": "rt",
                        "weights_dir": "w",
                        "tokenizer_dir": "tok",
                        "runtime_artifacts": {"trt": "rt/fused.plan"},
                    },
                }
            ),
            encoding="utf-8",
        )

        paths = resolve_model_package_paths(str(package))

        assert paths.engine_mode == "trt"
        assert paths.engine_dir == str(package / "rt")
        assert paths.weights_dir == str(package / "w")
        assert paths.tokenizer_dir == str(package / "tok")
        assert paths.runtime_artifact_path == str(package / "rt" / "fused.plan")


class TestToModelConfig:
    def test_conversion(self):
        arch = ModelArchConfig()
        cfg = EngineConfig()
        mc = to_model_config(arch, cfg)
        assert mc.num_layers == arch.num_layers
        assert mc.kv_heads == arch.kv_heads
        assert mc.cp_num_stages == arch.cp_num_stages
        assert mc.c2w_sliding_window == arch.c2w_sliding_window
        assert mc.max_seq_len == cfg.scheduler.max_seq_len


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class _patch_env:
    def __init__(self, env_vars: dict):
        self._vars = env_vars
        self._orig = {}

    def __enter__(self):
        for k, v in self._vars.items():
            self._orig[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *args):
        for k in self._vars:
            if self._orig[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = self._orig[k]
