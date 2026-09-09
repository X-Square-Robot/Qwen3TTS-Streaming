"""Executor must consume the package resolver's canonical artifact paths."""

import json
from pathlib import Path

import pytest
import torch

import engine.backend.executor as executor_module
from engine.backend.executor import Executor
from engine.backend.kv_cache_pool import ModelConfig
from engine.config import ModelPackagePaths
from engine.runtime.release_gate import ReleaseGate
from engine.core.speech_state_bundle import SpeechStateBundleValidation


class _FakeTRTEngine:
    def __init__(self, path, device):
        self.path = path
        self.device = device

    def load(self):
        return None


def _package_paths(tmp_path, runtime_relative="engines/bf16/nested"):
    package = tmp_path / "package"
    runtime = package if runtime_relative == "." else package / runtime_relative
    weights = package / "weights"
    runtime.mkdir(parents=True)
    weights.mkdir()
    manifest = package / "triton_manifest.json"
    manifest.write_text(json.dumps({"variant": "nested"}), encoding="utf-8")
    plan = runtime / "custom.plan"
    plan.write_bytes(b"plan")
    return ModelPackagePaths(
        package_dir=str(package),
        engine_dir=str(runtime),
        weights_dir=str(weights),
        tokenizer_dir=str(package / "tokenizer"),
        manifest_path=str(manifest),
        runtime_artifact_path=str(plan),
    )


def _executor(paths):
    executor = Executor.__new__(Executor)
    executor._package_paths = paths
    executor._engine_dir = Path(paths.engine_dir)
    executor._weights_dir = Path(paths.weights_dir)
    executor._device = torch.device("cpu")
    executor._config = ModelConfig()
    executor._max_batch = 1
    executor._max_seq_len = 4
    executor._max_input_len = 0
    executor._manifest = {}
    executor._fused_engine = None
    executor._graph_decode = None
    executor._speech_state_bundle_validation = SpeechStateBundleValidation(
        False, "bundle_not_loaded"
    )
    executor._release_gate = ReleaseGate.disabled()
    executor._kv_pool = None
    return executor


@pytest.mark.parametrize("runtime_relative", [".", "engines/bf16/nested"])
def test_executor_uses_package_manifest_plan_and_bundle_root(
    tmp_path, monkeypatch, runtime_relative
):
    paths = _package_paths(tmp_path, runtime_relative)
    executor = _executor(paths)
    seen = {}

    def validate(manifest, *, bundle_root, runtime_artifact_path):
        seen["manifest"] = manifest
        seen["bundle_root"] = bundle_root
        seen["runtime_artifact_path"] = runtime_artifact_path
        return SpeechStateBundleValidation(False, "missing_speech_state")

    monkeypatch.setattr(executor_module, "TRTEngine", _FakeTRTEngine)
    monkeypatch.setattr(
        executor_module, "KVCachePool", lambda **_kwargs: object()
    )
    monkeypatch.setattr(executor_module, "validate_speech_state_bundle", validate)
    monkeypatch.setattr(
        executor_module,
        "evaluate_release_gate",
        lambda manifest, evidence: seen.update(evidence=evidence)
        or ReleaseGate.disabled(),
    )
    for name in (
        "_apply_runtime_profile_limits",
        "_validate_io_dtype_consistency",
        "_discover_c2w_io_names",
        "_discover_cursor_io_names",
        "_init_c2w_state_arenas",
        "_init_graph_decode",
    ):
        monkeypatch.setattr(Executor, name, lambda self: None)

    executor.load()

    assert executor._fused_engine.path == paths.runtime_artifact_path
    assert seen["bundle_root"] == Path(paths.package_dir)
    assert seen["runtime_artifact_path"] == Path(paths.runtime_artifact_path)
    assert seen["manifest"] == {"variant": "nested"}
    assert seen["evidence"] is None


def test_executor_without_package_paths_keeps_legacy_runtime_discovery(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    plan = runtime / "model.plan"
    plan.write_bytes(b"plan")
    (runtime / "triton_manifest.json").write_text("{}", encoding="utf-8")
    executor = _executor(
        ModelPackagePaths(
            package_dir=str(tmp_path),
            engine_dir=str(runtime),
            weights_dir=str(tmp_path / "weights"),
            tokenizer_dir=str(tmp_path / "tokenizer"),
            manifest_path=str(runtime / "triton_manifest.json"),
            runtime_artifact_path=str(plan),
        )
    )
    del executor._package_paths
    seen = {}
    monkeypatch.setattr(executor_module, "TRTEngine", _FakeTRTEngine)
    monkeypatch.setattr(
        executor_module, "KVCachePool", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        executor_module,
        "validate_speech_state_bundle",
        lambda manifest, *, bundle_root, runtime_artifact_path: (
            seen.__setitem__("root", bundle_root)
            or SpeechStateBundleValidation(False, "missing_speech_state")
        ),
    )
    for name in (
        "_apply_runtime_profile_limits",
        "_validate_io_dtype_consistency",
        "_discover_c2w_io_names",
        "_discover_cursor_io_names",
        "_init_c2w_state_arenas",
        "_init_graph_decode",
    ):
        monkeypatch.setattr(Executor, name, lambda self: None)

    executor.load()

    assert executor._fused_engine.path == str(plan)
    assert seen["root"] == tmp_path
