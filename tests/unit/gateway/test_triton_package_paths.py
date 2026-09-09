"""Triton capability discovery must use the canonical package root."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.config import ModelPackagePaths
from engine.core.speech_state_bundle import SpeechStateBundleValidation
from engine.gateway.triton_realtime_server import _runtime_capabilities_from_package


@pytest.mark.parametrize("runtime_relative", [".", "runtime", "engines/bf16/nested"])
def test_triton_uses_package_dir_for_bundle_and_evidence(
    tmp_path, monkeypatch, runtime_relative
):
    package = tmp_path / "package"
    runtime = package / runtime_relative
    runtime.mkdir(parents=True)
    weights = package / "weights"
    weights.mkdir()
    manifest_path = package / "triton_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "native_cursor": {},
                "speech_state": {
                    "model_fingerprint": "model-v1",
                    "runtime_fingerprint": "runtime-v1",
                },
            }
        ),
        encoding="utf-8",
    )
    # The package-root evidence is intentionally malformed while the runtime
    # copy is complete.  Canonical package-root precedence must keep native
    # progress closed rather than silently selecting the runtime copy.
    (package / "capability_evidence.json").write_text(
        json.dumps({"schema_version": "invalid"}), encoding="utf-8"
    )
    if runtime != package:
        (runtime / "capability_evidence.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_fingerprint": "model-v1",
                    "runtime_fingerprint": "runtime-v1",
                    "performance": {"verified": True},
                    "native_cursor": {
                        "trt_numeric_verified": True,
                        "progress_e2e_verified": True,
                        "monotonic_verified": True,
                    },
                }
            ),
            encoding="utf-8",
        )
    plan = runtime / "model.plan"
    plan.write_bytes(b"plan")
    paths = ModelPackagePaths(
        package_dir=str(package),
        engine_dir=str(runtime),
        weights_dir=str(weights),
        tokenizer_dir=str(package / "tokenizer"),
        manifest_path=str(manifest_path),
        runtime_artifact_path=str(plan),
    )
    seen = {}
    monkeypatch.setattr(
        "engine.gateway.triton_realtime_server.load_model_manifest",
        lambda *_args, **_kwargs: SimpleNamespace(
            tts_model_type="custom_voice",
            supported_task_types=("custom_voice",),
            variant="custom-1.7b",
            native_cursor={"enabled": True, "progress_available": True},
            engine_profile=SimpleNamespace(
                max_batch_size=1,
                max_input_len=1,
                max_seq_len=1,
                engine_dtype="bf16",
                triton_io_float_dtype="fp32",
            ),
        ),
    )
    monkeypatch.setattr(
        "engine.gateway.triton_realtime_server.validate_speech_state_bundle",
        lambda manifest, *, bundle_root, runtime_artifact_path: (
            seen.__setitem__("bundle_root", bundle_root)
            or SpeechStateBundleValidation(False, "missing_speech_state")
        ),
    )

    capabilities = _runtime_capabilities_from_package(paths, "")

    assert seen["bundle_root"] == Path(paths.package_dir)
    assert capabilities["native_cursor"]["progress_available"] is False
    assert capabilities["native_cursor"]["reason"] == "release_evidence_missing"
