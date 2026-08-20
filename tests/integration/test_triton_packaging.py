"""Regression tests for Triton packaging: manifest generation, config rendering, and assembly."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PY = REPO_ROOT / "scripts" / "python"
sys.path.insert(0, str(SCRIPTS_PY))

from generate_triton_configs import (  # noqa: E402
    generate_configs,
    render_code2wav_streaming,
    render_orchestrator,
    render_orchestrator_http,
    render_speech_tokenizer_encoder_trt,
    render_talker_unified_trt,
    triton_io_float_pbtxt_from_manifest,
)
from triton_manifest_io import build_manifest_for_export, load_manifest  # noqa: E402


FIXTURE_MANIFEST = REPO_ROOT / "tools" / "data" / "triton_manifest_custom_1_7b.json"
FIXTURE_LAYOUT = (
    REPO_ROOT / "workspace" / "exported" / "custom-1.7b" / "code2wav_state_layout.json"
)


# ---------------------------------------------------------------------------
#  Helpers for assembly tests
# ---------------------------------------------------------------------------


def _write_custom_export(exported_dir: Path) -> None:
    variant_dir = exported_dir / "custom-1.7b"
    variant_dir.mkdir(parents=True)
    (exported_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "ngc_tag": "25.03",
                "ngc_image": "nvcr.io/nvidia/tritonserver:25.03-py3",
                "tensorrt_version": "10.9.0",
                "gpu_sm": "sm_89",
                "engine_dtype": "bf16",
                "engines": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    manifest["package"] = {
        "schema_version": 1,
        "layout": "triton_model_version",
        "model_package_dir": "/models/tts_orchestrator/1",
        "runtime_dir": "runtime",
        "weights_dir": "weights",
        "tokenizer_dir": "tokenizer",
        "manifest": "runtime/triton_manifest.json",
        "runtime_artifacts": {
            "trt": "runtime/model.plan",
            "onnx": "runtime/model.onnx",
        },
        "optional_assets": {
            "speaker_encoder": "runtime/speaker_encoder.onnx",
            "speech_tokenizer_encoder": "runtime/speech_tokenizer_encoder.onnx",
            "speech_tokenizer_codec_fused": "runtime/speech_tokenizer_codec_fused.onnx",
        },
    }
    (variant_dir / "triton_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (variant_dir / "MODEL_VERSION").write_text(
        "zehan@20260601\n",
        encoding="utf-8",
    )
    (variant_dir / "talker_code2wav_fused.engine").write_bytes(b"fake-plan")
    weights_dir = variant_dir / "weights"
    weights_dir.mkdir()
    (weights_dir / "config.json").write_text(
        json.dumps(
            {
                "talker_hidden_size": 2048,
                "talker_num_heads": 16,
                "talker_num_kv_heads": 8,
                "talker_head_dim": 128,
                "talker_num_layers": 28,
                "talker_vocab_size": 3072,
            }
        ),
        encoding="utf-8",
    )

    tokenizer_dir = exported_dir / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "speech_tokenizer_encoder.onnx").write_bytes(b"fake-onnx")
    (tokenizer_dir / "code2wav_decoder.engine").write_bytes(b"fake-engine")

    model_dir = exported_dir.parent / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model_dir.mkdir(parents=True)
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


def _assemble(
    exported_dir: Path,
    repo_dir: Path,
    variant: str,
    engine_mode: str,
    *,
    package_date: str = "2026-08-20",
) -> None:
    cmd = """
set -euo pipefail
source "{REPO_ROOT}/scripts/bash/lib/triton.sh"
assemble_model_repo "$1" "$2" "$3" "$4" "1"
""".format(REPO_ROOT=REPO_ROOT)
    subprocess.run(
        [
            "bash",
            "-c",
            cmd,
            "bash",
            str(exported_dir),
            variant,
            str(repo_dir),
            engine_mode,
        ],
        cwd=REPO_ROOT,
        check=True,
        env={
            **os.environ,
            "QWEN3_TTS_ENGINE_BUILD_VERSION": "engine-builder@20260820_v1",
            "QWEN3_TTS_PACKAGER": "packager-test",
            "QWEN3_TTS_PACKAGE_DATE": package_date,
        },
    )


# ===========================================================================
#  Manifest generation & config rendering tests
#  (from test_triton_manifest_generator.py)
# ===========================================================================


def test_render_talker_unified_28_layers_bf16():
    talker = {
        "hidden_size": 2048,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_layers": 28,
        "vocab_size": 3072,
    }
    text = render_talker_unified_trt(talker, "bf16")
    assert 'name: "talker_unified"' in text
    assert "TYPE_BF16" in text
    assert text.count('name: "past_kv_') == 28 * 2
    assert 'name: "past_kv_27_v"' in text
    assert 'name: "present_kv_27_k"' in text


def test_render_code2wav_trt_bf16_has_8_kv_layers():
    text = render_code2wav_streaming("trt", "bf16")
    assert 'name: "code2wav"' in text
    assert 'name: "past_kv_7_v"' in text
    assert 'name: "conv_state_16"' in text
    assert 'name: "new_transconv_overlap_3"' in text


def test_render_orchestrator_custom_variant():
    orch = {
        "tts_model_type": "custom_voice",
        "supported_task_types": "custom_voice",
        "max_decode_steps": "4096",
        "audio_chunk_frames": "25",
        "first_chunk_frames": "4",
        "model_package_dir": "/models/tts_orchestrator/1",
    }
    text = render_orchestrator("custom-1.7b", orch)
    assert 'string_value: "custom-1.7b"' in text
    assert 'string_value: "custom_voice"' in text
    assert 'key: "model_package_dir"' in text
    assert 'string_value: "/models/tts_orchestrator/1"' in text
    assert 'key: "do_sample"\n  value: { string_value: "false" }' in text


def test_render_orchestrator_http_custom_variant():
    orch = {
        "tts_model_type": "custom_voice",
        "supported_task_types": "custom_voice",
    }
    text = render_orchestrator_http("custom-1.7b", orch)
    assert 'name: "tts_orchestrator_http"' in text
    assert "kind: KIND_CPU" in text
    assert 'string_value: "tts_orchestrator"' in text
    assert 'name: "audio_chunk"' in text


def test_render_speech_tokenizer_encoder_trt():
    text = render_speech_tokenizer_encoder_trt()
    assert "TYPE_INT64" in text and "audio_codes" in text


def test_load_manifest_merges_orchestrator_defaults(tmp_path):
    mpath = tmp_path / "triton_manifest.json"
    mpath.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": "custom-1.7b",
                "code2wav_fused": {
                    "num_code2wav_hidden_layers": 8,
                    "c2w_state_input_names": [],
                    "c2w_state_output_names": [],
                    "initial_state_shapes": [],
                },
            }
        ),
        encoding="utf-8",
    )
    m = load_manifest(mpath, output_repo=None)
    assert m["orchestrator"]["tts_model_type"] == "custom_voice"
    assert m["orchestrator"]["model_package_dir"] == "/models/tts_orchestrator/1"
    assert m["orchestrator"]["do_sample"] == "false"


def test_build_manifest_for_export_roundtrip():
    if not FIXTURE_LAYOUT.is_file():
        pytest.skip("workspace exported layout not present")
    lay = json.loads(FIXTURE_LAYOUT.read_text(encoding="utf-8"))
    wc = {
        "talker_hidden_size": 2048,
        "talker_num_kv_heads": 8,
        "talker_head_dim": 128,
        "talker_num_layers": 28,
        "talker_vocab_size": 3072,
        "talker_num_heads": 16,
    }
    m = build_manifest_for_export("custom-1.7b", wc, lay)
    assert m["schema_version"] == 2
    assert m.get("triton_io_float_dtype") == "bf16"
    assert m["package"]["layout"] == "triton_model_version"
    assert m["package"]["runtime_artifacts"]["trt"] == "runtime/model.plan"
    assert m["package"]["optional_assets"] == {}
    assert m["code2wav_fused"]["packed_kv"] is True
    assert m["code2wav_fused"]["c2w_kv_heads"] == 16
    assert m["code2wav_fused"]["c2w_head_dim"] == 64
    assert m["code2wav_fused"]["c2w_state_input_names"][0] == "c2w_conv_state_0"


def test_triton_io_float_dtype_from_manifest_defaults_fp32():
    assert (
        triton_io_float_pbtxt_from_manifest({}) == "TYPE_FP32"
    )  # missing key defaults fp32
    assert (
        triton_io_float_pbtxt_from_manifest({"triton_io_float_dtype": "bf16"})
        == "TYPE_BF16"
    )


def test_generate_configs_minimal_repo(tmp_path):
    if not FIXTURE_MANIFEST.is_file():
        pytest.fail("missing fixture manifest")
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    # Minimal fake repo: only orchestrator is exposed to Triton.
    (tmp_path / "tts_orchestrator" / "1" / "runtime").mkdir(parents=True)
    (tmp_path / "tts_orchestrator_http" / "1").mkdir(parents=True)
    (tmp_path / "tts_orchestrator" / "1" / "runtime" / "model.plan").write_text("stub")
    generate_configs(manifest, tmp_path, "trt", engine_dtype="bf16")
    orch_cfg = (tmp_path / "tts_orchestrator" / "config.pbtxt").read_text()
    orch_http_cfg = (tmp_path / "tts_orchestrator_http" / "config.pbtxt").read_text()
    assert "tts_orchestrator" in orch_cfg
    assert "custom-1.7b" in orch_cfg
    assert 'key: "model_package_dir"' in orch_cfg
    assert "/models/tts_orchestrator/1" in orch_cfg
    assert 'name: "tts_orchestrator_http"' in orch_http_cfg
    assert 'string_value: "tts_orchestrator"' in orch_http_cfg
    assert not (tmp_path / "talker_code2wav_fused" / "config.pbtxt").exists()


def test_generate_configs_uses_package_version(tmp_path):
    if not FIXTURE_MANIFEST.is_file():
        pytest.fail("missing fixture manifest")
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    manifest.setdefault("package", {})["model_package_dir"] = (
        "/models/tts_orchestrator/2"
    )
    manifest.setdefault("orchestrator", {})["model_package_dir"] = (
        "/models/tts_orchestrator/2"
    )

    # Minimal fake repo for version 2: only the versioned orchestrator package is present.
    (tmp_path / "tts_orchestrator" / "2" / "runtime").mkdir(parents=True)
    (tmp_path / "tts_orchestrator_http" / "2").mkdir(parents=True)
    (tmp_path / "tts_orchestrator" / "2" / "runtime" / "model.plan").write_text("stub")

    generate_configs(manifest, tmp_path, "trt", engine_dtype="bf16")

    orch_cfg = (tmp_path / "tts_orchestrator" / "config.pbtxt").read_text()
    orch_http_cfg = (tmp_path / "tts_orchestrator_http" / "config.pbtxt").read_text()
    assert "/models/tts_orchestrator/2" in orch_cfg
    assert 'name: "tts_orchestrator_http"' in orch_http_cfg
    assert 'string_value: "tts_orchestrator"' in orch_http_cfg


# ===========================================================================
#  Assembly & packaging tests
#  (from test_triton_assembly_packaging.py)
# ===========================================================================


def test_custom_trt_package_excludes_verification_and_icl_assets(tmp_path):
    exported_dir = tmp_path / "exported"
    repo_dir = tmp_path / "model_repository"
    _write_custom_export(exported_dir)

    _assemble(exported_dir, repo_dir, "custom-1.7b", "trt")

    package_dir = repo_dir / "tts_orchestrator" / "1"
    runtime_dir = package_dir / "runtime"
    assert not (repo_dir / "triton_manifest.json").exists()
    assert not (repo_dir / "artifact_manifest.json").exists()
    assert (runtime_dir / "model.plan").is_file()
    assert (package_dir / "MODEL_VERSION").read_text(encoding="utf-8") == (
        "zehan@20260601\n"
    )
    assert (package_dir / "MODEL_VERSION").stat().st_mode & 0o222 == 0
    engine_version_path = package_dir / "ENGINE_BUILD_VERSION"
    assert engine_version_path.read_text(encoding="utf-8") == (
        "engine-builder@20260820_v1\n"
    )
    assert engine_version_path.stat().st_mode & 0o222 == 0
    package_info_path = package_dir / "PACKAGE_INFO.json"
    assert json.loads(package_info_path.read_text(encoding="utf-8")) == {
        "package_info_schema_version": 1,
        "packager": "packager-test",
        "packaged_on": "2026-08-20",
    }
    assert package_info_path.stat().st_mode & 0o222 == 0
    assert (package_dir / "artifact_manifest.json").is_file()
    assert (runtime_dir / "artifact_manifest.json").is_file()
    assert not (runtime_dir / "speech_tokenizer_encoder.onnx").exists()
    assert not (package_dir / "tokenizer" / "code2wav_decoder.engine").exists()
    assert not (repo_dir / "speech_tokenizer_encoder").exists()

    manifest = json.loads(
        (runtime_dir / "triton_manifest.json").read_text(encoding="utf-8")
    )
    optional_assets = manifest["package"]["optional_assets"]
    assert optional_assets == {}


def test_assembly_rejects_package_date_with_time_component(tmp_path):
    exported_dir = tmp_path / "exported"
    repo_dir = tmp_path / "model_repository"
    _write_custom_export(exported_dir)

    with pytest.raises(subprocess.CalledProcessError):
        _assemble(
            exported_dir,
            repo_dir,
            "custom-1.7b",
            "trt",
            package_date="2026-08-20T16:20:02+08:00",
        )
