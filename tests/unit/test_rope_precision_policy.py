from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PY = REPO_ROOT / "scripts" / "python"
sys.path.insert(0, str(SCRIPTS_PY))

from ensure_rope_precision import (  # noqa: E402
    apply_rope_precision_policy,
    env_truthy,
    update_manifest_file,
)


def test_rope_fix_is_enabled_by_default_policy():
    manifest = {
        "engine_dtype": "bf16",
        "engine_profile": {"engine_dtype": "bf16"},
        "cp_precision": "bf16",
    }

    apply_rope_precision_policy(manifest, enabled=True)

    assert manifest["rope_precision"] == "fp32"
    assert manifest["engine_profile"]["rope_precision"] == "fp32"
    assert manifest["cp_precision"] == "bf16"


def test_hidden_debug_disable_removes_stale_fix_marker():
    manifest = {
        "rope_precision": "fp32",
        "engine_profile": {"rope_precision": "fp32", "max_seq_len": 512},
        "engine_dtype": "bf16",
    }

    apply_rope_precision_policy(manifest, enabled=False)

    assert "rope_precision" not in manifest
    assert "rope_precision" not in manifest["engine_profile"]
    assert manifest["engine_profile"]["max_seq_len"] == 512


def test_debug_environment_values_are_intentionally_narrow():
    assert all(env_truthy(value) for value in ("1", "true", "YES", "on"))
    assert not any(env_truthy(value) for value in (None, "", "0", "false", "debug"))


def test_manifest_file_uses_hidden_environment_switch(tmp_path, monkeypatch):
    manifest_path = tmp_path / "triton_manifest.json"
    manifest_path.write_text(
        json.dumps({"engine_profile": {"max_seq_len": 512}}), encoding="utf-8"
    )

    monkeypatch.delenv("QWEN3_DISABLE_ROPE_FP32", raising=False)
    update_manifest_file(manifest_path, disable=False)
    enabled = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert enabled["rope_precision"] == "fp32"
    assert enabled["engine_profile"]["rope_precision"] == "fp32"

    monkeypatch.setenv("QWEN3_DISABLE_ROPE_FP32", "1")
    update_manifest_file(manifest_path, disable=True)
    disabled = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "rope_precision" not in disabled
    assert "rope_precision" not in disabled["engine_profile"]
