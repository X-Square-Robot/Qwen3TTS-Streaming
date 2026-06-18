"""Regression tests for Triton orchestrator sampling defaults."""

from __future__ import annotations

import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ORCH_DIR = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if str(ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(ORCH_DIR))

sys.modules.setdefault(
    "triton_python_backend_utils",
    types.SimpleNamespace(TRITONSERVER_RESPONSE_COMPLETE_FINAL=1),
)

from model import _env_string, _param_string, _parse_bool  # noqa: E402
from engine.config import SamplingConfig  # noqa: E402


def _resolved_do_sample(params: dict) -> bool:
    return _parse_bool(
        _param_string(
            params,
            "do_sample",
            _env_string("ENGINE_SAMPLING_DO_SAMPLE", "DO_SAMPLE", default="false"),
        ),
        False,
    )


def test_triton_orchestrator_do_sample_defaults_to_greedy(monkeypatch):
    monkeypatch.delenv("ENGINE_SAMPLING_DO_SAMPLE", raising=False)
    monkeypatch.delenv("DO_SAMPLE", raising=False)

    assert _resolved_do_sample({}) is False


def test_engine_sampling_config_defaults_to_greedy():
    assert SamplingConfig().do_sample is False


def test_triton_orchestrator_prefers_engine_sampling_env(monkeypatch):
    monkeypatch.setenv("ENGINE_SAMPLING_DO_SAMPLE", "false")
    monkeypatch.setenv("DO_SAMPLE", "1")

    assert _resolved_do_sample({}) is False


def test_triton_orchestrator_model_config_overrides_env(monkeypatch):
    monkeypatch.setenv("ENGINE_SAMPLING_DO_SAMPLE", "false")

    assert _resolved_do_sample({"do_sample": {"string_value": "true"}}) is True
