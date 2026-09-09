"""Capability discovery must consume the executor's property-based contract."""

from types import SimpleNamespace

import pytest

import engine.server as server_module
from engine.backend.executor import Executor
from engine.backend.speech_state import NullSpeechStateAdapter
from engine.runtime.release_gate import ReleaseGate
from engine.core.speech_state_bundle import SpeechStateBundleValidation
from engine.server import TTSEngine


def _engine(*, graph_enabled=True, evidence_verified=False):
    executor = Executor.__new__(Executor)
    executor._manifest = {
        "native_cursor": {
            "enabled": True,
            "progress_available": True,
            "cursor_head_sha256": "a" * 64,
            "cursor_vocab_sha256": "a" * 64,
        }
    }
    executor._cursor_enabled = graph_enabled
    executor._release_gate = (
        ReleaseGate(True, False, "verified", "release_evidence_missing")
        if evidence_verified
        else ReleaseGate.disabled()
    )
    executor._speech_state_adapter = NullSpeechStateAdapter()
    executor._speech_state_bundle_validation = SpeechStateBundleValidation(
        False, "missing_speech_state"
    )
    engine = TTSEngine()
    engine._executor = executor
    engine._engine_loop = SimpleNamespace(health_stats=lambda: {"running": True})
    return engine


@pytest.mark.parametrize("surface", ["describe_capabilities", "health_stats"])
@pytest.mark.parametrize("evidence_verified", [False, True])
def test_server_reads_executor_capability_properties(surface, evidence_verified):
    engine = _engine(evidence_verified=evidence_verified)

    capability = getattr(engine, surface)()

    assert capability["native_cursor"]["enabled"] is True
    assert capability["native_cursor"]["progress_available"] is False
    if evidence_verified:
        assert capability["native_cursor"]["reason"] == "cursor_labelizer_not_loaded"
    else:
        assert capability["native_cursor"]["reason"] == "release_evidence_missing"
    assert capability["speech_state"]["supported"] is False
    assert capability["speech_state"]["reason"] == "missing_speech_state"


def test_server_standard_plan_preserves_loaded_bundle_rejection_reason():
    engine = _engine(graph_enabled=False)

    capability = engine.describe_capabilities()

    assert capability["native_cursor"] == {"enabled": False}
    assert capability["speech_state"]["reason"] == "missing_speech_state"


def test_server_capability_reader_keeps_method_based_executor_compatibility():
    engine = TTSEngine()
    native = {"enabled": True, "progress_available": False, "reason": "not_ready"}
    engine._executor = SimpleNamespace(
        native_cursor_capability=lambda: native,
        speech_state_capability_reason=lambda: "model_weights_hash_mismatch",
    )

    capability = engine.describe_capabilities()

    assert capability["native_cursor"] == native
    assert capability["speech_state"]["reason"] == "model_weights_hash_mismatch"
    capability["native_cursor"]["enabled"] = False
    assert native["enabled"] is True


def test_server_capability_reader_observes_runtime_gate_changes():
    engine = _engine(evidence_verified=True)
    engine._cursor_plan_adapter_factory = lambda: object()
    assert engine.describe_capabilities()["native_cursor"]["progress_available"] is True

    engine._executor._release_gate = ReleaseGate.disabled("runtime_evidence_rejected")

    capability = engine.describe_capabilities()
    assert capability["native_cursor"]["progress_available"] is False
    assert capability["native_cursor"]["reason"] == "runtime_evidence_rejected"


def test_server_builds_cursor_plan_factory_only_after_labelizer_load(
    tmp_path, monkeypatch
):
    engine = _engine(evidence_verified=True)
    head = tmp_path / "qwen3_tts_12hz_la1_seed0.pt"
    head.write_bytes(b"head")
    engine._executor._cursor_head_path = head
    seen = {}

    class _Labelizer:
        def __call__(self, text):
            return (len(text),)

    def load(path, *, expected_vocab_sha256=None):
        seen["path"] = path
        seen["sha"] = expected_vocab_sha256
        return _Labelizer()

    monkeypatch.setattr(server_module, "load_native_cursor_labelizer", load)

    factory = engine._build_cursor_plan_adapter_factory()

    assert factory is not None
    assert seen == {"path": head, "sha": "a" * 64}
    adapter = factory()
    assert adapter._labelize("abc") == (3,)
