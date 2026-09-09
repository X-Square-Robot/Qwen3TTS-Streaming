from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import engine.server as server_module
from engine.backend.ref_audio_processor import ReferenceAudioSupport
from engine.config import EngineConfig, EngineProfileConfig, ModelArchConfig
from engine.core.types import SessionConfig
from engine.server import TTSEngine
from engine.runtime.model_version import ModelVersionError
from qwen3tts_protocol.protocol import SUPPORTED_VAD_STRATEGIES


@dataclass
class _StubSupportProbe:
    support: ReferenceAudioSupport


class _StubFrontend:
    def __init__(self):
        self.calls = []

    async def create_session(
        self, session_id: str, *, config, on_audio=None, on_done=None, on_event=None
    ):
        self.calls.append((session_id, config, on_audio, on_done, on_event))
        return {"session_id": session_id, "config": config}


class _StubReferenceProcessor:
    def __init__(self, support: ReferenceAudioSupport):
        self.support = support
        self.calls = []

    def process(self, ref_audio: bytes, *, require_ref_codec: bool, cache_tag: str):
        self.calls.append((ref_audio, require_ref_codec, cache_tag))
        return SimpleNamespace(duration_sec=1.25)


def _available_ref_support(*, codec: bool = True) -> ReferenceAudioSupport:
    return ReferenceAudioSupport(
        available=True,
        speaker_encoder_path=Path("speaker_encoder.engine"),
        speech_tokenizer_codec_fused_path=(
            Path("speech_tokenizer_codec_fused.engine") if codec else None
        ),
        ref_codec_reason=(
            "" if codec else "speech_tokenizer_codec_fused_trt_missing: missing codec"
        ),
    )


def test_validate_session_config_rejects_unsupported_task_type():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="custom-1.7b",
            tts_model_type="custom_voice",
            supported_task_types=("custom_voice",),
        )
    )

    with pytest.raises(ValueError, match="has loaded model_type 'custom_voice'"):
        engine._validate_session_config(
            SessionConfig(task_type="voice_clone", ref_audio=b"x")
        )


def test_start_rejects_missing_model_version_before_loading_runtime(tmp_path):
    cfg = EngineConfig()
    cfg.paths.model_package_dir = str(tmp_path)
    engine = TTSEngine(config=cfg)

    with pytest.raises(ModelVersionError, match="MODEL_VERSION"):
        asyncio.run(engine.start())

    assert engine._tokenizer is None
    assert engine._executor is None


def test_start_rejects_missing_engine_build_version_before_loading_runtime(tmp_path):
    model_version = tmp_path / "MODEL_VERSION"
    model_version.write_text("zehan@20260818\n", encoding="utf-8")
    model_version.chmod(0o444)
    cfg = EngineConfig()
    cfg.paths.model_package_dir = str(tmp_path)
    engine = TTSEngine(config=cfg)

    with pytest.raises(ModelVersionError, match="ENGINE_BUILD_VERSION"):
        asyncio.run(engine.start())

    assert engine._tokenizer is None
    assert engine._executor is None


def test_validate_session_config_uses_default_ref_audio_for_base(monkeypatch, tmp_path):
    ref_path = tmp_path / "base_ref.wav"
    ref_path.write_bytes(b"default-wav")
    monkeypatch.setattr(
        server_module,
        "_DEFAULT_BASE_REF_AUDIO_CANDIDATES",
        (str(ref_path),),
    )
    monkeypatch.setattr(
        server_module,
        "_DEFAULT_BASE_REF_TEXT",
        "默认参考文本",
    )

    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="base")
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.ref_audio == b"default-wav"
    assert config.ref_text == "默认参考文本"
    assert config.x_vector_only is False
    assert config.ref_source == "default"
    assert config.ref_id == "default"


def test_prime_configured_reference_cache_processes_default_and_aliases(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first_txt = tmp_path / "first.txt"
    second_txt = tmp_path / "second.txt"
    first.write_bytes(b"first-wav")
    second.write_bytes(b"second-wav")
    first_txt.write_text("first ref", encoding="utf-8")
    second_txt.write_text("second ref", encoding="utf-8")

    cfg = EngineConfig()
    cfg.references.default = "first"
    cfg.references.entries = {
        "first": {
            "audio_path": str(first),
            "ref_text_path": str(first_txt),
        },
        "second": {
            "audio_path": str(second),
            "ref_text_path": str(second_txt),
        },
    }
    engine = TTSEngine(
        config=cfg,
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        ),
    )
    processor = _StubReferenceProcessor(_available_ref_support())
    engine._ref_audio_processor = processor

    engine._prime_configured_reference_cache()

    assert processor.calls == [
        (b"first-wav", True, "voice_clone_icl"),
        (b"second-wav", True, "voice_clone_icl"),
    ]


def test_validate_session_config_requires_ref_audio_when_default_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        server_module,
        "_DEFAULT_BASE_REF_AUDIO_CANDIDATES",
        (str(tmp_path / "missing.wav"),),
    )

    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    with pytest.raises(ValueError, match="no default Base reference audio"):
        engine._validate_session_config(SessionConfig(task_type="base"))


def test_validate_session_config_reports_unavailable_ref_audio_processor():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(
        ReferenceAudioSupport(
            available=False,
            reason="speaker_encoder_trt_missing: missing speaker_encoder TensorRT engine",
        )
    )

    with pytest.raises(
        ValueError, match="voice_clone is not available in standalone mode"
    ):
        engine._validate_session_config(SessionConfig(task_type="base", ref_audio=b"x"))


def test_start_session_validates_before_delegating():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="custom-1.7b",
            tts_model_type="custom_voice",
            supported_task_types=("custom_voice",),
        )
    )
    frontend = _StubFrontend()
    engine._frontend = frontend

    config = SessionConfig(task_type="custom_voice", speaker="Serena")
    result = asyncio.run(engine.start_session("sid-1", config=config))

    assert result["session_id"] == "sid-1"
    assert len(frontend.calls) == 1
    assert frontend.calls[0][1] is config


def test_start_session_offloads_only_when_ref_audio_needs_preprocessing(monkeypatch):
    """Contract: validation runs inline on the loop for cheap configs; the
    to_thread hop is reserved for configs whose reference audio actually
    needs feature extraction (ref_audio present, no precomputed embedding).
    The unconditional hop cost two event-loop requeues per session open and
    was removed after it dominated nothing but added latency."""
    import threading

    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="custom-1.7b",
            tts_model_type="custom_voice",
            supported_task_types=("custom_voice",),
        )
    )
    engine._frontend = _StubFrontend()

    seen_threads: list[int] = []

    def _recording_validate(config):
        seen_threads.append(threading.get_ident())
        if config.ref_audio:
            time.sleep(0.05)

    monkeypatch.setattr(
        engine, "_validate_and_prepare_session_config", _recording_validate
    )

    async def _run():
        loop_thread = threading.get_ident()

        # Cheap config (no ref audio): validation runs inline on the loop.
        await engine.start_session(
            "sid-inline",
            config=SessionConfig(task_type="custom_voice", speaker="Serena"),
        )
        assert seen_threads[-1] == loop_thread

        # Ref-audio config: heavy preprocessing goes off-loop and the loop
        # stays responsive while it blocks.
        task = asyncio.create_task(
            engine.start_session(
                "sid-offload",
                config=SessionConfig(
                    task_type="custom_voice",
                    speaker="Serena",
                    ref_audio=b"\x00\x01",
                ),
            )
        )
        ticks = []

        async def _tick():
            await asyncio.sleep(0.01)
            ticks.append("tick")

        await asyncio.wait_for(_tick(), timeout=0.03)
        assert ticks == ["tick"]
        await task
        assert seen_threads[-1] != loop_thread

    asyncio.run(_run())


def test_validate_session_config_binds_empty_task_type_to_loaded_model():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="custom-1.7b",
            tts_model_type="custom_voice",
            supported_task_types=("custom_voice",),
        )
    )

    config = SessionConfig(task_type="", speaker="Serena")
    engine._validate_session_config(config)

    assert config.task_type == "custom_voice"


def test_validate_session_config_maps_base_model_to_internal_voice_clone():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="", ref_audio=b"wav")
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.x_vector_only is True


def test_validate_session_config_maps_base_ref_text_to_icl():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="base", ref_audio=b"wav", ref_text="你好")
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.x_vector_only is False
    assert config.speaker is None
    assert config.ref_source == "explicit"


def test_validate_session_config_explicit_reference_ignores_speaker_alias():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(
        task_type="base",
        speaker="vivian",
        ref_audio=b"wav",
        ref_text="你好",
    )
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.speaker is None
    assert config.ref_source == "explicit"
    assert config.ref_id == "vivian"
    assert config.x_vector_only is False


def test_validate_session_config_maps_icl_model_to_internal_voice_clone():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="icl-1.7b",
            tts_model_type="icl",
            supported_task_types=("icl",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="", ref_audio=b"wav", ref_text="你好")
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.x_vector_only is False


def test_validate_session_config_accepts_public_voice_clone_for_icl_model():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="icl-1.7b",
            tts_model_type="icl",
            supported_task_types=("icl",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="voice_clone", ref_audio=b"wav", ref_text="你好")
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.x_vector_only is False


def test_validate_session_config_requires_ref_text_for_icl():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="icl-1.7b",
            tts_model_type="icl",
            supported_task_types=("icl",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    with pytest.raises(
        ValueError, match="ref_text is required for loaded model_type 'icl'"
    ):
        engine._validate_session_config(
            SessionConfig(task_type="icl", ref_audio=b"wav")
        )


def test_validate_session_config_uses_default_ref_audio_for_icl(monkeypatch, tmp_path):
    ref_path = tmp_path / "base_ref.wav"
    ref_path.write_bytes(b"default-wav")
    monkeypatch.setattr(
        server_module,
        "_DEFAULT_BASE_REF_AUDIO_CANDIDATES",
        (str(ref_path),),
    )
    monkeypatch.setattr(
        server_module,
        "_DEFAULT_BASE_REF_TEXT",
        "默认参考文本",
    )

    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="icl-1.7b",
            tts_model_type="icl",
            supported_task_types=("icl",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="icl")
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.ref_audio == b"default-wav"
    assert config.ref_text == "默认参考文本"
    assert config.x_vector_only is False
    assert config.ref_source == "default"


def test_validate_session_config_resolves_base_speaker_as_reference_alias(tmp_path):
    ref_path = tmp_path / "vivian.wav"
    ref_path.write_bytes(b"alias-wav")
    cfg = EngineConfig()
    cfg.references.entries = {
        "Vivian": {
            "audio_path": str(ref_path),
            "ref_text": "别名参考文本",
        }
    }

    engine = TTSEngine(
        config=cfg,
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        ),
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="base", speaker="vivian")
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.speaker is None
    assert config.ref_id == "Vivian"
    assert config.ref_source == "registry"
    assert config.ref_audio == b"alias-wav"
    assert config.ref_text == "别名参考文本"


def test_validate_session_config_applies_registry_language_for_auto(tmp_path):
    ref_path = tmp_path / "vivian.wav"
    ref_path.write_bytes(b"alias-wav")
    cfg = EngineConfig()
    cfg.references.entries = {
        "Vivian": {
            "audio_path": str(ref_path),
            "ref_text": "别名参考文本",
            "language": "zh",
        }
    }

    engine = TTSEngine(
        config=cfg,
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        ),
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="base", speaker="vivian", language="auto")
    engine._validate_session_config(config)

    assert config.language == "zh"


def test_validate_session_config_keeps_explicit_language_over_registry(tmp_path):
    ref_path = tmp_path / "vivian.wav"
    ref_path.write_bytes(b"alias-wav")
    cfg = EngineConfig()
    cfg.references.entries = {
        "Vivian": {
            "audio_path": str(ref_path),
            "ref_text": "别名参考文本",
            "language": "zh",
        }
    }

    engine = TTSEngine(
        config=cfg,
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        ),
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="base", speaker="vivian", language="en")
    engine._validate_session_config(config)

    assert config.language == "en"


def test_validate_session_config_explicit_reference_does_not_apply_registry_language(
    tmp_path,
):
    ref_path = tmp_path / "vivian.wav"
    ref_path.write_bytes(b"alias-wav")
    cfg = EngineConfig()
    cfg.references.entries = {
        "Vivian": {
            "audio_path": str(ref_path),
            "ref_text": "别名参考文本",
            "language": "zh",
        }
    }

    engine = TTSEngine(
        config=cfg,
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        ),
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(
        task_type="base",
        speaker="vivian",
        ref_audio=b"wav",
        ref_text="你好",
        language="auto",
    )
    engine._validate_session_config(config)

    assert config.language == "auto"
    assert config.ref_source == "explicit"


def test_validate_session_config_uses_configured_default_reference(tmp_path):
    ref_path = tmp_path / "default.wav"
    ref_path.write_bytes(b"default-registry-wav")
    cfg = EngineConfig()
    cfg.references.default = "Vivian"
    cfg.references.entries = {
        "Vivian": {
            "audio_path": str(ref_path),
            "ref_text": "默认 registry 参考文本",
        }
    }

    engine = TTSEngine(
        config=cfg,
        model_arch=ModelArchConfig(
            variant="icl-1.7b",
            tts_model_type="icl",
            supported_task_types=("icl",),
        ),
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="icl")
    engine._validate_session_config(config)

    assert config.task_type == "voice_clone"
    assert config.ref_id == "Vivian"
    assert config.ref_source == "default"
    assert config.ref_audio == b"default-registry-wav"
    assert config.ref_text == "默认 registry 参考文本"


def test_validate_session_config_reads_reference_assets_from_model_package(tmp_path):
    package_dir = tmp_path / "tts_orchestrator" / "1"
    ref_dir = package_dir / "resources" / "speakers" / "vivian"
    ref_dir.mkdir(parents=True)
    (ref_dir / "ref.wav").write_bytes(b"packaged-wav")
    (ref_dir / "ref.txt").write_text("packaged reference text\n", encoding="utf-8")

    cfg = EngineConfig()
    cfg.paths.model_package_dir = str(package_dir)
    cfg.references.entries = {
        "vivian": {
            "audio_path": "resources/speakers/vivian/ref.wav",
            "ref_text_path": "resources/speakers/vivian/ref.txt",
        }
    }

    engine = TTSEngine(
        config=cfg,
        model_arch=ModelArchConfig(
            variant="icl-1.7b",
            tts_model_type="icl",
            supported_task_types=("icl",),
        ),
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    config = SessionConfig(task_type="icl", speaker="vivian")
    engine._validate_session_config(config)

    assert config.ref_audio == b"packaged-wav"
    assert config.ref_text == "packaged reference text"
    assert config.ref_source == "registry"
    assert config.ref_id == "vivian"


def test_validate_session_config_missing_reference_alias_errors():
    cfg = EngineConfig()
    cfg.references.entries = {
        "default": {
            "audio_path": "/tmp/does-not-matter.wav",
            "ref_text": "默认文本",
        }
    }
    engine = TTSEngine(
        config=cfg,
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        ),
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support())

    with pytest.raises(ValueError, match="reference_not_found"):
        engine._validate_session_config(
            SessionConfig(task_type="base", speaker="missing")
        )


def test_validate_session_config_requires_instruct_for_voice_design():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="voice-design-1.7b",
            tts_model_type="voice_design",
            supported_task_types=("voice_design",),
        )
    )

    with pytest.raises(
        ValueError, match="instruct is required for loaded model_type 'voice_design'"
    ):
        engine._validate_session_config(SessionConfig(task_type="voice_design"))


def test_health_stats_exposes_loaded_model_binding():
    class _StubEngineLoop:
        def health_stats(self):
            return {"running": True}

    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="custom-1.7b",
            tts_model_type="custom_voice",
            supported_task_types=("custom_voice",),
        )
    )
    engine._engine_loop = _StubEngineLoop()

    stats = engine.health_stats()
    assert stats["variant"] == "custom-1.7b"
    assert stats["loaded_model_type"] == "custom_voice"
    assert stats["declared_supported_task_types"] == ["custom_voice"]


def test_describe_capabilities_reports_loaded_model_contract():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="custom-1.7b",
            tts_model_type="custom_voice",
            supported_task_types=("custom_voice",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(
        ReferenceAudioSupport(
            available=False,
            reason="variant 'custom-1.7b' is not a base model; voice_clone is unsupported",
        )
    )
    engine._supported_speakers = ("serena", "vivian")
    engine._supported_languages = ("auto", "chinese", "english")

    cap = engine.describe_capabilities()
    assert cap["variant"] == "custom-1.7b"
    assert cap["loaded_model_type"] == "custom_voice"
    assert cap["declared_supported_task_types"] == ["custom_voice"]
    assert cap["supported_input_modes"] == [
        "token",
        "clause",
        "long_segment",
        "full_text",
    ]
    assert cap["supported_group_policies"] == ["none", "auto"]
    assert cap["supported_speakers"] == ["serena", "vivian"]
    assert cap["supported_languages"] == ["auto", "chinese", "english"]
    assert cap["speech_state"]["supported"] is False
    assert cap["speech_state"]["reason"] == "adapter_disabled"
    assert cap["ref_audio_available"] is False
    assert cap["protocol_version"] == "tts-session-v2alpha1"
    assert "vad_policy" in cap["supported_output_policy_features"]
    assert set(cap["supported_vad_strategies"]) <= set(SUPPORTED_VAD_STRATEGIES)
    assert {"disabled", "energy"} <= set(cap["supported_vad_strategies"])
    assert "server_ttft_ms" in cap["supported_timing_fields"]


def test_describe_capabilities_reports_missing_icl_codec():
    engine = TTSEngine(
        model_arch=ModelArchConfig(
            variant="base-1.7b",
            tts_model_type="base",
            supported_task_types=("base",),
        )
    )
    engine._ref_audio_processor = _StubSupportProbe(_available_ref_support(codec=False))

    cap = engine.describe_capabilities()

    assert cap["speaker_encoder_available"] is True
    assert cap["ref_codec_available"] is False
    assert cap["icl_available"] is False
    assert cap["ref_audio_available"] is False
    assert "speech_tokenizer_codec_fused_trt_missing" in cap["ref_audio_reason"]


def test_runtime_profile_rejects_oversized_batch():
    arch = ModelArchConfig(
        variant="custom-1.7b",
        engine_profile=EngineProfileConfig(max_batch_size=16, max_seq_len=512),
    )

    with pytest.raises(
        ValueError, match="runtime max_batch_size=32 exceeds engine profile"
    ):
        TTSEngine(
            config=EngineConfig(), model_arch=arch, max_batch_size=32, max_seq_len=512
        )


def test_runtime_profile_rejects_oversized_seq_len():
    arch = ModelArchConfig(
        variant="custom-1.7b",
        engine_profile=EngineProfileConfig(max_batch_size=64, max_seq_len=512),
    )

    with pytest.raises(
        ValueError, match="runtime max_seq_len=1024 exceeds engine profile"
    ):
        TTSEngine(
            config=EngineConfig(), model_arch=arch, max_batch_size=32, max_seq_len=1024
        )
