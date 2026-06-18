from __future__ import annotations

import struct
import wave
from collections import OrderedDict

import pytest

from engine.backend.ref_audio_processor import (
    ReferenceAudioProcessor,
    ReferenceAudioFeatures,
    ReferenceAudioSupport,
    _decode_wav_bytes,
    _validate_reference_audio_quality,
)


def test_probe_ignores_onnx_for_trt_standalone(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    speaker_engine = runtime / "speaker_encoder.engine"
    speaker_engine.write_bytes(b"speaker-trt")
    (runtime / "speech_tokenizer_codec_fused.onnx").write_bytes(b"onnx")

    processor = ReferenceAudioProcessor(str(runtime), "base-1.7b")

    assert processor.support.speaker_encoder_path == speaker_engine
    assert processor.support.speech_tokenizer_codec_fused_path is None


def test_probe_accepts_model_plan_layout(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "speaker_encoder").mkdir(parents=True)
    (runtime / "speech_tokenizer_codec_fused").mkdir(parents=True)
    speaker_plan = runtime / "speaker_encoder" / "model.plan"
    codec_plan = runtime / "speech_tokenizer_codec_fused" / "model.plan"
    speaker_plan.write_bytes(b"speaker-plan")
    codec_plan.write_bytes(b"codec-plan")

    processor = ReferenceAudioProcessor(str(runtime), "icl-1.7b")

    assert processor.support.speaker_encoder_path == speaker_plan
    assert processor.support.speech_tokenizer_codec_fused_path == codec_plan


def test_missing_icl_codec_engine_reports_explicit_error(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    speaker_engine = runtime / "speaker_encoder.engine"
    speaker_engine.write_bytes(b"speaker-trt")
    processor = ReferenceAudioProcessor(str(runtime), "base-1.7b")
    processor._support = ReferenceAudioSupport(
        available=True,
        speaker_encoder_path=speaker_engine,
        speech_tokenizer_codec_fused_path=None,
    )

    with pytest.raises(RuntimeError, match="speech_tokenizer_codec_fused_trt_missing"):
        processor.process(b"not-a-wav", require_ref_codec=True)


def test_feature_cache_key_changes_with_artifact_fingerprint(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    speaker_engine = runtime / "speaker_encoder.engine"
    codec_engine = runtime / "speech_tokenizer_codec_fused.engine"
    speaker_engine.write_bytes(b"speaker-v1")
    codec_engine.write_bytes(b"codec-v1")

    processor = ReferenceAudioProcessor(str(runtime), "base-1.7b")
    processor._support = ReferenceAudioSupport(
        available=True,
        speaker_encoder_path=speaker_engine,
        speech_tokenizer_codec_fused_path=codec_engine,
    )
    key_before = processor._feature_cache_key(
        "audio-sha",
        require_ref_codec=True,
        cache_tag="voice_clone_icl",
    )

    codec_engine.write_bytes(b"codec-v2-has-different-size")
    key_after = processor._feature_cache_key(
        "audio-sha",
        require_ref_codec=True,
        cache_tag="voice_clone_icl",
    )

    assert key_before != key_after


def test_reference_audio_quality_validation_errors_and_warns():
    import numpy as np

    with pytest.raises(ValueError, match="ref_audio_too_short"):
        _validate_reference_audio_quality(np.zeros(12000, dtype=np.float32), 0.5)

    with pytest.raises(ValueError, match=r"exceeds 8\.00s"):
        _validate_reference_audio_quality(np.zeros(24_000 * 9, dtype=np.float32), 9.0)

    warnings = _validate_reference_audio_quality(
        np.zeros(24_000 * 2, dtype=np.float32),
        2.0,
    )
    assert any("outside the recommended 3-8s range" in item for item in warnings)
    assert any("near-silent" in item for item in warnings)


def test_reference_audio_quality_uses_dynamic_profile_cap():
    import numpy as np

    with pytest.raises(ValueError, match=r"exceeds 6\.00s"):
        _validate_reference_audio_quality(
            np.zeros(24_000 * 7, dtype=np.float32),
            7.0,
            max_duration_sec=6.0,
        )

    warnings = _validate_reference_audio_quality(
        np.zeros(int(24_000 * 10.5), dtype=np.float32),
        10.5,
        max_duration_sec=12.0,
    )
    assert any("outside the recommended 3-10s range" in item for item in warnings)


def test_codec_profile_max_duration_prefers_trt_profile():
    class _FakeCodecEngine:
        def get_input_profile_max_shape(self, name):
            assert name == "waveform"
            return (1, 1, 144_000)

    assert ReferenceAudioProcessor._infer_codec_max_duration_sec(_FakeCodecEngine()) == 6.0


def test_codec_profile_max_duration_falls_back_to_8s():
    class _FakeCodecEngine:
        def get_input_profile_max_shape(self, name):
            return None

    assert ReferenceAudioProcessor._infer_codec_max_duration_sec(_FakeCodecEngine()) == 8.0


def test_decode_wav_bytes_supports_ieee_float_wav():
    samples = [0.0, 0.5, -0.25, 1.0]
    raw = b"".join(struct.pack("<f", item) for item in samples)
    fmt = struct.pack("<HHIIHH", 3, 1, 24000, 24000 * 4, 4, 32)
    data = (
        b"RIFF"
        + struct.pack("<I", 4 + (8 + len(fmt)) + (8 + len(raw)))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(raw))
        + raw
    )

    audio, sample_rate = _decode_wav_bytes(data)

    assert sample_rate == 24000
    assert audio.tolist() == pytest.approx(samples)


def test_decode_wav_bytes_keeps_pcm_wav_support(tmp_path):
    wav_path = tmp_path / "ref.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(struct.pack("<hhh", 0, 16384, -8192))

    audio, sample_rate = _decode_wav_bytes(wav_path.read_bytes())

    assert sample_rate == 24000
    assert audio.tolist() == pytest.approx([0.0, 0.5, -0.25])


def test_feature_cache_lru_evicts_oldest_and_refreshes_hits(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    processor = ReferenceAudioProcessor(
        str(runtime),
        "base-1.7b",
        cache_enabled=True,
        cache_max_entries=2,
    )
    first = ReferenceAudioFeatures(spk_embedding="first")
    second = ReferenceAudioFeatures(spk_embedding="second")
    third = ReferenceAudioFeatures(spk_embedding="third")

    processor._cache_put("a", first)
    processor._cache_put("b", second)
    assert processor._cache_get("a") is first
    processor._cache_put("c", third)

    assert list(processor._cache.keys()) == ["a", "c"]
    assert processor._cache_get("b") is None


def test_feature_cache_can_be_disabled(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    processor = ReferenceAudioProcessor(
        str(runtime),
        "base-1.7b",
        cache_enabled=False,
        cache_max_entries=2,
    )

    processor._cache_put("a", ReferenceAudioFeatures(spk_embedding="first"))

    assert processor._cache == OrderedDict()
    assert processor._cache_get("a") is None
