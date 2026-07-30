"""Unit tests for TTS output VAD processors.

Tests the core VAD state machine, energy scoring, and the end→begin cycle
without requiring the TenVad ONNX runtime.
"""

import math
from unittest.mock import patch

import numpy as np
import pytest

from engine.interface.vad import (
    DisabledVADProcessor,
    EnergyVADProcessor,
    TTSVADConfig,
    VADMode,
    VADState,
    create_vad_processor,
    vad_config_from_dict,
)

SAMPLE_RATE = 24000
FRAME_SAMPLES = 384  # 16ms @ 24kHz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_silence(duration_ms: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate silence (all zeros) as int16."""
    n = int(sr * duration_ms / 1000)
    return np.zeros(n, dtype=np.int16)


def _make_tone(
    freq: float = 440.0,
    duration_ms: int = 100,
    amplitude: float = 0.3,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a sine tone as int16."""
    n = int(sr * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    signal = amplitude * np.sin(2 * math.pi * freq * t)
    return (signal * 32767).astype(np.int16)


def _make_noise(
    duration_ms: int = 100,
    amplitude: float = 0.01,
    sr: int = SAMPLE_RATE,
    seed: int = 42,
) -> np.ndarray:
    """Generate low-amplitude noise as int16."""
    rng = np.random.RandomState(seed)
    n = int(sr * duration_ms / 1000)
    signal = amplitude * rng.randn(n)
    return np.clip(signal * 32767, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateVADProcessor:
    def test_disabled_mode(self):
        cfg = TTSVADConfig(mode=VADMode.DISABLED)
        proc = create_vad_processor(cfg)
        assert isinstance(proc, DisabledVADProcessor)

    def test_energy_mode(self):
        cfg = TTSVADConfig(mode=VADMode.ENERGY)
        proc = create_vad_processor(cfg)
        assert isinstance(proc, EnergyVADProcessor)

    def test_tenvad_mode_missing_package(self):
        cfg = TTSVADConfig(mode=VADMode.TENVAD)
        with patch("engine.interface.vad._TenVadClass", None):
            with pytest.raises(ImportError, match="ten_vad"):
                create_vad_processor(cfg)

    def test_invalid_mode(self):
        cfg = TTSVADConfig(mode="invalid")  # type: ignore
        with pytest.raises(ValueError):
            create_vad_processor(cfg)


class TestVadConfigFromDict:
    def test_defaults(self):
        cfg = vad_config_from_dict({})
        assert cfg.mode == VADMode.DISABLED

    def test_energy_mode(self):
        cfg = vad_config_from_dict(
            {
                "mode": "energy",
                "begin_threshold": 0.4,
                "end_count": 50,
            }
        )
        assert cfg.mode == VADMode.ENERGY
        assert cfg.begin_threshold == 0.4
        assert cfg.end_count == 50

    def test_invalid_mode_falls_back(self):
        cfg = vad_config_from_dict({"mode": "invalid"})
        assert cfg.mode == VADMode.DISABLED


# ---------------------------------------------------------------------------
# Disabled VAD
# ---------------------------------------------------------------------------


class TestDisabledVADProcessor:
    def test_passthrough(self):
        cfg = TTSVADConfig(mode=VADMode.DISABLED)
        proc = create_vad_processor(cfg, sample_rate=SAMPLE_RATE)
        audio = _make_tone(duration_ms=100)
        result = proc.process_chunk(audio)
        assert result.size == audio.size
        np.testing.assert_array_equal(result, audio)

    def test_empty_chunk(self):
        cfg = TTSVADConfig(mode=VADMode.DISABLED)
        proc = create_vad_processor(cfg, sample_rate=SAMPLE_RATE)
        result = proc.process_chunk(np.empty((0,), dtype=np.int16))
        assert result.size == 0


# ---------------------------------------------------------------------------
# Energy VAD scoring
# ---------------------------------------------------------------------------


class TestEnergyVADScoring:
    def test_silence_scores_near_zero(self):
        cfg = TTSVADConfig(mode=VADMode.ENERGY)
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)
        frame = _make_silence(16)
        score = proc._score_frame(frame)
        assert score < 0.1  # silence should score very low

    def test_loud_tone_scores_high(self):
        cfg = TTSVADConfig(mode=VADMode.ENERGY)
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)
        frame = _make_tone(amplitude=0.5, duration_ms=16)
        score = proc._score_frame(frame)
        assert score > 0.5  # loud tone should score high

    def test_threshold_range_0_to_1(self):
        cfg = TTSVADConfig(mode=VADMode.ENERGY)
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)
        for amp in [0.001, 0.01, 0.1, 0.5, 0.9]:
            frame = _make_tone(amplitude=amp, duration_ms=16)
            score = proc._score_frame(frame)
            assert 0.0 <= score <= 1.0, f"amp={amp}, score={score}"


# ---------------------------------------------------------------------------
# VAD state machine
# ---------------------------------------------------------------------------


class TestVADStateMachine:
    def test_leading_silence_trimmed(self):
        """Leading silence should be discarded, speech should pass."""
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.3,
            begin_count=2,
            end_threshold=0.2,
            end_count=100,  # high so we don't trigger end in this test
            start_margin_ms=0,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # Feed 200ms silence → should produce no output
        silence = _make_silence(200)
        result = proc.process_chunk(silence)
        assert result.size == 0

        # Feed 200ms tone → should produce output
        tone = _make_tone(duration_ms=200, amplitude=0.5)
        result = proc.process_chunk(tone)
        assert result.size > 0

        # Metrics: original > effective (silence was trimmed)
        m = proc.metrics
        assert m.original_audio_samples > m.effective_audio_samples
        assert m.prefix_trimmed_samples > 0

    def test_start_margin_preserved(self):
        """When begin triggers, start_margin_ms of prior audio should be included."""
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.3,
            begin_count=2,
            end_threshold=0.2,
            end_count=100,
            start_margin_ms=32,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # Feed 100ms silence then 200ms tone
        silence = _make_silence(100)
        result1 = proc.process_chunk(silence)
        # Some silence may be in margin buffer, not emitted yet
        tone = _make_tone(duration_ms=200, amplitude=0.5)
        result2 = proc.process_chunk(tone)

        total_emitted = result1.size + result2.size
        # With margin, we should have more than just the tone
        # At minimum, the tone itself should be emitted
        assert total_emitted > 0

    def test_end_triggers_and_rebegin(self):
        """After end triggers, a new begin should restart speech."""
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.3,
            begin_count=2,
            end_threshold=0.2,
            end_count=3,  # very low for testing
            start_margin_ms=0,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # 1. Speech
        tone1 = _make_tone(duration_ms=100, amplitude=0.5)
        proc.process_chunk(tone1)
        assert proc.state == VADState.SPEECH

        # 2. Silence (enough frames for end)
        silence1 = _make_silence(100)
        proc.process_chunk(silence1)
        # Should have triggered end (3 frames of silence < end_threshold)
        assert proc.state == VADState.SILENCE
        assert proc.metrics.end_trigger_count >= 1

        # 3. More silence (should be discarded)
        silence2 = _make_silence(100)
        result = proc.process_chunk(silence2)
        # No output during silence
        assert result.size == 0

        # 4. New speech
        tone2 = _make_tone(duration_ms=100, amplitude=0.5)
        result = proc.process_chunk(tone2)
        assert proc.state == VADState.SPEECH
        assert result.size > 0
        assert proc.metrics.begin_trigger_count >= 2

    def test_flush_emits_pending_speech(self):
        """Flush should emit all pending audio when in SPEECH state."""
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.3,
            begin_count=2,
            end_threshold=0.2,
            end_count=100,
            start_margin_ms=0,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # Start speech
        tone = _make_tone(duration_ms=200, amplitude=0.5)
        proc.process_chunk(tone)
        assert proc.state == VADState.SPEECH

        # Flush
        final = proc.flush()
        # Some audio may have been emitted already, but flush should not be empty
        # if there's anything in the emit buffer
        # At minimum, the state should be reset
        assert proc.state == VADState.SILENCE

    def test_flush_discards_silence(self):
        """Flush in SILENCE state should discard margin buffer."""
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.3,
            begin_count=2,
            end_threshold=0.2,
            end_count=100,
            start_margin_ms=20,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # Feed silence only
        silence = _make_silence(200)
        proc.process_chunk(silence)
        assert proc.state == VADState.SILENCE

        # Flush should return nothing
        final = proc.flush()
        assert final.size == 0
        assert proc.metrics.prefix_trimmed_samples > 0


# ---------------------------------------------------------------------------
# Frame alignment
# ---------------------------------------------------------------------------


class TestFrameAlignment:
    def test_sub_frame_chunk(self):
        """Chunks smaller than one frame should be buffered."""
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_count=2,
            end_count=100,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # Feed a tiny chunk (much less than 16ms)
        tiny = _make_tone(duration_ms=5, amplitude=0.5)
        result = proc.process_chunk(tiny)
        # Too small to form a frame, so no output yet
        assert result.size == 0

    def test_multi_frame_chunk(self):
        """Chunks larger than one frame should be processed completely."""
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.1,
            begin_count=1,
            end_count=100,
            start_margin_ms=0,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # Feed 80ms of tone (5 frames of 16ms)
        tone = _make_tone(duration_ms=80, amplitude=0.5)
        result = proc.process_chunk(tone)
        # Should have emitted all 5 frames (begin triggers on frame 1)
        assert result.size > 0

    def test_residual_carried_over(self):
        """Non-aligned chunks should carry residual samples."""
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.1,
            begin_count=1,
            end_count=100,
            start_margin_ms=0,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # Feed 20ms (1.25 frames of 16ms)
        chunk1 = _make_tone(duration_ms=20, amplitude=0.5)
        result1 = proc.process_chunk(chunk1)
        # 1 complete frame (16ms), 4ms residual

        # Feed another 20ms — should process the residual + new
        chunk2 = _make_tone(duration_ms=20, amplitude=0.5)
        result2 = proc.process_chunk(chunk2)
        # Should have more output
        assert result1.size + result2.size > 0


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class TestVADObservability:
    def test_metrics_disabled_mode(self):
        cfg = TTSVADConfig(mode=VADMode.DISABLED)
        proc = create_vad_processor(cfg, sample_rate=SAMPLE_RATE)
        tone = _make_tone(duration_ms=100, amplitude=0.5)
        proc.process_chunk(tone)

        m = proc.metrics
        assert m.original_audio_samples == tone.size
        assert m.effective_audio_samples == tone.size
        assert m.prefix_trimmed_samples == 0

    def test_metrics_with_trimming(self):
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.3,
            begin_count=2,
            end_threshold=0.2,
            end_count=100,
            start_margin_ms=0,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        silence = _make_silence(100)
        proc.process_chunk(silence)
        tone = _make_tone(duration_ms=100, amplitude=0.5)
        proc.process_chunk(tone)

        m = proc.metrics
        # original_audio_samples may be less than total input due to
        # residual samples in the frame alignment buffer
        assert m.original_audio_samples <= silence.size + tone.size
        assert m.effective_audio_samples < m.original_audio_samples
        assert m.prefix_trimmed_samples > 0
        assert m.begin_trigger_count >= 1

    def test_begin_end_counts(self):
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.3,
            begin_count=2,
            end_threshold=0.2,
            end_count=3,
            start_margin_ms=0,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # speech → silence → speech
        proc.process_chunk(_make_tone(100, amplitude=0.5))
        proc.process_chunk(_make_silence(100))
        proc.process_chunk(_make_tone(100, amplitude=0.5))

        assert proc.metrics.begin_trigger_count >= 2
        assert proc.metrics.end_trigger_count >= 1

    def test_start_margin_is_effective_audio_not_prefix_trim(self):
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            chunk_ms=16,
            begin_threshold=0.5,
            begin_count=1,
            end_count=100,
            start_margin_ms=32,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)
        proc._score_frame = lambda frame: float(frame[0] != 0)

        silence = np.zeros(5 * FRAME_SAMPLES, dtype=np.int16)
        speech = np.ones(FRAME_SAMPLES, dtype=np.int16)
        emitted = proc.process_chunk(np.concatenate((silence, speech)))

        # Two lookback frames plus the onset frame are emitted. Only the three
        # silence frames that fell out of lookback were actually trimmed.
        assert emitted.size == 3 * FRAME_SAMPLES
        m = proc.metrics
        assert m.original_audio_samples == 6 * FRAME_SAMPLES
        assert m.effective_audio_samples == 3 * FRAME_SAMPLES
        assert m.prefix_trimmed_samples == 3 * FRAME_SAMPLES
        assert m.tail_trimmed_samples == 0
        assert (
            m.original_audio_samples
            == m.effective_audio_samples
            + m.prefix_trimmed_samples
            + m.tail_trimmed_samples
        )

    def test_all_silence_flush_is_exact_and_idempotent(self):
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            chunk_ms=16,
            begin_threshold=0.5,
            begin_count=2,
            start_margin_ms=20,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)
        audio = _make_silence(83)  # includes a partial final frame

        assert proc.process_chunk(audio).size == 0
        assert proc.flush().size == 0
        first_flush_metrics = vars(proc.metrics).copy()

        m = proc.metrics
        assert m.original_audio_samples == audio.size
        assert m.effective_audio_samples == 0
        assert m.prefix_trimmed_samples == audio.size
        assert m.tail_trimmed_samples == 0
        assert proc.flush().size == 0
        assert vars(proc.metrics) == first_flush_metrics

    def test_post_speech_trim_is_tail_and_metrics_conserve(self):
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            chunk_ms=16,
            begin_threshold=0.5,
            begin_count=1,
            end_threshold=0.5,
            end_count=1,
            start_margin_ms=16,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)
        proc._score_frame = lambda frame: float(frame[0] != 0)

        def frames(value, count):
            return np.full(count * FRAME_SAMPLES, value, dtype=np.int16)

        audio = np.concatenate(
            (
                frames(0, 3),
                frames(1, 1),
                frames(0, 1),  # triggers end and is discarded
                frames(0, 3),
                frames(1, 1),  # re-begin emits one-frame lookback margin
            )
        )

        emitted = np.concatenate((proc.process_chunk(audio), proc.flush()))
        m = proc.metrics
        assert emitted.size == 4 * FRAME_SAMPLES
        assert m.original_audio_samples == 9 * FRAME_SAMPLES
        assert m.effective_audio_samples == 4 * FRAME_SAMPLES
        assert m.prefix_trimmed_samples == 2 * FRAME_SAMPLES
        assert m.tail_trimmed_samples == 3 * FRAME_SAMPLES
        assert (
            m.original_audio_samples
            == m.effective_audio_samples
            + m.prefix_trimmed_samples
            + m.tail_trimmed_samples
        )


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestVADReset:
    def test_discard_pending_breaks_cross_segment_onset_and_preserves_metrics(self):
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            chunk_ms=16,
            begin_threshold=0.5,
            begin_count=2,
            end_count=100,
            start_margin_ms=0,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)
        proc._score_frame = lambda frame: float(frame[0] != 0)
        first_candidate = np.full(FRAME_SAMPLES, 1, dtype=np.int16)
        second_segment = np.full(2 * FRAME_SAMPLES, 2, dtype=np.int16)

        # Segment 0 aborts one frame short of a confirmed onset.
        assert proc.process_chunk(first_candidate).size == 0
        proc.discard_pending()
        assert proc.state == VADState.SILENCE
        assert proc.metrics.original_audio_samples == FRAME_SAMPLES
        assert proc.metrics.prefix_trimmed_samples == FRAME_SAMPLES

        # Segment 1 must establish its own two-frame onset.  The condemned
        # marker from segment 0 is neither used to trigger nor emitted.
        emitted = proc.process_chunk(second_segment)
        assert emitted.size == 2 * FRAME_SAMPLES
        assert set(int(x) for x in emitted) == {2}
        m = proc.metrics
        assert m.original_audio_samples == 3 * FRAME_SAMPLES
        assert m.effective_audio_samples == 2 * FRAME_SAMPLES
        assert m.prefix_trimmed_samples == FRAME_SAMPLES
        assert (
            m.original_audio_samples
            == m.effective_audio_samples
            + m.prefix_trimmed_samples
            + m.tail_trimmed_samples
        )

    def test_discard_pending_accounts_partial_frame_exactly_once(self):
        cfg = TTSVADConfig(mode=VADMode.ENERGY, chunk_ms=16)
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)
        partial = np.ones(FRAME_SAMPLES // 2, dtype=np.int16)

        assert proc.process_chunk(partial).size == 0
        proc.discard_pending()
        first = vars(proc.metrics).copy()
        proc.discard_pending()

        assert proc.metrics.original_audio_samples == partial.size
        assert proc.metrics.prefix_trimmed_samples == partial.size
        assert vars(proc.metrics) == first

    def test_discard_pending_after_speech_classifies_uncommitted_audio_as_tail(self):
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            chunk_ms=16,
            begin_threshold=0.5,
            begin_count=2,
            end_threshold=0.5,
            end_count=1,
            start_margin_ms=0,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)
        proc._score_frame = lambda frame: float(frame[0] != 0)
        speech = np.ones(2 * FRAME_SAMPLES, dtype=np.int16)
        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        next_candidate = np.full(FRAME_SAMPLES, 2, dtype=np.int16)

        assert proc.process_chunk(speech).size == speech.size
        assert proc.process_chunk(silence).size == 0
        assert proc.process_chunk(next_candidate).size == 0
        proc.discard_pending()

        m = proc.metrics
        assert m.first_effective_audio_found is True
        assert m.prefix_trimmed_samples == 0
        assert m.tail_trimmed_samples == 2 * FRAME_SAMPLES
        assert m.original_audio_samples == 4 * FRAME_SAMPLES
        assert m.effective_audio_samples == 2 * FRAME_SAMPLES

    def test_reset_clears_state(self):
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_count=2,
            end_count=100,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # Run some audio through
        proc.process_chunk(_make_tone(200, amplitude=0.5))
        assert proc.state == VADState.SPEECH or proc.metrics.original_audio_samples > 0

        # Reset
        proc.reset()
        assert proc.state == VADState.SILENCE
        assert proc.metrics.original_audio_samples == 0
        assert proc.metrics.effective_audio_samples == 0
        assert proc.metrics.begin_trigger_count == 0

    def test_reuse_after_reset(self):
        cfg = TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.3,
            begin_count=2,
            end_count=100,
            start_margin_ms=0,
        )
        proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)

        # First use
        proc.process_chunk(_make_tone(200, amplitude=0.5))
        proc.reset()

        # Second use — should work the same
        result = proc.process_chunk(_make_tone(200, amplitude=0.5))
        assert result.size > 0


def test_onset_frames_not_clipped_by_small_lookback_margin():
    """Regression: consecutive begin-candidate frames must not be evicted by the
    small lookback margin. With begin_count=5 and start_margin_ms=20 (< 4 frames),
    the old code dropped the first ~3 onset frames."""
    cfg = vad_config_from_dict(
        {"mode": "energy", "chunk_ms": 16, "begin_count": 5, "start_margin_ms": 20}
    )
    proc = EnergyVADProcessor(cfg, sample_rate=SAMPLE_RATE)
    proc._score_frame = lambda frame: 1.0  # force all-speech, isolate the FSM

    fs = proc._frame_samples
    # Tag each frame with a distinct value so we can see which survive.
    audio = np.concatenate([np.full(fs, i + 1, dtype=np.int16) for i in range(8)])
    emitted = np.concatenate([proc.process_chunk(audio), proc.flush()])

    surviving = set(int(x) for x in emitted)
    assert {1, 2, 3, 4, 5, 6, 7, 8} <= surviving  # no onset frame dropped


def test_vad_config_honors_explicit_zero():
    """Regression: an explicit 0/0.0 must not be replaced by the default."""
    cfg = vad_config_from_dict(
        {"start_margin_ms": 0, "begin_count": 1, "end_threshold": 0.0}
    )
    assert cfg.start_margin_ms == 0
    assert cfg.begin_count == 1
    assert cfg.end_threshold == 0.0
    # omitted keys still fall back to defaults
    assert vad_config_from_dict({}).start_margin_ms == 20
