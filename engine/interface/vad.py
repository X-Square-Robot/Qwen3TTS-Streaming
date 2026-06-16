"""TTS output VAD processors.

Stream-oriented voice activity detection for trimming leading silence and
intercepting hallucination noise in TTS output audio.  Designed for the TTS
use case (single continuous speech segment), NOT for ASR.

Three modes:
  - ``disabled``: pass-through, no filtering
  - ``energy``:   preemphasis + Hamming window + log-energy + dB-scale threshold
  - ``tenvad``:   TenVad ONNX inference + TTS-specific state machine

All modes share the same parameter interface (0~1 float thresholds, frame
counts, start margin).  Each mode maps thresholds internally to its own
physical scale.

Architecture: per-session stateful streaming processor.  Audio arrives as
variable-size PCM chunks; the processor splits them into fixed-size frames,
applies VAD logic, and returns only the audio that should be emitted.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class VADMode(Enum):
    DISABLED = "disabled"
    ENERGY = "energy"
    TENVAD = "tenvad"


@dataclass
class TTSVADConfig:
    """Unified VAD configuration for TTS output gating."""

    mode: VADMode = VADMode.DISABLED
    chunk_ms: int = 16
    begin_threshold: float = 0.6   # 0~1, mapped internally
    begin_count: int = 5           # consecutive frames above begin_threshold
    end_threshold: float = 0.35    # 0~1, mapped internally
    end_count: int = 31            # consecutive frames below end_threshold (~500ms)
    start_margin_ms: int = 20      # lookback on begin trigger

    # Energy-mode internals
    preemphasis: float = 0.97

    # TenVAD-mode internals
    tenvad_hop_size: int = 256     # 256 samples @ 16kHz = 16ms
    tenvad_threshold: float = 0.5  # inner model threshold

    @property
    def enabled(self) -> bool:
        return self.mode != VADMode.DISABLED


# ---------------------------------------------------------------------------
# VAD state machine
# ---------------------------------------------------------------------------

class VADState(Enum):
    SILENCE = auto()
    SPEECH = auto()


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

@dataclass
class VADMetrics:
    """Accumulated observability for a session's VAD processing."""

    prefix_trimmed_samples: int = 0
    tail_trimmed_samples: int = 0
    original_audio_samples: int = 0
    effective_audio_samples: int = 0
    begin_trigger_count: int = 0
    end_trigger_count: int = 0
    first_effective_audio_found: bool = False

    @property
    def prefix_trimmed_ms(self) -> float:
        return self.prefix_trimmed_samples  # caller divides by sample_rate

    @property
    def original_audio_ms(self) -> float:
        return self.original_audio_samples

    @property
    def effective_audio_ms(self) -> float:
        return self.effective_audio_samples


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TTSVADProcessor(ABC):
    """Per-session streaming VAD processor.

    Usage::

        vad = create_vad_processor(config, sample_rate=24000)
        for chunk in audio_chunks:
            emit = vad.process_chunk(chunk)
            if emit.size > 0:
                send_to_client(emit)
        final = vad.flush()
        if final.size > 0:
            send_to_client(final)
        vad.reset()
    """

    def __init__(self, config: TTSVADConfig, sample_rate: int = 24000) -> None:
        self._config = config
        self._sample_rate = sample_rate
        self._state = VADState.SILENCE
        self._metrics = VADMetrics()

        # Frame size in samples
        self._frame_samples = max(1, int(round(sample_rate * config.chunk_ms / 1000.0)))

        # Input buffer: accumulate partial frames from variable-size chunks
        self._input_buffer: np.ndarray = np.empty((0,), dtype=np.int16)

        # Start margin buffer: ring buffer of recent frames for lookback
        self._margin_samples = max(
            0, int(round(sample_rate * config.start_margin_ms / 1000.0))
        )
        self._margin_buffer: np.ndarray = np.empty((0,), dtype=np.int16)

        # Pending emit buffer: audio confirmed as speech, awaiting batch emit
        self._emit_buffer: list[np.ndarray] = []

        # State counters
        self._begin_counter: int = 0
        self._end_counter: int = 0

    @property
    def config(self) -> TTSVADConfig:
        return self._config

    @property
    def state(self) -> VADState:
        return self._state

    @property
    def metrics(self) -> VADMetrics:
        return self._metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, pcm_int16: np.ndarray) -> np.ndarray:
        """Process a PCM chunk, return audio that should be emitted now."""
        if not self._config.enabled or pcm_int16.size == 0:
            self._metrics.original_audio_samples += pcm_int16.size
            self._metrics.effective_audio_samples += pcm_int16.size
            return pcm_int16

        # Append to input buffer
        if self._input_buffer.size > 0:
            self._input_buffer = np.concatenate((self._input_buffer, pcm_int16))
        else:
            self._input_buffer = pcm_int16.copy()

        # Process complete frames
        n_frames = self._input_buffer.size // self._frame_samples
        if n_frames == 0:
            return self._drain_emit_buffer()

        consumed = n_frames * self._frame_samples
        frames = self._input_buffer[:consumed].reshape(n_frames, self._frame_samples)
        self._input_buffer = self._input_buffer[consumed:]

        for i in range(n_frames):
            frame = frames[i]
            self._metrics.original_audio_samples += frame.size
            self._process_frame(frame)

        return self._drain_emit_buffer()

    def flush(self) -> np.ndarray:
        """Audio stream ending: emit all pending audio and reset."""
        if not self._config.enabled:
            result = self._input_buffer.copy() if self._input_buffer.size > 0 else np.empty((0,), dtype=np.int16)
            self._input_buffer = np.empty((0,), dtype=np.int16)
            return result

        # Process remaining partial frame (pad with zeros)
        if self._input_buffer.size > 0:
            padded = np.zeros(self._frame_samples, dtype=np.int16)
            padded[:self._input_buffer.size] = self._input_buffer
            self._metrics.original_audio_samples += self._input_buffer.size
            self._process_frame(padded)
            self._input_buffer = np.empty((0,), dtype=np.int16)

        # Flush: emit everything pending regardless of state
        if self._state == VADState.SILENCE:
            # We were in silence — the margin buffer has audio that was held back.
            # On flush, if we never entered speech, discard it (it was leading silence).
            # If we had entered speech at some point, the margin was already consumed.
            self._metrics.prefix_trimmed_samples += self._margin_buffer.size
            self._margin_buffer = np.empty((0,), dtype=np.int16)
        else:
            # In speech state — emit all pending
            if self._margin_buffer.size > 0:
                self._emit_buffer.append(self._margin_buffer.copy())
                self._metrics.effective_audio_samples += self._margin_buffer.size
                self._margin_buffer = np.empty((0,), dtype=np.int16)

        result = self._drain_emit_buffer()
        self._state = VADState.SILENCE
        self._begin_counter = 0
        self._end_counter = 0
        return result

    def reset(self) -> None:
        """Reset all state for reuse."""
        self._state = VADState.SILENCE
        self._input_buffer = np.empty((0,), dtype=np.int16)
        self._margin_buffer = np.empty((0,), dtype=np.int16)
        self._emit_buffer = []
        self._begin_counter = 0
        self._end_counter = 0
        self._metrics = VADMetrics()

    # ------------------------------------------------------------------
    # Frame-level processing (subclass implements scoring)
    # ------------------------------------------------------------------

    @abstractmethod
    def _score_frame(self, frame_int16: np.ndarray) -> float:
        """Return a 0~1 speech score for this frame.

        - energy mode: dB-scale energy normalized to 0~1
        - tenvad mode: TenVad probability
        """
        ...

    def _process_frame(self, frame_int16: np.ndarray) -> None:
        """Core VAD state machine for one frame."""
        score = self._score_frame(frame_int16)
        cfg = self._config

        if self._state == VADState.SILENCE:
            # Check for begin
            if score >= cfg.begin_threshold:
                self._begin_counter += 1
            else:
                self._begin_counter = 0

            if self._begin_counter >= cfg.begin_count:
                # Begin triggered!
                self._state = VADState.SPEECH
                self._end_counter = 0
                self._begin_counter = 0
                self._metrics.begin_trigger_count += 1

                # Emit margin buffer + this frame
                if self._margin_buffer.size > 0:
                    self._emit_buffer.append(self._margin_buffer.copy())
                    self._metrics.effective_audio_samples += self._margin_buffer.size
                    self._margin_buffer = np.empty((0,), dtype=np.int16)

                if not self._metrics.first_effective_audio_found:
                    self._metrics.first_effective_audio_found = True

                self._emit_buffer.append(frame_int16.copy())
                self._metrics.effective_audio_samples += frame_int16.size
            else:
                # Still silence — add to margin buffer
                self._update_margin_buffer(frame_int16)
                self._metrics.prefix_trimmed_samples += frame_int16.size

        elif self._state == VADState.SPEECH:
            # Check for end
            if score < cfg.end_threshold:
                self._end_counter += 1
            else:
                self._end_counter = 0

            if self._end_counter >= cfg.end_count:
                # End triggered!
                self._state = VADState.SILENCE
                self._begin_counter = 0
                self._end_counter = 0
                self._metrics.end_trigger_count += 1
                # The end_count frames that triggered end are discarded
                # (they were below end_threshold = noise/silence)
                self._metrics.tail_trimmed_samples += frame_int16.size
                # Start fresh margin buffer
                self._margin_buffer = np.empty((0,), dtype=np.int16)
            else:
                # Still speech — emit
                self._emit_buffer.append(frame_int16.copy())
                self._metrics.effective_audio_samples += frame_int16.size

    def _update_margin_buffer(self, frame_int16: np.ndarray) -> None:
        """Add frame to margin buffer, evicting oldest if over capacity."""
        if self._margin_samples <= 0:
            return
        self._margin_buffer = np.concatenate((self._margin_buffer, frame_int16))
        if self._margin_buffer.size > self._margin_samples:
            excess = self._margin_buffer.size - self._margin_samples
            self._margin_buffer = self._margin_buffer[excess:]

    def _drain_emit_buffer(self) -> np.ndarray:
        """Concatenate and return all pending emit audio."""
        if not self._emit_buffer:
            return np.empty((0,), dtype=np.int16)
        result = np.concatenate(self._emit_buffer)
        self._emit_buffer = []
        return result


# ---------------------------------------------------------------------------
# Disabled (pass-through)
# ---------------------------------------------------------------------------

class DisabledVADProcessor(TTSVADProcessor):
    """No-op VAD: passes all audio through unchanged."""

    def _score_frame(self, frame_int16: np.ndarray) -> float:
        return 1.0  # always speech


# ---------------------------------------------------------------------------
# Energy VAD
# ---------------------------------------------------------------------------

class EnergyVADProcessor(TTSVADProcessor):
    """Log-energy VAD with preemphasis + Hamming window + dB-scale threshold.

    Scoring:
      1. Preemphasis: y[n] = x[n] - a * x[n-1]
      2. Hamming window
      3. energy = sum(y^2)
      4. dB = 10 * log10(energy / (N * 32768^2) + eps)   — absolute reference
      5. score = clamp((dB + 80) / 80, 0, 1)              — -80dB→0, 0dB→1

    The dB scale makes thresholds frame-size-independent and intuitive.
    Preemphasis amplifies high frequencies, making unvoiced consonants
    (f, s, sh) more visible to the energy detector.
    """

    def __init__(self, config: TTSVADConfig, sample_rate: int = 24000) -> None:
        super().__init__(config, sample_rate)
        self._preemphasis = config.preemphasis
        self._hamming = np.hamming(self._frame_samples).astype(np.float64)
        self._prev_sample: float = 0.0  # for preemphasis continuity

    def _score_frame(self, frame_int16: np.ndarray) -> float:
        x = frame_int16.astype(np.float64)
        a = self._preemphasis

        # Preemphasis
        if a > 0.0:
            y = np.empty_like(x)
            y[0] = x[0] - a * self._prev_sample
            y[1:] = x[1:] - a * x[:-1]
            self._prev_sample = float(x[-1])
        else:
            y = x

        # Hamming window
        y = y * self._hamming

        # Energy
        energy = float(np.sum(y * y))

        # dB scale (absolute reference: full-scale sine)
        n = len(y)
        ref = n * 32768.0 * 32768.0  # N * (2^15)^2
        db = 10.0 * math.log10(energy / ref + 1e-10)

        # Normalize to 0~1: -80dB → 0, 0dB → 1
        score = (db + 80.0) / 80.0
        return max(0.0, min(1.0, score))

    def reset(self) -> None:
        super().reset()
        self._prev_sample = 0.0


# ---------------------------------------------------------------------------
# TenVAD
# ---------------------------------------------------------------------------

_TenVadClass = None
_tenvad_import_error: Optional[Exception] = None

try:
    from ten_vad import TenVad
    _TenVadClass = TenVad
except ImportError as e:
    _tenvad_import_error = e


class TenVADProcessor(TTSVADProcessor):
    """TenVad ONNX model + TTS-specific state machine.

    Only the inference core (TenVad.process() → probability + flags) is used.
    The ASR-oriented multi-segment state machine from workspace/ten_vad.py is
    NOT reused — this class implements the TTS-specific begin/end logic from
    TTSVADProcessor.

    TenVad requires 16kHz int16 input.  This processor handles downsampling
    from the engine's native 24kHz internally.
    """

    def __init__(self, config: TTSVADConfig, sample_rate: int = 24000) -> None:
        super().__init__(config, sample_rate)

        if _TenVadClass is None:
            raise ImportError(
                f"ten_vad package is required for tenvad mode: "
                f"{_tenvad_import_error or 'not installed'}"
            )

        # TenVad operates at 16kHz
        self._tenvad_sr = 16000
        self._tenvad_hop = config.tenvad_hop_size  # 256 samples @ 16kHz = 16ms

        # Create TenVad instance
        self._vad = _TenVadClass(
            hop_size=self._tenvad_hop,
            threshold=config.tenvad_threshold,
        )

        # Downsampling state: 24kHz → 16kHz (3:2 ratio)
        # We accumulate 24kHz samples and emit 16kHz samples.
        # Simple approach: linear interpolation for every 3→2 conversion.
        self._downsample_buffer: np.ndarray = np.empty((0,), dtype=np.int16)
        self._tenvad_frame_samples = self._tenvad_hop  # 256 @ 16kHz

        # TenVad's frame in 24kHz space: we need enough 24kHz samples
        # to produce one 16kHz frame of 256 samples.
        # 256 * 24000 / 16000 = 384 samples at 24kHz
        self._src_frame_samples = max(
            1, int(round(self._tenvad_frame_samples * sample_rate / self._tenvad_sr))
        )

    def _score_frame(self, frame_int16: np.ndarray) -> float:
        """Run TenVad inference on one frame, return probability 0~1."""
        # Downsample 24kHz → 16kHz
        downsampled = self._downsample(frame_int16)

        if downsampled.size < self._tenvad_frame_samples:
            # Not enough samples after downsampling — pad with zeros
            padded = np.zeros(self._tenvad_frame_samples, dtype=np.int16)
            padded[:downsampled.size] = downsampled
            downsampled = padded
        elif downsampled.size > self._tenvad_frame_samples:
            downsampled = downsampled[:self._tenvad_frame_samples]

        try:
            probability, flags = self._vad.process(downsampled)
            return float(probability)
        except Exception as e:
            logger.warning("TenVad process error: %s", e)
            return 0.0

    def _downsample(self, pcm_24k: np.ndarray) -> np.ndarray:
        """Simple 24kHz → 16kHz downsampling via linear interpolation.

        3 samples at 24kHz → 2 samples at 16kHz.
        """
        if pcm_24k.size == 0:
            return np.empty((0,), dtype=np.int16)

        src = pcm_24k.astype(np.float64)
        src_len = src.shape[0]
        dst_len = int(round(src_len * self._tenvad_sr / self._sample_rate))
        if dst_len <= 0:
            return np.empty((0,), dtype=np.int16)

        src_x = np.linspace(0.0, src_len / self._sample_rate, num=src_len, endpoint=False)
        dst_x = np.linspace(0.0, src_len / self._sample_rate, num=dst_len, endpoint=False)
        result = np.interp(dst_x, src_x, src)
        return np.clip(result, -32768, 32767).astype(np.int16)

    def reset(self) -> None:
        super().reset()
        self._downsample_buffer = np.empty((0,), dtype=np.int16)
        # Recreate TenVad instance (it has internal state)
        if _TenVadClass is not None:
            self._vad = _TenVadClass(
                hop_size=self._tenvad_hop,
                threshold=self._config.tenvad_threshold,
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vad_processor(
    config: TTSVADConfig,
    sample_rate: int = 24000,
) -> TTSVADProcessor:
    """Create a VAD processor for the given mode."""
    if not config.enabled:
        return DisabledVADProcessor(config, sample_rate)

    if config.mode == VADMode.ENERGY:
        return EnergyVADProcessor(config, sample_rate)

    if config.mode == VADMode.TENVAD:
        return TenVADProcessor(config, sample_rate)

    raise ValueError(f"Unsupported VAD mode: {config.mode}")


def vad_config_from_dict(d: dict) -> TTSVADConfig:
    """Build TTSVADConfig from a dict (e.g. from VADConfig.config or protocol)."""
    mode_str = str(d.get("mode", "disabled") or "disabled").strip().lower()
    mode = VADMode(mode_str) if mode_str in {m.value for m in VADMode} else VADMode.DISABLED

    return TTSVADConfig(
        mode=mode,
        chunk_ms=int(d.get("chunk_ms", 16) or 16),
        begin_threshold=float(d.get("begin_threshold", 0.6) or 0.6),
        begin_count=int(d.get("begin_count", 5) or 5),
        end_threshold=float(d.get("end_threshold", 0.35) or 0.35),
        end_count=int(d.get("end_count", 31) or 31),
        start_margin_ms=int(d.get("start_margin_ms", 20) or 20),
        preemphasis=float(d.get("preemphasis", 0.97) or 0.97),
        tenvad_hop_size=int(d.get("tenvad_hop_size", 256) or 256),
        tenvad_threshold=float(d.get("tenvad_threshold", 0.5) or 0.5),
    )
