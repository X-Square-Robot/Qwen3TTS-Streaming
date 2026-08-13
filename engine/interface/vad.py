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
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

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
    begin_threshold: float = 0.6  # 0~1, mapped internally
    begin_count: int = 5  # consecutive frames above begin_threshold
    end_threshold: float = 0.35  # 0~1, mapped internally
    end_count: int = 31  # consecutive frames below end_threshold (~500ms)
    start_margin_ms: int = 20  # lookback on begin trigger

    # Energy-mode internals
    preemphasis: float = 0.97

    # TenVAD-mode internals
    tenvad_hop_size: int = 256  # 256 samples @ 16kHz = 16ms
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


@dataclass(frozen=True)
class SampleProvenanceSpan:
    """Source attribution for a half-open native-sample range.

    The range is local to the ``AttributedSamples`` object that carries it.
    ``progress_event`` is a candidate only; output sample coordinates are
    assigned by the shared output processor after VAD and resampling.
    """

    sample_start: int
    sample_end: int
    segment_idx: int = -1
    source_frame_start: int = 0
    source_frame_end: int = 0
    progress_event: Optional[Any] = None


@dataclass
class AttributedSamples:
    """Native PCM together with run-length provenance spans."""

    samples: np.ndarray
    provenance: list[SampleProvenanceSpan]


def _shift_spans(
    spans: list[SampleProvenanceSpan], offset: int
) -> list[SampleProvenanceSpan]:
    return [
        SampleProvenanceSpan(
            max(0, span.sample_start + offset),
            max(0, span.sample_end + offset),
            span.segment_idx,
            span.source_frame_start,
            span.source_frame_end,
            span.progress_event,
        )
        for span in spans
        if span.sample_end > span.sample_start
    ]


def _slice_spans(
    spans: list[SampleProvenanceSpan], start: int, end: int
) -> list[SampleProvenanceSpan]:
    """Clip local provenance spans to ``[start, end)`` and rebase to zero."""

    clipped: list[SampleProvenanceSpan] = []
    for span in spans:
        left = max(start, span.sample_start)
        right = min(end, span.sample_end)
        if right <= left:
            continue
        clipped.append(
            SampleProvenanceSpan(
                left - start,
                right - start,
                span.segment_idx,
                span.source_frame_start,
                span.source_frame_end,
                span.progress_event,
            )
        )
    return clipped


def _concat_spans(
    blocks: list[tuple[np.ndarray, list[SampleProvenanceSpan]]]
) -> AttributedSamples:
    if not blocks:
        return AttributedSamples(np.empty((0,), dtype=np.int16), [])
    samples = np.concatenate([audio for audio, _ in blocks])
    provenance: list[SampleProvenanceSpan] = []
    offset = 0
    for audio, spans in blocks:
        provenance.extend(_shift_spans(spans, offset))
        offset += audio.size
    return AttributedSamples(samples, provenance)


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
        self._input_provenance: list[SampleProvenanceSpan] = []

        # Start margin buffer: ring buffer of recent frames for lookback
        self._margin_samples = max(
            0, int(round(sample_rate * config.start_margin_ms / 1000.0))
        )
        self._margin_buffer: np.ndarray = np.empty((0,), dtype=np.int16)
        self._margin_provenance: list[SampleProvenanceSpan] = []

        # Pending emit buffer: audio confirmed as speech, awaiting batch emit
        self._emit_buffer: list[tuple[np.ndarray, list[SampleProvenanceSpan]]] = []

        # Candidate onset frames: consecutive above-threshold frames while the
        # begin counter is climbing. Held uncapped (bounded by begin_count) so a
        # real speech onset is never evicted by the small lookback margin.
        self._begin_buffer: list[tuple[np.ndarray, list[SampleProvenanceSpan]]] = []

        # State counters
        self._begin_counter: int = 0
        self._end_counter: int = 0

        # L2 observability side-channel (decoupled from logging — the gateway
        # owns session context and emits). Off by default ⇒ zero cost.
        self._record_transitions: bool = False
        self._transitions: list[dict] = []

    def enable_transition_recording(self, flag: bool) -> None:
        self._record_transitions = flag

    def drain_transitions(self) -> list[dict]:
        if not self._transitions:
            return []
        out = self._transitions
        self._transitions = []
        return out

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
        return self.process_attributed_chunk(
            pcm_int16,
            [SampleProvenanceSpan(0, int(pcm_int16.size))],
        ).samples

    def process_attributed_chunk(
        self,
        pcm_int16: np.ndarray,
        provenance: list[SampleProvenanceSpan],
    ) -> AttributedSamples:
        """Process native PCM while retaining source provenance.

        The legacy :meth:`process_chunk` API remains a thin ndarray wrapper.
        Gateways that need text alignment use this method so prefix margin,
        onset candidates, and tail decisions carry the original segment/event
        attribution through the state machine.
        """
        pcm_int16 = np.asarray(pcm_int16, dtype=np.int16).reshape(-1)
        if pcm_int16.size == 0:
            return AttributedSamples(pcm_int16, [])
        if not self._config.enabled:
            self._metrics.original_audio_samples += pcm_int16.size
            self._metrics.effective_audio_samples += pcm_int16.size
            return AttributedSamples(pcm_int16, _slice_spans(provenance, 0, pcm_int16.size))

        # Append to input buffer
        if self._input_buffer.size > 0:
            self._input_buffer = np.concatenate((self._input_buffer, pcm_int16))
            self._input_provenance.extend(
                _shift_spans(provenance, self._input_buffer.size - pcm_int16.size)
            )
        else:
            self._input_buffer = pcm_int16.copy()
            self._input_provenance = _slice_spans(provenance, 0, pcm_int16.size)

        # Process complete frames
        n_frames = self._input_buffer.size // self._frame_samples
        if n_frames == 0:
            return self._drain_emit_buffer_attributed()

        consumed = n_frames * self._frame_samples
        frames = self._input_buffer[:consumed].reshape(n_frames, self._frame_samples)
        self._input_buffer = self._input_buffer[consumed:]
        frame_provenance = _slice_spans(self._input_provenance, 0, consumed)
        self._input_provenance = _shift_spans(
            _slice_spans(self._input_provenance, consumed, consumed + self._input_buffer.size),
            -consumed,
        )

        for i in range(n_frames):
            frame = frames[i]
            frame_spans = _slice_spans(
                frame_provenance,
                i * self._frame_samples,
                (i + 1) * self._frame_samples,
            )
            self._metrics.original_audio_samples += frame.size
            self._process_frame(frame, provenance=frame_spans)

        return self._drain_emit_buffer_attributed()

    def flush(self) -> np.ndarray:
        """Audio stream ending: emit all pending audio and reset."""
        return self.flush_attributed().samples

    def flush_attributed(self) -> AttributedSamples:
        """Flush the detector and return retained samples with provenance."""
        if not self._config.enabled:
            result = (
                self._input_buffer.copy()
                if self._input_buffer.size > 0
                else np.empty((0,), dtype=np.int16)
            )
            self._input_buffer = np.empty((0,), dtype=np.int16)
            provenance = self._input_provenance
            self._input_provenance = []
            return AttributedSamples(result, _slice_spans(provenance, 0, result.size))

        # Process remaining partial frame (pad only for scoring).  The state
        # machine receives the real sample count so padding is never emitted or
        # included in trim metrics.
        if self._input_buffer.size > 0:
            valid_samples = self._input_buffer.size
            padded = np.zeros(self._frame_samples, dtype=np.int16)
            padded[:valid_samples] = self._input_buffer
            self._metrics.original_audio_samples += valid_samples
            self._process_frame(
                padded,
                valid_samples=valid_samples,
                provenance=_slice_spans(self._input_provenance, 0, valid_samples),
            )
            self._input_buffer = np.empty((0,), dtype=np.int16)
            self._input_provenance = []

        # Flush: emit everything pending regardless of state
        if self._state == VADState.SILENCE:
            # The stream ended before this lookback could accompany a confirmed
            # onset. Discard it as prefix silence before the first speech, or as
            # tail silence after speech has already been emitted.
            self._record_trimmed_samples(self._margin_buffer.size)
            self._margin_buffer = np.empty((0,), dtype=np.int16)
            self._margin_provenance = []
            # Candidate onset frames that never confirmed as speech are discarded
            # too (an unconfirmed begin run at end-of-stream).
            for candidate, _ in self._begin_buffer:
                self._record_trimmed_samples(candidate.size)
            self._begin_buffer = []
        else:
            # In speech state — emit all pending
            if self._margin_buffer.size > 0:
                self._emit_buffer.append(
                    (self._margin_buffer.copy(), self._margin_provenance)
                )
                self._metrics.effective_audio_samples += self._margin_buffer.size
                self._margin_buffer = np.empty((0,), dtype=np.int16)
                self._margin_provenance = []

        result = self._drain_emit_buffer_attributed()
        self._state = VADState.SILENCE
        self._begin_counter = 0
        self._end_counter = 0
        self._begin_buffer = []
        return result

    def discard_pending(self) -> None:
        """Discard uncommitted audio at a condemned segment boundary.

        Unlike :meth:`reset`, this preserves cumulative metrics.  Complete
        frames in the margin/onset buffers were already counted as original
        input, while a partial input frame was not; finalize each exactly once
        before clearing the streaming and detector state.  This prevents an
        onset candidate from one aborted segment being completed by the next
        segment without erasing the trim evidence from observability.
        """
        if self._input_buffer.size > 0:
            self._metrics.original_audio_samples += self._input_buffer.size
            self._record_trimmed_samples(self._input_buffer.size)
        self._record_trimmed_samples(self._margin_buffer.size)
        for candidate, _ in self._begin_buffer:
            self._record_trimmed_samples(candidate.size)

        self._clear_stream_state()
        self._reset_detector_state()

    def reset(self) -> None:
        """Reset all state for reuse."""
        self._clear_stream_state()
        self._metrics = VADMetrics()
        self._reset_detector_state()

    def _clear_stream_state(self) -> None:
        """Clear stream buffers/counters without touching accumulated metrics."""
        self._state = VADState.SILENCE
        self._input_buffer = np.empty((0,), dtype=np.int16)
        self._input_provenance = []
        self._margin_buffer = np.empty((0,), dtype=np.int16)
        self._margin_provenance = []
        self._emit_buffer = []
        self._begin_buffer = []
        self._begin_counter = 0
        self._end_counter = 0

    def _reset_detector_state(self) -> None:
        """Reset subclass-specific streaming detector state."""

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

    def _process_frame(
        self,
        frame_int16: np.ndarray,
        valid_samples: Optional[int] = None,
        provenance: Optional[list[SampleProvenanceSpan]] = None,
    ) -> None:
        """Core VAD state machine for one frame."""
        score = self._score_frame(frame_int16)
        cfg = self._config
        audio = (
            frame_int16
            if valid_samples is None
            else frame_int16[: max(0, min(valid_samples, frame_int16.size))]
        )
        frame_provenance = _slice_spans(
            provenance or [SampleProvenanceSpan(0, frame_int16.size)],
            0,
            audio.size,
        )

        if self._state == VADState.SILENCE:
            # Check for begin
            if score >= cfg.begin_threshold:
                self._begin_counter += 1
                # Hold this candidate onset frame uncapped (see _begin_buffer).
                self._begin_buffer.append((audio.copy(), frame_provenance))
            else:
                # A below-threshold frame breaks the run: the candidate frames
                # were not speech after all → demote them to lookback silence
                # (capped). Samples are counted as trimmed only when evicted
                # from lookback, because retained margin may still be emitted.
                for candidate, candidate_provenance in self._begin_buffer:
                    self._update_margin_buffer(candidate, candidate_provenance)
                self._begin_buffer = []
                self._begin_counter = 0
                self._update_margin_buffer(audio, frame_provenance)

            if self._begin_counter >= cfg.begin_count:
                # Begin triggered!
                self._state = VADState.SPEECH
                self._end_counter = 0
                self._begin_counter = 0
                self._metrics.begin_trigger_count += 1
                if self._record_transitions:
                    self._transitions.append(
                        {
                            "event": "begin",
                            "score": round(float(score), 4),
                            "threshold": cfg.begin_threshold,
                            "trigger_count": self._metrics.begin_trigger_count,
                        }
                    )

                # Emit the lookback margin (silence before the onset) ...
                if self._margin_buffer.size > 0:
                    self._emit_buffer.append(
                        (self._margin_buffer.copy(), self._margin_provenance)
                    )
                    self._metrics.effective_audio_samples += self._margin_buffer.size
                    self._margin_buffer = np.empty((0,), dtype=np.int16)
                    self._margin_provenance = []

                if not self._metrics.first_effective_audio_found:
                    self._metrics.first_effective_audio_found = True

                # ... then ALL candidate onset frames (uncapped) — this is the
                # real speech onset that the old code clipped via the margin cap.
                for candidate, candidate_provenance in self._begin_buffer:
                    self._emit_buffer.append((candidate, candidate_provenance))
                    self._metrics.effective_audio_samples += candidate.size
                self._begin_buffer = []

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
                if self._record_transitions:
                    self._transitions.append(
                        {
                            "event": "end",
                            "score": round(float(score), 4),
                            "threshold": cfg.end_threshold,
                            "trigger_count": self._metrics.end_trigger_count,
                        }
                    )
                # Discard the frame that crosses the end-count threshold. The
                # preceding low-score frames were already emitted as speech.
                self._record_trimmed_samples(audio.size)
                # Start fresh margin buffer
                self._margin_buffer = np.empty((0,), dtype=np.int16)
                self._margin_provenance = []
            else:
                # Still speech — emit
                self._emit_buffer.append((audio.copy(), frame_provenance))
                self._metrics.effective_audio_samples += audio.size

    def _record_trimmed_samples(self, sample_count: int) -> None:
        """Classify samples once their final disposition is known.

        Silence before the first confirmed onset is prefix trim.  Once any
        effective audio has been found, discarded samples belong to the tail
        (including gaps after an end transition).
        """
        if sample_count <= 0:
            return
        if self._metrics.first_effective_audio_found:
            self._metrics.tail_trimmed_samples += sample_count
        else:
            self._metrics.prefix_trimmed_samples += sample_count

    def _update_margin_buffer(
        self,
        frame_int16: np.ndarray,
        provenance: Optional[list[SampleProvenanceSpan]] = None,
    ) -> None:
        """Add audio to lookback and account only samples actually evicted."""
        if frame_int16.size == 0:
            return
        if self._margin_samples <= 0:
            self._record_trimmed_samples(frame_int16.size)
            return
        self._margin_buffer = np.concatenate((self._margin_buffer, frame_int16))
        offset = self._margin_buffer.size - frame_int16.size
        self._margin_provenance.extend(
            _shift_spans(
                provenance or [SampleProvenanceSpan(0, frame_int16.size)],
                offset,
            )
        )
        if self._margin_buffer.size > self._margin_samples:
            excess = self._margin_buffer.size - self._margin_samples
            self._margin_buffer = self._margin_buffer[excess:]
            self._margin_provenance = _shift_spans(
                _slice_spans(
                    self._margin_provenance,
                    excess,
                    excess + self._margin_buffer.size,
                ),
                -excess,
            )
            self._record_trimmed_samples(excess)

    def _drain_emit_buffer(self) -> np.ndarray:
        """Concatenate and return all pending emit audio."""
        return self._drain_emit_buffer_attributed().samples

    def _drain_emit_buffer_attributed(self) -> AttributedSamples:
        """Concatenate pending audio while retaining run-length provenance."""
        result = _concat_spans(self._emit_buffer)
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

    def _reset_detector_state(self) -> None:
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
            padded[: downsampled.size] = downsampled
            downsampled = padded
        elif downsampled.size > self._tenvad_frame_samples:
            downsampled = downsampled[: self._tenvad_frame_samples]

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

        src_x = np.linspace(
            0.0, src_len / self._sample_rate, num=src_len, endpoint=False
        )
        dst_x = np.linspace(
            0.0, src_len / self._sample_rate, num=dst_len, endpoint=False
        )
        result = np.interp(dst_x, src_x, src)
        return np.clip(result, -32768, 32767).astype(np.int16)

    def _reset_detector_state(self) -> None:
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
    mode = (
        VADMode(mode_str)
        if mode_str in {m.value for m in VADMode}
        else VADMode.DISABLED
    )

    # Use `if k in d else default` rather than `d.get(k) or default` so an
    # explicit 0 / 0.0 (e.g. start_margin_ms=0 to disable lookback) is honored
    # instead of being silently replaced by the default.
    def _num(key, default, cast):
        return cast(d[key]) if d.get(key) is not None else default

    return TTSVADConfig(
        mode=mode,
        chunk_ms=_num("chunk_ms", 16, int),
        begin_threshold=_num("begin_threshold", 0.6, float),
        begin_count=_num("begin_count", 5, int),
        end_threshold=_num("end_threshold", 0.35, float),
        end_count=_num("end_count", 31, int),
        start_margin_ms=_num("start_margin_ms", 20, int),
        preemphasis=_num("preemphasis", 0.97, float),
        tenvad_hop_size=_num("tenvad_hop_size", 256, int),
        tenvad_threshold=_num("tenvad_threshold", 0.5, float),
    )
