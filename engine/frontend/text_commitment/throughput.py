"""Online rate and audio-credit estimates for streaming continuity control.

This module is intentionally independent from semantic commit decisions.  It
only answers whether a pending semantic wait can probably be hidden by already
generated audio.
"""
from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class StreamingRateMetrics:
    raw_tokens: int = 0
    normalized_tokens: int = 0
    codec_frames: int = 0
    generated_audio_ms: float = 0.0
    estimated_played_audio_ms: float = 0.0
    audio_credit_ms: float = 0.0
    lambda_raw: float = 0.0
    lambda_norm: float = 0.0
    t_service_ms: float = 0.0
    r_audio_ms_per_s: float = 0.0
    safe_wait_ms: float = 0.0


class AudioCreditEstimator:
    """Track three unit domains and estimate safe semantic wait budget."""

    def __init__(self, *, codec_frame_rate: float = 12.5, reserve_ms: float = 0.0) -> None:
        self.codec_frame_rate = max(float(codec_frame_rate), 0.001)
        self.reserve_ms = max(float(reserve_ms), 0.0)
        self._raw = 0
        self._norm = 0
        self._frames = 0
        self._generated_ms = 0.0
        self._played_ms = 0.0
        self._started_at = time.monotonic()
        self._service_ms = 0.0
        self._recovery_ms = 0.0
        self._jitter_ms = 0.0

    def observe_raw_tokens(self, count: int = 1) -> None:
        self._raw += max(int(count), 0)

    def observe_normalized_tokens(self, count: int = 1, *, service_ms: float | None = None) -> None:
        self._norm += max(int(count), 0)
        if service_ms is not None and service_ms >= 0:
            self._service_ms = float(service_ms)

    def observe_codec_frames(self, count: int) -> None:
        count = max(int(count), 0)
        self._frames += count
        self._generated_ms += count * 1000.0 / self.codec_frame_rate

    def observe_played_audio(self, elapsed_ms: float) -> None:
        self._played_ms = max(self._played_ms, float(elapsed_ms))

    def set_reserves(self, *, recovery_ms: float = 0.0, jitter_ms: float = 0.0, network_ms: float = 0.0) -> None:
        self._recovery_ms = max(float(recovery_ms), 0.0)
        self._jitter_ms = max(float(jitter_ms), 0.0) + max(float(network_ms), 0.0)

    def snapshot(self, *, now: float | None = None) -> StreamingRateMetrics:
        elapsed_s = max((time.monotonic() if now is None else now) - self._started_at, 1e-3)
        raw_rate = self._raw / elapsed_s
        expansion = self._norm / self._raw if self._raw else 0.0
        norm_rate = expansion * raw_rate
        service_rate = 1000.0 / self._service_ms if self._service_ms > 0 else float("inf")
        # k is codec frames per normalized token; if no text has arrived yet,
        # there is no meaningful audio throughput estimate.
        k = self._frames / self._norm if self._norm else 0.0
        effective_norm_rate = min(norm_rate, service_rate)
        r_audio = effective_norm_rate * k * 1000.0 / self.codec_frame_rate
        credit = self._generated_ms - self._played_ms - self.reserve_ms
        safe_wait = max(0.0, credit - self._recovery_ms - self._jitter_ms)
        return StreamingRateMetrics(
            raw_tokens=self._raw,
            normalized_tokens=self._norm,
            codec_frames=self._frames,
            generated_audio_ms=self._generated_ms,
            estimated_played_audio_ms=self._played_ms,
            audio_credit_ms=credit,
            lambda_raw=raw_rate,
            lambda_norm=norm_rate,
            t_service_ms=self._service_ms,
            r_audio_ms_per_s=r_audio,
            safe_wait_ms=safe_wait,
        )

