"""Diagnostic-only acoustic features for blind-review prioritisation.

None of these measurements is allowed to create a confirmed hallucination
label.  They are intentionally transparent features that help reviewers find
non-linguistic tails which ASR may omit entirely.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def acoustic_diagnostics(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
) -> dict[str, Any]:
    """Return stable whole-file and frame-level noise screening features."""

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if audio.size == 0:
        return {
            "sample_count": 0,
            "duration_s": 0.0,
            "finite_fraction": 1.0,
            "clipping_fraction": 0.0,
            "rms_dbfs": None,
            "dc_offset": None,
            "spectral_flatness_median": None,
            "high_frequency_ratio_median": None,
            "flat_audible_longest_s": 0.0,
        }

    finite = np.isfinite(audio)
    clean = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    rms = float(np.sqrt(np.mean(np.square(clean, dtype=np.float64))))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    frame_length = max(16, round(sample_rate * frame_ms / 1000.0))
    hop_length = max(8, round(sample_rate * hop_ms / 1000.0))

    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - validated by environment preflight
        raise RuntimeError(
            "long-form acoustic diagnostics require librosa in the validation venv"
        ) from exc

    spectrum = np.abs(
        librosa.stft(
            clean,
            n_fft=frame_length,
            hop_length=hop_length,
            win_length=frame_length,
            center=False,
        )
    )
    if spectrum.shape[1] == 0:
        spectrum = np.abs(
            librosa.stft(
                np.pad(clean, (0, max(0, frame_length - clean.size))),
                n_fft=frame_length,
                hop_length=hop_length,
                win_length=frame_length,
                center=False,
            )
        )
    power = np.square(spectrum, dtype=np.float64)
    flatness = librosa.feature.spectral_flatness(S=power).reshape(-1)
    frame_rms = librosa.feature.rms(
        S=spectrum,
        frame_length=frame_length,
    ).reshape(-1)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
    high_mask = frequencies >= min(8000.0, sample_rate * 0.4)
    total_energy = power.sum(axis=0)
    high_ratio = (
        power[high_mask].sum(axis=0) / np.maximum(total_energy, 1e-20)
        if np.any(high_mask)
        else np.zeros_like(total_energy)
    )
    flat_audible = (flatness >= 0.6) & (frame_rms >= 10 ** (-50.0 / 20.0))

    return {
        "sample_count": int(audio.size),
        "duration_s": audio.size / float(sample_rate),
        "finite_fraction": float(np.mean(finite)),
        "clipping_fraction": float(np.mean(np.abs(clean) >= 0.999)),
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": 20.0 * math.log10(max(float(np.max(np.abs(clean))), 1e-12)),
        "dc_offset": float(np.mean(clean, dtype=np.float64)),
        "spectral_flatness_median": float(np.median(flatness)),
        "spectral_flatness_p95": float(np.quantile(flatness, 0.95)),
        "high_frequency_ratio_median": float(np.median(high_ratio)),
        "high_frequency_ratio_p95": float(np.quantile(high_ratio, 0.95)),
        "flat_audible_longest_s": (
            _longest_true_run(flat_audible) * hop_length / float(sample_rate)
        ),
        "diagnostic_only": True,
    }
