"""Transient contracts shared by long-form synthesis arm implementations."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .models import ArmKind, RunStatus


DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_SPEAKER = "001"
DEFAULT_LANGUAGE = "auto"
_MAX_TORCH_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class AudioChunkRecord:
    """Serializable coordinates and identity for one collected audio chunk.

    ``sample_start``/``sample_end`` address :attr:`CollectedRun.samples`.
    ``output_sample_*`` preserve delivery coordinates after VAD or guarded
    delivery.  PCM is retained once on the run; the digest maps each chunk to
    its original wire payload without duplicating audio in JSON sidecars.
    """

    sequence_index: int
    chunk_index: int
    sample_start: int
    sample_end: int
    sample_count: int
    sample_rate: int
    channels: int
    encoding: str
    first_chunk: bool
    final_chunk: bool
    output_sample_start: int | None
    output_sample_end: int | None
    meta: dict[str, str]
    pcm_sha256: str


@dataclass
class CollectedRun:
    """Transient, typed result shared by all three synthesis arms."""

    arm: ArmKind
    seed: int
    session_id: str
    status: RunStatus
    samples: np.ndarray
    sample_rate: int
    duration_s: float
    events: list[dict[str, Any]] = field(default_factory=list)
    audio_chunks: list[AudioChunkRecord] = field(default_factory=list)
    error: str | None = None
    total_ms: int = 0
    ttft_ms: int | None = None
    terminal_event: str | None = None
    eos_reason: str | None = None
    sampling_seed: int | None = None

    @property
    def chunks(self) -> int:
        return len(self.audio_chunks)

    @property
    def pcm_bytes(self) -> bytes:
        """Canonical little-endian float32 PCM for hashing/persistence."""

        return np.asarray(self.samples, dtype="<f4").reshape(-1).tobytes()


class ArmAdapter(Protocol):
    """Narrow contract consumed by the serial long-form experiment runner."""

    arm: ArmKind

    def collect(
        self,
        text: str,
        *,
        session_id: str,
        seed: int = 0,
    ) -> CollectedRun: ...


def stable_sampling_seed(
    base_seed: int,
    session_id: str,
    segment_id: int = 0,
) -> int:
    """Match the gateway-bound engine session/segment seed derivation.

    The gateway keeps the public session ID as ``sampling_identity`` while the
    engine registry uses a private UUID.  Before hashing, the executor expands
    that identity to ``"<public sid>:<segment>"`` and then appends the segment
    index once more.  Reproducing both steps here is required for the official
    PyTorch arm to receive the same torch seed as the endpoint arms.
    """

    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(int(base_seed)).encode("utf-8"))
    engine_identity = f"{session_id}:{int(segment_id)}"
    for part in (engine_identity, int(segment_id)):
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") & _MAX_TORCH_SEED


def finish_run(
    arm: ArmKind,
    *,
    session_id: str,
    seed: int,
    status: RunStatus,
    started_at: float,
    samples: np.ndarray | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    events: list[dict[str, Any]] | None = None,
    audio_chunks: list[AudioChunkRecord] | None = None,
    error: str | None = None,
    ttft_ms: int | None = None,
    terminal_event: str | None = None,
    eos_reason: str | None = None,
    sampling_seed: int | None = None,
) -> CollectedRun:
    """Finalize common timing/audio fields without arm-specific branching."""

    audio = (
        np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples is not None
        else np.empty(0, dtype=np.float32)
    )
    return CollectedRun(
        arm=arm,
        seed=int(seed),
        session_id=session_id,
        status=status,
        samples=audio,
        sample_rate=int(sample_rate),
        duration_s=(audio.size / float(sample_rate) if sample_rate > 0 else 0.0),
        events=list(events or []),
        audio_chunks=list(audio_chunks or []),
        error=error,
        total_ms=round((time.perf_counter() - started_at) * 1000.0),
        ttft_ms=ttft_ms,
        terminal_event=terminal_event,
        eos_reason=eos_reason,
        sampling_seed=sampling_seed,
    )


__all__ = [
    "ArmAdapter",
    "AudioChunkRecord",
    "CollectedRun",
    "DEFAULT_LANGUAGE",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SPEAKER",
    "finish_run",
    "stable_sampling_seed",
]
