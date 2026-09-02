"""Official PyTorch collector for the same-checkpoint long-form arm."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .arm_types import (
    AudioChunkRecord,
    CollectedRun,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEAKER,
    finish_run,
    stable_sampling_seed,
)
from .models import ArmKind, RunStatus


OfficialModelFactory = Callable[[Path], Any]
SeedSetter = Callable[[int], None]


def _default_seed_setter(seed: int) -> None:
    """Seed every RNG used by the official generation stack on demand."""

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    import torch

    torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "manual_seed_all"):
        cuda.manual_seed_all(seed)


def _official_model_factory(
    *,
    device_map: str,
    dtype: str,
    attn_implementation: str | None,
) -> OfficialModelFactory:
    def load(checkpoint: Path) -> Any:
        # Heavy imports stay behind model loading in the dedicated environment.
        import torch
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        dtype_name = str(dtype).strip().lower()
        dtype_value = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }.get(dtype_name)
        if dtype_value is None:
            raise ValueError(f"unsupported official PyTorch dtype: {dtype!r}")
        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": dtype_value,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        return Qwen3TTSModel.from_pretrained(str(checkpoint), **kwargs)

    return load


def _official_waveform(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    audio = np.asarray(value, dtype=np.float32)
    if audio.ndim == 2 and 1 in audio.shape:
        audio = audio.reshape(-1)
    if audio.ndim != 1:
        raise ValueError(f"official waveform must be mono, got shape {audio.shape}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _official_chunk(
    samples: np.ndarray,
    sample_rate: int,
    sampling_seed: int,
) -> AudioChunkRecord:
    pcm_bytes = samples.astype("<f4", copy=False).tobytes()
    return AudioChunkRecord(
        sequence_index=0,
        chunk_index=0,
        sample_start=0,
        sample_end=int(samples.size),
        sample_count=int(samples.size),
        sample_rate=sample_rate,
        channels=1,
        encoding="pcm_f32",
        first_chunk=True,
        final_chunk=True,
        output_sample_start=0,
        output_sample_end=int(samples.size),
        meta={
            "source": "official_pytorch",
            "sampling_seed": str(sampling_seed),
        },
        pcm_sha256=hashlib.sha256(pcm_bytes).hexdigest(),
    )


class OfficialPyTorchArmAdapter:
    """Same-checkpoint baseline through the official high-level PyTorch API."""

    arm = ArmKind.PYTORCH_0818

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        speaker: str = DEFAULT_SPEAKER,
        language: str = "Auto",
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str | None = "eager",
        sampling_base_seed: int = 0,
        generation_kwargs: Mapping[str, Any] | None = None,
        model_factory: OfficialModelFactory | None = None,
        seed_setter: SeedSetter | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.speaker = speaker
        self.language = language
        # Endpoint engines derive sampling from their server base seed plus the
        # public SID.  ``collect(seed)`` is only the experiment's trial label.
        self.sampling_base_seed = int(sampling_base_seed)
        self.generation_kwargs = dict(generation_kwargs or {})
        self._model_factory = model_factory or _official_model_factory(
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self._seed_setter = seed_setter or _default_seed_setter
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_factory(self.checkpoint)
        return self._model

    def close(self) -> None:
        self._model = None

    def __enter__(self) -> OfficialPyTorchArmAdapter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def collect(
        self,
        text: str,
        *,
        session_id: str,
        seed: int = 0,
    ) -> CollectedRun:
        """Collect the default whole-document call on sampling segment zero."""

        return self._collect_segment(
            text,
            session_id=session_id,
            trial_seed=seed,
            segment_index=0,
            generation_kwargs=self.generation_kwargs,
        )

    def collect_segment(
        self,
        text: str,
        session_id: str,
        trial_seed: int,
        segment_index: int,
        generation_kwargs_override: Mapping[str, Any] | None = None,
    ) -> CollectedRun:
        """Collect one frozen segment with its true engine segment identity."""

        if (
            not isinstance(segment_index, int)
            or isinstance(segment_index, bool)
            or segment_index < 0
        ):
            raise ValueError("segment_index must be a non-negative integer")
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.update(dict(generation_kwargs_override or {}))
        return self._collect_segment(
            text,
            session_id=session_id,
            trial_seed=trial_seed,
            segment_index=segment_index,
            generation_kwargs=generation_kwargs,
        )

    def _collect_segment(
        self,
        text: str,
        *,
        session_id: str,
        trial_seed: int,
        segment_index: int,
        generation_kwargs: Mapping[str, Any],
    ) -> CollectedRun:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if not session_id:
            raise ValueError("session_id must not be empty")

        started_at = time.perf_counter()
        sampling_seed = stable_sampling_seed(
            self.sampling_base_seed, session_id, segment_index
        )
        try:
            model = self._get_model()
        except Exception as exc:  # noqa: BLE001 - loader failure is evidence
            return finish_run(
                self.arm,
                session_id=session_id,
                seed=trial_seed,
                status=RunStatus.ERROR,
                started_at=started_at,
                error=f"{type(exc).__name__}: {exc}",
                sampling_seed=sampling_seed,
            )

        try:
            self._seed_setter(sampling_seed)
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                speaker=self.speaker,
                language=self.language,
                **generation_kwargs,
            )
            if not isinstance(wavs, (list, tuple)) or len(wavs) != 1:
                raise ValueError(
                    "official generate_custom_voice must return exactly one waveform"
                )
            samples = _official_waveform(wavs[0])
            sample_rate = int(sample_rate)
        except Exception as exc:  # noqa: BLE001 - generation failure is evidence
            return finish_run(
                self.arm,
                session_id=session_id,
                seed=trial_seed,
                status=RunStatus.TTS_FAILED,
                started_at=started_at,
                error=f"{type(exc).__name__}: {exc}",
                sampling_seed=sampling_seed,
            )

        chunks = [_official_chunk(samples, sample_rate, sampling_seed)]
        failure: str | None = None
        if sample_rate != DEFAULT_SAMPLE_RATE:
            failure = (
                f"official arm returned {sample_rate} Hz; expected "
                f"{DEFAULT_SAMPLE_RATE} Hz"
            )
        elif samples.size == 0:
            failure = "official arm produced no audio"
        return finish_run(
            self.arm,
            session_id=session_id,
            seed=trial_seed,
            status=RunStatus.TTS_FAILED if failure else RunStatus.OK,
            started_at=started_at,
            error=failure,
            audio_chunks=chunks,
            samples=samples,
            sample_rate=sample_rate,
            sampling_seed=sampling_seed,
        )


OfficialPyTorchArm = OfficialPyTorchArmAdapter


__all__ = ["OfficialPyTorchArm", "OfficialPyTorchArmAdapter"]
