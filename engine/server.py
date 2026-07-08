"""TTS Engine server: wires asyncio frontend + GPU engine thread.

Architecture:

    ┌────────── asyncio event loop (main thread) ──────────┐
    │                                                       │
    │  gRPC / WebSocket  ─►  FrontendInterface ─► Dispatcher │
    │                                       │         │      │
    │       ▲                                    │          │
    │       │ audio chunks                       │          │
    │       │ (call_soon_threadsafe)              │          │
    │  session.result_queue  ◄───────────────┐   │          │
    └───────────────────────────────────────┼───┼──────────┘
                                            │   │
                    ┌── Engine Thread ───────┼───┼──────────┐
                    │                       │   ▼           │
                    │  engine_inbox.get() → EngineLoop      │
                    │                      ┌────────────┐   │
                    │                      │ Executor    │   │
                    │                      │ KVCachePool │   │
                    │                      │ PrefillBld  │   │
                    │                      │ PrefixCache │   │
                    │                      └────────────┘   │
                    └───────────────────────────────────────┘

Usage:
    python -m engine.server --config engine.yaml
    python -m engine.server --model-package-dir /models/tts_orchestrator/<version>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import queue
import signal
from pathlib import Path
from typing import AsyncIterator, Optional

import torch

from .config import (
    EngineConfig,
    ModelArchConfig,
    apply_model_package_paths,
    load_config,
    load_model_manifest,
    resolve_model_package_paths,
    to_model_config,
)
from .runtime.fingerprint import (
    FingerprintCheckError,
    enforce_engine_fingerprint,
)
from .core.mlfq import MLFQConfig
from .core import observability as obs
from .core.types import SessionConfig
from .interface import normalize_capabilities
from .frontend.interface import FrontendInterface
from .frontend.spliter.tokenizer import LightQwen3TTSTokenizer
from .backend.engine_loop import EngineLoop
from .backend.executor import Executor
from .backend.prefill import EmbeddingWeights, PrefillBuilder
from .backend.ref_audio_processor import ReferenceAudioProcessor

logger = logging.getLogger(__name__)


_EXTERNAL_TO_INTERNAL_TASK_TYPE = {
    "base": "voice_clone",
    "icl": "voice_clone",
    "voice_clone": "voice_clone",
    "custom_voice": "custom_voice",
    "voice_design": "voice_design",
    "instruct": "voice_design",
}

_DEFAULT_BASE_REF_TEXT = (
    os.environ.get("ENGINE_DEFAULT_REF_TEXT")
    or os.environ.get("ENGINE_DEFAULT_BASE_REF_TEXT")
    or "爱护环境，人人有责，让我们一起守护前海石公园环境卫生！"
)

_DEFAULT_BASE_REF_AUDIO_CANDIDATES = (
    os.environ.get("ENGINE_DEFAULT_REF_AUDIO_PATH", ""),
    os.environ.get("ENGINE_DEFAULT_BASE_REF_AUDIO_PATH", ""),
    "workspace/default_refs/base_ref.wav",
)


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "out of memory" in msg and "cuda" in msg


class TTSEngine:
    """Top-level engine that owns both the asyncio world and the GPU thread."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        model_arch: Optional[ModelArchConfig] = None,
        *,
        tokenizer_dir: str = "",
        weights_dir: str = "",
        engine_dir: str = "",
        device_id: int = 0,
        max_batch_size: int = 48,
        max_sessions: int = 128,
        max_seq_len: int = 512,
    ):
        self._cfg = config or EngineConfig()
        self._model_arch = model_arch or ModelArchConfig()

        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._engine_inbox = queue.Queue(maxsize=4096)
        self._async_inbox: Optional[asyncio.Queue] = None

        self._tokenizer_dir = tokenizer_dir or self._cfg.paths.tokenizer_dir
        self._weights_dir = weights_dir or self._cfg.paths.weights_dir
        self._engine_dir = engine_dir or self._cfg.paths.engine_dir
        self._device_id = device_id
        self._max_batch = (
            max_batch_size
            if max_batch_size != 48
            else self._cfg.scheduler.max_batch_size
        )
        self._max_sessions = (
            max_sessions if max_sessions != 128 else self._cfg.server.max_sessions
        )
        self._max_seq_len = (
            max_seq_len if max_seq_len != 512 else self._cfg.scheduler.max_seq_len
        )
        self._validate_runtime_profile_bounds()
        self._cfg.scheduler.max_batch_size = self._max_batch
        self._cfg.scheduler.max_seq_len = self._max_seq_len
        self._cfg.server.max_sessions = self._max_sessions

        self._tokenizer: Optional[LightQwen3TTSTokenizer] = None
        self._frontend: Optional[FrontendInterface] = None
        self._executor: Optional[Executor] = None
        self._engine_loop: Optional[EngineLoop] = None
        self._relay_task: Optional[asyncio.Task] = None
        self._ref_audio_processor: Optional[ReferenceAudioProcessor] = None
        self._default_base_ref_audio: Optional[bytes] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _validate_runtime_profile_bounds(self) -> None:
        """Fail early if requested runtime limits exceed the built TRT profile."""
        profile = self._model_arch.engine_profile
        variant = self._model_arch.variant or "unknown"

        # Surface the per-submodule precision the engine claims to be built with,
        # and warn on the streaming-hallucination knob (CP not in fp32). This is
        # the only runtime visibility into cp_precision — it is otherwise baked
        # into the plan and never cross-checked.
        logger.info(
            "Engine precision: base=%s backbone=%s cp=%s code2wav=%s (variant=%s)",
            self._model_arch.dtype or "bf16",
            profile.backbone_precision or "(base)",
            profile.cp_precision or "(base)",
            profile.code2wav_precision or "(base)",
            variant,
        )
        cp_prec = (profile.cp_precision or self._model_arch.dtype or "bf16").lower()
        if cp_prec in ("bf16", "fp16"):
            logger.warning(
                "Code Predictor precision is %s — bf16/fp16 CP causes near-tie "
                "argmax flips and streaming hallucination. Rebuild with "
                "--cp-precision fp32 unless this is intentional.",
                cp_prec,
            )

        # When the manifest carries no engine_profile bounds, the hard checks
        # below are skipped and the only remaining guard is the executor's silent
        # clamp to the loaded plan. Make that loud so an operator-requested limit
        # is never quietly lowered.
        if profile.max_batch_size <= 0 and profile.max_seq_len <= 0:
            logger.warning(
                "Manifest has no engine_profile bounds for variant '%s'; runtime "
                "max_batch/max_seq are NOT pre-validated and will be silently "
                "clamped to the loaded TRT plan. Rebuild Phase B to record bounds.",
                variant,
            )

        if profile.max_batch_size > 0 and self._max_batch > profile.max_batch_size:
            raise ValueError(
                f"runtime max_batch_size={self._max_batch} exceeds engine profile "
                f"max_batch_size={profile.max_batch_size} for variant '{variant}'. "
                "Lower --max-batch / ENGINE_SCHEDULER_MAX_BATCH_SIZE, or rebuild Phase B with "
                f"`bash scripts/bash/build_engines.sh --variant {variant} "
                f"--max-batch-size {self._max_batch}`."
            )
        if profile.max_seq_len > 0 and self._max_seq_len > profile.max_seq_len:
            raise ValueError(
                f"runtime max_seq_len={self._max_seq_len} exceeds engine profile "
                f"max_seq_len={profile.max_seq_len} for variant '{variant}'. "
                "Lower --max-seq-len / ENGINE_SCHEDULER_MAX_SEQ_LEN, or rebuild Phase B with "
                f"`bash scripts/bash/build_engines.sh --variant {variant} "
                f"--max-seq-len {self._max_seq_len}`."
            )

    async def start(self) -> None:
        """Initialize all components, warm up, and start the engine thread."""
        self._loop = asyncio.get_event_loop()
        self._async_inbox = asyncio.Queue(maxsize=4096)

        self._tokenizer = LightQwen3TTSTokenizer(self._tokenizer_dir)
        self._ref_audio_processor = ReferenceAudioProcessor(
            self._engine_dir,
            self._model_arch.variant,
            device_id=self._device_id,
            cache_enabled=self._cfg.reference_cache.enabled,
            cache_max_entries=self._cfg.reference_cache.max_entries,
        )
        self._prime_configured_reference_cache()

        model_config = to_model_config(self._model_arch, self._cfg)
        sampling = self._cfg.sampling
        package_paths = (
            resolve_model_package_paths(self._cfg.paths.model_package_dir)
            if self._cfg.paths.model_package_dir
            else None
        )
        runtime_artifact = package_paths.runtime_artifact_path if package_paths else ""
        if runtime_artifact:
            logger.info(
                "Resolved model package: package=%s mode=%s runtime_artifact=%s",
                package_paths.package_dir,
                package_paths.engine_mode,
                runtime_artifact,
            )
        self._executor = Executor(
            engine_dir=self._engine_dir,
            weights_dir=self._weights_dir,
            device_id=self._device_id,
            max_batch_size=self._max_batch,
            max_seq_len=self._max_seq_len,
            model_config=model_config,
            do_sample=sampling.do_sample,
            temperature=sampling.temperature,
            repetition_penalty=sampling.repetition_penalty,
            random_seed=sampling.random_seed,
        )
        try:
            self._executor.load()
        except Exception as exc:
            if _is_cuda_oom(exc):
                # The artifact fingerprint validates GPU identity (SM/driver/TRT),
                # not memory capacity, so an engine sized for a larger GPU passes
                # the check and then OOMs here. Make that actionable.
                raise RuntimeError(
                    f"GPU out of memory loading the engine for variant "
                    f"'{self._model_arch.variant}' (max_batch_size={self._max_batch}). "
                    "The Phase B profile was likely sized for a larger GPU than this host; "
                    "lower --max-batch / ENGINE_SCHEDULER_MAX_BATCH_SIZE, or rebuild Phase B "
                    "for this GPU's memory."
                ) from exc
            raise
        self._max_batch = self._executor.max_batch_size
        self._max_seq_len = self._executor.max_seq_len

        # Cross-check the loaded plan's prefill bound against the manifest so a
        # mismatch surfaces at startup rather than mid-stream on the first
        # over-long request (the per-request guard lives deep in the executor).
        _prof_max_in = self._model_arch.engine_profile.max_input_len
        _plan_max_in = self._executor.max_input_len
        if _prof_max_in > 0 and _plan_max_in > 0 and _prof_max_in != _plan_max_in:
            logger.warning(
                "manifest max_input_len=%d disagrees with loaded TRT plan "
                "max_input_len=%d (variant=%s); prefill is enforced at the plan value.",
                _prof_max_in,
                _plan_max_in,
                self._model_arch.variant or "unknown",
            )

        sc = self._cfg.spliter
        self._frontend = FrontendInterface(
            engine_inbox=self._async_inbox,
            tokenizer=self._tokenizer,
            max_sessions=self._max_sessions,
            engine_max_decode_len=self._max_seq_len,
            prefill_len=sc.prefill_len,
            ema_ratio=sc.ema_ratio_initial,
            max_concurrent_segments=sc.max_concurrent_segments,
            ema_alpha=sc.ema_alpha,
            ema_overflow_alpha=sc.ema_overflow_alpha,
            ema_min_ratio=sc.ema_min_ratio,
            ema_max_ratio=sc.ema_max_ratio,
            safety_margin=sc.safety_margin,
            l1_split_cap_ratio=sc.l1_split_cap_ratio,
            l2_split_cap_ratio=sc.l2_split_cap_ratio,
            l3_split_cap_ratio=sc.l3_split_cap_ratio,
        )
        prefill_builder = None
        if self._weights_dir:
            try:
                pf = self._cfg.prefill
                try:
                    emb_weights = EmbeddingWeights(
                        self._weights_dir,
                        self._device_id,
                        default_speaker=pf.default_speaker,
                        fallback_speaker=pf.fallback_speaker,
                    )
                except Exception as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    logger.warning(
                        "Loading embedding weights on CUDA failed with OOM; retrying on CPU: %s",
                        exc,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    emb_weights = EmbeddingWeights(
                        self._weights_dir,
                        self._device_id,
                        device="cpu",
                        default_speaker=pf.default_speaker,
                        fallback_speaker=pf.fallback_speaker,
                    )
                # The engine trusts the manifest's architecture, but the actual
                # weights are loaded separately. If they describe different models
                # (stale/foreign manifest), fail cleanly at startup instead of
                # crashing on a raw tensor-shape mismatch at first inference.
                if (
                    self._model_arch.hidden_size
                    and emb_weights.hidden_size != self._model_arch.hidden_size
                ):
                    raise ValueError(
                        f"weights hidden_size={emb_weights.hidden_size} disagrees with "
                        f"manifest architecture hidden_size={self._model_arch.hidden_size} "
                        f"for variant '{self._model_arch.variant}': the loaded weights and "
                        "triton_manifest.json describe different models — re-export/rebuild."
                    )
                self._executor.set_embedding_weights(emb_weights)
                prefill_builder = PrefillBuilder(
                    emb_weights,
                    self._tokenizer,
                    output_device=self._executor._device,
                )
                logger.info(
                    "PrefillBuilder loaded from %s (weights_device=%s, output_device=%s)",
                    self._weights_dir,
                    emb_weights.device,
                    self._executor._device,
                )
            except Exception as e:
                logger.warning("Could not load embedding weights: %s", e)

        if self._cfg.server.warmup_rounds > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._executor.warmup(n_rounds=self._cfg.server.warmup_rounds)

        sched = self._cfg.scheduler
        pc = self._cfg.prefix_cache
        mlfq_cfg = MLFQConfig(
            q1_threshold=sched.mlfq_q1_threshold,
            q2_threshold=sched.mlfq_q2_threshold,
            aging_interval=sched.mlfq_aging_interval,
            starvation_limit=sched.mlfq_starvation_limit,
        )

        self._engine_loop = EngineLoop(
            engine_inbox=self._engine_inbox,
            async_loop=self._loop,
            executor=self._executor,
            prefill_builder=prefill_builder,
            max_batch_size=self._max_batch,
            mlfq_config=mlfq_cfg,
            prefix_cache_max_entries=pc.max_entries if pc.enabled else 0,
            prefix_cache_max_len=pc.max_prefix_len,
            max_idle_sec=sched.max_idle_sec,
            max_queue_size=sched.max_queue_size,
            session_timeout_sec=sched.session_timeout_sec,
            min_pad_steps=sched.min_pad_steps,
            pad_silence_peak_threshold=sched.pad_silence_peak_threshold,
            pad_silence_mean_abs_threshold=sched.pad_silence_mean_abs_threshold,
            max_slots_per_session=self._cfg.spliter.max_concurrent_segments,
        )
        self._engine_loop.start()

        self._relay_task = asyncio.create_task(self._relay_inbox())
        logger.info(
            "TTS Engine started (max_batch=%d, max_sessions=%d, mlfq=%s, prefix_cache=%s)",
            self._max_batch,
            self._max_sessions,
            "enabled",
            "enabled" if pc.enabled else "disabled",
        )

        await self._prime_prefix_cache()

    async def _prime_prefix_cache(self) -> None:
        """Prewarm the prefix KV cache for configured speakers.

        The executor warmup warms TRT/CUDA compute but not the prefix cache,
        which is keyed by (task_type, language, speaker, ...) and populated on
        the first real prefill.  For each configured speaker we run one real
        (output-discarded) synthesis so its prefix is cached and the first real
        request hits it (cold ~50ms TTFT -> warm ~20ms).  Requires prefix cache
        enabled and a running engine loop (called at the end of ``start``).
        """
        speakers = self._cfg.server.prewarm_speakers
        if not speakers or not self._cfg.prefix_cache.enabled:
            return

        timeout = self._cfg.server.request_timeout_sec
        for speaker in speakers:
            speaker = str(speaker).strip()
            if not speaker:
                continue
            session_id = f"__prewarm__{speaker}"
            done = asyncio.Event()

            async def _on_audio(_sid: str, _data: bytes) -> None:
                return  # discard warmup audio

            async def _on_done(_sid: str, _metrics: dict, _done=done) -> None:
                _done.set()

            try:
                await self.start_session(
                    session_id,
                    config=SessionConfig(task_type="custom_voice", speaker=speaker),
                    on_audio=_on_audio,
                    on_done=_on_done,
                )
                await self.push_text_input(session_id, "你好")
                await self.mark_input_complete(session_id)
                await asyncio.wait_for(done.wait(), timeout=timeout)
                logger.info("Prefix cache prewarmed for speaker %r", speaker)
            except Exception as exc:  # noqa: BLE001 — prewarm must never block startup
                logger.warning(
                    "Prefix cache prewarm failed for speaker %r: %s", speaker, exc
                )

    async def stop(self) -> None:
        if self._relay_task:
            self._relay_task.cancel()
        if self._engine_loop:
            self._engine_loop.stop()
        if self._executor:
            self._executor.shutdown()
        logger.info("TTS Engine stopped")

    # ------------------------------------------------------------------
    # Public API (called by Gateway / gRPC handlers)
    # ------------------------------------------------------------------

    async def synthesize_stream(
        self,
        session_id: str,
        *,
        speaker_key: Optional[str] = None,
        task_type: Optional[str] = None,
        ref_audio: Optional[bytes] = None,
    ) -> AsyncIterator[bytes]:
        """Create a session and yield audio chunks as they are produced."""
        audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        async def on_audio(sid: str, data: bytes) -> None:
            await audio_queue.put(data)

        async def on_done(sid: str, metrics: dict) -> None:
            await audio_queue.put(None)

        await self.start_session(
            session_id,
            config=SessionConfig(
                task_type=task_type or "",
                speaker=speaker_key,
                ref_audio=ref_audio,
            ),
            on_audio=on_audio,
            on_done=on_done,
        )

        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def start_session(
        self,
        session_id: str,
        *,
        config: SessionConfig,
        on_audio=None,
        on_done=None,
        on_event=None,
    ):
        """Create a configured session through the frontend interface.

        This is the public session-entry API for gateways/adapters. It keeps the
        transport layer from reaching into frontend internals directly.
        """
        # Hop to a worker thread only when reference-audio feature extraction
        # will actually run (voice_clone with raw ref audio) — that is real
        # blocking work.  For everything else validation is microseconds of
        # string checks, while an unconditional to_thread costs two extra
        # event-loop requeues per open; under a 128-way burst each requeue
        # waits behind the whole ready queue, and this dominated session
        # ingest serialization (t0→request.accepted avg ~83ms, measured via
        # workspace/mp_burst_probe.py).
        if config.ref_audio and config.spk_embedding is None:
            await asyncio.to_thread(self._validate_and_prepare_session_config, config)
        else:
            self._validate_and_prepare_session_config(config)
        return await self._frontend.create_session(
            session_id,
            config=config,
            on_audio=on_audio,
            on_done=on_done,
            on_event=on_event,
        )

    async def push_text_input(self, session_id: str, text: str) -> None:
        """Transport-facing text ingress; frontend converts it to tokens."""
        await self._frontend.push_text_input(session_id, text)

    async def feed_full_text(self, session_id: str, text: str) -> None:
        """Offline mode: set complete text, pre-split, drive all segments."""
        await self._frontend.feed_full_text(session_id, text)

    async def mark_input_complete(self, session_id: str) -> None:
        """Signal that the transport has finished sending text input."""
        await self._frontend.mark_input_complete(session_id)

    async def cancel(self, session_id: str) -> None:
        await self._frontend.cancel_session(session_id)

    def health_stats(self) -> dict:
        """Return engine health metrics (safe to call from asyncio thread)."""
        if self._engine_loop is None:
            return {"running": False}
        stats = self._engine_loop.health_stats()
        stats["variant"] = self._model_arch.variant
        stats["loaded_model_type"] = self._loaded_model_type()
        if self._model_arch.supported_task_types:
            stats["declared_supported_task_types"] = list(
                self._model_arch.supported_task_types
            )
        profile = self._model_arch.engine_profile
        if profile.max_batch_size or profile.max_seq_len:
            stats["engine_profile"] = {
                "max_batch_size": profile.max_batch_size,
                "max_input_len": profile.max_input_len,
                "max_seq_len": profile.max_seq_len,
                "engine_dtype": profile.engine_dtype,
                "triton_io_float_dtype": profile.triton_io_float_dtype,
            }
        if self._ref_audio_processor is not None:
            stats.update(self._reference_capabilities())
        return stats

    def describe_capabilities(self) -> dict:
        """Return static standalone capability metadata for clients."""
        ref_caps = self._reference_capabilities()

        return normalize_capabilities(
            {
                "variant": self._model_arch.variant,
                "loaded_model_type": self._loaded_model_type(),
                "declared_supported_task_types": list(
                    self._model_arch.supported_task_types or ()
                ),
                "supported_input_modes": [
                    "token",
                    "clause",
                    "long_segment",
                    "full_text",
                ],
                "supported_group_policies": ["none", "auto"],
                "supported_audio_formats": [
                    {"encoding": "pcm_f32", "sample_rate": 24000, "channels": 1},
                    {"encoding": "pcm_f32", "sample_rate": 16000, "channels": 1},
                    {"encoding": "pcm_s16le", "sample_rate": 24000, "channels": 1},
                    {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
                ],
                **ref_caps,
                "engine_profile": {
                    "max_batch_size": self._model_arch.engine_profile.max_batch_size,
                    "max_input_len": self._model_arch.engine_profile.max_input_len,
                    "max_seq_len": self._model_arch.engine_profile.max_seq_len,
                    "engine_dtype": self._model_arch.engine_profile.engine_dtype,
                    "triton_io_float_dtype": self._model_arch.engine_profile.triton_io_float_dtype,
                },
            }
        )

    def _reference_capabilities(self) -> dict:
        support = (
            self._ref_audio_processor.support
            if self._ref_audio_processor is not None
            else None
        )
        loaded_model_type = self._loaded_model_type()
        speaker_encoder_available = bool(
            support is not None and support.speaker_encoder_available
        )
        ref_codec_available = bool(support is not None and support.ref_codec_available)
        icl_available = bool(support is not None and support.icl_available)
        if loaded_model_type in ("base", "icl"):
            ref_audio_available = icl_available
        else:
            ref_audio_available = False

        ref_audio_reason = ""
        ref_codec_reason = ""
        if support is None:
            ref_audio_reason = "reference-audio processor unavailable"
        else:
            ref_audio_reason = support.reason or ""
            ref_codec_reason = support.ref_codec_reason or ""
            if loaded_model_type in ("base", "icl") and not ref_audio_available:
                ref_audio_reason = ref_audio_reason or ref_codec_reason

        return {
            "ref_audio_available": ref_audio_available,
            "speaker_encoder_available": speaker_encoder_available,
            "ref_codec_available": ref_codec_available,
            "icl_available": icl_available,
            "ref_audio_max_duration_sec": (
                float(support.ref_audio_max_duration_sec)
                if support is not None
                else 0.0
            ),
            "ref_c2w_warm_state_available": bool(
                support is not None and support.ref_c2w_warm_state_available
            ),
            "ref_audio_reason": ref_audio_reason,
            "ref_codec_reason": ref_codec_reason,
        }

    def _validate_and_prepare_session_config(self, config: SessionConfig) -> None:
        self._validate_session_config(config)
        self._prepare_reference_audio_features(config)

    def _validate_session_config(self, config: SessionConfig) -> None:
        loaded_model_type = self._loaded_model_type()
        requested = (config.task_type or "").strip()

        if loaded_model_type != "unknown":
            if requested and requested != loaded_model_type:
                raise ValueError(
                    f"standalone engine has loaded model_type '{loaded_model_type}', "
                    f"but client requested task_type '{requested}'. "
                    "Omit task_type or send the same loaded model type."
                )
            model_type = loaded_model_type
        elif not requested:
            raise ValueError(
                "task_type is required because the loaded engine manifest does not declare tts_model_type"
            )
        else:
            model_type = requested

        self._validate_model_specific_fields(model_type, config)
        config.task_type = self._internal_task_type_for_model(model_type)
        self._log_session_config_debug(
            "session_config_resolved",
            loaded_model_type=loaded_model_type,
            requested_task_type=requested,
            resolved_model_type=model_type,
            internal_task_type=config.task_type,
            config=config,
        )

        task_type = (config.task_type or "").strip()
        if task_type == "voice_clone":
            if not config.ref_audio:
                raise ValueError("ref_audio is required for task_type 'voice_clone'")
            support = (
                self._ref_audio_processor.support
                if self._ref_audio_processor is not None
                else None
            )
            if support is None or not support.speaker_encoder_available:
                reason = (
                    support.reason
                    if support is not None
                    else "reference-audio processor unavailable"
                )
                raise ValueError(
                    f"voice_clone is not available in standalone mode: {reason}"
                )
            if not config.x_vector_only and not support.ref_codec_available:
                reason = support.ref_codec_reason or support.reason
                raise ValueError(
                    f"voice_clone ICL is not available in standalone mode: {reason}"
                )

    def _loaded_model_type(self) -> str:
        model_type = (self._model_arch.tts_model_type or "").strip()
        if model_type and model_type != "unknown":
            return model_type
        supported = tuple(
            t.strip()
            for t in self._model_arch.supported_task_types or ()
            if t and t.strip()
        )
        if len(supported) == 1:
            return supported[0]
        return "unknown"

    def _internal_task_type_for_model(self, model_type: str) -> str:
        normalized = (model_type or "").strip()
        internal = _EXTERNAL_TO_INTERNAL_TASK_TYPE.get(normalized)
        if internal:
            return internal
        raise ValueError(f"Unknown loaded model type: '{normalized}'")

    def _apply_default_base_reference(self, config: SessionConfig) -> None:
        using_default_ref_audio = False
        if not config.ref_audio:
            config.ref_audio = self._load_default_base_ref_audio()
            using_default_ref_audio = True
        if using_default_ref_audio and not (config.ref_text or "").strip():
            config.ref_text = _DEFAULT_BASE_REF_TEXT

    def _load_default_base_ref_audio(self) -> bytes:
        if self._default_base_ref_audio is not None:
            return self._default_base_ref_audio

        checked_paths = []
        for raw_path in _DEFAULT_BASE_REF_AUDIO_CANDIDATES:
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            checked_paths.append(str(path))
            try:
                if not path.is_file():
                    continue
                data = path.read_bytes()
            except OSError as exc:
                logger.warning(
                    "Could not read default Base ref_audio %s: %s", path, exc
                )
                continue
            if not data:
                logger.warning("Default Base ref_audio %s is empty", path)
                continue
            self._default_base_ref_audio = data
            logger.info(
                "Loaded default Base ref_audio from %s (%d bytes)", path, len(data)
            )
            return data

        raise ValueError(
            "ref_audio is required for loaded model_type 'base' and no default Base "
            "reference audio was found; checked: " + ", ".join(checked_paths)
        )

    def _resolve_reference_entry(self, ref_id: str) -> tuple[bytes, str, str, str]:
        refs = self._cfg.references
        entries = refs.entries or {}
        normalized = (ref_id or "").strip().lower()
        if not normalized:
            raise ValueError("reference_not_found: empty reference alias")

        raw_entry = None
        raw_key = ""
        for key, value in entries.items():
            if str(key).strip().lower() == normalized:
                raw_key = str(key).strip()
                raw_entry = value
                break
        if raw_entry is None:
            raise ValueError(f"reference_not_found: {ref_id!r}")
        if not isinstance(raw_entry, dict):
            raise ValueError(f"reference_not_found: entry {ref_id!r} is not an object")

        audio_path = str(raw_entry.get("audio_path") or "").strip()
        ref_text = str(raw_entry.get("ref_text") or "").strip()
        ref_text_path = str(
            raw_entry.get("ref_text_path") or raw_entry.get("text_path") or ""
        ).strip()
        language = str(raw_entry.get("language") or "").strip()
        if not audio_path:
            raise ValueError(f"reference_not_found: entry {ref_id!r} has no audio_path")
        if not ref_text and ref_text_path:
            ref_text = self._read_reference_text_path(ref_text_path)
        if not ref_text:
            raise ValueError(
                f"ref_text_required: reference {ref_id!r} has empty ref_text"
            )
        return (
            self._read_reference_audio_path(audio_path),
            ref_text,
            raw_key or normalized,
            language,
        )

    def _reference_path_candidates(self, raw_path: str) -> list[Path]:
        path = Path(raw_path).expanduser()
        candidates = [path] if path.is_absolute() else []
        if not path.is_absolute():
            if self._cfg.paths.model_package_dir:
                candidates.append(Path(self._cfg.paths.model_package_dir) / path)
            candidates.append(Path(__file__).resolve().parents[1] / path)
            candidates.append(Path.cwd() / path)
        return candidates

    def _read_reference_audio_path(self, raw_path: str) -> bytes:
        candidates = self._reference_path_candidates(raw_path)
        checked: list[str] = []
        for candidate in candidates:
            checked.append(str(candidate))
            try:
                if not candidate.is_file():
                    continue
                data = candidate.read_bytes()
            except OSError as exc:
                logger.warning("Could not read reference audio %s: %s", candidate, exc)
                continue
            if data:
                return data
            logger.warning("Reference audio %s is empty", candidate)
        raise ValueError(
            "reference_not_found: reference audio path not found or empty; checked: "
            + ", ".join(checked)
        )

    def _read_reference_text_path(self, raw_path: str) -> str:
        candidates = self._reference_path_candidates(raw_path)
        checked: list[str] = []
        for candidate in candidates:
            checked.append(str(candidate))
            try:
                if not candidate.is_file():
                    continue
                text = candidate.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.warning("Could not read reference text %s: %s", candidate, exc)
                continue
            if text:
                return text
            logger.warning("Reference text %s is empty", candidate)
        raise ValueError(
            "ref_text_required: reference text path not found or empty; checked: "
            + ", ".join(checked)
        )

    def _resolve_default_reference(self) -> tuple[bytes, str, str, str, str]:
        refs = self._cfg.references
        if refs.entries:
            ref_id = (refs.default or "default").strip() or "default"
            try:
                audio, text, resolved_id, language = self._resolve_reference_entry(
                    ref_id
                )
            except ValueError as exc:
                raise ValueError(f"default_reference_missing: {exc}") from exc
            return audio, text, resolved_id, "default", language
        try:
            return (
                self._load_default_base_ref_audio(),
                _DEFAULT_BASE_REF_TEXT,
                "default",
                "default",
                "",
            )
        except ValueError as exc:
            raise ValueError(f"default_reference_missing: {exc}") from exc

    def _prime_configured_reference_cache(self) -> None:
        if self._ref_audio_processor is None:
            return
        if not self._cfg.reference_cache.enabled:
            return
        if self._loaded_model_type() not in ("base", "icl", "voice_clone"):
            return

        support = self._ref_audio_processor.support
        if not support.icl_available:
            reason = support.ref_codec_reason or support.reason
            logger.info("Skipping Base/ICL reference cache priming: %s", reason)
            return

        targets: list[tuple[str, str, bytes]] = []
        seen_ids: set[str] = set()
        try:
            audio, _text, ref_id, source, _language = self._resolve_default_reference()
            targets.append((source, ref_id, audio))
            seen_ids.add(ref_id.strip().lower())
        except Exception as exc:
            logger.warning(
                "Could not resolve default Base/ICL reference for cache priming: %s",
                exc,
            )

        for raw_id in (self._cfg.references.entries or {}).keys():
            ref_id = str(raw_id).strip()
            if not ref_id or ref_id.lower() in seen_ids:
                continue
            try:
                audio, _text, resolved_id, _language = self._resolve_reference_entry(
                    ref_id
                )
            except Exception as exc:
                logger.warning(
                    "Could not resolve Base/ICL reference %s for cache priming: %s",
                    ref_id,
                    exc,
                )
                continue
            targets.append(("registry", resolved_id, audio))
            seen_ids.add(resolved_id.strip().lower())

        for source, ref_id, audio in targets:
            try:
                features = self._ref_audio_processor.process(
                    audio,
                    require_ref_codec=True,
                    cache_tag="voice_clone_icl",
                )
                logger.info(
                    "Primed Base/ICL reference cache: %s (%s, %.2fs)",
                    ref_id,
                    source,
                    features.duration_sec,
                )
            except Exception as exc:
                logger.warning(
                    "Could not prime Base/ICL reference cache for %s (%s): %s",
                    ref_id,
                    source,
                    exc,
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_reference_language(self, config: SessionConfig, language: str) -> None:
        language = (language or "").strip()
        if not language:
            return
        current = (config.language or "").strip().lower()
        if not current or current == "auto":
            config.language = language

    def _mark_reference_metadata(
        self,
        config: SessionConfig,
        *,
        source: str,
        ref_id: Optional[str] = None,
    ) -> None:
        config.ref_source = source
        config.ref_id = (ref_id or config.ref_id or "").strip() or None
        if config.ref_audio:
            config.ref_audio_sha256 = hashlib.sha256(config.ref_audio).hexdigest()
        if (config.ref_text or "").strip():
            config.ref_text_hash = hashlib.sha1(
                (config.ref_text or "").strip().encode("utf-8"),
            ).hexdigest()
        config.ref_preprocess_runtime = "trt"
        self._log_session_config_debug(
            "reference_metadata_resolved",
            loaded_model_type=self._loaded_model_type(),
            requested_task_type=config.task_type,
            resolved_model_type=self._loaded_model_type(),
            internal_task_type=config.task_type,
            config=config,
        )

    def _resolve_voice_clone_reference(
        self, model_type: str, config: SessionConfig
    ) -> None:
        normalized = (model_type or "").strip()
        has_audio = bool(config.ref_audio)
        has_text = bool((config.ref_text or "").strip())
        alias = (config.speaker or "").strip()

        if has_audio and has_text:
            self._mark_reference_metadata(
                config, source="explicit", ref_id=alias or None
            )
            config.speaker = None
            config.x_vector_only = False
            return

        if has_audio and not has_text:
            if normalized == "icl":
                raise ValueError(
                    "ref_text_required: ref_text is required for loaded model_type 'icl'"
                )
            self._mark_reference_metadata(
                config, source="explicit", ref_id=alias or None
            )
            config.speaker = None
            config.x_vector_only = True
            return

        if has_text and not has_audio:
            raise ValueError(
                f"ref_audio is required when ref_text is provided for loaded model_type '{normalized}'"
            )

        if alias:
            audio, text, resolved_id, language = self._resolve_reference_entry(alias)
            config.ref_audio = audio
            config.ref_text = text
            self._apply_reference_language(config, language)
            self._mark_reference_metadata(config, source="registry", ref_id=resolved_id)
            config.speaker = None
            config.x_vector_only = False
            return

        audio, text, resolved_id, source, language = self._resolve_default_reference()
        config.ref_audio = audio
        config.ref_text = text
        self._apply_reference_language(config, language)
        self._mark_reference_metadata(config, source=source, ref_id=resolved_id)
        config.speaker = None
        config.x_vector_only = False

    def _validate_model_specific_fields(
        self, model_type: str, config: SessionConfig
    ) -> None:
        normalized = (model_type or "").strip()
        if normalized == "base":
            self._resolve_voice_clone_reference(normalized, config)
            if not config.ref_audio:
                raise ValueError("ref_audio is required for loaded model_type 'base'")
            if config.instruct:
                raise ValueError(
                    "instruct is not supported for loaded model_type 'base'"
                )
            return

        if normalized in ("icl",):
            self._resolve_voice_clone_reference(normalized, config)
            if not config.ref_audio:
                raise ValueError("ref_audio is required for loaded model_type 'icl'")
            if not (config.ref_text or "").strip():
                raise ValueError(
                    "ref_text_required: ref_text is required for loaded model_type 'icl'"
                )
            if config.instruct:
                raise ValueError(
                    "instruct is not supported for loaded model_type 'icl'"
                )
            config.x_vector_only = False
            return

        if normalized in ("voice_design", "instruct"):
            if not (config.instruct or "").strip():
                raise ValueError(
                    "instruct is required for loaded model_type 'voice_design'"
                )
            if config.speaker:
                raise ValueError(
                    "speaker is not supported for loaded model_type 'voice_design'"
                )
            if config.ref_audio:
                raise ValueError(
                    "ref_audio is not supported for loaded model_type 'voice_design'"
                )
            if config.ref_text:
                raise ValueError(
                    "ref_text is not supported for loaded model_type 'voice_design'"
                )
            if config.x_vector_only:
                raise ValueError(
                    "x_vector_only is not supported for loaded model_type 'voice_design'"
                )
            return

        if normalized == "custom_voice":
            if config.ref_audio:
                raise ValueError(
                    "ref_audio is not supported for loaded model_type 'custom_voice'"
                )
            if config.ref_text:
                raise ValueError(
                    "ref_text is not supported for loaded model_type 'custom_voice'"
                )
            if config.x_vector_only:
                raise ValueError(
                    "x_vector_only is not supported for loaded model_type 'custom_voice'"
                )
            return

        if normalized == "voice_clone":
            return

        raise ValueError(f"Unknown loaded model type: '{normalized}'")

    def _prepare_reference_audio_features(self, config: SessionConfig) -> None:
        if (config.task_type or "").strip() != "voice_clone":
            return
        if config.spk_embedding is not None:
            return
        if not config.ref_audio:
            return
        if self._ref_audio_processor is None:
            raise ValueError("reference-audio processor unavailable")

        features = self._ref_audio_processor.process(
            config.ref_audio,
            require_ref_codec=not bool(config.x_vector_only),
            cache_tag=(
                "voice_clone_icl"
                if not bool(config.x_vector_only)
                else "voice_clone_xvec"
            ),
        )
        config.spk_embedding = features.spk_embedding
        config.ref_feature_cache_key = getattr(features, "cache_key", "") or ""
        for warning_msg in getattr(features, "warnings", []) or []:
            config.ref_warnings.append(str(warning_msg))
        if not config.x_vector_only:
            config.ref_codec_sum_vec = features.ref_codec_sum_vec
            config.ref_audio_codes = features.ref_audio_codes
            config.ref_c2w_kv = features.ref_c2w_kv
            config.ref_c2w_conv_states = features.ref_c2w_conv_states
            config.ref_c2w_transconv_states = features.ref_c2w_transconv_states
            config.ref_c2w_frame_idx = features.ref_c2w_frame_idx
        self._log_reference_feature_debug(config)

    def _log_session_config_debug(
        self,
        stage: str,
        *,
        loaded_model_type: str,
        requested_task_type: str,
        resolved_model_type: str,
        internal_task_type: str,
        config: SessionConfig,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Session config observability: %s",
            {
                "stage": stage,
                "loaded_model_type": loaded_model_type,
                "requested_task_type": requested_task_type,
                "resolved_model_type": resolved_model_type,
                "internal_task_type": internal_task_type,
                "speaker": config.speaker,
                "language": config.language,
                "input_mode": config.input_mode.value if config.input_mode else "",
                "group_policy": config.group_policy.value
                if config.group_policy
                else "",
                "has_ref_audio": bool(config.ref_audio),
                "has_ref_text": bool((config.ref_text or "").strip()),
                "has_instruct": bool((config.instruct or "").strip()),
                "x_vector_only": bool(config.x_vector_only),
                "ref_source": config.ref_source,
                "ref_id": config.ref_id,
                "ref_audio_sha256_prefix": config.ref_audio_sha256[:12],
                "ref_text_hash_prefix": config.ref_text_hash[:12],
                "ref_preprocess_runtime": config.ref_preprocess_runtime,
            },
        )

    def _log_reference_feature_debug(self, config: SessionConfig) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Reference codec observability: %s",
            {
                "task_type": config.task_type,
                "ref_source": config.ref_source,
                "ref_id": config.ref_id,
                "x_vector_only": bool(config.x_vector_only),
                "ref_feature_cache_key": config.ref_feature_cache_key,
                "spk_embedding_ready": config.spk_embedding is not None,
                "ref_codec_sum_vec_ready": config.ref_codec_sum_vec is not None,
                "ref_audio_codes_ready": config.ref_audio_codes is not None,
                "ref_c2w_kv_ready": config.ref_c2w_kv is not None,
                "ref_c2w_conv_states_ready": bool(config.ref_c2w_conv_states),
                "ref_c2w_transconv_states_ready": bool(config.ref_c2w_transconv_states),
                "ref_c2w_frame_idx": int(config.ref_c2w_frame_idx),
                "warnings": list(config.ref_warnings),
            },
        )

    # ------------------------------------------------------------------
    # Internal: relay asyncio.Queue → stdlib queue.Queue
    # ------------------------------------------------------------------

    async def _relay_inbox(self) -> None:
        try:
            while True:
                req = await self._async_inbox.get()
                self._engine_inbox.put_nowait(req)
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _install_async_logging() -> None:
    """Route all logging through a queue to a dedicated writer thread.

    Lifecycle events are JSON log lines written synchronously to stdout
    (docker json-file driver) from the hot threads — the engine loop and the
    gateway asyncio loop.  Under a 128-session burst that stdout write + GIL
    contention measurably stretches both.  stdlib QueueHandler/QueueListener
    makes hot-thread logging enqueue-only; formatting and I/O happen on the
    listener thread.
    """
    import atexit
    import queue as _queue
    from logging.handlers import QueueHandler, QueueListener

    root = logging.getLogger()
    handlers = root.handlers[:]
    if not handlers or any(isinstance(h, QueueHandler) for h in handlers):
        return
    log_queue: _queue.SimpleQueue = _queue.SimpleQueue()
    for h in handlers:
        root.removeHandler(h)
    root.addHandler(QueueHandler(log_queue))
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()
    atexit.register(listener.stop)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _install_async_logging()

    parser = argparse.ArgumentParser(description="TTS Engine Server")
    parser.add_argument(
        "--config",
        default="engine.yaml",
        help="Path to engine.yaml config file (default: engine.yaml)",
    )
    parser.add_argument(
        "--model-package-dir",
        default="",
        help="Path to shared model package (tts_orchestrator/<version>)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--max-batch", type=int, default=0, help="Override scheduler.max_batch_size"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=0, help="Override scheduler.max_seq_len"
    )
    parser.add_argument(
        "--max-sessions", type=int, default=0, help="Override server.max_sessions"
    )
    parser.add_argument("--port", type=int, default=0, help="Override server.port")
    parser.add_argument(
        "--ws-port",
        type=int,
        default=-1,
        help="Override server.websocket_port (-1 keeps config)",
    )
    parser.add_argument("--ws-path", default="", help="Override server.websocket_path")
    args = parser.parse_args()

    cli_overrides: dict = {}
    if args.model_package_dir:
        cli_overrides.setdefault("paths", {})["model_package_dir"] = (
            args.model_package_dir
        )
    if args.max_batch > 0:
        cli_overrides.setdefault("scheduler", {})["max_batch_size"] = args.max_batch
    if args.max_seq_len > 0:
        cli_overrides.setdefault("scheduler", {})["max_seq_len"] = args.max_seq_len
    if args.max_sessions > 0:
        cli_overrides.setdefault("server", {})["max_sessions"] = args.max_sessions
    if args.port > 0:
        cli_overrides.setdefault("server", {})["port"] = args.port
    if args.ws_port >= 0:
        cli_overrides.setdefault("server", {})["websocket_port"] = args.ws_port
    if args.ws_path:
        cli_overrides.setdefault("server", {})["websocket_path"] = args.ws_path

    cfg = load_config(args.config, cli_overrides=cli_overrides)

    # Install the observability control plane and drive the root log level from
    # the resolved config. ENGINE_OBS_LEVEL is accepted as a shorthand alias for
    # ENGINE_OBSERVABILITY_LEVEL (the generic env override).
    obs_alias = os.environ.get("ENGINE_OBS_LEVEL", "").strip()
    if obs_alias:
        cfg.observability.level = obs_alias
    obs.configure(
        cfg.observability.level,
        cfg.observability.max_session_level,
        text_capture=cfg.observability.text_capture,
        text_preview_chars=cfg.observability.text_preview_chars,
        health_interval_sec=cfg.observability.health_interval_sec,
    )
    logging.getLogger().setLevel(obs.to_logging_level(obs.global_level()))

    # L3: at the dump level, auto-enable the EngineDebugDumper from config so the
    # single `observability.level: dump` knob drives the tensor dumper (it reads
    # ENGINE_DUMP_* at construction). Explicit ENGINE_DUMP_* env always wins.
    if obs.global_level() >= obs.ObsLevel.DUMP and cfg.observability.dump_dir:
        os.environ.setdefault("ENGINE_DUMP_DIR", cfg.observability.dump_dir)
        os.environ.setdefault("ENGINE_DUMP_LIMIT", str(cfg.observability.dump_limit))
        os.environ.setdefault(
            "ENGINE_DUMP_INCLUDE_WAV",
            "true" if cfg.observability.dump_include_wav else "false",
        )
        if cfg.observability.dump_sessions:
            os.environ.setdefault(
                "ENGINE_DUMP_SESSIONS", cfg.observability.dump_sessions
            )

    if cfg.paths.model_package_dir:
        apply_model_package_paths(
            cfg,
            resolve_model_package_paths(cfg.paths.model_package_dir),
        )

    # Strict runtime fingerprint guard.  Refuses to start when the engine
    # plan in this package was compiled for a GPU / TRT version that
    # disagrees with what this process can actually use.  Bypass with
    # QWEN3_ALLOW_FINGERPRINT_MISMATCH=1 (debugging only).  See
    # engine/runtime/fingerprint.py for details.
    if cfg.paths.model_package_dir:
        try:
            enforce_engine_fingerprint(
                cfg.paths.model_package_dir,
                device_index=args.device,
            )
        except FingerprintCheckError as exc:
            logger.error("Engine fingerprint check failed:\n%s", exc)
            raise SystemExit(2) from exc

    engine_dir = cfg.paths.engine_dir
    tokenizer_dir = cfg.paths.tokenizer_dir
    model_arch = load_model_manifest(engine_dir, cfg, tokenizer_dir=tokenizer_dir)

    async def run():
        engine = TTSEngine(
            config=cfg,
            model_arch=model_arch,
            device_id=args.device,
        )
        await engine.start()

        stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        port = cfg.server.port
        websocket_port = cfg.server.websocket_port
        websocket_path = cfg.server.websocket_path
        health_port = cfg.server.health_port

        grpc_task = None
        websocket_task = None
        health_task = None

        try:
            from .gateway.grpc_server import serve as grpc_serve

            grpc_task = asyncio.create_task(
                grpc_serve(engine, port, stop_event=stop_event),
            )
            logger.info("gRPC server launched on port %d", port)
        except Exception as e:
            logger.warning("gRPC server not started: %s", e)

        if websocket_port > 0:
            try:
                from .gateway.websocket_server import serve as websocket_serve

                websocket_task = asyncio.create_task(
                    websocket_serve(
                        engine,
                        websocket_port,
                        stop_event=stop_event,
                        path=websocket_path,
                    ),
                )
                logger.info(
                    "WebSocket server launched on port %d path %s",
                    websocket_port,
                    websocket_path,
                )
            except Exception as e:
                logger.warning("WebSocket server not started: %s", e)

        if health_port > 0:
            health_task = asyncio.create_task(
                _run_health_server(engine, health_port, stop_event),
            )

        logger.info("Engine ready, press Ctrl+C to stop")
        await stop_event.wait()

        if grpc_task:
            try:
                await grpc_task
            except Exception as e:
                logger.warning("gRPC server shutdown: %s", e)
        if websocket_task:
            try:
                await websocket_task
            except Exception as e:
                logger.warning("WebSocket server shutdown: %s", e)
        if health_task:
            try:
                await health_task
            except Exception as e:
                logger.warning("Health server shutdown: %s", e)
        await engine.stop()

    # uvloop cuts event-loop overhead 2-4x vs stock asyncio.  The gateway's
    # session dispatch and audio fan-out share this loop (and the GIL) with
    # the engine thread, so loop efficiency directly affects burst TTFT.
    try:
        import uvloop
    except ImportError:
        logger.info("uvloop not available; using stock asyncio event loop")
        asyncio.run(run())
    else:
        logger.info("Using uvloop event loop")
        uvloop.run(run())


async def _run_health_server(
    engine: TTSEngine,
    port: int,
    stop_event: asyncio.Event,
) -> None:
    """Minimal HTTP health / metrics endpoint.

    Uses aiohttp if available; otherwise falls back to a simple
    asyncio.start_server implementation.
    """
    import json as _json

    try:
        from aiohttp import web  # type: ignore[import-untyped]

        async def handle_health(request):
            stats = engine.health_stats()
            return web.json_response(stats)

        app = web.Application()
        app.router.add_get("/health", handle_health)
        app.router.add_get("/metrics", handle_health)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        try:
            await site.start()
            logger.info("Health/metrics HTTP server on port %d (aiohttp)", port)
            await stop_event.wait()
        finally:
            await runner.cleanup()
        return
    except ImportError:
        pass

    async def _handle_connection(reader, writer):
        await reader.read(4096)
        stats = engine.health_stats()
        body = _json.dumps(stats).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n" % len(body) + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(_handle_connection, "0.0.0.0", port)
    logger.info("Health/metrics HTTP server on port %d (asyncio)", port)
    async with server:
        serve_task = asyncio.create_task(server.serve_forever())
        try:
            await stop_event.wait()
        finally:
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    main()
