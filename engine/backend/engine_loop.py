"""Engine loop: runs in a dedicated thread, owns all GPU resources.

Pipeline design for high GPU utilization:

    ┌─────────────────── Iteration K ───────────────────────────┐
    │                                                           │
    │  Phase 1 (CPU):  Process step K-1 output                  │
    │     • scatter KV to pool                                  │
    │     • compute next_embed from codec_sum                   │
    │     • check EOS, send audio chunks                        │
    │                                                           │
    │  Phase 2 (CPU):  Prefill new sessions (rare, ~10ms each)  │
    │     • runs on separate CUDA stream                        │
    │     • only when new sessions arrive                       │
    │                                                           │
    │  Phase 3 (CPU→GPU):  Build & launch step K                │
    │     • pool gather KV → batched tensor                     │
    │     • launch TRT on compute_stream (non-blocking)         │
    │                                                           │
    │  Phase 4 (CPU ∥ GPU):  Housekeeping while GPU computes    │
    │     • drain inbox                                         │
    │     • evict idle slots                                    │
    │     • check session timeouts                              │
    │                                                           │
    │  Phase 5 (sync):  wait GPU → prev_output for next iter    │
    └───────────────────────────────────────────────────────────┘

    This ordering guarantees autoregressive correctness:
    step K reads KV that includes step K-1's output (Phase 1 runs
    BEFORE Phase 3).  CPU housekeeping overlaps with GPU compute.

Level 2 pipelining:
    Multiple segments of the same session may decode in parallel.
    Each segment owns its own KV slot.  The engine batches all active
    slots regardless of session, and routes results by (session_id,
    segment_idx).

    Prefill priority: FIRST_SEGMENT > CONTINUATION > PREFETCHED.

Scheduling:
    MLFQ (Multi-Level Feedback Queue) dynamically assigns decode
    priority per segment.  Anti-starvation aging boosts long-waiting
    segments to prevent starvation.

Slot management:
    Idle slots past max_idle_sec are evicted.  Backpressure prevents
    new sessions when the queue depth exceeds max_queue_size.

Prefix KV caching:
    System prompt KV tensors are cached across requests with the same
    speaker/task configuration, eliminating redundant prefill GPU work.

Thread safety:
    - Runs entirely in its own thread.
    - Reads from engine_inbox (stdlib queue.Queue, thread-safe).
    - Writes results via asyncio.loop.call_soon_threadsafe().
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Dict, Optional

import numpy as np
import torch

from ..core.mlfq import MLFQConfig, MLFQMeta, MLFQScheduler
from ..core.types import (
    EngineRequest,
    EngineResult,
    RequestPriority,
    RequestType,
    ResultType,
)
from ..core.lifecycle import LifecycleLogger
from ..core import observability as obs
from .executor import Executor, StepOutput
from .kv_cache_pool import SlotKVState
from .prefix_cache import PrefixKVCache
from .prefill import PrefillBuilder, PrefillPlan, TaskType, parse_task_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-segment tracking (engine-thread side)
# ---------------------------------------------------------------------------


class EngineSegment:
    """Engine thread's view of one segment: owns a KV slot + decode state."""

    __slots__ = (
        "session_id",
        "segment_idx",
        "slot",
        "state",
        "priority",
        "input_complete",
        "prefill_plan",
        "trailing_idx",
        "text_tokens_consumed",
        "decode_start_frame",
        "mlfq_meta",
        "pending_token_ids",
        "eos_trailing_added",
        "first_raw_audio_sent",
        "dequeued_at",
        "prefill_started_at",
        "prefill_completed_at",
        "cache_hit",
        "cache_tokens_reused",
        "max_decode_batch",
        "prefix_probe_key",
        "prefix_probe_task_type",
        "prefix_probe_done",
    )

    def __init__(
        self,
        session_id: str,
        segment_idx: int,
        priority: RequestPriority = RequestPriority.FIRST_SEGMENT,
    ):
        self.session_id = session_id
        self.segment_idx = segment_idx
        self.slot: Optional[SlotKVState] = None
        self.state: str = "pending_prefill"
        self.priority = priority
        self.input_complete: bool = False
        self.prefill_plan: Optional[PrefillPlan] = None
        self.trailing_idx: int = 0
        self.text_tokens_consumed: int = 0
        self.decode_start_frame: int = 0
        self.mlfq_meta: MLFQMeta = MLFQMeta()
        self.pending_token_ids: list[int] = []
        self.eos_trailing_added: bool = False
        self.first_raw_audio_sent: bool = False
        self.dequeued_at: Optional[float] = None
        self.prefill_started_at: Optional[float] = None
        self.prefill_completed_at: Optional[float] = None
        self.cache_hit: bool = False
        self.cache_tokens_reused: int = 0
        # Largest decode batch this segment was ever co-scheduled in (continuous
        # batching observability — answers "拼没拼 batch"). 0 until first decode.
        self.max_decode_batch: int = 0
        # Memoized prefix-cache key + task type for batch-admission probing.
        # Both are immutable per segment; cache membership is re-checked on
        # every probe (entries can be LRU-evicted between iterations).
        self.prefix_probe_key: Optional[str] = None
        self.prefix_probe_task_type: Optional[TaskType] = None
        self.prefix_probe_done: bool = False


class EngineSessionGroup:
    """Groups all segments belonging to one session."""

    __slots__ = (
        "session_id",
        "request",
        "result_queue",
        "segments",
        "input_complete_all",
        "created_at",
        "overflow_token_ids",
        "first_text_dequeued_at",
    )

    def __init__(self, session_id: str, request: EngineRequest):
        self.session_id = session_id
        self.request = request
        self.result_queue: Optional[asyncio.Queue] = request.result_queue
        self.segments: Dict[int, EngineSegment] = {}
        self.input_complete_all: bool = False
        self.created_at: float = time.monotonic()
        self.overflow_token_ids: list[int] = []
        self.first_text_dequeued_at: Optional[float] = None

    @property
    def active_slot_count(self) -> int:
        return sum(
            1
            for seg in self.segments.values()
            if seg.state in ("pending_prefill", "active") and seg.slot is not None
        )


# ---------------------------------------------------------------------------
# Segment key
# ---------------------------------------------------------------------------


def _seg_key(session_id: str, segment_idx: int) -> str:
    return f"{session_id}:{segment_idx}"


# ---------------------------------------------------------------------------
# Engine loop
# ---------------------------------------------------------------------------


class EngineLoop:
    """GPU-owning engine thread with pipelined decode and priority scheduling."""

    def __init__(
        self,
        engine_inbox: queue.Queue,
        async_loop: asyncio.AbstractEventLoop,
        executor: Executor,
        prefill_builder: Optional[PrefillBuilder] = None,
        *,
        max_batch_size: int = 48,
        mlfq_config: Optional[MLFQConfig] = None,
        prefix_cache_max_entries: int = 16,
        prefix_cache_max_len: int = 512,
        max_idle_sec: float = 10.0,
        max_queue_size: int = 256,
        session_timeout_sec: float = 300.0,
        min_pad_steps: int = 4,
        pad_silence_peak_threshold: float = 5e-4,
        pad_silence_mean_abs_threshold: float = 2e-4,
        max_slots_per_session: int = 2,
    ):
        self._inbox = engine_inbox
        self._async_loop = async_loop
        self._executor = executor
        self._prefill_builder = prefill_builder
        self._max_batch = max_batch_size
        self._max_idle_sec = max_idle_sec
        self._max_queue_size = max_queue_size
        self._session_timeout_sec = session_timeout_sec
        self._min_pad_steps = min_pad_steps
        self._pad_silence_peak_threshold = float(pad_silence_peak_threshold)
        self._pad_silence_mean_abs_threshold = float(pad_silence_mean_abs_threshold)
        self._max_slots_per_session = max(1, int(max_slots_per_session))

        self._groups: Dict[str, EngineSessionGroup] = {}
        self._seg_by_slot: Dict[int, EngineSegment] = {}
        # When set (during _process_step_output), _send_result appends
        # (queue, result) here instead of scheduling one loop callback per
        # result; the whole step's results are flushed with a single
        # call_soon_threadsafe (1 wakeup instead of one per slot per step).
        self._result_batch: Optional[list] = None

        self._mlfq = MLFQScheduler(mlfq_config or MLFQConfig())
        self._prefix_cache = PrefixKVCache(
            max_entries=prefix_cache_max_entries,
            max_prefix_len=prefix_cache_max_len,
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_eviction_check: float = 0.0

        self._total_steps: int = 0
        self._total_prefills: int = 0
        self._total_sessions: int = 0
        self._total_eos: int = 0
        self._total_timeouts: int = 0
        self._total_evictions: int = 0
        self._last_health_emit: float = 0.0
        # Per-iteration phase timing, aggregated and logged every few seconds
        # (loop-tax observability: where does the iteration wall time go).
        self._phase_acc = {k: 0.0 for k in ("p1", "p2", "p3", "p4", "p5")}
        self._phase_iters: int = 0
        self._phase_width_sum: int = 0
        self._phase_last_log: float = 0.0
        # L2 batch_compose(decode): only emit when the decode batch size changes.
        self._last_decode_batch: int = 0

        self._embed_device = self._executor._device
        self._embed_dtype = self._executor._config.dtype
        self._hidden_size = self._executor._config.hidden_size
        self._tts_pad_embed = (
            prefill_builder.w.tts_pad_embed.to(
                device=self._embed_device,
                dtype=self._embed_dtype,
            ).clone()
            if prefill_builder is not None
            else torch.zeros(
                1,
                1,
                self._hidden_size,
                device=self._embed_device,
                dtype=self._embed_dtype,
            )
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="engine-loop",
            daemon=True,
        )
        self._thread.start()
        logger.info("Engine loop started (max_batch=%d)", self._max_batch)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Engine loop stopped")

    def thread_alive(self) -> bool:
        """True while the engine-loop thread is actually running.

        ``_running`` is a lifecycle flag, not liveness: it stays True if the
        loop thread dies on an uncaught exception. Health checks must use
        this instead.
        """
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._run_inner()
        except Exception:
            logger.exception("Engine loop crashed")
            raise
        finally:
            # Health checks read this; without it a dead engine thread keeps
            # reporting running=True forever.
            self._running = False

    def _run_inner(self) -> None:
        prev_output: Optional[StepOutput] = None

        while self._running:
            t0 = time.monotonic()
            # --- Phase 1: Process previous step output (MUST run before
            # building next inputs to satisfy autoregressive dependency) ---
            if prev_output is not None:
                self._process_step_output(prev_output)
                prev_output = None

            # --- Phase 1.5: Drain inbox EARLY so newly arrived requests
            # are immediately available for prefill/decode in this
            # iteration, instead of waiting until the next one. ---
            self._drain_inbox()
            t1 = time.monotonic()

            # --- Phase 2: Prefill pending sessions FIRST.
            # Prefill and decode share one TRT execution context, so they
            # must run serially. First audio chunk is still produced during
            # prefill, so first_chunk_latency = time-to-prefill. ---
            try:
                self._try_prefill_pending()
            except Exception:
                logger.exception("Prefill failed unexpectedly")
                self._cleanup_failed_prefills()
            t2 = time.monotonic()

            # --- Phase 3: Launch decode for active slots. ---
            active_slots = self._get_active_slots_mlfq()
            gpu_future = None
            if active_slots:
                gpu_future = self._executor.launch_decode_step(active_slots)
            t3 = time.monotonic()

            # --- Phase 4: While decode runs, do CPU housekeeping.
            # Draining here is what makes batch admission "natural": requests
            # arriving during a decode step accumulate as pending segments and
            # are admitted together at the next step boundary (Phase 2).
            # Admission itself must NOT run here even for cache hits: measured
            # twice (2026-07-07), mid-flight admission is ~4x slower per
            # session than at the boundary — legacy-default-stream implicit
            # sync stalls its H2D copies behind the running step, and on a
            # dedicated side stream the caching allocator cannot reuse
            # default-stream blocks and falls back to device-synchronizing
            # cudaMalloc.  At the boundary the GPU is idle and the allocator
            # hits its cache (~16-30ms per full wave post log-diet). ---
            self._drain_inbox()
            self._try_evict_idle_slots()
            self._try_timeout_sessions()
            self._maybe_emit_health()
            t4 = time.monotonic()

            # --- Phase 5: Wait for GPU, store output for next iteration ---
            if gpu_future is not None:
                prev_output = gpu_future.wait()
                self._total_steps += 1
                self._mlfq.tick(self._all_active_mlfq_metas())
            else:
                if not self._has_work():
                    time.sleep(0.001)
            self._note_iteration_timing(
                t0, t1, t2, t3, t4, time.monotonic(), len(active_slots)
            )

    def _note_iteration_timing(
        self,
        t0: float,
        t1: float,
        t2: float,
        t3: float,
        t4: float,
        t5: float,
        width: int,
    ) -> None:
        """Aggregate per-iteration phase durations; log one line every ~2s.

        Answers "where does iteration wall time go at width W": p1 = process
        prev output + drain, p2 = admission, p3 = launch, p4 = housekeeping,
        p5 = GPU wait.  Idle iterations (width 0, nothing pending) are
        skipped so the averages describe loaded behavior.
        """
        if width == 0:
            return
        acc = self._phase_acc
        acc["p1"] += t1 - t0
        acc["p2"] += t2 - t1
        acc["p3"] += t3 - t2
        acc["p4"] += t4 - t3
        acc["p5"] += t5 - t4
        self._phase_iters += 1
        self._phase_width_sum += width
        now = t5
        if now - self._phase_last_log < 2.0:
            return
        n = self._phase_iters
        logger.info(
            "Loop timing over %d iters (avg width %.1f): "
            "p1_process=%.1fms p2_admit=%.1fms p3_launch=%.1fms "
            "p4_housekeep=%.1fms p5_gpu_wait=%.1fms iter=%.1fms",
            n,
            self._phase_width_sum / n,
            acc["p1"] / n * 1000.0,
            acc["p2"] / n * 1000.0,
            acc["p3"] / n * 1000.0,
            acc["p4"] / n * 1000.0,
            acc["p5"] / n * 1000.0,
            sum(acc.values()) / n * 1000.0,
        )
        self._phase_last_log = now
        for k in acc:
            acc[k] = 0.0
        self._phase_iters = 0
        self._phase_width_sum = 0

    # ------------------------------------------------------------------
    # Inbox
    # ------------------------------------------------------------------

    @staticmethod
    def _session_obs_level(group: Optional["EngineSessionGroup"]):
        """Per-session resolved observability level (None ⇒ use global), for
        gating L2 decision logs from the engine thread."""
        sc = group.request.session_config if group and group.request else None
        return getattr(sc, "observability_level", None) if sc else None

    def _maybe_emit_health(self) -> None:
        """Emit the periodic L1 ``engine.health`` gauge (aggregate engine state).

        Cadence is ``observability.health_interval_sec`` (0 disables). Cheap:
        one ``time.monotonic()`` check per loop iteration.
        """
        interval = obs.health_interval_sec()
        if interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_health_emit < interval:
            return
        self._last_health_emit = now
        kv_pool = self._executor.kv_pool
        cache_stats = self._prefix_cache.stats
        LifecycleLogger.emit(
            session_id="-",
            phase="engine.health",
            active_sessions=len(self._groups),
            kv_used=kv_pool.used_count if kv_pool else None,
            kv_free=kv_pool.free_count if kv_pool else None,
            queue_depth=self._inbox.qsize(),
            prefix_cache_hit_rate=round(cache_stats.get("hit_rate", 0.0), 3),
            total_steps=self._total_steps,
            total_prefills=self._total_prefills,
            total_eos=self._total_eos,
            total_evictions=self._total_evictions,
            total_timeouts=self._total_timeouts,
        )

    def _drain_inbox(self) -> None:
        drained = 0
        queue_depth = 0
        while True:
            try:
                req: EngineRequest = self._inbox.get_nowait()
            except queue.Empty:
                break
            # Record dequeued timestamp
            now = time.monotonic()
            req.dequeued_at = now
            # Emit lifecycle event for first text dequeue (START_TOKENS only)
            if req.type == RequestType.START_TOKENS:
                wait_ms = 0.0
                if req.enqueued_at is not None:
                    wait_ms = (now - req.enqueued_at) * 1000.0
                group = self._groups.get(req.session_id)
                if group is not None and group.first_text_dequeued_at is None:
                    group.first_text_dequeued_at = now
                    # Write to ServerTimingAccumulator if available. `req` here
                    # is the bare START_TOKENS request, which never carries
                    # session_config (see Dispatcher.dispatch_segment_actions),
                    # so the accumulator must be resolved via the session
                    # group's original NEW_SESSION request instead.
                    acc = self._get_group_timing_accumulator(group)
                    if acc is not None:
                        acc.first_text_dequeued_monotonic = now
                    LifecycleLogger.emit(
                        session_id=req.session_id,
                        phase="text.first_dequeued",
                        segment_idx=req.segment_idx,
                        request_id=(
                            req.session_config.timing.request_id
                            if req.session_config
                            else None
                        )
                        or None,
                        monotonic_ts=now,
                        wait_ms=round(wait_ms),
                        queue_depth_at_dequeue=self._inbox.qsize(),
                    )
            self._handle_request(req)
            drained += 1
        if drained > 0:
            logger.debug("Drained %d requests (queue_depth=%d)", drained, queue_depth)

    def _handle_request(self, req: EngineRequest) -> None:
        if req.type == RequestType.NEW_SESSION:
            replacing_existing = req.session_id in self._groups
            if not replacing_existing and len(self._groups) >= self._max_queue_size:
                logger.warning(
                    "Backpressure: rejecting session %s (active=%d >= limit=%d)",
                    req.session_id,
                    len(self._groups),
                    self._max_queue_size,
                )
                if req.result_queue is not None:
                    self._async_loop.call_soon_threadsafe(
                        req.result_queue.put_nowait,
                        EngineResult(
                            type=ResultType.ERROR,
                            session_id=req.session_id,
                            error_msg="Server overloaded, please retry later",
                        ),
                    )
                return
            if replacing_existing:
                logger.warning(
                    "NEW_SESSION replacing existing backend session: %s",
                    req.session_id,
                )
                self._remove_session(req.session_id)
            group = EngineSessionGroup(req.session_id, req)
            self._groups[req.session_id] = group
            self._total_sessions += 1
            logger.debug("New session group: %s", req.session_id)
            kv_pool = self._executor.kv_pool
            LifecycleLogger.emit(
                session_id=req.session_id,
                phase="session.registered",
                request_id=(
                    req.session_config.timing.request_id if req.session_config else None
                )
                or None,
                free_slots=kv_pool.free_count if kv_pool else None,
                active_sessions=len(self._groups),
            )

        elif req.type == RequestType.START_TOKENS:
            group = self._groups.get(req.session_id)
            if group is None:
                logger.warning("START_TOKENS for unknown session: %s", req.session_id)
                return
            seg = EngineSegment(
                req.session_id,
                req.segment_idx,
                req.priority,
            )
            seg.dequeued_at = req.dequeued_at
            if group.overflow_token_ids:
                seg.pending_token_ids.extend(group.overflow_token_ids)
                seg.text_tokens_consumed += len(group.overflow_token_ids)
                logger.info(
                    "Prepended %d overflow tokens to %s seg=%d",
                    len(group.overflow_token_ids),
                    req.session_id,
                    req.segment_idx,
                )
                group.overflow_token_ids.clear()
            if req.token_ids:
                seg.pending_token_ids.extend(req.token_ids)
                seg.text_tokens_consumed += len(req.token_ids)
            old_seg = group.segments.get(req.segment_idx)
            if old_seg is not None:
                logger.warning(
                    "START_TOKENS replacing existing segment: %s seg=%d state=%s",
                    req.session_id,
                    req.segment_idx,
                    old_seg.state,
                )
                self._release_segment_slot(old_seg)
            group.segments[req.segment_idx] = seg
            if req.result_queue is not None:
                group.result_queue = req.result_queue
            logger.debug(
                "New segment: %s seg=%d prio=%s",
                req.session_id,
                req.segment_idx,
                req.priority.name,
            )

        elif req.type == RequestType.APPEND_TOKENS:
            group = self._groups.get(req.session_id)
            if group is None:
                return
            if not req.token_ids:
                return
            seg = group.segments.get(req.segment_idx)
            if seg is None or seg.state == "done":
                group.overflow_token_ids.extend(req.token_ids)
                logger.debug(
                    "Overflow %d tokens for %s seg=%d (seg_state=%s, overflow_total=%d)",
                    len(req.token_ids),
                    req.session_id,
                    req.segment_idx,
                    seg.state if seg else "MISSING",
                    len(group.overflow_token_ids),
                )
                return
            seg.pending_token_ids.extend(req.token_ids)
            seg.text_tokens_consumed += len(req.token_ids)
            if (
                seg.state == "active"
                and seg.slot is not None
                and self._prefill_builder is not None
            ):
                self._append_trailing_tokens(seg.slot, req.token_ids)
                self._resume_streaming_segment_if_ready(seg)
            else:
                logger.debug(
                    "APPEND_TOKENS %d tokens for %s seg=%d state=%s (pre-prefill accumulate)",
                    len(req.token_ids),
                    req.session_id,
                    req.segment_idx,
                    seg.state,
                )

        elif req.type == RequestType.SEGMENT_TOKENS_DONE:
            group = self._groups.get(req.session_id)
            if group is None:
                return
            seg = group.segments.get(req.segment_idx)
            if seg:
                seg.input_complete = True
                if (
                    req.append_eos
                    and seg.state == "active"
                    and seg.slot is not None
                    and not seg.eos_trailing_added
                    and self._prefill_builder is not None
                ):
                    self._append_eos_trailing(seg)
                    self._resume_streaming_segment_if_ready(seg)
                elif req.append_eos:
                    seg.eos_trailing_added = False
                if seg.state == "done":
                    self._check_session_done(group)

        elif req.type == RequestType.SESSION_TOKENS_DONE:
            group = self._groups.get(req.session_id)
            if group is None:
                return
            group.input_complete_all = True
            self._check_session_done(group)

        elif req.type == RequestType.CANCEL_SESSION:
            group = self._groups.get(req.session_id)
            if group is not None:
                self._send_result(
                    group,
                    EngineResult(
                        type=ResultType.SESSION_DONE,
                        session_id=req.session_id,
                        metrics={"cancelled": True},
                    ),
                )
            self._remove_session(req.session_id)

    # ------------------------------------------------------------------
    # Priority-based prefill
    # ------------------------------------------------------------------

    def _try_prefill_pending(self) -> None:
        """Admit ALL pending segments at the step boundary, not just one.

        Runs right after the previous decode step's output was processed
        (finished slots released) and before the next step launches, so
        requests that accumulated during the in-flight step join the next
        decode batch together:

          1. batch admission for prefix-cache hits — one flat embedding
             lookup + one broadcast KV restore per cache entry, no TRT;
          2. serial TRT prefill for the remainder (cache misses / ICL),
             which must not overlap decode (shared TRT context).
        """
        batched = self._admit_cache_hit_batch()
        count = 0
        while count < self._max_batch:
            if not self._try_prefill_one():
                break
            count += 1
        if batched + count > 1:
            logger.info(
                "Prefilled %d segments in one pass (%d batched cache-hit)",
                batched + count,
                batched,
            )

    def _prefix_cache_admittable(
        self,
        group: EngineSessionGroup,
        seg: EngineSegment,
    ) -> bool:
        """True if this pending segment can be admitted from the prefix KV
        cache alone (no TRT prefill call).

        Invalid task_type or uncacheable configs return False and fall
        through to the serial prefill pass, which owns error reporting.
        """
        if self._prefill_builder is None:
            return False
        if not seg.prefix_probe_done:
            seg.prefix_probe_done = True
            req_cfg = group.request.session_config
            task_type_str = (
                req_cfg.task_type
                if req_cfg is not None
                else (group.request.task_type or "custom_voice")
            )
            try:
                task_type = parse_task_type(
                    task_type_str,
                    x_vector_only=(
                        req_cfg.x_vector_only if req_cfg is not None else False
                    ),
                )
            except ValueError:
                return False
            if task_type == TaskType.VOICE_CLONE_ICL:
                return False
            seg.prefix_probe_task_type = task_type
            seg.prefix_probe_key = self._prefill_builder.compute_cache_key(
                task_type,
                req_cfg.language if req_cfg is not None else "auto",
                req_cfg.speaker
                if req_cfg is not None
                else group.request.speaker_key,
                req_cfg.instruct if req_cfg is not None else None,
                (
                    list(req_cfg.instruct_spec.token_ids)
                    if req_cfg is not None and req_cfg.instruct_spec is not None
                    else None
                ),
                spk_embedding=(
                    req_cfg.spk_embedding if req_cfg is not None else None
                ),
            )
        return self._prefix_cache.contains(seg.prefix_probe_key)

    def _admit_cache_hit_batch(self) -> int:
        """Batch-admit all pending prefix-cache-hit segments in one pass.

        Burst TTFT was dominated by per-session admission cost serialized
        against ever-wider decode steps (~1.4ms × N sessions).  Cache-hit
        admission needs no TRT call, so the per-session GPU work is
        vectorized across the whole batch:

          - one flat text-embedding lookup for all sessions' suffix tokens;
          - one broadcast KV-pool restore per distinct cache entry;
          - one zero-fill per C2W state shape (row views per slot).

        Admission is capped by free KV slots (= max_batch − occupied), per
        the continuous-batching invariant.  Returns the number admitted.
        """
        kv_pool = self._executor.kv_pool
        if kv_pool is None or self._prefill_builder is None:
            return 0
        # Test doubles may predate the batch-admission API — fall back to the
        # serial pass entirely.
        if not hasattr(self._prefill_builder, "build_suffix_batch") or not hasattr(
            self._executor, "make_zero_states_batch"
        ):
            return 0
        budget = min(kv_pool.free_count, self._max_batch)
        if budget <= 0:
            return 0

        # -- Collect pending segments in global priority order --
        candidates: list[tuple[EngineSegment, EngineSessionGroup]] = []
        for group in self._groups.values():
            if group.active_slot_count >= self._max_slots_per_session:
                continue
            for seg in group.segments.values():
                if seg.state != "pending_prefill":
                    continue
                if not seg.pending_token_ids:
                    continue
                candidates.append((seg, group))
        if not candidates:
            return 0
        candidates.sort(key=lambda pair: pair[0].priority.value)

        # Walk in priority order: cache hits are picked for the batch; cache
        # misses also consume budget so a slot stays free for the serial TRT
        # pass — batch admission must not starve higher-priority misses.
        picked: list[tuple[EngineSegment, EngineSessionGroup]] = []
        picked_per_group: Dict[str, int] = {}
        for seg, group in candidates:
            if budget <= 0:
                break
            in_batch = picked_per_group.get(group.session_id, 0)
            if group.active_slot_count + in_batch >= self._max_slots_per_session:
                continue
            if not self._prefix_cache_admittable(group, seg):
                budget -= 1
                continue
            budget -= 1
            picked_per_group[group.session_id] = in_batch + 1
            picked.append((seg, group))
        if not picked:
            return 0

        return self._admit_picked(kv_pool, picked)

    def _admit_picked(
        self,
        kv_pool,
        picked: list[tuple[EngineSegment, EngineSessionGroup]],
    ) -> int:
        """Allocate, restore, and activate the picked cache-hit segments."""
        # -- Allocate slots (cheap metadata + async row zero) --
        batch_start = time.monotonic()
        t_collect = batch_start
        admitted: list[tuple[EngineSegment, EngineSessionGroup]] = []
        for seg, group in picked:
            slot = kv_pool.allocate(_seg_key(seg.session_id, seg.segment_idx))
            if slot is None:
                break
            seg.slot = slot
            slot.segment_idx = int(seg.segment_idx)
            self._seg_by_slot[slot.slot_id] = seg
            self._mlfq.on_segment_created(seg.mlfq_meta)
            seg.prefill_started_at = batch_start
            admitted.append((seg, group))
        if not admitted:
            return 0
        t_alloc = time.monotonic()

        # -- Batched suffix embeds: one flat lookup for every session --
        suffixes = self._prefill_builder.build_suffix_batch(
            [seg.pending_token_ids for seg, _ in admitted],
            [seg.input_complete for seg, _ in admitted],
        )
        t_suffix = time.monotonic()

        # -- Batched zero C2W states + token counts (row views per slot) --
        # Prefer persistent slot-indexed arenas (zeroed in place, no alloc);
        # fall back to fresh batch allocations when arenas are unavailable.
        take_rows = getattr(self._executor, "take_zeroed_state_rows", None)
        zero_states = (
            take_rows([seg.slot.slot_id for seg, _ in admitted])
            if take_rows is not None
            else None
        )
        arena_backed = zero_states is not None
        if zero_states is None:
            zero_states = self._executor.make_zero_states_batch(len(admitted))
        token_counts_rows = torch.zeros(
            len(admitted),
            self._executor._config.codec_vocab_size,
            device=self._embed_device,
            dtype=torch.int64,
        ).split(1, dim=0)
        t_zeros = time.monotonic()

        # -- Broadcast KV restore, one pool write per distinct cache entry --
        by_entry: Dict[str, list[int]] = {}
        entries: Dict[str, object] = {}
        for i, (seg, group) in enumerate(admitted):
            entry = self._prefix_cache.get(seg.prefix_probe_key)
            if entry is None:
                # Should not happen (no puts since the probe); fall back to
                # the serial pass for this segment.
                self._seg_by_slot.pop(seg.slot.slot_id, None)
                kv_pool.release(seg.slot.slot_id)
                seg.slot = None
                continue
            entries[seg.prefix_probe_key] = entry
            by_entry.setdefault(seg.prefix_probe_key, []).append(i)

        use_pool = (
            getattr(kv_pool, "_preallocate", False)
            and getattr(kv_pool, "_talker_kv_pool", None) is not None
        )
        for key, idxs in by_entry.items():
            entry = entries[key]
            if use_pool:
                kv_pool.restore_prefix_batch(
                    [admitted[i][0].slot.slot_id for i in idxs],
                    entry.talker_kv,
                    entry.prefix_len,
                )
            else:
                for i in idxs:
                    admitted[i][0].slot.talker_kv = entry.talker_kv.clone()
            for i in idxs:
                admitted[i][0].slot.past_len = entry.prefix_len
        t_restore = time.monotonic()

        # -- Per-slot assembly + completion (CPU bookkeeping only) --
        completed = 0
        prime_ms = 0.0
        complete_ms = 0.0
        for i, (seg, group) in enumerate(admitted):
            if seg.slot is None:
                continue
            entry = entries[seg.prefix_probe_key]
            request_embeds, trailing = suffixes[i]
            t0 = time.monotonic()
            self._prime_decode_after_prefix_prefill(
                seg.slot,
                request_embeds,
                trailing,
                source="prefix_cache_prefix_only",
                zero_states=zero_states[i],
                token_counts=token_counts_rows[i],
            )
            if arena_backed:
                seg.slot.c2w_arena_backed = True
                seg.slot.c2w_write_in_a = False
            seg.cache_hit = True
            seg.cache_tokens_reused = entry.prefix_len
            seg.eos_trailing_added = seg.input_complete
            prefill_metrics = self._prefill_metrics(
                seg.prefix_probe_task_type,
                group.request.session_config,
            )
            t1 = time.monotonic()
            self._complete_prefill(group, seg, seg.slot, prefill_metrics, None, False)
            complete_ms += (time.monotonic() - t1) * 1000.0
            prime_ms += (t1 - t0) * 1000.0
            completed += 1
        if completed:
            logger.info(
                "Batch-admitted %d cache-hit segments (%d distinct prefixes, "
                "%.1fms total: alloc=%.1f suffix=%.1f zeros=%.1f restore=%.1f "
                "prime=%.1f complete=%.1f)",
                completed,
                len(by_entry),
                (time.monotonic() - batch_start) * 1000.0,
                (t_alloc - t_collect) * 1000.0,
                (t_suffix - t_alloc) * 1000.0,
                (t_zeros - t_suffix) * 1000.0,
                (t_restore - t_zeros) * 1000.0,
                prime_ms,
                complete_ms,
            )
        return completed

    def _try_prefill_one(self) -> bool:
        """Run prefill for the highest-priority pending segment.

        Uses prefix KV cache when available to skip redundant GPU work.
        Returns True if a segment was prefilled, False otherwise.
        """
        kv_pool = self._executor.kv_pool
        if kv_pool is None or kv_pool.free_count == 0:
            return False

        best: Optional[EngineSegment] = None
        best_group: Optional[EngineSessionGroup] = None

        for group in self._groups.values():
            if group.active_slot_count >= self._max_slots_per_session:
                continue
            for seg in group.segments.values():
                if seg.state != "pending_prefill":
                    continue
                if not seg.pending_token_ids:
                    continue
                if best is None or seg.priority.value < best.priority.value:
                    best = seg
                    best_group = group

        if best is None or best_group is None:
            return False

        slot = kv_pool.allocate(_seg_key(best.session_id, best.segment_idx))
        if slot is None:
            return False
        best.slot = slot
        slot.segment_idx = int(best.segment_idx)
        self._seg_by_slot[slot.slot_id] = best
        self._mlfq.on_segment_created(best.mlfq_meta)

        if self._prefill_builder is not None:
            req_cfg = best_group.request.session_config
            task_type_str = (
                req_cfg.task_type
                if req_cfg is not None
                else (best_group.request.task_type or "custom_voice")
            )
            try:
                task_type = parse_task_type(
                    task_type_str,
                    x_vector_only=(
                        req_cfg.x_vector_only if req_cfg is not None else False
                    ),
                )
            except ValueError as exc:
                logger.error("Invalid task_type for %s: %s", best.session_id, exc)
                best.state = "error"
                self._seg_by_slot.pop(slot.slot_id, None)
                kv_pool.release(slot.slot_id)
                best.slot = None
                self._send_result(
                    best_group,
                    EngineResult(
                        type=ResultType.ERROR,
                        session_id=best.session_id,
                        segment_idx=best.segment_idx,
                        error_msg=str(exc),
                    ),
                )
                self._remove_session(best.session_id)
                return False
            prefill_metrics = self._prefill_metrics(task_type, req_cfg)
            # Record prefill start time
            best.prefill_started_at = time.monotonic()
            # Non-ICL tasks can check cache before building the full plan. ICL
            # needs the plan first because its request suffix includes ref
            # codec frames plus target text.
            cache_key = None
            cached = None
            if task_type != TaskType.VOICE_CLONE_ICL:
                cache_key = self._prefill_builder.compute_cache_key(
                    task_type,
                    req_cfg.language if req_cfg is not None else "auto",
                    req_cfg.speaker
                    if req_cfg is not None
                    else best_group.request.speaker_key,
                    req_cfg.instruct if req_cfg is not None else None,
                    (
                        list(req_cfg.instruct_spec.token_ids)
                        if req_cfg is not None and req_cfg.instruct_spec is not None
                        else None
                    ),
                    spk_embedding=(
                        req_cfg.spk_embedding if req_cfg is not None else None
                    ),
                )
                cached = self._prefix_cache.get(cache_key)

            if (
                task_type != TaskType.VOICE_CLONE_ICL
                and cached is not None
                and best.pending_token_ids
            ):
                # ── Cache HIT: restore prefix KV and let decode consume first text token ──
                best.cache_hit = True
                best.cache_tokens_reused = cached.prefix_len
                req_embeds, trailing = self._prefill_builder.build_suffix_from_ids(
                    best.pending_token_ids,
                    include_eos=best.input_complete,
                )
                self._apply_prefix_cache_hit(
                    slot,
                    cached,
                    req_embeds,
                    trailing,
                )
                best.eos_trailing_added = best.input_complete
                prefill_audio, prefill_eos = None, False
                logger.info(
                    "Prefix cache hit: copied %d KV tokens for %s "
                    "(slot=%d, decode will consume first text token in batch)",
                    cached.prefix_len,
                    best.session_id,
                    slot.slot_id,
                )
            else:
                # ── Cache MISS: full token-native plan + TRT prefill ──
                plan = self._prefill_builder.build_plan_from_ids(
                    task_type=task_type,
                    token_ids=best.pending_token_ids,
                    language=req_cfg.language if req_cfg is not None else "auto",
                    speaker=req_cfg.speaker
                    if req_cfg is not None
                    else best_group.request.speaker_key,
                    instruct=req_cfg.instruct if req_cfg is not None else None,
                    instruct_token_ids=(
                        list(req_cfg.instruct_spec.token_ids)
                        if req_cfg is not None and req_cfg.instruct_spec is not None
                        else None
                    ),
                    spk_embedding=(
                        req_cfg.spk_embedding if req_cfg is not None else None
                    ),
                    ref_text=req_cfg.ref_text if req_cfg is not None else None,
                    ref_text_token_ids=(
                        list(req_cfg.ref_text_spec.token_ids)
                        if req_cfg is not None and req_cfg.ref_text_spec is not None
                        else None
                    ),
                    ref_codec_sum_vec=(
                        req_cfg.ref_codec_sum_vec if req_cfg is not None else None
                    ),
                    ref_audio_sha256=(
                        req_cfg.ref_audio_sha256 if req_cfg is not None else None
                    ),
                    ref_feature_cache_key=(
                        req_cfg.ref_feature_cache_key if req_cfg is not None else None
                    ),
                    include_eos=best.input_complete,
                )
                best.prefill_plan = plan
                slot.prefill_source = "full_prefill"

                if req_cfg is not None and req_cfg.ref_warnings:
                    for warning_msg in req_cfg.ref_warnings:
                        self._send_result(
                            best_group,
                            EngineResult(
                                type=ResultType.WARNING,
                                session_id=best.session_id,
                                segment_idx=best.segment_idx,
                                warning_msg=str(warning_msg),
                            ),
                        )
                if plan.warnings:
                    for warning_msg in plan.warnings:
                        self._send_result(
                            best_group,
                            EngineResult(
                                type=ResultType.WARNING,
                                session_id=best.session_id,
                                segment_idx=best.segment_idx,
                                warning_msg=str(warning_msg),
                            ),
                        )

                # ICL keeps the reference codec path as one complete prefill.
                # Splitting it into a cached reference prefix and request suffix
                # changes the fused Talker/C2W state boundary and can corrupt
                # generation quality.
                split_prefix_prefill = (
                    task_type != TaskType.VOICE_CLONE_ICL
                    and plan.cacheable_prefix_embeds is not None
                    and plan.request_prefill_embeds is not None
                    and int(plan.request_prefill_embeds.shape[1]) == 1
                )
                if split_prefix_prefill:
                    self._executor.prefill_prefix_only(
                        slot,
                        plan.cacheable_prefix_embeds,
                    )
                    prefill_audio, prefill_eos = None, False
                else:
                    if task_type == TaskType.VOICE_CLONE_ICL:
                        self._apply_ref_c2w_warm_state(best_group, best, req_cfg)
                    prefill_audio, prefill_eos = self._executor.prefill(
                        slot,
                        plan.prefill_embeds,
                    )
                if task_type != TaskType.VOICE_CLONE_ICL:
                    # Populate cache — read from pool when preallocated.
                    effective_key = plan.prefix_cache_key or cache_key
                    if (
                        effective_key is not None
                        and plan.cacheable_prefix_embeds is not None
                    ):
                        prefix_len = int(plan.cacheable_prefix_embeds.shape[1])
                        prefix_kv = self._read_prefix_kv(slot, prefix_len)
                        if prefix_kv is not None:
                            self._prefix_cache.put(
                                effective_key,
                                prefix_kv,
                                prefix_len,
                            )

                    if split_prefix_prefill:
                        self._prime_decode_after_prefix_prefill(
                            slot,
                            plan.request_prefill_embeds,
                            plan.trailing,
                            source="full_prefill_prefix_only",
                        )
                        best.eos_trailing_added = best.input_complete
                    else:
                        self._attach_trailing_after_prefill(slot, plan.trailing)
                        best.eos_trailing_added = best.input_complete
                elif not split_prefix_prefill:
                    self._attach_trailing_after_prefill(slot, plan.trailing)
                    best.eos_trailing_added = best.input_complete
        else:
            prefill_metrics = {}
            best.prefill_started_at = time.monotonic()
            prefill_audio, prefill_eos = self._executor.prefill(
                slot,
                torch.zeros(
                    1,
                    1,
                    self._hidden_size,
                    device=self._embed_device,
                    dtype=self._embed_dtype,
                ),
            )

        # Serial admissions (cold TRT prefill, ICL, serial cache hit) produce
        # freestanding C2W state tensors; move them into the slot's arena
        # rows so the post-step scatter stays batched for every active slot.
        adopt = getattr(self._executor, "adopt_c2w_states", None)
        if adopt is not None:
            adopt(slot)

        self._complete_prefill(
            best_group,
            best,
            slot,
            prefill_metrics,
            prefill_audio,
            prefill_eos,
        )
        return True

    def _complete_prefill(
        self,
        group: EngineSessionGroup,
        seg: EngineSegment,
        slot: SlotKVState,
        prefill_metrics: dict,
        prefill_audio: Optional[bytes],
        prefill_eos: bool,
    ) -> None:
        """Shared admission tail: activate the segment and emit observability.

        Used by both the serial prefill path and batch cache-hit admission.
        """
        seg.state = "active"
        seg.decode_start_frame = slot.frame_idx
        self._total_prefills += 1

        # -- Prefill timing observability --
        prefill_end = time.monotonic()
        seg.prefill_completed_at = prefill_end
        if seg.prefill_started_at is not None:
            prefill_duration_ms = (prefill_end - seg.prefill_started_at) * 1000.0
        else:
            prefill_duration_ms = 0.0

        # Determine cache hit status. slot.prefill_source is only ever assigned
        # "full_prefill" or "prefix_cache_prefix_only" (see
        # _apply_prefix_cache_hit -> _prime_decode_after_prefix_prefill); the
        # literal "prefix_cache_hit" was never assigned anywhere, so this check
        # always evaluated to False and silently clobbered the correct
        # seg.cache_hit = True set by the cache-hit paths.
        cache_hit = (
            slot.prefill_source == "prefix_cache_prefix_only"
            if hasattr(slot, "prefill_source")
            else False
        )
        seg.cache_hit = cache_hit

        # Write to ServerTimingAccumulator if available
        acc = self._get_group_timing_accumulator(group)
        if acc is not None:
            if (
                acc.prefill_started_monotonic is None
                and seg.prefill_started_at is not None
            ):
                acc.prefill_started_monotonic = seg.prefill_started_at
            acc.prefill_completed_monotonic = prefill_end
            acc.cache_hit = cache_hit
            acc.cache_tokens_reused = seg.cache_tokens_reused

        # Emit prefill completed lifecycle event
        LifecycleLogger.emit(
            session_id=seg.session_id,
            phase="engine.prefill.completed",
            segment_idx=seg.segment_idx,
            request_id=(
                group.request.session_config.timing.request_id
                if group.request.session_config
                else None
            )
            or None,
            monotonic_ts=prefill_end,
            prefill_duration_ms=round(prefill_duration_ms, 3),
            cache_hit=cache_hit,
            cache_tokens_reused=seg.cache_tokens_reused,
        )

        # L2 prefill_detail: why this prefill looked the way it did.
        session_level = self._session_obs_level(group)
        if obs.is_enabled(obs.ObsLevel.DEBUG, session_level):
            sc = group.request.session_config
            LifecycleLogger.emit(
                session_id=seg.session_id,
                phase="prefill_detail",
                segment_idx=seg.segment_idx,
                min_level=obs.ObsLevel.DEBUG,
                session_level=session_level,
                prefill_source=str(getattr(slot, "prefill_source", "") or ""),
                language=getattr(sc, "language", "") if sc else "",
                speaker=getattr(sc, "speaker", None) if sc else None,
                ref_source=getattr(sc, "ref_source", "") if sc else "",
                ref_id=getattr(sc, "ref_id", None) if sc else None,
                ref_audio_sha256=getattr(sc, "ref_audio_sha256", "") if sc else "",
                x_vector_only=getattr(sc, "x_vector_only", False) if sc else False,
            )

        # Add timing to prefill_metrics
        if seg.prefill_started_at is not None:
            prefill_metrics["prefill_started_at"] = str(seg.prefill_started_at)
        prefill_metrics["prefill_completed_at"] = str(prefill_end)
        prefill_metrics["prefill_duration_ms"] = f"{prefill_duration_ms:.3f}"
        prefill_metrics["cache_hit"] = "true" if cache_hit else "false"
        if seg.cache_tokens_reused > 0:
            prefill_metrics["cache_tokens_reused"] = str(seg.cache_tokens_reused)
        if seg.dequeued_at is not None:
            prefill_metrics["first_text_dequeued_at"] = str(seg.dequeued_at)
        if any(
            key in prefill_metrics
            for key in (
                "ref_source",
                "ref_id",
                "ref_audio_sha256",
                "ref_text_hash",
                "icl_cache_hit",
                "icl_cache_miss",
                "ref_preprocess_runtime",
            )
        ):
            logger.info(
                "Prefill metadata: session=%s segment=%d %s",
                seg.session_id,
                seg.segment_idx,
                " ".join(
                    f"{key}={value}" for key, value in sorted(prefill_metrics.items())
                ),
            )
        self._send_result(
            group,
            EngineResult(
                type=ResultType.PREFILL_DONE,
                session_id=seg.session_id,
                segment_idx=seg.segment_idx,
                metrics=prefill_metrics,
            ),
        )

        if prefill_audio and len(prefill_audio) > 0:
            self._send_result(
                group,
                EngineResult(
                    type=ResultType.AUDIO_CHUNK,
                    session_id=seg.session_id,
                    segment_idx=seg.segment_idx,
                    audio_bytes=prefill_audio,
                ),
            )
        if prefill_eos:
            self._handle_segment_eos(group, seg)
            return
        logger.debug(
            "Prefill done: %s seg=%d prio=%s (slot=%d, past_len=%d, "
            "trailing=%d, input_complete=%s, tokens=%d)",
            seg.session_id,
            seg.segment_idx,
            seg.priority.name,
            slot.slot_id,
            slot.past_len,
            len(slot.trailing),
            seg.input_complete,
            len(seg.pending_token_ids),
        )

    # ------------------------------------------------------------------
    # Prefix cache helpers
    # ------------------------------------------------------------------

    def _apply_prefix_cache_hit(
        self,
        slot: SlotKVState,
        cached,
        request_prefill_embeds: torch.Tensor,
        trailing: list,
    ) -> None:
        """Set up slot from cached prefix KV without any TRT call.

        Copies the cached talker KV into the KV pool (or slot) and
        prepares the slot for decode.  The suffix token is stored as
        next_embed so the first decode step processes it with full
        attention to the cached KV.
        """
        self._restore_prefix_cache(slot, cached)
        self._prime_decode_after_prefix_prefill(
            slot,
            request_prefill_embeds,
            trailing,
            source="prefix_cache_prefix_only",
        )

    def _restore_prefix_cache(self, slot: SlotKVState, cached) -> None:
        kv_pool = self._executor.kv_pool
        kv_pool.init_kv_tensors(slot)
        prefix_len = cached.prefix_len

        if kv_pool._preallocate and kv_pool._talker_kv_pool is not None:
            kv_pool._talker_kv_pool[slot.slot_id, :, :, :prefix_len, :] = (
                cached.talker_kv[0, :, :, :prefix_len, :]
            )
        else:
            slot.talker_kv = cached.talker_kv.clone()
        slot.past_len = prefix_len

    def _apply_ref_c2w_warm_state(
        self,
        group: EngineSessionGroup,
        seg: EngineSegment,
        req_cfg,
    ) -> None:
        if req_cfg is None or req_cfg.ref_c2w_kv is None or seg.slot is None:
            return
        warmed = self._executor.apply_c2w_warm_state(
            seg.slot,
            req_cfg.ref_c2w_kv,
            req_cfg.ref_c2w_conv_states,
            req_cfg.ref_c2w_transconv_states,
            req_cfg.ref_c2w_frame_idx,
        )
        if warmed:
            logger.info(
                "Applied Code2Wav ref warm state for %s (slot=%d, frame_idx=%d)",
                seg.session_id,
                seg.slot.slot_id,
                int(req_cfg.ref_c2w_frame_idx),
            )

    def _attach_trailing_after_prefill(
        self,
        slot: SlotKVState,
        trailing: list,
    ) -> None:
        slot.trailing = trailing
        if slot.next_embed is not None and slot.trailing:
            first_trail = slot.trailing[0].to(slot.next_embed.dtype)
            slot.next_embed = (slot.next_embed + first_trail).to(torch.float32)
            slot.text_idx = 1

    def _prefill_metrics(self, task_type: TaskType, req_cfg) -> dict:
        metrics: dict[str, str] = {"task_type": task_type.value}
        if req_cfg is None:
            return metrics
        for attr in (
            "ref_source",
            "ref_id",
            "ref_audio_sha256",
            "ref_text_hash",
            "ref_preprocess_runtime",
        ):
            value = getattr(req_cfg, attr, None)
            if value:
                text = str(value)
                if attr in ("ref_audio_sha256", "ref_text_hash"):
                    text = text[:12]
                metrics[attr] = text
        return metrics

    def _prime_decode_after_prefix_prefill(
        self,
        slot: SlotKVState,
        request_prefill_embeds: torch.Tensor,
        trailing: list,
        *,
        source: str,
        zero_states: Optional[tuple] = None,
        token_counts: Optional[torch.Tensor] = None,
    ) -> None:
        """Prepare slot so decode step0 consumes the first text token.

        ``zero_states``/``token_counts`` let batch admission inject
        pre-allocated per-slot rows (one zero-fill kernel per state shape
        for the whole batch) instead of allocating per slot here.
        """
        slot.prefill_source = source
        slot.frame_idx = 0
        slot.pad_start_frame = -1
        slot.pad_consecutive_silence = 0

        slot.c2w_kv = None
        slot.c2w_len = 0
        kv_pool = self._executor.kv_pool
        slot.c2w_pooled = bool(
            kv_pool is not None
            and getattr(kv_pool, "_preallocate", False)
            and getattr(kv_pool, "_c2w_kv_pool", None) is not None
        )
        if zero_states is not None:
            conv, transconv, conv_write, transconv_write = zero_states
            slot.c2w_conv_states = conv
            slot.c2w_transconv_states = transconv
            slot.init_pingpong_buffers(
                conv_write=conv_write,
                transconv_write=transconv_write,
            )
        else:
            slot.c2w_conv_states = self._executor.make_zero_conv_states()
            slot.c2w_transconv_states = self._executor.make_zero_transconv_states()
            slot.init_pingpong_buffers()

        if token_counts is not None:
            slot.token_counts = token_counts
        else:
            cfg = self._executor._config
            slot.token_counts = torch.zeros(
                1,
                cfg.codec_vocab_size,
                device=self._embed_device,
                dtype=torch.int64,
            )
        slot.next_embed = self._coerce_embed_tensor(
            request_prefill_embeds,
            dtype=torch.float32,
        )
        slot.last_codec_sum = None
        slot.trailing = [self._coerce_embed_tensor(t) for t in trailing]
        slot.text_idx = 0

    def _resume_streaming_segment_if_ready(self, seg: EngineSegment) -> None:
        """Resume a paused streaming segment when new trailing text/EOS arrives."""
        slot = seg.slot
        if slot is None or slot.last_codec_sum is None or slot.next_embed is not None:
            return
        if not slot.trailing or slot.text_idx >= len(slot.trailing):
            return
        text_add = slot.trailing[slot.text_idx].to(slot.last_codec_sum.dtype)
        slot.text_idx += 1
        slot.next_embed = (slot.last_codec_sum + text_add).to(torch.float32)
        slot.last_codec_sum = None
        slot.pad_start_frame = -1
        slot.pad_consecutive_silence = 0
        logger.debug(
            "Resumed paused streaming segment %s:%d (trailing=%d, text_idx=%d)",
            seg.session_id,
            seg.segment_idx,
            len(slot.trailing),
            slot.text_idx,
        )

    def _read_prefix_kv(
        self,
        slot: SlotKVState,
        prefix_len: int,
    ) -> Optional[torch.Tensor]:
        """Read prefix KV from pool or slot for cache population."""
        kv_pool = self._executor.kv_pool
        if kv_pool._preallocate and kv_pool._talker_kv_pool is not None:
            return kv_pool._talker_kv_pool[
                slot.slot_id : slot.slot_id + 1,
                :,
                :,
                :prefix_len,
                :,
            ].clone()
        if slot.talker_kv is not None:
            return slot.talker_kv[:, :, :, :prefix_len, :].clone()
        return None

    # ------------------------------------------------------------------
    # Decode batch (MLFQ-ordered)
    # ------------------------------------------------------------------

    def _get_active_slots_mlfq(self) -> list[SlotKVState]:
        """Build decode batch using MLFQ priority ordering."""
        kv_pool = self._executor.kv_pool
        max_seq = kv_pool.max_seq_len if kv_pool else float("inf")
        evict_pairs: list[tuple[EngineSessionGroup, EngineSegment]] = []
        candidates: list[EngineSegment] = []
        for group in self._groups.values():
            for seg in group.segments.values():
                if seg.state != "active" or seg.slot is None:
                    continue
                slot = seg.slot
                if slot.past_len >= max_seq:
                    evict_pairs.append((group, seg))
                    continue
                if slot.next_embed is None:
                    if slot.trailing and slot.text_idx < len(slot.trailing):
                        slot.next_embed = self._coerce_embed_tensor(
                            slot.trailing[slot.text_idx],
                            dtype=torch.float32,
                        )
                        slot.text_idx += 1
                    else:
                        continue
                candidates.append(seg)

        for group, seg in evict_pairs:
            logger.warning(
                "Segment hit max_seq_len (%d): %s seg=%d, forcing EOS (overflow)",
                max_seq,
                seg.session_id,
                seg.segment_idx,
            )
            self._handle_segment_eos(group, seg, overflow=True)

        if not candidates:
            return []

        ordered = self._mlfq.select_batch(
            candidates,
            self._max_batch,
            get_meta=lambda seg: seg.mlfq_meta,
        )
        return [seg.slot for seg in ordered]

    def _all_active_mlfq_metas(self) -> list[MLFQMeta]:
        """Collect all active segment MLFQ metas for aging."""
        return [
            seg.mlfq_meta
            for group in self._groups.values()
            for seg in group.segments.values()
            if seg.state == "active"
        ]

    # ------------------------------------------------------------------
    # Idle slot eviction
    # ------------------------------------------------------------------

    def _try_evict_idle_slots(self) -> None:
        """Periodically check for idle slots and evict them."""
        now = time.monotonic()
        if now - self._last_eviction_check < 1.0:
            return
        self._last_eviction_check = now

        kv_pool = self._executor.kv_pool
        if kv_pool is None:
            return

        while True:
            candidate = kv_pool.find_eviction_candidate(self._max_idle_sec)
            if candidate is None:
                break
            evicted_session_key = kv_pool.force_evict(candidate.slot_id)
            if evicted_session_key is None:
                break

            seg = self._seg_by_slot.pop(candidate.slot_id, None)
            if seg is None:
                continue

            group = self._groups.get(seg.session_id)
            if group is None:
                continue

            seg.state = "evicted"
            seg.slot = None
            self._send_result(
                group,
                EngineResult(
                    type=ResultType.ERROR,
                    session_id=seg.session_id,
                    segment_idx=seg.segment_idx,
                    error_msg=f"Slot evicted: idle > {self._max_idle_sec}s",
                ),
            )
            self._total_evictions += 1
            logger.warning(
                "Evicted segment %s:%d due to idle timeout",
                seg.session_id,
                seg.segment_idx,
            )
            self._remove_session(seg.session_id)

    # ------------------------------------------------------------------
    # Result processing (runs while GPU does next step)
    # ------------------------------------------------------------------

    def _process_step_output(self, output: StepOutput) -> None:
        self._result_batch = []
        try:
            self._process_step_output_inner(output)
        finally:
            batch, self._result_batch = self._result_batch, None
            if batch:
                self._async_loop.call_soon_threadsafe(
                    self._deliver_result_batch, batch
                )

    def _process_step_output_inner(self, output: StepOutput) -> None:
        kv_pool = self._executor.kv_pool
        use_pool = kv_pool is not None and kv_pool._preallocate

        # Batch-level KV scatter to pool (single operation, avoids per-slot split)
        if use_pool and output.batch_talker_kv is not None:
            slot_ids = [s.slot_id for s in output.slots]
            kv_pool.scatter_talker_kv_delta(
                slot_ids,
                output.batch_talker_kv,
                output.original_past_lens,
            )

        batch_size = len(output.slots)
        # One batch-level clone of the token-count staging output, sliced into
        # per-slot row views (1 kernel instead of one small clone per slot).
        # The clone is required: the staging tensor is rewritten next step.
        updated_tc_rows = (
            output.updated_tc.clone().split(1, dim=0)
            if output.updated_tc is not None
            else None
        )
        # L2 batch_compose(decode): emit only when the batch size changes, to
        # show continuous-batching composition over time without per-step flood.
        if batch_size != self._last_decode_batch and obs.is_enabled(obs.ObsLevel.DEBUG):
            members = []
            for s in output.slots:
                m = self._seg_by_slot.get(s.slot_id)
                if m is not None:
                    members.append(
                        {"session_id": m.session_id, "segment_id": m.segment_idx}
                    )
            LifecycleLogger.emit(
                session_id="-",
                phase="batch_compose",
                min_level=obs.ObsLevel.DEBUG,
                stage="decode",
                batch_size=batch_size,
                prev_batch_size=self._last_decode_batch,
                members=members,
            )
        self._last_decode_batch = batch_size
        # Arena-backed slots whose C2W state writes are deferred to one
        # batched indexed copy per state after this loop.
        arena_scatter: list[tuple[SlotKVState, int]] = []
        # Pooled slots whose C2W KV append is deferred to one batched
        # sliding-window shift+write after this loop.
        c2w_append: list[tuple[SlotKVState, int]] = []
        for i, slot in enumerate(output.slots):
            seg = self._seg_by_slot.get(slot.slot_id)
            if seg is None:
                continue
            if seg.max_decode_batch == 0:
                # First decode step for this segment (L1 lifecycle marker).
                group = self._groups.get(seg.session_id)
                LifecycleLogger.emit(
                    session_id=seg.session_id,
                    phase="engine.decode.first_step",
                    segment_idx=seg.segment_idx,
                    request_id=(
                        group.request.session_config.timing.request_id
                        if group and group.request.session_config
                        else None
                    )
                    or None,
                )
            if batch_size > seg.max_decode_batch:
                seg.max_decode_batch = batch_size
            group = self._groups.get(seg.session_id)
            if group is None:
                continue

            if output.batch_c2w_kv is not None:
                if slot.c2w_pooled:
                    # Deferred: one batched sliding-window append for all
                    # pooled slots after this loop.
                    c2w_append.append((slot, i))
                else:
                    kv = output.batch_c2w_kv[i : i + 1]
                    c2w_max_past = self._executor._config.c2w_sliding_window - 1
                    if slot.c2w_kv is None:
                        slot.c2w_kv = kv.clone()
                    else:
                        slot.c2w_kv = torch.cat([slot.c2w_kv, kv], dim=3)
                        if slot.c2w_kv.shape[3] > c2w_max_past:
                            slot.c2w_kv = slot.c2w_kv[
                                :, :, :, -c2w_max_past:, :
                            ].contiguous()
                        else:
                            slot.c2w_kv = slot.c2w_kv.contiguous()
            if not use_pool:
                if output.batch_talker_kv is not None:
                    kv = output.batch_talker_kv[i : i + 1]
                    if slot.talker_kv is None:
                        slot.talker_kv = kv.clone()
                    else:
                        slot.talker_kv = torch.cat(
                            [slot.talker_kv, kv], dim=3
                        ).contiguous()

            if output.used_pingpong and slot.pingpong_ready:
                # batch=1 zero-copy: TRT wrote directly to write bufs
                slot.flip_c2w_buffers()
            elif (
                slot.pingpong_ready
                and slot.c2w_arena_backed
                and output.batch_c2w_conv is not None
            ):
                # batch>1 arena-backed: deferred to one indexed copy per
                # state for the whole batch (see flush after this loop).
                arena_scatter.append((slot, i))
            elif slot.pingpong_ready:
                # batch>1 legacy per-slot copy into write bufs
                conv_rows, transconv_rows = self._c2w_output_rows(output, i)
                slot.copy_c2w_and_flip(conv_rows, transconv_rows)
            else:
                # Fallback: clone (first step or non-pingpong slot)
                conv_rows, transconv_rows = self._c2w_output_rows(output, i)
                if conv_rows:
                    slot.c2w_conv_states = [
                        t.clone() for t in conv_rows if t is not None
                    ]
                if transconv_rows:
                    slot.c2w_transconv_states = [
                        t.clone() for t in transconv_rows if t is not None
                    ]
            if updated_tc_rows is not None:
                slot.token_counts = updated_tc_rows[i]
            slot.past_len += 1
            slot.frame_idx += 1
            slot.touch()
            self._mlfq.on_step_done(seg.mlfq_meta)

            # --- Determine text_add and track pad phase ---
            in_pad = False
            if output.codec_sum is not None:
                if slot.trailing and slot.text_idx < len(slot.trailing):
                    text_add = slot.trailing[slot.text_idx].to(output.codec_sum.dtype)
                    slot.text_idx += 1
                    slot.pad_start_frame = -1
                    slot.pad_consecutive_silence = 0
                    slot.last_codec_sum = None
                    slot.next_embed = (output.codec_sum[i : i + 1] + text_add).to(
                        torch.float32
                    )
                elif not seg.input_complete:
                    # True streaming pause: preserve the latest codec_sum and
                    # wait for more text instead of injecting pad tokens, which
                    # creates artificial silences and prosody discontinuities.
                    slot.last_codec_sum = output.codec_sum[i : i + 1].clone()
                    slot.next_embed = None
                    slot.pad_start_frame = -1
                    slot.pad_consecutive_silence = 0
                    logger.debug(
                        "Paused streaming segment %s:%d awaiting text (frame=%d, past=%d)",
                        seg.session_id,
                        seg.segment_idx,
                        slot.frame_idx,
                        slot.past_len,
                    )
                else:
                    text_add = self._tts_pad_embed.to(output.codec_sum.dtype)
                    in_pad = True
                    slot.last_codec_sum = None
                    if slot.pad_start_frame < 0:
                        slot.pad_start_frame = slot.frame_idx
                    slot.next_embed = (output.codec_sum[i : i + 1] + text_add).to(
                        torch.float32
                    )
            else:
                slot.next_embed = None

            # --- Pad phase controls ---
            # Only two safeguards:
            #   1) Dynamic silence abort — stricter as KV budget shrinks
            #   2) KV overflow (past_len >= max_seq_len) — handled by
            #      _get_active_slots_mlfq before the next decode step
            pad_steps = (
                (slot.frame_idx - slot.pad_start_frame)
                if in_pad and slot.pad_start_frame >= 0
                else 0
            )

            if output.eos_flags[i]:
                self._handle_segment_eos(group, seg)
            else:
                audio = output.audio_chunks[i]

                if in_pad:
                    if audio is not None and len(audio) > 0:
                        if self._is_pad_silence(audio):
                            slot.pad_consecutive_silence += 1
                        else:
                            slot.pad_consecutive_silence = 0

                    if pad_steps >= self._min_pad_steps:
                        kv_pool = self._executor.kv_pool
                        max_seq = kv_pool.max_seq_len if kv_pool else 512
                        remaining_kv = max(0, max_seq - slot.past_len)
                        silence_limit = self._dynamic_silence_limit(remaining_kv)
                        if slot.pad_consecutive_silence > silence_limit:
                            logger.info(
                                "Silence abort: %s seg=%d silence=%d limit=%d "
                                "pad=%d remaining_kv=%d",
                                seg.session_id,
                                seg.segment_idx,
                                slot.pad_consecutive_silence,
                                silence_limit,
                                pad_steps,
                                remaining_kv,
                            )
                            # L2 pad_phase: why this segment got silence-aborted.
                            session_level = self._session_obs_level(group)
                            if obs.is_enabled(obs.ObsLevel.DEBUG, session_level):
                                LifecycleLogger.emit(
                                    session_id=seg.session_id,
                                    phase="pad_phase",
                                    segment_idx=seg.segment_idx,
                                    min_level=obs.ObsLevel.DEBUG,
                                    session_level=session_level,
                                    decision="silence_abort",
                                    pad_steps=pad_steps,
                                    pad_consecutive_silence=slot.pad_consecutive_silence,
                                    silence_limit=silence_limit,
                                    remaining_kv=remaining_kv,
                                )
                            self._handle_segment_eos(
                                group,
                                seg,
                                eos_reason="silence_abort",
                            )
                            continue

                if audio is not None and len(audio) > 0:
                    # -- First raw audio observability --
                    audio_metrics: dict = {}
                    if not seg.first_raw_audio_sent:
                        seg.first_raw_audio_sent = True
                        now_mono = time.monotonic()
                        # Write to ServerTimingAccumulator if available
                        group_acc = self._get_group_timing_accumulator(group)
                        if (
                            group_acc is not None
                            and group_acc.first_raw_audio_monotonic is None
                        ):
                            group_acc.first_raw_audio_monotonic = now_mono
                        LifecycleLogger.emit(
                            session_id=seg.session_id,
                            phase="engine.audio.first_raw",
                            segment_idx=seg.segment_idx,
                            request_id=(
                                group.request.session_config.timing.request_id
                                if group.request.session_config
                                else None
                            )
                            or None,
                            monotonic_ts=now_mono,
                        )
                        audio_metrics["first_raw_audio_at"] = str(now_mono)
                        if seg.dequeued_at is not None:
                            audio_metrics["first_text_dequeued_at"] = str(
                                seg.dequeued_at
                            )

                    self._send_result(
                        group,
                        EngineResult(
                            type=ResultType.AUDIO_CHUNK,
                            session_id=seg.session_id,
                            segment_idx=seg.segment_idx,
                            audio_bytes=audio,
                            metrics=audio_metrics,
                        ),
                    )

        if arena_scatter:
            # Slots released mid-loop (EOS/silence abort) are dropped: their
            # rows are stale until the next admission re-zeroes them.
            live = [
                (slot, pos)
                for slot, pos in arena_scatter
                if slot.c2w_arena_backed and not slot.is_free
            ]
            if live:
                self._executor.scatter_c2w_states_batch(
                    [slot for slot, _ in live],
                    [pos for _, pos in live],
                    output.batch_c2w_conv,
                    output.batch_c2w_transconv,
                )
                for slot, _ in live:
                    slot.flip_c2w_buffers()

        if c2w_append:
            live = [
                (slot, pos)
                for slot, pos in c2w_append
                if slot.c2w_pooled and not slot.is_free
            ]
            if live:
                kv_pool.append_c2w_frames(
                    [slot.slot_id for slot, _ in live],
                    output.batch_c2w_kv[
                        torch.tensor(
                            [pos for _, pos in live],
                            device=output.batch_c2w_kv.device,
                            dtype=torch.long,
                        )
                    ],
                )
                c2w_cap = self._executor._config.c2w_sliding_window - 1
                for slot, _ in live:
                    slot.c2w_len = min(slot.c2w_len + 1, c2w_cap)

    @staticmethod
    def _c2w_output_rows(
        output: StepOutput,
        i: int,
    ) -> tuple[list, list]:
        """Per-slot row views of the step's C2W state outputs.

        Prefers the batch-level tensors; falls back to the pre-split lists
        (test-constructed StepOutputs).
        """
        if output.batch_c2w_conv is not None:
            conv = [
                t[i : i + 1] if t is not None else None
                for t in output.batch_c2w_conv
            ]
            transconv = [
                t[i : i + 1] if t is not None else None
                for t in (output.batch_c2w_transconv or [])
            ]
            return conv, transconv
        conv = output.split_c2w_conv[i] if output.split_c2w_conv else []
        transconv = (
            output.split_c2w_transconv[i] if output.split_c2w_transconv else []
        )
        return conv, transconv

    @staticmethod
    def _dynamic_silence_limit(remaining_kv: int) -> int:
        """Silence frame threshold — stricter as KV budget shrinks.

        Mirrors old engine DecodeSessionFSM.dynamic_silence_limit:
        more patience early (remaining > 100 → 12 frames), increasingly
        aggressive as the slot approaches max_seq_len (≤ 20 → 1 frame).
        """
        if remaining_kv > 100:
            return 12
        if remaining_kv > 50:
            return 6
        if remaining_kv > 20:
            return 3
        return 1

    def _is_pad_silence(self, audio: bytes) -> bool:
        """Detect near-silent pad-phase audio frames.

        The previous peak-only `1e-4` threshold missed pathological pad loops
        that produce almost-flat chunks with tiny residual noise around
        `3e-4 ~ 5e-4`. Use both peak and mean absolute amplitude so we catch
        repeated near-silence without clipping normal quiet speech too
        aggressively.
        """
        audio_np = np.frombuffer(audio, dtype=np.float32)
        if audio_np.size == 0:
            return False
        abs_audio = np.abs(audio_np)
        peak = float(abs_audio.max(initial=0.0))
        mean_abs = float(abs_audio.mean())
        return (
            peak <= self._pad_silence_peak_threshold
            and mean_abs <= self._pad_silence_mean_abs_threshold
        )

    def _handle_segment_eos(
        self,
        group: EngineSessionGroup,
        seg: EngineSegment,
        *,
        overflow: bool = False,
        eos_reason: Optional[str] = None,
    ) -> None:
        """Handle EOS for one segment.

        ``eos_reason`` records *why* the segment ended (daily L1 observability):
        ``codec_eos`` (model emitted EOS — normal), ``kv_overflow`` (hit the KV
        budget cap), or ``silence_abort`` (pad-phase silence heuristic). When not
        given it is derived from ``overflow``.
        """
        self._total_eos += 1
        if eos_reason is None:
            eos_reason = "kv_overflow" if overflow else "codec_eos"
        audio_steps = 0
        slot_snapshot: Optional[dict] = None
        if seg.slot:
            audio_steps = seg.slot.frame_idx - seg.decode_start_frame
            sl = seg.slot
            slot_snapshot = {
                "slot_id": getattr(sl, "slot_id", None),
                "past_len": getattr(sl, "past_len", None),
                "frame_idx": getattr(sl, "frame_idx", None),
                "text_idx": getattr(sl, "text_idx", None),
                "trailing_len": len(getattr(sl, "trailing", []) or []),
                "c2w_kv_len": (
                    int(sl.c2w_kv.shape[3])
                    if getattr(sl, "c2w_kv", None) is not None
                    else int(getattr(sl, "c2w_len", 0))
                ),
            }
        text_tokens = seg.text_tokens_consumed
        audio_text_ratio = round(audio_steps / text_tokens, 2) if text_tokens else 0.0
        batched = seg.max_decode_batch > 1
        metrics = {
            "audio_steps": audio_steps,
            "text_tokens": text_tokens,
            "segment_idx": seg.segment_idx,
            "overflow": overflow,
            "eos_reason": eos_reason,
            "audio_text_ratio": audio_text_ratio,
            "batched": batched,
            "batch_size_seen": seg.max_decode_batch,
            "cache_hit": seg.cache_hit,
        }

        seg.state = "done"
        self._release_segment_slot(seg)

        self._send_result(
            group,
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id=seg.session_id,
                segment_idx=seg.segment_idx,
                metrics=metrics,
            ),
        )
        LifecycleLogger.emit(
            session_id=seg.session_id,
            phase="engine.segment.eos",
            segment_idx=seg.segment_idx,
            eos_reason=eos_reason,
            audio_steps=audio_steps,
            text_tokens=text_tokens,
            audio_text_ratio=audio_text_ratio,
            batched=batched,
            batch_size_seen=seg.max_decode_batch,
            cache_hit=seg.cache_hit,
        )
        logger.info(
            "Segment EOS: %s seg=%d reason=%s audio_steps=%d text_tokens=%d "
            "ratio=%.2f batch=%d",
            seg.session_id,
            seg.segment_idx,
            eos_reason,
            audio_steps,
            text_tokens,
            audio_text_ratio,
            seg.max_decode_batch,
        )

        # L2 segment_synthesis: sampling params + anomaly heuristics that flag
        # the C4 hallucination / non-termination signature (answers "合成为什么错").
        session_level = self._session_obs_level(group)
        if obs.is_enabled(obs.ObsLevel.DEBUG, session_level):
            kv_pool = self._executor.kv_pool
            max_seq = kv_pool.max_seq_len if kv_pool else 512
            anomaly: list[str] = []
            if eos_reason == "kv_overflow" or audio_steps >= max_seq - 1:
                anomaly.append("hit_kv_cap")
            if eos_reason == "silence_abort":
                anomaly.append("silence_aborted")
            if eos_reason != "codec_eos":
                anomaly.append("no_codec_eos")
            if audio_text_ratio > 10.0:
                anomaly.append("ratio_outlier")
            reason = (
                "ran to KV cap without codec EOS — likely hallucination tail"
                if "hit_kv_cap" in anomaly
                else "pad-phase silence abort"
                if "silence_aborted" in anomaly
                else "normal codec EOS"
            )
            LifecycleLogger.emit(
                session_id=seg.session_id,
                phase="segment_synthesis",
                segment_idx=seg.segment_idx,
                min_level=obs.ObsLevel.DEBUG,
                session_level=session_level,
                do_sample=self._executor._do_sample,
                temperature=self._executor._temperature,
                repetition_penalty=self._executor._repetition_penalty,
                audio_steps=audio_steps,
                text_tokens=text_tokens,
                audio_text_ratio=audio_text_ratio,
                eos_reason=eos_reason,
                anomaly=anomaly,
                reason=reason,
            )
            if slot_snapshot is not None:
                LifecycleLogger.emit(
                    session_id=seg.session_id,
                    phase="slot_state",
                    segment_idx=seg.segment_idx,
                    min_level=obs.ObsLevel.DEBUG,
                    session_level=session_level,
                    **slot_snapshot,
                )

        self._check_session_done(group)

    def _check_session_done(self, group: EngineSessionGroup) -> None:
        """Check if ALL segments are done and no more are expected.

        Requires input_complete_all (SESSION_TOKENS_DONE received) so that
        streaming sessions don't conclude before all text has arrived.
        Also waits for overflow_token_ids to be drained into new segments.
        """
        if not group.input_complete_all:
            logger.debug(
                "Session %s _check_session_done: input_complete_all=False, skipping",
                group.session_id,
            )
            return

        if group.overflow_token_ids:
            logger.warning(
                "Session %s has %d overflow tokens pending — waiting for new segment",
                group.session_id,
                len(group.overflow_token_ids),
            )
            return

        all_done = all(s.state == "done" for s in group.segments.values())
        if not all_done:
            seg_states = {idx: s.state for idx, s in group.segments.items()}
            logger.debug(
                "Session %s _check_session_done: not all done, seg_states=%s",
                group.session_id,
                seg_states,
            )
            return

        logger.info(
            "Session %s _check_session_done: ALL DONE, sending SESSION_DONE",
            group.session_id,
        )
        self._send_result(
            group,
            EngineResult(
                type=ResultType.SESSION_DONE,
                session_id=group.session_id,
            ),
        )
        self._remove_session(group.session_id)

    # ------------------------------------------------------------------
    # Session cleanup
    # ------------------------------------------------------------------

    def _remove_session(self, session_id: str) -> None:
        group = self._groups.pop(session_id, None)
        if group is None:
            return
        for seg in group.segments.values():
            self._release_segment_slot(seg)

    def _release_segment_slot(self, seg: EngineSegment) -> None:
        slot = seg.slot
        if slot is None:
            return
        slot_id = slot.slot_id
        kv_pool = self._executor.kv_pool
        if kv_pool is not None:
            kv_pool.release(slot_id)
        self._seg_by_slot.pop(slot_id, None)
        seg.slot = None

    def _cleanup_failed_prefills(self) -> None:
        failed_sessions: list[str] = []
        for group in list(self._groups.values()):
            for seg in group.segments.values():
                if seg.state == "pending_prefill" and seg.slot is not None:
                    failed_sessions.append(group.session_id)
                    self._send_result(
                        group,
                        EngineResult(
                            type=ResultType.ERROR,
                            session_id=seg.session_id,
                            segment_idx=seg.segment_idx,
                            error_msg="Prefill failed unexpectedly",
                        ),
                    )
                    logger.warning(
                        "Cleaning failed prefill session %s seg=%d slot=%d",
                        seg.session_id,
                        seg.segment_idx,
                        seg.slot.slot_id,
                    )
                    break
        for session_id in failed_sessions:
            self._remove_session(session_id)

    # ------------------------------------------------------------------
    # Result delivery (cross-thread)
    # ------------------------------------------------------------------

    def _send_result(
        self,
        group: EngineSessionGroup,
        result: EngineResult,
    ) -> None:
        if group.result_queue is None:
            return
        q = group.result_queue
        if self._result_batch is not None:
            self._result_batch.append((q, result))
            return
        self._async_loop.call_soon_threadsafe(q.put_nowait, result)

    @staticmethod
    def _deliver_result_batch(items: list) -> None:
        """Runs on the asyncio loop: fan a step's results out in order.

        Per-item isolation matters: session result queues are bounded, and
        one clogged session's QueueFull must only drop its own item — not
        abort delivery for every other session in the same step batch.
        """
        for q, result in items:
            try:
                q.put_nowait(result)
            except asyncio.QueueFull:
                logger.warning(
                    "Result queue full, dropping %s for session %s",
                    getattr(result, "type", "?"),
                    getattr(result, "session_id", "?"),
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_work(self) -> bool:
        return any(
            seg.state in ("pending_prefill", "active")
            for group in self._groups.values()
            for seg in group.segments.values()
        )

    # ------------------------------------------------------------------
    # Text embedding helpers (for streaming APPEND_TOKENS)
    # ------------------------------------------------------------------

    def _append_trailing_tokens(
        self,
        slot: SlotKVState,
        token_ids: list[int],
    ) -> None:
        """Embed new token IDs and append to slot's trailing list.

        Called when APPEND_TOKENS arrives after a segment has already been
        prefilled, so the model can see the new text during decode.
        """
        w = self._prefill_builder.w
        ids_tensor = torch.tensor(
            [token_ids],
            device=w.device,
            dtype=torch.int64,
        )
        with torch.no_grad():
            embed = w.text_embed(ids_tensor)
        embed = self._coerce_embed_tensor(embed)
        for i in range(embed.shape[1]):
            slot.trailing.append(embed[:, i : i + 1, :].clone())
        logger.debug(
            "Appended %d trailing tokens (total=%d, text_idx=%d)",
            len(token_ids),
            len(slot.trailing),
            slot.text_idx,
        )

    def _append_eos_trailing(self, seg: EngineSegment) -> None:
        """Append tts_eos_embed to trailing when SEGMENT_TOKENS_DONE arrives post-prefill."""
        w = self._prefill_builder.w
        seg.slot.trailing.append(self._coerce_embed_tensor(w.tts_eos_embed).clone())
        seg.eos_trailing_added = True
        logger.debug(
            "Appended EOS trailing for seg=%d (total=%d)",
            seg.segment_idx,
            len(seg.slot.trailing),
        )

    def _coerce_embed_tensor(
        self,
        tensor: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        target_dtype = dtype
        if target_dtype is None and tensor.is_floating_point():
            target_dtype = self._embed_dtype
        return tensor.to(
            device=self._embed_device,
            dtype=target_dtype if target_dtype is not None else tensor.dtype,
        )

    # ------------------------------------------------------------------
    # Session timeout
    # ------------------------------------------------------------------

    def _try_timeout_sessions(self) -> None:
        """Cancel sessions that exceed the maximum allowed duration."""
        if self._session_timeout_sec <= 0:
            return
        now = time.monotonic()
        to_cancel: list[str] = []
        for sid, group in self._groups.items():
            if now - group.created_at > self._session_timeout_sec:
                to_cancel.append(sid)
        for sid in to_cancel:
            group = self._groups.get(sid)
            if group is None:
                continue
            self._total_timeouts += 1
            self._send_result(
                group,
                EngineResult(
                    type=ResultType.ERROR,
                    session_id=sid,
                    error_msg=f"Session timeout ({self._session_timeout_sec}s exceeded)",
                ),
            )
            self._remove_session(sid)
            logger.warning("Session %s timed out", sid)

    # ------------------------------------------------------------------
    # ServerTimingAccumulator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_timing_accumulator(req: EngineRequest):
        """Get the ServerTimingAccumulator from an EngineRequest, if present."""
        if req.session_config is None:
            return None
        extra = req.session_config.timing.extra
        if not isinstance(extra, dict):
            return None
        acc = extra.get("_server_timing_accumulator")
        if acc is not None and hasattr(acc, "to_meta_dict"):
            return acc
        return None

    def _get_group_timing_accumulator(self, group: EngineSessionGroup):
        """Get the ServerTimingAccumulator from an EngineSessionGroup."""
        return self._get_timing_accumulator(group.request)

    # ------------------------------------------------------------------
    # Health / metrics (thread-safe read)
    # ------------------------------------------------------------------

    def health_stats(self) -> dict:
        """Return a snapshot of engine health metrics.

        Called from the asyncio thread; reads only atomic int/float fields
        so no lock is needed.
        """
        kv_pool = self._executor.kv_pool
        return {
            "running": self._running,
            "active_sessions": len(self._groups),
            "active_slots": kv_pool.used_count if kv_pool else 0,
            "free_slots": kv_pool.free_count if kv_pool else 0,
            "pool_utilization": kv_pool.utilization if kv_pool else 0.0,
            "total_steps": self._total_steps,
            "total_prefills": self._total_prefills,
            "total_sessions": self._total_sessions,
            "total_eos": self._total_eos,
            "total_evictions": self._total_evictions,
            "total_timeouts": self._total_timeouts,
            "prefix_cache_stats": self._prefix_cache.stats,
        }
