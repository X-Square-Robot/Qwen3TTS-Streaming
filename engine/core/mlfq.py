"""Multi-Level Feedback Queue for per-segment dynamic priority scheduling.

Adapted from the old BLS engine's mlfq_scheduler.py to work with the
new EngineSegment model.  The MLFQ ensures fair GPU time distribution:

  Q0 (high):   new sessions / freshly prefilled segments (TTFB critical)
  Q1 (medium): segments that consumed > q1_threshold decode steps
  Q2 (low):    long-running segments, still making progress

Anti-starvation: segments stuck at Q2 for > starvation_limit global steps
are boosted back to Q0 to prevent indefinite starving.

Thread safety:
    Called only from the engine thread — no locking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class MLFQConfig:
    """Tuneable MLFQ parameters (loaded from engine.yaml → SchedulerConfig)."""

    q1_threshold: int = 50
    q2_threshold: int = 200
    aging_interval: int = 100
    starvation_limit: int = 50


@dataclass
class MLFQMeta:
    """Per-segment MLFQ tracking state."""

    level: int = 0
    decode_steps: int = 0
    last_scheduled_at: int = 0
    steps_since_schedule: int = 0

    def reset(self) -> None:
        self.level = 0
        self.decode_steps = 0
        self.steps_since_schedule = 0


class MLFQScheduler:
    """Stateful MLFQ scheduler for the engine loop.

    Usage in EngineLoop._run():
        # After building candidate segments:
        ordered = mlfq.select_batch(candidates, max_batch_size)
        # After each decode step:
        for seg_meta in batch_metas:
            mlfq.on_step_done(seg_meta)
        mlfq.tick()  # advance global step, run aging
    """

    def __init__(self, config: Optional[MLFQConfig] = None):
        self._cfg = config or MLFQConfig()
        self._global_step: int = 0
        # ids of metas scheduled into a batch during the current step;
        # consumed (and cleared) by tick() to age the segments that were
        # *not* scheduled. See tick() for the anti-starvation rationale.
        self._scheduled_this_step: set[int] = set()

    @property
    def global_step(self) -> int:
        return self._global_step

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------

    def on_segment_created(self, meta: MLFQMeta) -> None:
        """Called when a new segment starts decoding."""
        meta.reset()

    def on_segment_boundary(self, meta: MLFQMeta) -> None:
        """Called at segment boundary — resets to Q0 for low latency."""
        meta.reset()

    def on_step_done(self, meta: MLFQMeta) -> None:
        """Called after each decode step for a segment.

        Note: ``steps_since_schedule`` is *not* advanced here. It tracks
        time since a segment was last selected into a batch, which is an
        attribute of the global step rather than of a per-segment decode
        step — so it is aged in :meth:`tick` for every active segment that
        was not scheduled this step.
        """
        meta.decode_steps += 1

        if meta.level == 0 and meta.decode_steps >= self._cfg.q1_threshold:
            meta.level = 1
        elif meta.level == 1 and meta.decode_steps >= self._cfg.q2_threshold:
            meta.level = 2

    def on_scheduled(self, meta: MLFQMeta) -> None:
        """Mark a segment as having been included in the current batch."""
        meta.last_scheduled_at = self._global_step
        meta.steps_since_schedule = 0
        self._scheduled_this_step.add(id(meta))

    # ------------------------------------------------------------------
    # Batch selection
    # ------------------------------------------------------------------

    def select_batch(
        self,
        candidates: List,
        max_batch: int,
        *,
        get_meta,
    ) -> List:
        """Order candidates by MLFQ priority and return up to max_batch.

        Args:
            candidates: list of segment-like objects (EngineSegment or similar)
            max_batch: maximum batch size
            get_meta: callable(candidate) -> MLFQMeta

        Returns:
            Ordered list, Q0 first, FIFO within each level.
        """
        buckets: dict[int, list] = {0: [], 1: [], 2: []}
        for seg in candidates:
            meta = get_meta(seg)
            lev = min(2, max(0, meta.level))
            buckets[lev].append(seg)

        result: list = []
        for lev in (0, 1, 2):
            for seg in buckets[lev]:
                if len(result) >= max_batch:
                    break
                self.on_scheduled(get_meta(seg))
                result.append(seg)
            if len(result) >= max_batch:
                break

        return result

    # ------------------------------------------------------------------
    # Global tick (aging / anti-starvation)
    # ------------------------------------------------------------------

    def tick(self, all_metas: Optional[List[MLFQMeta]] = None) -> int:
        """Advance global step counter and run anti-starvation aging.

        Args:
            all_metas: all active segment MLFQMeta objects (for aging check).

        Returns:
            Number of segments boosted by anti-starvation.
        """
        # Age every active segment that was *not* scheduled into this step's
        # batch. Scheduled segments had ``steps_since_schedule`` reset to 0 in
        # on_scheduled, so this measures global steps since last scheduled and
        # lets the starvation boost below actually fire for segments that the
        # batch never picks (e.g. when candidates exceed max_batch).
        if all_metas:
            scheduled = self._scheduled_this_step
            for meta in all_metas:
                if id(meta) not in scheduled:
                    meta.steps_since_schedule += 1
        self._scheduled_this_step.clear()

        self._global_step += 1
        boosted = 0

        if (
            all_metas
            and self._cfg.aging_interval > 0
            and self._global_step % self._cfg.aging_interval == 0
        ):
            for meta in all_metas:
                if (
                    meta.level >= 2
                    and meta.steps_since_schedule >= self._cfg.starvation_limit
                ):
                    meta.level = 0
                    meta.steps_since_schedule = 0
                    boosted += 1
            if boosted > 0:
                logger.debug(
                    "MLFQ aging at step %d: boosted %d starved segments",
                    self._global_step,
                    boosted,
                )

        return boosted
