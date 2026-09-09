"""Backend-facing dispatcher for frontend session traffic.

This module intentionally stays below the gateway/interface boundary.
It knows how to translate session state + Spliter actions into
``EngineRequest`` messages for the backend thread, but it does not own
external session lifecycle APIs.
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import List

from ..core.lifecycle import LifecycleLogger
from ..core.session import Session, SegmentOrderMeta
from ..core.timing import ServerTimingAccumulator
from ..core.types import EngineRequest, RequestPriority, RequestType
from .spliter import SegmentAction
from .spliter.driver import ActionType

logger = logging.getLogger(__name__)


class Dispatcher:
    """Translate frontend session actions into backend requests."""

    def __init__(self, engine_inbox: asyncio.Queue):
        self._engine_inbox = engine_inbox

    async def submit_new_session(self, session: Session) -> None:
        """Register a new session with the backend engine thread."""
        config = session.config
        req = EngineRequest(
            type=RequestType.NEW_SESSION,
            session_id=session.session_id,
            session_config=config,
            speaker_key=config.speaker,
            task_type=config.task_type,
            ref_audio=config.ref_audio,
            result_queue=session.result_queue,
            enqueued_at=time.monotonic(),
        )
        await self._engine_inbox.put(req)

    async def submit_cancel(self, session_id: str, *, reason: str = "") -> None:
        req = EngineRequest(
            type=RequestType.CANCEL_SESSION,
            session_id=session_id,
            cancel_reason=str(reason or "") or None,
            enqueued_at=time.monotonic(),
        )
        await self._engine_inbox.put(req)

    async def submit_session_tokens_done(self, session_id: str) -> None:
        req = EngineRequest(
            type=RequestType.SESSION_TOKENS_DONE,
            session_id=session_id,
            enqueued_at=time.monotonic(),
        )
        await self._engine_inbox.put(req)

    async def submit_cursor_plan(self, session: Session) -> None:
        """Publish an owner-safe segment plan to active segments.

        A segment gets only complete TN owners.  If its bounds are unknown or
        cut through an owner, the session returns an inactive plan for that
        segment and it stays on EMA. Queue ordering gives the engine thread a
        happens-before edge relative to later token requests from this event-
        loop turn.
        """
        if session.cursor_label_plan is None:
            return
        for segment_idx in tuple(session.segment_order):
            plan = session.cursor_plan_for_segment(segment_idx)
            if plan is None:
                continue
            session.cursor_segment_plans[segment_idx] = plan
            await self._engine_inbox.put(
                EngineRequest(
                    type=RequestType.UPDATE_CURSOR_PLAN,
                    session_id=session.session_id,
                    segment_idx=segment_idx,
                    cursor_label_plan=plan,
                    enqueued_at=time.monotonic(),
                )
            )

    async def dispatch_segment_actions(
        self,
        session: Session,
        actions: List[SegmentAction],
    ) -> None:
        """Translate Spliter actions into EngineRequests and submit them."""
        for sa in actions:
            seg_idx = sa.segment_idx
            action = sa.action
            priority = self._segment_priority(session, sa)
            # Spliter always emits explicit coordinates (a streaming segment is
            # its own group with group_idx == segment_idx); no -1 sentinel.
            order_meta = SegmentOrderMeta(
                group_idx=sa.group_idx,
                local_idx=sa.local_idx,
                group_final=sa.group_final,
            )

            if action.type == ActionType.PREFILL:
                session.segments_submitted += 1
                session.segment_order[seg_idx] = order_meta
                segment_plan = session.cursor_plan_for_segment(seg_idx)
                if segment_plan is not None:
                    session.cursor_segment_plans[seg_idx] = segment_plan

                now = time.monotonic()
                # Record first text enqueue timestamp on session
                if session.first_text_enqueued_at is None:
                    session.first_text_enqueued_at = now
                    # Write to ServerTimingAccumulator if present
                    acc = session.config.timing.extra.get("_server_timing_accumulator")
                    if isinstance(acc, ServerTimingAccumulator):
                        acc.first_text_enqueued_monotonic = now
                    # Emit lifecycle event
                    LifecycleLogger.emit(
                        session_id=session.session_id,
                        phase="text.first_enqueued",
                        segment_idx=seg_idx,
                        request_id=session.config.timing.request_id or None,
                        turn_id=session.config.timing.turn_id or None,
                        monotonic_ts=now,
                        queue_depth=self._engine_inbox.qsize(),
                    )

                req = EngineRequest(
                    type=RequestType.START_TOKENS,
                    session_id=session.session_id,
                    segment_idx=seg_idx,
                    priority=priority,
                    token_ids=[action.token],
                    cursor_label_plan=segment_plan,
                    result_queue=session.result_queue,
                    enqueued_at=now,
                )
                await self._engine_inbox.put(req)

            elif action.type == ActionType.DECODE:
                req = EngineRequest(
                    type=RequestType.APPEND_TOKENS,
                    session_id=session.session_id,
                    segment_idx=seg_idx,
                    priority=priority,
                    token_ids=[action.token],
                    enqueued_at=time.monotonic(),
                )
                await self._engine_inbox.put(req)

            elif action.type in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                req = EngineRequest(
                    type=RequestType.SEGMENT_TOKENS_DONE,
                    session_id=session.session_id,
                    segment_idx=seg_idx,
                    append_eos=(action.type == ActionType.FLUSH_EOS),
                    enqueued_at=time.monotonic(),
                )
                await self._engine_inbox.put(req)

    async def maybe_send_session_tokens_done(self, session: Session) -> None:
        """Signal session-level token completion when no more groups remain."""
        if session.engine_tokens_done_sent or session.spliter is None:
            return
        if not session.input_complete:
            return
        if not self._spliter_has_dispatched_all_input(session):
            return
        await self.submit_session_tokens_done(session.session_id)
        session.engine_tokens_done_sent = True

    @staticmethod
    def _spliter_has_dispatched_all_input(session: Session) -> bool:
        spliter = session.spliter
        if spliter is None:
            return True

        pending = getattr(spliter, "_pending", None)
        if pending:
            return False

        drivers = getattr(spliter, "_drivers", {}) or {}
        flushing = getattr(spliter, "_flushing", set()) or set()
        done = getattr(spliter, "_done", set()) or set()
        for segment_idx in drivers:
            if segment_idx not in flushing and segment_idx not in done:
                return False
        return True

    def _segment_priority(
        self,
        session: Session,
        sa: SegmentAction,
    ) -> RequestPriority:
        """Lower numeric value means higher urgency."""
        segment_idx = sa.segment_idx
        if segment_idx == 0 and session.segments_submitted == 0:
            return RequestPriority.FIRST_SEGMENT
        if sa.local_idx > 0:
            return RequestPriority.CONTINUATION
        if segment_idx > 0 and (segment_idx - 1) in (
            session.spliter._flushing if session.spliter else set()
        ):
            return RequestPriority.CONTINUATION
        return RequestPriority.PREFETCHED
