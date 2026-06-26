"""Gateway-facing frontend interface.

The interface owns external session lifecycle and input semantics:

- create/cancel sessions for transport adapters
- tokenize and route text according to declared input mode
- consume backend results and surface ordered audio callbacks

It delegates backend request emission to ``frontend.dispatcher.Dispatcher``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Dict, Optional, TYPE_CHECKING

from ..core.session import Session, SegmentOrderMeta
from ..core.types import (
    EngineResult,
    GroupPolicy,
    InputMode,
    ResultType,
    SegmentToken,
    SessionConfig,
    SessionState,
    TokenizedText,
)
from ..core.lifecycle import LifecycleLogger
from ..core.timing import ServerTimingAccumulator
from ..text_normalization import strip_emoji
from .dispatcher import Dispatcher
from .spliter import Spliter
from .spliter.driver import ActionType
from .spliter.reorder import AudioReorder

if TYPE_CHECKING:
    from .spliter.tokenizer import LightQwen3TTSTokenizer

logger = logging.getLogger(__name__)

_WHITESPACE_TO_STRIP = str.maketrans({
    "\n": "",
    "\r": "",
    "\t": " ",
    "\u3000": "",
})


def _normalize_tts_text(text: str) -> str:
    """Remove formatting whitespace that harms tokenization/prosody."""
    text = strip_emoji((text or "").translate(_WHITESPACE_TO_STRIP))
    while "  " in text:
        text = text.replace("  ", " ")
    return text


class FrontendInterface:
    """Gateway-facing session and text interface."""

    def __init__(
        self,
        engine_inbox: asyncio.Queue,
        tokenizer: LightQwen3TTSTokenizer,
        *,
        max_sessions: int = 128,
        engine_max_decode_len: int = 512,
        prefill_len: int = 12,
        ema_ratio: float = 5.0,
        max_concurrent_segments: int = 2,
        ema_alpha: float = 0.1,
        ema_overflow_alpha: float = 0.5,
        ema_min_ratio: float = 2.0,
        ema_max_ratio: float = 10.0,
        safety_margin: int = 8,
        l1_split_cap_ratio: float = 0.70,
        l2_split_cap_ratio: float = 0.80,
        l3_split_cap_ratio: float = 0.90,
    ):
        self._dispatcher = Dispatcher(engine_inbox)
        self._tokenizer = tokenizer
        self._max_sessions = max_sessions
        self._engine_max = engine_max_decode_len
        self._prefill_len = prefill_len
        self._ema_ratio = ema_ratio
        self._max_concurrent = max_concurrent_segments
        self._ema_alpha = ema_alpha
        self._ema_overflow_alpha = ema_overflow_alpha
        self._ema_min_ratio = ema_min_ratio
        self._ema_max_ratio = ema_max_ratio
        self._safety_margin = safety_margin
        self._l1_split_cap_ratio = l1_split_cap_ratio
        self._l2_split_cap_ratio = l2_split_cap_ratio
        self._l3_split_cap_ratio = l3_split_cap_ratio

        self._sessions: Dict[str, Session] = {}
        self._consumer_tasks: Dict[str, asyncio.Task] = {}

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def create_session(
        self,
        session_id: str,
        *,
        config: Optional[SessionConfig] = None,
        speaker_key: Optional[str] = None,
        task_type: str = "custom",
        ref_audio: Optional[bytes] = None,
        on_audio: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
        on_event: Optional[Callable] = None,
    ) -> Session:
        if session_id in self._sessions:
            await self.cancel_session(session_id)

        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError(f"Max sessions ({self._max_sessions}) reached")

        if config is None:
            config = SessionConfig(
                task_type=task_type,
                speaker=speaker_key,
                ref_audio=ref_audio,
            )
        self._prepare_session_config(config)

        session = Session(session_id=session_id, config=config)
        session.spliter = Spliter(
            engine_max_decode_len=self._engine_max,
            prefill_len=self._prefill_len,
            ema_ratio=self._ema_ratio,
            max_concurrent=self._max_concurrent,
            ema_alpha=self._ema_alpha,
            ema_overflow_alpha=self._ema_overflow_alpha,
            ema_min_ratio=self._ema_min_ratio,
            ema_max_ratio=self._ema_max_ratio,
            safety_margin=self._safety_margin,
            l1_split_cap_ratio=self._l1_split_cap_ratio,
            l2_split_cap_ratio=self._l2_split_cap_ratio,
            l3_split_cap_ratio=self._l3_split_cap_ratio,
        )
        session.reorder = AudioReorder()
        session.event_callback = on_event
        self._sessions[session_id] = session

        # Emit lifecycle events
        LifecycleLogger.emit(
            session_id=session_id,
            phase="session.config.validated",
            request_id=config.timing.request_id or None,
            turn_id=config.timing.turn_id or None,
            input_mode=config.input_mode.value,
            vad_strategy=config.output_policy.vad.strategy or "disabled",
            protocol_version=str(config.timing.extra.get("client_protocol_version", "")),
        )
        LifecycleLogger.emit(
            session_id=session_id,
            phase="session.created",
            request_id=config.timing.request_id or None,
            turn_id=config.timing.turn_id or None,
            monotonic_ts=session.created_at,
            speaker_id=config.speaker,
        )

        # Set session_created on ServerTimingAccumulator if present
        acc = config.timing.extra.get("_server_timing_accumulator")
        if isinstance(acc, ServerTimingAccumulator):
            acc.session_created_monotonic = session.created_at

        task = asyncio.create_task(
            self._consume_results(session, on_audio=on_audio, on_done=on_done, on_event=on_event)
        )
        self._consumer_tasks[session_id] = task

        await self._dispatcher.submit_new_session(session)
        logger.info("Session %s created (active: %d)", session_id, self.active_count)
        return session

    async def cancel_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.state = SessionState.DONE
        await self._dispatcher.submit_cancel(session_id)
        task = self._consumer_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        self._cleanup_session(session_id)

    async def push_text_input(self, session_id: str, text: str) -> None:
        """Feed transport text according to the session's declared input mode."""
        session = self._sessions.get(session_id)
        if session is None or session.state == SessionState.DONE:
            return
        text = _normalize_tts_text(text)

        mode = session.config.input_mode
        if not text:
            return
        if mode != InputMode.TOKEN and not text.strip():
            return
        if mode == InputMode.FULL_TEXT:
            session.append_text(text)
            return

        tokens = self._tokenize_segment_text(text)
        if not tokens:
            return

        spliter: Spliter = session.spliter
        if mode == InputMode.LONG_SEGMENT and session.config.group_policy != GroupPolicy.NONE:
            seg_actions = spliter.push_group_tokens(tokens)
        else:
            seg_actions = spliter.feed_tokens(tokens)
        await self._dispatch_segment_actions(session, seg_actions)

    async def feed_full_text(self, session_id: str, text: str) -> None:
        """Explicit offline mode: set complete text, pre-split, drive all segments."""
        session = self._sessions.get(session_id)
        if session is None or session.state == SessionState.DONE:
            return
        text = _normalize_tts_text(text).strip()
        if not text:
            return

        tokens = self._tokenize_segment_text(text)
        if not tokens:
            return

        session.mark_input_complete()
        seg_actions = session.spliter.set_full_text(tokens)
        await self._dispatch_segment_actions(session, seg_actions)
        await self._dispatcher.maybe_send_session_tokens_done(session)

    async def mark_input_complete(self, session_id: str) -> None:
        """Upstream signals no more transport input will arrive."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.mark_input_complete()

        mode = session.config.input_mode
        if mode == InputMode.FULL_TEXT:
            full_text = session.drain_text()
            if full_text.strip():
                await self.feed_full_text(session_id, full_text)
            else:
                await self._dispatcher.submit_session_tokens_done(session_id)
                session.engine_tokens_done_sent = True
            return

        if mode == InputMode.LONG_SEGMENT and session.config.group_policy != GroupPolicy.NONE:
            await self._dispatcher.maybe_send_session_tokens_done(session)
            return

        seg_actions = session.spliter.input_done()
        await self._dispatch_segment_actions(session, seg_actions)
        await self._dispatcher.maybe_send_session_tokens_done(session)

    async def _consume_results(
        self,
        session: Session,
        *,
        on_audio: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
        on_event: Optional[Callable] = None,
    ) -> None:
        try:
            while True:
                result: EngineResult = await session.result_queue.get()

                if result.type == ResultType.AUDIO_CHUNK:
                    session.record_first_audio()
                    audio = result.audio_bytes or b""
                    session.total_audio_bytes += len(audio)

                    # Propagate raw audio timestamp from engine thread
                    if result.metrics and "first_raw_audio_at" in result.metrics:
                        try:
                            session.first_raw_audio_at = float(result.metrics["first_raw_audio_at"])
                        except (ValueError, TypeError):
                            pass

                    reorder = session.reorder
                    meta = session.segment_order.get(
                        result.segment_idx,
                        SegmentOrderMeta(result.segment_idx, 0, True),
                    )
                    ready = reorder.push(meta.group_idx, meta.local_idx, audio)
                    if ready and on_audio:
                        for chunk in ready:
                            await on_audio(session.session_id, chunk)

                elif result.type == ResultType.PREFILL_DONE:
                    # Propagate prefill timing from engine thread
                    if result.metrics:
                        if "prefill_started_at" in result.metrics and session.prefill_started_at is None:
                            try:
                                session.prefill_started_at = float(result.metrics["prefill_started_at"])
                            except (ValueError, TypeError):
                                pass
                        if "prefill_completed_at" in result.metrics and session.prefill_completed_at is None:
                            try:
                                session.prefill_completed_at = float(result.metrics["prefill_completed_at"])
                            except (ValueError, TypeError):
                                pass
                        if "first_text_dequeued_at" in result.metrics and session.first_text_dequeued_at is None:
                            try:
                                session.first_text_dequeued_at = float(result.metrics["first_text_dequeued_at"])
                            except (ValueError, TypeError):
                                pass

                    if on_event:
                        await on_event(
                            session.session_id,
                            {
                                "type": "prefill_done",
                                "segment_idx": result.segment_idx,
                                "text": session.segment_texts.get(result.segment_idx, ""),
                                "meta": {
                                    str(k): str(v)
                                    for k, v in (result.metrics or {}).items()
                                },
                            },
                        )

                elif result.type == ResultType.SEGMENT_END:
                    seg_idx = result.segment_idx
                    session.segments_done += 1

                    reorder = session.reorder
                    meta = session.segment_order.pop(
                        seg_idx,
                        SegmentOrderMeta(seg_idx, 0, True),
                    )
                    ready = reorder.mark_done(
                        meta.group_idx, meta.local_idx, group_final=meta.group_final,
                    )
                    if ready and on_audio:
                        for chunk in ready:
                            await on_audio(session.session_id, chunk)

                    if result.metrics:
                        audio_steps = result.metrics.get("audio_steps", 0)
                        text_tokens = result.metrics.get("text_tokens", 0)
                        overflow = result.metrics.get("overflow", False)
                        if audio_steps > 0 and text_tokens > 0:
                            session.spliter.update_ratio(
                                audio_steps, text_tokens, overflow=overflow,
                            )

                    if on_event:
                        metrics = {
                            str(k): str(v) for k, v in (result.metrics or {}).items()
                        }
                        segment_text = session.segment_texts.pop(seg_idx, "")
                        # Add segment-level timing observability
                        metrics["segment_id"] = str(seg_idx)
                        if segment_text:
                            preview = segment_text[:64] + "..." if len(segment_text) > 64 else segment_text
                            metrics["segment_text_preview"] = preview
                        if "audio_steps" in (result.metrics or {}):
                            metrics["segment_decode_steps"] = str(result.metrics["audio_steps"])
                        if "text_tokens" in (result.metrics or {}):
                            metrics["segment_text_tokens"] = str(result.metrics["text_tokens"])
                        if "cache_hit" in (result.metrics or {}):
                            metrics["segment_cache_hit"] = str(result.metrics["cache_hit"])
                        if "prefill_duration_ms" in (result.metrics or {}):
                            metrics["segment_prefill_ms"] = str(result.metrics["prefill_duration_ms"])
                        session.segment_token_emitted_count.pop(seg_idx, None)
                        session.text_boundary_emitted.discard(seg_idx)
                        await on_event(
                            session.session_id,
                            {
                                "type": "segment_end",
                                "segment_idx": seg_idx,
                                "text": segment_text,
                                "meta": metrics,
                            },
                        )

                    new_actions = session.spliter.on_segment_done(seg_idx)
                    if new_actions:
                        await self._dispatch_segment_actions(session, new_actions)
                    await self._dispatcher.maybe_send_session_tokens_done(session)

                elif result.type == ResultType.RATIO_UPDATE:
                    if session.spliter and result.ema_ratio > 0:
                        session.spliter._ema_ratio = result.ema_ratio

                elif result.type == ResultType.WARNING:
                    if on_event:
                        await on_event(
                            session.session_id,
                            {
                                "type": "warning",
                                "segment_idx": result.segment_idx,
                                "message": result.warning_msg or "",
                                "text": session.segment_texts.get(result.segment_idx, ""),
                            },
                        )

                elif result.type == ResultType.SESSION_DONE:
                    session.state = SessionState.DONE
                    if on_done:
                        await on_done(session.session_id, result.metrics)
                    break

                elif result.type == ResultType.ERROR:
                    logger.error("Session %s error: %s",
                                 session.session_id, result.error_msg)
                    session.state = SessionState.DONE
                    # Emit structured error lifecycle event
                    LifecycleLogger.emit(
                        session_id=session.session_id,
                        phase="session.error",
                        request_id=session.config.timing.request_id or None,
                        error_type="engine_error",
                        error_message=result.error_msg or "",
                        current_phase=session.state.value,
                        segments_completed=session.segments_done,
                    )
                    error_meta = {
                        "error_phase": session.state.value,
                        "error_type": "engine_error",
                        "error_message": str(result.error_msg or ""),
                        "segments_completed": str(session.segments_done),
                    }
                    if on_done:
                        await on_done(session.session_id, {"error": result.error_msg, **error_meta})
                    break

        except asyncio.CancelledError:
            pass
        finally:
            self._cleanup_session(session.session_id)

    def _cleanup_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        self._consumer_tasks.pop(session_id, None)
        if session:
            # Compute structured summary metrics
            summary: dict[str, Any] = {
                "session_id": session_id,
                "segments_done": session.segments_done,
                "segments_submitted": session.segments_submitted,
                "total_audio_bytes": session.total_audio_bytes,
            }

            # Derived timing metrics
            if session.session_create_to_first_raw_audio_ms is not None:
                summary["session_create_to_first_raw_audio_ms"] = round(
                    session.session_create_to_first_raw_audio_ms, 3
                )
            if session.first_raw_audio_at is not None and session.first_text_enqueued_at is not None:
                summary["first_text_enqueue_to_first_raw_audio_ms"] = round(
                    (session.first_raw_audio_at - session.first_text_enqueued_at) * 1000, 3
                )
            if session.first_raw_audio_at is not None and session.first_text_dequeued_at is not None:
                summary["first_text_dequeue_to_first_raw_audio_ms"] = round(
                    (session.first_raw_audio_at - session.first_text_dequeued_at) * 1000, 3
                )
            if session.prefill_completed_at is not None and session.prefill_started_at is not None:
                summary["engine_prefill_ms"] = round(
                    (session.prefill_completed_at - session.prefill_started_at) * 1000, 3
                )
            if session.first_raw_audio_at is not None and session.first_text_dequeued_at is not None:
                summary["first_raw_to_first_effective_audio_ms"] = 0.0  # will be updated by output pipeline

            # Emit session.completed lifecycle event
            LifecycleLogger.emit(
                session_id=session_id,
                phase="session.completed",
                request_id=session.config.timing.request_id or None,
                turn_id=session.config.timing.turn_id or None,
                # ``summary`` carries its own "session_id" key (used by the
                # human-readable log below); drop it here so it doesn't collide
                # with the explicit session_id= argument.
                **{k: v for k, v in summary.items() if k != "session_id"},
            )

            # Also log a human-readable summary
            latency = session.session_create_to_first_raw_audio_ms
            logger.info(
                "Session %s cleaned up (session_create_to_first_raw_audio_ms=%.1fms, segments=%d/%d, summary=%s)",
                session_id,
                latency or -1,
                session.segments_done,
                session.segments_submitted,
                json.dumps(summary, ensure_ascii=False),
            )

    async def _dispatch_segment_actions(
        self,
        session: Session,
        actions: list,
    ) -> None:
        if not actions:
            return
        self._record_segment_text(actions, session)
        await self._dispatcher.dispatch_segment_actions(session, actions)
        await self._emit_text_token_events(session, actions)
        await self._emit_text_boundary_events(session, actions)

    def _record_segment_text(self, actions: list, session: Session) -> None:
        for sa in actions:
            if sa.token_text and sa.action.type in (ActionType.PREFILL, ActionType.DECODE):
                session.segment_texts[sa.segment_idx] = (
                    session.segment_texts.get(sa.segment_idx, "") + sa.token_text
                )

    @staticmethod
    def _segment_event_meta(sa) -> dict[str, str]:
        return {
            "group_idx": str(sa.group_idx),
            "local_idx": str(sa.local_idx),
            "group_final": "true" if sa.group_final else "false",
        }

    async def _emit_text_token_events(self, session: Session, actions: list) -> None:
        on_event = session.event_callback
        if on_event is None:
            return
        for sa in actions:
            if sa.action.type not in (ActionType.PREFILL, ActionType.DECODE):
                continue
            if not sa.token_text:
                continue
            token_idx = session.segment_token_emitted_count.get(sa.segment_idx, 0)
            session.segment_token_emitted_count[sa.segment_idx] = token_idx + 1
            await on_event(
                session.session_id,
                {
                    "type": "text_token",
                    "segment_idx": sa.segment_idx,
                    "text": sa.token_text,
                    "meta": {
                        **self._segment_event_meta(sa),
                        "token_idx": str(token_idx),
                        "punct_level": str(Spliter.classify_punct_level(sa.token_text)),
                        "text_complete": "false",
                    },
                },
            )

    async def _emit_text_boundary_events(self, session: Session, actions: list) -> None:
        on_event = session.event_callback
        if on_event is None:
            return
        for sa in actions:
            if sa.action.type not in (ActionType.FLUSH_EOS, ActionType.FLUSH_NOP):
                continue
            if sa.segment_idx in session.text_boundary_emitted:
                continue
            session.text_boundary_emitted.add(sa.segment_idx)
            await on_event(
                session.session_id,
                {
                    "type": "text_boundary_commit",
                    "segment_idx": sa.segment_idx,
                    "text": session.segment_texts.get(sa.segment_idx, ""),
                    "meta": {
                        **self._segment_event_meta(sa),
                        "boundary_reason": sa.action.type.value,
                        "text_complete": "true",
                    },
                },
            )

    def _prepare_session_config(self, config: SessionConfig) -> None:
        """Canonicalize session-level prompt text once at session creation."""
        config.instruct_spec = self._tokenize_prompt_text(config.instruct, field_name="instruct")
        config.instruct = config.instruct_spec.text if config.instruct_spec else None
        config.ref_text_spec = self._tokenize_prompt_text(config.ref_text, field_name="ref_text")
        config.ref_text = config.ref_text_spec.text if config.ref_text_spec else None

    def _tokenize_prompt_text(self, text: Optional[str], *, field_name: str) -> Optional[TokenizedText]:
        raw_text = text or ""
        normalized = _normalize_tts_text(raw_text).strip()
        self._log_tokenization_debug(
            field_name,
            raw_text=raw_text,
            normalized_text=normalized,
        )
        if not normalized:
            return None
        return TokenizedText(
            text=normalized,
            token_ids=self._encode_ids(normalized),
        )

    def _tokenize_segment_text(self, text: str) -> list[SegmentToken]:
        self._log_tokenization_debug(
            "segment_text",
            raw_text=text,
            normalized_text=text,
        )
        ids, texts = self._tokenizer.encode_with_text(text, add_special_tokens=False)
        return [
            SegmentToken(
                token_id=token_id,
                text=token_text,
                punct_level=Spliter.classify_punct_level(token_text),
            )
            for token_id, token_text in zip(ids, texts)
        ]

    def _encode_ids(self, text: str) -> list[int]:
        if hasattr(self._tokenizer, "encode_ids"):
            return list(self._tokenizer.encode_ids(text, add_special_tokens=False))
        ids, _ = self._tokenizer.encode_with_text(text, add_special_tokens=False)
        return list(ids)

    def _log_tokenization_debug(
        self,
        field_name: str,
        *,
        raw_text: str,
        normalized_text: str,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return

        payload = {
            "field": field_name,
            "raw_text": raw_text,
            "raw_text_repr": repr(raw_text),
            "normalized_text": normalized_text,
            "normalized_text_repr": repr(normalized_text),
        }
        if normalized_text:
            snapshot_fn = getattr(self._tokenizer, "debug_snapshot", None)
            if callable(snapshot_fn):
                payload["tokenizer"] = snapshot_fn(
                    normalized_text,
                    add_special_tokens=False,
                )
            else:
                ids, pieces = self._tokenizer.encode_with_text(
                    normalized_text,
                    add_special_tokens=False,
                )
                payload["tokenizer"] = {
                    "ids": [int(token_id) for token_id in ids],
                    "pieces": [
                        {
                            "index": idx,
                            "id": int(token_id),
                            "span": token_text,
                            "span_repr": repr(token_text),
                        }
                        for idx, (token_id, token_text) in enumerate(zip(ids, pieces))
                    ],
                }

        logger.debug(
            "Tokenizer observability: %s",
            json.dumps(payload, ensure_ascii=False),
        )
