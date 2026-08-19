"""WebSocket streaming gateway for the standalone TTS engine.

Protocol:
  Client text frames:
    {"type":"start","session_id":"...","config":{...}}
    {"type":"text","text":"...","seq_no":1}
    {"type":"end"}
    {"type":"stop"}  # alias for end
    {"type":"cancel"}
    {"type":"oneshot","session_id":"...","text":"...","config":{...}}
    {"type":"get_capabilities"}

  Server text frames:
    {"type":"event","event":{...}}
    {"type":"capabilities","capabilities":{...}}

  Server binary frames:
    raw PCM audio bytes matching the audio format declared by the ``start`` event.

A connection carries at most one active synthesis session, but it may carry
multiple sessions serially.  A session ``done``/``error`` event ends only that
logical session; the websocket remains available for the next ``start``.

The wire ``session_id`` is a client correlation ID.  Every logical request is
mapped to a fresh server-generated engine session ID, so equal client IDs on
different connections never share engine state.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ssl

from ..core.types import (
    AudioConfig,
    AudioEncoding,
    GroupPolicy,
    InputMode,
    SessionConfig,
)
from ..core.timing import ServerTimingAccumulator
from ..core.lifecycle import LifecycleLogger
from ..interface import (
    StreamingOutputProcessor,
    SessionStartRequest,
    build_done_event,
    build_forward_event,
    build_start_event,
    parse_output_policy,
    parse_timing_context,
    serialize_stream_event,
    to_core_output_policy,
    to_core_timing_context,
)
from ..interface.vad import (
    create_vad_processor,
)
from ..frontend.hold_window import PrefixGateGuardBypass
from .grpc_server import _build_vad_config, _inject_vad_metrics
from .capabilities import (
    REALTIME_MAX_MESSAGE_BYTES,
    REFERENCE_AUDIO_MAX_BYTES,
    RuntimeType,
    build_gateway_capabilities,
)
from .session_identity import GatewaySessionIdentity
from .websocket_resume import (
    ResumeAttachment,
    ResumeDelivery,
    ResumeFailure,
    ResumeFence,
    ResumeProtocolError,
    ResumableSession,
    ResumableSessionRegistry,
    start_fingerprint,
)

if TYPE_CHECKING:
    from ..server import HealthState, TTSEngine

try:
    from aiohttp import WSMsgType, web
except ImportError:  # pragma: no cover - exercised in environments without aiohttp
    WSMsgType = None
    web = None


_WEBSOCKET_AUDIO_QUEUE_MAXSIZE = int(
    os.environ.get("ENGINE_WEBSOCKET_AUDIO_QUEUE_MAXSIZE", "4096") or "4096"
)
_WEBSOCKET_REQUEST_QUEUE_MAXSIZE = int(
    os.environ.get("ENGINE_WEBSOCKET_REQUEST_QUEUE_MAXSIZE", "64") or "64"
)
_CAPABILITIES_PATH = "/v1/capabilities"
_OPENAI_REALTIME_PATH = "/v1/realtime"
_WEBSOCKET_HEARTBEAT_SEC = float(
    os.environ.get("ENGINE_WEBSOCKET_HEARTBEAT_SEC", "30") or "30"
)
_WEBSOCKET_REUSABLE_META_KEY = "websocket_connection_reusable"
_WEBSOCKET_RESUME_GRACE_SEC = float(
    os.environ.get("ENGINE_WEBSOCKET_STREAM_RESUME_GRACE_SEC", "30") or "30"
)
_WEBSOCKET_RESUME_MAX_BUFFER_BYTES = int(
    os.environ.get("ENGINE_WEBSOCKET_STREAM_RESUME_MAX_BUFFER_BYTES", "16777216")
    or "16777216"
)
_SUPPORTED_WEBSOCKET_FEATURES = ["persistent_sessions_v1", "stream_resume_v1"]

logger = logging.getLogger(__name__)


class WebSocketGateway:
    """Bridge one websocket to a serial sequence of TTS engine sessions."""

    def __init__(
        self,
        engine: TTSEngine,
        *,
        stream_resume_grace_seconds: float = _WEBSOCKET_RESUME_GRACE_SEC,
        stream_resume_max_buffer_bytes: int = _WEBSOCKET_RESUME_MAX_BUFFER_BYTES,
    ):
        self._engine = engine
        self._resume_registry = ResumableSessionRegistry(
            engine,
            grace_seconds=stream_resume_grace_seconds,
            max_buffer_bytes=stream_resume_max_buffer_bytes,
        )
        from ..session import SessionService
        from .session_backend import StandaloneSessionBackend

        self._session_service = SessionService(
            StandaloneSessionBackend(self, engine),
        )

    @property
    def session_service(self):
        """Shared typed execution service used by secondary wire adapters."""

        return self._session_service

    def _capabilities(self) -> dict[str, Any]:
        return build_gateway_capabilities(
            self._engine.describe_capabilities(),
            runtime_type=RuntimeType.STANDALONE,
            backend="in-process",
            native_path=_normalize_ws_path("/v1/ws"),
            resume_grace_ms=int(round(self._resume_registry.grace_seconds * 1000.0)),
            resume_max_buffer_bytes=self._resume_registry.max_buffer_bytes,
        )

    async def handle_capabilities(self, request):
        return web.json_response(self._capabilities())

    async def close(self) -> None:
        await self._session_service.close()
        await self._resume_registry.close()

    async def create_session(
        self,
        identity: GatewaySessionIdentity,
        *,
        start_request: SessionStartRequest,
        outbound_queue: asyncio.Queue,
    ) -> None:
        """Create one engine session for a protocol-neutral backend.

        The public method is the migration seam for Realtime and Triton
        adapters.  ``_create_session`` remains as a compatibility alias for
        older integrations and the native v2 handler.
        """

        await self._create_session(
            identity,
            start_request=start_request,
            outbound_queue=outbound_queue,
        )

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse(
            heartbeat=_WEBSOCKET_HEARTBEAT_SEC,
            max_msg_size=REALTIME_MAX_MESSAGE_BYTES,
        )
        await ws.prepare(request)

        client_session_id: str | None = None
        internal_session_id: str | None = None
        # A fresh queue is allocated for every logical session.  Engine
        # callbacks can race with cancellation/completion; keeping their old
        # queue detached prevents a late callback from leaking audio/events
        # into the next session that reuses this websocket.
        outbound_queue: asyncio.Queue | None = None
        request_queue: asyncio.Queue = asyncio.Queue(
            maxsize=_WEBSOCKET_REQUEST_QUEUE_MAXSIZE
        )
        connection_closed = False
        input_closed = False
        start_request: SessionStartRequest | None = None
        resume_session: ResumableSession | None = None
        resume_attachment: ResumeAttachment | None = None
        playback_played_sample = 0
        playback_buffered_sample = 0
        playback_output_sample = 0
        request_task: asyncio.Task | None = None
        outbound_task: asyncio.Task | None = None
        pump_task = asyncio.create_task(self._pump_messages(ws, request_queue))

        async def reset_session() -> None:
            """Release the current logical session without closing ``ws``."""

            nonlocal client_session_id, internal_session_id
            nonlocal start_request, outbound_queue, outbound_task
            nonlocal input_closed, resume_session, resume_attachment
            nonlocal playback_played_sample, playback_buffered_sample
            nonlocal playback_output_sample
            # Usually the queue-get task delivered the terminal itself. It can
            # still be pending when cancel sends its terminal directly.
            if outbound_task is not None and not outbound_task.done():
                outbound_task.cancel()
                try:
                    await outbound_task
                except asyncio.CancelledError:
                    pass
            outbound_task = None
            client_session_id = None
            internal_session_id = None
            input_closed = False
            start_request = None
            outbound_queue = None
            resume_session = None
            resume_attachment = None
            playback_played_sample = 0
            playback_buffered_sample = 0
            playback_output_sample = 0

        async def send_outbound_frame(frame: dict[str, Any]) -> bool:
            """Send one queued frame; return true when it ends the session."""

            nonlocal connection_closed, playback_output_sample
            terminal = _is_terminal_frame(frame)
            event_type = str(frame.get("event", {}).get("type", ""))
            reusable = terminal and event_type == "done" and input_closed
            if reusable:
                # Older gateways closed the physical socket after each terminal
                # event. The explicit marker lets a new SDK distinguish this
                # persistent protocol and safely fall back when it is absent.
                event = frame.setdefault("event", {})
                event.setdefault("meta", {})[_WEBSOCKET_REUSABLE_META_KEY] = "true"
            await _send_frame(ws, frame)
            if frame.get("type") == "audio":
                playback_output_sample = max(
                    playback_output_sample,
                    int(
                        (frame.get("audio") or {})
                        .get("meta", {})
                        .get("output_sample_end", playback_output_sample)
                        or playback_output_sample
                    ),
                )
            if not terminal:
                return False
            await reset_session()
            if not reusable:
                # An engine error (or an unexpected early done while input is
                # still open) can race with already queued client text. Closing
                # prevents those old frames from contaminating a later session.
                connection_closed = True
            return True

        async def send_resume_delivery(delivery: ResumeDelivery) -> bool:
            """Send one replayable delivery without changing its boundaries."""

            nonlocal connection_closed
            session = resume_session
            attachment = resume_attachment
            if session is None or attachment is None:
                raise RuntimeError("resumable delivery has no websocket attachment")
            if not await session.is_current_generation(attachment.generation):
                connection_closed = True
                return False

            frame = delivery.frame
            if frame.get("type") == "audio":
                audio = frame["audio"]
                await ws.send_json(
                    {
                        "type": "audio_header",
                        "delivery_seq": delivery.delivery_seq,
                        "start_sample": delivery.start_sample,
                        "end_sample": delivery.end_sample,
                        "audio": {
                            "sample_rate": audio["sample_rate"],
                            "encoding": audio["encoding"],
                            "channels": audio["channels"],
                            "meta": audio.get("meta") or {},
                        },
                    }
                )
                # The header and binary frame are one indivisible logical
                # delivery. A transport failure between them is recovered by
                # replaying this same sequence number on the next attachment.
                await ws.send_bytes(audio["pcm_data"])
            else:
                event = frame.get("event", {})
                if event.get("type") == "done" and session.input_closed:
                    event.setdefault("meta", {})[_WEBSOCKET_REUSABLE_META_KEY] = "true"
                await ws.send_json(frame)
            return _is_terminal_frame(frame)

        try:
            while True:
                if request_task is None and not connection_closed:
                    request_task = asyncio.create_task(request_queue.get())
                if (
                    outbound_task is None
                    and internal_session_id
                    and outbound_queue is not None
                ):
                    outbound_task = asyncio.create_task(outbound_queue.get())

                wait_set = {
                    task for task in (request_task, outbound_task) if task is not None
                }
                if not wait_set:
                    break

                done, _ = await asyncio.wait(
                    wait_set, return_when=asyncio.FIRST_COMPLETED
                )

                # Flush a ready outbound frame before handling a control frame,
                # preserving callback queue order while the client continues
                # sending text full-duplex.
                if outbound_task in done:
                    frame = outbound_task.result()
                    outbound_task = None
                    if resume_session is not None:
                        if isinstance(frame, ResumeFence):
                            connection_closed = True
                            break
                        if isinstance(frame, ResumeFailure):
                            if not ws.closed:
                                await ws.send_json(
                                    {
                                        "type": "resume_error",
                                        "code": frame.code,
                                        "message": frame.message,
                                    }
                                )
                            connection_closed = True
                            break
                        if not isinstance(frame, ResumeDelivery):
                            raise RuntimeError(
                                "resumable websocket queue contained an invalid frame"
                            )
                        await send_resume_delivery(frame)
                        if connection_closed:
                            break
                        # A resumable terminal remains in the registry until
                        # terminal_ack, so it can be replayed if the physical
                        # connection dies immediately after delivery.
                    else:
                        # Capture the queue: a terminal frame resets the connection's
                        # current queue, but coalescing this batch must stay bound to
                        # the session that produced it.
                        current_queue = outbound_queue
                        if current_queue is None:  # defensive; no session owns frame
                            raise RuntimeError(
                                "websocket outbound frame has no session"
                            )
                        while True:
                            leftover = None
                            if frame.get("type") == "audio":
                                frame, leftover = _coalesce_queued_audio_frames(
                                    frame, current_queue
                                )
                            if await send_outbound_frame(frame):
                                break
                            if leftover is None:
                                break
                            frame = leftover

                        if connection_closed:
                            break

                if request_task in done:
                    kind, payload = request_task.result()
                    request_task = None

                    if kind == "error":
                        raise payload

                    if kind == "closed":
                        connection_closed = True
                    else:
                        message = payload
                        msg_type = str(message.get("type", "") or "").strip().lower()

                        if msg_type == "get_capabilities":
                            await ws.send_json(
                                {
                                    "type": "capabilities",
                                    _WEBSOCKET_REUSABLE_META_KEY: True,
                                    "capabilities": self._capabilities(),
                                }
                            )
                            continue

                        if msg_type == "start":
                            if internal_session_id is not None:
                                raise ValueError(
                                    "websocket session has already been started"
                                )
                            start_request = _start_request_from_ws_message(
                                message,
                                default_mode=InputMode.AUTO,
                            )
                            identity = GatewaySessionIdentity.create(
                                message.get("session_id")
                            )
                            resume_spec = _resume_start_spec(message)
                            if resume_spec is None:
                                outbound_queue = asyncio.Queue(
                                    maxsize=_WEBSOCKET_AUDIO_QUEUE_MAXSIZE
                                )
                                client_session_id = identity.client_session_id
                                internal_session_id = identity.internal_session_id
                                await self._create_session(
                                    identity,
                                    start_request=start_request,
                                    outbound_queue=outbound_queue,
                                )
                            else:
                                token, last_delivery_seq, audio_sample = resume_spec
                                (
                                    session,
                                    created,
                                ) = await self._resume_registry.claim_start(
                                    token=token,
                                    identity=identity,
                                    start_request=start_request,
                                    config_fingerprint=start_fingerprint(message),
                                )
                                if not created:
                                    await session.wait_until_ready()
                                    # The server-owned request and identities are
                                    # authoritative for an idempotent start retry.
                                    start_request = session.start_request
                                attachment = await session.attach(
                                    last_delivery_seq=last_delivery_seq,
                                    audio_through_sample=audio_sample,
                                )
                                resume_session = session
                                resume_attachment = attachment
                                outbound_queue = attachment.queue
                                client_session_id = session.client_session_id
                                internal_session_id = session.internal_session_id
                                input_closed = session.input_closed
                                if created:
                                    try:
                                        await self._create_session(
                                            session.identity,
                                            start_request=start_request,
                                            outbound_queue=session,
                                        )
                                    except BaseException as exc:
                                        await self._resume_registry.fail_initialization(
                                            session, exc
                                        )
                                        raise
                                    else:
                                        await session.mark_initialized()
                                else:
                                    await ws.send_json(await session.resume_info())

                        elif msg_type == "resume":
                            if internal_session_id is not None:
                                raise ResumeProtocolError(
                                    "websocket_session_active",
                                    "websocket already has an active logical session",
                                )
                            token, last_delivery_seq, audio_sample = (
                                _resume_request_spec(message)
                            )
                            session = await self._resume_registry.find(token)
                            attachment = await session.attach(
                                last_delivery_seq=last_delivery_seq,
                                audio_through_sample=audio_sample,
                            )
                            resume_session = session
                            resume_attachment = attachment
                            outbound_queue = attachment.queue
                            start_request = session.start_request
                            client_session_id = session.client_session_id
                            internal_session_id = session.internal_session_id
                            input_closed = session.input_closed
                            await ws.send_json(await session.resume_info())

                        elif msg_type == "oneshot":
                            if internal_session_id is not None:
                                raise ValueError(
                                    "websocket session has already been started"
                                )
                            start_request = _start_request_from_ws_message(
                                message,
                                default_mode=InputMode.FULL_TEXT,
                            )
                            start_request.config.input_mode = InputMode.FULL_TEXT
                            if start_request.config.group_policy == GroupPolicy.NONE:
                                start_request.config.group_policy = GroupPolicy.AUTO
                            outbound_queue = asyncio.Queue(
                                maxsize=_WEBSOCKET_AUDIO_QUEUE_MAXSIZE
                            )
                            identity = GatewaySessionIdentity.create(
                                message.get("session_id")
                            )
                            client_session_id = identity.client_session_id
                            internal_session_id = identity.internal_session_id
                            await self._create_session(
                                identity,
                                start_request=start_request,
                                outbound_queue=outbound_queue,
                            )
                            input_closed = True
                            text = str(message.get("text", "") or "")
                            if not text:
                                raise ValueError(
                                    "oneshot request requires non-empty 'text'"
                                )
                            start_request.initial_text = text
                            await self._engine.push_text_input(
                                internal_session_id, text
                            )
                            await self._engine.mark_input_complete(internal_session_id)

                        elif msg_type == "text":
                            if not internal_session_id:
                                raise ValueError("received 'text' before 'start'")
                            if start_request is not None:
                                client_ts_ms = _coerce_ws_int(
                                    message.get("client_timestamp_ms"), 0
                                )
                                if client_ts_ms > 0:
                                    start_request.timing.client_text_ts_ms = (
                                        client_ts_ms
                                    )
                            text = str(message.get("text", "") or "")
                            if resume_session is not None:
                                if resume_attachment is None:
                                    raise RuntimeError(
                                        "resumable session has no attachment"
                                    )
                                seq_no = _required_nonnegative_ws_int(
                                    message, "seq_no", positive=True
                                )
                                acked_seq, duplicate = await resume_session.accept_text(
                                    resume_attachment.generation,
                                    seq_no=seq_no,
                                    text=text,
                                    push=lambda: self._engine.push_text_input(
                                        internal_session_id, text
                                    ),
                                )
                                if await resume_session.is_current_generation(
                                    resume_attachment.generation
                                ):
                                    await ws.send_json(
                                        {
                                            "type": "text_ack",
                                            "through_seq": acked_seq,
                                            "duplicate": duplicate,
                                        }
                                    )
                            else:
                                await self._engine.push_text_input(
                                    internal_session_id,
                                    text,
                                )

                        elif msg_type in {"end", "stop"}:
                            if not internal_session_id:
                                raise ValueError(
                                    f"received '{msg_type}' before 'start'"
                                )
                            if start_request is not None:
                                client_ts_ms = _coerce_ws_int(
                                    message.get("client_timestamp_ms"), 0
                                )
                                if client_ts_ms > 0:
                                    start_request.timing.client_end_ts_ms = client_ts_ms
                            if resume_session is not None:
                                if resume_attachment is None:
                                    raise RuntimeError(
                                        "resumable session has no attachment"
                                    )
                                raw_final_seq = message.get("final_seq_no")
                                final_seq_no = (
                                    resume_session.acked_text_seq
                                    if raw_final_seq is None
                                    else _required_nonnegative_ws_int(
                                        message, "final_seq_no", positive=False
                                    )
                                )
                                (
                                    accepted_final,
                                    duplicate,
                                ) = await resume_session.close_input(
                                    resume_attachment.generation,
                                    final_seq_no=final_seq_no,
                                    close=lambda: self._engine.mark_input_complete(
                                        internal_session_id
                                    ),
                                )
                                input_closed = resume_session.input_closed
                                if await resume_session.is_current_generation(
                                    resume_attachment.generation
                                ):
                                    await ws.send_json(
                                        {
                                            "type": "input_ack",
                                            "final_seq_no": accepted_final,
                                            "acked_text_seq": resume_session.acked_text_seq,
                                            "duplicate": duplicate,
                                        }
                                    )
                            else:
                                input_closed = True
                                await self._engine.mark_input_complete(
                                    internal_session_id
                                )

                        elif msg_type == "ack":
                            if resume_session is None or resume_attachment is None:
                                raise ResumeProtocolError(
                                    "resume_not_enabled",
                                    "delivery ACK is only valid for a resumable stream",
                                )
                            await resume_session.acknowledge(
                                resume_attachment.generation,
                                through_delivery_seq=_required_nonnegative_ws_int(
                                    message,
                                    "through_delivery_seq",
                                    positive=False,
                                ),
                                audio_through_sample=_required_nonnegative_ws_int(
                                    message,
                                    "audio_through_sample",
                                    positive=False,
                                ),
                            )

                        elif msg_type == "terminal_ack":
                            if resume_session is None or resume_attachment is None:
                                raise ResumeProtocolError(
                                    "resume_not_enabled",
                                    "terminal_ack is only valid for a resumable stream",
                                )
                            await self._resume_registry.terminal_ack(
                                resume_session,
                                resume_attachment.generation,
                                through_delivery_seq=_required_nonnegative_ws_int(
                                    message,
                                    "through_delivery_seq",
                                    positive=False,
                                ),
                                audio_through_sample=_required_nonnegative_ws_int(
                                    message,
                                    "audio_through_sample",
                                    positive=False,
                                ),
                            )
                            await reset_session()

                        elif msg_type == "playback_progress":
                            try:
                                played = _required_nonnegative_ws_int(
                                    message, "played_through_sample", positive=False
                                )
                                buffered = _required_nonnegative_ws_int(
                                    message, "buffered_through_sample", positive=False
                                )
                                if resume_session is not None:
                                    if resume_attachment is None:
                                        raise ResumeProtocolError(
                                            "resume_attachment_missing",
                                            "resumable session has no attachment",
                                        )
                                    observed_delivery_seq = (
                                        _required_nonnegative_ws_int(
                                            message,
                                            "observed_delivery_seq",
                                            positive=False,
                                        )
                                        if "observed_delivery_seq" in message
                                        else None
                                    )
                                    await resume_session.record_playback_progress(
                                        resume_attachment.generation,
                                        played_through_sample=played,
                                        buffered_through_sample=buffered,
                                        observed_delivery_seq=observed_delivery_seq,
                                    )
                                else:
                                    if buffered < played:
                                        raise ResumeProtocolError(
                                            "invalid_playback_progress",
                                            "buffered_through_sample must be >= played_through_sample",
                                        )
                                    # Playback feedback may be duplicated by a
                                    # reconnecting client.  Ignore only a
                                    # wholly stale pair; a partial rollback is
                                    # still a protocol error.
                                    if (
                                        played <= playback_played_sample
                                        and buffered <= playback_buffered_sample
                                    ):
                                        continue
                                    if played < playback_played_sample:
                                        raise ResumeProtocolError(
                                            "invalid_playback_progress",
                                            "played_through_sample cannot move backwards",
                                        )
                                    if buffered < playback_buffered_sample:
                                        raise ResumeProtocolError(
                                            "invalid_playback_progress",
                                            "buffered_through_sample cannot move backwards",
                                        )
                                    if buffered > playback_output_sample:
                                        raise ResumeProtocolError(
                                            "invalid_playback_progress",
                                            "buffered_through_sample is ahead of server output",
                                        )
                                    playback_played_sample = played
                                    playback_buffered_sample = buffered
                            except ResumeProtocolError as exc:
                                await ws.send_json(
                                    {
                                        "type": "playback_progress_error",
                                        "code": exc.code,
                                        "message": str(exc),
                                    }
                                )

                        elif msg_type == "cancel":
                            if internal_session_id:
                                cancelled_internal_session_id = internal_session_id
                                cancelled_client_session_id = client_session_id or ""
                                cancel_reason = str(message.get("reason", "") or "")

                                if resume_session is not None:
                                    await resume_session.publish_cancelled(
                                        cancel_reason
                                    )
                                    input_closed = True
                                    continue

                                # Stop a parked queue read before detaching the
                                # per-session queue.  Frames already sent before
                                # this control message remain valid; queued frames
                                # are intentionally discarded by abandoning the
                                # queue after the engine is cancelled.
                                if (
                                    outbound_task is not None
                                    and not outbound_task.done()
                                ):
                                    outbound_task.cancel()
                                    try:
                                        await outbound_task
                                    except asyncio.CancelledError:
                                        pass
                                    outbound_task = None

                                await self._engine.cancel(cancelled_internal_session_id)
                                await ws.send_json(
                                    _make_event_frame(
                                        event_type="done",
                                        session_id=cancelled_client_session_id,
                                        message=cancel_reason,
                                        meta={
                                            "terminal_reason": "cancelled",
                                            "cancel_reason": cancel_reason,
                                            _WEBSOCKET_REUSABLE_META_KEY: "true",
                                        },
                                    )
                                )
                                # Reset detaches this session's queue. It remains
                                # reachable only from old callbacks; the next
                                # session cannot observe their late frames.
                                await reset_session()

                        else:
                            raise ValueError(
                                f"unsupported websocket message type: '{msg_type or '<empty>'}'"
                            )

                if connection_closed:
                    break

        except asyncio.CancelledError:
            logger.info(
                "WebSocket stream cancelled: client=%s internal=%s",
                client_session_id,
                internal_session_id,
            )
        except ResumeProtocolError as exc:
            logger.warning(
                "Resumable WebSocket protocol error: client=%s internal=%s code=%s",
                client_session_id,
                internal_session_id,
                exc.code,
            )
            if not ws.closed:
                await ws.send_json(
                    {
                        "type": "resume_error",
                        "code": exc.code,
                        "message": str(exc),
                    }
                )
        except Exception as exc:
            logger.error(
                "WebSocket stream error: client=%s internal=%s: %s",
                client_session_id,
                internal_session_id,
                exc,
            )
            if not ws.closed:
                await ws.send_json(
                    _make_event_frame(
                        event_type="error",
                        session_id=client_session_id or "",
                        message=str(exc),
                    )
                )
        finally:
            for task in (request_task, outbound_task, pump_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (request_task, outbound_task, pump_task):
                if task is not None:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            if resume_session is not None and resume_attachment is not None:
                await resume_session.detach(resume_attachment.generation)
            elif internal_session_id:
                await self._engine.cancel(internal_session_id)
            if not ws.closed:
                await ws.close()

        return ws

    async def _create_session(
        self,
        identity: GatewaySessionIdentity,
        *,
        start_request: SessionStartRequest,
        outbound_queue: asyncio.Queue,
    ) -> None:
        client_session_id = identity.client_session_id
        internal_session_id = identity.internal_session_id
        start_request.session_id = client_session_id
        config = start_request.config
        identity.bind_engine_config(config)

        # Create server timing accumulator for cross-thread observability
        import time as _time

        timing_acc = ServerTimingAccumulator()
        timing_acc.request_received_epoch_ms = int(round(_time.time() * 1000.0))
        timing_acc.session_created_epoch_ms = timing_acc.request_received_epoch_ms
        timing_acc.vad_policy = config.output_policy.vad.strategy or "disabled"
        timing_acc.text_input_mode = config.input_mode.value

        LifecycleLogger.emit(
            session_id=internal_session_id,
            phase="request.accepted",
            request_id=config.timing.request_id or None,
            turn_id=config.timing.turn_id or None,
            transport="websocket",
            client_session_id=client_session_id,
            client_request_ts_ms=config.timing.client_request_ts_ms or None,
        )

        # Store accumulator reference in timing extra for engine thread access
        config.timing.extra["_server_timing_accumulator"] = timing_acc

        # Create per-session VAD processor from config
        from .grpc_server import ENGINE_SAMPLE_RATE

        vad_config = _build_vad_config(config)
        vad_processor = create_vad_processor(vad_config, sample_rate=ENGINE_SAMPLE_RATE)
        output_processor = StreamingOutputProcessor(
            start_request,
            vad_processor=vad_processor,
            native_sample_rate=ENGINE_SAMPLE_RATE,
            timing_accumulator=timing_acc,
        )
        pipeline = output_processor.pipeline

        async def _enqueue_frames(frames: list[dict[str, Any]]) -> None:
            if not frames:
                return
            if isinstance(outbound_queue, ResumableSession):
                await outbound_queue.put_batch(frames)
                return
            for frame in frames:
                await outbound_queue.put(frame)

        vad_enabled = vad_processor.config.enabled
        first_effective_logged = False
        prefix_gate_guard_bypass = (
            PrefixGateGuardBypass(vad_processor.discard_pending)
            if vad_enabled
            else None
        )
        if prefix_gate_guard_bypass is not None:
            # Internal shared state: the frontend bypasses playhead pacing only
            # while this gateway-owned VAD is still consuming the prefix.
            config.timing.extra["_prefix_gate_guard_bypass"] = prefix_gate_guard_bypass

        def snapshot_prefix_bypass() -> tuple[int, float]:
            if prefix_gate_guard_bypass is None:
                return 0, 0.0
            bypass_chunks = prefix_gate_guard_bypass.bypassed_chunks
            bypass_audio_ms = (
                prefix_gate_guard_bypass.bypassed_audio_bytes
                / (ENGINE_SAMPLE_RATE * 4)
                * 1000.0
            )
            timing_acc.guarded_delivery_prefix_bypass_chunks = bypass_chunks
            timing_acc.guarded_delivery_prefix_bypass_audio_ms = bypass_audio_ms
            return bypass_chunks, bypass_audio_ms

        def log_first_effective_audio(audio_bytes: int) -> None:
            """Emit the first audible-output boundary with VAD context once."""
            nonlocal first_effective_logged
            if first_effective_logged:
                return
            first_effective_logged = True
            if prefix_gate_guard_bypass is not None:
                prefix_gate_guard_bypass.mark_first_effective(audio_bytes)
            bypass_chunks, bypass_audio_ms = snapshot_prefix_bypass()
            vad_metrics = vad_processor.metrics
            prefix_trimmed_ms = (
                vad_metrics.prefix_trimmed_samples / ENGINE_SAMPLE_RATE * 1000.0
            )
            gating_ms = None
            if (
                timing_acc.first_raw_audio_monotonic is not None
                and timing_acc.first_effective_audio_monotonic is not None
            ):
                gating_ms = (
                    timing_acc.first_effective_audio_monotonic
                    - timing_acc.first_raw_audio_monotonic
                ) * 1000.0
            LifecycleLogger.emit(
                session_id=internal_session_id,
                phase="output.audio.first_effective",
                request_id=config.timing.request_id or None,
                client_session_id=client_session_id,
                session_level=config.observability_level,
                vad_policy=vad_processor.config.mode.value,
                prefix_trim_applied=prefix_trimmed_ms > 0.0,
                prefix_trimmed_ms=round(prefix_trimmed_ms, 3),
                first_raw_to_first_effective_audio_ms=(
                    round(gating_ms, 3) if gating_ms is not None else None
                ),
                guarded_delivery_prefix_bypass_chunks=bypass_chunks,
                guarded_delivery_prefix_bypass_audio_ms=round(bypass_audio_ms, 3),
            )

        async def on_audio(sid: str, data: bytes) -> None:
            batch = output_processor.process(data)
            frames: list[dict[str, Any]] = []
            if batch.audio is not None and batch.audio.pcm_bytes:
                log_first_effective_audio(len(batch.audio.pcm_bytes))
                frames.append(
                    _make_audio_frame(
                        batch.audio.pcm_bytes,
                        batch.audio.audio,
                        meta=batch.audio.meta,
                    )
                )
            for event in (*batch.anchors, *batch.events):
                frames.append(
                    _make_event_frame_from_contract(
                        build_forward_event(client_session_id, event, start_request)
                    )
                )
            await _enqueue_frames(frames)

        async def on_event(sid: str, event: dict) -> None:
            batch = output_processor.process_event(event)
            frames = []
            for item in (*batch.anchors, *batch.events):
                frames.append(
                    _make_event_frame_from_contract(
                        build_forward_event(client_session_id, item, start_request)
                    )
                )
            await _enqueue_frames(frames)

        async def on_done(sid: str, metrics: dict) -> None:
            for batch in output_processor.finish(
                emit_final=not bool(metrics.get("error") or metrics.get("cancelled"))
            ):
                frames = []
                if batch.audio is not None and batch.audio.pcm_bytes:
                    log_first_effective_audio(len(batch.audio.pcm_bytes))
                    frames.append(
                        _make_audio_frame(
                            batch.audio.pcm_bytes,
                            batch.audio.audio,
                            meta=batch.audio.meta,
                        )
                    )
                for event in (*batch.anchors, *batch.events):
                    frames.append(
                        _make_event_frame_from_contract(
                            build_forward_event(client_session_id, event, start_request)
                        )
                    )
                await _enqueue_frames(frames)

            # All-silence sessions never cross the first-effective boundary,
            # so snapshot the bypass counters unconditionally at completion.
            snapshot_prefix_bypass()
            # Inject VAD observability into metrics
            _inject_vad_metrics(vad_processor, pipeline, metrics)
            await _enqueue_frames(
                [
                    _make_event_frame_from_contract(
                        build_done_event(client_session_id, metrics, pipeline)
                    )
                ]
            )

        await self._engine.start_session(
            internal_session_id,
            config=config,
            on_audio=on_audio,
            on_done=on_done,
            on_event=on_event,
        )
        await outbound_queue.put(
            _make_event_frame_from_contract(
                build_start_event(client_session_id, start_request)
            )
        )
        logger.info(
            "WebSocket session started: client=%s internal=%s",
            client_session_id,
            internal_session_id,
        )

    async def _pump_messages(
        self,
        ws,
        request_queue: asyncio.Queue,
    ) -> None:
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid websocket JSON payload: {exc}"
                        ) from exc
                    if not isinstance(payload, dict):
                        raise ValueError("websocket payload must be a JSON object")
                    await request_queue.put(("request", payload))
                    continue
                if msg.type == WSMsgType.BINARY:
                    raise ValueError(
                        "binary client frames are not supported; send JSON control messages only"
                    )
                if msg.type == WSMsgType.ERROR:
                    raise msg.data
        except Exception as exc:
            await request_queue.put(("error", exc))
        finally:
            await request_queue.put(("closed", None))


def _coalesce_queued_audio_frames(
    frame: dict[str, Any],
    outbound_queue: asyncio.Queue,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """WebSocket twin of grpc_server._coalesce_queued_audio.

    Merges backlogged audio frames (zero added latency: only what is
    already queued), preserving the head's meta and stopping at any frame
    carrying non-standard meta, a non-audio frame, or the byte cap.
    Returns (possibly-merged frame, leftover frame or None); the leftover
    MUST be processed by the caller before waiting on the queue again.
    """
    from .grpc_server import _COALESCE_MAX_BYTES, _MERGEABLE_META_KEYS

    if frame.get("type") != "audio":
        return frame, None
    head = frame["audio"]
    parts: list[bytes] | None = None
    leftover: dict[str, Any] | None = None
    total_bytes = len(head["pcm_data"])
    merged_sample_end = int((head.get("meta") or {}).get("output_sample_end", "0") or 0)
    while total_bytes < _COALESCE_MAX_BYTES and not outbound_queue.empty():
        nxt = outbound_queue.get_nowait()
        audio = nxt.get("audio") if nxt.get("type") == "audio" else None
        if (
            audio is not None
            and set(audio.get("meta") or ()) <= _MERGEABLE_META_KEYS
            and audio["sample_rate"] == head["sample_rate"]
            and audio["encoding"] == head["encoding"]
            and audio["channels"] == head["channels"]
        ):
            if parts is None:
                parts = [head["pcm_data"]]
            parts.append(audio["pcm_data"])
            total_bytes += len(audio["pcm_data"])
            merged_sample_end = max(
                merged_sample_end,
                int((audio.get("meta") or {}).get("output_sample_end", "0") or 0),
            )
            continue
        leftover = nxt
        break
    if parts is None:
        return frame, leftover
    merged_audio = dict(head)
    merged_audio["pcm_data"] = b"".join(parts)
    merged_audio["meta"] = {
        **dict(head.get("meta") or {}),
        "output_sample_end": str(merged_sample_end),
    }
    merged = dict(frame)
    merged["audio"] = merged_audio
    return merged, leftover


def _make_audio_frame(
    pcm_bytes: bytes,
    audio_config: AudioConfig,
    *,
    meta: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "audio",
        "audio": {
            "pcm_data": pcm_bytes,
            "sample_rate": audio_config.sample_rate,
            "encoding": audio_config.encoding.value,
            "channels": audio_config.channels,
            "meta": meta or {},
        },
    }


def _make_event_frame(
    *,
    event_type: str,
    session_id: str = "",
    segment_id: int = -1,
    text: str = "",
    message: str = "",
    audio_format: AudioConfig | None = None,
    meta: dict[str, str] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": event_type,
        "session_id": session_id,
        "segment_id": segment_id,
        "text": text,
        "message": message,
        "meta": meta or {},
    }
    if audio_format is not None:
        event["audio"] = {
            "encoding": audio_format.encoding.value,
            "sample_rate": audio_format.sample_rate,
            "channels": audio_format.channels,
        }
    return {
        "type": "event",
        "event": event,
    }


def _make_event_frame_from_contract(event) -> dict[str, Any]:
    payload = serialize_stream_event(event)
    audio = payload.get("audio")
    return _make_event_frame(
        event_type=str(payload.get("type", "") or ""),
        session_id=str(payload.get("session_id", "") or ""),
        segment_id=int(payload.get("segment_id", -1)),
        text=str(payload.get("text", "") or ""),
        message=str(payload.get("message", "") or ""),
        audio_format=(
            AudioConfig(
                encoding=_audio_encoding_from_ws_value(
                    str(audio.get("encoding", "pcm_f32"))
                ),
                sample_rate=int(audio.get("sample_rate", 24000)),
                channels=int(audio.get("channels", 1)),
            )
            if isinstance(audio, dict)
            else None
        ),
        meta={str(k): str(v) for k, v in (payload.get("meta", {}) or {}).items()},
    )


async def _send_frame(ws, frame: dict[str, Any]) -> None:
    if frame.get("type") == "audio":
        await ws.send_bytes(frame["audio"]["pcm_data"])
        return
    await ws.send_json(frame)


def _is_terminal_frame(frame: dict[str, Any]) -> bool:
    if frame.get("type") != "event":
        return False
    return frame.get("event", {}).get("type") in {"done", "error"}


def add_health_routes(app, health_state: HealthState) -> None:
    """Expose the engine health probes on the gateway port as well.

    Same routes and semantics as the dedicated health port (see
    ``engine.server.HealthState``), for platforms that can only probe the
    service port. Two caveats versus the health port: this surface binds
    only after the model load (probes see connection-refused during the
    load window), and it answers from the main event loop, so it also
    exercises the actual serving path.
    """

    async def handle_probe(request):
        stats, code = health_state.payload_and_status(request.path)
        return web.json_response(stats, status=code)

    for route in health_state.ROUTES:
        app.router.add_get(route, handle_probe)


async def serve(
    engine: TTSEngine,
    port: int,
    *,
    stop_event: asyncio.Event,
    path: str = "/v1/ws",
    started: asyncio.Event | None = None,
    health_state: HealthState | None = None,
    realtime_usage_recorder: Any = None,
    ssl_context: ssl.SSLContext | None = None,
) -> None:
    """Start the websocket gateway using aiohttp.

    ``started`` is set once the port is bound so readiness can cover
    "gateway actually listening". ``health_state`` additionally mounts the
    health probe routes on this port.
    """
    if web is None:
        logger.error("aiohttp not installed. Run: pip install aiohttp")
        return

    ws_path = _normalize_ws_path(path)
    gateway = WebSocketGateway(engine)
    from ..session import ResumableSessionRegistry
    from .native_session_gateway import NativeSessionGateway
    from .openai_realtime import OpenAIRealtimeGateway

    # Both public protocols share the same transport-neutral execution
    # contract. Native delivery reliability is owned by the typed projector,
    # not by the standalone engine adapter.
    resume_registry = ResumableSessionRegistry(gateway.session_service)
    native_gateway = NativeSessionGateway(
        gateway.session_service,
        capabilities=gateway._capabilities,
        resume_registry=resume_registry,
    )
    realtime_gateway = OpenAIRealtimeGateway(
        engine,
        session_service=gateway.session_service,
        resume_registry=resume_registry,
        usage_recorder=realtime_usage_recorder,
    )
    app = web.Application()
    app.router.add_get(_CAPABILITIES_PATH, gateway.handle_capabilities)
    app.router.add_get(_OPENAI_REALTIME_PATH, realtime_gateway.handle_websocket)
    app.router.add_get(ws_path, native_gateway.handle_websocket)
    from ..distribution.sdk import mount_sdk_routes
    from ..distribution.site import mount_demo_config_route

    sdk_distribution = mount_sdk_routes(app)
    mount_demo_config_route(
        app,
        runtime_type=RuntimeType.STANDALONE.value,
        capabilities_provider=gateway._capabilities,
        sdk_distribution=sdk_distribution,
    )
    if health_state is not None:
        add_health_routes(app, health_state)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(
        runner,
        "0.0.0.0",
        port,
        ssl_context=ssl_context,
    )
    try:
        await site.start()
        if started is not None:
            started.set()
        logger.info(
            "%s gateway listening on port %d (legacy %s, realtime %s, capabilities %s)",
            "HTTPS/WSS" if ssl_context is not None else "HTTP/WS",
            port,
            ws_path,
            _OPENAI_REALTIME_PATH,
            _CAPABILITIES_PATH,
        )
        await stop_event.wait()
    finally:
        await runner.cleanup()
        await native_gateway.close()
        await gateway.close()


def _session_config_from_ws_message(
    message: dict[str, Any],
    *,
    default_mode: InputMode,
) -> SessionConfig:
    return _start_request_from_ws_message(message, default_mode=default_mode).config


def parse_session_start_request(
    message: dict[str, Any],
    *,
    default_mode: InputMode,
) -> SessionStartRequest:
    raw = _extract_ws_config_payload(message)
    output_policy = parse_output_policy(_extract_ws_output_policy(raw))
    timing = parse_timing_context(raw.get("timing") or raw.get("timing_context"))
    protocol_version = str(raw.get("protocol_version") or "").strip()
    if protocol_version and "client_protocol_version" not in timing.extra:
        timing.extra["client_protocol_version"] = protocol_version
    cfg = SessionConfig(
        task_type=str(raw.get("task_type", "") or ""),
        language=str(raw.get("language", "auto") or "auto"),
        speaker=_optional_str(raw.get("speaker")),
        instruct=_optional_str(raw.get("instruct")),
        ref_audio=_decode_optional_base64(raw.get("ref_audio")),
        ref_text=_optional_str(raw.get("ref_text")),
        x_vector_only=_coerce_ws_bool(raw.get("x_vector_only", False)),
        input_mode=_input_mode_from_ws_value(
            raw.get("input_mode"), default_mode=default_mode
        ),
        group_policy=_group_policy_from_ws_value(raw.get("group_policy")),
        audio=_audio_config_from_ws_value(raw.get("audio")),
        output_policy=to_core_output_policy(output_policy),
        timing=to_core_timing_context(timing),
    )
    _validate_audio_config(cfg.audio)
    return SessionStartRequest(
        session_id=str(message.get("session_id", "") or ""),
        config=cfg,
        output_policy=output_policy,
        timing=timing,
        initial_text=str(message.get("text", "") or ""),
    )


# Compatibility alias for integrations that imported the old private helper.
# New adapters must use the public parser above.
_start_request_from_ws_message = parse_session_start_request


def _extract_ws_output_policy(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("output_policy")
    policy = dict(value) if isinstance(value, dict) else {}
    if "vad_policy" not in policy:
        if isinstance(raw.get("vad_policy"), dict):
            policy["vad_policy"] = raw["vad_policy"]
        elif isinstance(raw.get("vad"), dict):
            policy["vad_policy"] = raw["vad"]
    return policy


def _extract_ws_config_payload(message: dict[str, Any]) -> dict[str, Any]:
    cfg = message.get("config")
    if cfg is None:
        return {
            key: value
            for key, value in message.items()
            if key not in {"type", "text", "seq_no", "session_id"}
        }
    if not isinstance(cfg, dict):
        raise ValueError("websocket 'config' must be an object")
    merged = dict(cfg)
    for field in (
        "task_type",
        "language",
        "speaker",
        "instruct",
        "ref_audio",
        "ref_text",
        "x_vector_only",
        "input_mode",
        "group_policy",
        "audio",
        "output_policy",
        "vad_policy",
        "timing",
        "timing_context",
        "protocol_version",
    ):
        if field not in merged and field in message:
            merged[field] = message[field]
    return merged


def _resume_start_spec(message: dict[str, Any]) -> tuple[str, int, int] | None:
    raw = message.get("resume")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ResumeProtocolError(
            "invalid_resume_request", "start.resume must be an object"
        )
    if not _coerce_ws_bool(raw.get("enabled", False)):
        return None
    token = _validated_resume_token(raw.get("token"))
    return (
        token,
        _mapping_nonnegative_int(raw, "last_delivery_seq", default=0),
        _mapping_nonnegative_int(raw, "audio_through_sample", default=0),
    )


def _resume_request_spec(message: dict[str, Any]) -> tuple[str, int, int]:
    token = _validated_resume_token(message.get("token"))
    return (
        token,
        _required_nonnegative_ws_int(message, "last_delivery_seq", positive=False),
        _required_nonnegative_ws_int(message, "audio_through_sample", positive=False),
    )


def _validated_resume_token(value: Any) -> str:
    token = str(value or "").strip()
    # UUID4 hex (the SDK spelling) and canonical UUID strings both carry a
    # full 128-bit identifier while keeping registry keys small and bounded.
    # Never include this capability secret in the error text or logs.
    if not token or len(token) > 36:
        raise ResumeProtocolError(
            "invalid_resume_token", "resume token must be a 128-bit UUID"
        )
    try:
        parsed = uuid.UUID(token)
    except (ValueError, AttributeError) as exc:
        raise ResumeProtocolError(
            "invalid_resume_token", "resume token must be a 128-bit UUID"
        ) from exc
    if parsed.int == 0:
        raise ResumeProtocolError(
            "invalid_resume_token", "resume token must be a 128-bit UUID"
        )
    return parsed.hex


def _mapping_nonnegative_int(
    mapping: dict[str, Any], field: str, *, default: int
) -> int:
    if field not in mapping:
        return default
    try:
        value = int(mapping[field])
    except (TypeError, ValueError) as exc:
        raise ResumeProtocolError(
            "invalid_resume_cursor", f"{field} must be an integer"
        ) from exc
    if value < 0:
        raise ResumeProtocolError(
            "invalid_resume_cursor", f"{field} must be non-negative"
        )
    return value


def _required_nonnegative_ws_int(
    message: dict[str, Any], field: str, *, positive: bool
) -> int:
    if field not in message:
        raise ResumeProtocolError(
            "missing_protocol_field", f"resumable message requires '{field}'"
        )
    value = _mapping_nonnegative_int(message, field, default=0)
    if positive and value <= 0:
        raise ResumeProtocolError(
            "invalid_text_sequence", f"{field} must be greater than zero"
        )
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _decode_optional_base64(value: Any) -> bytes | None:
    if value in (None, ""):
        return None
    if isinstance(value, bytes):
        decoded = value
    else:
        try:
            decoded = base64.b64decode(str(value), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("websocket 'ref_audio' must be valid base64") from exc
    if len(decoded) > REFERENCE_AUDIO_MAX_BYTES:
        raise ValueError(
            "websocket 'ref_audio' exceeds the advertised "
            f"{REFERENCE_AUDIO_MAX_BYTES}-byte limit"
        )
    return decoded


def _coerce_ws_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, "", 0):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no"}:
            return False
        if normalized in {"true", "1", "yes"}:
            return True
    return bool(value)


def _coerce_ws_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _input_mode_from_ws_value(value: Any, *, default_mode: InputMode) -> InputMode:
    if value in (None, "", 0):
        return default_mode
    if isinstance(value, int):
        mapping = {
            1: InputMode.TOKEN,
            2: InputMode.CLAUSE,
            3: InputMode.LONG_SEGMENT,
            4: InputMode.FULL_TEXT,
            5: InputMode.AUTO,
        }
        if value in mapping:
            return mapping[value]
    normalized = str(value).strip().lower()
    mapping = {
        "auto": InputMode.AUTO,
        "token": InputMode.TOKEN,
        "clause": InputMode.CLAUSE,
        "long_segment": InputMode.LONG_SEGMENT,
        "full_text": InputMode.FULL_TEXT,
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"unsupported input_mode: {value!r}")


def _group_policy_from_ws_value(value: Any) -> GroupPolicy:
    if value in (None, "", 0):
        return GroupPolicy.AUTO
    if isinstance(value, int):
        mapping = {
            1: GroupPolicy.NONE,
            2: GroupPolicy.AUTO,
        }
        if value in mapping:
            return mapping[value]
    normalized = str(value).strip().lower()
    mapping = {
        "none": GroupPolicy.NONE,
        "auto": GroupPolicy.AUTO,
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"unsupported group_policy: {value!r}")


def _audio_config_from_ws_value(value: Any) -> AudioConfig:
    if value is None:
        return AudioConfig()
    if not isinstance(value, dict):
        raise ValueError("websocket 'audio' must be an object")
    encoding = _audio_encoding_from_ws_value(value.get("encoding"))
    sample_rate = int(value.get("sample_rate", 24000) or 24000)
    channels = int(value.get("channels", 1) or 1)
    return AudioConfig(
        sample_rate=sample_rate,
        encoding=encoding,
        channels=channels,
    )


def _audio_encoding_from_ws_value(value: Any) -> AudioEncoding:
    if value in (None, "", 0, 1):
        return AudioEncoding.PCM_F32
    if value == 2:
        return AudioEncoding.PCM_S16LE
    normalized = str(value).strip().lower()
    if normalized == "pcm_f32":
        return AudioEncoding.PCM_F32
    if normalized == "pcm_s16le":
        return AudioEncoding.PCM_S16LE
    raise ValueError(f"unsupported audio encoding: {value!r}")


def _validate_audio_config(audio: AudioConfig) -> None:
    if audio.channels != 1:
        raise ValueError(f"Unsupported channel count: {audio.channels} (mono only)")
    if audio.sample_rate not in (16000, 24000):
        raise ValueError(
            f"Unsupported sample_rate: {audio.sample_rate} (expected 16000 or 24000)"
        )
    if audio.encoding not in (AudioEncoding.PCM_F32, AudioEncoding.PCM_S16LE):
        raise ValueError(f"Unsupported audio encoding: {audio.encoding}")


def _normalize_ws_path(path: str) -> str:
    normalized = str(path or "/v1/ws").strip()
    if not normalized:
        normalized = "/v1/ws"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized
