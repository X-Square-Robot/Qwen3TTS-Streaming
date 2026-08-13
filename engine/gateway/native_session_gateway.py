"""Native binary WebSocket projection over the canonical session service.

The standalone gateway has a mature resume-aware handler.  Triton sidecars do
not own that engine registry, so they use this intentionally small wire
projector over :class:`SessionService`: the logical session, ordering and
terminal semantics remain shared while the native v2 JSON/binary frames stay
independent from OpenAI Realtime events.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from ..core.types import InputMode
from ..session import (
    AppendText,
    AudioOutput,
    CompleteInput,
    EventOutput,
    SessionHandle,
    SessionProtocolError,
    SessionService,
    StartedOutput,
    TerminalStatus,
)
from .session_identity import GatewaySessionIdentity
from .websocket_server import parse_session_start_request

try:
    from aiohttp import WSMsgType, web
except ImportError:  # pragma: no cover - deployment dependency
    WSMsgType = None
    web = None


logger = logging.getLogger(__name__)

NATIVE_WEBSOCKET_PATH = "/v1/ws"
NATIVE_WEBSOCKET_PROTOCOL = "tts-session-v2alpha1"
_HEARTBEAT_SECONDS = 30.0


class NativeSessionGateway:
    """Serve native v2 frames from a shared typed session service."""

    def __init__(
        self,
        service: SessionService,
        *,
        capabilities: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._service = service
        self._capabilities_provider = capabilities

    async def handle_capabilities(self, _request):
        return web.json_response(self._capabilities())

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=_HEARTBEAT_SECONDS)
        await ws.prepare(request)
        handle: SessionHandle | None = None
        output_task: asyncio.Task | None = None
        input_closed = False
        output_sample_limit = 0
        playback_played_sample = 0
        playback_buffered_sample = 0

        async def send_output(active: SessionHandle) -> None:
            nonlocal handle, output_task, input_closed, output_sample_limit
            try:
                async for output in active.outputs():
                    if isinstance(output, StartedOutput):
                        await ws.send_json(
                            {
                                "type": "event",
                                "event": {
                                    "type": "start",
                                    "session_id": output.session_id,
                                    "audio": {
                                        "encoding": output.audio.encoding,
                                        "sample_rate": output.audio.sample_rate,
                                        "channels": output.audio.channels,
                                    },
                                    "meta": dict(output.meta),
                                },
                            }
                        )
                    elif isinstance(output, AudioOutput):
                        output_sample_limit = max(
                            output_sample_limit, output.output_sample_end
                        )
                        # Header + binary is one logical audio delivery.  The
                        # absolute sample range is based on final output PCM,
                        # so SDKs can join it to text anchors without using
                        # send-time byte guesses.
                        await ws.send_json(
                            {
                                "type": "audio_header",
                                "start_sample": output.output_sample_start,
                                "end_sample": output.output_sample_end,
                                "audio": {
                                    "encoding": output.audio.encoding,
                                    "sample_rate": output.audio.sample_rate,
                                    "channels": output.audio.channels,
                                    "meta": {
                                        **dict(output.meta),
                                        "output_sample_start": str(
                                            output.output_sample_start
                                        ),
                                        "output_sample_end": str(
                                            output.output_sample_end
                                        ),
                                        "output_sample_rate": str(
                                            output.audio.sample_rate
                                        ),
                                    },
                                },
                            }
                        )
                        await ws.send_bytes(output.pcm_bytes)
                    elif isinstance(output, EventOutput):
                        await ws.send_json(
                            {
                                "type": "event",
                                "event": {
                                    "type": output.event_type,
                                    "session_id": output.session_id,
                                    "segment_id": output.segment_id,
                                    "text": output.text,
                                    "message": output.message,
                                    "meta": dict(output.meta),
                                },
                            }
                        )
                    else:
                        event_type = (
                            "error"
                            if output.status == TerminalStatus.FAILED
                            else "done"
                        )
                        meta = {str(k): str(v) for k, v in output.metrics.items()}
                        meta.setdefault("terminal_reason", output.status.value)
                        # The marker is harmless to clients and allows a native
                        # SDK to keep a healthy Triton socket for serial calls.
                        meta.setdefault("websocket_connection_reusable", "true")
                        await ws.send_json(
                            {
                                "type": "event",
                                "event": {
                                    "type": event_type,
                                    "session_id": output.session_id,
                                    "message": output.message,
                                    "meta": meta,
                                },
                            }
                        )
                await self._service.close_session(active.internal_session_id)
                input_closed = False
                handle = None
            except (ConnectionError, RuntimeError):
                # The receive loop owns connection teardown.  A peer may close
                # between two output frames; do not turn that normal transport
                # race into an unhandled task exception.
                logger.debug("native websocket output detached", exc_info=True)
            except asyncio.CancelledError:
                raise
            finally:
                if output_task is asyncio.current_task():
                    output_task = None

        async def stop_active(reason: str = "") -> None:
            nonlocal handle, output_task, input_closed
            active = handle
            if active is None:
                return
            try:
                await active.cancel(reason)
            except Exception:
                logger.exception("native session cancellation failed")
            if output_task is not None and not output_task.done():
                try:
                    await output_task
                except asyncio.CancelledError:
                    pass
            await self._service.close_session(active.internal_session_id)
            handle = None
            output_task = None
            input_closed = False

        async def send_error(exc: Exception) -> None:
            code = getattr(exc, "code", "invalid_request")
            if not ws.closed:
                await ws.send_json(
                    {
                        "type": "event",
                        "event": {
                            "type": "error",
                            "session_id": handle.client_session_id if handle else "",
                            "message": str(exc),
                            "meta": {"code": str(code)},
                        },
                    }
                )

        try:
            async for message in ws:
                if message.type == WSMsgType.ERROR:
                    raise RuntimeError(str(message.data))
                if message.type == WSMsgType.BINARY:
                    raise ValueError(
                        "binary client frames are not supported; send JSON control messages only"
                    )
                if message.type != WSMsgType.TEXT:
                    continue
                payload = _json_object(message.data)
                message_type = str(payload.get("type") or "").strip().lower()

                if message_type == "get_capabilities":
                    await ws.send_json(
                        {
                            "type": "capabilities",
                            "websocket_connection_reusable": True,
                            "capabilities": self._capabilities(),
                        }
                    )
                    continue

                if message_type == "playback_progress":
                    played = _required_int(
                        payload, "played_through_sample", positive=False
                    )
                    buffered = _required_int(
                        payload, "buffered_through_sample", positive=False
                    )
                    try:
                        if buffered < played:
                            raise ValueError(
                                "buffered_through_sample must be >= played_through_sample"
                            )
                        if played <= playback_played_sample and buffered <= playback_buffered_sample:
                            continue
                        if played < playback_played_sample:
                            raise ValueError("played_through_sample cannot move backwards")
                        if buffered < playback_buffered_sample:
                            raise ValueError("buffered_through_sample cannot move backwards")
                        if buffered > output_sample_limit:
                            raise ValueError(
                                "buffered_through_sample is ahead of server output"
                            )
                        playback_played_sample = played
                        playback_buffered_sample = buffered
                    except ValueError as exc:
                        await ws.send_json(
                            {
                                "type": "playback_progress_error",
                                "code": "invalid_playback_progress",
                                "message": str(exc),
                            }
                        )
                    continue

                if message_type == "start":
                    if handle is not None:
                        raise SessionProtocolError(
                            "session_active", "websocket session has already been started"
                        )
                    start_request = parse_session_start_request(
                        payload, default_mode=InputMode.AUTO
                    )
                    identity = GatewaySessionIdentity.create(payload.get("session_id"))
                    handle = await self._service.create(
                        identity, start_request=start_request
                    )
                    output_task = asyncio.create_task(send_output(handle))
                    continue

                if message_type == "oneshot":
                    if handle is not None:
                        raise SessionProtocolError(
                            "session_active", "websocket session has already been started"
                        )
                    start_request = parse_session_start_request(
                        payload, default_mode=InputMode.FULL_TEXT
                    )
                    identity = GatewaySessionIdentity.create(payload.get("session_id"))
                    handle = await self._service.create(
                        identity, start_request=start_request
                    )
                    output_task = asyncio.create_task(send_output(handle))
                    text = str(payload.get("text") or "")
                    if not text:
                        raise ValueError("oneshot request requires non-empty 'text'")
                    await handle.append_text(AppendText(seq_no=1, text=text))
                    await handle.complete_input(CompleteInput(final_seq_no=1))
                    input_closed = True
                    continue

                if handle is None:
                    raise SessionProtocolError(
                        "session_not_started", "received input before 'start'"
                    )
                if message_type == "text":
                    seq_no = _required_int(payload, "seq_no", positive=True)
                    ack = await handle.append_text(
                        AppendText(seq_no=seq_no, text=str(payload.get("text") or ""))
                    )
                    await ws.send_json(
                        {
                            "type": "text_ack",
                            "through_seq": ack.seq_no,
                            "duplicate": ack.duplicate,
                        }
                    )
                elif message_type in {"end", "stop"}:
                    final_seq = _required_int(
                        payload,
                        "final_seq_no",
                        positive=False,
                        default=handle.accepted_text_seq,
                    )
                    ack = await handle.complete_input(
                        CompleteInput(final_seq_no=final_seq)
                    )
                    input_closed = True
                    await ws.send_json(
                        {
                            "type": "input_ack",
                            "final_seq_no": ack.seq_no,
                            "duplicate": ack.duplicate,
                        }
                    )
                elif message_type == "cancel":
                    await stop_active(str(payload.get("reason") or ""))
                else:
                    raise ValueError(
                        f"unsupported websocket message type: '{message_type or '<empty>'}'"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("native websocket request failed: %s", exc)
            await send_error(exc)
        finally:
            if handle is not None:
                await stop_active("websocket_closed")
            if output_task is not None and not output_task.done():
                output_task.cancel()
                try:
                    await output_task
                except asyncio.CancelledError:
                    pass
            if not ws.closed:
                await ws.close()
        return ws

    def _capabilities(self) -> dict[str, Any]:
        if self._capabilities_provider is None:
            return {
                "protocols": {
                    "native_websocket": {
                        "path": NATIVE_WEBSOCKET_PATH,
                        "current": NATIVE_WEBSOCKET_PROTOCOL,
                        "supported": [NATIVE_WEBSOCKET_PROTOCOL],
                        "features": ["persistent_sessions_v1"],
                    }
                }
            }
        return dict(self._capabilities_provider())


def _json_object(raw: Any) -> dict[str, Any]:
    import json

    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid websocket JSON payload: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("websocket payload must be a JSON object")
    return value


def _required_int(
    payload: dict[str, Any],
    key: str,
    *,
    positive: bool,
    default: int | None = None,
) -> int:
    raw = payload.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(
            f"{key} must be {'positive' if positive else 'non-negative'}"
        )
    return value


__all__ = [
    "NATIVE_WEBSOCKET_PATH",
    "NATIVE_WEBSOCKET_PROTOCOL",
    "NativeSessionGateway",
]
