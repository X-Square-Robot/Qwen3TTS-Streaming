"""OpenAI Realtime-compatible WebSocket facade for streaming TTS.

The public wire protocol follows the OpenAI Realtime event lifecycle.  Qwen's
token-level text ingress is exposed as one deliberately namespaced extension:

* ``qwen.input_text_buffer.append``
* ``qwen.input_text_buffer.commit``

This is necessary because Realtime defines full-duplex transport and streaming
audio output, but does not define appending text to an already-created input
item.  Standard clients can instead send a complete ``conversation.item.create``
followed by ``response.create``.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from ..core.lifecycle import LifecycleLogger
from ..core.types import InputMode
from ..session import SessionService
from .session_identity import GatewaySessionIdentity
from .websocket_server import parse_session_start_request

if TYPE_CHECKING:
    from ..interface import SessionStartRequest
    from ..server import TTSEngine

try:
    from aiohttp import WSMsgType, web
except ImportError:  # pragma: no cover
    WSMsgType = None
    web = None


logger = logging.getLogger(__name__)

OPENAI_REALTIME_PATH = "/v1/realtime"
OPENAI_REALTIME_PROTOCOL = "openai-realtime-v1"
QWEN_REALTIME_EXTENSION_PROTOCOL = "qwen-realtime-v1"
QWEN_TEXT_BUFFER_EXTENSION = "qwen.input_text_buffer.v1"
QWEN_TEXT_PROGRESS_EXTENSION = "qwen.text_progress.v1"
QWEN_PLAYBACK_ACK_EXTENSION = "qwen.playback_ack.v1"
_HEARTBEAT_SECONDS = float(
    os.environ.get("ENGINE_WEBSOCKET_HEARTBEAT_SEC", "30") or "30"
)
_OUTBOUND_QUEUE_MAXSIZE = int(
    os.environ.get("ENGINE_WEBSOCKET_AUDIO_QUEUE_MAXSIZE", "4096") or "4096"
)

SessionStarter = Callable[..., Awaitable[None]]
UsageRecorder = Callable[[dict[str, Any]], Any]


class RealtimeSessionBackend(Protocol):
    """Transport-neutral execution boundary used by the Realtime facade.

    The standalone implementation delegates to ``TTSEngine``.  A Triton
    sidecar can implement the same contract over bidirectional gRPC without
    making OpenAI wire events part of the Triton model contract.
    """

    async def start(
        self,
        identity: GatewaySessionIdentity,
        *,
        start_request: "SessionStartRequest",
        outbound_queue: asyncio.Queue,
    ) -> None: ...

    async def push_text(self, session_id: str, text: str) -> None: ...

    async def complete_input(self, session_id: str) -> None: ...

    async def cancel(self, session_id: str) -> None: ...

    def count_text_tokens(self, text: str) -> int: ...


class EngineRealtimeBackend:
    """Realtime execution adapter for the in-process standalone engine."""

    def __init__(self, engine: "TTSEngine", session_starter: SessionStarter) -> None:
        self._engine = engine
        self._session_starter = session_starter

    async def start(
        self,
        identity: GatewaySessionIdentity,
        *,
        start_request: "SessionStartRequest",
        outbound_queue: asyncio.Queue,
    ) -> None:
        await self._session_starter(
            identity, start_request=start_request, outbound_queue=outbound_queue
        )

    async def push_text(self, session_id: str, text: str) -> None:
        await self._engine.push_text_input(session_id, text)

    async def complete_input(self, session_id: str) -> None:
        await self._engine.mark_input_complete(session_id)

    async def cancel(self, session_id: str) -> None:
        await self._engine.cancel(session_id)

    def count_text_tokens(self, text: str) -> int:
        return self._engine.count_text_tokens(text)


class RealtimeProtocolError(ValueError):
    def __init__(self, code: str, message: str, *, param: str | None = None):
        super().__init__(message)
        self.code = code
        self.param = param


@dataclass
class _SessionSettings:
    session_id: str
    model: str
    instructions: str = ""
    voice: str = ""
    sample_rate: int = 24000
    qwen: dict[str, Any] = field(default_factory=dict)

    def public_payload(self) -> dict[str, Any]:
        public_qwen = {
            key: value for key, value in self.qwen.items() if key not in {"ref_audio"}
        }
        return {
            "type": "realtime",
            "object": "realtime.session",
            "id": self.session_id,
            "model": self.model,
            "output_modalities": ["audio"],
            "instructions": self.instructions,
            "audio": {
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.sample_rate,
                    },
                    "voice": self.voice or None,
                }
            },
            "qwen": {
                **public_qwen,
                "protocol_version": QWEN_REALTIME_EXTENSION_PROTOCOL,
                "text_buffer_extension": QWEN_TEXT_BUFFER_EXTENSION,
                "text_progress_extension": QWEN_TEXT_PROGRESS_EXTENSION,
                "playback_ack_extension": QWEN_PLAYBACK_ACK_EXTENSION,
            },
        }

    def apply_update(self, update: Any, *, voice_locked: bool) -> None:
        if not isinstance(update, dict):
            raise RealtimeProtocolError(
                "invalid_session",
                "session.update requires a session object",
                param="session",
            )

        session_type = update.get("type")
        if session_type not in (None, "", "realtime"):
            raise RealtimeProtocolError(
                "unsupported_session_type",
                "only realtime synthesis sessions are supported",
                param="session.type",
            )

        model = update.get("model")
        if model not in (None, ""):
            self.model = str(model)
        if "instructions" in update:
            self.instructions = str(update.get("instructions") or "")

        modalities = update.get("output_modalities", update.get("modalities"))
        if modalities is not None:
            if not isinstance(modalities, list) or "audio" not in modalities:
                raise RealtimeProtocolError(
                    "unsupported_modality",
                    "this endpoint requires audio in output_modalities",
                    param="session.output_modalities",
                )

        voice: Any = update.get("voice")
        audio = update.get("audio")
        if audio is not None and not isinstance(audio, dict):
            raise RealtimeProtocolError(
                "invalid_audio_config",
                "session.audio must be an object",
                param="session.audio",
            )
        output_audio = audio.get("output") if isinstance(audio, dict) else None
        if output_audio is not None and not isinstance(output_audio, dict):
            raise RealtimeProtocolError(
                "invalid_audio_config",
                "session.audio.output must be an object",
                param="session.audio.output",
            )
        if isinstance(output_audio, dict):
            if "voice" in output_audio:
                voice = output_audio.get("voice")
            if "format" in output_audio:
                self.sample_rate = _parse_output_audio_format(
                    output_audio.get("format")
                )
        if "output_audio_format" in update:
            self.sample_rate = _parse_output_audio_format(
                update.get("output_audio_format")
            )

        if voice is not None:
            new_voice = str(voice or "")
            if voice_locked and new_voice != self.voice:
                raise RealtimeProtocolError(
                    "voice_change_not_allowed",
                    "voice cannot be changed after audio has been emitted",
                    param="session.audio.output.voice",
                )
            self.voice = new_voice

        qwen = update.get("qwen")
        if qwen is not None:
            if not isinstance(qwen, dict):
                raise RealtimeProtocolError(
                    "invalid_qwen_config",
                    "session.qwen must be an object",
                    param="session.qwen",
                )
            self.qwen.update(qwen)


@dataclass
class _TextBuffer:
    chunks: list[str] = field(default_factory=list)
    committed: bool = False
    next_sequence: int = 1
    accepted_sequences: dict[int, str] = field(default_factory=dict)

    def reset_for_response(self) -> None:
        """Start a fresh Qwen incremental-input journal.

        The buffer is response-scoped.  Keeping its sequence journal on the
        physical connection made a second serial response reject sequence 1
        and could leak accepted text into the next response.
        """

        self.chunks.clear()
        self.committed = False
        self.next_sequence = 1
        self.accepted_sequences.clear()

    def append(self, text: Any, sequence: Any) -> tuple[int, bool]:
        value = str(text or "")
        if not value:
            raise RealtimeProtocolError(
                "invalid_text", "append text must not be empty", param="text"
            )
        try:
            seq = int(sequence)
        except (TypeError, ValueError) as exc:
            raise RealtimeProtocolError(
                "invalid_sequence", "sequence must be an integer", param="sequence"
            ) from exc
        prior = self.accepted_sequences.get(seq)
        if prior is not None:
            if prior != value:
                raise RealtimeProtocolError(
                    "sequence_conflict",
                    "a repeated sequence must contain the same text",
                    param="sequence",
                )
            return seq, True
        if seq != self.next_sequence:
            raise RealtimeProtocolError(
                "sequence_gap",
                f"expected sequence {self.next_sequence}, received {seq}",
                param="sequence",
            )
        if self.committed:
            raise RealtimeProtocolError(
                "input_already_committed",
                "cannot append after input commit",
                param="type",
            )
        self.accepted_sequences[seq] = value
        self.next_sequence += 1
        self.chunks.append(value)
        return seq, False


@dataclass
class _ResponseState:
    response_id: str
    item_id: str
    identity: GatewaySessionIdentity
    start_request: "SessionStartRequest"
    model: str
    voice: str
    instructions: str
    ref_text: str
    sample_rate: int
    queue: asyncio.Queue
    pending_chunks: list[str] = field(default_factory=list)
    commit_requested: bool = False
    input_chunks: list[str] = field(default_factory=list)
    engine_started: bool = False
    input_complete: bool = False
    audio_samples: int = 0
    finished: bool = False
    task: asyncio.Task | None = None
    ingest_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    finish_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class OpenAIRealtimeGateway:
    """Serve an OpenAI Realtime-compatible, full-duplex TTS connection."""

    def __init__(
        self,
        engine: "TTSEngine | None",
        *,
        session_starter: SessionStarter | None = None,
        backend: RealtimeSessionBackend | None = None,
        session_service: SessionService | None = None,
        usage_recorder: UsageRecorder | None = None,
    ) -> None:
        self._legacy_owner = None
        self._service_backend = None
        if backend is None and session_service is not None:
            from .session_backend import RealtimeSessionServiceBackend

            self._service_backend = RealtimeSessionServiceBackend(session_service)
            backend = self._service_backend
        if backend is None and session_starter is None:
            if engine is None:
                raise RuntimeError(
                    "Realtime gateway requires an engine or a session backend"
                )
            # Reuse the canonical output/VAD pipeline while the old gateway and
            # the Realtime facade coexist.  The backend boundary can later be
            # replaced by a Triton gRPC sidecar without changing this handler.
            from .websocket_server import WebSocketGateway

            self._legacy_owner = WebSocketGateway(engine)
            session_starter = self._legacy_owner._create_session
        if backend is None:
            if engine is None or session_starter is None:  # defensive
                raise RuntimeError("Realtime gateway requires a session backend")
            backend = EngineRealtimeBackend(engine, session_starter)
        self._backend = backend
        self._usage_recorder = usage_recorder

    async def close(self) -> None:
        if self._service_backend is not None:
            await self._service_backend.close()
        elif self._legacy_owner is not None:
            await self._legacy_owner.close()

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=_HEARTBEAT_SECONDS)
        await ws.prepare(request)
        connection = _RealtimeConnection(
            backend=self._backend,
            usage_recorder=self._usage_recorder,
            ws=ws,
            model=str(request.query.get("model") or "qwen3-tts-realtime"),
        )
        await connection.run()
        return ws


class _RealtimeConnection:
    def __init__(
        self,
        *,
        backend: RealtimeSessionBackend,
        usage_recorder: UsageRecorder | None,
        ws: Any,
        model: str,
    ) -> None:
        self._backend = backend
        self._usage_recorder = usage_recorder
        self._ws = ws
        self._settings = _SessionSettings(
            session_id=f"sess_{uuid.uuid4().hex}", model=model
        )
        self._conversation_text: list[str] = []
        self._buffer = _TextBuffer()
        self._active: _ResponseState | None = None
        self._voice_locked = False

    async def run(self) -> None:
        await self._send(
            {
                "type": "session.created",
                "session": self._settings.public_payload(),
            }
        )
        try:
            async for message in self._ws:
                if message.type == WSMsgType.TEXT:
                    client_event_id: str | None = None
                    try:
                        event = json.loads(message.data)
                        if not isinstance(event, dict):
                            raise RealtimeProtocolError(
                                "invalid_event", "client event must be a JSON object"
                            )
                        client_event_id = _optional_id(event.get("event_id"))
                        await self._handle_event(event)
                    except json.JSONDecodeError as exc:
                        await self._send_error(
                            RealtimeProtocolError("invalid_json", str(exc))
                        )
                    except RealtimeProtocolError as exc:
                        await self._send_error(exc, client_event_id=client_event_id)
                    except Exception as exc:  # keep request errors non-fatal
                        logger.exception("Realtime client event failed")
                        await self._send_error(
                            RealtimeProtocolError("server_error", str(exc)),
                            client_event_id=client_event_id,
                            error_type="server_error",
                        )
                    continue
                if message.type == WSMsgType.BINARY:
                    await self._send_error(
                        RealtimeProtocolError(
                            "unsupported_binary_input",
                            "binary input is not supported by this TTS-only endpoint",
                        )
                    )
                    continue
                if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        finally:
            await self._cancel_active(reason="connection_closed", send_events=False)
            if not self._ws.closed:
                await self._ws.close()

    async def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "session.update":
            candidate = _SessionSettings(
                session_id=self._settings.session_id,
                model=self._settings.model,
                instructions=self._settings.instructions,
                voice=self._settings.voice,
                sample_rate=self._settings.sample_rate,
                qwen=dict(self._settings.qwen),
            )
            candidate.apply_update(
                event.get("session"), voice_locked=self._voice_locked
            )
            self._settings = candidate
            await self._send(
                {
                    "type": "session.updated",
                    "session": self._settings.public_payload(),
                }
            )
            return
        if event_type == "conversation.item.create":
            await self._create_conversation_item(event)
            return
        if event_type == "response.create":
            await self._create_response(event)
            return
        if event_type == "response.cancel":
            await self._cancel_active(reason="client_cancelled", send_events=True)
            return
        if event_type == "qwen.input_text_buffer.append":
            await self._append_text(event)
            return
        if event_type == "qwen.input_text_buffer.commit":
            await self._commit_text()
            return
        if event_type == "qwen.playback.ack":
            await self._ack_playback(event)
            return
        raise RealtimeProtocolError(
            "unsupported_event",
            f"unsupported client event type: {event_type!r}",
            param="type",
        )

    async def _create_conversation_item(self, event: dict[str, Any]) -> None:
        item = event.get("item")
        if not isinstance(item, dict):
            raise RealtimeProtocolError(
                "invalid_item",
                "conversation.item.create requires an item object",
                param="item",
            )
        text = _text_from_item(item)
        if not text:
            raise RealtimeProtocolError(
                "invalid_item",
                "input item must contain non-empty input_text",
                param="item.content",
            )
        item_id = str(item.get("id") or f"item_{uuid.uuid4().hex}")
        normalized_item = {
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": str(item.get("role") or "user"),
            "content": [{"type": "input_text", "text": text}],
        }
        self._conversation_text.append(text)
        common = {
            "previous_item_id": event.get("previous_item_id"),
            "item": normalized_item,
        }
        await self._send({"type": "conversation.item.added", **common})
        await self._send({"type": "conversation.item.done", **common})

    async def _create_response(self, event: dict[str, Any]) -> None:
        if self._active is not None and not self._active.finished:
            raise RealtimeProtocolError(
                "response_in_progress",
                "only one active response is supported per connection",
                param="type",
            )
        response_options = event.get("response") or {}
        if not isinstance(response_options, dict):
            raise RealtimeProtocolError(
                "invalid_response",
                "response.create.response must be an object",
                param="response",
            )

        explicit_text = _text_from_response_input(response_options.get("input"))
        conversation_text = "".join(self._conversation_text)
        self._conversation_text.clear()
        buffered_chunks = list(self._buffer.chunks)
        buffer_committed = self._buffer.committed
        self._buffer.chunks.clear()
        self._buffer.committed = False

        initial_chunks: list[str] = []
        if explicit_text:
            initial_chunks.append(explicit_text)
        elif conversation_text:
            initial_chunks.append(conversation_text)
        initial_chunks.extend(buffered_chunks)
        complete_standard_input = bool(explicit_text or conversation_text)
        commit_requested = buffer_committed or (
            complete_standard_input and not buffered_chunks
        )

        response_id = f"resp_{uuid.uuid4().hex}"
        item_id = f"item_{uuid.uuid4().hex}"
        identity = GatewaySessionIdentity.create(response_id)
        instructions = str(
            response_options.get("instructions", self._settings.instructions) or ""
        )
        start_request = self._build_start_request(
            response_id=response_id,
            instructions=instructions,
        )
        state = _ResponseState(
            response_id=response_id,
            item_id=item_id,
            identity=identity,
            start_request=start_request,
            model=self._settings.model,
            voice=self._settings.voice,
            instructions=str(start_request.config.instruct or ""),
            ref_text=str(start_request.config.ref_text or ""),
            sample_rate=start_request.config.audio.sample_rate,
            queue=asyncio.Queue(maxsize=_OUTBOUND_QUEUE_MAXSIZE),
            pending_chunks=initial_chunks,
            commit_requested=commit_requested,
            input_chunks=list(initial_chunks),
        )
        self._active = state

        # The chunks above belong to this response.  Future incremental input
        # must start a new, response-scoped sequence journal.
        self._buffer.reset_for_response()

        await self._send(
            {"type": "response.created", "response": self._response_payload(state)}
        )
        await self._send(
            {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": self._output_item(
                    state, status="in_progress", include_content=False
                ),
            }
        )
        await self._send(
            {
                "type": "response.content_part.added",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "audio", "audio": "", "transcript": ""},
            }
        )
        state.task = asyncio.create_task(self._run_response(state))

    def _build_start_request(
        self, *, response_id: str, instructions: str
    ) -> "SessionStartRequest":
        qwen = dict(self._settings.qwen)
        timing = dict(qwen.get("timing") or {})
        timing.setdefault("request_id", response_id)
        timing_extra = dict(timing.get("extra") or {})
        timing_extra["api_protocol"] = OPENAI_REALTIME_PROTOCOL
        timing["extra"] = timing_extra
        config: dict[str, Any] = {
            "protocol_version": OPENAI_REALTIME_PROTOCOL,
            "task_type": str(qwen.get("task_type") or ""),
            "language": str(qwen.get("language") or "auto"),
            "speaker": str(qwen.get("speaker") or self._settings.voice or "") or None,
            "instruct": instructions or qwen.get("instruct"),
            "ref_audio": qwen.get("ref_audio"),
            "ref_text": qwen.get("ref_text"),
            "x_vector_only": bool(qwen.get("x_vector_only", False)),
            "input_mode": qwen.get("input_mode", "auto"),
            "group_policy": qwen.get("group_policy", "auto"),
            "audio": {
                "encoding": "pcm_s16le",
                "sample_rate": self._settings.sample_rate,
                "channels": 1,
            },
            "output_policy": qwen.get("output_policy", {}),
            "timing": timing,
        }
        return parse_session_start_request(
            {"type": "start", "session_id": response_id, "config": config},
            default_mode=InputMode.AUTO,
        )

    async def _run_response(self, state: _ResponseState) -> None:
        try:
            await self._backend.start(
                state.identity,
                start_request=state.start_request,
                outbound_queue=state.queue,
            )
            async with state.ingest_lock:
                state.engine_started = True
                pending = list(state.pending_chunks)
                state.pending_chunks.clear()
                for text in pending:
                    await self._backend.push_text(
                        state.identity.internal_session_id, text
                    )
                if state.commit_requested and not state.input_complete:
                    state.input_complete = True
                    await self._backend.complete_input(
                        state.identity.internal_session_id
                    )

            while not state.finished:
                frame = await state.queue.get()
                if frame.get("type") == "audio":
                    audio = frame.get("audio") or {}
                    pcm = audio.get("pcm_data") or b""
                    channels = int(audio.get("channels", 1) or 1)
                    if pcm:
                        # The adapter always asks the shared output pipeline for
                        # signed 16-bit PCM; duration is the billing ground truth.
                        state.audio_samples += len(pcm) // (2 * channels)
                        self._voice_locked = True
                        await self._send(
                            {
                                "type": "response.output_audio.delta",
                                "response_id": state.response_id,
                                "item_id": state.item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": base64.b64encode(pcm).decode("ascii"),
                            }
                        )
                    continue
                event = frame.get("event") or {}
                event_type = str(event.get("type") or "")
                if event_type == "done":
                    await self._finish_response(state, status="completed")
                    return
                if event_type in {
                    "text_token",
                    "text_boundary_commit",
                    "text_progress",
                }:
                    # These are Qwen extensions carried on the Realtime data
                    # channel.  Standard Realtime clients can ignore unknown
                    # namespaced events; Qwen clients use them to render the
                    # source text cursor and future ASR/alignment revisions.
                    event_meta = dict(event.get("meta") or {})
                    if event_type == "text_progress" and "output_sample_end" not in event_meta:
                        # A legacy backend may still emit the coarse EMA
                        # percentage, but bytes sent on this facade are not a
                        # playback/alignment coordinate. Forward the legacy
                        # fields without manufacturing a v1 anchor.
                        event_meta.pop("anchor_seq", None)
                        event_meta.pop("alignment_final", None)
                    raw_segment_id = event.get("segment_id")
                    if raw_segment_id is None:
                        raw_segment_id = event.get("segment_idx", -1)
                    segment_id = int(raw_segment_id)
                    await self._send(
                        {
                            "type": f"qwen.{event_type}",
                            "response_id": state.response_id,
                            "item_id": state.item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "segment_id": segment_id,
                            "text": str(event.get("text") or ""),
                            "meta": event_meta,
                        }
                    )
                    continue
                if event_type == "error":
                    message = str(event.get("message") or "engine synthesis failed")
                    await self._send_error(
                        RealtimeProtocolError("synthesis_failed", message),
                        error_type="server_error",
                    )
                    await self._finish_response(
                        state, status="failed", failure_message=message
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Realtime response failed: %s", state.response_id)
            await self._send_error(
                RealtimeProtocolError("synthesis_failed", str(exc)),
                error_type="server_error",
            )
            try:
                await self._backend.cancel(state.identity.internal_session_id)
            except Exception:
                logger.exception("Failed to cancel errored Realtime response")
            await self._finish_response(
                state, status="failed", failure_message=str(exc)
            )

    async def _append_text(self, event: dict[str, Any]) -> None:
        target = self._active
        if target is None or target.finished:
            seq, duplicate = self._buffer.append(
                event.get("text"), event.get("sequence")
            )
        else:
            async with target.ingest_lock:
                if target.finished:
                    seq, duplicate = self._buffer.append(
                        event.get("text"), event.get("sequence")
                    )
                else:
                    if target.input_complete:
                        raise RealtimeProtocolError(
                            "input_already_committed",
                            "cannot append after input commit",
                        )
                    seq, duplicate = self._buffer.append(
                        event.get("text"), event.get("sequence")
                    )
                    text = self._buffer.accepted_sequences[seq]
                    # _TextBuffer owns connection-wide sequence/idempotency;
                    # move active-response chunks out of its holding list.
                    if not duplicate:
                        self._buffer.chunks.pop()
                        target.input_chunks.append(text)
                        if target.engine_started:
                            await self._backend.push_text(
                                target.identity.internal_session_id, text
                            )
                        else:
                            target.pending_chunks.append(text)
        await self._send(
            {
                "type": "qwen.input_text_buffer.ack",
                "sequence": seq,
                "duplicate": duplicate,
            }
        )

    async def _commit_text(self) -> None:
        state = self._active
        if state is None or state.finished:
            self._buffer.committed = True
            await self._send({"type": "qwen.input_text_buffer.committed"})
            return
        async with state.ingest_lock:
            if not state.input_complete:
                state.commit_requested = True
                if state.engine_started:
                    state.input_complete = True
                    await self._backend.complete_input(
                        state.identity.internal_session_id
                    )
        await self._send(
            {
                "type": "qwen.input_text_buffer.committed",
                "response_id": state.response_id,
            }
        )

    async def _ack_playback(self, event: dict[str, Any]) -> None:
        response_id = str(event.get("response_id") or "")
        if not response_id:
            raise RealtimeProtocolError(
                "invalid_playback_ack",
                "qwen.playback.ack requires response_id",
                param="response_id",
            )
        for name in ("played_audio_sample_end", "played_text_char_end"):
            value = event.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RealtimeProtocolError(
                    "invalid_playback_ack",
                    f"{name} must be a non-negative integer",
                    param=name,
                )
        await self._send(
            {
                "type": "qwen.playback.ack",
                "response_id": response_id,
                "played_audio_sample_end": int(event.get("played_audio_sample_end", 0)),
                "played_text_char_end": int(event.get("played_text_char_end", 0)),
                "is_estimate": bool(event.get("is_estimate", True)),
                "accepted": True,
            }
        )

    async def _cancel_active(self, *, reason: str, send_events: bool) -> None:
        state = self._active
        if state is None or state.finished:
            if send_events:
                await self._send_error(
                    RealtimeProtocolError(
                        "no_active_response", "there is no active response to cancel"
                    )
                )
            return
        task = state.task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        try:
            await self._backend.cancel(state.identity.internal_session_id)
        except Exception:
            logger.exception(
                "Failed to cancel Realtime response: %s", state.response_id
            )
        await self._finish_response(
            state,
            status="cancelled",
            cancellation_reason=reason,
            send_events=send_events,
        )

    async def _finish_response(
        self,
        state: _ResponseState,
        *,
        status: str,
        failure_message: str = "",
        cancellation_reason: str = "",
        send_events: bool = True,
    ) -> None:
        async with state.finish_lock:
            if state.finished:
                return
            state.finished = True
            usage = self._usage(state)
            await self._record_usage(state, status=status, usage=usage)
            item_status = "completed" if status == "completed" else "incomplete"
            if send_events and not self._ws.closed:
                common = {
                    "response_id": state.response_id,
                    "item_id": state.item_id,
                    "output_index": 0,
                    "content_index": 0,
                }
                await self._send({"type": "response.output_audio.done", **common})
                await self._send(
                    {
                        "type": "response.content_part.done",
                        **common,
                        "part": {"type": "audio", "transcript": ""},
                    }
                )
                await self._send(
                    {
                        "type": "response.output_item.done",
                        "response_id": state.response_id,
                        "output_index": 0,
                        "item": self._output_item(
                            state, status=item_status, include_content=True
                        ),
                    }
                )
                await self._send(
                    {
                        "type": "response.done",
                        "response": self._response_payload(
                            state,
                            status=status,
                            usage=usage,
                            failure_message=failure_message,
                            cancellation_reason=cancellation_reason,
                        ),
                    }
                )
            if self._active is state:
                self._active = None
                # Voice locking protects one response after its first audio;
                # a later serial response may negotiate a different voice.
                self._voice_locked = False

    def _usage(self, state: _ResponseState) -> dict[str, Any]:
        input_text_tokens = sum(
            self._backend.count_text_tokens(value)
            for value in (
                "".join(state.input_chunks),
                state.instructions,
                state.ref_text,
            )
            if value
        )
        output_audio_tokens = math.ceil(
            state.audio_samples / (state.sample_rate * 0.05)
        )
        input_details = {
            "text_tokens": input_text_tokens,
            "audio_tokens": 0,
            "image_tokens": 0,
            "cached_tokens": 0,
            "cached_tokens_details": {
                "text_tokens": 0,
                "audio_tokens": 0,
                "image_tokens": 0,
            },
        }
        output_details = {"text_tokens": 0, "audio_tokens": output_audio_tokens}
        return {
            "total_tokens": input_text_tokens + output_audio_tokens,
            "input_tokens": input_text_tokens,
            "output_tokens": output_audio_tokens,
            "input_token_details": input_details,
            "output_token_details": output_details,
        }

    async def _record_usage(
        self, state: _ResponseState, *, status: str, usage: dict[str, Any]
    ) -> None:
        record = {
            "session_id": self._settings.session_id,
            "response_id": state.response_id,
            "model": state.model,
            "status": status,
            "usage": usage,
        }
        LifecycleLogger.emit(
            session_id=state.identity.internal_session_id,
            phase="billing.usage",
            transport="openai_realtime",
            realtime_session_id=self._settings.session_id,
            response_id=state.response_id,
            model=state.model,
            status=status,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            output_audio_tokens=usage["output_token_details"]["audio_tokens"],
        )
        if self._usage_recorder is None:
            return
        try:
            result = self._usage_recorder(record)
            if inspect.isawaitable(result):
                await result
        except Exception:
            # Usage has already reached the structured lifecycle log.  A
            # durable recorder failure is operationally visible but must not
            # suppress the terminal protocol event.
            logger.exception("Realtime usage recorder failed: %s", state.response_id)

    def _output_item(
        self, state: _ResponseState, *, status: str, include_content: bool
    ) -> dict[str, Any]:
        return {
            "id": state.item_id,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": (
                [{"type": "audio", "transcript": ""}] if include_content else []
            ),
        }

    def _response_payload(
        self,
        state: _ResponseState,
        *,
        status: str = "in_progress",
        usage: dict[str, Any] | None = None,
        failure_message: str = "",
        cancellation_reason: str = "",
    ) -> dict[str, Any]:
        status_details: dict[str, Any] | None = None
        if status == "cancelled":
            status_details = {
                "type": "cancelled",
                "reason": cancellation_reason or "client_cancelled",
            }
        elif status == "failed":
            status_details = {
                "type": "failed",
                "error": {
                    "type": "server_error",
                    "code": "synthesis_failed",
                    "message": failure_message or "engine synthesis failed",
                },
            }
        return {
            "object": "realtime.response",
            "id": state.response_id,
            "status": status,
            "status_details": status_details,
            "output": (
                []
                if status == "in_progress"
                else [
                    self._output_item(
                        state,
                        status="completed" if status == "completed" else "incomplete",
                        include_content=True,
                    )
                ]
            ),
            "conversation_id": self._settings.session_id,
            "output_modalities": ["audio"],
            "voice": state.voice or None,
            "usage": usage,
            "metadata": {
                "qwen_final_output_sample": str(state.audio_samples),
                "qwen_output_sample_rate": str(state.sample_rate),
            },
        }

    async def _send_error(
        self,
        error: RealtimeProtocolError,
        *,
        client_event_id: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        await self._send(
            {
                "type": "error",
                "error": {
                    "type": error_type,
                    "code": error.code,
                    "message": str(error),
                    "param": error.param,
                    "event_id": client_event_id,
                },
            }
        )

    async def _send(self, event: dict[str, Any]) -> None:
        if self._ws.closed:
            return
        payload = {"event_id": f"event_{uuid.uuid4().hex}", **event}
        await self._ws.send_json(payload)


def _parse_output_audio_format(value: Any) -> int:
    if value in (None, "", "pcm16"):
        return 24000
    if not isinstance(value, dict):
        raise RealtimeProtocolError(
            "unsupported_audio_format",
            "output format must be pcm16 or an audio/pcm format object",
            param="session.audio.output.format",
        )
    media_type = str(value.get("type") or "")
    if media_type != "audio/pcm":
        raise RealtimeProtocolError(
            "unsupported_audio_format",
            "only signed 16-bit audio/pcm output is supported",
            param="session.audio.output.format.type",
        )
    try:
        rate = int(value.get("rate", 24000) or 24000)
    except (TypeError, ValueError) as exc:
        raise RealtimeProtocolError(
            "unsupported_audio_format",
            "audio/pcm rate must be an integer",
            param="session.audio.output.format.rate",
        ) from exc
    if rate not in (16000, 24000):
        raise RealtimeProtocolError(
            "unsupported_audio_format",
            "audio/pcm rate must be 16000 or 24000",
            param="session.audio.output.format.rate",
        )
    return rate


def _text_from_item(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("input_text", "text"):
            parts.append(str(part.get("text") or ""))
    return "".join(parts)


def _text_from_response_input(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, list):
        raise RealtimeProtocolError(
            "invalid_response_input",
            "response.input must be an array",
            param="response.input",
        )
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            raise RealtimeProtocolError(
                "invalid_response_input",
                "response.input entries must be item objects",
                param="response.input",
            )
        parts.append(_text_from_item(item))
    return "".join(parts)


def _optional_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


__all__ = [
    "EngineRealtimeBackend",
    "OPENAI_REALTIME_PATH",
    "OPENAI_REALTIME_PROTOCOL",
    "QWEN_TEXT_BUFFER_EXTENSION",
    "OpenAIRealtimeGateway",
    "RealtimeSessionBackend",
]
