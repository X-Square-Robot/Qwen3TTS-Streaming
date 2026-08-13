"""Execution backends and wire shims for the shared session service.

The legacy standalone gateway still owns the mature VAD/output pipeline.  This
module adapts its queue into typed session outputs while that pipeline is being
migrated.  No Realtime JSON is sent to the legacy gateway: Realtime consumes
the typed outputs produced here and projects them to its own events.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..core.types import AudioConfig, AudioEncoding
from ..session import (
    AppendText,
    AudioFormat,
    AudioOutput,
    CompleteInput,
    EventOutput,
    ExecutionHandle,
    SessionOutput,
    SessionService,
    StartedOutput,
    TerminalOutput,
    TerminalStatus,
)
from .session_identity import GatewaySessionIdentity

if TYPE_CHECKING:
    from ..interface import SessionStartRequest
    from ..server import TTSEngine
    from .triton_realtime_backend import TritonRealtimeBackend
    from .websocket_server import WebSocketGateway


def _audio_format(
    payload: dict[str, Any], fallback: AudioConfig
) -> tuple[str, int, int]:
    return (
        str(payload.get("encoding") or fallback.encoding.value),
        int(payload.get("sample_rate") or fallback.sample_rate),
        int(payload.get("channels") or fallback.channels),
    )


def _sample_count(pcm: bytes, encoding: str, channels: int) -> int:
    width = 4 if encoding == AudioEncoding.PCM_F32.value else 2
    divisor = max(1, width * channels)
    return len(pcm) // divisor


def _raw_frame_to_output(
    frame: dict[str, Any],
    *,
    start_request: "SessionStartRequest",
    sample_cursor: int,
) -> tuple[SessionOutput, int] | None:
    frame_type = str(frame.get("type") or "")
    if frame_type == "audio":
        audio = dict(frame.get("audio") or {})
        pcm = bytes(audio.get("pcm_data") or b"")
        if not pcm:
            return None
        encoding, sample_rate, channels = _audio_format(
            audio, start_request.config.audio
        )
        meta = {str(k): str(v) for k, v in (audio.get("meta") or {}).items()}
        start = int(meta.get("output_sample_start", sample_cursor) or sample_cursor)
        end = int(
            meta.get("output_sample_end", start + _sample_count(pcm, encoding, channels))
            or start
        )
        return (
            AudioOutput(
                session_id=start_request.session_id,
                pcm_bytes=pcm,
                audio=AudioFormat(encoding, sample_rate, channels),
                output_sample_start=start,
                output_sample_end=end,
                meta=meta,
            ),
            end,
        )

    if frame_type != "event":
        return None
    event = dict(frame.get("event") or {})
    event_type = str(event.get("type") or "")
    session_id = str(event.get("session_id") or start_request.session_id)
    meta = {str(k): str(v) for k, v in (event.get("meta") or {}).items()}
    if event_type == "start":
        audio_payload = dict(event.get("audio") or {})
        encoding, sample_rate, channels = _audio_format(
            audio_payload, start_request.config.audio
        )
        return (
            StartedOutput(
                session_id=session_id,
                audio=AudioFormat(encoding, sample_rate, channels),
                meta=meta,
            ),
            sample_cursor,
        )
    if event_type in {"done", "error"}:
        reason = str(meta.get("terminal_reason") or "")
        status = (
            TerminalStatus.CANCELLED
            if reason == "cancelled"
            else TerminalStatus.FAILED
            if event_type == "error"
            else TerminalStatus.COMPLETED
        )
        return (
            TerminalOutput(
                session_id=session_id,
                status=status,
                message=str(event.get("message") or ""),
                metrics={str(k): v for k, v in meta.items()},
            ),
            sample_cursor,
        )
    return (
        EventOutput(
            session_id=session_id,
            event_type=event_type,
            segment_id=int(event.get("segment_id", -1) or -1),
            text=str(event.get("text") or ""),
            message=str(event.get("message") or ""),
            meta=meta,
        ),
        sample_cursor,
    )


class _LegacyExecutionHandle:
    def __init__(self, engine: "TTSEngine", internal_session_id: str) -> None:
        self._engine = engine
        self._internal_session_id = internal_session_id
        self._pump: asyncio.Task | None = None

    def set_pump(self, task: asyncio.Task) -> None:
        self._pump = task

    async def push_text(self, text: str) -> None:
        await self._engine.push_text_input(self._internal_session_id, text)

    async def complete_input(self) -> None:
        await self._engine.mark_input_complete(self._internal_session_id)

    async def cancel(self, reason: str = "") -> None:
        del reason
        await self._engine.cancel(self._internal_session_id)

    async def close(self) -> None:
        if self._pump is not None and not self._pump.done():
            self._pump.cancel()
            try:
                await self._pump
            except asyncio.CancelledError:
                pass


class StandaloneSessionBackend:
    """Bridge the existing standalone output pipeline into SessionService."""

    def __init__(self, gateway: "WebSocketGateway", engine: "TTSEngine") -> None:
        self._gateway = gateway
        self._engine = engine
        self._queues: set[asyncio.Queue] = set()

    async def start(
        self,
        identity: GatewaySessionIdentity,
        *,
        start_request: "SessionStartRequest",
        emit,
    ) -> ExecutionHandle:
        raw_queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
        self._queues.add(raw_queue)
        handle = _LegacyExecutionHandle(self._engine, identity.internal_session_id)

        async def pump() -> None:
            cursor = 0
            try:
                while True:
                    frame = await raw_queue.get()
                    converted = _raw_frame_to_output(
                        frame,
                        start_request=start_request,
                        sample_cursor=cursor,
                    )
                    if converted is None:
                        continue
                    output, cursor = converted
                    await emit(output)
                    if isinstance(output, TerminalOutput):
                        return
            finally:
                self._queues.discard(raw_queue)

        pump_task = asyncio.create_task(pump())
        handle.set_pump(pump_task)
        try:
            await self._gateway.create_session(
                identity,
                start_request=start_request,
                outbound_queue=raw_queue,
            )
        except BaseException:
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
            self._queues.discard(raw_queue)
            raise
        return handle

    async def close(self) -> None:
        self._queues.clear()


class RealtimeSessionServiceBackend:
    """Expose SessionService through the legacy Realtime backend shape."""

    def __init__(self, service: SessionService) -> None:
        self._service = service
        self._handles: dict[str, Any] = {}
        self._forwarders: dict[str, asyncio.Task] = {}
        self._next_seq: dict[str, int] = {}

    async def start(
        self,
        identity: GatewaySessionIdentity,
        *,
        start_request: "SessionStartRequest",
        outbound_queue: asyncio.Queue,
    ) -> None:
        handle = await self._service.create(identity, start_request=start_request)
        self._handles[identity.internal_session_id] = handle
        self._next_seq[identity.internal_session_id] = 1

        async def forward() -> None:
            try:
                async for output in handle.outputs():
                    await outbound_queue.put(
                        self._to_legacy_frame(output, start_request)
                    )
            finally:
                self._handles.pop(identity.internal_session_id, None)
                self._next_seq.pop(identity.internal_session_id, None)
                # A terminal output owns the complete logical response.  Drop
                # its service entry immediately so long-lived Realtime
                # connections do not accumulate finished Triton executions.
                if handle.terminal is not None:
                    await self._service.close_session(identity.internal_session_id)

        self._forwarders[identity.internal_session_id] = asyncio.create_task(forward())

    async def push_text(self, session_id: str, text: str) -> None:
        handle = self._require(session_id)
        seq = self._next_seq[session_id]
        await handle.append_text(AppendText(seq_no=seq, text=text))
        self._next_seq[session_id] = seq + 1

    async def complete_input(self, session_id: str) -> None:
        handle = self._require(session_id)
        await handle.complete_input(
            CompleteInput(final_seq_no=self._next_seq[session_id] - 1)
        )

    async def cancel(self, session_id: str) -> None:
        handle = self._handles.get(session_id)
        if handle is not None:
            await handle.cancel("realtime_cancelled")

    def count_text_tokens(self, text: str) -> int:
        counter = getattr(self._service.backend, "count_text_tokens", None)
        return int(counter(text)) if callable(counter) else len(text)

    async def close(self) -> None:
        for task in list(self._forwarders.values()):
            if not task.done():
                task.cancel()
        for task in list(self._forwarders.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._forwarders.clear()
        self._handles.clear()
        await self._service.close()

    def _require(self, session_id: str):
        handle = self._handles.get(session_id)
        if handle is None:
            raise RuntimeError(f"session {session_id} not found")
        return handle

    @staticmethod
    def _to_legacy_frame(output: SessionOutput, start_request):
        if isinstance(output, StartedOutput):
            return {
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
        if isinstance(output, AudioOutput):
            meta = dict(output.meta)
            meta.setdefault("output_sample_start", str(output.output_sample_start))
            meta.setdefault("output_sample_end", str(output.output_sample_end))
            return {
                "type": "audio",
                "audio": {
                    "pcm_data": output.pcm_bytes,
                    "encoding": output.audio.encoding,
                    "sample_rate": output.audio.sample_rate,
                    "channels": output.audio.channels,
                    "meta": meta,
                },
            }
        if isinstance(output, TerminalOutput):
            event_type = "error" if output.status == TerminalStatus.FAILED else "done"
            meta = {str(k): str(v) for k, v in output.metrics.items()}
            meta.setdefault("terminal_reason", output.status.value)
            return {
                "type": "event",
                "event": {
                    "type": event_type,
                    "session_id": output.session_id,
                    "message": output.message,
                    "meta": meta,
                },
            }
        return {
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


class _TritonExecutionHandle:
    def __init__(self, backend: "TritonRealtimeBackend", session_id: str) -> None:
        self._backend = backend
        self._session_id = session_id
        self._pump: asyncio.Task | None = None

    def set_pump(self, task: asyncio.Task) -> None:
        self._pump = task

    async def push_text(self, text: str) -> None:
        await self._backend.push_text(self._session_id, text)

    async def complete_input(self) -> None:
        await self._backend.complete_input(self._session_id)

    async def cancel(self, reason: str = "") -> None:
        del reason
        await self._backend.cancel(self._session_id)

    async def close(self) -> None:
        if self._pump is not None and not self._pump.done():
            self._pump.cancel()
            try:
                await self._pump
            except asyncio.CancelledError:
                pass


class TritonSessionBackend:
    """Adapt the Triton stream backend to the canonical session contract."""

    def __init__(self, backend: "TritonRealtimeBackend") -> None:
        self._backend = backend
        self._queues: set[asyncio.Queue] = set()

    async def start(
        self,
        identity: GatewaySessionIdentity,
        *,
        start_request: "SessionStartRequest",
        emit,
    ) -> ExecutionHandle:
        raw_queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
        self._queues.add(raw_queue)
        handle = _TritonExecutionHandle(self._backend, identity.internal_session_id)

        async def pump() -> None:
            cursor = 0
            try:
                while True:
                    frame = await raw_queue.get()
                    converted = _raw_frame_to_output(
                        frame,
                        start_request=start_request,
                        sample_cursor=cursor,
                    )
                    if converted is None:
                        continue
                    output, cursor = converted
                    await emit(output)
                    if isinstance(output, TerminalOutput):
                        return
            finally:
                self._queues.discard(raw_queue)

        pump_task = asyncio.create_task(pump())
        handle.set_pump(pump_task)
        try:
            await self._backend.start(
                identity,
                start_request=start_request,
                outbound_queue=raw_queue,
            )
        except BaseException:
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
            self._queues.discard(raw_queue)
            raise
        return handle

    async def close(self) -> None:
        self._queues.clear()
        await self._backend.close()


__all__ = [
    "RealtimeSessionServiceBackend",
    "StandaloneSessionBackend",
    "TritonSessionBackend",
]
