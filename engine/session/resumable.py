"""Transport-neutral ownership for resumable logical synthesis sessions."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .reliability import DeliveryAttachment, DeliveryLedger, LedgerError
from .service import AppendText, CompleteInput, InputAck, SessionHandle, SessionService

if TYPE_CHECKING:
    from ..gateway.session_identity import GatewaySessionIdentity
    from ..interface import SessionStartRequest


class ResumableSessionError(ValueError):
    """Stable protocol-neutral failure for resume lifecycle operations."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ResumableLogicalSession:
    """One execution and replay ledger shared by physical attachments."""

    def __init__(
        self,
        registry: "ResumableSessionRegistry",
        *,
        token: str,
        config_fingerprint: str,
        protocol: str,
        identity: "GatewaySessionIdentity",
        start_request: "SessionStartRequest",
        metadata: dict[str, Any] | None = None,
        output_observer: Callable[[Any], Awaitable[None] | None] | None = None,
    ) -> None:
        self.registry = registry
        self.token = token
        self.config_fingerprint = config_fingerprint
        self.protocol = protocol
        self.identity = identity
        self.start_request = start_request
        self.metadata = dict(metadata or {})
        self.output_observer = output_observer
        self.ledger = DeliveryLedger(
            token=token,
            config_fingerprint=config_fingerprint,
            protocol=protocol,
            max_buffer_bytes=registry.max_buffer_bytes,
            attachment_queue_maxsize=registry.attachment_queue_maxsize,
        )
        self.handle: SessionHandle | None = None
        self.ready = asyncio.Event()
        self.initialization_error: BaseException | None = None
        self.played_through_sample = 0
        self.buffered_through_sample = 0
        self._input_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._pump_task: asyncio.Task | None = None
        self._grace_task: asyncio.Task | None = None
        self._terminal_task: asyncio.Task | None = None
        self._closed = False

    @property
    def client_session_id(self) -> str:
        return self.identity.client_session_id

    @property
    def internal_session_id(self) -> str:
        return self.identity.internal_session_id

    @property
    def accepted_text_seq(self) -> int:
        return self.handle.accepted_text_seq if self.handle is not None else 0

    @property
    def input_closed(self) -> bool:
        return bool(self.handle is not None and self.handle.input_complete)

    async def initialize(self) -> None:
        try:
            self.handle = await self.registry.service.create(
                self.identity, start_request=self.start_request
            )
            self._pump_task = asyncio.create_task(self._pump_outputs())
        except BaseException as exc:
            self.initialization_error = exc
            raise
        finally:
            self.ready.set()

    async def wait_ready(self) -> None:
        await self.ready.wait()
        if self.initialization_error is not None:
            raise ResumableSessionError(
                "resume_session_initialization_failed",
                "resumable stream could not be initialized",
            ) from self.initialization_error

    async def attach(
        self,
        *,
        last_delivery_seq: int = 0,
        audio_through_sample: int = 0,
    ) -> DeliveryAttachment:
        await self.wait_ready()
        async with self._lifecycle_lock:
            if self._closed:
                raise ResumableSessionError(
                    "resume_session_not_found",
                    "resumable stream is no longer available",
                )
            self._cancel_task("_grace_task")
            try:
                return await self.ledger.attach(
                    last_delivery_seq=last_delivery_seq,
                    audio_through_sample=audio_through_sample,
                )
            except LedgerError as exc:
                raise ResumableSessionError(exc.code, str(exc)) from exc

    async def detach(self, generation: int) -> bool:
        detached = await self.ledger.detach(generation)
        if detached and self.ledger.terminal is None:
            async with self._lifecycle_lock:
                self._cancel_task("_grace_task")
                self._grace_task = asyncio.create_task(
                    self.registry.expire_detached(self, generation)
                )
        return detached

    async def append_text(self, generation: int, *, seq_no: int, text: str) -> InputAck:
        async with self._input_lock:
            await self._require_generation(generation)
            assert self.handle is not None
            return await self.handle.append_text(AppendText(seq_no=seq_no, text=text))

    async def complete_input(self, generation: int, *, final_seq_no: int) -> InputAck:
        async with self._input_lock:
            await self._require_generation(generation)
            assert self.handle is not None
            return await self.handle.complete_input(
                CompleteInput(final_seq_no=final_seq_no)
            )

    async def cancel(self, generation: int, reason: str = "") -> None:
        await self._require_generation(generation)
        assert self.handle is not None
        await self.handle.cancel(reason)

    async def acknowledge(
        self,
        generation: int,
        *,
        through_delivery_seq: int,
        audio_through_sample: int,
    ) -> None:
        try:
            await self.ledger.acknowledge(
                generation,
                through_delivery_seq=through_delivery_seq,
                audio_through_sample=audio_through_sample,
            )
        except LedgerError as exc:
            raise ResumableSessionError(exc.code, str(exc)) from exc

    async def terminal_ack(
        self,
        generation: int,
        *,
        through_delivery_seq: int,
        audio_through_sample: int,
    ) -> None:
        await self.acknowledge(
            generation,
            through_delivery_seq=through_delivery_seq,
            audio_through_sample=audio_through_sample,
        )
        try:
            await self.ledger.terminal_ack(generation)
        except LedgerError as exc:
            raise ResumableSessionError(exc.code, str(exc)) from exc
        await self.registry.remove(self, fence=False)

    async def record_playback_progress(
        self,
        generation: int,
        *,
        played_through_sample: int,
        buffered_through_sample: int,
        observed_delivery_seq: int,
    ) -> None:
        await self._require_generation(generation)
        if played_through_sample < 0 or buffered_through_sample < 0:
            raise ResumableSessionError(
                "invalid_playback_progress", "playback samples must be non-negative"
            )
        if buffered_through_sample < played_through_sample:
            raise ResumableSessionError(
                "invalid_playback_progress",
                "buffered_through_sample must be >= played_through_sample",
            )
        if (
            played_through_sample <= self.played_through_sample
            and buffered_through_sample <= self.buffered_through_sample
        ):
            return
        if (
            played_through_sample < self.played_through_sample
            or buffered_through_sample < self.buffered_through_sample
        ):
            raise ResumableSessionError(
                "invalid_playback_progress", "playback progress cannot move backwards"
            )
        try:
            observed_limit = await self.ledger.audio_end_through_delivery(
                observed_delivery_seq
            )
        except LedgerError as exc:
            raise ResumableSessionError(exc.code, str(exc)) from exc
        if buffered_through_sample > observed_limit:
            raise ResumableSessionError(
                "invalid_playback_progress",
                "buffered_through_sample is ahead of observed output",
            )
        self.played_through_sample = played_through_sample
        self.buffered_through_sample = buffered_through_sample

    def resume_info(self) -> dict[str, Any]:
        return {
            "acked_text_seq": self.accepted_text_seq,
            "input_closed": self.input_closed,
            "last_delivery_seq": self.ledger.last_delivery_seq,
            "audio_through_sample": self.ledger.next_audio_sample,
            **self.metadata,
        }

    async def close(self, *, fence: bool, cancel: bool) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._cancel_task("_grace_task")
            self._cancel_task("_terminal_task")
        handle = self.handle
        if cancel and handle is not None and handle.terminal is None:
            try:
                await handle.cancel("resume_session_expired")
            except Exception:
                pass
        await self.ledger.close(fence=fence)
        if self._pump_task is not None and not self._pump_task.done():
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
        await self.registry.service.close_session(self.internal_session_id)

    async def _pump_outputs(self) -> None:
        assert self.handle is not None
        try:
            async for output in self.handle.outputs():
                await self.ledger.publish(output)
                if self.output_observer is not None:
                    observed = self.output_observer(output)
                    if inspect.isawaitable(observed):
                        await observed
                if self.ledger.terminal is not None:
                    self.registry.schedule_terminal_expiry(self)
        except LedgerError:
            if self.handle.terminal is None:
                await self.handle.cancel("resume_buffer_exceeded")

    async def _require_generation(self, generation: int) -> None:
        try:
            await self.ledger.require_generation(generation)
        except LedgerError as exc:
            raise ResumableSessionError(exc.code, str(exc)) from exc

    def _cancel_task(self, name: str) -> None:
        task = getattr(self, name)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        setattr(self, name, None)


class ResumableSessionRegistry:
    """Process-local owner for reliable sessions shared by all wire adapters."""

    def __init__(
        self,
        service: SessionService,
        *,
        grace_seconds: float = 30.0,
        max_buffer_bytes: int = 16 * 1024 * 1024,
        attachment_queue_maxsize: int = 4096,
    ) -> None:
        self.service = service
        self.grace_seconds = max(0.0, float(grace_seconds))
        self.max_buffer_bytes = max(1, int(max_buffer_bytes))
        self.attachment_queue_maxsize = max(1, int(attachment_queue_maxsize))
        self._sessions: dict[str, ResumableLogicalSession] = {}
        self._lock = asyncio.Lock()

    async def claim_start(
        self,
        *,
        token: str,
        config_fingerprint: str,
        protocol: str,
        identity: "GatewaySessionIdentity",
        start_request: "SessionStartRequest",
        metadata: dict[str, Any] | None = None,
        output_observer: Callable[[Any], Awaitable[None] | None] | None = None,
    ) -> tuple[ResumableLogicalSession, bool]:
        if not token:
            raise ResumableSessionError(
                "invalid_resume_token", "resume token is required"
            )
        async with self._lock:
            current = self._sessions.get(token)
            if current is not None:
                if (
                    current.config_fingerprint != config_fingerprint
                    or current.protocol != protocol
                ):
                    raise ResumableSessionError(
                        "resume_token_conflict",
                        "resume token is already bound to another logical session",
                    )
                return current, False
            session = ResumableLogicalSession(
                self,
                token=token,
                config_fingerprint=config_fingerprint,
                protocol=protocol,
                identity=identity,
                start_request=start_request,
                metadata=metadata,
                output_observer=output_observer,
            )
            self._sessions[token] = session
        try:
            await session.initialize()
        except BaseException:
            await self.remove(session, fence=True, cancel=True)
            raise
        return session, True

    async def find(
        self, token: str, *, protocol: str | None = None
    ) -> ResumableLogicalSession:
        async with self._lock:
            session = self._sessions.get(token)
        if session is None or (protocol is not None and session.protocol != protocol):
            raise ResumableSessionError(
                "resume_session_not_found", "resumable stream is no longer available"
            )
        await session.wait_ready()
        return session

    async def remove(
        self,
        session: ResumableLogicalSession,
        *,
        fence: bool,
        cancel: bool = False,
    ) -> bool:
        async with self._lock:
            if self._sessions.get(session.token) is not session:
                return False
            del self._sessions[session.token]
        await session.close(fence=fence, cancel=cancel)
        return True

    async def expire_detached(
        self, session: ResumableLogicalSession, generation: int
    ) -> None:
        try:
            await asyncio.sleep(self.grace_seconds)
            if await session.ledger.has_attachment():
                return
            await self.remove(session, fence=False, cancel=True)
        except asyncio.CancelledError:
            return

    def schedule_terminal_expiry(self, session: ResumableLogicalSession) -> None:
        async def expire() -> None:
            try:
                await asyncio.sleep(self.grace_seconds)
                await self.remove(session, fence=True, cancel=False)
            except asyncio.CancelledError:
                return

        session._cancel_task("_terminal_task")
        session._terminal_task = asyncio.create_task(expire())

    async def close(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            await session.close(fence=True, cancel=True)


__all__ = [
    "ResumableLogicalSession",
    "ResumableSessionError",
    "ResumableSessionRegistry",
]
