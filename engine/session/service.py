"""Session lifecycle and input idempotency independent of any wire protocol."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

from .types import SessionOutput, TerminalOutput, TerminalStatus

if TYPE_CHECKING:
    from ..gateway.session_identity import GatewaySessionIdentity
    from ..interface import SessionStartRequest


class SessionProtocolError(ValueError):
    """A stable error raised for an invalid logical-session command."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class AppendText:
    seq_no: int
    text: str
    client_timestamp_ms: int = 0


@dataclass(frozen=True, slots=True)
class CompleteInput:
    final_seq_no: int
    client_timestamp_ms: int = 0


@dataclass(frozen=True, slots=True)
class InputAck:
    seq_no: int
    duplicate: bool = False


class ExecutionHandle(Protocol):
    async def push_text(self, text: str) -> None: ...

    async def complete_input(self) -> None: ...

    async def cancel(self, reason: str = "") -> None: ...

    async def close(self) -> None: ...


OutputSink = Callable[[SessionOutput], Awaitable[None]]


class ExecutionBackend(Protocol):
    async def start(
        self,
        identity: "GatewaySessionIdentity",
        *,
        start_request: "SessionStartRequest",
        emit: OutputSink,
    ) -> ExecutionHandle: ...

    async def close(self) -> None: ...


class SessionHandle:
    """One logical response, shared by successive physical attachments."""

    def __init__(
        self,
        service: "SessionService",
        identity: "GatewaySessionIdentity",
        start_request: "SessionStartRequest",
    ) -> None:
        self.service = service
        self.identity = identity
        self.start_request = start_request
        self._outputs: asyncio.Queue[SessionOutput] = asyncio.Queue(
            maxsize=service.output_queue_maxsize
        )
        self._input_lock = asyncio.Lock()
        self._terminal_lock = asyncio.Lock()
        self._execution: ExecutionHandle | None = None
        self._accepted_text: dict[int, str] = {}
        self._next_seq = 1
        self._input_complete = False
        self._terminal: TerminalOutput | None = None
        self._started = asyncio.Event()
        self._closed = False

    @property
    def internal_session_id(self) -> str:
        return self.identity.internal_session_id

    @property
    def client_session_id(self) -> str:
        return self.identity.client_session_id

    @property
    def terminal(self) -> TerminalOutput | None:
        return self._terminal

    async def start(self) -> None:
        if self._execution is not None:
            return
        self._execution = await self.service.backend.start(
            self.identity,
            start_request=self.start_request,
            emit=self._emit,
        )
        self._started.set()

    async def wait_started(self) -> None:
        await self._started.wait()

    async def append_text(self, command: AppendText) -> InputAck:
        if command.seq_no <= 0:
            raise SessionProtocolError("invalid_text_seq", "seq_no must be positive")
        if not command.text:
            raise SessionProtocolError("empty_text", "text must not be empty")
        async with self._input_lock:
            self._ensure_input_open()
            previous = self._accepted_text.get(command.seq_no)
            if previous is not None:
                if previous != command.text:
                    raise SessionProtocolError(
                        "text_seq_conflict", "same seq_no was used for different text"
                    )
                return InputAck(command.seq_no, duplicate=True)
            if command.seq_no != self._next_seq:
                raise SessionProtocolError(
                    "text_seq_gap",
                    f"expected seq_no {self._next_seq}, got {command.seq_no}",
                )
            assert self._execution is not None
            self._accepted_text[command.seq_no] = command.text
            self._next_seq += 1
            await self._execution.push_text(command.text)
            return InputAck(command.seq_no)

    async def complete_input(self, command: CompleteInput) -> InputAck:
        if command.final_seq_no < 0:
            raise SessionProtocolError(
                "invalid_final_seq", "final_seq_no must be non-negative"
            )
        async with self._input_lock:
            if self._input_complete:
                if command.final_seq_no != self._next_seq - 1:
                    raise SessionProtocolError(
                        "final_seq_conflict", "input was already completed"
                    )
                return InputAck(command.final_seq_no, duplicate=True)
            self._ensure_input_open()
            accepted = self._next_seq - 1
            if command.final_seq_no != accepted:
                raise SessionProtocolError(
                    "final_seq_mismatch",
                    f"expected final_seq_no {accepted}, got {command.final_seq_no}",
                )
            assert self._execution is not None
            self._input_complete = True
            await self._execution.complete_input()
            return InputAck(command.final_seq_no)

    async def cancel(self, reason: str = "") -> None:
        async with self._input_lock:
            if self._terminal is not None:
                return
            self._input_complete = True
            execution = self._execution
        if execution is not None:
            await execution.cancel(reason)
        await self._emit_terminal(
            TerminalOutput(
                session_id=self.client_session_id,
                status=TerminalStatus.CANCELLED,
                message=reason,
                metrics={"cancel_reason": reason} if reason else {},
            )
        )

    async def outputs(self):
        """Yield outputs until (and including) the unique terminal output."""
        while True:
            output = await self._outputs.get()
            yield output
            if isinstance(output, TerminalOutput):
                return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        execution = self._execution
        if execution is not None:
            await execution.close()

    async def _emit(self, output: SessionOutput) -> None:
        if isinstance(output, TerminalOutput):
            await self._emit_terminal(output)
            return
        if self._terminal is not None or self._closed:
            return
        await self._outputs.put(output)

    async def _emit_terminal(self, output: TerminalOutput) -> None:
        async with self._terminal_lock:
            if self._terminal is not None:
                return
            self._terminal = output
            await self._outputs.put(output)

    def _ensure_input_open(self) -> None:
        if self._terminal is not None:
            raise SessionProtocolError("session_terminal", "session is terminal")
        if self._input_complete:
            raise SessionProtocolError("input_already_complete", "input is complete")
        if self._execution is None:
            raise SessionProtocolError("session_not_started", "session is not started")


class SessionService:
    """Create and own logical sessions over a transport-neutral backend."""

    def __init__(
        self,
        backend: ExecutionBackend,
        *,
        output_queue_maxsize: int = 4096,
    ) -> None:
        if output_queue_maxsize <= 0:
            raise ValueError("output_queue_maxsize must be positive")
        self.backend = backend
        self.output_queue_maxsize = output_queue_maxsize
        self._sessions: dict[str, SessionHandle] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        identity: "GatewaySessionIdentity",
        *,
        start_request: "SessionStartRequest",
    ) -> SessionHandle:
        async with self._lock:
            if identity.internal_session_id in self._sessions:
                raise SessionProtocolError(
                    "duplicate_session", "internal session already exists"
                )
            handle = SessionHandle(self, identity, start_request)
            self._sessions[identity.internal_session_id] = handle
        try:
            await handle.start()
        except BaseException:
            async with self._lock:
                self._sessions.pop(identity.internal_session_id, None)
            raise
        return handle

    async def get(self, internal_session_id: str) -> SessionHandle:
        async with self._lock:
            handle = self._sessions.get(internal_session_id)
        if handle is None:
            raise SessionProtocolError("session_not_found", "session was not found")
        return handle

    async def close_session(self, internal_session_id: str) -> None:
        async with self._lock:
            handle = self._sessions.pop(internal_session_id, None)
        if handle is not None:
            await handle.close()

    async def close(self) -> None:
        async with self._lock:
            handles = list(self._sessions.values())
            self._sessions.clear()
        for handle in handles:
            await handle.close()
        await self.backend.close()

