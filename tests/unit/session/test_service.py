from __future__ import annotations

import pytest

from engine.core.types import AudioConfig
from engine.gateway.session_identity import GatewaySessionIdentity
from engine.session import (
    AppendText,
    AudioFormat,
    CompleteInput,
    EventOutput,
    ExecutionHandle,
    SessionProtocolError,
    SessionService,
    StartedOutput,
    TerminalOutput,
    TerminalStatus,
)


class _Handle:
    def __init__(self, emit):
        self.emit = emit
        self.text: list[str] = []
        self.completed = 0
        self.cancelled = 0
        self.closed = 0

    async def push_text(self, text: str) -> None:
        self.text.append(text)

    async def complete_input(self) -> None:
        self.completed += 1

    async def cancel(self, reason: str = "") -> None:
        self.cancelled += 1
        await self.emit(
            TerminalOutput(
                session_id="sid",
                status=TerminalStatus.CANCELLED,
                message=reason,
            )
        )

    async def close(self) -> None:
        self.closed += 1


class _Backend:
    def __init__(self):
        self.handle: _Handle | None = None

    async def start(self, identity, *, start_request, emit) -> ExecutionHandle:
        self.handle = _Handle(emit)
        await emit(
            StartedOutput(
                session_id=identity.client_session_id,
                audio=AudioFormat.from_config(AudioConfig()),
            )
        )
        return self.handle

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_session_service_enforces_order_and_idempotency():
    backend = _Backend()
    service = SessionService(backend)
    identity = GatewaySessionIdentity.create("sid")
    handle = await service.create(identity, start_request=type("Start", (), {})())

    first = await anext(handle.outputs())
    assert isinstance(first, StartedOutput)

    accepted = await handle.append_text(AppendText(1, "hello"))
    duplicate = await handle.append_text(AppendText(1, "hello"))
    assert accepted.duplicate is False
    assert duplicate.duplicate is True
    assert backend.handle is not None
    assert backend.handle.text == ["hello"]

    with pytest.raises(SessionProtocolError, match="expected seq_no 2"):
        await handle.append_text(AppendText(3, "gap"))
    with pytest.raises(SessionProtocolError, match="different text"):
        await handle.append_text(AppendText(1, "conflict"))

    await handle.complete_input(CompleteInput(1))
    assert backend.handle.completed == 1
    duplicate = await handle.complete_input(CompleteInput(1))
    assert duplicate.duplicate is True
    with pytest.raises(SessionProtocolError, match="input is complete"):
        await handle.append_text(AppendText(2, "late"))


@pytest.mark.asyncio
async def test_session_service_emits_one_terminal_and_cancel_is_idempotent():
    backend = _Backend()
    service = SessionService(backend)
    identity = GatewaySessionIdentity.create("sid")
    handle = await service.create(identity, start_request=type("Start", (), {})())
    outputs = handle.outputs()
    assert isinstance(await anext(outputs), StartedOutput)

    await handle.cancel("client_cancelled")
    terminal = await anext(outputs)
    assert isinstance(terminal, TerminalOutput)
    assert terminal.status is TerminalStatus.CANCELLED
    await handle.cancel("duplicate_cancel")
    assert backend.handle.cancelled == 1

    # A late backend callback cannot add a second terminal or any post-terminal
    # output to the logical session.
    await backend.handle.emit(
        EventOutput(session_id="sid", event_type="late", meta={"x": "1"})
    )
    with pytest.raises(StopAsyncIteration):
        await anext(outputs)
