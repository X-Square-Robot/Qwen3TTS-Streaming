from __future__ import annotations

import pytest

from engine.gateway.session_identity import GatewaySessionIdentity
from engine.session import (
    AppendText,
    AudioFormat,
    AudioOutput,
    CompleteInput,
    ExecutionHandle,
    SessionService,
    StartedOutput,
    TerminalOutput,
    TerminalStatus,
)


class _TraceExecution:
    def __init__(self, emit, trace):
        self.emit = emit
        self.trace = trace

    async def push_text(self, text: str) -> None:
        self.trace.append(("append_text", text))
        await self.emit(
            AudioOutput(
                "sid",
                b"\x00\x00" * 4,
                AudioFormat("pcm_s16le", 24000, 1),
                0,
                4,
            )
        )

    async def complete_input(self) -> None:
        self.trace.append(("complete_input",))
        await self.emit(TerminalOutput("sid", TerminalStatus.COMPLETED))

    async def cancel(self, reason: str = "") -> None:
        self.trace.append(("cancel", reason))

    async def close(self) -> None:
        self.trace.append(("close",))


class _TraceBackend:
    def __init__(self):
        self.trace: list[tuple] = []

    async def start(self, identity, *, start_request, emit) -> ExecutionHandle:
        self.trace.append(("start", identity.client_session_id))
        await emit(
            StartedOutput(
                identity.client_session_id,
                AudioFormat("pcm_s16le", 24000, 1),
            )
        )
        return _TraceExecution(emit, self.trace)

    async def close(self) -> None:
        self.trace.append(("backend_close",))


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment", ["standalone", "triton"])
async def test_same_canonical_trace_for_each_execution_deployment(deployment):
    del deployment  # Both deployments implement the same typed contract.
    backend = _TraceBackend()
    service = SessionService(backend)
    handle = await service.create(
        GatewaySessionIdentity.create("sid"),
        start_request=type("Start", (), {})(),
    )
    outputs = handle.outputs()
    assert isinstance(await anext(outputs), StartedOutput)
    await handle.append_text(AppendText(1, "hello"))
    await handle.complete_input(CompleteInput(1))
    assert isinstance(await anext(outputs), AudioOutput)
    assert isinstance(await anext(outputs), TerminalOutput)
    assert [entry[0] for entry in backend.trace[:3]] == [
        "start",
        "append_text",
        "complete_input",
    ]
    await service.close()
