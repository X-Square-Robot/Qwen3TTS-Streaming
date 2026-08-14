from __future__ import annotations

import base64
import json

import pytest

from engine.session import (
    AudioFormat,
    AudioOutput,
    EventOutput,
    ExecutionHandle,
    SessionService,
    StartedOutput,
    TerminalOutput,
    TerminalStatus,
)
from engine.gateway.openai_realtime import OpenAIRealtimeGateway
from engine.gateway.session_backend import (
    RealtimeSessionServiceBackend,
    StandaloneSessionBackend,
    TritonSessionBackend,
)


class _Execution:
    def __init__(self, emit):
        self.emit = emit

    async def push_text(self, text: str) -> None:
        await self.emit(
            AudioOutput(
                session_id="response",
                pcm_bytes=b"\x00\x00" * 1200,
                audio=AudioFormat("pcm_s16le", 24000, 1),
                output_sample_start=0,
                output_sample_end=1200,
            )
        )
        await self.emit(
            EventOutput(
                session_id="response",
                event_type="text_progress",
                segment_id=0,
                meta={"text_progress": "1.0"},
            )
        )

    async def complete_input(self) -> None:
        await self.emit(
            TerminalOutput(session_id="response", status=TerminalStatus.COMPLETED)
        )

    async def cancel(self, reason: str = "") -> None:
        await self.emit(
            TerminalOutput(
                session_id="response",
                status=TerminalStatus.CANCELLED,
                message=reason,
            )
        )

    async def close(self) -> None:
        return None


class _Backend:
    async def start(self, identity, *, start_request, emit) -> ExecutionHandle:
        await emit(
            StartedOutput(
                session_id=identity.client_session_id,
                audio=AudioFormat("pcm_s16le", 24000, 1),
            )
        )
        return _Execution(emit)

    async def close(self) -> None:
        return None

    def count_text_tokens(self, text: str) -> int:
        return len(text) + 10


async def _receive_json(ws, *, timeout: float = 1.0) -> dict:
    message = await ws.receive(timeout=timeout)
    assert message.type.name == "TEXT"
    return json.loads(message.data)


@pytest.mark.asyncio
async def test_realtime_uses_typed_session_service_outputs():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    service = SessionService(_Backend())
    gateway = OpenAIRealtimeGateway(None, session_service=service)
    app = web.Application()
    app.router.add_get("/v1/realtime", gateway.handle_websocket)
    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/realtime?model=test")
            created = await _receive_json(ws)
            assert created["type"] == "session.created"
            await ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    },
                }
            )
            assert (await _receive_json(ws))["type"] == "conversation.item.added"
            assert (await _receive_json(ws))["type"] == "conversation.item.done"
            await ws.send_json({"type": "response.create"})

            seen: list[dict] = []
            while True:
                event = await _receive_json(ws)
                seen.append(event)
                if event["type"] == "response.done":
                    break
            assert any(event["type"] == "response.output_audio.delta" for event in seen)
            delta = next(
                event
                for event in seen
                if event["type"] == "response.output_audio.delta"
            )
            assert len(base64.b64decode(delta["delta"])) == 2400
            assert any(event["type"] == "qwen.text_progress" for event in seen)
            assert seen[-1]["type"] == "response.done"
            assert seen[-1]["response"]["usage"]["input_tokens"] == 15
            await ws.close()

    await gateway.close()


def test_session_backend_token_counters_are_forwarded_without_character_fallback():
    class TokenOwner:
        def count_text_tokens(self, text: str) -> int:
            assert text == "hello"
            return 37

        async def close(self) -> None:
            return None

    standalone = StandaloneSessionBackend(None, TokenOwner())
    triton = TritonSessionBackend(TokenOwner())

    assert (
        RealtimeSessionServiceBackend(SessionService(standalone)).count_text_tokens(
            "hello"
        )
        == 37
    )
    assert (
        RealtimeSessionServiceBackend(SessionService(triton)).count_text_tokens("hello")
        == 37
    )


def test_realtime_session_backend_rejects_missing_tokenizer_counter():
    class MissingCounter:
        async def close(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="count_text_tokens"):
        RealtimeSessionServiceBackend(SessionService(MissingCounter()))
