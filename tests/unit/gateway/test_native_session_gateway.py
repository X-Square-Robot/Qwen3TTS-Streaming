from __future__ import annotations

import pytest

from engine.core.types import AudioConfig, AudioEncoding, SessionConfig
from engine.gateway.native_session_gateway import NativeSessionGateway
from engine.gateway.session_identity import GatewaySessionIdentity
from engine.interface import SessionStartRequest
from engine.session import (
    AudioFormat,
    AudioOutput,
    ExecutionHandle,
    SessionService,
    StartedOutput,
    TerminalOutput,
    TerminalStatus,
)


class _Handle:
    def __init__(self, emit):
        self.emit = emit

    async def push_text(self, text: str) -> None:
        del text
        await self.emit(
            AudioOutput(
                session_id="native-1",
                pcm_bytes=b"\x00\x00\x80?",
                audio=AudioFormat("pcm_f32", 24000, 1),
                output_sample_start=0,
                output_sample_end=1,
            )
        )

    async def complete_input(self) -> None:
        await self.emit(
            TerminalOutput(session_id="native-1", status=TerminalStatus.COMPLETED)
        )

    async def cancel(self, reason: str = "") -> None:
        await self.emit(
            TerminalOutput(
                session_id="native-1",
                status=TerminalStatus.CANCELLED,
                message=reason,
            )
        )

    async def close(self) -> None:
        return None


class _Backend:
    def __init__(self):
        self.start_count = 0

    async def start(self, identity, *, start_request, emit) -> ExecutionHandle:
        self.start_count += 1
        await emit(
            StartedOutput(
                session_id=identity.client_session_id,
                audio=AudioFormat.from_config(start_request.config.audio),
            )
        )
        return _Handle(emit)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_native_gateway_projects_typed_session_to_binary_frames():
    pytest.importorskip("aiohttp")
    from aiohttp import WSMsgType, web
    from aiohttp.test_utils import TestClient, TestServer

    service = SessionService(_Backend())
    gateway = NativeSessionGateway(service)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json(
                {
                    "type": "start",
                    "session_id": "native-1",
                    "config": {
                        "audio": {
                            "encoding": "pcm_f32",
                            "sample_rate": 24000,
                            "channels": 1,
                        }
                    },
                }
            )
            start = await ws.receive(timeout=1)
            assert start.type == WSMsgType.TEXT
            assert start.json()["event"]["type"] == "start"

            await ws.send_json({"type": "text", "seq_no": 1, "text": "hello"})
            observed_binary = False
            observed_header = False
            observed_ack = False
            for _ in range(3):
                message = await ws.receive(timeout=1)
                if message.type == WSMsgType.BINARY:
                    observed_binary = message.data == b"\x00\x00\x80?"
                else:
                    payload = message.json()
                    observed_header |= payload["type"] == "audio_header"
                    observed_ack |= payload["type"] == "text_ack"
            assert observed_binary and observed_header and observed_ack

            await ws.send_json({"type": "stop", "final_seq_no": 1})
            terminal = None
            for _ in range(2):
                message = await ws.receive(timeout=1)
                if message.type == WSMsgType.TEXT:
                    payload = message.json()
                    if payload.get("event", {}).get("type") == "done":
                        terminal = payload
            assert terminal is not None
            assert terminal["event"]["session_id"] == "native-1"

            await ws.close()
    await service.close()


def test_native_gateway_backend_contract_is_constructible():
    config = SessionConfig(audio=AudioConfig(encoding=AudioEncoding.PCM_F32))
    request = SessionStartRequest(session_id="x", config=config)
    assert request.config.audio.sample_rate == 24000
    assert GatewaySessionIdentity.create("x").client_session_id == "x"


@pytest.mark.asyncio
async def test_native_gateway_preserves_unsequenced_legacy_frames():
    pytest.importorskip("aiohttp")
    from aiohttp import WSMsgType, web
    from aiohttp.test_utils import TestClient, TestServer

    service = SessionService(_Backend())
    gateway = NativeSessionGateway(service)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json({"type": "start", "session_id": "legacy"})
            assert (await ws.receive_json(timeout=1))["event"]["type"] == "start"

            await ws.send_json({"type": "text", "text": "hello"})
            header = await ws.receive_json(timeout=1)
            assert header["type"] == "audio_header"
            assert (await ws.receive(timeout=1)).type == WSMsgType.BINARY

            await ws.send_json({"type": "end"})
            terminal = await ws.receive_json(timeout=1)
            assert terminal["event"]["type"] == "done"
            await ws.close()

    await gateway.close()
    await service.close()


@pytest.mark.asyncio
async def test_native_gateway_resumes_same_typed_session_and_replays_exact_delivery():
    pytest.importorskip("aiohttp")
    from aiohttp import WSMsgType, web
    from aiohttp.test_utils import TestClient, TestServer

    backend = _Backend()
    service = SessionService(backend)
    gateway = NativeSessionGateway(service, stream_resume_grace_seconds=1.0)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)
    token = "native-resume-token-0001"

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json(
                {
                    "type": "start",
                    "session_id": "native-1",
                    "resume": {"enabled": True, "token": token},
                    "config": {
                        "audio": {
                            "encoding": "pcm_f32",
                            "sample_rate": 24000,
                            "channels": 1,
                        }
                    },
                }
            )
            started = await ws.receive_json(timeout=1)
            assert started["delivery_seq"] == 1
            await ws.send_json({"type": "text", "seq_no": 1, "text": "hello"})
            assert (await ws.receive_json(timeout=1))["type"] == "text_ack"
            header = await ws.receive_json(timeout=1)
            assert header["delivery_seq"] == 2
            assert (header["start_sample"], header["end_sample"]) == (0, 1)
            assert (await ws.receive(timeout=1)).type == WSMsgType.BINARY
            await ws.close()

            resumed_ws = await client.ws_connect("/v1/ws")
            await resumed_ws.send_json(
                {
                    "type": "resume",
                    "token": token,
                    "last_delivery_seq": 1,
                    "audio_through_sample": 0,
                }
            )
            resumed = await resumed_ws.receive_json(timeout=1)
            assert resumed["type"] == "resumed"
            assert resumed["acked_text_seq"] == 1
            replay_header = await resumed_ws.receive_json(timeout=1)
            assert replay_header["delivery_seq"] == 2
            assert (await resumed_ws.receive(timeout=1)).type == WSMsgType.BINARY
            assert backend.start_count == 1

            await resumed_ws.send_json({"type": "text", "seq_no": 1, "text": "hello"})
            duplicate = await resumed_ws.receive_json(timeout=1)
            assert duplicate["duplicate"] is True
            await resumed_ws.send_json({"type": "stop", "final_seq_no": 1})
            assert (await resumed_ws.receive_json(timeout=1))["type"] == "input_ack"
            terminal = await resumed_ws.receive_json(timeout=1)
            assert terminal["event"]["type"] == "done"
            await resumed_ws.send_json(
                {
                    "type": "terminal_ack",
                    "through_delivery_seq": terminal["delivery_seq"],
                    "audio_through_sample": 1,
                }
            )
            await resumed_ws.close()

    await gateway.close()
    await service.close()
