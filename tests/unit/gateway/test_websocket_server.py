from __future__ import annotations

import asyncio
import base64
import json

import pytest

from engine.core.types import AudioEncoding, GroupPolicy, InputMode
from engine.gateway.websocket_server import (
    WebSocketGateway,
    _normalize_ws_path,
    _start_request_from_ws_message,
    _session_config_from_ws_message,
)


def test_session_config_from_ws_message_maps_json_fields():
    request = {
        "type": "start",
        "session_id": "sid-ws",
        "config": {
            "task_type": "custom_voice",
            "speaker": "Serena",
            "input_mode": "token",
            "group_policy": "none",
            "ref_audio": base64.b64encode(b"wav").decode(),
            "audio": {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
            },
        },
    }

    cfg = _session_config_from_ws_message(request, default_mode=InputMode.LONG_SEGMENT)

    assert cfg.task_type == "custom_voice"
    assert cfg.speaker == "Serena"
    assert cfg.input_mode == InputMode.TOKEN
    assert cfg.group_policy == GroupPolicy.NONE
    assert cfg.ref_audio == b"wav"
    assert cfg.audio.encoding == AudioEncoding.PCM_S16LE
    assert cfg.audio.sample_rate == 16000


def test_session_config_from_ws_message_supports_top_level_legacy_fields():
    request = {
        "type": "oneshot",
        "task_type": "voice_design",
        "instruct": "warm and calm",
        "audio": {
            "encoding": 1,
            "sample_rate": 24000,
            "channels": 1,
        },
    }

    cfg = _session_config_from_ws_message(request, default_mode=InputMode.FULL_TEXT)

    assert cfg.task_type == "voice_design"
    assert cfg.instruct == "warm and calm"
    assert cfg.input_mode == InputMode.FULL_TEXT
    assert cfg.group_policy == GroupPolicy.AUTO
    assert cfg.audio.encoding == AudioEncoding.PCM_F32


def test_start_request_from_ws_message_round_trips_output_policy_and_timing():
    request = {
        "type": "start",
        "session_id": "sid-contract",
        "config": {
            "task_type": "custom_voice",
            "protocol_version": "tts-session-v2alpha1",
            "output_policy": {
                "vad_policy": {
                    "enabled": True,
                    "strategy": "tail_guard",
                    "implementation": "tenvad",
                    "config": {"max_non_speech_frames": 8},
                },
                "chunk_ms": 16,
                "packet_format": "raw_pcm",
                "emit_text_events": False,
                "config": {"transport": "websocket"},
            },
            "timing_context": {
                "request_id": "req-ws",
                "turn_id": "turn-ws",
                "client_request_ts_ms": 1710000000000,
                "extra": {"client_clock": "synced"},
            },
        },
    }

    start = _start_request_from_ws_message(request, default_mode=InputMode.LONG_SEGMENT)

    assert start.output_policy.vad.enabled is True
    assert start.output_policy.vad.strategy == "tail_guard"
    assert start.output_policy.vad.implementation == "tenvad"
    assert start.output_policy.emit_text_events is False
    assert start.timing.request_id == "req-ws"
    assert start.timing.turn_id == "turn-ws"
    assert start.timing.extra["client_clock"] == "synced"
    assert start.timing.extra["client_protocol_version"] == "tts-session-v2alpha1"
    assert start.config.output_policy.vad.strategy == "tail_guard"
    assert start.config.timing.request_id == "req-ws"


def test_start_request_from_ws_message_accepts_top_level_vad_policy_alias():
    request = {
        "type": "start",
        "session_id": "sid-vad-alias",
        "task_type": "custom_voice",
        "vad_policy": {
            "enabled": True,
            "strategy": "prefix_trim",
            "implementation": "energy",
        },
    }

    start = _start_request_from_ws_message(request, default_mode=InputMode.LONG_SEGMENT)

    assert start.output_policy.vad.enabled is True
    assert start.output_policy.vad.strategy == "prefix_trim"
    assert start.config.output_policy.vad.implementation == "energy"


def test_normalize_ws_path_adds_leading_slash():
    assert _normalize_ws_path("stream/ws") == "/stream/ws"
    assert _normalize_ws_path("/v1/ws") == "/v1/ws"


@pytest.mark.asyncio
async def test_websocket_gateway_streams_audio_and_events_when_aiohttp_available():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    class _StubEngine:
        def __init__(self):
            self._on_audio = None
            self._on_done = None
            self.cancel_calls = []

        def describe_capabilities(self):
            return {
                "variant": "custom-1.7b",
                "loaded_model_type": "custom_voice",
            }

        async def start_session(
            self, session_id, *, config, on_audio=None, on_done=None, on_event=None
        ):
            self._on_audio = on_audio
            self._on_done = on_done
            return session_id

        async def push_text_input(self, session_id, text):
            await self._on_audio(session_id, b"\x00\x00\x00\x00")

        async def mark_input_complete(self, session_id):
            await self._on_done(session_id, {})

        async def cancel(self, session_id):
            self.cancel_calls.append(session_id)

    gateway = WebSocketGateway(_StubEngine())
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
                    "session_id": "sid-stream",
                    "config": {
                        "task_type": "custom_voice",
                        "speaker": "Serena",
                    },
                }
            )
            await ws.send_json({"type": "text", "text": "你好"})
            await ws.send_json({"type": "end"})

            first = await ws.receive(timeout=0.2)
            second = await ws.receive(timeout=0.2)
            third = await ws.receive(timeout=0.2)

            assert first.type == aiohttp.WSMsgType.TEXT
            first_event = json.loads(first.data)["event"]
            assert first_event["type"] == "start"
            assert first_event["meta"]["protocol_version"] == "tts-session-v2alpha1"
            assert first_event["meta"]["vad_enabled"] == "false"
            assert second.type == aiohttp.WSMsgType.BINARY
            assert second.data == b"\x00\x00\x00\x00"
            assert third.type == aiohttp.WSMsgType.TEXT
            third_event = json.loads(third.data)["event"]
            assert third_event["type"] == "done"
            assert third_event["meta"]["timing_contract"] == "server_monotonic_v1"
            assert third_event["meta"]["audio_chunk_count"] == "1"

            await ws.close()


class _ReusableStubEngine:
    """Small callback-driven engine for websocket connection-lifecycle tests."""

    def __init__(self):
        self.callbacks = {}
        self.cancel_calls = []
        self.started_sessions = []

    def describe_capabilities(self):
        return {
            "variant": "custom-1.7b",
            "loaded_model_type": "custom_voice",
        }

    async def start_session(
        self, session_id, *, config, on_audio=None, on_done=None, on_event=None
    ):
        self.started_sessions.append(session_id)
        self.callbacks[session_id] = {
            "on_audio": on_audio,
            "on_done": on_done,
            "on_event": on_event,
        }
        return session_id

    async def push_text_input(self, session_id, text):
        # One float32 sample; its bytes make cross-session leakage observable.
        marker = b"\x00\x00\x80?" if session_id.endswith("2") else b"\x00\x00\x00?"
        await self.callbacks[session_id]["on_audio"](session_id, marker)

    async def mark_input_complete(self, session_id):
        await self.callbacks[session_id]["on_done"](session_id, {})

    async def cancel(self, session_id):
        self.cancel_calls.append(session_id)


class _YieldingStubEngine(_ReusableStubEngine):
    """Force queue-consumer scheduling races at callback yield boundaries."""

    async def push_text_input(self, session_id, text):
        await self.callbacks[session_id]["on_audio"](session_id, b"\x00\x00\x00?")
        await asyncio.sleep(0)
        await self.callbacks[session_id]["on_audio"](session_id, b"\x00\x00\x80?")

    async def mark_input_complete(self, session_id):
        await self.callbacks[session_id]["on_audio"](session_id, b"\x00\x00\x00\xbf")
        await asyncio.sleep(0)
        await self.callbacks[session_id]["on_done"](session_id, {})


@pytest.mark.asyncio
async def test_websocket_reuses_connection_for_serial_sessions_and_stop_alias():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")

            await ws.send_json({"type": "get_capabilities"})
            capabilities = json.loads((await ws.receive(timeout=1.0)).data)
            assert capabilities["type"] == "capabilities"
            assert capabilities["websocket_connection_reusable"] is True

            await ws.send_json({"type": "start", "session_id": "serial-1"})
            await ws.send_json({"type": "text", "text": "first"})
            await ws.send_json({"type": "end"})

            first_start = await ws.receive(timeout=1.0)
            first_audio = await ws.receive(timeout=1.0)
            first_done = await ws.receive(timeout=1.0)
            assert json.loads(first_start.data)["event"]["type"] == "start"
            assert first_audio.type == aiohttp.WSMsgType.BINARY
            first_done_event = json.loads(first_done.data)["event"]
            assert first_done_event["type"] == "done"
            assert first_done_event["meta"]["websocket_connection_reusable"] == "true"
            assert not ws.closed

            # The same physical websocket accepts another start after the first
            # terminal event. ``stop`` is the graceful ``end`` alias.
            await ws.send_json({"type": "start", "session_id": "serial-2"})
            await ws.send_json({"type": "text", "text": "second"})
            await ws.send_json({"type": "stop"})

            second_start = await ws.receive(timeout=1.0)
            second_audio = await ws.receive(timeout=1.0)
            second_done = await ws.receive(timeout=1.0)
            assert json.loads(second_start.data)["event"]["session_id"] == "serial-2"
            assert second_audio.type == aiohttp.WSMsgType.BINARY
            assert json.loads(second_done.data)["event"]["type"] == "done"
            assert engine.started_sessions == ["serial-1", "serial-2"]
            assert engine.cancel_calls == []

            await ws.close()


@pytest.mark.asyncio
async def test_outbound_audio_keeps_callback_enqueue_order_across_yield():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    gateway = WebSocketGateway(_YieldingStubEngine())
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json({"type": "start", "session_id": "ordered"})
            assert (
                json.loads((await ws.receive(timeout=1.0)).data)["event"]["type"]
                == "start"
            )

            await ws.send_json({"type": "text", "text": "trigger"})
            audio = await ws.receive(timeout=1.0)
            assert audio.type == aiohttp.WSMsgType.BINARY
            # Backlogged chunks may be coalesced, but their byte order must
            # remain identical to callback enqueue order.
            assert audio.data == b"\x00\x00\x00?\x00\x00\x80?"

            await ws.send_json({"type": "cancel"})
            assert (
                json.loads((await ws.receive(timeout=1.0)).data)["event"]["type"]
                == "done"
            )


@pytest.mark.asyncio
async def test_stop_never_allows_done_to_overtake_final_audio():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    gateway = WebSocketGateway(_YieldingStubEngine())
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json({"type": "start", "session_id": "final-order"})
            assert (
                json.loads((await ws.receive(timeout=1.0)).data)["event"]["type"]
                == "start"
            )

            await ws.send_json({"type": "stop"})
            final_audio = await ws.receive(timeout=1.0)
            done = await ws.receive(timeout=1.0)
            assert final_audio.type == aiohttp.WSMsgType.BINARY
            assert final_audio.data == b"\x00\x00\x00\xbf"
            assert json.loads(done.data)["event"]["type"] == "done"


@pytest.mark.asyncio
async def test_cancel_emits_done_keeps_connection_and_isolates_old_queue():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")

            await ws.send_json({"type": "start", "session_id": "cancel-1"})
            started = await ws.receive(timeout=1.0)
            assert json.loads(started.data)["event"]["type"] == "start"

            await ws.send_json(
                {"type": "cancel", "reason": "caller interrupted playback"}
            )
            terminal = await ws.receive(timeout=1.0)
            terminal_event = json.loads(terminal.data)["event"]
            assert terminal_event["type"] == "done"
            assert terminal_event["session_id"] == "cancel-1"
            assert terminal_event["meta"]["terminal_reason"] == "cancelled"
            assert (
                terminal_event["meta"]["cancel_reason"] == "caller interrupted playback"
            )
            assert terminal_event["meta"]["websocket_connection_reusable"] == "true"
            assert engine.cancel_calls == ["cancel-1"]
            assert not ws.closed

            # Simulate an engine callback racing after cancellation. It still
            # targets cancel-1's detached queue and must never appear in the
            # next logical session on this connection.
            await engine.callbacks["cancel-1"]["on_audio"]("cancel-1", b"\x00\x00\x10A")

            await ws.send_json({"type": "start", "session_id": "cancel-2"})
            second_start = await ws.receive(timeout=1.0)
            assert json.loads(second_start.data)["event"]["session_id"] == "cancel-2"

            await ws.send_json({"type": "text", "text": "new session"})
            second_audio = await ws.receive(timeout=1.0)
            assert second_audio.type == aiohttp.WSMsgType.BINARY
            assert second_audio.data == b"\x00\x00\x80?"

            await ws.send_json({"type": "stop"})
            second_done = await ws.receive(timeout=1.0)
            assert json.loads(second_done.data)["event"]["type"] == "done"

            await ws.close()


@pytest.mark.asyncio
async def test_engine_error_terminal_closes_connection_without_reuse_marker():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")

            await ws.send_json({"type": "start", "session_id": "failed-1"})
            first_start = await ws.receive(timeout=1.0)
            assert json.loads(first_start.data)["event"]["type"] == "start"

            await engine.callbacks["failed-1"]["on_done"](
                "failed-1", {"error": "synthetic engine failure"}
            )
            failed = await ws.receive(timeout=1.0)
            failed_event = json.loads(failed.data)["event"]
            assert failed_event["type"] == "error"
            assert failed_event["message"] == "synthetic engine failure"
            assert "websocket_connection_reusable" not in failed_event["meta"]
            closed = await ws.receive(timeout=1.0)
            assert closed.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
            }


@pytest.mark.asyncio
async def test_engine_error_drops_already_queued_text_before_connection_close():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json({"type": "start", "session_id": "failed-race"})
            assert (
                json.loads((await ws.receive(timeout=1.0)).data)["event"]["type"]
                == "start"
            )

            # Queue an engine terminal and an old-session text frame in the
            # same event-loop turn. The client must see exactly the engine
            # error followed by close, never a second protocol error and never
            # a reusable marker.
            await engine.callbacks["failed-race"]["on_done"](
                "failed-race", {"error": "boom"}
            )
            await ws.send_json({"type": "text", "text": "too late"})

            failed = json.loads((await ws.receive(timeout=1.0)).data)["event"]
            assert failed["type"] == "error"
            assert failed["message"] == "boom"
            assert "websocket_connection_reusable" not in failed["meta"]
            closed = await ws.receive(timeout=1.0)
            assert closed.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
            }


@pytest.mark.asyncio
async def test_official_sdk_reuses_one_real_websocket_for_two_sessions():
    """Exercise the SDK pool and gateway lifecycle together over a real socket."""
    pytest.importorskip("websocket")
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from qwen3tts import SessionStartRequest, SynthesisConfig
    from qwen3tts._adapters.engine_websocket import EngineWebSocketAdapter

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine)
    handshake_count = 0

    async def counted_handler(request):
        nonlocal handshake_count
        handshake_count += 1
        return await gateway.handle_websocket(request)

    app = web.Application()
    app.router.add_get("/v1/ws", counted_handler)

    server = TestServer(app)
    async with server:
        ws_url = str(server.make_url("/v1/ws")).replace("http://", "ws://", 1)

        def run_two_sessions():
            adapter = EngineWebSocketAdapter(
                ws_url,
                timeout=2.0,
                connect_timeout=2.0,
            )
            observed = []
            try:
                for session_id in ("sdk-1", "sdk-2"):
                    session = adapter.open_stream(
                        SessionStartRequest(
                            session_id=session_id,
                            config=SynthesisConfig(),
                        )
                    )
                    session.send_text("hello")
                    session.stop()
                    observed.append(
                        [
                            getattr(message, "type", "audio")
                            for message in session.iter_messages()
                        ]
                    )
            finally:
                adapter.close()
            return observed

        observed = await asyncio.to_thread(run_two_sessions)

    assert handshake_count == 1
    assert observed == [
        ["start", "audio", "done"],
        ["start", "audio", "done"],
    ]
