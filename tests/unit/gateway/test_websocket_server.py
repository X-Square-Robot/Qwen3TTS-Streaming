from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid

import numpy as np
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


@pytest.mark.asyncio
async def test_websocket_logs_first_effective_audio_with_prefix_trim(
    monkeypatch,
):
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    lifecycle_events = []
    monkeypatch.setattr(
        "engine.gateway.websocket_server.LifecycleLogger.emit",
        lambda **fields: lifecycle_events.append(fields),
    )

    class _VadStubEngine:
        async def start_session(
            self, session_id, *, config, on_audio=None, on_done=None, on_event=None
        ):
            self._on_audio = on_audio
            self._on_done = on_done
            self._timing_acc = config.timing.extra["_server_timing_accumulator"]
            return session_id

        async def push_text_input(self, session_id, text):
            self._timing_acc.first_raw_audio_monotonic = time.monotonic()
            silence = np.zeros(9600, dtype=np.float32)  # 400 ms @ 24 kHz
            samples = np.arange(2400, dtype=np.float32)
            tone = (0.5 * np.sin(2.0 * np.pi * 440.0 * samples / 24000.0)).astype(
                np.float32
            )
            await self._on_audio(session_id, np.concatenate((silence, tone)).tobytes())

        async def mark_input_complete(self, session_id):
            await self._on_done(session_id, {})

        async def cancel(self, session_id):
            return None

    gateway = WebSocketGateway(_VadStubEngine())
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
                    "session_id": "sid-vad",
                    "config": {
                        "task_type": "custom_voice",
                        "output_policy": {
                            "vad_policy": {
                                "enabled": True,
                                "strategy": "energy",
                                "implementation": "energy",
                                "chunk_ms": 10,
                                "begin_threshold": 0.1,
                                "begin_count": 2,
                                "start_margin_ms": 20,
                            }
                        },
                    },
                }
            )
            await ws.send_json({"type": "text", "text": "你好"})
            await ws.send_json({"type": "end"})

            received_done = None
            for _ in range(5):
                message = await ws.receive(timeout=0.5)
                if message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)["event"]
                    if event["type"] == "done":
                        received_done = event
                        break

            first_effective = next(
                event
                for event in lifecycle_events
                if event.get("phase") == "output.audio.first_effective"
            )
            assert first_effective["vad_policy"] == "energy"
            assert first_effective["prefix_trim_applied"] is True
            assert first_effective["prefix_trimmed_ms"] > 0
            assert first_effective["first_raw_to_first_effective_audio_ms"] >= 0
            assert received_done is not None
            assert received_done["meta"]["server_prefix_trim_applied"] == "true"
            assert float(received_done["meta"]["server_ttft_effective_ms"]) > float(
                received_done["meta"]["server_ttft_raw_ms"]
            )

            await ws.close()


class _ReusableStubEngine:
    """Small callback-driven engine for websocket connection-lifecycle tests."""

    def __init__(self):
        self.callbacks = {}
        self.cancel_calls = []
        self.started_sessions = []
        self.started_client_sessions = []
        self.internal_by_client = {}
        self.client_by_internal = {}

    def describe_capabilities(self):
        return {
            "variant": "custom-1.7b",
            "loaded_model_type": "custom_voice",
        }

    async def start_session(
        self, session_id, *, config, on_audio=None, on_done=None, on_event=None
    ):
        client_session_id = config.timing.extra["_sampling_identity"]
        self.started_sessions.append(session_id)
        self.started_client_sessions.append(client_session_id)
        self.internal_by_client.setdefault(client_session_id, []).append(session_id)
        self.client_by_internal[session_id] = client_session_id
        self.callbacks[session_id] = {
            "on_audio": on_audio,
            "on_done": on_done,
            "on_event": on_event,
        }
        return session_id

    async def push_text_input(self, session_id, text):
        # One float32 sample; its bytes make cross-session leakage observable.
        client_session_id = self.client_by_internal[session_id]
        marker = (
            b"\x00\x00\x80?" if client_session_id.endswith("2") else b"\x00\x00\x00?"
        )
        await self.callbacks[session_id]["on_audio"](session_id, marker)

    async def mark_input_complete(self, session_id):
        await self.callbacks[session_id]["on_done"](session_id, {})

    async def cancel(self, session_id):
        self.cancel_calls.append(session_id)

    def internal_for(self, client_session_id, index=-1):
        return self.internal_by_client[client_session_id][index]


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
            assert capabilities["capabilities"]["supported_websocket_features"] == [
                "persistent_sessions_v1",
                "stream_resume_v1",
            ]
            assert capabilities["capabilities"]["supported_progress_features"] == [
                "text_progress_anchor_v1",
                "playback_progress_v1",
                "qwen.text_progress.v1",
            ]
            assert capabilities["capabilities"]["stream_resume_grace_ms"] == 30000
            assert capabilities["capabilities"]["supported_api_protocols"] == [
                "tts-session-v2alpha1",
                "openai-realtime-v1",
            ]
            assert capabilities["capabilities"]["openai_realtime_path"] == (
                "/v1/realtime"
            )

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
            assert engine.started_client_sessions == ["serial-1", "serial-2"]
            assert len(set(engine.started_sessions)) == 2
            assert set(engine.started_sessions).isdisjoint({"serial-1", "serial-2"})
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
            cancel_1_internal = engine.internal_for("cancel-1")
            assert engine.cancel_calls == [cancel_1_internal]
            assert not ws.closed

            # Simulate an engine callback racing after cancellation. It still
            # targets cancel-1's detached queue and must never appear in the
            # next logical session on this connection.
            await engine.callbacks[cancel_1_internal]["on_audio"](
                cancel_1_internal, b"\x00\x00\x10A"
            )

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

            failed_internal = engine.internal_for("failed-1")
            await engine.callbacks[failed_internal]["on_done"](
                failed_internal, {"error": "synthetic engine failure"}
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
            failed_internal = engine.internal_for("failed-race")
            await engine.callbacks[failed_internal]["on_done"](
                failed_internal, {"error": "boom"}
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
async def test_same_client_session_id_on_two_websockets_is_engine_isolated():
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
            ws_a = await client.ws_connect("/v1/ws")
            ws_b = await client.ws_connect("/v1/ws")

            await ws_a.send_json({"type": "start", "session_id": "duplicate"})
            await ws_b.send_json({"type": "start", "session_id": "duplicate"})
            start_a = json.loads((await ws_a.receive(timeout=1.0)).data)["event"]
            start_b = json.loads((await ws_b.receive(timeout=1.0)).data)["event"]

            assert start_a["session_id"] == "duplicate"
            assert start_b["session_id"] == "duplicate"
            internal_a, internal_b = engine.internal_by_client["duplicate"]
            assert internal_a != internal_b
            assert "duplicate" not in {internal_a, internal_b}

            await ws_a.send_json({"type": "cancel", "reason": "only a"})
            cancelled = json.loads((await ws_a.receive(timeout=1.0)).data)["event"]
            assert cancelled["session_id"] == "duplicate"
            assert engine.cancel_calls == [internal_a]

            # The second physical connection still owns a distinct engine
            # execution even though its public correlation ID is identical.
            await ws_b.send_json({"type": "text", "text": "still running"})
            assert (await ws_b.receive(timeout=1.0)).type == aiohttp.WSMsgType.BINARY
            await ws_b.send_json({"type": "stop"})
            done_b = json.loads((await ws_b.receive(timeout=1.0)).data)["event"]
            assert done_b["type"] == "done"
            assert done_b["session_id"] == "duplicate"
            assert engine.cancel_calls == [internal_a]

            await ws_a.close()
            await ws_b.close()


@pytest.mark.asyncio
async def test_websocket_generates_and_echoes_client_id_when_omitted():
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
            await ws.send_json({"type": "start"})
            start = json.loads((await ws.receive(timeout=1.0)).data)["event"]

            generated_client_id = start["session_id"]
            assert generated_client_id
            internal_id = engine.internal_for(generated_client_id)
            assert generated_client_id != internal_id

            await ws.send_json({"type": "cancel"})
            done = json.loads((await ws.receive(timeout=1.0)).data)["event"]
            assert done["session_id"] == generated_client_id
            assert engine.cancel_calls == [internal_id]


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

    # The official SDK opts into active-stream resume. Explicitly close the
    # process-local registry just as ``serve()`` does during gateway shutdown,
    # so retained terminal windows do not outlive this isolated test loop.
    await gateway.close()

    assert handshake_count == 1
    assert observed == [
        ["start", "audio", "done"],
        ["start", "audio", "done"],
    ]


@pytest.mark.asyncio
async def test_official_sdk_resumes_real_websocket_without_audio_loss_or_restart():
    """Drop the live TCP websocket and recover the same engine execution."""

    pytest.importorskip("websocket")
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from qwen3tts import SessionStartRequest, SynthesisConfig
    from qwen3tts._adapters.engine_websocket import EngineWebSocketAdapter
    from qwen3tts._internal.raw_websocket import ws_close

    class _RecoveryStubEngine(_ReusableStubEngine):
        def __init__(self):
            super().__init__()
            self.text_calls = []

        async def push_text_input(self, session_id, text):
            self.text_calls.append((session_id, text))
            marker = b"\x00\x00\x00?" if text == "first" else b"\x00\x00\x80?"
            await self.callbacks[session_id]["on_audio"](session_id, marker)

    engine = _RecoveryStubEngine()
    gateway = WebSocketGateway(engine, stream_resume_grace_seconds=1.0)
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

        def run_recovered_stream():
            adapter = EngineWebSocketAdapter(
                ws_url,
                timeout=2.0,
                connect_timeout=1.0,
                active_stream_resume=True,
                stream_resume_attempts=2,
                stream_resume_timeout=2.0,
                stream_resume_ack_interval=1,
                max_connections=1,
                max_idle_connections=1,
            )
            try:
                session = adapter.open_stream(
                    SessionStartRequest(
                        session_id="sdk-resume",
                        config=SynthesisConfig(),
                    )
                )
                messages = session.iter_messages(post_send_idle_timeout=2.0)
                session.send_text("first")
                first_event = next(messages)
                first_audio = next(messages)
                assert first_event.type == "start"
                assert first_audio.pcm_bytes == b"\x00\x00\x00?"

                # Close from outside the SDK reader thread. Depending on the
                # scheduler, either the reader or this next send becomes the
                # recovery leader; both paths share the same resume contract.
                ws_close(session._conn)
                session.send_text("second")
                session.stop()
                tail = list(messages)
                return first_audio.pcm_bytes, tail
            finally:
                adapter.close()

        first_pcm, tail = await asyncio.to_thread(run_recovered_stream)

    await gateway.close()

    tail_audio = [
        message.pcm_bytes for message in tail if hasattr(message, "pcm_bytes")
    ]
    tail_events = [getattr(message, "type", "") for message in tail]
    assert first_pcm == b"\x00\x00\x00?"
    assert tail_audio == [b"\x00\x00\x80?"]
    assert tail_events[-1] == "done"
    assert handshake_count == 2
    assert len(engine.started_sessions) == 1
    internal_id = engine.started_sessions[0]
    assert engine.text_calls == [
        (internal_id, "first"),
        (internal_id, "second"),
    ]
    assert engine.cancel_calls == []


@pytest.mark.asyncio
async def test_resumable_stream_reconnects_to_same_engine_and_replays_exact_audio():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine, stream_resume_grace_seconds=0.5)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)
    token = uuid.uuid4().hex

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json(
                {
                    "type": "start",
                    "session_id": "resume-one",
                    "resume": {"enabled": True, "token": token},
                }
            )
            start = json.loads((await ws.receive(timeout=1.0)).data)
            assert start["event"]["type"] == "start"
            assert start["delivery_seq"] == 1

            await ws.send_json({"type": "text", "seq_no": 1, "text": "first"})
            text_ack = json.loads((await ws.receive(timeout=1.0)).data)
            assert text_ack == {
                "type": "text_ack",
                "through_seq": 1,
                "duplicate": False,
            }
            first_header = json.loads((await ws.receive(timeout=1.0)).data)
            first_audio = await ws.receive(timeout=1.0)
            assert first_header["type"] == "audio_header"
            assert first_header["delivery_seq"] == 2
            assert first_header["start_sample"] == 0
            assert first_header["end_sample"] == 1
            assert first_audio.type == aiohttp.WSMsgType.BINARY

            internal_id = engine.internal_for("resume-one")
            await ws.close()
            await asyncio.sleep(0.02)
            assert engine.cancel_calls == []

            # Output produced while no websocket is attached remains in the
            # bounded replay ring and keeps its exact callback boundary.
            await engine.callbacks[internal_id]["on_audio"](
                internal_id, b"\x00\x00\x80?"
            )

            resumed_ws = await client.ws_connect("/v1/ws")
            await resumed_ws.send_json(
                {
                    "type": "resume",
                    "token": token,
                    "last_delivery_seq": 2,
                    "audio_through_sample": 1,
                }
            )
            resumed = json.loads((await resumed_ws.receive(timeout=1.0)).data)
            assert resumed["type"] == "resumed"
            assert resumed["acked_text_seq"] == 1
            assert resumed["input_closed"] is False

            replay_header = json.loads((await resumed_ws.receive(timeout=1.0)).data)
            replay_audio = await resumed_ws.receive(timeout=1.0)
            assert replay_header["delivery_seq"] == 3
            assert replay_header["start_sample"] == 1
            assert replay_header["end_sample"] == 2
            assert replay_audio.data == b"\x00\x00\x80?"
            assert engine.started_sessions == [internal_id]

            await resumed_ws.send_json({"type": "stop", "final_seq_no": 1})
            input_ack = json.loads((await resumed_ws.receive(timeout=1.0)).data)
            assert input_ack["type"] == "input_ack"
            assert input_ack["final_seq_no"] == 1
            done = json.loads((await resumed_ws.receive(timeout=1.0)).data)
            assert done["event"]["type"] == "done"
            assert done["delivery_seq"] == 4
            assert done["event"]["meta"]["websocket_connection_reusable"] == "true"

            await resumed_ws.send_json(
                {
                    "type": "terminal_ack",
                    "through_delivery_seq": 4,
                    "audio_through_sample": 2,
                }
            )
            await asyncio.sleep(0.01)
            assert engine.cancel_calls == []
            await resumed_ws.close()


@pytest.mark.asyncio
async def test_resumable_text_is_idempotent_and_rejects_sequence_gap():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine, stream_resume_grace_seconds=0.05)
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
                    "session_id": "resume-seq",
                    "resume": {"enabled": True, "token": uuid.uuid4().hex},
                }
            )
            await ws.receive(timeout=1.0)  # start delivery

            await ws.send_json({"type": "text", "seq_no": 1, "text": "same"})
            assert json.loads((await ws.receive(timeout=1.0)).data)["through_seq"] == 1
            await ws.receive(timeout=1.0)  # audio header
            await ws.receive(timeout=1.0)  # audio bytes

            await ws.send_json({"type": "text", "seq_no": 1, "text": "same"})
            duplicate = json.loads((await ws.receive(timeout=1.0)).data)
            assert duplicate["type"] == "text_ack"
            assert duplicate["duplicate"] is True

            # A missing sequence is rejected before touching the engine.
            await ws.send_json({"type": "text", "seq_no": 3, "text": "gap"})
            error = json.loads((await ws.receive(timeout=1.0)).data)
            assert error["type"] == "resume_error"
            assert error["code"] == "text_sequence_gap"
            assert len(engine.started_sessions) == 1
            await asyncio.sleep(0.07)
            assert engine.cancel_calls == [engine.internal_for("resume-seq")]


@pytest.mark.asyncio
async def test_resumable_disconnect_expires_and_cancels_engine_exactly_once():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine, stream_resume_grace_seconds=0.02)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)
    token = uuid.uuid4().hex

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json(
                {
                    "type": "start",
                    "session_id": "expires",
                    "resume": {"enabled": True, "token": token},
                }
            )
            await ws.receive(timeout=1.0)
            internal_id = engine.internal_for("expires")
            await ws.close()
            await asyncio.sleep(0.06)
            assert engine.cancel_calls == [internal_id]

            retry = await client.ws_connect("/v1/ws")
            await retry.send_json(
                {
                    "type": "resume",
                    "token": token,
                    "last_delivery_seq": 0,
                    "audio_through_sample": 0,
                }
            )
            error = json.loads((await retry.receive(timeout=1.0)).data)
            assert error["type"] == "resume_error"
            assert error["code"] == "resume_session_not_found"
            await retry.close()
            assert engine.cancel_calls == [internal_id]


@pytest.mark.asyncio
async def test_duplicate_resumable_start_is_idempotent_and_fences_old_handler():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine, stream_resume_grace_seconds=0.5)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)
    token = uuid.uuid4().hex
    start_payload = {
        "type": "start",
        "session_id": "duplicate-start",
        "config": {"speaker": "Serena"},
        "resume": {"enabled": True, "token": token},
    }

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            original = await client.ws_connect("/v1/ws")
            await original.send_json(start_payload)
            assert (
                json.loads((await original.receive(timeout=1.0)).data)["delivery_seq"]
                == 1
            )
            internal_id = engine.internal_for("duplicate-start")

            replacement = await client.ws_connect("/v1/ws")
            await replacement.send_json(start_payload)
            resumed = json.loads((await replacement.receive(timeout=1.0)).data)
            replayed_start = json.loads((await replacement.receive(timeout=1.0)).data)
            assert resumed["type"] == "resumed"
            assert replayed_start["delivery_seq"] == 1
            assert engine.started_sessions == [internal_id]
            await asyncio.sleep(0.01)
            assert engine.cancel_calls == []

            conflict = await client.ws_connect("/v1/ws")
            conflicting_payload = dict(start_payload)
            conflicting_payload["config"] = {"speaker": "Ryan"}
            await conflict.send_json(conflicting_payload)
            error = json.loads((await conflict.receive(timeout=1.0)).data)
            assert error["type"] == "resume_error"
            assert error["code"] == "resume_token_conflict"
            assert token not in json.dumps(error)
            await conflict.close()

            await replacement.send_json({"type": "cancel", "reason": "done"})
            terminal = json.loads((await replacement.receive(timeout=1.0)).data)
            assert terminal["event"]["type"] == "done"
            assert terminal["delivery_seq"] == 2
            await replacement.send_json(
                {
                    "type": "terminal_ack",
                    "through_delivery_seq": 2,
                    "audio_through_sample": 0,
                }
            )
            await asyncio.sleep(0.01)
            assert engine.cancel_calls == [internal_id]
            await original.close()
            await replacement.close()


@pytest.mark.asyncio
async def test_resumable_buffer_overflow_fails_explicitly_and_cancels_once():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(
        engine,
        stream_resume_grace_seconds=0.02,
        stream_resume_max_buffer_bytes=4096,
    )
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
                    "session_id": "overflow",
                    "resume": {
                        "enabled": True,
                        "token": uuid.uuid4().hex,
                    },
                }
            )
            await ws.receive(timeout=1.0)
            internal_id = engine.internal_for("overflow")

            await engine.callbacks[internal_id]["on_audio"](internal_id, b"\x00" * 8192)
            failure = json.loads((await ws.receive(timeout=1.0)).data)
            assert failure["type"] == "resume_error"
            assert failure["code"] == "resume_buffer_exceeded"
            await asyncio.sleep(0.06)
            assert engine.cancel_calls == [internal_id]


@pytest.mark.asyncio
async def test_resume_cursor_validation_and_terminal_ttl_cleanup():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine, stream_resume_grace_seconds=0.2)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)
    token = uuid.uuid4().hex

    async def rejected_resume(client, *, delivery_seq, audio_sample):
        ws = await client.ws_connect("/v1/ws")
        await ws.send_json(
            {
                "type": "resume",
                "token": token,
                "last_delivery_seq": delivery_seq,
                "audio_through_sample": audio_sample,
            }
        )
        error = json.loads((await ws.receive(timeout=1.0)).data)
        await ws.close()
        return error

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json(
                {
                    "type": "start",
                    "session_id": "cursor-check",
                    "resume": {"enabled": True, "token": token},
                }
            )
            start = json.loads((await ws.receive(timeout=1.0)).data)
            assert start["delivery_seq"] == 1
            await ws.send_json(
                {
                    "type": "ack",
                    "through_delivery_seq": 1,
                    "audio_through_sample": 0,
                }
            )
            # ACKs are intentionally one-way; let the handler apply the trim
            # before closing the transport used to send it.
            await asyncio.sleep(0.01)
            await ws.close()

            # Cursor 0 has been trimmed by the cumulative ACK.
            too_old = await rejected_resume(client, delivery_seq=0, audio_sample=0)
            assert too_old["code"] == "resume_window_exceeded"

            # The retained event cursor has produced no audio samples.
            bad_sample = await rejected_resume(client, delivery_seq=1, audio_sample=1)
            assert bad_sample["code"] == "resume_audio_cursor_mismatch"

            ahead = await rejected_resume(client, delivery_seq=2, audio_sample=0)
            assert ahead["code"] == "invalid_resume_cursor"

            resumed_ws = await client.ws_connect("/v1/ws")
            await resumed_ws.send_json(
                {
                    "type": "resume",
                    "token": token,
                    "last_delivery_seq": 1,
                    "audio_through_sample": 0,
                }
            )
            assert (
                json.loads((await resumed_ws.receive(timeout=1.0)).data)["type"]
                == "resumed"
            )
            await resumed_ws.send_json({"type": "stop", "final_seq_no": 0})
            assert (
                json.loads((await resumed_ws.receive(timeout=1.0)).data)["type"]
                == "input_ack"
            )
            terminal = json.loads((await resumed_ws.receive(timeout=1.0)).data)
            assert terminal["event"]["type"] == "done"
            await resumed_ws.close()  # deliberately omit terminal_ack

            await asyncio.sleep(0.25)
            expired = await rejected_resume(client, delivery_seq=2, audio_sample=0)
            assert expired["code"] == "resume_session_not_found"
            # Natural completion is not spuriously cancelled by terminal TTL.
            assert engine.cancel_calls == []


@pytest.mark.asyncio
async def test_failed_initialization_removes_token_even_when_cancel_raises():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    class _FailingStartEngine(_ReusableStubEngine):
        async def start_session(
            self,
            session_id,
            *,
            config,
            on_audio=None,
            on_done=None,
            on_event=None,
        ):
            self.started_sessions.append(session_id)
            raise RuntimeError("synthetic start failure")

        async def cancel(self, session_id):
            self.cancel_calls.append(session_id)
            raise RuntimeError("synthetic cancel failure")

    engine = _FailingStartEngine()
    gateway = WebSocketGateway(engine, stream_resume_grace_seconds=0.1)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)
    token = uuid.uuid4().hex

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json(
                {
                    "type": "start",
                    "session_id": "failed-start",
                    "resume": {"enabled": True, "token": token},
                }
            )
            # The immediate error spelling is less important than ensuring
            # the token record is gone despite cancel itself raising.
            await ws.receive(timeout=1.0)
            await ws.close()

            retry = await client.ws_connect("/v1/ws")
            await retry.send_json(
                {
                    "type": "resume",
                    "token": token,
                    "last_delivery_seq": 0,
                    "audio_through_sample": 0,
                }
            )
            error = json.loads((await retry.receive(timeout=1.0)).data)
            assert error["type"] == "resume_error"
            assert error["code"] == "resume_session_not_found"
            assert len(engine.cancel_calls) == 1


@pytest.mark.asyncio
async def test_legacy_active_disconnect_still_cancels_immediately():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    engine = _ReusableStubEngine()
    gateway = WebSocketGateway(engine, stream_resume_grace_seconds=1.0)
    app = web.Application()
    app.router.add_get("/v1/ws", gateway.handle_websocket)

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/ws")
            await ws.send_json({"type": "start", "session_id": "legacy-close"})
            await ws.receive(timeout=1.0)
            internal_id = engine.internal_for("legacy-close")
            await ws.close()
            await asyncio.sleep(0.02)
            assert engine.cancel_calls == [internal_id]
