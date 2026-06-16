from __future__ import annotations

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

        async def start_session(self, session_id, *, config, on_audio=None, on_done=None, on_event=None):
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
