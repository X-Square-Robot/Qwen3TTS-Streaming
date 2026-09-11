from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest

from qwen3tts import (
    AudioChunk,
    AudioFormat,
    SessionStartRequest,
    SynthesisConfig,
    TTSClient,
)
from qwen3tts.constants import TRANSPORT_OPENAI_REALTIME
from engine.gateway.openai_realtime import OpenAIRealtimeGateway
from engine.gateway.websocket_server import WebSocketGateway
from engine.session import SessionCapacityError


class _RealtimeStubEngine:
    def __init__(self) -> None:
        self.callbacks = {}
        self.configs = {}
        self.pushed: list[tuple[str, str]] = []
        self.cancel_calls: list[str] = []

    def describe_capabilities(self):
        return {"variant": "custom-1.7b", "loaded_model_type": "custom_voice"}

    def count_text_tokens(self, text: str) -> int:
        # Deterministic stand-in for the live model tokenizer.
        return len(text)

    async def start_session(
        self, session_id, *, config, on_audio=None, on_done=None, on_event=None
    ):
        self.configs[session_id] = config
        self.callbacks[session_id] = {
            "on_audio": on_audio,
            "on_done": on_done,
            "on_event": on_event,
        }
        return session_id

    async def push_text_input(self, session_id: str, text: str) -> None:
        self.pushed.append((session_id, text))
        # 1,200 samples at 24 kHz is exactly 50 ms and therefore one
        # OpenAI Realtime output audio token.
        audio = np.linspace(-0.25, 0.25, 1200, dtype=np.float32).tobytes()
        await self.callbacks[session_id]["on_audio"](session_id, audio)
        await self.callbacks[session_id]["on_event"](
            session_id,
            {
                "type": "text_progress",
                "segment_idx": 0,
                "meta": {
                    "progress_basis": "ema_frame_ratio_v1",
                    "progress_quality": "rough",
                    "source_frame_start": "0",
                    "source_frame_end": "1",
                    "text_token_start": "0",
                    "text_token_end": "1",
                    "text_token_count": "2",
                    "text_progress": "0.5",
                    "progress_final": "false",
                },
            },
        )

    async def mark_input_complete(self, session_id: str) -> None:
        await self.callbacks[session_id]["on_done"](session_id, {})

    async def cancel(self, session_id: str) -> None:
        self.cancel_calls.append(session_id)


class _CapacityLimitedStubEngine(_RealtimeStubEngine):
    async def start_session(
        self, session_id, *, config, on_audio=None, on_done=None, on_event=None
    ):
        raise SessionCapacityError(128)


async def _receive_json(ws, *, timeout: float = 1.0) -> dict:
    message = await ws.receive(timeout=timeout)
    assert message.type.name == "TEXT"
    return json.loads(message.data)


async def _receive_until(
    ws, event_type: str, *, limit: int = 20
) -> tuple[dict, list[dict]]:
    seen = []
    for _ in range(limit):
        event = await _receive_json(ws)
        seen.append(event)
        if event["type"] == event_type:
            return event, seen
    raise AssertionError(
        f"did not receive {event_type}; saw {[e['type'] for e in seen]}"
    )


def _test_app(engine, *, usage_recorder=None, reliable=False):
    pytest.importorskip("aiohttp")
    from aiohttp import web

    legacy = WebSocketGateway(engine)
    if reliable:
        realtime = OpenAIRealtimeGateway(
            engine,
            session_service=legacy.session_service,
            usage_recorder=usage_recorder,
        )
    else:
        realtime = OpenAIRealtimeGateway(
            engine,
            session_starter=legacy._create_session,
            usage_recorder=usage_recorder,
        )
    app = web.Application()
    app.router.add_get("/v1/realtime", realtime.handle_websocket)

    async def close_gateway(_app):
        await realtime.close()

    app.on_cleanup.append(close_gateway)
    return app


@pytest.mark.asyncio
async def test_standard_realtime_tts_lifecycle_and_usage():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    engine = _RealtimeStubEngine()
    usage_records = []
    server = TestServer(_test_app(engine, usage_recorder=usage_records.append))
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/realtime?model=qwen3-tts-realtime")
            created = await _receive_json(ws)
            assert created["type"] == "session.created"
            assert created["session"]["object"] == "realtime.session"
            assert created["session"]["qwen"]["text_buffer_extension"] == (
                "qwen.input_text_buffer.v1"
            )

            await ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": "calm",
                        "output_modalities": ["audio"],
                        "audio": {
                            "output": {
                                "voice": "Serena",
                                "format": {"type": "audio/pcm", "rate": 24000},
                            }
                        },
                        "qwen": {"task_type": "custom_voice"},
                    },
                }
            )
            updated = await _receive_json(ws)
            assert updated["type"] == "session.updated"
            assert updated["session"]["audio"]["output"]["voice"] == "Serena"

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
            done, events = await _receive_until(ws, "response.done")
            event_types = [event["type"] for event in events]
            assert event_types[:3] == [
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
            ]
            delta = next(
                event
                for event in events
                if event["type"] == "response.output_audio.delta"
            )
            assert len(base64.b64decode(delta["delta"])) == 2400
            assert float(delta["qwen_server_ttft_ms"]) >= 0
            progress = next(
                event for event in events if event["type"] == "qwen.text_progress"
            )
            assert progress["segment_id"] == 0
            assert progress["meta"]["text_token_end"] == "1"
            # The stub deliberately emits a legacy event without an
            # attributed output range. The gateway forwards its EMA fields but
            # must not infer playback coordinates from sent bytes.
            assert "output_sample_end" not in progress["meta"]
            assert events.index(progress) > events.index(delta)
            assert event_types[-4:] == [
                "response.output_audio.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.done",
            ]

            usage = done["response"]["usage"]
            metadata = done["response"]["metadata"]
            assert float(metadata["qwen_server_ttft_ms"]) >= 0
            assert float(metadata["qwen_server_total_ms"]) >= float(
                metadata["qwen_server_ttft_ms"]
            )
            assert usage["input_tokens"] == len("hello") + len("calm")
            assert usage["output_tokens"] == 1
            assert usage["output_token_details"]["audio_tokens"] == 1
            assert usage["total_tokens"] == len("hellocalm") + 1
            assert len(usage_records) == 1
            assert usage_records[0]["response_id"] == done["response"]["id"]
            config = next(iter(engine.configs.values()))
            assert config.audio.encoding.value == "pcm_s16le"
            assert config.speaker == "Serena"
            await ws.close()


@pytest.mark.asyncio
async def test_new_sdk_openai_realtime_transport_interoperates_with_gateway():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestServer

    engine = _RealtimeStubEngine()
    server = TestServer(_test_app(engine))
    async with server:
        endpoint = str(server.make_url("/v1/realtime")).replace("http://", "ws://")
        client = await asyncio.to_thread(
            TTSClient.connect,
            endpoint,
            transport=TRANSPORT_OPENAI_REALTIME,
            verify=False,
        )
        try:
            config = SynthesisConfig(
                task_type="custom_voice",
                speaker="Serena",
                instruct="calm",
                audio=AudioFormat(encoding="pcm_f32", sample_rate=24000, channels=1),
            )
            config.timing_context.request_id = "client-request-1"
            result = await asyncio.to_thread(
                client.synthesize_bytes,
                "hello",
                request=config,
            )
        finally:
            await asyncio.to_thread(client.close)

    assert result.transport == TRANSPORT_OPENAI_REALTIME
    assert result.audio_format.encoding == "pcm_s16le"
    assert len(result.audio_bytes) == 2400
    assert result.details["usage"]["input_tokens"] == len("hello") + len("calm")
    assert result.details["usage"]["output_tokens"] == 1
    engine_config = next(iter(engine.configs.values()))
    assert engine_config.timing.request_id == "client-request-1"
    assert engine_config.timing.extra["api_protocol"] == "openai-realtime-v1"


@pytest.mark.asyncio
async def test_new_sdk_incremental_realtime_is_full_duplex_and_exposes_usage():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestServer

    engine = _RealtimeStubEngine()
    server = TestServer(_test_app(engine))
    async with server:
        endpoint = str(server.make_url("/v1/realtime")).replace("http://", "ws://")
        client = await asyncio.to_thread(
            TTSClient.connect,
            endpoint,
            transport=TRANSPORT_OPENAI_REALTIME,
            verify=False,
        )
        try:
            session = await asyncio.to_thread(
                client.open_stream,
                SessionStartRequest(
                    session_id="sdk-stream",
                    config=SynthesisConfig(task_type="custom_voice"),
                ),
            )
            await asyncio.to_thread(session.send_text, "hel")
            await asyncio.to_thread(session.send_text, "lo")
            await asyncio.to_thread(session.end)
            messages = await asyncio.to_thread(
                lambda: list(session.iter_messages(post_send_idle_timeout=1.0))
            )
        finally:
            await asyncio.to_thread(client.close)

    assert (
        len([message for message in messages if isinstance(message, AudioChunk)]) == 2
    )
    progress_events = [
        message
        for message in messages
        if getattr(message, "type", "") == "text_progress"
    ]
    assert len(progress_events) == 2
    assert progress_events[0].meta["progress_basis"] == "ema_frame_ratio_v1"
    assert session.response_status == "completed"
    assert session.usage["input_tokens"] == len("hello")
    assert session.usage["output_token_details"]["audio_tokens"] == 2
    assert [text for _, text in engine.pushed] == ["hel", "lo"]


@pytest.mark.asyncio
async def test_sdk_resumes_realtime_response_without_restarting_execution():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestServer
    from qwen3tts._internal.raw_websocket import ws_close

    engine = _RealtimeStubEngine()
    server = TestServer(_test_app(engine, reliable=True))
    async with server:
        endpoint = str(server.make_url("/v1/realtime")).replace("http://", "ws://")
        client = await asyncio.to_thread(
            TTSClient.connect,
            endpoint,
            transport=TRANSPORT_OPENAI_REALTIME,
            verify=False,
        )
        try:
            session = await asyncio.to_thread(
                client.open_stream,
                SessionStartRequest(
                    session_id="sdk-realtime-resume",
                    config=SynthesisConfig(task_type="custom_voice"),
                ),
            )
            messages = session.iter_messages(post_send_idle_timeout=2.0)
            await asyncio.to_thread(session.send_text, "hel")

            first_audio = None
            while first_audio is None:
                candidate = await asyncio.to_thread(next, messages)
                if isinstance(candidate, AudioChunk):
                    first_audio = candidate
            ws_close(session._conn)

            await asyncio.to_thread(session.send_text, "lo")
            await asyncio.to_thread(session.end)
            tail = await asyncio.to_thread(lambda: list(messages))
        finally:
            await asyncio.to_thread(client.close)

    audio = [first_audio] + [item for item in tail if isinstance(item, AudioChunk)]
    assert len(audio) == 2
    assert [item.output_sample_start for item in audio] == [0, 1200]
    assert [item.output_sample_end for item in audio] == [1200, 2400]
    assert session.response_status == "completed"
    assert len(engine.configs) == 1
    assert [text for _, text in engine.pushed] == ["hel", "lo"]


@pytest.mark.asyncio
async def test_reliable_realtime_validates_absolute_playback_cursor():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    engine = _RealtimeStubEngine()
    server = TestServer(_test_app(engine, reliable=True))
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/realtime")
            created = await _receive_json(ws)
            assert created["session"]["qwen"]["response_resume_extension"] == (
                "qwen.response_resume.v1"
            )
            await ws.send_json(
                {
                    "type": "response.create",
                    "response": {
                        "metadata": {"qwen_resume_token": "resume-token-0123456789"}
                    },
                }
            )
            for _ in range(3):
                await _receive_json(ws)
            await ws.send_json(
                {
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 1,
                    "text": "hello",
                }
            )
            delta, _ = await _receive_until(ws, "response.output_audio.delta")
            assert delta["qwen_output_sample_start"] == 0
            assert delta["qwen_output_sample_end"] == 1200

            await ws.send_json(
                {
                    "type": "qwen.playback.ack",
                    "response_id": delta["response_id"],
                    "played_through_sample": 100,
                    "buffered_through_sample": 1200,
                    "observed_delivery_seq": delta["qwen_delivery_seq"],
                }
            )
            accepted, _ = await _receive_until(ws, "qwen.playback.ack")
            assert accepted["accepted"] is True
            assert accepted["played_through_sample"] == 100

            await ws.send_json(
                {
                    "type": "qwen.playback.ack",
                    "response_id": delta["response_id"],
                    "played_through_sample": 100,
                    "buffered_through_sample": 1201,
                    "observed_delivery_seq": delta["qwen_delivery_seq"],
                }
            )
            rejected, _ = await _receive_until(ws, "error")
            assert rejected["error"]["code"] == "invalid_playback_progress"
            await ws.close()


@pytest.mark.asyncio
async def test_reliable_realtime_accepts_bigint_playback_cursor_strings():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    engine = _RealtimeStubEngine()
    server = TestServer(_test_app(engine, reliable=True))
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/realtime")
            await _receive_json(ws)
            await ws.send_json(
                {
                    "type": "response.create",
                    "response": {
                        "metadata": {"qwen_resume_token": "resume-token-0123456789"}
                    },
                }
            )
            for _ in range(3):
                await _receive_json(ws)
            await ws.send_json(
                {
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 1,
                    "text": "hello",
                }
            )
            delta, _ = await _receive_until(ws, "response.output_audio.delta")
            await ws.send_json(
                {
                    "type": "qwen.playback.ack",
                    "response_id": delta["response_id"],
                    "played_through_sample": "100",
                    "buffered_through_sample": "1200",
                    "observed_delivery_seq": delta["qwen_delivery_seq"],
                }
            )
            accepted, _ = await _receive_until(ws, "qwen.playback.ack")
            assert accepted["accepted"] is True
            assert accepted["played_through_sample"] == 100
            await ws.close()


@pytest.mark.asyncio
async def test_reliable_realtime_records_async_usage_exactly_once():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    usage_records = []
    release_recorder = asyncio.Event()

    async def recorder(record):
        usage_records.append(record)
        await release_recorder.wait()

    async def release_after_first_record():
        while not usage_records:
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        release_recorder.set()

    engine = _RealtimeStubEngine()
    server = TestServer(_test_app(engine, reliable=True, usage_recorder=recorder))
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/realtime")
            await _receive_json(ws)
            await ws.send_json(
                {
                    "type": "response.create",
                    "response": {
                        "metadata": {"qwen_resume_token": "usage-once-token-0123456789"}
                    },
                }
            )
            for _ in range(3):
                await _receive_json(ws)
            await ws.send_json(
                {
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 1,
                    "text": "hello",
                }
            )
            await ws.send_json({"type": "qwen.input_text_buffer.commit"})
            release_task = asyncio.create_task(release_after_first_record())
            done, _ = await _receive_until(ws, "response.done")
            await release_task

            assert done["response"]["usage"]["input_tokens"] == len("hello")
            assert len(usage_records) == 1
            assert usage_records[0]["response_id"] == done["response"]["id"]
            await ws.close()


@pytest.mark.asyncio
async def test_qwen_text_append_is_full_duplex_and_idempotent():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    engine = _RealtimeStubEngine()
    server = TestServer(_test_app(engine))
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/realtime")
            await _receive_json(ws)  # session.created
            await ws.send_json({"type": "response.create"})
            for expected in (
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
            ):
                assert (await _receive_json(ws))["type"] == expected

            await ws.send_json(
                {
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 1,
                    "text": "hel",
                }
            )
            delta, first_events = await _receive_until(
                ws, "response.output_audio.delta"
            )
            assert delta["delta"]
            if not any(e["type"] == "qwen.input_text_buffer.ack" for e in first_events):
                ack, _ = await _receive_until(ws, "qwen.input_text_buffer.ack")
                assert ack["duplicate"] is False

            await ws.send_json(
                {
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 1,
                    "text": "hel",
                }
            )
            duplicate, _ = await _receive_until(ws, "qwen.input_text_buffer.ack")
            assert duplicate["duplicate"] is True

            await ws.send_json(
                {
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 2,
                    "text": "lo",
                }
            )
            await _receive_until(ws, "qwen.input_text_buffer.ack")
            await ws.send_json({"type": "qwen.input_text_buffer.commit"})

            done, tail = await _receive_until(ws, "response.done")
            assert any(
                event["type"] == "qwen.input_text_buffer.committed" for event in tail
            )
            usage = done["response"]["usage"]
            assert usage["input_tokens"] == len("hello")
            assert usage["output_token_details"]["audio_tokens"] == 2
            assert [text for _, text in engine.pushed] == ["hel", "lo"]
            await ws.close()


@pytest.mark.asyncio
async def test_sequence_error_is_nonfatal_and_next_event_is_accepted():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    engine = _RealtimeStubEngine()
    server = TestServer(_test_app(engine))
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/realtime")
            await _receive_json(ws)
            await ws.send_json(
                {
                    "event_id": "client-gap",
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 2,
                    "text": "bad",
                }
            )
            error = await _receive_json(ws)
            assert error["type"] == "error"
            assert error["error"]["code"] == "sequence_gap"
            assert error["error"]["event_id"] == "client-gap"

            await ws.send_json(
                {
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 1,
                    "text": "ok",
                }
            )
            ack = await _receive_json(ws)
            assert ack["type"] == "qwen.input_text_buffer.ack"
            assert not ws.closed
            await ws.close()


@pytest.mark.asyncio
async def test_session_capacity_rejection_is_a_stable_protocol_error():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    engine = _CapacityLimitedStubEngine()
    server = TestServer(_test_app(engine))
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/realtime")
            await _receive_json(ws)
            await ws.send_json(
                {
                    "event_id": "capacity-check",
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 1,
                    "text": "hello",
                }
            )
            ack = await _receive_json(ws)
            assert ack["type"] == "qwen.input_text_buffer.ack"
            await ws.send_json(
                {"type": "response.create", "event_id": "capacity-check"}
            )
            error, _ = await _receive_until(ws, "error")
            assert error["type"] == "error"
            assert error["error"]["code"] == "max_sessions"
            assert error["error"]["type"] == "server_error"
            assert "Max sessions (128) reached" in error["error"]["message"]
            assert not ws.closed
            await ws.close()


@pytest.mark.asyncio
async def test_cancel_returns_partial_usage_and_records_it_once():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    engine = _RealtimeStubEngine()
    usage_records = []
    server = TestServer(_test_app(engine, usage_recorder=usage_records.append))
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/realtime")
            await _receive_json(ws)
            await ws.send_json({"type": "response.create"})
            for _ in range(3):
                await _receive_json(ws)
            await ws.send_json(
                {
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 1,
                    "text": "partial",
                }
            )
            await _receive_until(ws, "response.output_audio.delta")
            await ws.send_json({"type": "response.cancel"})
            done, _ = await _receive_until(ws, "response.done")
            assert done["response"]["status"] == "cancelled"
            assert done["response"]["usage"]["input_tokens"] == len("partial")
            assert done["response"]["usage"]["output_tokens"] == 1
            assert len(usage_records) == 1
            assert usage_records[0]["status"] == "cancelled"
            assert len(engine.cancel_calls) == 1
            await ws.close()


@pytest.mark.asyncio
async def test_disconnect_still_records_partial_usage_once():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    engine = _RealtimeStubEngine()
    usage_records = []
    server = TestServer(_test_app(engine, usage_recorder=usage_records.append))
    async with server:
        client = TestClient(server)
        async with client:
            ws = await client.ws_connect("/v1/realtime")
            await _receive_json(ws)
            await ws.send_json({"type": "response.create"})
            for _ in range(3):
                await _receive_json(ws)
            await ws.send_json(
                {
                    "type": "qwen.input_text_buffer.append",
                    "sequence": 1,
                    "text": "bill me",
                }
            )
            await _receive_until(ws, "response.output_audio.delta")
            await ws.close()

            for _ in range(20):
                if usage_records:
                    break
                await asyncio.sleep(0)
            assert len(usage_records) == 1
            assert usage_records[0]["status"] == "cancelled"
            assert usage_records[0]["usage"]["input_tokens"] == len("bill me")
            assert usage_records[0]["usage"]["output_tokens"] == 1
