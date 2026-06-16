from __future__ import annotations

import asyncio

import pytest

from qwen3_tts_protocol import (
    AudioFormat,
    BytesResult,
    SessionStartRequest,
    StreamEvent,
    SynthesisConfig,
)
from qwen3_tts_client.async_client import AsyncTTSClient


class _FakeAdapter:
    def __init__(self):
        self.calls = []

    def get_capabilities(self):
        self.calls.append("get_capabilities")
        return {"variant": "fake"}

    def synthesize_bytes(self, text: str, *, request):
        self.calls.append(("synthesize_bytes", text))
        return BytesResult(
            audio_bytes=b"\x00\x00\x00\x00" * 2,
            audio_format=AudioFormat(),
            session_id=request.session_id,
            transport="fake",
            events=[StreamEvent(type="done", session_id=request.session_id)],
            warnings=[],
            details={},
        )

    def open_stream(self, start_request):
        self.calls.append(("open_stream", start_request.session_id))
        class _Session:
            session_id = start_request.session_id
            transport = "fake"
            degraded_to_oneshot = False
            def send_text(self, text, **kw): pass
            def end(self, **kw): pass
            def cancel(self, reason=""): pass
            def iter_messages(self): return iter(())
        return _Session()


class TestAsyncTTSClient:
    def test_delegates_get_capabilities(self):
        fake_adapter = _FakeAdapter()
        detected = type("D", (), {"transport": "fake", "probe_report": []})()
        sync_client = type(
            "TTSClient",
            (),
            {
                "endpoint": "fake",
                "_adapter": fake_adapter,
                "resolved_transport": "fake",
                "probe_report": [],
                "detected_transport": detected,
                "get_capabilities": fake_adapter.get_capabilities,
                "synthesize_bytes": fake_adapter.synthesize_bytes,
                "open_stream": fake_adapter.open_stream,
            },
        )()
        async_client = AsyncTTSClient(sync_client)

        async def _run():
            caps = await async_client.get_capabilities()
            return caps

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result == {"variant": "fake"}

    def test_delegates_synthesize_bytes(self):
        fake_adapter = _FakeAdapter()
        detected = type("D", (), {"transport": "fake", "probe_report": []})()

        def _synthesize_bytes(self_unused, text: str, *, request=None):
            return fake_adapter.synthesize_bytes(text, request=request)

        sync_client = type(
            "TTSClient",
            (),
            {
                "endpoint": "fake",
                "_adapter": fake_adapter,
                "resolved_transport": "fake",
                "probe_report": [],
                "detected_transport": detected,
                "get_capabilities": fake_adapter.get_capabilities,
                "synthesize_bytes": _synthesize_bytes,
                "open_stream": fake_adapter.open_stream,
            },
        )()
        async_client = AsyncTTSClient(sync_client)

        async def _run():
            result = await async_client.synthesize_bytes("hello", request=SynthesisConfig(task_type="custom_voice"))
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert isinstance(result, BytesResult)
        assert len(result.audio_bytes) > 0

    def test_properties_mirror_sync_client(self):
        fake_adapter = _FakeAdapter()
        detected = type("D", (), {"transport": "engine-websocket", "probe_report": []})()
        sync_client = type(
            "TTSClient",
            (),
            {
                "endpoint": "ws://localhost:50052/v1/ws",
                "_adapter": fake_adapter,
                "resolved_transport": "engine-websocket",
                "probe_report": [{"transport": "engine-websocket", "ok": True}],
                "detected_transport": detected,
            },
        )()
        async_client = AsyncTTSClient(sync_client)

        assert async_client.endpoint == "ws://localhost:50052/v1/ws"
        assert async_client.resolved_transport == "engine-websocket"
        assert len(async_client.probe_report) == 1
