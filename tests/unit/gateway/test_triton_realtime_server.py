from __future__ import annotations

import json

import pytest

from engine.gateway.triton_realtime_server import JsonlUsageRecorder, create_app


class _Backend:
    model_name = "tts_orchestrator"
    model_version = "7"

    def __init__(self, *, ready=True):
        self.ready = ready
        self.closed = False

    async def is_ready(self):
        return self.ready

    async def close(self):
        self.closed = True

    async def start(self, identity, *, start_request, outbound_queue):
        raise AssertionError("not used by these route tests")

    async def push_text(self, session_id, text):
        raise AssertionError("not used by these route tests")

    async def complete_input(self, session_id):
        raise AssertionError("not used by these route tests")

    async def cancel(self, session_id):
        return None

    def count_text_tokens(self, text):
        return len(text)


@pytest.mark.asyncio
async def test_sidecar_health_and_capabilities_routes():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    backend = _Backend()
    server = TestServer(create_app(backend))
    async with server:
        client = TestClient(server)
        async with client:
            health = await (await client.get("/health")).json()
            assert health == {
                "status": "ready",
                "running": True,
                "backend": "triton",
                "model": "tts_orchestrator",
            }
            response = await client.get("/v1/capabilities")
            assert response.status == 200
            capabilities = await response.json()
            assert capabilities["supported_api_protocols"] == ["openai-realtime-v1"]
            assert capabilities["openai_realtime_path"] == "/v1/realtime"
            assert capabilities["model_version"] == "7"
            assert capabilities["usage"]["output_audio_token_ms"] == 50
            assert capabilities["stream_resume_grace_ms"] == 30000
            assert capabilities["protocols"]["native_websocket"]["features"] == [
                "persistent_sessions_v1",
                "stream_resume_v1",
                "playback_progress_v1",
            ]
            assert (
                "qwen.response_resume.v1"
                in capabilities["protocols"]["openai_realtime"]["supported_extensions"]
            )
            assert (
                "active_response_resume"
                in capabilities["protocols"]["openai_realtime"]["features"]
            )
    assert backend.closed is True


@pytest.mark.asyncio
async def test_sidecar_health_is_503_while_triton_is_unavailable():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    server = TestServer(create_app(_Backend(ready=False)))
    async with server:
        client = TestClient(server)
        async with client:
            response = await client.get("/health")
            assert response.status == 503
            assert (await response.json())["running"] is False


@pytest.mark.asyncio
async def test_jsonl_usage_recorder_writes_complete_billing_records(tmp_path):
    path = tmp_path / "billing" / "usage.jsonl"
    recorder = JsonlUsageRecorder(str(path))
    record = {
        "response_id": "resp_1",
        "status": "cancelled",
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }

    await recorder(record)

    assert json.loads(path.read_text(encoding="utf-8")) == record
