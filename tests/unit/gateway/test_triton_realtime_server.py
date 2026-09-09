from __future__ import annotations

import json

import pytest

from engine.config import ModelPackagePaths
from engine.gateway.triton_realtime_server import (
    JsonlUsageRecorder,
    _runtime_capabilities_from_package,
    create_app,
)


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


def test_runtime_capabilities_use_actual_package_voice_metadata(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "config.json").write_text(
        json.dumps(
            {
                "spk_id": {"Vivian": 1, "Serena": 2},
                "codec_language_id": {"English": 10, "Chinese": 11},
            }
        ),
        encoding="utf-8",
    )
    package_paths = ModelPackagePaths(
        package_dir=str(tmp_path),
        engine_dir=str(tmp_path / "runtime"),
        weights_dir=str(weights_dir),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        manifest_path=str(tmp_path / "triton_manifest.json"),
        runtime_artifact_path=str(tmp_path / "runtime/model.plan"),
    )
    monkeypatch.setattr(
        "engine.gateway.triton_realtime_server.load_model_manifest",
        lambda *_args, **_kwargs: type(
            "Arch",
            (),
            {
                "tts_model_type": "custom_voice",
                "supported_task_types": ("custom_voice",),
                "variant": "custom-1.7b",
                "native_cursor": {
                    "enabled": True,
                    "progress_available": True,
                },
                "engine_profile": type(
                    "Profile",
                    (),
                    {
                        "max_batch_size": 8,
                        "max_input_len": 128,
                        "max_seq_len": 512,
                        "engine_dtype": "bf16",
                        "triton_io_float_dtype": "fp32",
                    },
                )(),
            },
        )(),
    )

    capabilities = _runtime_capabilities_from_package(package_paths, "")

    assert capabilities["supported_speakers"] == ["serena", "vivian"]
    assert capabilities["supported_languages"] == ["auto", "chinese", "english"]
    assert capabilities["native_cursor"]["progress_available"] is False
    assert capabilities["native_cursor"]["reason"] == "release_evidence_missing"
    assert capabilities["speech_state"] == {
        "supported": False,
        "reason": "missing_speech_state",
    }


def test_runtime_capabilities_reject_malformed_cursor_progress_flag(
    tmp_path, monkeypatch
):
    package_paths = ModelPackagePaths(
        package_dir=str(tmp_path),
        engine_dir=str(tmp_path / "runtime"),
        weights_dir=str(tmp_path / "weights"),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        manifest_path=str(tmp_path / "triton_manifest.json"),
        runtime_artifact_path=str(tmp_path / "runtime/model.plan"),
    )
    monkeypatch.setattr(
        "engine.gateway.triton_realtime_server.load_model_manifest",
        lambda *_args, **_kwargs: type(
            "Arch",
            (),
            {
                "tts_model_type": "custom_voice",
                "supported_task_types": ("custom_voice",),
                "variant": "custom-1.7b",
                "native_cursor": {
                    "enabled": True,
                    "progress_available": "true",
                },
                "engine_profile": type(
                    "Profile",
                    (),
                    {
                        "max_batch_size": 8,
                        "max_input_len": 128,
                        "max_seq_len": 512,
                        "engine_dtype": "bf16",
                        "triton_io_float_dtype": "fp32",
                    },
                )(),
            },
        )(),
    )

    capabilities = _runtime_capabilities_from_package(package_paths, "")

    assert capabilities["native_cursor"]["progress_available"] is False
    assert capabilities["native_cursor"]["reason"] == (
        "malformed_native_cursor_capability"
    )


def test_runtime_capabilities_always_expose_fail_closed_speech_state(tmp_path):
    package_paths = ModelPackagePaths(
        package_dir=str(tmp_path),
        engine_dir=str(tmp_path / "runtime"),
        weights_dir=str(tmp_path / "weights"),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        manifest_path=str(tmp_path / "triton_manifest.json"),
        runtime_artifact_path=str(tmp_path / "runtime/model.plan"),
    )

    capabilities = _runtime_capabilities_from_package(package_paths, "")

    assert capabilities["speech_state"]["supported"] is False
    assert capabilities["speech_state"]["reason"] == "missing_speech_state"


@pytest.mark.asyncio
async def test_sidecar_health_capabilities_sdk_and_demo_config_routes(
    tmp_path, monkeypatch
):
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    backend = _Backend()
    monkeypatch.setenv("DEMO_ENABLED", "true")
    wheel_name = "qwen3_tts_client-0.2.0-py3-none-any.whl"
    (tmp_path / wheel_name).write_bytes(b"wheel")
    server = TestServer(create_app(backend, sdk_dir=tmp_path))
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
            assert capabilities["supported_api_protocols"] == [
                "tts-session-v2alpha1",
                "openai-realtime-v1",
            ]
            assert capabilities["schema_version"] == "qwen.tts.capabilities.v1"
            assert capabilities["runtime"]["type"] == "triton"
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
            sdk_redirect = await client.get("/sdk", allow_redirects=False)
            assert sdk_redirect.status == 308
            assert sdk_redirect.headers["Location"] == "./sdk/"
            sdk_index = await client.get("/sdk/")
            assert wheel_name in await sdk_index.text()
            demo_config_response = await client.get("/demo/config.json")
            assert demo_config_response.status == 200
            assert demo_config_response.headers["Cache-Control"] == "no-store"
            demo_config = await demo_config_response.json()
            assert demo_config["runtime_type"] == "triton"
            assert demo_config["endpoints"]["openai_realtime_url"] == ("../v1/realtime")
            assert demo_config["python_sdk"]["download_url"] == (f"../sdk/{wheel_name}")
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
async def test_demo_config_is_404_when_explicitly_disabled(tmp_path, monkeypatch):
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    monkeypatch.setenv("DEMO_ENABLED", "false")
    server = TestServer(create_app(_Backend(), sdk_dir=tmp_path))
    async with server:
        async with TestClient(server) as client:
            assert (await client.get("/demo/config.json")).status == 404


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
