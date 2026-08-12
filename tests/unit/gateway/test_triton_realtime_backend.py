from __future__ import annotations

import asyncio
import json
import types

import numpy as np
import pytest

from engine.core.types import (
    AudioConfig,
    AudioEncoding,
    GroupPolicy,
    InputMode,
    SessionConfig,
)
from engine.gateway.session_identity import GatewaySessionIdentity
from engine.gateway.triton_realtime_backend import TritonRealtimeBackend
from engine.interface import SessionStartRequest


class _FakeInferInput:
    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data = None

    def set_data_from_numpy(self, data):
        self.data = data


class _FakeRequestedOutput:
    def __init__(self, name):
        self.name = name


class _FakeResult:
    def __init__(
        self,
        event_type: str,
        *,
        payload: dict | None = None,
        audio: bytes = b"",
        is_final: bool = False,
    ) -> None:
        self._values = {
            "audio_chunk": np.array([audio], dtype=object),
            "event_type": np.array([event_type], dtype=object),
            "event_json": np.array(
                [json.dumps(payload or {}, ensure_ascii=False)], dtype=object
            ),
            "is_final": np.array([is_final], dtype=bool),
        }

    def as_numpy(self, name):
        return self._values.get(name)


class _FakeTritonClient:
    def __init__(self) -> None:
        self.callback = None
        self.requests: list[tuple[dict, dict]] = []
        self.stop_calls = 0
        self.close_calls = 0

    def start_stream(self, *, callback, headers=None):
        self.callback = callback
        self.stream_headers = headers

    def async_stream_infer(self, **kwargs):
        payload = json.loads(kwargs["inputs"][0].data.reshape(-1)[0])
        self.requests.append((payload, kwargs))
        action = payload["action"]
        if action == "init":
            self.callback(
                result=_FakeResult(
                    "start",
                    payload={
                        "session_id": payload["session_id"],
                        "audio_format": {
                            "encoding": "pcm_s16le",
                            "sample_rate": 24000,
                            "channels": 1,
                        },
                    },
                )
            )
        elif action == "append_text":
            self.callback(
                result=_FakeResult(
                    "audio",
                    payload={
                        "session_id": payload["session_id"],
                        "meta": {"first_audio_chunk": "true"},
                    },
                    audio=b"\x01\x00" * 1200,
                )
            )
        elif action in {"text_complete", "cancel"}:
            self.callback(
                result=_FakeResult(
                    "done",
                    payload={
                        "session_id": payload["session_id"],
                        "meta": {"cancelled": str(action == "cancel").lower()},
                    },
                    is_final=True,
                )
            )

    def stop_stream(self):
        self.stop_calls += 1

    def close(self):
        self.close_calls += 1


def _fake_grpc(client):
    return types.SimpleNamespace(
        InferInput=_FakeInferInput,
        InferRequestedOutput=_FakeRequestedOutput,
        InferenceServerClient=lambda url: client,
    )


def _start_request() -> SessionStartRequest:
    config = SessionConfig(
        task_type="custom_voice",
        language="zh",
        speaker="Vivian",
        instruct="calm",
        input_mode=InputMode.AUTO,
        group_policy=GroupPolicy.AUTO,
        audio=AudioConfig(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=24000,
            channels=1,
        ),
    )
    config.timing.request_id = "request-1"
    return SessionStartRequest(session_id="public", config=config)


@pytest.mark.asyncio
async def test_triton_backend_maps_one_response_to_one_bidirectional_stream(
    monkeypatch,
):
    fake_client = _FakeTritonClient()
    monkeypatch.setattr(
        "engine.gateway.triton_realtime_backend._require_triton",
        lambda: (np, _fake_grpc(fake_client)),
    )
    backend = TritonRealtimeBackend(
        "triton:8001",
        model_version="7",
        token_counter=lambda text: len(text.split()),
        headers={"authorization": "Bearer internal"},
    )
    identity = GatewaySessionIdentity.create("public-sampling-id")
    outbound = asyncio.Queue()

    await backend.start(
        identity,
        start_request=_start_request(),
        outbound_queue=outbound,
    )
    start = await asyncio.wait_for(outbound.get(), timeout=1)
    assert start["event"]["type"] == "start"

    await backend.push_text(identity.internal_session_id, "hello realtime")
    audio = await asyncio.wait_for(outbound.get(), timeout=1)
    assert audio["type"] == "audio"
    assert audio["audio"]["pcm_data"] == b"\x01\x00" * 1200
    assert audio["audio"]["encoding"] == "pcm_s16le"

    await backend.complete_input(identity.internal_session_id)
    done = await asyncio.wait_for(outbound.get(), timeout=1)
    assert done["event"]["type"] == "done"

    payloads = [payload for payload, _kwargs in fake_client.requests]
    assert [payload["action"] for payload in payloads] == [
        "init",
        "append_text",
        "text_complete",
    ]
    init = payloads[0]
    assert init["session_id"] == identity.internal_session_id
    assert init["input_mode"] == "auto"
    assert init["audio"]["encoding"] == "pcm_s16le"
    assert init["timing"]["extra"]["_sampling_identity"] == "public-sampling-id"
    assert all(
        kwargs["model_version"] == "7" for _payload, kwargs in fake_client.requests
    )
    assert fake_client.stream_headers == {"authorization": "Bearer internal"}

    for _ in range(10):
        if identity.internal_session_id not in backend._sessions:
            break
        await asyncio.sleep(0)
    assert identity.internal_session_id not in backend._sessions
    assert fake_client.stop_calls == 1
    assert fake_client.close_calls == 1


@pytest.mark.asyncio
async def test_triton_backend_cancel_waits_for_terminal_and_closes_once(monkeypatch):
    fake_client = _FakeTritonClient()
    monkeypatch.setattr(
        "engine.gateway.triton_realtime_backend._require_triton",
        lambda: (np, _fake_grpc(fake_client)),
    )
    backend = TritonRealtimeBackend("triton:8001", token_counter=lambda _text: 0)
    identity = GatewaySessionIdentity.create("public")
    outbound = asyncio.Queue()
    await backend.start(
        identity,
        start_request=_start_request(),
        outbound_queue=outbound,
    )
    await outbound.get()

    await backend.cancel(identity.internal_session_id)
    terminal = await asyncio.wait_for(outbound.get(), timeout=1)
    assert terminal["event"]["type"] == "done"
    assert fake_client.requests[-1][0]["action"] == "cancel"
    assert identity.internal_session_id not in backend._sessions
    assert fake_client.stop_calls == 1
    assert fake_client.close_calls == 1


@pytest.mark.asyncio
async def test_triton_backend_readiness_and_token_counter(monkeypatch):
    class ReadyClient:
        def __init__(self):
            self.calls = []
            self.closed = False

        def is_server_live(self, **kwargs):
            self.calls.append(("live", kwargs))
            return True

        def is_server_ready(self, **kwargs):
            self.calls.append(("ready", kwargs))
            return True

        def is_model_ready(self, model_name, **kwargs):
            self.calls.append((model_name, kwargs))
            return True

        def close(self):
            self.closed = True

    ready_client = ReadyClient()
    monkeypatch.setattr(
        "engine.gateway.triton_realtime_backend._require_triton",
        lambda: (
            np,
            types.SimpleNamespace(
                InferenceServerClient=lambda url: ready_client,
            ),
        ),
    )
    backend = TritonRealtimeBackend(
        "triton:8001", token_counter=lambda text: len(text) + 3
    )

    assert backend.count_text_tokens("bill") == 7
    assert await backend.is_ready(timeout=0.5) is True
    assert ready_client.closed is True
    assert ready_client.calls[-1][0] == "tts_orchestrator"
    assert ready_client.calls[-1][1]["model_version"] == ""
