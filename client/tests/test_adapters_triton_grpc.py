"""Regression test for the triton-grpc terminal guarantee.

If a stream ends via ``is_final=True`` on an audio chunk without an explicit
``end``/``error`` event, the worker must still emit a terminal so the queue
sentinel is enqueued and ``iter_messages()`` does not block forever.
"""

from __future__ import annotations

import threading
import types

import pytest

pytest.importorskip("tritonclient")
import numpy as np

from qwen3tts_protocol import (
    AudioChunk,
    SessionStartRequest,
    StreamEvent,
    SynthesisConfig,
)
import qwen3tts._adapters.triton_grpc as tg


class _FakeClient:
    """Fires the stream callback once per infer with a *final* audio chunk and
    no end event — the exact shape that used to hang the consumer."""

    def __init__(self, url):
        self._cb = None

    def start_stream(self, callback):
        self._cb = callback

    def async_stream_infer(self, **kwargs):
        if self._cb is not None:
            self._cb(result=object(), error=None)

    def stop_stream(self):
        pass


def test_is_final_without_end_event_does_not_hang(monkeypatch):
    fake_grpcclient = types.SimpleNamespace(
        InferenceServerClient=lambda url: _FakeClient(url),
        InferInput=lambda *a, **k: types.SimpleNamespace(
            set_data_from_numpy=lambda *a, **k: None
        ),
        InferRequestedOutput=lambda name: name,
    )
    monkeypatch.setattr(tg, "_require_triton", lambda: (np, fake_grpcclient))
    # Canned extraction: an audio chunk flagged final, with NO end/error event.
    canned = {
        "event_type": "audio",
        "event_json": None,
        "audio_chunk": "x",
        "is_final": True,
    }
    monkeypatch.setattr(
        tg, "_scalar_from_result", lambda result, name: canned.get(name)
    )
    monkeypatch.setattr(
        tg, "_decode_audio_bytes_field", lambda value, payload: b"\x00\x00\x00\x00"
    )

    adapter = tg.TritonGrpcAdapter("localhost:8001", model_name="m", timeout=2.0)
    session = tg.TritonGrpcStreamSession(
        adapter,
        SessionStartRequest(
            session_id="s", config=SynthesisConfig(task_type="custom_voice")
        ),
    )
    session.end()  # close the send side so the worker reaches its terminal logic

    # Drain on a watchdog thread: without the fix this never returns.
    collected: list = []

    def _drain():
        collected.extend(session.iter_messages())

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()
    drainer.join(timeout=5.0)

    assert not drainer.is_alive(), "iter_messages() hung — no terminal emitted"
    assert any(isinstance(m, AudioChunk) for m in collected)
    assert any(isinstance(m, StreamEvent) and m.type == "done" for m in collected)
