"""Triton orchestrator output-policy contract regressions."""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ORCH_DIR = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if str(ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(ORCH_DIR))

sys.modules.setdefault(
    "triton_python_backend_utils",
    types.SimpleNamespace(TRITONSERVER_RESPONSE_COMPLETE_FINAL=1),
)

from engine.core.types import InputMode  # noqa: E402
from model import TritonPythonModel, _parse_input_mode  # noqa: E402


def test_session_config_preserves_complete_output_policy():
    model = TritonPythonModel.__new__(TritonPythonModel)
    model._loaded_model_type = "custom_voice"
    raw_policy = {
        "vad_policy": {
            "enabled": True,
            "strategy": "tenvad",
            "implementation": "onnx",
            "config": {"tenvad_threshold": 0.73},
            "chunk_ms": 8,
            "begin_threshold": 0.91,
            "begin_count": 3,
            "end_threshold": 0.21,
            "end_count": 47,
            "start_margin_ms": 12,
        },
        "chunk_ms": 40,
        "packet_format": "raw_pcm",
        "emit_text_events": False,
        "config": {"delivery": "guarded", "delivery_window_ms": "160"},
    }

    config = model._session_config_from_request(
        {
            "task_type": "custom_voice",
            "output_policy": raw_policy,
        },
        streaming=True,
    )

    vad = config.output_policy.vad
    assert vad.enabled is True
    assert vad.strategy == "tenvad"
    assert vad.implementation == "onnx"
    assert vad.config == {"tenvad_threshold": 0.73}
    assert vad.chunk_ms == 8
    assert vad.begin_threshold == 0.91
    assert vad.begin_count == 3
    assert vad.end_threshold == 0.21
    assert vad.end_count == 47
    assert vad.start_margin_ms == 12
    assert config.output_policy.chunk_ms == 40
    assert config.output_policy.packet_format == "raw_pcm"
    assert config.output_policy.emit_text_events is False
    assert config.output_policy.config == {
        "delivery": "guarded",
        "delivery_window_ms": "160",
    }


def test_realtime_auto_input_mode_and_timing_context_are_preserved():
    model = TritonPythonModel.__new__(TritonPythonModel)
    model._loaded_model_type = "custom_voice"

    config = model._session_config_from_request(
        {
            "task_type": "custom_voice",
            "input_mode": "auto",
            "timing": {
                "request_id": "request-1",
                "extra": {"_sampling_identity": "public-session"},
            },
        },
        streaming=True,
    )

    assert config.input_mode is InputMode.AUTO
    assert config.timing.request_id == "request-1"
    assert config.timing.extra["_sampling_identity"] == "public-session"
    assert _parse_input_mode(5, default_mode=InputMode.LONG_SEGMENT) is InputMode.AUTO


def test_cancel_finalizes_original_decoupled_response_sender():
    class Engine:
        def __init__(self):
            self.cancelled = []

        async def cancel(self, session_id):
            self.cancelled.append(session_id)

    model = TritonPythonModel.__new__(TritonPythonModel)
    model._engine = Engine()
    model._sessions_lock = threading.Lock()
    model._active_sessions = {"private-session"}
    model._session_senders = {"private-session": "original-sender"}
    model._run_coro = asyncio.run
    sent_events = []
    final_senders = []
    model._send_event = lambda sender, **kwargs: sent_events.append((sender, kwargs))
    model._safe_send_final = final_senders.append

    model._handle_cancel({"session_id": "private-session"}, "cancel-control-sender")

    assert model._engine.cancelled == ["private-session"]
    assert sent_events == [
        (
            "original-sender",
            {
                "event_type": "done",
                "session_id": "private-session",
                "meta": {"cancelled": "true"},
                "is_final": True,
            },
        )
    ]
    assert final_senders == ["cancel-control-sender"]
    assert "private-session" not in model._active_sessions
    assert "private-session" not in model._session_senders
