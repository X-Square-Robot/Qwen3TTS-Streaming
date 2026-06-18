from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from demo_api import llm_pk
from demo_api.audio_assets import attach_audio_to_result, scheduled_start_ms
from demo_api.audio_store import AudioStore
from demo_api.jobs import ConcurrencyJobManager
from demo_api.schemas import RunMetrics, RunResult, normalize_backend_result, percentile, summarize_ttft
from demo_api.triton_client import TtsRequest, build_action_payload


def test_normalize_triton_streaming_result_round_trips():
    result = normalize_backend_result(
        {
            "run_id": "stream",
            "backend": "triton_streaming",
            "label": "Streaming TTS (token-by-token)",
            "mode": "engine starts on first simulated token",
            "source": "live_triton",
            "metrics": {
                "client_ttfb_ms": 18,
                "first_playable_ms": 18,
                "simulated_llm_complete_ms": 870,
                "total_ms": 4200,
            },
            "events": [
                {"type": "request_started", "t_ms": 0},
                {"type": "first_audio_chunk", "t_ms": 18},
            ],
        }
    )

    assert result["backend"] == "triton_streaming"
    assert result["metrics"]["client_ttfb_ms"] == 18
    assert result["metrics"]["simulated_llm_complete_ms"] == 870
    assert result["audio_format"] == {"encoding": "pcm_f32", "sample_rate": 24000, "channels": 1}


def test_normalize_unknown_backend_raises():
    with pytest.raises(ValueError):
        normalize_backend_result({"backend": "official_pytorch_offline"})


def test_scheduled_start_ms_uses_first_playable():
    metrics = {
        "client_ttfb_ms": 20,
        "first_playable_ms": 18,
    }
    assert scheduled_start_ms(metrics, "triton_streaming") == pytest.approx(18)
    assert scheduled_start_ms(metrics, "triton_offline") == pytest.approx(18)


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 0.5) == pytest.approx(25.0)
    assert percentile([10, 20, 30, 40], 0.9) == pytest.approx(37.0)


def test_summarize_ttft_reports_distribution():
    summary = summarize_ttft([100, 120, 180, 220, 300])

    assert summary["count"] == 5
    assert summary["avg_ttft_ms"] == pytest.approx(184.0)
    assert summary["p50_ttft_ms"] == pytest.approx(180.0)
    assert summary["p90_ttft_ms"] == pytest.approx(268.0)
    assert summary["max_ttft_ms"] == pytest.approx(300.0)


def test_simulated_concurrency_does_not_attach_fake_audio():
    async def run_job():
        manager = ConcurrencyJobManager(enable_live=False)
        job = await manager.create_job({"concurrency": 4, "live": False})
        queue = await manager.subscribe(job)
        audio_seen = False
        while True:
            message = await asyncio.wait_for(queue.get(), timeout=2)
            if message.get("type") == "lane_update" and message.get("audio"):
                audio_seen = True
            if message.get("type") == "summary":
                return audio_seen, message

    audio_seen, summary = asyncio.run(run_job())
    assert audio_seen is False
    assert summary["source"] == "simulated"
    assert summary["count"] == 4


def test_missing_raw_audio_is_not_replaced_with_synthetic_wav(tmp_path):
    result = RunResult(
        run_id="fixture-row",
        backend="triton_streaming",
        label="Streaming TTS",
        mode="test",
        source="fixture",
        metrics=RunMetrics(first_playable_ms=24, audio_duration_ms=1600),
    )

    attach_audio_to_result(result, AudioStore(tmp_path))

    assert result.audio == {}
    assert result.warnings == [
        "No raw audio captured; playback disabled instead of attaching synthetic audio."
    ]
    assert list(tmp_path.iterdir()) == []


def test_action_payload_can_pin_streaming_input_mode():
    request = TtsRequest(
        text="你好",
        speaker="Serena",
        language="zh",
        input_mode="token",
        group_policy="none",
    )

    payload = build_action_payload("init", "sid-token", request=request)

    assert payload["input_mode"] == "token"
    assert payload["group_policy"] == "none"
    assert payload["speaker"] == "Serena"


def test_tokenize_for_simulation_uses_original_token_spans(monkeypatch):
    class _Tokenizer:
        def encode_with_text(self, text, add_special_tokens=False):
            assert text == "Hello world"
            assert add_special_tokens is False
            return [9707, 1879], ["Hello", " world"]

    monkeypatch.setattr(llm_pk, "_get_tokenizer", lambda: _Tokenizer())

    assert llm_pk._tokenize_for_simulation("Hello world") == [
        (9707, "Hello"),
        (1879, " world"),
    ]


def test_llm_pk_streaming_init_uses_token_mode(monkeypatch):
    sent_payloads = []

    class _Tokenizer:
        def encode_with_text(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [1, 2], ["你好", "，"]

    class _Result:
        def __init__(self, event_type="", payload=None, audio=b"", is_final=False):
            self._values = {
                "event_type": np.array([event_type], dtype=object),
                "event_json": np.array(
                    [json.dumps(payload or {}, ensure_ascii=False) if payload is not None else ""],
                    dtype=object,
                ),
                "audio_chunk": np.array([audio], dtype=object),
                "is_final": np.array([is_final], dtype=bool),
            }

        def as_numpy(self, name):
            return self._values.get(name)

    class _InferInput:
        def __init__(self, name, shape, datatype):
            self.name = name
            self.shape = shape
            self.datatype = datatype
            self.data = None

        def set_data_from_numpy(self, data):
            self.data = data

    class _RequestedOutput:
        def __init__(self, name):
            self.name = name

    class _Client:
        def __init__(self, url):
            self.url = url
            self.callback = None

        def start_stream(self, callback):
            self.callback = callback

        def stop_stream(self):
            pass

        def async_stream_infer(self, model_name, inputs, outputs):
            raw = inputs[0].data.reshape(-1)[0]
            payload = json.loads(
                raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            )
            sent_payloads.append(payload)
            action = payload.get("action", "synthesize")
            if action == "init":
                self.callback(
                    _Result(
                        "start",
                        {
                            "audio_format": {
                                "encoding": "pcm_f32",
                                "sample_rate": 24000,
                                "channels": 1,
                            }
                        },
                    ),
                    None,
                )
            elif action == "append_text":
                self.callback(
                    _Result("audio", {"meta": {"phase": "token"}}, b"\x00\x00\x00\x00"),
                    None,
                )
                self.callback(_Result(is_final=True), None)
            elif action == "text_complete":
                self.callback(
                    _Result(
                        "segment_end",
                        {
                            "text": "你好，",
                            "meta": {
                                "segment_idx": "0",
                                "audio_steps": "2",
                                "text_tokens": "2",
                            },
                        },
                    ),
                    None,
                )
                self.callback(_Result("end", {"meta": {}}, is_final=True), None)

    class _GrpcModule:
        InferInput = _InferInput
        InferRequestedOutput = _RequestedOutput
        InferenceServerClient = _Client

    monkeypatch.setattr(llm_pk, "_get_tokenizer", lambda: _Tokenizer())
    monkeypatch.setattr(llm_pk, "_load_triton_client", lambda: _GrpcModule)

    result = asyncio.run(
        llm_pk.run_streaming(
            text="你好，",
            ms_per_token=0.001,
            speaker="Serena",
            language="zh",
            endpoint="fake:8001",
            model_name="tts_orchestrator",
        )
    )

    assert sent_payloads[0]["action"] == "init"
    assert sent_payloads[0]["input_mode"] == "token"
    assert sent_payloads[0]["group_policy"] == "none"
    append_texts = [
        payload.get("text")
        for payload in sent_payloads
        if payload.get("action") == "append_text"
    ]
    assert append_texts == [
        "你好",
        "，",
    ]
    assert result.metrics.chunks == 2
