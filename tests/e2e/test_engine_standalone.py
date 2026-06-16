#!/usr/bin/env python3
"""
E2E test & benchmark for the standalone TTS engine (gRPC on port 50051).

Tests the engine with:
  1. Single-request smoke test
  2. Streaming text input (init -> text chunks -> text_complete)
  2a. Strict token-scale streaming input
  2b. CustomVoice + instruct (preset speaker + style instruction; skipped on 0.6b)
  3. Multi-session concurrent requests (1, 2, 4 sessions)
  4. Long text rollover (medium / very long / streaming long)
  5. BadCase tests (empty text, whitespace, invalid task_type, etc.)
  6. Performance metrics: first-chunk latency, total latency, RTF, throughput

Pytest usage:
    python -m engine.server --config engine.yaml   # terminal 1
    pytest tests/e2e/test_engine_standalone.py -v -s

Manual benchmark usage:
    python tests/tools/engine_standalone_benchmark.py --host localhost --port 50051
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.engine_standalone import (
    CUSTOM_VOICE_INSTRUCT_EN,
    CUSTOM_VOICE_INSTRUCT_ZH,
    DEFAULT_TOKEN_STREAM_TEXT,
    GRPC_HOST,
    GRPC_PORT,
    LONG_TEXT,
    SAMPLE_RATE,
    VERY_LONG_TEXT,
    _GATEWAY_IMPORT_ERROR,
    _build_token_stream_chunks_or_skip,
    _check_server,
    _custom_voice_instruct_supported,
    _get_capabilities,
    _require_gateway,
    _synthesize_oneshot,
    _synthesize_streaming,
    _test_cancel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def engine_addr():
    if _GATEWAY_IMPORT_ERROR is not None:
        pytest.skip(f"engine gateway protobuf import failed: {_GATEWAY_IMPORT_ERROR}")
    if not _check_server(GRPC_HOST, GRPC_PORT):
        pytest.skip(
            f"Standalone engine not reachable at {GRPC_HOST}:{GRPC_PORT}. "
            f"Start with: python -m engine.server --config engine.yaml"
        )
    return GRPC_HOST, GRPC_PORT


class TestEngineSmokeAndStreaming:
    """Smoke + streaming tests against the standalone engine."""

    def test_get_capabilities(self, engine_addr):
        host, port = engine_addr
        cap = _get_capabilities(host, port)
        assert cap["loaded_model_type"]
        assert cap["supported_audio_formats"]

    def test_single_smoke(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="你好，这是一个测试。", speaker="Serena")
        assert r.error is None, f"Synthesis failed: {r.error}"
        assert r.num_chunks >= 1
        assert r.total_samples >= SAMPLE_RATE * 0.1

    def test_english(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="Hello, how are you today?", speaker="Serena")
        assert r.error is None, f"Synthesis failed: {r.error}"
        assert r.total_samples >= SAMPLE_RATE * 0.1

    def test_streaming_text(self, engine_addr):
        host, port = engine_addr
        pb2, _ = _require_gateway()
        r = asyncio.run(
            _synthesize_streaming(
                host,
                port,
                init_speaker="Serena",
                init_task_type="custom_voice",
                input_mode=pb2.INPUT_MODE_CLAUSE,
                text_chunks=["你好，", "这是流式测试。"],
                chunk_delay_ms=100,
            )
        )
        assert r.error is None, f"Streaming synthesis failed: {r.error}"
        assert r.total_samples >= SAMPLE_RATE * 0.1

    def test_streaming_token_text(self, engine_addr):
        host, port = engine_addr
        pb2, _ = _require_gateway()
        token_chunks = _build_token_stream_chunks_or_skip()
        r = asyncio.run(
            _synthesize_streaming(
                host,
                port,
                init_speaker="Serena",
                init_task_type="custom_voice",
                input_mode=pb2.INPUT_MODE_TOKEN,
                group_policy=pb2.GROUP_POLICY_NONE,
                text_chunks=token_chunks,
                chunk_delay_ms=30,
                session_id="pytest-stream-token",
            )
        )
        assert r.error is None, f"TOKEN-mode streaming synthesis failed: {r.error}"
        assert r.text == DEFAULT_TOKEN_STREAM_TEXT
        assert r.total_samples >= SAMPLE_RATE * 0.1


class TestEngineCustomVoiceInstruct:
    """CustomVoice + instruct (skipped for non-custom_voice or 0.6b)."""

    def test_custom_instruct_oneshot(self, engine_addr):
        host, port = engine_addr
        if not _custom_voice_instruct_supported(host, port):
            pytest.skip("custom instruct needs custom_voice 1.7b+ (not 0.6b)")
        r = _synthesize_oneshot(
            host,
            port,
            text="你好，带风格指令的测试。",
            speaker="Serena",
            instruct=CUSTOM_VOICE_INSTRUCT_ZH,
        )
        assert r.error is None, f"Synthesis failed: {r.error}"
        assert r.total_samples >= SAMPLE_RATE * 0.1

    def test_custom_instruct_streaming(self, engine_addr):
        host, port = engine_addr
        pb2, _ = _require_gateway()
        if not _custom_voice_instruct_supported(host, port):
            pytest.skip("custom instruct needs custom_voice 1.7b+ (not 0.6b)")
        r = asyncio.run(
            _synthesize_streaming(
                host,
                port,
                init_speaker="Ethan",
                init_task_type="custom_voice",
                init_instruct=CUSTOM_VOICE_INSTRUCT_EN,
                input_mode=pb2.INPUT_MODE_CLAUSE,
                text_chunks=["Hello, ", "instruct streaming test."],
                chunk_delay_ms=80,
            )
        )
        assert r.error is None, f"Streaming failed: {r.error}"
        assert r.total_samples >= SAMPLE_RATE * 0.1


class TestEngineLongText:
    """Long text rollover tests."""

    def test_medium_long(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text=LONG_TEXT, speaker="Serena", timeout=180)
        assert r.error is None, f"Long text failed: {r.error}"
        assert r.duration_sec >= 1.0, f"Audio too short: {r.duration_sec:.2f}s"

    def test_very_long(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text=VERY_LONG_TEXT, speaker="Serena", timeout=300)
        assert r.error is None, f"Very long text failed: {r.error}"
        assert r.duration_sec >= 3.0, f"Audio too short: {r.duration_sec:.2f}s"


class TestEngineBadCases:
    """Boundary conditions and error handling."""

    def test_empty_text(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="", speaker="Serena", timeout=15)
        assert r.error is not None or r.total_samples == 0

    def test_whitespace_text(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="   \n\t  ", speaker="Serena", timeout=15)
        assert r.error is not None or r.total_samples == 0

    def test_single_char(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="好", speaker="Serena", timeout=30)
        assert r.error is None
        assert r.total_samples > 0

    def test_cancel_stream(self, engine_addr):
        host, port = engine_addr
        r = _test_cancel(host, port)
        assert r.error is None


class TestEnginePerformance:
    """Performance baseline (relaxed thresholds for CI)."""

    def test_first_chunk_latency(self, engine_addr):
        host, port = engine_addr
        r = _synthesize_oneshot(host, port, text="今天天气真好。", speaker="Serena")
        assert r.error is None
        assert r.first_chunk_ms is not None
        print(f"\n  first_chunk={r.first_chunk_ms:.0f}ms  RTF={r.rtf:.2f}")
        assert r.first_chunk_ms < 30_000, f"First chunk too slow: {r.first_chunk_ms:.0f}ms"
