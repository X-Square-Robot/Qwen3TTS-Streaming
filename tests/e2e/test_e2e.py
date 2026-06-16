"""
L3 E2E tests: TTS Orchestrator via Triton gRPC (T3.1, T3.3, T3.4).

Requires Triton server running with tts_orchestrator loaded, e.g.:
  bash scripts/bash/build_triton.sh assemble --engine-mode onnx --variant custom-1.7b
  bash scripts/bash/build_triton.sh run

Production assemble includes talker_code2wav_fused (+ optional speaker_encoder,
speech_tokenizer_codec_fused for base ICL). Legacy talker_unified + code2wav is optional.

Skip all tests if server not reachable at localhost:8001.
"""
import json
import time
from pathlib import Path

import numpy as np
import pytest
from tests.support.triton_streaming import (
    StreamResult,
    build_request_payload,
    infer_stream,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Default gRPC port
GRPC_HOST = "localhost"
GRPC_PORT = 8001


def _grpc_client():
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        pytest.skip("tritonclient[grpc] not installed")
    return grpcclient


def _server_ready():
    try:
        import tritonclient.grpc as grpcclient
        c = grpcclient.InferenceServerClient(url=f"{GRPC_HOST}:{GRPC_PORT}")
        return c.is_server_ready()
    except Exception:
        return False


@pytest.fixture(scope="module")
def client():
    _grpc_client()
    if not _server_ready():
        pytest.skip(f"Triton server not ready at {GRPC_HOST}:{GRPC_PORT}")
    import tritonclient.grpc as grpcclient
    return grpcclient.InferenceServerClient(url=f"{GRPC_HOST}:{GRPC_PORT}")


def _stream_tts(client, req_dict, timeout=60):
    """Send request, collect streaming audio chunks. Returns (chunks_list, first_chunk_sec, total_sec, error_msg)."""
    grpcclient = _grpc_client()
    stream = infer_stream(client, grpcclient, req_dict, timeout=timeout)
    chunks = [stream.audio] if stream.audio is not None and stream.audio.size else []
    first_sec = (stream.first_chunk_ms / 1000.0) if stream.first_chunk_ms is not None else None
    total = stream.total_ms / 1000.0
    return chunks, first_sec, total, stream.error


def _minimal_wav_base64(duration_sec=0.5, sample_rate=16000):
    """Minimal 16kHz mono WAV as base64 (for voice_clone ref_audio)."""
    n = int(duration_sec * sample_rate)
    samples = np.zeros(n, dtype=np.float32)
    import struct
    import base64
    import io
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + n * 2))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", n * 2))
    buf.write(struct.pack("<%dh" % n, *(0,) * n))
    return base64.b64encode(buf.getvalue()).decode()


# ---- T3.1 ONNX/TRT backend E2E ----
# Note: custom-1.7b supports custom_voice only; design-1.7b supports voice_design.
# Tests use custom_voice (works with custom-1.7b). For voice_design, deploy design-1.7b.


def test_e2e_voice_design_or_custom(client):
    """T3.1a: voice_design (design variant) or custom_voice (custom variant), streaming audio."""
    # Prefer custom_voice (works with custom-1.7b); voice_design only for design-1.7b
    chunks, first_sec, total, err = _stream_tts(client, {
        "text": "你好，这是测试",
        "task_type": "custom_voice",
        "speaker": "zhitian",
    })
    assert err is None, f"Request failed: {err}"
    assert len(chunks) >= 1, "Expected at least one audio chunk"
    full = np.concatenate(chunks)
    assert full.size >= 1
    assert first_sec is not None
    print(f"\n[E2E T3.1a] voice_design first_chunk_s={first_sec:.3f} total_s={total:.3f} samples={full.size}")
    # T3.4: first chunk latency < 200ms (relaxed for CI)
    assert first_sec < 30.0, f"First chunk too slow: {first_sec:.2f}s"


def test_e2e_custom_voice(client):
    """T3.1b: custom_voice with speaker + instruct."""
    chunks, _, _, err = _stream_tts(client, {
        "text": "你好",
        "task_type": "custom_voice",
        "speaker": "zhitian",
        "instruct": "温柔",
    })
    assert err is None, f"Request failed: {err}"
    assert len(chunks) >= 1
    assert np.concatenate(chunks).size >= 1


@pytest.mark.skip(reason="voice_clone requires valid ref_audio; use manual test with real audio")
def test_e2e_voice_clone_xvec(client):
    """T3.1c: voice_clone with ref_audio, x_vector_only."""
    ref_b64 = _minimal_wav_base64()
    chunks, _, _, err = _stream_tts(client, {
        "text": "你好",
        "task_type": "voice_clone",
        "ref_audio": ref_b64,
        "x_vector_only": True,
    })
    assert err is None
    assert len(chunks) >= 1


@pytest.mark.skip(reason="voice_clone_icl requires valid ref_audio; use manual test")
def test_e2e_voice_clone_icl(client):
    """T3.1d: voice_clone_icl with ref_audio + ref_text."""
    ref_b64 = _minimal_wav_base64()
    chunks, _, _, err = _stream_tts(client, {
        "text": "你好",
        "task_type": "voice_clone",
        "ref_audio": ref_b64,
        "ref_text": "参考文本",
    })
    assert err is None
    assert len(chunks) >= 1


# ---- T3.3 Error handling ----

def test_e2e_error_empty_text(client):
    """T3.3a: empty text -> server returns error."""
    _, _, _, err = _stream_tts(client, {"text": "", "task_type": "custom_voice", "speaker": "zhitian"})
    assert err is not None
    assert "text" in err.lower() or "required" in err.lower() or "empty" in err.lower()


def test_e2e_error_invalid_task_type(client):
    """T3.3b: invalid task_type -> server returns error."""
    _, _, _, err = _stream_tts(client, {"text": "你好", "task_type": "invalid_type"})
    assert err is not None


def test_e2e_error_voice_clone_no_ref_audio(client):
    """T3.3c: voice_clone without ref_audio -> server returns error."""
    _, _, _, err = _stream_tts(client, {"text": "你好", "task_type": "voice_clone"})
    assert err is not None
    assert "ref_audio" in err.lower() or "required" in err.lower() or "voice_clone" in err.lower()


def test_e2e_error_voice_clone_bad_base64(client):
    """T3.3d: voice_clone with invalid ref_audio base64 -> RuntimeError."""
    _, _, _, err = _stream_tts(client, {
        "text": "你好",
        "task_type": "voice_clone",
        "ref_audio": "not_valid_base64!!!",
    })
    assert err is not None


# ---- T3.4 Performance baseline ----

def test_e2e_first_chunk_latency(client):
    """T3.4: first audio chunk latency (target < 200ms in prod)."""
    chunks, first_sec, total_sec, err = _stream_tts(client, {
        "text": "今天天气真好。",
        "task_type": "custom_voice",
        "speaker": "zhitian",
    })
    assert err is None
    assert len(chunks) >= 1
    assert first_sec is not None
    # Log metrics for E2E report; assertion relaxed for CI
    first_ms = first_sec * 1000
    total_ms = total_sec * 1000
    samples = sum(c.size for c in chunks)
    print(f"\n[E2E T3.4] first_chunk_latency_ms={first_ms:.0f} total_ms={total_ms:.0f} chunks={len(chunks)} samples={samples}")
    assert first_sec < 15.0, f"First chunk latency {first_ms:.0f}ms (target < 200ms in prod)"
