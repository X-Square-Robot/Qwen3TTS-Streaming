#!/usr/bin/env python3
"""Generate engine audio for quality comparison with reference (PyTorch) output.

Calls the running engine via gRPC for the same texts as gen_reference_audio.py,
saves WAV files side by side for A/B listening.

Usage:
  python tests/tools/gen_engine_audio.py
"""

import sys
import time
import uuid

import grpc
import numpy as np

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from qwen3tts_tools.audio import save_wav
from qwen3tts_tools.common import REPO_ROOT

from engine.gateway import tts_pb2, tts_pb2_grpc

SAMPLE_RATE = 24000
HOST = "localhost"
PORT = 50051

TEXTS = {
    "test1": "你好，这是单路测试。",
    "test2": "你好，这是流式文本输入测试。",
    "test3": "你好，今天天气真好。",
    "test4": "欢迎来到人工智能语音合成的世界。",
}
def _audio_chunk_to_f32(audio_chunk) -> np.ndarray:
    encoding = getattr(audio_chunk, "encoding", tts_pb2.AUDIO_ENCODING_PCM_F32)
    if encoding == tts_pb2.AUDIO_ENCODING_PCM_S16LE:
        return np.frombuffer(audio_chunk.pcm_data, dtype=np.int16).astype(np.float32) / 32767.0
    return np.frombuffer(audio_chunk.pcm_data, dtype=np.float32)


def _is_terminal_event(resp) -> tuple[bool, str]:
    which = resp.WhichOneof("response")
    if which == "event":
        if resp.event.type == "error":
            return True, resp.event.message
        if resp.event.type in ("done", "end"):
            return True, ""
    elif which == "status":
        if resp.status.event == "error":
            return True, resp.status.message
        if resp.status.event == "done":
            return True, ""
    return False, ""


def synthesize(text: str, speaker: str = "Serena") -> tuple[np.ndarray | None, float, int]:
    sid = uuid.uuid4().hex[:12]
    channel = grpc.insecure_channel(f"{HOST}:{PORT}")
    stub = tts_pb2_grpc.TTSServiceStub(channel)

    chunks = []
    first_ts = None
    t0 = time.perf_counter()
    try:
        request = tts_pb2.SynthesizeOnceRequest(
            session_id=sid,
            text=text,
            config=tts_pb2.SessionConfig(
                task_type="custom_voice",
                speaker=speaker,
                input_mode=tts_pb2.INPUT_MODE_FULL_TEXT,
                group_policy=tts_pb2.GROUP_POLICY_AUTO,
                audio=tts_pb2.AudioFormat(
                    encoding=tts_pb2.AUDIO_ENCODING_PCM_F32,
                    sample_rate=SAMPLE_RATE,
                    channels=1,
                ),
            ),
        )
        for resp in stub.SynthesizeOnce(request, timeout=60):
            which = resp.WhichOneof("response")
            if which == "audio":
                if first_ts is None:
                    first_ts = time.perf_counter()
                chunks.append(_audio_chunk_to_f32(resp.audio))
            else:
                done, message = _is_terminal_event(resp)
                if message:
                    print(f"  ERROR: {message}")
                if done:
                    break
    except grpc.RpcError as e:
        print(f"  gRPC error: {e.code().name}: {e.details()}")
    finally:
        channel.close()

    elapsed = time.perf_counter() - t0
    if chunks:
        audio = np.concatenate(chunks)
        return audio, elapsed, len(chunks)
    return None, elapsed, 0


def main():
    out_dir = REPO_ROOT / "workspace" / "audio_samples" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    channel = grpc.insecure_channel(f"{HOST}:{PORT}")
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    except grpc.FutureTimeoutError:
        print(f"Engine not reachable at {HOST}:{PORT}")
        sys.exit(1)
    finally:
        channel.close()

    print("Engine connected. Generating audio...")

    for name, text in TEXTS.items():
        print(f"\nGenerating [{name}]: {text}")
        audio, elapsed, n_chunks = synthesize(text)
        if audio is not None:
            duration = len(audio) / SAMPLE_RATE
            out_path = out_dir / f"engine_{name}.wav"
            save_wav(audio, out_path, sample_rate=SAMPLE_RATE)
            print(f"  -> {out_path.name}: {len(audio)} samples, {duration:.2f}s, "
                  f"{n_chunks} chunks, took {elapsed:.1f}s")
        else:
            print(f"  -> FAILED: no audio returned")

    print(f"\nAll engine audio saved to: {out_dir}")
    print("Compare proto_*.wav (PyTorch) vs engine_*.wav (TRT engine):")
    for name in TEXTS:
        proto = out_dir / f"proto_{name}.wav"
        engine = out_dir / f"engine_{name}.wav"
        p_size = proto.stat().st_size if proto.exists() else 0
        e_size = engine.stat().st_size if engine.exists() else 0
        print(f"  {name}: proto={p_size:,}B  engine={e_size:,}B")


if __name__ == "__main__":
    main()
