#!/usr/bin/env python3
"""Shared standalone-engine support for pytest E2E and manual tools."""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports("scripts_python", "repo")

from qwen3tts_tools.audio import save_wav
from qwen3tts_tools.common import (
    DEFAULT_ENGINE_GRPC,
    DEFAULT_SAMPLE_RATE,
    parse_host_port,
    story_path as default_story_path,
)
from tests.support.token_streaming import (
    DEFAULT_TOKEN_STREAM_TEXT,
    TokenChunkingUnavailable,
    build_token_text_chunks,
)

try:
    from engine.gateway import tts_pb2, tts_pb2_grpc
except Exception as exc:  # protobuf runtime mismatches also surface here.
    tts_pb2 = None
    tts_pb2_grpc = None
    _GATEWAY_IMPORT_ERROR = exc
else:
    _GATEWAY_IMPORT_ERROR = None

GRPC_HOST, GRPC_PORT = parse_host_port(DEFAULT_ENGINE_GRPC, default_port=50051)
SAMPLE_RATE = DEFAULT_SAMPLE_RATE
OUTPUT_DIR = REPO_ROOT / "workspace" / "audio_samples" / "engine"


def _require_gateway():
    if _GATEWAY_IMPORT_ERROR is not None:
        raise RuntimeError(f"engine gateway protobuf import failed: {_GATEWAY_IMPORT_ERROR}")
    return tts_pb2, tts_pb2_grpc


@dataclass
class TTSResult:
    session_id: str
    text: str
    first_chunk_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    start_to_first_audio_ms: Optional[float] = None
    total_ms: float = 0.0
    num_chunks: int = 0
    total_samples: int = 0
    error: Optional[str] = None
    audio: Optional[np.ndarray] = None
    warnings: list = field(default_factory=list)
    audio_chunk_intervals_ms: list[float] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return self.total_samples / SAMPLE_RATE if self.total_samples > 0 else 0.0

    @property
    def rtf(self) -> float:
        if self.duration_sec <= 0 or self.total_ms <= 0:
            return 0.0
        return (self.total_ms / 1000) / self.duration_sec

    @property
    def decode_step_mean_ms(self) -> Optional[float]:
        if not self.audio_chunk_intervals_ms:
            return None
        return statistics.mean(self.audio_chunk_intervals_ms)

    @property
    def decode_step_p50_ms(self) -> Optional[float]:
        if not self.audio_chunk_intervals_ms:
            return None
        return statistics.median(self.audio_chunk_intervals_ms)

    @property
    def decode_step_p95_ms(self) -> Optional[float]:
        if not self.audio_chunk_intervals_ms:
            return None
        vals = sorted(self.audio_chunk_intervals_ms)
        idx = min(len(vals) - 1, max(0, int(round(0.95 * (len(vals) - 1)))))
        return vals[idx]


def _compute_intervals_ms(timestamps: list[float]) -> list[float]:
    if len(timestamps) < 2:
        return []
    return [
        (timestamps[i] - timestamps[i - 1]) * 1000.0
        for i in range(1, len(timestamps))
    ]


TEST_TEXTS = [
    "你好，今天天气真好。",
    "欢迎来到人工智能语音合成的世界。",
    "技术创新推动着社会不断前进。",
    "我们正在测试多路并发的语音合成能力。",
    "这是第五路测试文本，用来验证系统的处理能力。",
    "深度学习让机器能够理解和生成自然的语音。",
    "云计算和边缘计算相结合，提供更好的用户体验。",
    "每一次迭代都让我们的系统变得更加完善。",
]

LONG_TEXT = (
    "人工智能正在深刻改变我们的世界。"
    "从语音识别到自然语言处理，从计算机视觉到机器人技术，"
    "AI的应用已经渗透到生活的方方面面。"
    "在医疗领域，AI可以辅助诊断疾病、发现新药物。"
    "在教育领域，AI可以提供个性化的学习方案。"
    "在交通领域，自动驾驶技术正在逐步成熟。"
    "未来，人工智能将继续推动社会进步，"
    "为人类创造更多的可能性。"
)

CUSTOM_VOICE_INSTRUCT_ZH = "用温柔、舒缓的语气朗读。"
CUSTOM_VOICE_INSTRUCT_EN = "Speak in a calm and friendly tone."

VERY_LONG_TEXT = (
    "在遥远的古代，人类就开始仰望星空，思考宇宙的奥秘。"
    "从古希腊的哲学家到中国的天文学家，人们不断探索着这个世界的本质。"
    "随着科学技术的发展，我们对宇宙的认知越来越深入。"
    "从伽利略的望远镜到哈勃太空望远镜，每一次技术的突破都让我们看到了更广阔的世界。"
    "如今，人工智能技术的发展为科学研究带来了前所未有的机遇。"
    "机器学习算法可以处理海量的天文数据，发现人眼无法察觉的规律。"
    "深度学习网络能够分析复杂的光谱数据，帮助我们理解恒星的演化过程。"
    "在医学领域，AI辅助诊断系统已经在某些疾病的识别上达到了专家水平。"
    "在交通出行方面，自动驾驶技术正在逐步走向成熟，有望彻底改变我们的出行方式。"
    "在教育领域，个性化学习系统能够根据每个学生的特点量身定制学习方案。"
    "在金融领域，智能风控系统可以实时监测交易异常，保护用户的资金安全。"
    "然而，技术的发展也带来了新的挑战和思考。"
    "如何确保人工智能的安全性和可控性？"
    "如何在自动化浪潮中保障就业和社会公平？"
    "这些问题需要我们全社会共同面对和解决。"
    "只有在技术进步与人文关怀之间找到平衡，"
    "才能让人工智能真正造福全人类，"
    "创造一个更加美好的未来。"
)


def _pcm_bytes_to_f32(pcm_data: bytes) -> np.ndarray:
    return np.frombuffer(pcm_data, dtype=np.float32)


def _audio_chunk_to_f32(audio_chunk) -> np.ndarray:
    pb2, _ = _require_gateway()
    encoding = getattr(audio_chunk, "encoding", pb2.AUDIO_ENCODING_PCM_F32)
    if encoding == pb2.AUDIO_ENCODING_PCM_S16LE:
        return np.frombuffer(audio_chunk.pcm_data, dtype=np.int16).astype(np.float32) / 32767.0
    return np.frombuffer(audio_chunk.pcm_data, dtype=np.float32)


def _handle_stream_control(resp, result: TTSResult) -> bool:
    which = resp.WhichOneof("response")
    if which == "event":
        et = resp.event.type
        if et == "warning" and resp.event.message:
            result.warnings.append(resp.event.message)
        if et == "error":
            result.error = resp.event.message
            return True
        if et in ("done", "end"):
            return True
        return False
    if which == "status":
        if resp.status.event == "error":
            result.error = resp.status.message
            return True
        if resp.status.event == "done":
            return True
    return False


def _make_session_config(
    *,
    task_type: str,
    speaker: str = "",
    instruct: str = "",
    input_mode=None,
    group_policy=None,
    sample_rate: int = SAMPLE_RATE,
    encoding=None,
):
    pb2, _ = _require_gateway()
    if input_mode is None:
        input_mode = pb2.INPUT_MODE_LONG_SEGMENT
    if group_policy is None:
        group_policy = pb2.GROUP_POLICY_AUTO
    if encoding is None:
        encoding = pb2.AUDIO_ENCODING_PCM_F32
    return pb2.SessionConfig(
        task_type=task_type,
        speaker=speaker,
        instruct=instruct or "",
        input_mode=input_mode,
        group_policy=group_policy,
        audio=pb2.AudioFormat(
            encoding=encoding,
            sample_rate=sample_rate,
            channels=1,
        ),
    )


def _save_wav(audio: np.ndarray, path: str | Path):
    save_wav(audio, path, sample_rate=SAMPLE_RATE)


def _check_server(host: str, port: int) -> bool:
    import grpc

    channel = grpc.insecure_channel(f"{host}:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=3)
        return True
    except grpc.FutureTimeoutError:
        return False
    finally:
        channel.close()


def _get_capabilities(host: str, port: int) -> dict:
    pb2, pb2_grpc = _require_gateway()
    import grpc

    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = pb2_grpc.TTSServiceStub(channel)
    try:
        resp = stub.GetCapabilities(pb2.GetCapabilitiesRequest(), timeout=10)
        return {
            "variant": resp.variant,
            "loaded_model_type": resp.loaded_model_type,
            "declared_supported_task_types": list(resp.declared_supported_task_types),
            "supported_input_modes": list(resp.supported_input_modes),
            "supported_group_policies": list(resp.supported_group_policies),
            "supported_audio_formats": [
                (fmt.encoding, fmt.sample_rate, fmt.channels)
                for fmt in resp.supported_audio_formats
            ],
            "ref_audio_available": resp.ref_audio_available,
            "ref_audio_reason": resp.ref_audio_reason,
        }
    finally:
        channel.close()


def _custom_voice_instruct_supported(host: str, port: int) -> bool:
    cap = _get_capabilities(host, port)
    mtype = (cap.get("loaded_model_type") or "").strip()
    if mtype != "custom_voice":
        return False
    v = (cap.get("variant") or "").lower()
    if "0.6" in v or "0b6" in v:
        return False
    return True


def _synthesize_oneshot(
    host: str,
    port: int,
    text: str,
    speaker: str = "Serena",
    task_type: str = "custom_voice",
    instruct: str = "",
    session_id: str = "",
    timeout: float = 120.0,
) -> TTSResult:
    pb2, pb2_grpc = _require_gateway()
    sid = session_id or uuid.uuid4().hex[:12]
    result = TTSResult(session_id=sid, text=text)

    import grpc

    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = pb2_grpc.TTSServiceStub(channel)

    chunks = []
    first_ts = None
    chunk_timestamps: list[float] = []

    t0 = time.perf_counter()
    request_sent_ts = t0
    try:
        request = pb2.SynthesizeOnceRequest(
            session_id=sid,
            text=text,
            config=_make_session_config(
                task_type=task_type,
                speaker=speaker,
                instruct=instruct,
                input_mode=pb2.INPUT_MODE_FULL_TEXT,
            ),
        )
        for resp in stub.SynthesizeOnce(request, timeout=timeout):
            which = resp.WhichOneof("response")
            if which == "audio":
                if first_ts is None:
                    first_ts = time.perf_counter()
                    chunk_timestamps.append(first_ts)
                else:
                    chunk_timestamps.append(time.perf_counter())
                chunks.append(_audio_chunk_to_f32(resp.audio))
            elif _handle_stream_control(resp, result):
                break
    except grpc.RpcError as e:
        result.error = f"gRPC {e.code().name}: {e.details()}"
    except Exception as e:
        result.error = str(e)
    finally:
        channel.close()

    elapsed = time.perf_counter() - t0
    result.total_ms = elapsed * 1000
    result.first_chunk_ms = (first_ts - t0) * 1000 if first_ts else None
    result.start_to_first_audio_ms = result.first_chunk_ms
    result.ttft_ms = (first_ts - request_sent_ts) * 1000 if first_ts else None
    result.audio_chunk_intervals_ms = _compute_intervals_ms(chunk_timestamps)
    result.num_chunks = len(chunks)
    if chunks:
        result.audio = np.concatenate(chunks)
        result.total_samples = result.audio.size

    return result


async def _synthesize_streaming(
    host: str,
    port: int,
    init_speaker: str,
    init_task_type: str,
    text_chunks: list[str],
    init_instruct: str = "",
    input_mode=None,
    group_policy=None,
    chunk_delay_ms: float = 50.0,
    session_id: str = "",
    timeout: float = 120.0,
) -> TTSResult:
    pb2, pb2_grpc = _require_gateway()
    if input_mode is None:
        input_mode = pb2.INPUT_MODE_LONG_SEGMENT
    if group_policy is None:
        group_policy = pb2.GROUP_POLICY_AUTO
    sid = session_id or uuid.uuid4().hex[:12]
    result = TTSResult(session_id=sid, text="".join(text_chunks))

    import grpc.aio as grpc_aio

    channel = grpc_aio.insecure_channel(f"{host}:{port}")
    stub = pb2_grpc.TTSServiceStub(channel)

    chunks: List[np.ndarray] = []
    first_ts = None
    chunk_timestamps: list[float] = []
    send_marks: dict[str, Optional[float]] = {
        "start_sent_ts": None,
        "first_text_sent_ts": None,
    }

    async def request_gen():
        send_marks["start_sent_ts"] = time.perf_counter()
        yield pb2.SynthesizeRequest(
            start=pb2.StartRequest(
                session_id=sid,
                config=_make_session_config(
                    task_type=init_task_type,
                    speaker=init_speaker,
                    instruct=init_instruct,
                    input_mode=input_mode,
                    group_policy=group_policy,
                ),
            )
        )
        for chunk_text in text_chunks:
            if chunk_delay_ms > 0:
                await asyncio.sleep(chunk_delay_ms / 1000)
            now = time.perf_counter()
            if send_marks["first_text_sent_ts"] is None:
                send_marks["first_text_sent_ts"] = now
            yield pb2.SynthesizeRequest(text=pb2.TextChunk(text=chunk_text))
        if chunk_delay_ms > 0:
            await asyncio.sleep(chunk_delay_ms / 1000)
        yield pb2.SynthesizeRequest(end=pb2.EndRequest())

    t0 = time.perf_counter()
    try:
        response_stream = stub.SynthesizeStream(request_gen(), timeout=timeout)
        async for resp in response_stream:
            which = resp.WhichOneof("response")
            if which == "audio":
                if first_ts is None:
                    first_ts = time.perf_counter()
                    chunk_timestamps.append(first_ts)
                else:
                    chunk_timestamps.append(time.perf_counter())
                chunks.append(_audio_chunk_to_f32(resp.audio))
            elif _handle_stream_control(resp, result):
                break
    except Exception as e:
        result.error = str(e)
    finally:
        await channel.close()

    elapsed = time.perf_counter() - t0
    result.total_ms = elapsed * 1000
    result.first_chunk_ms = (first_ts - t0) * 1000 if first_ts else None
    result.start_to_first_audio_ms = (
        (first_ts - send_marks["start_sent_ts"]) * 1000
        if first_ts and send_marks["start_sent_ts"] is not None else None
    )
    result.ttft_ms = (
        (first_ts - send_marks["first_text_sent_ts"]) * 1000
        if first_ts and send_marks["first_text_sent_ts"] is not None else None
    )
    result.audio_chunk_intervals_ms = _compute_intervals_ms(chunk_timestamps)
    result.num_chunks = len(chunks)
    if chunks:
        result.audio = np.concatenate(chunks)
        result.total_samples = result.audio.size

    return result


def _print_result(result: TTSResult, label: str = ""):
    prefix = f"  [{label}]" if label else "  "
    if result.error:
        print(f"{prefix} ERROR: {result.error}")
        return
    if result.warnings:
        for warning in result.warnings:
            print(f"{prefix} WARNING: {warning}")
    fc = f"{result.first_chunk_ms:.0f}ms" if result.first_chunk_ms is not None else "N/A"
    ttft = f"{result.ttft_ms:.0f}ms" if result.ttft_ms is not None else "N/A"
    start_to_first = (
        f"{result.start_to_first_audio_ms:.0f}ms"
        if result.start_to_first_audio_ms is not None else "N/A"
    )
    step_mean = (
        f"{result.decode_step_mean_ms:.1f}ms"
        if result.decode_step_mean_ms is not None else "N/A"
    )
    step_p50 = (
        f"{result.decode_step_p50_ms:.1f}ms"
        if result.decode_step_p50_ms is not None else "N/A"
    )
    step_p95 = (
        f"{result.decode_step_p95_ms:.1f}ms"
        if result.decode_step_p95_ms is not None else "N/A"
    )
    print(
        f"{prefix} session={result.session_id}"
        f"  first_chunk={fc}"
        f"  ttft={ttft}"
        f"  start_to_first_audio={start_to_first}"
        f"  decode_step_mean={step_mean}"
        f"  p50={step_p50}"
        f"  p95={step_p95}"
        f"  total={result.total_ms:.0f}ms"
        f"  chunks={result.num_chunks}"
        f"  audio={result.duration_sec:.2f}s"
        f"  RTF={result.rtf:.2f}"
    )


def _print_summary(results: list[TTSResult], label: str):
    ok = [result for result in results if result.error is None and result.first_chunk_ms is not None]
    fail = [result for result in results if result.error is not None]
    print(f"\n  -- {label} Summary --")
    print(f"  Total: {len(results)}  OK: {len(ok)}  Failed: {len(fail)}")
    if not ok:
        return
    first_chunks = [result.first_chunk_ms for result in ok]
    ttfts = [result.ttft_ms for result in ok if result.ttft_ms is not None]
    totals = [result.total_ms for result in ok]
    rtfs = [result.rtf for result in ok if result.rtf > 0]
    durations = [result.duration_sec for result in ok]
    decode_steps = [
        step
        for result in ok
        for step in result.audio_chunk_intervals_ms
    ]
    print(
        f"  First-chunk latency:  min={min(first_chunks):.0f}ms  "
        f"median={statistics.median(first_chunks):.0f}ms  "
        f"max={max(first_chunks):.0f}ms  "
        f"mean={statistics.mean(first_chunks):.0f}ms"
    )
    if ttfts:
        print(
            f"  TTFT:                 min={min(ttfts):.0f}ms  "
            f"median={statistics.median(ttfts):.0f}ms  "
            f"max={max(ttfts):.0f}ms  "
            f"mean={statistics.mean(ttfts):.0f}ms"
        )
    print(
        f"  Total latency:        min={min(totals):.0f}ms  "
        f"median={statistics.median(totals):.0f}ms  "
        f"max={max(totals):.0f}ms  "
        f"mean={statistics.mean(totals):.0f}ms"
    )
    if decode_steps:
        print(
            f"  Decode-step interval: min={min(decode_steps):.1f}ms  "
            f"p50={statistics.median(decode_steps):.1f}ms  "
            f"p95={sorted(decode_steps)[min(len(decode_steps) - 1, max(0, int(round(0.95 * (len(decode_steps) - 1)))) )]:.1f}ms  "
            f"mean={statistics.mean(decode_steps):.1f}ms"
        )
    if rtfs:
        print(
            f"  RTF (wall/audio):     min={min(rtfs):.2f}  "
            f"median={statistics.median(rtfs):.2f}  "
            f"max={max(rtfs):.2f}"
        )
    total_audio = sum(durations)
    wall_time = max(totals) / 1000
    effective_throughput = total_audio / wall_time if wall_time > 0 else 0
    print(
        f"  Aggregate throughput: {total_audio:.1f}s audio / {wall_time:.1f}s wall "
        f"= {effective_throughput:.2f}x realtime"
    )


def _p95(values: list[float]) -> float:
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round(0.95 * (len(vals) - 1)))))
    return vals[idx]


def run_single_smoke(host: str, port: int, output_dir: Path) -> TTSResult:
    print("\n" + "=" * 60)
    print("  Test 1: Single Request Smoke Test")
    print("=" * 60)

    result = _synthesize_oneshot(
        host,
        port,
        text="你好，这是单路测试。",
        speaker="Serena",
        session_id="smoke-single",
    )
    _print_result(result, "single")
    if result.audio is not None and result.audio.size > 0:
        out_path = output_dir / "test1_single_smoke.wav"
        _save_wav(result.audio, out_path)
        print(f"  Saved: {out_path}")

    return result


def run_streaming_text(host: str, port: int, output_dir: Path) -> TTSResult:
    pb2, _ = _require_gateway()
    print("\n" + "=" * 60)
    print("  Test 2: Streaming Text Input")
    print("=" * 60)

    result = asyncio.run(
        _synthesize_streaming(
            host,
            port,
            init_speaker="Serena",
            init_task_type="custom_voice",
            input_mode=pb2.INPUT_MODE_CLAUSE,
            text_chunks=[
                "你好，这是流式文本输入测试。",
                "我们正在验证",
                "文本追加功能",
                "是否工作正常。",
            ],
            chunk_delay_ms=200,
            session_id="stream-text",
        )
    )
    _print_result(result, "stream")
    if result.audio is not None and result.audio.size > 0:
        out_path = output_dir / "test2_streaming_text.wav"
        _save_wav(result.audio, out_path)
        print(f"  Saved: {out_path}")

    return result


def _build_token_stream_chunks_or_skip() -> list[str]:
    try:
        tokenized = build_token_text_chunks(DEFAULT_TOKEN_STREAM_TEXT)
    except TokenChunkingUnavailable as exc:
        pytest.skip(str(exc))
    except ValueError as exc:
        pytest.fail(f"Invalid TOKEN-mode test chunks: {exc}")
    return tokenized.chunks


def run_token_streaming_text(host: str, port: int, output_dir: Path) -> TTSResult:
    pb2, _ = _require_gateway()
    print("\n" + "=" * 60)
    print("  Test 2a: Token-Scale Streaming Text Input")
    print("=" * 60)

    tokenized = build_token_text_chunks(DEFAULT_TOKEN_STREAM_TEXT)
    print(
        "  Token chunks: "
        f"count={tokenized.chunk_count} max_chars={tokenized.max_chunk_chars} "
        f"tokenizer={tokenized.tokenizer_dir}"
    )

    result = asyncio.run(
        _synthesize_streaming(
            host,
            port,
            init_speaker="Serena",
            init_task_type="custom_voice",
            input_mode=pb2.INPUT_MODE_TOKEN,
            group_policy=pb2.GROUP_POLICY_NONE,
            text_chunks=tokenized.chunks,
            chunk_delay_ms=30,
            session_id="stream-token",
        )
    )
    _print_result(result, "stream-token")
    if result.audio is not None and result.audio.size > 0:
        out_path = output_dir / "test2a_streaming_token.wav"
        _save_wav(result.audio, out_path)
        print(f"  Saved: {out_path}")

    return result


def run_custom_voice_instruct(host: str, port: int, output_dir: Path) -> list[TTSResult]:
    pb2, _ = _require_gateway()
    print("\n" + "=" * 60)
    print("  Test 2b: CustomVoice + Instruct")
    print("=" * 60)

    if not _custom_voice_instruct_supported(host, port):
        cap = _get_capabilities(host, port)
        print(
            "  SKIPPED: need loaded_model_type=custom_voice and non-0.6b variant "
            f"(got type={cap.get('loaded_model_type')!r} variant={cap.get('variant')!r})"
        )
        return []

    results: list[TTSResult] = []

    print("\n  --- 2b-1: SynthesizeOnce with instruct (ZH) ---")
    r1 = _synthesize_oneshot(
        host,
        port,
        text="你好，这是带指令的预置音色测试。",
        speaker="Serena",
        instruct=CUSTOM_VOICE_INSTRUCT_ZH,
        session_id="custom-instruct-oneshot",
    )
    _print_result(r1, "instruct-oneshot")
    if r1.audio is not None and r1.audio.size > 0:
        out_path = output_dir / "test2b_custom_instruct_oneshot.wav"
        _save_wav(r1.audio, out_path)
        print(f"  Saved: {out_path}")
    results.append(r1)

    print("\n  --- 2b-2: Streaming with instruct (EN instruct + EN text) ---")
    r2 = asyncio.run(
        _synthesize_streaming(
            host,
            port,
            init_speaker="Serena",
            init_task_type="custom_voice",
            init_instruct=CUSTOM_VOICE_INSTRUCT_EN,
            input_mode=pb2.INPUT_MODE_CLAUSE,
            text_chunks=[
                "Hello, this is a streaming test ",
                "with instruct for preset voice.",
            ],
            chunk_delay_ms=150,
            session_id="custom-instruct-stream",
        )
    )
    _print_result(r2, "instruct-stream")
    if r2.audio is not None and r2.audio.size > 0:
        out_path = output_dir / "test2b_custom_instruct_stream.wav"
        _save_wav(r2.audio, out_path)
        print(f"  Saved: {out_path}")
    results.append(r2)

    return results


def run_concurrent(host: str, port: int, concurrency: int, output_dir: Path) -> list[TTSResult]:
    print("\n" + "=" * 60)
    print(f"  Test 3: Concurrent Requests (concurrency={concurrency})")
    print("=" * 60)

    def _run_one(idx: int) -> TTSResult:
        text = TEST_TEXTS[idx % len(TEST_TEXTS)]
        return _synthesize_oneshot(
            host,
            port,
            text=text,
            speaker="Serena",
            session_id=f"concurrent-{idx}",
        )

    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_run_one, i): i for i in range(concurrency)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                _print_result(result, f"c{idx}")
            except Exception as exc:
                print(f"  [c{idx}] EXCEPTION: {exc}")
                results.append(TTSResult(session_id=f"concurrent-{idx}", text="", error=str(exc)))

    wall_time = time.perf_counter() - t0
    print(f"\n  Wall time for {concurrency} concurrent: {wall_time:.2f}s")
    _print_summary(results, f"Concurrent x{concurrency}")

    for i, result in enumerate(results):
        if result.audio is not None and result.audio.size > 0:
            out_path = output_dir / f"test3_concurrent_{concurrency}x_{i}.wav"
            _save_wav(result.audio, out_path)

    return results


def stress_concurrent(
    host: str,
    port: int,
    *,
    concurrency: int,
    rounds: int,
    warmup_rounds: int = 0,
) -> list[list[TTSResult]]:
    print("\n" + "=" * 60)
    print(
        f"  Stress Test: Concurrent Requests "
        f"(concurrency={concurrency}, warmup={warmup_rounds}, rounds={rounds})"
    )
    print("=" * 60)

    all_rounds: list[list[TTSResult]] = []
    measured_rounds: list[list[TTSResult]] = []

    total_rounds = warmup_rounds + rounds
    for round_idx in range(total_rounds):
        measured = round_idx >= warmup_rounds
        phase = "measure" if measured else "warmup"
        label = f"{phase}-{round_idx - warmup_rounds + 1}" if measured else f"warmup-{round_idx + 1}"
        print(f"\n  --- Round {round_idx + 1}/{total_rounds} ({label}) ---")

        def _run_one(idx: int) -> TTSResult:
            text = TEST_TEXTS[idx % len(TEST_TEXTS)]
            return _synthesize_oneshot(
                host,
                port,
                text=text,
                speaker="Serena",
                session_id=f"stress-{round_idx}-{idx}",
            )

        t0 = time.perf_counter()
        round_results: list[TTSResult] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_run_one, i): i for i in range(concurrency)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    round_results.append(result)
                except Exception as exc:
                    round_results.append(
                        TTSResult(session_id=f"stress-{round_idx}-{idx}", text="", error=str(exc))
                    )

        wall_time = time.perf_counter() - t0
        ok = sum(1 for result in round_results if result.error is None)
        fail = len(round_results) - ok
        first_chunks = [
            result.first_chunk_ms
            for result in round_results
            if result.error is None and result.first_chunk_ms is not None
        ]
        totals = [result.total_ms for result in round_results if result.error is None]
        print(
            f"  Round result: ok={ok}/{len(round_results)} fail={fail} "
            f"wall={wall_time:.2f}s"
        )
        if first_chunks:
            print(
                f"  First-chunk: median={statistics.median(first_chunks):.0f}ms "
                f"p95={_p95(first_chunks):.0f}ms max={max(first_chunks):.0f}ms"
            )
        if totals:
            print(
                f"  Total latency: median={statistics.median(totals):.0f}ms "
                f"p95={_p95(totals):.0f}ms max={max(totals):.0f}ms"
            )

        all_rounds.append(round_results)
        if measured:
            measured_rounds.append(round_results)

    if not measured_rounds:
        return all_rounds

    flat = [result for round_results in measured_rounds for result in round_results]
    print("\n  --- Stress Aggregate ---")
    _print_summary(flat, f"Stress x{concurrency} rounds={rounds}")

    round_success = [
        sum(1 for result in round_results if result.error is None) / len(round_results)
        for round_results in measured_rounds
        if round_results
    ]
    if round_success:
        print(
            f"  Round success-rate: min={min(round_success) * 100:.1f}% "
            f"median={statistics.median(round_success) * 100:.1f}% "
            f"max={max(round_success) * 100:.1f}%"
        )

    return all_rounds


def run_long_text(host: str, port: int, output_dir: Path) -> list[TTSResult]:
    pb2, _ = _require_gateway()
    print("\n" + "=" * 60)
    print("  Test 4: Long Text Rollover")
    print("=" * 60)

    results = []

    print("\n  --- 4a: Medium long text ---")
    r1 = _synthesize_oneshot(
        host,
        port,
        text=LONG_TEXT,
        speaker="Serena",
        session_id="longtext-medium",
        timeout=180,
    )
    _print_result(r1, "medium-long")
    if r1.audio is not None and r1.audio.size > 0:
        out_path = output_dir / "test4a_long_text_medium.wav"
        _save_wav(r1.audio, out_path)
        print(f"  Saved: {out_path}")
    results.append(r1)

    print("\n  --- 4b: Very long text (multi-segment) ---")
    r2 = _synthesize_oneshot(
        host,
        port,
        text=VERY_LONG_TEXT,
        speaker="Serena",
        session_id="longtext-verylong",
        timeout=300,
    )
    _print_result(r2, "very-long")
    if r2.audio is not None and r2.audio.size > 0:
        out_path = output_dir / "test4b_long_text_verylong.wav"
        _save_wav(r2.audio, out_path)
        print(f"  Saved: {out_path}")
        print(f"  Audio duration: {r2.duration_sec:.2f}s (~{len(VERY_LONG_TEXT)} chars)")
    results.append(r2)

    print("\n  --- 4c: Streaming long text ---")
    sentences = [
        "人工智能正在深刻改变我们的世界。",
        "从语音识别到自然语言处理，从计算机视觉到机器人技术，AI的应用已经渗透到生活的方方面面。",
        "在医疗领域，AI可以辅助诊断疾病、发现新药物。在教育领域，AI可以提供个性化的学习方案。",
        "在交通领域，自动驾驶技术正在逐步成熟。未来，人工智能将继续推动社会进步，为人类创造更多的可能性。",
    ]
    r3 = asyncio.run(
        _synthesize_streaming(
            host,
            port,
            init_speaker="Serena",
            init_task_type="custom_voice",
            input_mode=pb2.INPUT_MODE_LONG_SEGMENT,
            text_chunks=sentences,
            chunk_delay_ms=300,
            session_id="stream-long",
            timeout=180,
        )
    )
    _print_result(r3, "stream-long")
    if r3.audio is not None and r3.audio.size > 0:
        out_path = output_dir / "test4c_streaming_long.wav"
        _save_wav(r3.audio, out_path)
        print(f"  Saved: {out_path}")
    results.append(r3)

    story_path = default_story_path()
    if story_path.is_file():
        print("\n  --- 4d: Story (long narrative) ---")
        story_text = story_path.read_text(encoding="utf-8").strip()
        print(f"  Story length: {len(story_text)} chars")
        r4 = _synthesize_oneshot(
            host,
            port,
            text=story_text,
            speaker="Serena",
            session_id="longtext-story",
            timeout=600,
            instruct=CUSTOM_VOICE_INSTRUCT_ZH,
        )
        _print_result(r4, "story")
        if r4.audio is not None and r4.audio.size > 0:
            out_path = output_dir / "test4d_story.wav"
            _save_wav(r4.audio, out_path)
            print(f"  Saved: {out_path}")
            print(f"  Audio duration: {r4.duration_sec:.2f}s (~{len(story_text)} chars)")
        results.append(r4)
    else:
        print(f"\n  --- 4d: Story SKIPPED (not found: {story_path}) ---")

    return results


def run_badcases(host: str, port: int, output_dir: Path) -> list:
    pb2, _ = _require_gateway()
    print("\n" + "=" * 60)
    print("  Test 5: BadCase Tests")
    print("=" * 60)
    results = []

    print("\n  --- 5a: Empty text (should error) ---")
    r = _synthesize_oneshot(host, port, text="", speaker="Serena", session_id="badcase-empty", timeout=15)
    expected_error = r.error is not None or r.total_samples == 0
    print(f"  Got error/empty: {expected_error} -> {'PASS' if expected_error else 'FAIL'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5a_empty_text", expected_error, r))

    print("\n  --- 5b: Whitespace-only text (should error) ---")
    r = _synthesize_oneshot(
        host,
        port,
        text="   \n\t  ",
        speaker="Serena",
        session_id="badcase-whitespace",
        timeout=15,
    )
    expected_error = r.error is not None or r.total_samples == 0
    print(f"  Got error/empty: {expected_error} -> {'PASS' if expected_error else 'FAIL'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5b_whitespace_text", expected_error, r))

    print("\n  --- 5c: Invalid task_type (should error) ---")
    r = _synthesize_oneshot(
        host,
        port,
        text="测试",
        task_type="nonexistent_task",
        session_id="badcase-invalid-task",
        timeout=15,
    )
    expected_error = r.error is not None
    print(f"  Got error: {expected_error} -> {'PASS' if expected_error else 'FAIL'}")
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5c_invalid_task", expected_error, r))

    print("\n  --- 5d: Streaming init -> text_complete (no text) ---")
    r = asyncio.run(
        _synthesize_streaming(
            host,
            port,
            init_speaker="Serena",
            init_task_type="custom_voice",
            input_mode=pb2.INPUT_MODE_CLAUSE,
            text_chunks=[],
            chunk_delay_ms=100,
            session_id="badcase-stream-notext",
            timeout=15,
        )
    )
    got_error_or_empty = r.error is not None or r.total_samples == 0
    print(
        f"  Error or empty: {got_error_or_empty} -> "
        f"{'PASS' if got_error_or_empty else 'FAIL (unexpected audio)'}"
    )
    if r.error:
        print(f"  Error msg: {r.error[:120]}")
    results.append(("5d_stream_no_text", got_error_or_empty, r))

    print("\n  --- 5e: Cancel mid-stream ---")
    r = _test_cancel(host, port)
    ok = r.error is None
    print(f"  Cancel clean: {ok} -> {'PASS' if ok else 'FAIL'}")
    _print_result(r, "cancel")
    results.append(("5e_cancel", ok, r))

    print("\n  --- 5f: Single character text ---")
    r = _synthesize_oneshot(
        host,
        port,
        text="好",
        speaker="Serena",
        session_id="badcase-single-char",
        timeout=30,
    )
    ok = r.error is None and r.total_samples > 0
    print(f"  Got audio: {ok} -> {'PASS' if ok else 'FAIL'}")
    _print_result(r, "single-char")
    if r.audio is not None and r.audio.size > 0:
        _save_wav(r.audio, output_dir / "test5f_single_char.wav")
    results.append(("5f_single_char", ok, r))

    print("\n  --- 5g: Pure punctuation text ---")
    r = _synthesize_oneshot(
        host,
        port,
        text="。。。！！！",
        speaker="Serena",
        session_id="badcase-punctuation",
        timeout=30,
    )
    _print_result(r, "punctuation")
    results.append(("5g_punctuation", True, r))

    print("\n  --- 5h: Streaming with slow text input (1s delay) ---")
    r = asyncio.run(
        _synthesize_streaming(
            host,
            port,
            init_speaker="Serena",
            init_task_type="custom_voice",
            input_mode=pb2.INPUT_MODE_CLAUSE,
            text_chunks=["你好，", "这是一个", "慢速输入的测试。"],
            chunk_delay_ms=1000,
            session_id="badcase-slow-stream",
            timeout=60,
        )
    )
    ok = r.error is None and r.total_samples > 0
    print(f"  Got audio: {ok} -> {'PASS' if ok else 'FAIL'}")
    _print_result(r, "slow-stream")
    if r.audio is not None and r.audio.size > 0:
        _save_wav(r.audio, output_dir / "test5h_slow_stream.wav")
    results.append(("5h_slow_stream", ok, r))

    print("\n  -- BadCase Summary --")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, result in results:
        status = "PASS" if ok else "FAIL"
        err = f" error={result.error[:60]}" if result.error else ""
        samples = f" samples={result.total_samples}" if result.total_samples else ""
        print(f"  {status:4s}  {name:30s}{err}{samples}")
    print(f"\n  BadCase: {passed}/{total} passed")

    return results


def _test_cancel(host: str, port: int) -> TTSResult:
    pb2, pb2_grpc = _require_gateway()
    sid = uuid.uuid4().hex[:12]
    result = TTSResult(session_id=sid, text="(cancel test)")

    import grpc

    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = pb2_grpc.TTSServiceStub(channel)

    def request_gen():
        yield pb2.SynthesizeRequest(
            start=pb2.StartRequest(
                session_id=sid,
                config=_make_session_config(
                    task_type="custom_voice",
                    speaker="Serena",
                    input_mode=pb2.INPUT_MODE_LONG_SEGMENT,
                ),
            )
        )
        yield pb2.SynthesizeRequest(text=pb2.TextChunk(text="这段文字将被取消。"))
        time.sleep(0.1)
        yield pb2.SynthesizeRequest(cancel=pb2.CancelRequest())

    chunks = []
    t0 = time.perf_counter()
    try:
        for resp in stub.SynthesizeStream(request_gen(), timeout=15):
            which = resp.WhichOneof("response")
            if which == "audio":
                chunks.append(_audio_chunk_to_f32(resp.audio))
            elif _handle_stream_control(resp, result):
                break
    except grpc.RpcError:
        pass
    except Exception as exc:
        result.error = str(exc)
    finally:
        channel.close()

    result.total_ms = (time.perf_counter() - t0) * 1000
    result.num_chunks = len(chunks)
    if chunks:
        result.audio = np.concatenate(chunks)
        result.total_samples = result.audio.size

    return result
