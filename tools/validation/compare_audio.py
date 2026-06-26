#!/usr/bin/env python3
"""Unified audio comparison tool for Qwen3-TTS-Triton.

Merges the functionality of nine legacy scripts into one CLI with subcommands:

  gen-triton       Generate WAVs via Triton streaming          (was gen_audio.py)
  gen-engine       Generate audio via engine gRPC              (was gen_engine_audio.py)
  gen-reference    Generate reference audio (official API)     (was gen_reference_audio.py)
  compare-triton   Official API vs Triton orchestrator A/B     (was compare_official_vs_triton_audio.py)
  compare-ort      Official API vs fused ONNX (ORT) A/B       (was compare_official_vs_fused_onnx.py)
  compare-full-chain  Full-chain 4-way listen compare          (was full_chain_audio_listen.py)
  compare-3way     Three-way: proto vs manual vs ORT           (was generate_audio_compare.py)
  fused-onnx       Run fused ONNX decode loop, save WAV       (was fused_onnx_audio.py)
  long-ab          Long-text A/B: official vs engine           (was long_streaming_listen_ab.py)

Usage:
  python tools/validation/compare_audio.py <mode> [options]
  python tools/validation/compare_audio.py gen-triton
  python tools/validation/compare_audio.py compare-triton --text "你好" --speaker serena
  python tools/validation/compare_audio.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Bootstrap import paths (must run before common/audio imports).
import sys
from pathlib import Path

# Ensure the validation tools directory is on sys.path for _bootstrap
_validation_dir = str(Path(__file__).resolve().parent)
if _validation_dir not in sys.path:
    sys.path.insert(0, _validation_dir)

from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()

from common import REPO_ROOT, bootstrap_project_imports

# ---------------------------------------------------------------------------
# Bootstrap: ensure all sub-project packages are importable.
# Modes that need fewer paths still work because Python ignores missing
# entries; we include the superset for simplicity.
# ---------------------------------------------------------------------------
bootstrap_project_imports(
    "repo", "scripts_python", "scripts_export", "scripts", "third_party_qwen",
)

# Ensure the repo-root ``tests`` package takes precedence over any
# ``tests`` namespace package that may exist in site-packages (e.g. from
# a pip-installed test-data package).  We insert REPO_ROOT at position 0
# and bust any cached ``tests`` module that resolved elsewhere.
import importlib
_repo_str = str(REPO_ROOT)
if sys.path[0] != _repo_str:
    sys.path.insert(0, _repo_str)
if "tests" in sys.modules:
    _t = sys.modules["tests"]
    if not getattr(_t, "__file__", None) or not str(getattr(_t, "__file__", "")).startswith(_repo_str):
        del sys.modules["tests"]

from qwen3_tts_protocol.triton_types import (
    build_request_payload,
    build_variant_request_payload,
)
from qwen3_tts_protocol.audio import save_wav, StreamResult
from tests.support.triton_streaming import infer_stream

# Lazy imports for heavy dependencies (torch, onnxruntime, grpc, etc.) are
# done inside each mode function so that modes that don't need them start fast.

SAMPLE_RATE = 24000
DEPRECATION_NOTICE = (
    "NOTE: This unified tool replaces the following legacy scripts:\n"
    "  gen_audio.py, gen_engine_audio.py, gen_reference_audio.py,\n"
    "  compare_official_vs_triton_audio.py, compare_official_vs_fused_onnx.py,\n"
    "  full_chain_audio_listen.py, generate_audio_compare.py,\n"
    "  fused_onnx_audio.py, long_streaming_listen_ab.py\n"
    "Those scripts are deprecated; prefer `compare_audio.py <mode>`."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_audio")

# ── Shared helpers ──────────────────────────────────────────────────────────

def _default_out(*parts: str) -> Path:
    """Build an output path under workspace/."""
    return REPO_ROOT / "workspace" / Path(*parts)


def _slug(s: str, max_len: int = 40) -> str:
    import re
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^\w一-鿿_-]", "", s)
    return (s[:max_len] if s else "run").strip("_") or "run"


def _check_triton_client():
    try:
        import tritonclient.grpc as _gc
        return _gc
    except ImportError:
        logger.error("tritonclient[grpc] not installed.  pip install tritonclient[grpc]")
        sys.exit(1)


def _check_ort():
    try:
        import onnxruntime as _ort
        return _ort
    except ImportError:
        logger.error("onnxruntime not installed.  pip install onnxruntime")
        sys.exit(1)


def _load_manifest(exported_dir: Path) -> Dict[str, Any]:
    p = exported_dir / "triton_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_utils():
    """Lazily import helpers from scripts/export/utils.py."""
    from utils import (
        setup_logging,
        resolve_model_path,
        resolve_device,
        has_model_weights,
        DEFAULT_OUTPUT_DIR,
    )
    return dict(
        setup_logging=setup_logging,
        resolve_model_path=resolve_model_path,
        resolve_device=resolve_device,
        has_model_weights=has_model_weights,
        DEFAULT_OUTPUT_DIR=DEFAULT_OUTPUT_DIR,
    )


# ── Mode: gen-triton ────────────────────────────────────────────────────────

SAMPLES_GEN_TRITON = [
    {"name": "01_greeting", "text": "你好，欢迎使用Qwen3-TTS语音合成系统。", "speaker": "zhitian"},
    {"name": "02_weather",  "text": "今天天气晴朗，万里无云，非常适合外出活动。", "speaker": "zhitian"},
    {"name": "03_story",    "text": "从前有座山，山上有座庙，庙里有个老和尚在给小和尚讲故事。", "speaker": "zhitian"},
    {"name": "04_english",  "text": "Hello, this is a test of the Qwen3 text to speech system. How does it sound?", "speaker": "zhitian"},
    {"name": "05_mixed",    "text": "深度学习领域的Transformer架构，自2017年提出以来，已经彻底改变了自然语言处理的格局。", "speaker": "zhitian"},
]


def _stream_tts(client, grpcclient, text: str, speaker: str, timeout: float = 60.0):
    req_dict = build_request_payload(text=text, task_type="custom_voice", speaker=speaker)
    stream = infer_stream(client, grpcclient, req_dict, timeout=timeout)
    chunks = [stream.audio] if stream.audio is not None and stream.audio.size else []
    first_sec = (stream.first_chunk_ms / 1000.0) if stream.first_chunk_ms is not None else None
    total = stream.total_ms / 1000.0
    return chunks, first_sec, total, stream.error


def mode_gen_triton(args):
    grpcclient = _check_triton_client()
    client = grpcclient.InferenceServerClient(url=args.triton_url)
    if not client.is_server_ready():
        logger.error("Triton server not ready at %s", args.triton_url)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else _default_out("audio_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(DEPRECATION_NOTICE)
    print(f"Generating {len(SAMPLES_GEN_TRITON)} audio samples -> {out_dir}/\n")

    for sample in SAMPLES_GEN_TRITON:
        name, text, speaker = sample["name"], sample["text"], sample["speaker"]
        print(f'  [{name}] "{text[:40]}..."')
        chunks, first_sec, total_sec, err = _stream_tts(client, text, speaker, timeout=args.timeout)
        if err:
            print(f"    ERROR: {err}")
            continue
        if not chunks:
            print("    WARNING: no audio chunks received")
            continue
        audio = np.concatenate(chunks)
        duration = audio.size / SAMPLE_RATE
        wav_path = out_dir / f"{name}.wav"
        save_wav(audio, wav_path, sample_rate=SAMPLE_RATE)
        print(f"    -> {wav_path.name}  duration={duration:.2f}s  "
              f"first_chunk={first_sec*1000:.0f}ms  total={total_sec*1000:.0f}ms  chunks={len(chunks)}")

    print(f"\nDone. Files in: {out_dir}/")


# ── Mode: gen-engine ────────────────────────────────────────────────────────

TEXTS_ENGINE = {
    "test1": "你好，这是单路测试。",
    "test2": "你好，这是流式文本输入测试。",
    "test3": "你好，今天天气真好。",
    "test4": "欢迎来到人工智能语音合成的世界。",
}


def _audio_chunk_to_f32(audio_chunk) -> np.ndarray:
    from engine.gateway import tts_pb2
    encoding = getattr(audio_chunk, "encoding", tts_pb2.AUDIO_ENCODING_PCM_F32)
    if encoding == tts_pb2.AUDIO_ENCODING_PCM_S16LE:
        return np.frombuffer(audio_chunk.pcm_data, dtype=np.int16).astype(np.float32) / 32767.0
    return np.frombuffer(audio_chunk.pcm_data, dtype=np.float32)


def _is_terminal_event(resp) -> Tuple[bool, str]:
    from engine.gateway import tts_pb2
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


def _engine_synthesize(host: str, port: int, text: str, speaker: str = "Serena",
                       timeout: float = 60.0) -> Tuple[Optional[np.ndarray], float, int]:
    import grpc
    from engine.gateway import tts_pb2, tts_pb2_grpc

    sid = uuid.uuid4().hex[:12]
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = tts_pb2_grpc.TTSServiceStub(channel)
    chunks: list = []
    first_ts = None
    t0 = time.perf_counter()
    try:
        request = tts_pb2.SynthesizeOnceRequest(
            session_id=sid, text=text,
            config=tts_pb2.SessionConfig(
                task_type="custom_voice", speaker=speaker,
                input_mode=tts_pb2.INPUT_MODE_FULL_TEXT,
                group_policy=tts_pb2.GROUP_POLICY_AUTO,
                audio=tts_pb2.AudioFormat(
                    encoding=tts_pb2.AUDIO_ENCODING_PCM_F32,
                    sample_rate=SAMPLE_RATE, channels=1,
                ),
            ),
        )
        for resp in stub.SynthesizeOnce(request, timeout=timeout):
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
        return np.concatenate(chunks), elapsed, len(chunks)
    return None, elapsed, 0


def mode_gen_engine(args):
    import grpc
    host, port = args.host, args.port
    channel = grpc.insecure_channel(f"{host}:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    except grpc.FutureTimeoutError:
        logger.error("Engine not reachable at %s:%d", host, port)
        sys.exit(1)
    finally:
        channel.close()

    out_dir = Path(args.out_dir) if args.out_dir else _default_out("audio_samples", "reference")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(DEPRECATION_NOTICE)
    print("Engine connected. Generating audio...")

    for name, text in TEXTS_ENGINE.items():
        print(f"\nGenerating [{name}]: {text}")
        audio, elapsed, n_chunks = _engine_synthesize(host, port, text, speaker=args.speaker)
        if audio is not None:
            duration = len(audio) / SAMPLE_RATE
            out_path = out_dir / f"engine_{name}.wav"
            save_wav(audio, out_path, sample_rate=SAMPLE_RATE)
            print(f"  -> {out_path.name}: {len(audio)} samples, {duration:.2f}s, "
                  f"{n_chunks} chunks, took {elapsed:.1f}s")
        else:
            print("  -> FAILED: no audio returned")

    print(f"\nAll engine audio saved to: {out_dir}")


# ── Mode: gen-reference ─────────────────────────────────────────────────────

TEXTS_REFERENCE = {
    "test1": "你好，这是单路测试。",
    "test2": "你好，这是流式文本输入测试。",
    "test3": "你好，今天天气真好。",
    "test4": "欢迎来到人工智能语音合成的世界。",
}


def mode_gen_reference(args):
    import soundfile as sf
    import torch

    utils = _resolve_utils()
    model_path = utils["resolve_model_path"](args.variant, args.models_dir)
    if not utils["has_model_weights"](model_path):
        logger.error("No weights for variant %s at %s", args.variant, model_path)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else _default_out("audio_samples", "reference")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(DEPRECATION_NOTICE)

    print(f"Loading model from {model_path} ...")
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    wrapper = Qwen3TTSModel.from_pretrained(
        str(model_path), device_map="cuda:0", dtype=torch.bfloat16,
    )
    print("Model loaded.")

    for name, text in TEXTS_REFERENCE.items():
        print(f"\nGenerating [{name}]: {text}")
        t0 = time.perf_counter()
        with torch.no_grad():
            wavs, sr = wrapper.generate_custom_voice(
                text=text, speaker=args.speaker, language="auto",
                do_sample=True, max_new_tokens=500,
            )
        elapsed = time.perf_counter() - t0
        wav = wavs[0]
        duration = len(wav) / sr
        out_path = out_dir / f"proto_{name}.wav"
        sf.write(str(out_path), wav, sr)
        print(f"  -> {out_path.name}: {len(wav)} samples, {duration:.2f}s, took {elapsed:.1f}s")

    print(f"\nAll reference audio saved to: {out_dir}")
    del wrapper
    torch.cuda.empty_cache()


# ── Mode: compare-triton ────────────────────────────────────────────────────

def mode_compare_triton(args):
    import soundfile as sf
    import torch

    utils = _resolve_utils()
    utils["setup_logging"]()
    device = utils["resolve_device"](args.device)
    path = utils["resolve_model_path"](args.variant, args.models_dir)
    if not utils["has_model_weights"](path):
        logger.error("No weights for variant %s at %s", args.variant, path)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else _default_out("audio_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(DEPRECATION_NOTICE)

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper
    logger.info("Loading official Qwen3TTSModel from %s ...", path)
    wrapper = TTSModelWrapper.from_pretrained(str(path), device_map=str(device), dtype=torch.float32)

    do_sample = not args.greedy
    gen_kwargs = dict(do_sample=do_sample, max_new_tokens=args.max_steps)
    if not do_sample:
        gen_kwargs["repetition_penalty"] = 1.0
        gen_kwargs["subtalker_dosample"] = False

    non_streaming = "design" in args.variant.lower()
    with torch.no_grad():
        if non_streaming:
            wavs, sr = wrapper.generate_voice_design(
                text=args.text, instruct=args.instruct, language=args.language,
                non_streaming_mode=non_streaming, **gen_kwargs,
            )
        else:
            wavs, sr = wrapper.generate_custom_voice(
                text=args.text, speaker=args.speaker, language=args.language,
                instruct=args.instruct, non_streaming_mode=non_streaming, **gen_kwargs,
            )

    wav_proto = wavs[0]
    sf.write(str(out_dir / "proto.wav"), wav_proto, sr)
    logger.info("Wrote proto.wav (official API, %d samples, %.2fs)", len(wav_proto), len(wav_proto) / sr)

    grpcclient = _check_triton_client()
    req = build_variant_request_payload(
        variant=args.variant, text=args.text, language=args.language,
        speaker=args.speaker, instruct=args.instruct,
    )
    logger.info("Triton request: %s", json.dumps(req, ensure_ascii=False))
    client = grpcclient.InferenceServerClient(url=args.triton_url)
    if not client.is_server_ready():
        logger.error("Triton not ready at %s", args.triton_url)
        sys.exit(1)

    stream = infer_stream(client, grpcclient, req, timeout=120)
    if stream.error:
        raise RuntimeError(stream.error)
    if stream.audio is None or stream.audio.size == 0:
        raise RuntimeError("No audio from Triton")
    sample_rate = int(stream.metadata.get("audio_format", {}).get("sample_rate", sr) or sr)
    wav_triton = stream.audio
    elapsed = stream.total_ms / 1000.0
    sf.write(str(out_dir / "triton.wav"), wav_triton, sample_rate)
    logger.info("Wrote triton.wav (Triton, %d samples, %.2fs, elapsed=%.2fs)",
                len(wav_triton), len(wav_triton) / sample_rate, elapsed)

    readme = out_dir / "LISTEN_README.txt"
    readme.write_text(
        "\n".join([
            "Official API vs Triton (listenability)",
            "========================================",
            f"variant: {args.variant}", f"text: {args.text[:80]}", "",
            "proto.wav  -- Qwen3TTSModel official high-level API.",
            "triton.wav -- tts_orchestrator gRPC stream (fused path).", "",
            f"Sampling: official uses do_sample={do_sample}; Triton uses orchestrator decode settings.", "",
            "These two are not expected to be bit-identical; compare by ear.",
        ]), encoding="utf-8",
    )
    logger.info("Done. Read %s and A/B listen to proto.wav vs triton.wav", readme)


# ── Mode: compare-ort ───────────────────────────────────────────────────────

def mode_compare_ort(args):
    import soundfile as sf
    import torch

    utils = _resolve_utils()
    utils["setup_logging"]()
    device = utils["resolve_device"](args.device)
    path = utils["resolve_model_path"](args.variant, args.models_dir)
    if not utils["has_model_weights"](path):
        logger.error("No weights for variant %s", args.variant)
        sys.exit(1)

    exported_dir = Path(utils["DEFAULT_OUTPUT_DIR"]) / args.variant
    onnx_path = exported_dir / "talker_code2wav_fused.onnx"
    if not onnx_path.is_file():
        logger.error("Missing fused ONNX: %s (run export_09)", onnx_path)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else _default_out("audio_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(DEPRECATION_NOTICE)

    ort = _check_ort()
    manifest = _load_manifest(exported_dir)
    talker = manifest.get("talker", {})
    num_layers = int(talker.get("num_layers", 28))
    num_kv_heads = int(talker.get("num_kv_heads", 8))
    head_dim = int(talker.get("head_dim", 128))

    logger.info("Loading ORT session: %s", onnx_path)
    sess = ort.InferenceSession(str(onnx_path), ort.SessionOptions(), providers=["CPUExecutionProvider"])

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper
    logger.info("Loading official Qwen3TTSModel ...")
    wrapper = TTSModelWrapper.from_pretrained(str(path), device_map=str(device), dtype=torch.float32)
    model = wrapper.model
    processor = wrapper.processor

    do_sample = not args.greedy
    gen_kwargs = dict(do_sample=do_sample, max_new_tokens=args.max_steps)
    if not do_sample:
        gen_kwargs["repetition_penalty"] = 1.0
        gen_kwargs["subtalker_dosample"] = False

    non_streaming = "design" in args.variant.lower()
    with torch.no_grad():
        if non_streaming:
            wavs, sr = wrapper.generate_voice_design(
                text=args.text, instruct=args.instruct, language=args.language,
                non_streaming_mode=non_streaming, **gen_kwargs,
            )
        else:
            wavs, sr = wrapper.generate_custom_voice(
                text=args.text, speaker=args.speaker, language=args.language,
                instruct=args.instruct, non_streaming_mode=non_streaming, **gen_kwargs,
            )
    wav_proto = wavs[0]
    sf.write(str(out_dir / "proto.wav"), wav_proto, sr)
    logger.info("Wrote proto.wav (official), len=%d, sr=%d", len(wav_proto), sr)

    # Reuse the full fused-ONNX loop from the legacy module
    from compare_official_vs_fused_onnx import (
        OFFICIAL_ASSISTANT_FMT,
        run_fused_onnx_loop,
    )
    from official_prefill import build_prefill_like_official

    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=args.text)
    tok_out = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = tok_out["input_ids"].to(device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    instruct_ids = None
    if args.instruct:
        instruct_text = wrapper._build_instruct_text(args.instruct)
        instruct_tok = processor(text=instruct_text, return_tensors="pt", padding=True)
        instruct_ids = instruct_tok["input_ids"].to(device=device, dtype=torch.long)
        if instruct_ids.dim() == 1:
            instruct_ids = instruct_ids.unsqueeze(0)

    prefill_embeds, trailing_list = build_prefill_like_official(
        model, input_ids, args.language, args.speaker or "", device,
        instruct_ids=instruct_ids, non_streaming_mode=non_streaming,
    )
    tts_pad_token_id = getattr(model.config, "tts_pad_token_id", 0)
    pad_id = torch.tensor([[tts_pad_token_id]], device=device, dtype=torch.long)
    pad_embed = model.talker.text_projection(model.talker.model.text_embedding(pad_id))

    B, S, _ = prefill_embeds.shape
    position_ids_1d = torch.arange(S, device=device, dtype=torch.int64)
    position_ids_prefill = position_ids_1d.reshape(1, 1, -1, 1).expand(B, 3, S, 1)
    codec_eos_id = int(model.config.talker_config.codec_eos_token_id)

    logger.info("Running fused ONNX loop (CPU ORT) ...")
    wav_fused = run_fused_onnx_loop(
        sess, manifest,
        inputs_embeds=prefill_embeds, position_ids_prefill=position_ids_prefill,
        trailing_text=trailing_list, pad_embed=pad_embed,
        num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        codec_eos_id=codec_eos_id, max_steps=args.max_steps,
    )
    sf.write(str(out_dir / "fused_onnx.wav"), wav_fused, sr)
    logger.info("Wrote fused_onnx.wav, len=%d (%.2fs @ %d Hz)", len(wav_fused), len(wav_fused) / sr, sr)

    readme = out_dir / "COMPARE_OFFICIAL_FUSED_ONNX.txt"
    readme.write_text(
        "\n".join([
            "proto.wav       -- Qwen3TTSModel official API (sampling unless --greedy).",
            "fused_onnx.wav  -- talker_code2wav_fused.onnx via ORT, greedy argmax.", "",
            "Sampling differs from greedy; lengths may not match. Compare by listening.",
        ]), encoding="utf-8",
    )
    logger.info("Done. See %s", readme)


# ── Mode: compare-full-chain ────────────────────────────────────────────────

def mode_compare_full_chain(args):
    import soundfile as sf
    import torch

    utils = _resolve_utils()
    utils["setup_logging"]()
    device = utils["resolve_device"](args.device)
    path = utils["resolve_model_path"](args.variant, args.models_dir)
    if not utils["has_model_weights"](path):
        logger.error("No weights for variant %s", args.variant)
        sys.exit(1)

    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
        if not text:
            logger.error("Empty --text-file")
            sys.exit(1)

    exported_dir = Path(utils["DEFAULT_OUTPUT_DIR"]) / args.variant
    onnx_path = exported_dir / "talker_code2wav_fused.onnx"
    if not onnx_path.is_file():
        logger.error("Missing fused ONNX: %s (run export_09)", onnx_path)
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out_root) if args.out_root else _default_out("audio_compare", "runs")
    out_dir = out_root / f"{stamp}_{_slug(text)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(DEPRECATION_NOTICE)
    logger.info("Output directory: %s", out_dir)

    ort = _check_ort()
    manifest = _load_manifest(exported_dir)
    talker = manifest.get("talker", {})
    num_layers = int(talker.get("num_layers", 28))
    num_kv_heads = int(talker.get("num_kv_heads", 8))
    head_dim = int(talker.get("head_dim", 128))

    sess = ort.InferenceSession(str(onnx_path), ort.SessionOptions(), providers=["CPUExecutionProvider"])
    onnx_input_names = [i.name for i in sess.get_inputs()]
    onnx_output_names = [o.name for o in sess.get_outputs()]

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper
    logger.info("Loading official Qwen3TTSModel ...")
    wrapper = TTSModelWrapper.from_pretrained(str(path), device_map=str(device), dtype=torch.float32)
    model = wrapper.model
    processor = wrapper.processor

    do_sample = not args.greedy
    gen_kwargs = dict(do_sample=do_sample, max_new_tokens=args.max_steps)
    if not do_sample:
        gen_kwargs["repetition_penalty"] = 1.0
        gen_kwargs["subtalker_dosample"] = False

    non_streaming = "design" in args.variant.lower()
    with torch.no_grad():
        if non_streaming:
            wavs, sr = wrapper.generate_voice_design(
                text=text, instruct=args.instruct, language=args.language,
                non_streaming_mode=non_streaming, **gen_kwargs,
            )
        else:
            wavs, sr = wrapper.generate_custom_voice(
                text=text, speaker=args.speaker, language=args.language,
                instruct=args.instruct, non_streaming_mode=non_streaming, **gen_kwargs,
            )
    wav_proto = wavs[0]
    sf.write(str(out_dir / "proto.wav"), wav_proto, sr)
    logger.info("Wrote proto.wav (%.2fs @ %d Hz)", len(wav_proto) / sr, sr)

    from compare_official_vs_fused_onnx import (
        OFFICIAL_ASSISTANT_FMT, run_fused_onnx_loop, run_fused_triton_loop,
    )
    from official_prefill import build_prefill_like_official

    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=text)
    tok_out = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = tok_out["input_ids"].to(device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    instruct_ids = None
    if args.instruct:
        instruct_text = wrapper._build_instruct_text(args.instruct)
        instruct_tok = processor(text=instruct_text, return_tensors="pt", padding=True)
        instruct_ids = instruct_tok["input_ids"].to(device=device, dtype=torch.long)
        if instruct_ids.dim() == 1:
            instruct_ids = instruct_ids.unsqueeze(0)

    prefill_embeds, trailing_list = build_prefill_like_official(
        model, input_ids, args.language, args.speaker or "", device,
        instruct_ids=instruct_ids, non_streaming_mode=non_streaming,
    )
    tts_pad_token_id = getattr(model.config, "tts_pad_token_id", 0)
    pad_id = torch.tensor([[tts_pad_token_id]], device=device, dtype=torch.long)
    pad_embed = model.talker.text_projection(model.talker.model.text_embedding(pad_id))

    B, S, _ = prefill_embeds.shape
    position_ids_1d = torch.arange(S, device=device, dtype=torch.int64)
    position_ids_prefill = position_ids_1d.reshape(1, 1, -1, 1).expand(B, 3, S, 1)
    codec_eos_id = int(model.config.talker_config.codec_eos_token_id)

    logger.info("Running fused ORT loop ...")
    wav_fused_ort = run_fused_onnx_loop(
        sess, manifest,
        inputs_embeds=prefill_embeds, position_ids_prefill=position_ids_prefill,
        trailing_text=trailing_list, pad_embed=pad_embed,
        num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        codec_eos_id=codec_eos_id, max_steps=args.max_steps,
    )
    sf.write(str(out_dir / "fused_onnx.wav"), wav_fused_ort, sr)
    logger.info("Wrote fused_onnx.wav (%.2fs)", len(wav_fused_ort) / sr)

    readme_lines = [
        "Full-chain audio listen (ear-first)", "====================================",
        f"variant: {args.variant}", f"text (first 120 chars): {text[:120]}",
        f"do_sample (official): {do_sample}", f"max_steps: {args.max_steps}", "",
        "Files:",
        "  proto.wav               -- Official Qwen3TTSModel API.",
        "  fused_onnx.wav          -- talker_code2wav_fused.onnx via ORT (CPU), greedy loop.",
        "  fused_triton_direct.wav -- Triton model talker_code2wav_fused.",
        "  orchestrator.wav        -- tts_orchestrator gRPC stream.", "",
    ]

    if not args.skip_triton:
        grpcclient = _check_triton_client()
        client = grpcclient.InferenceServerClient(url=args.triton_url)
        if not client.is_server_ready():
            logger.error("Triton not ready at %s", args.triton_url)
            sys.exit(1)

        cfg = client.get_model_config("talker_code2wav_fused", as_json=True)["config"]
        triton_input_dtypes = {item["name"]: item["data_type"] for item in cfg["input"]}

        logger.info("Running Triton talker_code2wav_fused (direct) ...")
        wav_fused_triton = run_fused_triton_loop(
            client, triton_input_dtypes, onnx_input_names, onnx_output_names, manifest,
            inputs_embeds=prefill_embeds, position_ids_prefill=position_ids_prefill,
            trailing_text=trailing_list, pad_embed=pad_embed,
            num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
            codec_eos_id=codec_eos_id, max_steps=args.max_steps,
        )
        sf.write(str(out_dir / "fused_triton_direct.wav"), wav_fused_triton, sr)
        logger.info("Wrote fused_triton_direct.wav (%.2fs)", len(wav_fused_triton) / sr)

        req = build_variant_request_payload(
            variant=args.variant, text=text, language=args.language,
            speaker=args.speaker, instruct=args.instruct,
        )
        logger.info("Orchestrator request: %s", json.dumps(req, ensure_ascii=False))
        stream = infer_stream(client, grpcclient, req, timeout=180)
        if stream.error:
            raise RuntimeError(stream.error)
        if stream.audio is None or stream.audio.size == 0:
            raise RuntimeError("No audio from tts_orchestrator")
        wav_orch = stream.audio
        elapsed = stream.total_ms / 1000.0
        orchestrator_sr = int(stream.metadata.get("audio_format", {}).get("sample_rate", sr) or sr)
        sf.write(str(out_dir / "orchestrator.wav"), wav_orch, orchestrator_sr)
        logger.info("Wrote orchestrator.wav (%.2fs, wall=%.2fs)", len(wav_orch) / orchestrator_sr, elapsed)
        readme_lines.append(f"orchestrator wall time: {elapsed:.2f}s")
    else:
        readme_lines.append("Triton steps skipped (--skip-triton).")

    (out_dir / "LISTEN_README.txt").write_text("\n".join(readme_lines), encoding="utf-8")
    logger.info("Done. Open %s and listen.", out_dir / "LISTEN_README.txt")


# ── Mode: compare-3way ──────────────────────────────────────────────────────

def mode_compare_3way(args):
    import soundfile as sf
    import torch

    utils = _resolve_utils()
    utils["setup_logging"]()
    device = utils["resolve_device"](args.device)
    path = utils["resolve_model_path"](args.variant, args.models_dir)
    if not utils["has_model_weights"](path):
        logger.error("No weights for variant %s", args.variant)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else _default_out("audio_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(DEPRECATION_NOTICE)

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper
    from official_prefill import build_prefill_like_official
    from verify_prototype_parity import run_manual_decode_loop
    from generate_audio_compare import run_ort_decode_loop, decode_codes_to_wav, OFFICIAL_ASSISTANT_FMT

    wrapper = TTSModelWrapper.from_pretrained(str(path), device_map=str(device), dtype=torch.float32)
    model = wrapper.model
    processor = wrapper.processor

    do_sample = not args.greedy
    non_streaming = "design" in args.variant.lower()
    codec_eos_id = int(model.config.talker_config.codec_eos_token_id)

    # (1) Prototype
    gen_kwargs = dict(do_sample=do_sample, max_new_tokens=args.max_steps)
    if not do_sample:
        gen_kwargs["repetition_penalty"] = 1.0
        gen_kwargs["subtalker_dosample"] = False
    with torch.no_grad():
        if non_streaming:
            wavs_proto, sr = wrapper.generate_voice_design(
                text=args.text, instruct=args.instruct, language=args.language,
                non_streaming_mode=non_streaming, **gen_kwargs,
            )
        else:
            wavs_proto, sr = wrapper.generate_custom_voice(
                text=args.text, speaker=args.speaker, language=args.language,
                instruct=args.instruct, non_streaming_mode=non_streaming, **gen_kwargs,
            )
    wav_proto = wavs_proto[0]
    sf.write(str(out_dir / "proto.wav"), wav_proto, sr)
    logger.info("Prototype: %d samples (%.2fs)", len(wav_proto), len(wav_proto) / sr)

    # Tokenize
    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=args.text)
    tok_out = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = tok_out["input_ids"].to(device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    instruct_ids = None
    if args.instruct:
        instruct_text = wrapper._build_instruct_text(args.instruct)
        instruct_tok = processor(text=instruct_text, return_tensors="pt", padding=True)
        instruct_ids = instruct_tok["input_ids"].to(device=device, dtype=torch.long)
        if instruct_ids.dim() == 1:
            instruct_ids = instruct_ids.unsqueeze(0)

    # (2) Manual PyTorch
    prefill_embeds, trailing_list = build_prefill_like_official(
        model, input_ids, args.language, args.speaker or "", device,
        instruct_ids=instruct_ids, non_streaming_mode=non_streaming,
    )
    tts_pad_token_id = getattr(model.config, "tts_pad_token_id", 0)
    with torch.no_grad():
        pad_id = torch.tensor([[tts_pad_token_id]], device=device, dtype=torch.long)
        pad_embed = model.talker.text_projection(model.talker.model.text_embedding(pad_id))
    codes_manual, _, manual_eos = run_manual_decode_loop(
        model, prefill_embeds, trailing_list, pad_embed, args.max_steps, codec_eos_id, device,
    )
    logger.info("Manual: T=%d, eos_step=%d", codes_manual.shape[0], manual_eos)

    # (3) ORT
    onnx_path = Path(utils["DEFAULT_OUTPUT_DIR"]) / args.variant / "talker_unified.onnx"
    if not onnx_path.exists():
        logger.error("ONNX not found: %s", onnx_path)
        sys.exit(1)
    ort = _check_ort()
    session = ort.InferenceSession(str(onnx_path), ort.SessionOptions(), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in session.get_outputs()]

    from generate_audio_compare import _load_talker_dims
    model_dir = Path(utils["DEFAULT_OUTPUT_DIR"]) / args.variant
    H, num_kv_heads, head_dim, num_layers = _load_talker_dims(model_dir)
    inputs_embeds_np = prefill_embeds.detach().cpu().float().numpy()
    pad_embed_np = pad_embed.detach().cpu().float().numpy()
    if pad_embed_np.ndim == 2:
        pad_embed_np = pad_embed_np.reshape(1, 1, -1)
    trailing_np = None
    if trailing_list:
        trailing_np = np.stack([t.detach().cpu().float().numpy() for t in trailing_list], axis=0)
        if trailing_np.ndim == 3:
            trailing_np = trailing_np[:, np.newaxis, :, :]
    codes_ort, ort_eos = run_ort_decode_loop(
        session, output_names, inputs_embeds_np, trailing_np, pad_embed_np,
        inputs_embeds_np.shape[1], num_layers, num_kv_heads, head_dim,
        args.max_steps, codec_eos_id,
    )
    logger.info("ORT: T=%d, eos_step=%d", codes_ort.shape[0], ort_eos)

    # (4) Optional TRT
    wav_trt = None
    if args.triton_url:
        try:
            grpcclient = _check_triton_client()
            triton_client = grpcclient.InferenceServerClient(url=args.triton_url)
            if not triton_client.is_server_ready():
                raise RuntimeError(f"Triton not ready at {args.triton_url}")
            req_dict = build_variant_request_payload(
                variant=args.variant, text=args.text, language=args.language,
                speaker=args.speaker or "serena", instruct=args.instruct or "",
            )
            stream = infer_stream(triton_client, grpcclient, req_dict, timeout=120)
            if stream.error:
                raise RuntimeError(f"Orchestrator error: {stream.error}")
            if stream.audio is not None and stream.audio.size:
                wav_trt = stream.audio.astype(np.float32)
                sample_rate = int(stream.metadata.get("audio_format", {}).get("sample_rate", sr) or sr)
                logger.info("TRT (orchestrator): %d samples (%.2fs)", len(wav_trt), len(wav_trt) / sample_rate)
        except Exception as e:
            logger.warning("TRT orchestrator failed: %s", e)

    def trim_at_eos(codes, eos_step):
        return codes[:eos_step] if eos_step >= 0 else codes

    wav_manual, _ = decode_codes_to_wav(model, trim_at_eos(codes_manual, manual_eos), device)
    sf.write(str(out_dir / "manual_pytorch.wav"), wav_manual, sr)
    wav_ort, _ = decode_codes_to_wav(model, trim_at_eos(codes_ort, ort_eos), device)
    sf.write(str(out_dir / "ort_fp32.wav"), wav_ort, sr)
    if wav_trt is not None:
        sf.write(str(out_dir / "trt_bf16.wav"), wav_trt, sr)

    report_lines = [
        "Audio compare report (proto / manual / ORT / TRT)",
        "==================================================",
        f"text: {args.text[:60]}...", f"variant: {args.variant}",
        f"do_sample: {do_sample}", f"max_steps: {args.max_steps}", "",
        "Audio lengths:",
        f"  prototype:  {len(wav_proto)} samples ({len(wav_proto)/sr:.2f} s)",
        f"  manual:     {len(wav_manual)} samples ({len(wav_manual)/sr:.2f} s), eos_step={manual_eos}",
        f"  ORT:        {len(wav_ort)} samples ({len(wav_ort)/sr:.2f} s), eos_step={ort_eos}",
    ]
    if wav_trt is not None:
        report_lines.append(f"  TRT:        {len(wav_trt)} samples ({len(wav_trt)/sr:.2f} s)")
    report_lines += ["", "Output WAV files (24 kHz):",
                     f"  proto.wav          - official API ({'sampling' if do_sample else 'greedy'})",
                     "  manual_pytorch.wav - manual PyTorch decode loop (greedy)",
                     "  ort_fp32.wav       - ORT talker_unified.onnx (greedy)"]
    if wav_trt is not None:
        report_lines.append("  trt_bf16.wav       - TRT via tts_orchestrator streaming")
    (out_dir / "compare_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("WAV and report saved to %s", out_dir)
    del model, wrapper
    torch.cuda.empty_cache()


# ── Mode: fused-onnx ────────────────────────────────────────────────────────

SLIDING_WINDOW = 72


def mode_fused_onnx(args):
    import torch

    print(DEPRECATION_NOTICE)
    exported_dir = REPO_ROOT / "workspace" / "exported" / args.variant
    onnx_path = exported_dir / "talker_code2wav_fused.onnx"
    weights_dir = exported_dir / "weights"
    manifest_path = exported_dir / "triton_manifest.json"

    model_dir_map = {
        "custom-0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "custom-1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }
    model_dir_name = model_dir_map.get(args.variant)
    if model_dir_name is None:
        logger.error("Unsupported variant: %s", args.variant)
        sys.exit(1)
    tokenizer_dir = REPO_ROOT / "workspace" / "models" / model_dir_name

    if not onnx_path.exists():
        logger.error("ONNX not found: %s", onnx_path)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    c2w_in_names = manifest["code2wav_fused"]["c2w_state_input_names"]
    c2w_out_names = manifest["code2wav_fused"]["c2w_state_output_names"]
    init_shapes = [tuple(s) for s in manifest["code2wav_fused"]["initial_state_shapes"]]
    num_layers = manifest["talker"]["num_layers"]
    num_kv_heads = manifest["talker"]["num_kv_heads"]
    head_dim = manifest["talker"]["head_dim"]
    codec_vocab_size = manifest["talker"]["vocab_size"]

    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
    from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer

    logger.info("Loading weights and tokenizer...")
    weights = EmbeddingWeights(str(weights_dir), device_id=0)
    tokenizer = load_lightweight_tokenizer(str(tokenizer_dir))
    builder = PrefillBuilder(weights, tokenizer)

    plan = builder.build_plan(TaskType.CUSTOM_VOICE, text=args.text, language=args.language, speaker=args.speaker)
    prefill_embeds = plan.prefill_embeds
    trailing = plan.trailing
    pad_embed = weights.tts_pad_embed
    codec_eos_id = int(weights.codec_eos_id)
    batch = 1

    logger.info("Prefill embeds: %s, trailing: %d, codec_eos_id: %d",
                prefill_embeds.shape, len(trailing), codec_eos_id)

    ort = _check_ort()
    providers = ["CUDAExecutionProvider"] if args.provider == "cuda" else ["CPUExecutionProvider"]
    logger.info("Loading fused ONNX model with %s ...", providers[0])
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_names = {i.name for i in sess.get_inputs()}
    output_names = [o.name for o in sess.get_outputs()]

    c2w_states_np = [np.zeros(shape, dtype=np.float32) for shape in init_shapes]
    talker_past_kv = np.empty((batch, num_layers * 2, num_kv_heads, 0, head_dim), dtype=np.float32)
    c2w_past_kv = np.empty((batch, 16, 16, 0, 64), dtype=np.float32)

    seq = prefill_embeds.shape[1]
    inp_np = prefill_embeds.detach().cpu().float().numpy()
    pos_np = (torch.arange(seq, dtype=torch.int64).reshape(1, 1, -1, 1)
              .expand(batch, 3, seq, 1).numpy())

    gumbel_noise = np.zeros((batch, 50), dtype=np.float32)
    cp_gumbel_noise = np.zeros((batch, 15, 50), dtype=np.float32)
    temperature = np.zeros((batch, 1), dtype=np.float32)
    penalty = np.ones((batch, 1), dtype=np.float32)
    token_counts = np.zeros((batch, codec_vocab_size), dtype=np.int64)

    def build_feed(inp, pos, cache_pos_val, past_kv_np, c2w_kv_np, c2w_st, tc):
        cur_seq = inp.shape[1]
        past_len = past_kv_np.shape[3]
        c2w_past_len = c2w_kv_np.shape[3]
        chunk_t = 1
        feed = {
            "input_embeds": inp.astype(np.float32),
            "position_ids": pos.astype(np.int64),
            "attention_bias": np.zeros((batch, 1, cur_seq, past_len + cur_seq), dtype=np.float32),
            "token_counts": tc.astype(np.int64),
            "gumbel_noise": gumbel_noise, "cp_gumbel_noise": cp_gumbel_noise,
            "temperature": temperature, "penalty": penalty,
            "cache_position": np.full((batch, chunk_t), cache_pos_val, dtype=np.float32),
            "c2w_attention_bias": np.zeros(
                (batch, 1, chunk_t, min(c2w_past_len + chunk_t, SLIDING_WINDOW)), dtype=np.float32),
            "talker_past_kv": past_kv_np.astype(np.float32),
            "c2w_past_kv": c2w_kv_np.astype(np.float32),
        }
        for name, st in zip(c2w_in_names, c2w_st):
            feed[name] = st.astype(np.float32)
        return {k: v for k, v in feed.items() if k in input_names}

    wav_chunks: list = []
    all_codec_0: list = []
    eos_step = -1

    logger.info("Running prefill (seq=%d) ...", seq)
    t0 = time.perf_counter()

    feed = build_feed(inp_np, pos_np, 0, talker_past_kv, c2w_past_kv, c2w_states_np, token_counts)
    outs = dict(zip(output_names, sess.run(output_names, feed)))

    codec_sum = outs["codec_sum"]
    full_codec = outs["full_codec"]
    wav_chunk = outs["wav"]
    token_counts = outs["updated_token_counts"].copy()
    talker_past_kv = outs["talker_new_kv"].copy()
    c2w_past_kv = outs["c2w_new_kv"].copy()
    kv_max = max(1, SLIDING_WINDOW - 1)
    if c2w_past_kv.shape[3] > kv_max:
        c2w_past_kv = c2w_past_kv[:, :, :, -kv_max:, :].copy()
    c2w_states_np = [outs[name].copy() for name in c2w_out_names]

    if wav_chunk is not None and wav_chunk.size > 0:
        wav_chunks.append(wav_chunk.flatten())
    codec_0 = int(full_codec[0, 0])
    all_codec_0.append(codec_0)
    if codec_0 == codec_eos_id:
        eos_step = 0

    text_add = trailing[0].detach().cpu().float().numpy() if len(trailing) > 0 else pad_embed.detach().cpu().float().numpy()
    if text_add.ndim == 2:
        text_add = text_add.reshape(1, 1, -1)
    next_input = (codec_sum.astype(np.float64) + text_add.astype(np.float64)).astype(np.float32)

    logger.info("Prefill done: codec_0=%d, wav_samples=%d", codec_0,
                wav_chunk.flatten().size if wav_chunk is not None else 0)

    for step in range(1, args.max_steps):
        if eos_step >= 0:
            break
        pos_step = np.full((batch, 3, 1, 1), seq + step - 1, dtype=np.int64)
        feed = build_feed(next_input, pos_step, step, talker_past_kv, c2w_past_kv, c2w_states_np, token_counts)
        outs = dict(zip(output_names, sess.run(output_names, feed)))

        codec_sum = outs["codec_sum"]
        full_codec = outs["full_codec"]
        wav_chunk = outs["wav"]
        token_counts = outs["updated_token_counts"].copy()
        talker_past_kv = np.concatenate([talker_past_kv, outs["talker_new_kv"]], axis=3).copy()
        c2w_past_kv = np.concatenate([c2w_past_kv, outs["c2w_new_kv"]], axis=3).copy()
        if c2w_past_kv.shape[3] > kv_max:
            c2w_past_kv = c2w_past_kv[:, :, :, -kv_max:, :].copy()
        c2w_states_np = [outs[name].copy() for name in c2w_out_names]

        if wav_chunk is not None and wav_chunk.size > 0:
            wav_chunks.append(wav_chunk.flatten())
        codec_0 = int(full_codec[0, 0])
        all_codec_0.append(codec_0)
        if codec_0 == codec_eos_id:
            eos_step = step

        text_add = (trailing[step].detach().cpu().float().numpy()
                    if step < len(trailing) else pad_embed.detach().cpu().float().numpy())
        if text_add.ndim == 2:
            text_add = text_add.reshape(1, 1, -1)
        next_input = (codec_sum.astype(np.float64) + text_add.astype(np.float64)).astype(np.float32)

        if step % 10 == 0:
            logger.info("  step %d: codec_0=%d", step, codec_0)

    elapsed = time.perf_counter() - t0
    if eos_step >= 0:
        wav_chunks = wav_chunks[:eos_step]

    if wav_chunks:
        full_wav = np.concatenate(wav_chunks).astype(np.float32)
        duration = len(full_wav) / SAMPLE_RATE
    else:
        full_wav = np.array([], dtype=np.float32)
        duration = 0.0

    out_dir = _default_out("audio_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fused_ort.wav"
    if full_wav.size > 0:
        save_wav(full_wav, out_path, sample_rate=SAMPLE_RATE)

    logger.info("=== RESULTS ===")
    logger.info("Text: %s", args.text)
    logger.info("Steps: %d, EOS step: %d", len(all_codec_0), eos_step)
    logger.info("WAV: %d samples, %.2f s", len(full_wav), duration)
    logger.info("Time: %.1f s", elapsed)
    logger.info("Saved: %s", out_path)


# ── Mode: long-ab ───────────────────────────────────────────────────────────

def mode_long_ab(args):
    import torch

    utils = _resolve_utils()
    utils["setup_logging"]()
    device = utils["resolve_device"](args.device)
    model_path = utils["resolve_model_path"](args.variant, args.models_dir)
    if not utils["has_model_weights"](model_path):
        raise SystemExit(f"No weights found at {model_path}")
    if not args.variant.lower().startswith("custom-"):
        raise SystemExit("This mode currently targets custom-* variants only.")

    from tests.support.engine_standalone import (
        GRPC_HOST, GRPC_PORT, LONG_TEXT, VERY_LONG_TEXT, SAMPLE_RATE as ESR,
        _check_server, _save_wav, _synthesize_oneshot,
    )

    text = args.text
    if not text:
        if args.case == "4a":
            text = LONG_TEXT
        elif args.case == "4b":
            text = VERY_LONG_TEXT

    out_dir = Path(args.out_dir) if args.out_dir else _default_out("audio_compare", f"listen_{args.case}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(DEPRECATION_NOTICE)

    manifest: dict = {
        "case": args.case, "variant": args.variant, "speaker": args.speaker,
        "language": args.language, "instruct": args.instruct,
        "text_chars": len(text), "sample_rate": ESR, "official": [], "engine": None,
    }

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper
    print(f"Loading official Qwen3TTSModel from {model_path} on {device} ...")
    wrapper = TTSModelWrapper.from_pretrained(
        str(model_path), device_map=str(device), dtype=torch.float32,
    )

    official_modes = [m.strip() for m in args.official_modes.split(",") if m.strip()]
    for mode in official_modes:
        if mode == "default":
            gen_kwargs = {"max_new_tokens": args.max_steps}
        elif mode == "engine_greedy":
            gen_kwargs = {
                "do_sample": False, "repetition_penalty": 1.05,
                "subtalker_dosample": False, "max_new_tokens": args.max_steps,
            }
        else:
            logger.warning("Unknown official mode: %s, skipping", mode)
            continue
        t0 = time.perf_counter()
        with torch.no_grad():
            wavs, sr = wrapper.generate_custom_voice(
                text=text, speaker=args.speaker, language=args.language,
                instruct=args.instruct, non_streaming_mode=False, **gen_kwargs,
            )
        elapsed = time.perf_counter() - t0
        wav = wavs[0]
        wav_path = out_dir / f"official_{mode}.wav"
        _save_wav(wav, str(wav_path))
        manifest["official"].append({
            "mode": mode, "path": str(wav_path),
            "duration_sec": len(wav) / sr if len(wav) else 0.0,
            "elapsed_sec": elapsed, "non_streaming_mode": False,
            "gen_kwargs": gen_kwargs,
        })
        print(f"Saved official {mode}: {wav_path}")

    if not args.skip_engine:
        host = args.host or GRPC_HOST
        port = args.port or GRPC_PORT
        if _check_server(host, port):
            result = _synthesize_oneshot(
                host, port, text=text, speaker=args.speaker,
                instruct=args.instruct, session_id=f"listen-{args.case}",
                timeout=args.timeout,
            )
            engine_entry = {
                "error": result.error, "warnings": result.warnings,
                "first_chunk_ms": result.first_chunk_ms, "ttft_ms": result.ttft_ms,
                "total_ms": result.total_ms, "duration_sec": result.duration_sec,
                "rtf": result.rtf,
            }
            if result.audio is not None and result.audio.size > 0:
                wav_path = out_dir / "engine_current.wav"
                _save_wav(result.audio, str(wav_path))
                engine_entry["path"] = str(wav_path)
                print(f"Saved engine current: {wav_path}")
            manifest["engine"] = engine_entry
        else:
            manifest["engine"] = {"error": f"engine server not reachable at {host}:{port}"}

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (out_dir / "LISTEN_README.txt").write_text(
        "\n".join([
            "Long Streaming Listen A/B", "=========================",
            f"case: {args.case}", f"variant: {args.variant}", f"text_chars: {len(text)}", "",
            "Questions to listen for:",
            "1. Does official_default also develop long noise / drift?",
            "2. If official_default is clean but engine_current is not, the gap is implementation/alignment.",
            "3. If both fail similarly, model/hparams become stronger suspects.", "",
            "Notes:",
            "- official_* uses Qwen3TTSModel with non_streaming_mode=False",
            "- engine_current is whatever the standalone server is currently running",
        ]) + "\n", encoding="utf-8",
    )
    print(f"Saved manifest and readme in: {out_dir}")


# ── CLI argument parser ─────────────────────────────────────────────────────

MODE_MAP = {
    "gen-triton":        mode_gen_triton,
    "gen-engine":        mode_gen_engine,
    "gen-reference":     mode_gen_reference,
    "compare-triton":    mode_compare_triton,
    "compare-ort":       mode_compare_ort,
    "compare-full-chain": mode_compare_full_chain,
    "compare-3way":      mode_compare_3way,
    "fused-onnx":        mode_fused_onnx,
    "long-ab":           mode_long_ab,
}


def _add_common_args(p, *, triton=False, engine=False, model=False, out=True):
    """Add frequently reused arguments to a sub-parser."""
    if triton:
        p.add_argument("--triton-url", default="localhost:8001", help="Triton gRPC URL")
        p.add_argument("--timeout", type=float, default=60.0, help="Request timeout (s)")
    if engine:
        p.add_argument("--host", default="localhost", help="Engine gRPC host")
        p.add_argument("--port", type=int, default=50051, help="Engine gRPC port")
        p.add_argument("--timeout", type=float, default=6000.0, help="Request timeout (s)")
    if model:
        p.add_argument("--variant", default="custom-1.7b", help="Model variant")
        p.add_argument("--speaker", default="serena", help="Speaker name")
        p.add_argument("--language", default="Chinese", help="Language")
        p.add_argument("--instruct", default="", help="VoiceDesign instruction")
        p.add_argument("--max-steps", type=int, default=500, help="Max decode steps")
        p.add_argument("--greedy", action="store_true", help="Use greedy (do_sample=False)")
        p.add_argument("--device", default=None, help="Torch device (auto-detect if omitted)")
        p.add_argument("--models-dir", default=None, help="Override models directory")
    if out:
        p.add_argument("--out-dir", default=None, help="Output directory override")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_audio",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True, help="Comparison / generation mode")

    # gen-triton
    p = sub.add_parser("gen-triton", help="Generate WAVs via Triton streaming")
    _add_common_args(p, triton=True, model=False)
    p.add_argument("--speaker", default="zhitian", help="Speaker name")

    # gen-engine
    p = sub.add_parser("gen-engine", help="Generate audio via engine gRPC")
    _add_common_args(p, engine=True, model=False)
    p.add_argument("--speaker", default="Serena", help="Speaker name")

    # gen-reference
    p = sub.add_parser("gen-reference", help="Generate reference audio (official API)")
    _add_common_args(p, model=True)
    p.set_defaults(variant="custom-1.7b", speaker="vivian")

    # compare-triton
    p = sub.add_parser("compare-triton", help="Official API vs Triton orchestrator A/B")
    _add_common_args(p, triton=True, model=True)
    p.add_argument("--text", default="你好，这是一段用于听感对比的测试语音。", help="Input text")

    # compare-ort
    p = sub.add_parser("compare-ort", help="Official API vs fused ONNX (ORT) A/B")
    _add_common_args(p, model=True)
    p.add_argument("--text", default="你好，这是一段用于对比的测试语音。", help="Input text")

    # compare-full-chain
    p = sub.add_parser("compare-full-chain", help="Full-chain 4-way listen compare")
    _add_common_args(p, triton=True, model=True, out=False)
    p.add_argument("--text", default="你好，这是一次官方 API、融合 ONNX、Triton 直连与编排器的全链路听感对比。")
    p.add_argument("--text-file", default=None, help="Read text from file (overrides --text)")
    p.add_argument("--out-root", default=None, help="Output root dir (default: workspace/audio_compare/runs/)")
    p.add_argument("--skip-triton", action="store_true", help="Skip Triton steps")

    # compare-3way
    p = sub.add_parser("compare-3way", help="Three-way: proto vs manual PyTorch vs ORT")
    _add_common_args(p, model=True)
    p.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。", help="Input text")
    p.add_argument("--triton-url", default="", help="Triton gRPC URL (empty=skip TRT path)")

    # fused-onnx
    p = sub.add_parser("fused-onnx", help="Run fused ONNX decode loop, save WAV")
    _add_common_args(p, model=False, out=False)
    p.add_argument("--variant", default="custom-1.7b", help="Model variant")
    p.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。", help="Input text")
    p.add_argument("--speaker", default="vivian", help="Speaker name")
    p.add_argument("--language", default="auto", help="Language")
    p.add_argument("--max-steps", type=int, default=300, help="Max decode steps")
    p.add_argument("--provider", default="cpu", choices=["cpu", "cuda"], help="ORT execution provider")

    # long-ab
    p = sub.add_parser("long-ab", help="Long-text A/B: official vs engine")
    _add_common_args(p, engine=True, model=True, out=False)
    p.add_argument("--case", choices=["4a", "4b"], default="4a", help="Text case preset")
    p.add_argument("--text", default=None, help="Override text (instead of case preset)")
    p.add_argument("--official-modes", default="default,engine_greedy",
                   help="Comma-separated official modes: default,engine_greedy")
    p.add_argument("--skip-engine", action="store_true", help="Skip engine gRPC call")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.mode not in MODE_MAP:
        parser.print_help()
        sys.exit(1)
    MODE_MAP[args.mode](args)


if __name__ == "__main__":
    main()
