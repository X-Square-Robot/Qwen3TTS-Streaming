#!/usr/bin/env python3
"""
Listenability A/B: official Python API vs Triton tts_orchestrator (fused talker+code2wav).

This does NOT assert numerical parity — it writes two WAV files for you to listen:

  - proto.wav       — `Qwen3TTSModel` high-level API (same as upstream usage).
  - triton.wav      — streaming audio from Triton (production path; uses fused engine when deployed).

Prerequisites:
  - conda env with qwen3-tts (or project venv) and weights under workspace/models.
  - Triton running with assembled repo, e.g.:
      bash scripts/bash/build_triton.sh assemble --engine-mode onnx --variant custom-1.7b
      bash scripts/bash/build_triton.sh run

Usage:
  conda activate qwen3-tts
  python tests/tools/compare_official_vs_triton_audio.py --variant custom-1.7b \\
      --text "你好，这是一段测试。" --speaker serena --triton-url localhost:8001

  # VoiceDesign variant:
  python tests/tools/compare_official_vs_triton_audio.py --variant design-1.7b \\
      --text "Hello" --instruct "Speak calmly." --triton-url localhost:8001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import soundfile as sf
import torch

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports("repo", "scripts_export", "scripts", "third_party_qwen")
from tests.support.triton_streaming import build_variant_request_payload, infer_stream

from utils import (
    setup_logging,
    resolve_model_path,
    resolve_device,
    has_model_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_official_triton")


def main():
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="custom-1.7b")
    p.add_argument("--text", default="你好，这是一段用于听感对比的测试语音。")
    p.add_argument("--language", default="Chinese")
    p.add_argument("--speaker", default="serena", help="custom-* only")
    p.add_argument("--instruct", default="", help="design-* only")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--triton-url", default="localhost:8001")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--models-dir", default=None)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="do_sample=False for official API (slightly more repeatable vs sampling)",
    )
    args = p.parse_args()

    device = resolve_device(args.device)
    path = resolve_model_path(args.variant, args.models_dir)
    if not has_model_weights(path):
        logger.error("No weights for variant %s at %s", args.variant, path)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "workspace" / "audio_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper

    logger.info("Loading official Qwen3TTSModel from %s ...", path)
    wrapper = TTSModelWrapper.from_pretrained(
        str(path), device_map=str(device), dtype=torch.float32
    )

    do_sample = not args.greedy
    gen_kwargs = dict(do_sample=do_sample, max_new_tokens=args.max_steps)
    if not do_sample:
        gen_kwargs["repetition_penalty"] = 1.0
        gen_kwargs["subtalker_dosample"] = False

    # Keep the official path aligned with the engine/orchestrator streaming-text
    # semantics for like-for-like listening on long-form cases.
    non_streaming = "design" in args.variant.lower()
    with torch.no_grad():
        if "design" in args.variant.lower():
            wavs, sr = wrapper.generate_voice_design(
                text=args.text,
                instruct=args.instruct,
                language=args.language,
                non_streaming_mode=non_streaming,
                **gen_kwargs,
            )
        else:
            wavs, sr = wrapper.generate_custom_voice(
                text=args.text,
                speaker=args.speaker,
                language=args.language,
                instruct=args.instruct,
                non_streaming_mode=non_streaming,
                **gen_kwargs,
            )

    wav_proto = wavs[0]
    sf.write(str(out_dir / "proto.wav"), wav_proto, sr)
    logger.info("Wrote %s (official API, %d samples, %.2fs)", out_dir / "proto.wav", len(wav_proto), len(wav_proto) / sr)

    # Triton (fused path in production)
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        logger.error("Install: pip install tritonclient[grpc]")
        sys.exit(1)

    req = build_variant_request_payload(
        variant=args.variant,
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
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
    # Orchestrator outputs float32 mono at model sample rate (typically 24kHz)
    sf.write(str(out_dir / "triton.wav"), wav_triton, sample_rate)
    logger.info(
        "Wrote %s (Triton orchestrator, %d samples, %.2fs, elapsed=%.2fs)",
        out_dir / "triton.wav",
        len(wav_triton),
        len(wav_triton) / sample_rate,
        elapsed,
    )

    readme = out_dir / "LISTEN_README.txt"
    readme.write_text(
        "\n".join(
            [
                "Official API vs Triton (listenability)",
                "========================================",
                f"variant: {args.variant}",
                f"text: {args.text[:80]}",
                "",
                "proto.wav  — Qwen3TTSModel official high-level API (this repo: qwen_tts.inference).",
                "triton.wav — tts_orchestrator gRPC stream (deployed fused talker_code2wav_fused path).",
                "",
                "Sampling: official uses do_sample=%s; Triton uses orchestrator decode settings."
                % do_sample,
                "",
                "These two are not expected to be bit-identical; compare by ear.",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Done. Read %s and A/B listen to proto.wav vs triton.wav", readme)


if __name__ == "__main__":
    main()
