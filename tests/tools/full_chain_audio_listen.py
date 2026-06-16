#!/usr/bin/env python3
"""
Full-chain listenability bundle (priority-2 validation, ear-first).

Writes WAV files for A/B listening in one timestamped directory:

  - proto.wav              — Official Qwen3TTSModel API (generate_*)
  - fused_onnx.wav         — ORT CPU, talker_code2wav_fused.onnx (greedy, orchestrator-style)
  - fused_triton_direct.wav — Triton `talker_code2wav_fused` (TRT or ONNX per deployment)
  - orchestrator.wav       — Triton `tts_orchestrator` streaming (production BLS path)

Prerequisites:
  - conda env qwen3-tts, weights under workspace/models
  - workspace/exported/<variant>/talker_code2wav_fused.onnx + triton_manifest.json
  - Triton running with assembled repo (same variant), e.g.:
      bash scripts/bash/build_triton.sh assemble --engine-mode trt --variant custom-1.7b
      bash scripts/bash/build_triton.sh run

Usage:
  cd /path/to/Qwen3-TTS-Triton && conda activate qwen3-tts
  python tests/tools/full_chain_audio_listen.py \\
      --variant custom-1.7b --text "你好，这是一段全链路听感对比。" --speaker serena \\
      --triton-url localhost:8001

  # Skip Triton (only local official + fused ORT):
  python tests/tools/full_chain_audio_listen.py --variant custom-1.7b --skip-triton

  # Longer passage (rollover stress: single orchestrator request; listen for cuts/glitches):
  python tests/tools/full_chain_audio_listen.py --variant custom-1.7b --text-file ./my.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports("repo", "scripts_python", "scripts_export", "scripts", "third_party_qwen")
from tests.support.triton_streaming import build_variant_request_payload, infer_stream

from official_prefill import build_prefill_like_official
from utils import (
    setup_logging,
    resolve_model_path,
    resolve_device,
    has_model_weights,
    DEFAULT_OUTPUT_DIR,
)

from compare_official_vs_fused_onnx import (
    OFFICIAL_ASSISTANT_FMT,
    run_fused_onnx_loop,
    run_fused_triton_loop,
    _load_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("full_chain_listen")


def _slug(s: str, max_len: int = 40) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^\w\u4e00-\u9fff_-]", "", s)
    return (s[:max_len] if s else "run").strip("_") or "run"


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="custom-1.7b")
    p.add_argument(
        "--text",
        default="你好，这是一次官方 API、融合 ONNX、Triton 直连与编排器的全链路听感对比。",
    )
    p.add_argument("--text-file", default=None, help="Read UTF-8 text from file (overrides --text)")
    p.add_argument("--language", default="Chinese")
    p.add_argument("--speaker", default="serena")
    p.add_argument("--instruct", default="")
    p.add_argument(
        "--max-steps",
        type=int,
        default=256,
        help="Fused ORT/direct loop max decode steps (avoid huge silent tails if EOS is late)",
    )
    p.add_argument("--triton-url", default="localhost:8001")
    p.add_argument(
        "--out-root",
        default=None,
        help="Default: workspace/audio_compare/runs/",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--models-dir", default=None)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="do_sample=False for official API (more repeatable vs sampling)",
    )
    p.add_argument(
        "--skip-triton",
        action="store_true",
        help="Only proto + fused_onnx (no Triton server required)",
    )
    args = p.parse_args()

    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
        if not text:
            logger.error("Empty --text-file")
            sys.exit(1)

    device = resolve_device(args.device)
    path = resolve_model_path(args.variant, args.models_dir)
    if not has_model_weights(path):
        logger.error("No weights for variant %s", args.variant)
        sys.exit(1)

    exported_dir = Path(DEFAULT_OUTPUT_DIR) / args.variant
    onnx_path = exported_dir / "talker_code2wav_fused.onnx"
    if not onnx_path.is_file():
        logger.error("Missing fused ONNX: %s (run export_09)", onnx_path)
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "workspace" / "audio_compare" / "runs"
    out_dir = out_root / f"{stamp}_{_slug(text)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    import onnxruntime as ort

    manifest = _load_manifest(exported_dir)
    talker = manifest.get("talker", {})
    num_layers = int(talker.get("num_layers", 28))
    num_kv_heads = int(talker.get("num_kv_heads", 8))
    head_dim = int(talker.get("head_dim", 128))

    sess = ort.InferenceSession(
        str(onnx_path),
        ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    onnx_input_names = [i.name for i in sess.get_inputs()]
    onnx_output_names = [o.name for o in sess.get_outputs()]

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper

    logger.info("Loading official Qwen3TTSModel ...")
    wrapper = TTSModelWrapper.from_pretrained(
        str(path), device_map=str(device), dtype=torch.float32
    )
    model = wrapper.model
    processor = wrapper.processor

    do_sample = not args.greedy
    gen_kwargs = dict(do_sample=do_sample, max_new_tokens=args.max_steps)
    if not do_sample:
        gen_kwargs["repetition_penalty"] = 1.0
        gen_kwargs["subtalker_dosample"] = False

    non_streaming = "design" in args.variant.lower()
    with torch.no_grad():
        if "design" in args.variant.lower():
            wavs, sr = wrapper.generate_voice_design(
                text=text,
                instruct=args.instruct,
                language=args.language,
                non_streaming_mode=non_streaming,
                **gen_kwargs,
            )
        else:
            wavs, sr = wrapper.generate_custom_voice(
                text=text,
                speaker=args.speaker,
                language=args.language,
                instruct=args.instruct,
                non_streaming_mode=non_streaming,
                **gen_kwargs,
            )
    wav_proto = wavs[0]
    sf.write(str(out_dir / "proto.wav"), wav_proto, sr)
    logger.info("Wrote proto.wav (%.2fs @ %d Hz)", len(wav_proto) / sr, sr)

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
        model,
        input_ids,
        args.language,
        args.speaker or "",
        device,
        instruct_ids=instruct_ids,
        non_streaming_mode=non_streaming,
    )

    tts_pad_token_id = getattr(model.config, "tts_pad_token_id", 0)
    pad_id = torch.tensor([[tts_pad_token_id]], device=device, dtype=torch.long)
    pad_embed = model.talker.text_projection(
        model.talker.model.text_embedding(pad_id)
    )

    B, S, _ = prefill_embeds.shape
    position_ids_1d = torch.arange(S, device=device, dtype=torch.int64)
    position_ids_prefill = position_ids_1d.reshape(1, 1, -1, 1).expand(B, 3, S, 1)

    codec_eos_id = int(model.config.talker_config.codec_eos_token_id)

    logger.info("Running fused ORT loop ...")
    wav_fused_ort = run_fused_onnx_loop(
        sess,
        manifest,
        inputs_embeds=prefill_embeds,
        position_ids_prefill=position_ids_prefill,
        trailing_text=trailing_list,
        pad_embed=pad_embed,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        codec_eos_id=codec_eos_id,
        max_steps=args.max_steps,
    )
    sf.write(str(out_dir / "fused_onnx.wav"), wav_fused_ort, sr)
    logger.info("Wrote fused_onnx.wav (%.2fs)", len(wav_fused_ort) / sr)

    readme_lines = [
        "Full-chain audio listen (ear-first)",
        "====================================",
        f"variant: {args.variant}",
        f"text (first 120 chars): {text[:120]}",
        f"do_sample (official): {do_sample}",
        f"max_steps: {args.max_steps}",
        "",
        "Files:",
        "  proto.wav               — Official Qwen3TTSModel API (sampling unless --greedy).",
        "  fused_onnx.wav          — talker_code2wav_fused.onnx via ORT (CPU), greedy loop.",
        "  fused_triton_direct.wav — Triton model talker_code2wav_fused (engine = TRT or ONNX per repo).",
        "  orchestrator.wav        — tts_orchestrator gRPC stream (BLS + fused backend).",
        "",
        "proto uses high-level generate_*; fused_* use greedy argmax like orchestrator decode.",
        "If proto is much longer than fused_onnx: official path often uses sampling (unless --greedy);",
        "fused paths stop at EOS or --max-steps.",
        "",
        "NOTE: fused_onnx runs in FP32 (CPU ORT). The model was trained in BF16;",
        "FP32 autoregressive decode accumulates numerical drift in KV cache, causing",
        "silence after ~8 steps. The loop stops early on detecting silence.",
        "Use fused_triton_direct (TRT BF16) or orchestrator for correct full-length audio.",
        "Listen for naturalness, clicks, noise, and drift vs proto.",
        "",
    ]

    if not args.skip_triton:
        try:
            import tritonclient.grpc as grpcclient
        except ImportError:
            logger.error("pip install tritonclient[grpc]")
            sys.exit(1)

        client = grpcclient.InferenceServerClient(url=args.triton_url)
        if not client.is_server_ready():
            logger.error("Triton not ready at %s", args.triton_url)
            sys.exit(1)

        cfg = client.get_model_config("talker_code2wav_fused", as_json=True)["config"]
        triton_input_dtypes = {
            item["name"]: item["data_type"] for item in cfg["input"]
        }

        logger.info("Running Triton talker_code2wav_fused (direct) ...")
        wav_fused_triton = run_fused_triton_loop(
            client,
            triton_input_dtypes,
            onnx_input_names,
            onnx_output_names,
            manifest,
            inputs_embeds=prefill_embeds,
            position_ids_prefill=position_ids_prefill,
            trailing_text=trailing_list,
            pad_embed=pad_embed,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            codec_eos_id=codec_eos_id,
            max_steps=args.max_steps,
        )
        sf.write(str(out_dir / "fused_triton_direct.wav"), wav_fused_triton, sr)
        logger.info("Wrote fused_triton_direct.wav (%.2fs)", len(wav_fused_triton) / sr)

        req = build_variant_request_payload(
            variant=args.variant,
            text=text,
            language=args.language,
            speaker=args.speaker,
            instruct=args.instruct,
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
        logger.info(
            "Wrote orchestrator.wav (%.2fs, wall=%.2fs)",
            len(wav_orch) / orchestrator_sr,
            elapsed,
        )
        readme_lines.append(f"orchestrator wall time: {elapsed:.2f}s")
    else:
        readme_lines.extend(
            [
                "Triton steps skipped (--skip-triton).",
                "",
            ]
        )

    (out_dir / "LISTEN_README.txt").write_text(
        "\n".join(readme_lines), encoding="utf-8"
    )
    logger.info("Done. Open %s and listen.", out_dir / "LISTEN_README.txt")


if __name__ == "__main__":
    main()
