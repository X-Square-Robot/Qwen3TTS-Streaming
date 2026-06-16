#!/usr/bin/env python3
"""Generate long-text A/B audio for manual listening.

This script focuses on one question:
does the official high-level API also degrade on long-text streaming semantics?

Outputs:
  - official_*.wav: official Qwen3TTSModel path with `non_streaming_mode=False`
  - engine_current.wav: current standalone engine output, if the gRPC server is reachable
  - manifest.json / LISTEN_README.txt: metadata for manual A/B listening
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

REPO_ROOT = bootstrap_tool_imports()
from qwen3tts_tools.common import bootstrap_project_imports

bootstrap_project_imports("scripts_export", "third_party_qwen")

from utils import has_model_weights, resolve_device, resolve_model_path, setup_logging
from tests.support.engine_standalone import (
    GRPC_HOST,
    GRPC_PORT,
    LONG_TEXT,
    SAMPLE_RATE,
    VERY_LONG_TEXT,
    _check_server,
    _save_wav,
    _synthesize_oneshot,
)


def _resolve_text(case: str, text_override: str | None) -> str:
    if text_override:
        return text_override
    if case == "4a":
        return LONG_TEXT
    if case == "4b":
        return VERY_LONG_TEXT
    raise ValueError(f"Unsupported case: {case}")


def _official_kwargs(mode: str, max_steps: int) -> dict:
    if mode == "default":
        return {"max_new_tokens": max_steps}
    if mode == "engine_greedy":
        return {
            "do_sample": False,
            "repetition_penalty": 1.05,
            "subtalker_dosample": False,
            "max_new_tokens": max_steps,
        }
    raise ValueError(f"Unsupported official mode: {mode}")


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["4a", "4b"], default="4a")
    p.add_argument("--text", default=None, help="Optional raw text override")
    p.add_argument("--variant", default="custom-1.7b")
    p.add_argument("--speaker", default="Vivian")
    p.add_argument("--language", default="Chinese")
    p.add_argument("--instruct", default="")
    p.add_argument("--models-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument(
        "--official-modes",
        default="default,engine_greedy",
        help="Comma-separated list: default,engine_greedy",
    )
    p.add_argument("--skip-engine", action="store_true")
    p.add_argument("--host", default=GRPC_HOST)
    p.add_argument("--port", type=int, default=GRPC_PORT)
    p.add_argument("--timeout", type=float, default=6000.0)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    if not args.variant.lower().startswith("custom-"):
        raise SystemExit("This script currently targets custom-* variants only.")

    text = _resolve_text(args.case, args.text)
    device = resolve_device(args.device)
    model_path = resolve_model_path(args.variant, args.models_dir)
    if not has_model_weights(model_path):
        raise SystemExit(f"No weights found at {model_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "workspace" / "audio_compare" / f"listen_{args.case}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "case": args.case,
        "variant": args.variant,
        "speaker": args.speaker,
        "language": args.language,
        "instruct": args.instruct,
        "text_chars": len(text),
        "sample_rate": SAMPLE_RATE,
        "official": [],
        "engine": None,
    }

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper

    print(f"Loading official Qwen3TTSModel from {model_path} on {device} ...")
    wrapper = TTSModelWrapper.from_pretrained(
        str(model_path), device_map=str(device), dtype=torch.float32,
    )

    official_modes = [m.strip() for m in args.official_modes.split(",") if m.strip()]
    for mode in official_modes:
        gen_kwargs = _official_kwargs(mode, args.max_steps)
        t0 = time.perf_counter()
        with torch.no_grad():
            wavs, sr = wrapper.generate_custom_voice(
                text=text,
                speaker=args.speaker,
                language=args.language,
                instruct=args.instruct,
                non_streaming_mode=False,
                **gen_kwargs,
            )
        elapsed = time.perf_counter() - t0
        wav = wavs[0]
        wav_path = out_dir / f"official_{mode}.wav"
        _save_wav(wav, str(wav_path))
        manifest["official"].append(
            {
                "mode": mode,
                "path": str(wav_path),
                "duration_sec": len(wav) / sr if len(wav) else 0.0,
                "elapsed_sec": elapsed,
                "non_streaming_mode": False,
                "gen_kwargs": gen_kwargs,
            }
        )
        print(f"Saved official {mode}: {wav_path}")

    if not args.skip_engine:
        if _check_server(args.host, args.port):
            result = _synthesize_oneshot(
                args.host,
                args.port,
                text=text,
                speaker=args.speaker,
                instruct=args.instruct,
                session_id=f"listen-{args.case}",
                timeout=args.timeout,
            )
            engine_entry = {
                "error": result.error,
                "warnings": result.warnings,
                "first_chunk_ms": result.first_chunk_ms,
                "ttft_ms": result.ttft_ms,
                "total_ms": result.total_ms,
                "duration_sec": result.duration_sec,
                "rtf": result.rtf,
            }
            if result.audio is not None and result.audio.size > 0:
                wav_path = out_dir / "engine_current.wav"
                _save_wav(result.audio, str(wav_path))
                engine_entry["path"] = str(wav_path)
                print(f"Saved engine current: {wav_path}")
            manifest["engine"] = engine_entry
        else:
            manifest["engine"] = {
                "error": f"engine server not reachable at {args.host}:{args.port}",
            }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = out_dir / "LISTEN_README.txt"
    readme.write_text(
        "\n".join(
            [
                "Long Streaming Listen A/B",
                "=========================",
                f"case: {args.case}",
                f"variant: {args.variant}",
                f"text_chars: {len(text)}",
                "",
                "Questions to listen for:",
                "1. Does official_default also develop long noise / drift?",
                "2. If official_default is clean but engine_current is not, the gap is implementation/alignment.",
                "3. If official_default and engine_current both fail similarly, model/hparams become stronger suspects.",
                "",
                "Notes:",
                "- official_* uses Qwen3TTSModel with non_streaming_mode=False",
                "- engine_current is whatever the standalone server is currently running",
                f"- full metadata: {manifest_path.name}",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved readme: {readme}")


if __name__ == "__main__":
    main()
