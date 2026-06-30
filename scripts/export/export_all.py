#!/usr/bin/env python3
"""
Master export script: exports all Qwen3-TTS components for Triton deployment.

Order (atomic subgraphs before composed ONNX):
  01. Embeddings → .pt + config
  02. Speaker Encoder → ONNX (base only)
  03. Speech Tokenizer Encoder → ONNX (shared)
  04. Speech Tokenizer + Codec 3D fused → ONNX (base only)
  05. Code Predictor → ONNX (verification)
  06. Code2Wav Decoder → ONNX (shared, verification)
  07. Talker backbone → ONNX (verification)
  08. Talker Unified (+CP+sum) → ONNX (verification)
  09. Talker + Code2Wav fused → ONNX (production)

Use --skip-verification to run only 01–04 + 09 (minimal for production TRT build).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from utils import (
    setup_logging,
    MODEL_VARIANTS,
    has_model_weights,
    resolve_device,
    resolve_dtype,
    DTYPE_NAMES,
)

logger = logging.getLogger("onnx_export")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Export all Qwen3-TTS components to ONNX / PyTorch weights"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=f"Model variant. Available: {list(MODEL_VARIANTS.keys())}. "
        "Default: export all downloaded variants",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Models directory (default: workspace/models)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: workspace/exported)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu | cuda | cuda:0 ... (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Target inference precision for embedding weights (default: bf16)",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip shared tokenizer ONNX exports (steps 03, 06)",
    )
    parser.add_argument(
        "--skip-speech-codec-fused",
        action="store_true",
        help="Skip step 04 (Speech Tokenizer + Codec 3D fused, base only)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip steps 05–08 (code_predictor, code2wav, talker_backbone, talker_unified)",
    )
    parser.add_argument(
        "--skip-talker-code2wav-fused",
        action="store_true",
        help="Skip step 09 (Talker+Code2Wav fused production ONNX)",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true", help="Skip step 01 (embeddings)"
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    logger.info(
        f"Export config: device={device}, target_dtype={DTYPE_NAMES[dtype]}, onnx_export=fp32"
    )

    t_start = time.time()
    results = {}

    from utils import DEFAULT_MODELS_DIR

    models_base = Path(args.models_dir) if args.models_dir else DEFAULT_MODELS_DIR

    if args.variant:
        variant_dir = models_base / MODEL_VARIANTS[args.variant]
        if not has_model_weights(variant_dir):
            logger.error(
                f"Variant '{args.variant}' has no weight files in {variant_dir}."
            )
            sys.exit(1)
        variants = [args.variant]
    else:
        variants = [
            v
            for v, dirname in MODEL_VARIANTS.items()
            if (models_base / dirname).exists()
            and has_model_weights(models_base / dirname)
        ]
        skipped = [
            v
            for v, dirname in MODEL_VARIANTS.items()
            if (models_base / dirname).exists()
            and not has_model_weights(models_base / dirname)
        ]
        if skipped:
            logger.warning(f"Skipping variants without weights: {skipped}")
        if not variants:
            logger.error(f"No model variants with weights found in {models_base}")
            sys.exit(1)
        logger.info(f"Detected variants with weights: {variants}")

    TOTAL = 9

    # ── Step 01: Embeddings ──
    if not args.skip_embeddings:
        for variant in variants:
            logger.info("=" * 60)
            logger.info(f"Step 01/{TOTAL}: [{variant}] Embeddings → .pt")
            logger.info("=" * 60)
            try:
                from export_01_embeddings import export_embeddings

                path = export_embeddings(
                    variant, args.models_dir, args.output_dir, device, dtype
                )
                results[f"{variant}/embeddings"] = ("OK", str(path))
            except Exception as e:
                logger.error(f"[{variant}] Embeddings failed: {e}", exc_info=True)
                results[f"{variant}/embeddings"] = ("FAILED", str(e))
    else:
        logger.info("Skipping step 01 (--skip-embeddings)")

    # ── Step 02: Speaker Encoder (base only) ──
    for variant in variants:
        if not variant.startswith("base-"):
            continue
        logger.info("=" * 60)
        logger.info(f"Step 02/{TOTAL}: [{variant}] Speaker Encoder → ONNX")
        logger.info("=" * 60)
        try:
            from export_02_speaker_encoder import export_speaker_encoder

            path = export_speaker_encoder(
                variant, args.models_dir, args.output_dir, device, dtype
            )
            results[f"{variant}/speaker_encoder"] = ("OK", path)
        except Exception as e:
            logger.error(f"[{variant}] Speaker Encoder failed: {e}", exc_info=True)
            results[f"{variant}/speaker_encoder"] = ("FAILED", str(e))

    # ── Step 03: Speech Tokenizer Encoder ──
    if not args.skip_tokenizer:
        logger.info("=" * 60)
        logger.info(f"Step 03/{TOTAL}: Speech Tokenizer Encoder → ONNX")
        logger.info("=" * 60)
        try:
            from export_03_speech_tokenizer_encoder import (
                export_speech_tokenizer_encoder,
            )

            path = export_speech_tokenizer_encoder(
                args.models_dir, args.output_dir, device, dtype
            )
            results["speech_tokenizer_encoder"] = ("OK", path)
        except Exception as e:
            logger.error(f"Speech Tokenizer Encoder failed: {e}", exc_info=True)
            results["speech_tokenizer_encoder"] = ("FAILED", str(e))
    else:
        logger.info("Skipping step 03 (--skip-tokenizer)")

    # ── Step 04: Speech + Codec 3D fused (base only) ──
    if not args.skip_speech_codec_fused:
        for variant in variants:
            if not variant.startswith("base-"):
                continue
            logger.info("=" * 60)
            logger.info(f"Step 04/{TOTAL}: [{variant}] Speech + Codec 3D fused → ONNX")
            logger.info("=" * 60)
            try:
                from export_04_speech_tokenizer_codec_fused import (
                    export_speech_tokenizer_codec_fused,
                )

                path = export_speech_tokenizer_codec_fused(
                    variant, args.models_dir, args.output_dir, device, dtype
                )
                results[f"{variant}/speech_tokenizer_codec_fused"] = ("OK", path)
            except FileNotFoundError as e:
                logger.warning(f"[{variant}] Step 04 skipped: {e}")
                results[f"{variant}/speech_tokenizer_codec_fused"] = ("SKIPPED", str(e))
            except Exception as e:
                logger.error(
                    f"[{variant}] Speech+Codec fused failed: {e}", exc_info=True
                )
                results[f"{variant}/speech_tokenizer_codec_fused"] = ("FAILED", str(e))
    elif args.skip_speech_codec_fused:
        logger.info("Skipping step 04 (--skip-speech-codec-fused)")

    # ── Steps 05–08: verification-only ONNX ──
    if not args.skip_verification:
        for variant in variants:
            logger.info("=" * 60)
            logger.info(
                f"Step 05/{TOTAL}: [{variant}] Code Predictor → ONNX (verification)"
            )
            logger.info("=" * 60)
            try:
                from export_05_code_predictor import export_code_predictor_unrolled

                path = export_code_predictor_unrolled(
                    variant, args.models_dir, args.output_dir, device
                )
                results[f"{variant}/code_predictor_unrolled"] = ("OK", path)
            except Exception as e:
                logger.error(
                    f"[{variant}] Code Predictor export failed: {e}", exc_info=True
                )
                results[f"{variant}/code_predictor_unrolled"] = ("FAILED", str(e))

        if not args.skip_tokenizer:
            logger.info("=" * 60)
            logger.info(f"Step 06/{TOTAL}: Code2Wav Decoder → ONNX")
            logger.info("=" * 60)
            try:
                from export_06_code2wav_decoder import export_code2wav_decoder

                path = export_code2wav_decoder(
                    args.models_dir, args.output_dir, device, dtype
                )
                results["code2wav_decoder"] = ("OK", path)
            except Exception as e:
                logger.error(f"Code2Wav Decoder failed: {e}", exc_info=True)
                results["code2wav_decoder"] = ("FAILED", str(e))
        else:
            logger.info("Skipping step 06 (--skip-tokenizer)")

        for variant in variants:
            logger.info("=" * 60)
            logger.info(f"Step 07/{TOTAL}: [{variant}] Talker backbone → ONNX")
            logger.info("=" * 60)
            try:
                from export_07_talker_backbone import export_talker_backbone

                out = export_talker_backbone(
                    variant, args.models_dir, args.output_dir, device
                )
                path = out.get("onnx", out) if isinstance(out, dict) else out
                results[f"{variant}/talker_backbone"] = ("OK", path)
            except Exception as e:
                logger.error(f"[{variant}] Talker backbone failed: {e}", exc_info=True)
                results[f"{variant}/talker_backbone"] = ("FAILED", str(e))

        for variant in variants:
            logger.info("=" * 60)
            logger.info(f"Step 08/{TOTAL}: [{variant}] Talker Unified → ONNX")
            logger.info("=" * 60)
            try:
                from export_08_talker_unified import export_talker_unified

                out = export_talker_unified(
                    variant, args.models_dir, args.output_dir, device
                )
                path = out.get("onnx", out) if isinstance(out, dict) else out
                results[f"{variant}/talker_unified"] = ("OK", path)
            except Exception as e:
                logger.error(f"[{variant}] Talker Unified failed: {e}", exc_info=True)
                results[f"{variant}/talker_unified"] = ("FAILED", str(e))
    else:
        logger.info("Skipping steps 05–08 (--skip-verification)")

    # ── Step 09: Talker + Code2Wav fused (production) ──
    if not args.skip_talker_code2wav_fused:
        for variant in variants:
            logger.info("=" * 60)
            logger.info(f"Step 09/{TOTAL}: [{variant}] Talker + Code2Wav fused → ONNX")
            logger.info("=" * 60)
            try:
                from export_09_talker_code2wav_fused import export_talker_code2wav_fused

                out = export_talker_code2wav_fused(
                    variant, args.models_dir, args.output_dir, device
                )
                path = out.get("onnx", out) if isinstance(out, dict) else out
                results[f"{variant}/talker_code2wav_fused"] = ("OK", path)
            except Exception as e:
                logger.error(
                    f"[{variant}] Talker+Code2Wav fused failed: {e}", exc_info=True
                )
                results[f"{variant}/talker_code2wav_fused"] = ("FAILED", str(e))
    else:
        logger.info("Skipping step 09 (--skip-talker-code2wav-fused)")

    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        f"  EXPORT SUMMARY  (target_dtype={DTYPE_NAMES[dtype]}, onnx=fp32, device={device})"
    )
    logger.info("=" * 60)

    n_ok = sum(1 for s, _ in results.values() if s == "OK")
    n_fail = sum(1 for s, _ in results.values() if s == "FAILED")

    for name, (status, detail) in results.items():
        icon = "✓" if status == "OK" else ("○" if status == "SKIPPED" else "✗")
        logger.info(f"  {icon} {name}: {status}")
        if status == "FAILED":
            logger.info(f"    → {detail}")

    logger.info("")
    logger.info(f"Total: {n_ok} succeeded, {n_fail} failed, {elapsed:.1f}s elapsed")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
