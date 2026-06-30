#!/usr/bin/env python3
"""
[Step 05] Export Speaker Encoder (ECAPA-TDNN) to ONNX.

Component: Speaker Encoder
Architecture: ECAPA-TDNN, mel_dim=128, enc_dim=1024
Input:  mel_spectrogram [B, T, 128]
Output: speaker_embedding [B, 1024]
Engine: ONNX Runtime
Usage: Called once per request (voice clone only)

Applies to: base-1.7b, base-0.6b (models with tts_model_type == "base")
"""

import argparse
import logging

import torch

from utils import (
    setup_logging,
    resolve_model_path,
    ensure_output_dir,
    load_tts_model,
    export_onnx,
    verify_onnx,
    to_numpy,
    resolve_device,
    resolve_dtype,
    add_common_args,
    MODEL_VARIANTS,
    ONNX_EXPORT_DTYPE,
)

logger = logging.getLogger("onnx_export")


def export_speaker_encoder(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    dtype: torch.dtype = ONNX_EXPORT_DTYPE,
) -> str:
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant)

    logger.info(f"Loading model: {variant} from {model_path} (fp32 for ONNX export)")
    model = load_tts_model(model_path, device=device, dtype=torch.float32)

    if model.speaker_encoder is None:
        logger.warning(
            f"Variant '{variant}' has no speaker encoder (not a base model). Skipping."
        )
        return None

    speaker_encoder = model.speaker_encoder.to(device).eval()

    B, T, mel_dim = 1, 200, 128
    dummy_mel = torch.randn(B, T, mel_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        ref_output = speaker_encoder(dummy_mel)

    onnx_path = str(out_dir / "speaker_encoder.onnx")
    export_onnx(
        model=speaker_encoder,
        dummy_inputs=(dummy_mel,),
        input_names=["mel"],
        output_names=["speaker_embedding"],
        dynamic_axes={
            "mel": {0: "batch", 1: "time"},
            "speaker_embedding": {0: "batch"},
        },
        onnx_path=onnx_path,
    )

    test_inputs = {"mel": to_numpy(dummy_mel)}
    torch_outputs = {"speaker_embedding": to_numpy(ref_output)}
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=1e-4)
    if ok:
        logger.info("Speaker Encoder ONNX verification PASSED")
    else:
        logger.warning(
            "Speaker Encoder ONNX verification FAILED (outputs differ beyond tolerance)"
        )

    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return onnx_path


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Export Speaker Encoder to ONNX")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model variant (e.g. base-1.7b). Default: export all base variants",
    )
    add_common_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    base_variants = [v for v in MODEL_VARIANTS if v.startswith("base-")]
    variants = [args.variant] if args.variant else base_variants

    for variant in variants:
        if variant not in MODEL_VARIANTS:
            logger.error(f"Unknown variant: {variant}")
            continue
        try:
            path = export_speaker_encoder(
                variant, args.models_dir, args.output_dir, device, dtype
            )
            if path:
                logger.info(f"[{variant}] Speaker Encoder exported: {path}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
