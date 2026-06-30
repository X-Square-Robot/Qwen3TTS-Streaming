#!/usr/bin/env python3
"""
[Step 07] Export Talker backbone ONNX only (layers + codec_head + KV).

Outputs: hidden, logits, present_kv_* — no Code Predictor or codec embedding sum.
For verification vs PyTorch and as the lower half before TalkerUnifiedFused (step 08).
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from talker_unified_modules import build_talker_backbone_module
from utils import (
    setup_logging,
    resolve_model_path,
    ensure_output_dir,
    load_tts_model,
    export_onnx,
    verify_onnx,
    to_numpy,
    resolve_device,
    add_common_args,
    MODEL_VARIANTS,
    ONNX_EXPORT_DTYPE,
)

logger = logging.getLogger("onnx_export")


def _export_talker_backbone_onnx(
    model,
    variant: str,
    output_dir: Path,
    device: str = "cpu",
    opset_version: int = 18,
) -> str:
    backbone, num_layers, hidden_size, num_kv_heads, head_dim = (
        build_talker_backbone_module(model, device=device)
    )

    B, one, S_past = 1, 1, 0
    dummy_embeds = torch.randn(
        B, one, hidden_size, device=device, dtype=ONNX_EXPORT_DTYPE
    )
    position_ids = torch.full((B, 3, one, 1), S_past, device=device, dtype=torch.long)

    past_list = []
    for _ in range(num_layers):
        past_list.append(
            torch.zeros(
                B,
                num_kv_heads,
                S_past,
                head_dim,
                device=device,
                dtype=ONNX_EXPORT_DTYPE,
            )
        )
        past_list.append(
            torch.zeros(
                B,
                num_kv_heads,
                S_past,
                head_dim,
                device=device,
                dtype=ONNX_EXPORT_DTYPE,
            )
        )

    with torch.no_grad():
        out = backbone(dummy_embeds, position_ids, *past_list)

    hidden, logits = out[0], out[1]
    logger.info(
        f"  Backbone shapes: hidden={hidden.shape}, logits={logits.shape}, layers={num_layers}"
    )

    output_names = ["hidden", "logits"]
    for i in range(num_layers):
        output_names.append(f"present_kv_{i}_k")
        output_names.append(f"present_kv_{i}_v")

    input_names = ["input_embeds", "position_ids"]
    for i in range(num_layers):
        input_names.append(f"past_kv_{i}_k")
        input_names.append(f"past_kv_{i}_v")

    dynamic_axes = {
        "input_embeds": {0: "batch", 1: "seq"},
        "position_ids": {0: "batch", 1: "three", 2: "seq"},
        "hidden": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_kv_{i}_k"] = {0: "batch", 2: "S_past"}
        dynamic_axes[f"past_kv_{i}_v"] = {0: "batch", 2: "S_past"}
        dynamic_axes[f"present_kv_{i}_k"] = {0: "batch", 2: "S_total"}
        dynamic_axes[f"present_kv_{i}_v"] = {0: "batch", 2: "S_total"}

    onnx_path = str(output_dir / "talker_backbone.onnx")
    dummy_inputs = (dummy_embeds, position_ids, *past_list)
    export_onnx(
        model=backbone,
        dummy_inputs=dummy_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        onnx_path=onnx_path,
        opset_version=opset_version,
        simplify=True,
    )

    test_inputs = {
        "input_embeds": to_numpy(dummy_embeds),
        "position_ids": position_ids.cpu().numpy(),
    }
    for i, t in enumerate(past_list):
        test_inputs[input_names[2 + i]] = to_numpy(t)
    torch_outputs = {
        output_names[i]: to_numpy(out[i]) for i in range(len(output_names))
    }
    atol = 2e-3 if ONNX_EXPORT_DTYPE == torch.float32 else 1e-1
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=atol, rtol=1e-2)
    if ok:
        logger.info("  Talker backbone ONNX verification PASSED")
    else:
        logger.warning("  Talker backbone ONNX verification had differences")

    return onnx_path


def export_talker_backbone(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
) -> dict:
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant)
    logger.info(f"Loading model: {variant} from {model_path}")
    model = load_tts_model(model_path, device="cpu", dtype=torch.float32)
    onnx_path = _export_talker_backbone_onnx(model, variant, out_dir, device=device)
    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return {"onnx": onnx_path}


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Export Talker backbone (no CP/sum) ONNX"
    )
    parser.add_argument("--variant", type=str, default=None)
    add_common_args(parser)
    args = parser.parse_args()
    device = resolve_device(args.device)
    variants = [args.variant] if args.variant else list(MODEL_VARIANTS.keys())
    for variant in variants:
        if variant not in MODEL_VARIANTS:
            logger.error(f"Unknown variant: {variant}")
            continue
        try:
            results = export_talker_backbone(
                variant, args.models_dir, args.output_dir, device
            )
            for key, path in results.items():
                logger.info(f"[{variant}] Talker backbone {key}: {path}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
