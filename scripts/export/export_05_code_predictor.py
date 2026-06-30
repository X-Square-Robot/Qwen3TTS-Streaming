#!/usr/bin/env python3
"""
Standalone Code Predictor export & verification (optional, not in main pipeline).

NOTE: In production, CP is fused into talker graphs (export_08 / export_09).
Step 05 exports a standalone ONNX for verification only.

The CodePredictorUnrolled class lives in utils.py and is imported here.

This script exists for:
  1. Standalone CP debugging / verification
  2. Exporting a standalone CP ONNX for comparison / profiling
  3. Single-stage fallback export (if the unrolled graph is too large for TRT)

ONNX export skips onnxsim (simplify=False): intermediate verification graph; simplification is slow.

Component: Code Predictor
Architecture: Qwen3-style, 5L, h=1024, GQA(16h/8kv), head_dim=128
              + (num_code_groups-1) codec embeddings + lm_heads
Input:  past_hidden [B, 1, 1024], codec_token_0 [B]
Output: codec_tokens [B, num_code_groups-1]

Applies to: ALL variants
"""

import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    CodePredictorUnrolled,
)

logger = logging.getLogger("onnx_export")


class CodePredictorSingleStage(nn.Module):
    """Single stage of the Code Predictor for fallback export.

    Called N times (N=num_code_groups-1) in a Python loop by the Orchestrator.
    Each call gets the full accumulated sequence + stage-specific lm_head weights.

    See architecture.md §5.6.
    """

    def __init__(self, code_predictor):
        super().__init__()
        self.transformer_layers = code_predictor.model.layers
        self.norm = code_predictor.model.norm
        self.rotary_emb = code_predictor.model.rotary_emb
        self.projection = code_predictor.small_to_mtp_projection
        self.hidden_size = code_predictor.config.hidden_size

    def forward(
        self,
        sequence: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sequence:       [B, S, hidden_size] (S grows from 2 to num_code_groups)
            lm_head_weight: [vocab_size, hidden_size]
            lm_head_bias:   [vocab_size] (zeros if no bias)
        Returns:
            logits:         [B, 1, vocab_size]
        """
        B, S, D = sequence.shape
        device = sequence.device

        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(sequence, position_ids)

        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=device, dtype=sequence.dtype),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        hidden = sequence
        for layer in self.transformer_layers:
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=torch.arange(S, device=device),
                position_embeddings=position_embeddings,
            )
            hidden = layer_out[0]

        hidden = self.norm(hidden)
        logits = F.linear(hidden[:, -1:, :], lm_head_weight, lm_head_bias)
        return logits


def export_code_predictor_unrolled(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    dtype: torch.dtype = ONNX_EXPORT_DTYPE,
) -> str:
    """Export the unrolled Code Predictor (all stages in one graph)."""
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant)

    logger.info(f"Loading model: {variant} from {model_path} (fp32 for ONNX export)")
    model = load_tts_model(model_path, device=device, dtype=torch.float32)

    talker = model.talker
    code_predictor = talker.code_predictor.to(device).eval()
    talker_codec_embedding = talker.model.codec_embedding.to(device).eval()

    wrapper = (
        CodePredictorUnrolled(code_predictor, talker_codec_embedding).to(device).eval()
    )

    hidden_size = talker.config.hidden_size
    B = 1
    dummy_hidden = torch.randn(B, 1, hidden_size, device=device, dtype=torch.float32)
    dummy_token = torch.randint(0, 3072, (B,), device=device)

    with torch.no_grad():
        ref_output = wrapper(dummy_hidden, dummy_token)

    logger.info(f"Unrolled Code Predictor output shape: {ref_output.shape}")

    onnx_path = str(out_dir / "code_predictor_unrolled.onnx")
    export_onnx(
        model=wrapper,
        dummy_inputs=(dummy_hidden, dummy_token),
        input_names=["past_hidden", "codec_token_0"],
        output_names=["codec_tokens"],
        dynamic_axes={
            "past_hidden": {0: "batch"},
            "codec_token_0": {0: "batch"},
            "codec_tokens": {0: "batch"},
        },
        onnx_path=onnx_path,
        simplify=False,
    )

    import onnx

    onnx_model = onnx.load(onnx_path)
    n_inits = len(onnx_model.graph.initializer)
    n_nodes = len(onnx_model.graph.node)
    logger.info(f"Weight sharing check: initializers={n_inits}, nodes={n_nodes}")
    if n_inits > 500:
        logger.warning(
            f"Initializer count ({n_inits}) seems high - Transformer weights may not be shared. "
            f"Expected ~60 (5L weights + {wrapper.num_stages} embeddings + {wrapper.num_stages} lm_heads)."
        )

    test_inputs = {
        "past_hidden": to_numpy(dummy_hidden),
        "codec_token_0": to_numpy(dummy_token),
    }
    torch_outputs = {"codec_tokens": to_numpy(ref_output)}
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=0)
    if ok:
        logger.info("Unrolled Code Predictor ONNX verification PASSED")
    else:
        logger.warning(
            "Unrolled Code Predictor ONNX verification FAILED (expected for argmax-based model)"
        )

    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return onnx_path


def export_code_predictor_single_stage(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    dtype: torch.dtype = ONNX_EXPORT_DTYPE,
) -> str:
    """Export the single-stage Code Predictor (fallback)."""
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant)

    logger.info(f"Loading model: {variant} from {model_path} (fp32 for ONNX export)")
    model = load_tts_model(model_path, device=device, dtype=torch.float32)

    talker = model.talker
    code_predictor = talker.code_predictor.to(device).eval()

    wrapper = CodePredictorSingleStage(code_predictor).to(device).eval()

    hidden_size = code_predictor.config.hidden_size
    vocab_size = code_predictor.config.vocab_size
    B, S = 1, 5
    dummy_seq = torch.randn(B, S, hidden_size, device=device, dtype=torch.float32)
    dummy_weight = torch.randn(
        vocab_size, hidden_size, device=device, dtype=torch.float32
    )
    dummy_bias = torch.zeros(vocab_size, device=device, dtype=torch.float32)

    with torch.no_grad():
        ref_output = wrapper(dummy_seq, dummy_weight, dummy_bias)

    logger.info(f"Single-stage Code Predictor output shape: {ref_output.shape}")

    onnx_path = str(out_dir / "code_predictor_single_stage.onnx")
    export_onnx(
        model=wrapper,
        dummy_inputs=(dummy_seq, dummy_weight, dummy_bias),
        input_names=["sequence", "lm_head_weight", "lm_head_bias"],
        output_names=["logits"],
        dynamic_axes={
            "sequence": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch"},
        },
        onnx_path=onnx_path,
        simplify=False,
    )

    test_inputs = {
        "sequence": to_numpy(dummy_seq),
        "lm_head_weight": to_numpy(dummy_weight),
        "lm_head_bias": to_numpy(dummy_bias),
    }
    torch_outputs = {"logits": to_numpy(ref_output)}
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=1e-4)
    if ok:
        logger.info("Single-stage Code Predictor ONNX verification PASSED")
    else:
        logger.warning("Single-stage Code Predictor ONNX verification FAILED")

    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return onnx_path


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Export Code Predictor to ONNX")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model variant. Default: export all variants",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["unrolled", "single_stage", "both"],
        help="Export mode: unrolled (primary), single_stage (fallback), or both",
    )
    add_common_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    variants = [args.variant] if args.variant else list(MODEL_VARIANTS.keys())

    for variant in variants:
        if variant not in MODEL_VARIANTS:
            logger.error(f"Unknown variant: {variant}")
            continue
        try:
            if args.mode in ("unrolled", "both"):
                path = export_code_predictor_unrolled(
                    variant, args.models_dir, args.output_dir, device, dtype
                )
                logger.info(f"[{variant}] Unrolled Code Predictor exported: {path}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped unrolled: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Unrolled export failed: {e}", exc_info=True)

        try:
            if args.mode in ("single_stage", "both"):
                path = export_code_predictor_single_stage(
                    variant, args.models_dir, args.output_dir, device, dtype
                )
                logger.info(f"[{variant}] Single-stage Code Predictor exported: {path}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped single-stage: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Single-stage export failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
