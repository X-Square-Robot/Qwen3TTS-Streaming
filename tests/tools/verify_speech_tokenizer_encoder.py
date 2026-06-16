#!/usr/bin/env python3
"""
Verify speech_tokenizer_encoder ONNX consistency.

Checks:
  1. ONNX vs PyTorch (num_quantizers=16) — output match on multiple input lengths
  2. num_quantizers=16 vs full 32 sliced — first 16 codebooks identical (proves early-stop correctness)
"""

import argparse
import logging

import numpy as np
import torch

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from qwen3tts_tools.common import bootstrap_project_imports

bootstrap_project_imports("scripts_export", "third_party_qwen")

from utils import (
    setup_logging,
    resolve_tokenizer_path,
    load_speech_tokenizer,
    to_numpy,
)

logger = logging.getLogger("verify_stoken")


def _patch_cdist_in_encoder(encoder: torch.nn.Module):
    from types import MethodType

    def _quantize_no_cdist(self, hidden_states):
        x = hidden_states.float()
        e = self.embed.to(x.device).float()
        dists = (
            (x * x).sum(dim=-1, keepdim=True)
            - 2.0 * torch.mm(x, e.t())
            + (e * e).sum(dim=-1).unsqueeze(0)
        )
        return dists.argmin(dim=-1)

    for mod in encoder.modules():
        if type(mod).__name__ == "MimiEuclideanCodebook":
            mod.quantize = MethodType(_quantize_no_cdist, mod)


def run_consistency_checks(
    onnx_path: str,
    tokenizer_path: str,
    device: str = "cpu",
    num_tests: int = 5,
) -> bool:
    """Run ONNX vs PyTorch and 16 vs 32 consistency checks. Returns True if all pass."""
    import onnxruntime as ort

    tokenizer_model = load_speech_tokenizer(tokenizer_path, device=device, dtype=torch.float32)
    encoder = tokenizer_model.encoder.to(device).eval()
    _patch_cdist_in_encoder(encoder)

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
        if device == "cpu"
        else ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    rng = np.random.default_rng(42)
    all_ok = True

    # Test lengths: 0.5s, 1s, 2s, 3s, 4s @ 24kHz
    sample_rate = 24000
    durations = [0.5, 1.0, 2.0, 3.0, 4.0][:num_tests]

    logger.info("=" * 60)
    logger.info("  Check 1: ONNX vs PyTorch (num_quantizers=16)")
    logger.info("=" * 60)

    for i, dur in enumerate(durations):
        n = int(sample_rate * dur)
        wav_np = rng.standard_normal((1, 1, n), dtype=np.float32) * 0.3
        wav_pt = torch.from_numpy(wav_np).to(device)

        with torch.no_grad():
            out_pt = encoder.encode(
                input_values=wav_pt, return_dict=True, num_quantizers=16
            ).audio_codes
        out_pt_np = to_numpy(out_pt)

        out_ort = sess.run(None, {"waveform": wav_np})[0]

        match = np.array_equal(out_pt_np, out_ort)
        max_diff = np.abs(out_pt_np.astype(np.float64) - out_ort.astype(np.float64)).max()
        logger.info(
            f"  [{i+1}/{len(durations)}] dur={dur}s, shape={out_ort.shape} "
            f"→ match={match}, max_diff={max_diff}"
        )
        if not match:
            all_ok = False

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Check 2: num_quantizers=16 vs full 32 (first 16 must match)")
    logger.info("=" * 60)

    wav_pt = torch.from_numpy(rng.standard_normal((1, 1, sample_rate * 2), dtype=np.float32) * 0.3).to(
        device
    )
    with torch.no_grad():
        out_16 = encoder.encode(input_values=wav_pt, return_dict=True, num_quantizers=16).audio_codes
        out_32 = encoder.encode(input_values=wav_pt, return_dict=True, num_quantizers=32).audio_codes
    out_32_slice = out_32[:, :16, :]
    match_16_32 = torch.equal(out_16, out_32_slice)
    logger.info(f"  out_16 shape: {out_16.shape}, out_32[:,:16,:] shape: {out_32_slice.shape}")
    logger.info(f"  num_quantizers=16 matches first 16 of full 32: {match_16_32}")
    if not match_16_32:
        diff_count = (out_16 != out_32_slice).sum().item()
        logger.warning(f"  Mismatch count: {diff_count}")
        all_ok = False

    del tokenizer_model
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_ok


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Verify speech_tokenizer_encoder ONNX consistency")
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to speech_tokenizer_encoder.onnx (default: workspace/exported/tokenizer/...)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Models directory for tokenizer (default: workspace/models)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-tests", type=int, default=5)
    args = parser.parse_args()

    tokenizer_path = resolve_tokenizer_path(args.models_dir)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        logger.error("Tokenizer not found at %s", tokenizer_path)
        sys.exit(1)

    onnx_path = args.onnx
    if not onnx_path:
        out_dir = REPO_ROOT / "workspace" / "exported" / "tokenizer"
        onnx_path = str(out_dir / "speech_tokenizer_encoder.onnx")
    if not Path(onnx_path).exists():
        logger.error("ONNX not found: %s", onnx_path)
        sys.exit(1)

    ok = run_consistency_checks(
        onnx_path=onnx_path,
        tokenizer_path=tokenizer_path,
        device=args.device,
        num_tests=args.num_tests,
    )

    if ok:
        logger.info("")
        logger.info("=== ALL CONSISTENCY CHECKS PASSED ===")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("=== SOME CHECKS FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
