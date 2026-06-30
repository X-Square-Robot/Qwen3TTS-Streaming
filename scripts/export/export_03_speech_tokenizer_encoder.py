#!/usr/bin/env python3
"""
[Step 02] Export Speech Tokenizer Encoder (MimiModel encoder-only) to ONNX.

Component: Speech Tokenizer Encoder
Architecture: MimiModel encoder (with VQ)
Input:  waveform [B, 1, samples]  (24kHz mono)
Output: audio_codes [B, num_quantizers, T_codes]
Engine: ONNX Runtime
Usage: Called once per request (ICL mode only, for base models)

The encoder is shared across all model variants (comes from the tokenizer checkpoint).
"""

import argparse
import logging

import torch

from utils import (
    setup_logging,
    resolve_tokenizer_path,
    ensure_output_dir,
    load_speech_tokenizer,
    export_onnx,
    verify_onnx,
    to_numpy,
    resolve_device,
    resolve_dtype,
    add_common_args,
    ONNX_EXPORT_DTYPE,
)

logger = logging.getLogger("onnx_export")


def _quantize_no_cdist(self, hidden_states):
    """Drop-in replacement for MimiEuclideanCodebook.quantize that avoids torch.cdist.

    torch.cdist's ONNX symbolic fails when dim sizes are not statically known
    after JIT trace (AssertionError: row_size_x1 is not None).  The equivalent
    squared-L2 via matmul is fully traceable.
    """
    x = hidden_states.float()  # [N, D]
    e = self.embed.to(x.device).float()  # [C, D]
    # ||x - e||^2 = ||x||^2 - 2 x·eᵀ + ||e||^2
    dists = (
        (x * x).sum(dim=-1, keepdim=True)
        - 2.0 * torch.mm(x, e.t())
        + (e * e).sum(dim=-1).unsqueeze(0)
    )
    return dists.argmin(dim=-1)


def _patch_cdist_in_encoder(encoder: torch.nn.Module):
    """Replace quantize() on all MimiEuclideanCodebook instances inside encoder."""
    from types import MethodType

    for mod in encoder.modules():
        cls_name = type(mod).__name__
        if cls_name == "MimiEuclideanCodebook":
            mod.quantize = MethodType(_quantize_no_cdist, mod)


class SpeechTokenizerEncoderWrapper(torch.nn.Module):
    """Wraps the MimiModel encoder for clean ONNX export.

    The original encoder.encode() returns a complex output structure.
    This wrapper exposes a simple forward(waveform) -> audio_codes interface.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        _patch_cdist_in_encoder(self.encoder)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_values: [B, 1, samples] - mono waveform at 24kHz
        Returns:
            audio_codes: [B, 16, T_codes] — num_quantizers=16 so only first 16 RVQ layers run
            (saves ~50% compute vs full 32; matches code2wav/encoder_valid_num_quantizers).
        """
        encoded = self.encoder.encode(
            input_values=input_values, return_dict=True, num_quantizers=16
        )
        return encoded.audio_codes


def export_speech_tokenizer_encoder(
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    dtype: torch.dtype = ONNX_EXPORT_DTYPE,
) -> str:
    tokenizer_path = resolve_tokenizer_path(models_dir)
    out_dir = ensure_output_dir(output_dir, "tokenizer")

    logger.info(
        f"Loading speech tokenizer from {tokenizer_path} (fp32 for ONNX export)"
    )
    tokenizer_model = load_speech_tokenizer(
        tokenizer_path, device=device, dtype=torch.float32
    )

    encoder = tokenizer_model.encoder.to(device).eval()
    wrapper = SpeechTokenizerEncoderWrapper(encoder).to(device).eval()

    B = 1
    sample_rate = 24000
    duration_sec = 3.0
    samples = int(sample_rate * duration_sec)
    dummy_wav = torch.randn(B, 1, samples, device=device, dtype=torch.float32)

    with torch.no_grad():
        ref_output = wrapper(dummy_wav)

    logger.info(f"Encoder output shape: {ref_output.shape}")

    onnx_path = str(out_dir / "speech_tokenizer_encoder.onnx")
    export_onnx(
        model=wrapper,
        dummy_inputs=(dummy_wav,),
        input_names=["waveform"],
        output_names=["audio_codes"],
        dynamic_axes={
            "waveform": {0: "batch", 2: "samples"},
            "audio_codes": {0: "batch", 2: "time"},
        },
        onnx_path=onnx_path,
    )

    cpu_wav = dummy_wav.cpu()
    for mod in wrapper.modules():
        if hasattr(mod, "_embed"):
            mod._embed = None
    cpu_wrapper = wrapper.cpu().eval()
    with torch.no_grad():
        cpu_ref = cpu_wrapper(cpu_wav)
    wrapper.to(device)

    test_inputs = {"waveform": to_numpy(cpu_wav)}
    torch_outputs = {"audio_codes": to_numpy(cpu_ref)}
    ok = verify_onnx(onnx_path, test_inputs, torch_outputs, atol=0)
    if ok:
        logger.info("Speech Tokenizer Encoder ONNX verification PASSED")
    else:
        logger.warning("Speech Tokenizer Encoder ONNX verification FAILED")

    del tokenizer_model
    if device != "cpu":
        torch.cuda.empty_cache()
    return onnx_path


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Export Speech Tokenizer Encoder to ONNX"
    )
    add_common_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    path = export_speech_tokenizer_encoder(
        args.models_dir, args.output_dir, device, dtype
    )
    logger.info(f"Speech Tokenizer Encoder exported: {path}")


if __name__ == "__main__":
    main()
