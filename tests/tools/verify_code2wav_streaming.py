#!/usr/bin/env python3
"""
Verify code2wav streaming vs prototype consistency.

1) Same codes → one-shot decoder.forward(codes) vs streaming wrapper (chunk_T=4).
   Ensures stateful streaming is mathematically equivalent to non-streaming.
2) Optionally: run short TTS to get real codec sequence, then decode with both
   prototype tokenizer.decode() and our streaming wrapper; compare and save WAVs.

Usage (from repo root, conda activate qwen3-tts):
  # Fixture test: same random codes -> one-shot decoder vs streaming (chunk_T=4); report max diff.
  python tests/tools/verify_code2wav_streaming.py --models-dir workspace/models
  # E2E: TTS -> codes -> decode with one-shot, streaming, and prototype tokenizer.decode; save WAVs for listening.
  python tests/tools/verify_code2wav_streaming.py --models-dir workspace/models --e2e --text "你好世界" --max-steps 40
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

# Prefer export utils (has load_speech_tokenizer + decoder RoPE patch).
bootstrap_project_imports("scripts_export", "scripts_python", "scripts", "third_party_qwen")

from utils import (
    setup_logging,
    resolve_tokenizer_path,
    resolve_model_path,
    load_speech_tokenizer,
    has_model_weights,
    resolve_device,
)
try:
    from utils import _patch_decoder_rotary_from_weights
except ImportError:
    _patch_decoder_rotary_from_weights = None

from code2wav_streaming import (
    Code2WavStreamingWrapper,
    create_initial_states,
    CHUNK_T,
    SAMPLES_PER_CHUNK,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_code2wav_streaming")

SAMPLES_PER_FRAME = 1920


def decode_one_shot(decoder: torch.nn.Module, codes: torch.Tensor) -> torch.Tensor:
    """Decoder forward on full codes [1, 16, T]. Returns wav [1, T*1920] (squeezed channel)."""
    with torch.no_grad():
        wav = decoder(codes)
    # decoder returns [B, 1, samples]; squeeze channel
    if wav.dim() == 3:
        wav = wav.squeeze(1)
    return wav


def decode_streaming(
    wrapper: Code2WavStreamingWrapper,
    codes: torch.Tensor,
    device: torch.device,
    codec_pad_id: int = 0,
    dtype: torch.dtype = torch.float32,
    wav_ref: torch.Tensor = None,
) -> torch.Tensor:
    """
    Decode codes [1, 16, T] using streaming wrapper (chunk_T=4).
    Last chunk padded to 4 with codec_pad_id if T % 4 != 0; output trimmed to T*1920.
    """
    B, Q, T = codes.shape
    assert B == 1 and Q == 16
    states = create_initial_states(wrapper.decoder, device, dtype, batch_size=B)
    wav_chunks = []
    frame_index = 0
    i = 0
    chunk_idx = 0
    while i < T:
        end = min(i + CHUNK_T, T)
        chunk = codes[:, :, i:end]
        n = chunk.shape[-1]
        if n < CHUNK_T:
            pad = torch.full(
                (B, Q, CHUNK_T - n), codec_pad_id, device=device, dtype=torch.long
            )
            chunk = torch.cat([chunk, pad], dim=-1)
        cache_position = torch.arange(
            frame_index, frame_index + CHUNK_T, device=device, dtype=torch.float32
        )
        with torch.no_grad():
            out = wrapper(chunk, cache_position, *states)
        wav_chunk = out[0]
        if wav_chunk.dim() == 3:
            wav_chunk = wav_chunk.squeeze(1)
        actual_samples = n * SAMPLES_PER_FRAME
        wav_chunks.append(wav_chunk[:, :actual_samples])
        if wav_ref is not None:
            ref_slice = wav_ref[:, i * SAMPLES_PER_FRAME : end * SAMPLES_PER_FRAME]
            stream_slice = wav_chunk[:, :actual_samples]
            diff = (ref_slice.float() - stream_slice.float()).abs()
            logger.info(
                "  chunk_%d (frames %d-%d): max_diff=%.2e mean_diff=%.2e",
                chunk_idx, i, end - 1, diff.max().item(), diff.mean().item(),
            )
        states = list(out[1:])
        frame_index += CHUNK_T
        i = end
        chunk_idx += 1
    return torch.cat(wav_chunks, dim=-1)


def compare_wav(
    wav_ref: np.ndarray, wav_stream: np.ndarray, name: str = "streaming"
) -> bool:
    """Compare two 1D float arrays; report max abs diff and pass/fail. Return True if pass."""
    if wav_ref.shape != wav_stream.shape:
        logger.error(
            "%s: shape mismatch ref %s vs %s %s",
            name, wav_ref.shape, name, wav_stream.shape,
        )
        return False
    diff = np.abs(wav_ref.astype(np.float64) - wav_stream.astype(np.float64))
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    ok = max_diff < 0.01 and mean_diff < 0.001
    logger.info(
        "%s vs ref: max_abs_diff=%.2e mean_abs_diff=%.2e -> %s",
        name, max_diff, mean_diff, "PASS" if ok else "FAIL",
    )
    return ok


def _trim_at_eos(codes: np.ndarray, eos_step: int) -> np.ndarray:
    """Trim codec sequence to frames before EOS (exclude EOS frame)."""
    if eos_step >= 0:
        return codes[:eos_step]
    return codes


def get_codes_from_tts(
    model_path: Path,
    text: str,
    max_steps: int,
    device: torch.device,
    variant: str = "design-1.7b",
) -> tuple:
    """Run manual decode loop to get codec sequence [T, 16].
    Returns (codes np.ndarray, sample_rate, eos_step, gold_wav_np_or_None).
    gold_wav is decoded by the TTS model's own speech_tokenizer as ground truth."""
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper
    from verify_prototype_parity import run_manual_decode_loop
    from official_prefill import build_prefill_like_official

    wrapper = TTSModelWrapper.from_pretrained(
        str(model_path), device_map=str(device), dtype=torch.float32
    )
    model = wrapper.model
    processor = wrapper.processor
    non_streaming = "design" in variant.lower()
    codec_eos_id = int(model.config.talker_config.codec_eos_token_id)

    # Must wrap text with assistant format, matching official generate_voice_design / generate_custom_voice
    formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    tok_out = processor(text=formatted_text, return_tensors="pt")
    input_ids = tok_out["input_ids"].to(device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    prefill_embeds, trailing_list = build_prefill_like_official(
        model, input_ids, "auto", "", device, non_streaming_mode=non_streaming
    )
    tts_pad_token_id = getattr(model.config, "tts_pad_token_id", 0)
    with torch.no_grad():
        pad_id = torch.tensor([[tts_pad_token_id]], device=device, dtype=torch.long)
        pad_embed = model.talker.text_projection(
            model.talker.model.text_embedding(pad_id)
        )
    codes_manual, _, eos_step = run_manual_decode_loop(
        model, prefill_embeds, trailing_list, pad_embed,
        max_steps, codec_eos_id, device,
    )

    # Produce gold-standard WAV using the TTS model's own speech_tokenizer
    gold_wav_np = None
    speech_tok = getattr(model, "speech_tokenizer", None)
    if speech_tok is None:
        speech_tok = getattr(wrapper, "speech_tokenizer", None)
    if speech_tok is not None:
        trimmed = codes_manual[:eos_step] if eos_step >= 0 else codes_manual
        trimmed = np.clip(trimmed, 0, 2047).astype(np.int64)
        codes_for_dec = torch.from_numpy(trimmed).long().to(device)
        try:
            wavs_gold, sr_gold = speech_tok.decode([{"audio_codes": codes_for_dec}])
            gold_wav_np = wavs_gold[0] if isinstance(wavs_gold[0], np.ndarray) else wavs_gold[0].cpu().numpy()
            logger.info("Gold decode (TTS model's speech_tokenizer): %d samples (%.2f s)",
                        len(gold_wav_np), len(gold_wav_np) / sr_gold)
        except Exception as e:
            logger.warning("Gold decode failed: %s", e)

    return codes_manual, 24000, eos_step, gold_wav_np


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Verify code2wav streaming parity with prototype"
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Models root (default: workspace/models)",
    )
    parser.add_argument(
        "--variant",
        default="design-1.7b",
        help="Model variant for E2E (default: design-1.7b)",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run E2E: TTS -> codes -> decode with prototype and streaming, save WAVs",
    )
    parser.add_argument("--text", default="你好世界。", help="Input text for E2E")
    parser.add_argument("--max-steps", type=int, default=40, help="Max decode steps for E2E")
    parser.add_argument("--out-dir", default=None, help="Output dir for WAVs (default: workspace/audio_compare)")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=24,
        help="Number of codec frames for fixture test (default 24, multiple of 4)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    models_dir = Path(args.models_dir) if args.models_dir else REPO_ROOT / "workspace" / "models"
    tokenizer_path = resolve_tokenizer_path(str(models_dir))

    logger.info("Loading speech tokenizer from %s ...", tokenizer_path)
    tokenizer_model = load_speech_tokenizer(
        tokenizer_path, device=str(device), dtype=torch.float32
    )
    decoder = tokenizer_model.decoder.to(device).eval()

    if _patch_decoder_rotary_from_weights is not None:
        _patch_decoder_rotary_from_weights(decoder)
    wrapper = Code2WavStreamingWrapper(decoder).to(device).eval()

    # Codec pad id from tokenizer/decoder config (Config object has getattr, not .get)
    _cfg = getattr(decoder, "config", None) or getattr(tokenizer_model, "config", None)
    codec_pad_id = getattr(_cfg, "pad_token_id", 0) if _cfg is not None else 0
    if not isinstance(codec_pad_id, int):
        codec_pad_id = 0
    logger.info("Using codec_pad_id=%s", codec_pad_id)

    if args.e2e:
        model_path = resolve_model_path(args.variant, str(models_dir))
        if not has_model_weights(model_path):
            logger.error("No weights for variant %s at %s", args.variant, model_path)
            sys.exit(1)
        logger.info("E2E: getting codes from TTS (variant=%s, max_steps=%s) ...", args.variant, args.max_steps)
        codes_np, sr, eos_step, gold_wav_np = get_codes_from_tts(
            model_path, args.text, args.max_steps, device, args.variant
        )
        # Trim at EOS so we only decode real speech frames (post-EOS steps produce noise)
        codes_np = _trim_at_eos(codes_np, eos_step)
        if codes_np.size == 0:
            logger.error("No frames after trim (eos_step=%s). Increase --max-steps or check TTS.", eos_step)
            sys.exit(1)
        T = codes_np.shape[0]
        logger.info("Got codes shape %s (%d frames after trim at EOS step %s)", codes_np.shape, T, eos_step)
        # Clamp to valid codebook indices [0, codebook_size-1] as in generate_audio_compare
        codebook_size = 2048
        codes_np = np.clip(codes_np, 0, codebook_size - 1).astype(np.int64)
        codes_t = torch.from_numpy(codes_np).long().to(device)
        if codes_t.dim() == 2:
            codes_t = codes_t.unsqueeze(0).permute(0, 2, 1)
        else:
            codes_t = codes_t.unsqueeze(0)
        # codes_t [1, 16, T]
    else:
        # Fixture: deterministic codes for parity test
        T = args.num_frames
        if T % CHUNK_T != 0:
            T = (T // CHUNK_T + 1) * CHUNK_T
        torch.manual_seed(42)
        codes_t = torch.randint(0, 2048, (1, 16, T), device=device, dtype=torch.long)
        logger.info("Fixture codes [1, 16, %d]", T)

    # Reference: one-shot decoder
    logger.info("Reference: one-shot decoder.forward(codes) ...")
    wav_ref = decode_one_shot(decoder, codes_t)
    wav_ref_np = wav_ref[0].float().cpu().numpy()

    # Streaming: chunk_T=4
    logger.info("Streaming: Code2WavStreamingWrapper chunk_T=4 ...")
    wav_stream = decode_streaming(
        wrapper, codes_t, device, codec_pad_id=codec_pad_id, dtype=torch.float32,
        wav_ref=wav_ref,
    )
    wav_stream_np = wav_stream[0].float().cpu().numpy()

    # Compare
    ok = compare_wav(wav_ref_np, wav_stream_np, "streaming")
    if not ok:
        logger.warning("Streaming vs one-shot decoder: FAIL (check implementation)")

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "workspace" / "audio_compare"
    if args.e2e:
        out_dir.mkdir(parents=True, exist_ok=True)

        def _write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
            if _HAS_SOUNDFILE:
                sf.write(str(path), wav, sr)
            else:
                import wave
                wav_int16 = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
                with wave.open(str(path), "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(sr)
                    f.writeframes(wav_int16.tobytes())

        # Diagnostic: audio statistics
        logger.info("wav_ref stats: min=%.4f max=%.4f mean=%.6f std=%.6f",
                     wav_ref_np.min(), wav_ref_np.max(), wav_ref_np.mean(), wav_ref_np.std())

        _write_wav(out_dir / "code2wav_ref_one_shot.wav", wav_ref_np, 24000)
        _write_wav(out_dir / "code2wav_streaming.wav", wav_stream_np, 24000)
        logger.info("Saved code2wav_ref_one_shot.wav and code2wav_streaming.wav in %s", out_dir)

        # Gold reference from TTS model's own speech_tokenizer (known-good decode path)
        if gold_wav_np is not None:
            _write_wav(out_dir / "code2wav_gold_tts_tokenizer.wav", gold_wav_np, 24000)
            logger.info("Saved code2wav_gold_tts_tokenizer.wav (TTS model's own speech_tokenizer)")
            logger.info("gold_wav stats: min=%.4f max=%.4f mean=%.6f std=%.6f",
                         gold_wav_np.min(), gold_wav_np.max(), gold_wav_np.mean(), gold_wav_np.std())
            min_len = min(len(wav_ref_np), len(gold_wav_np))
            diff = np.abs(wav_ref_np[:min_len].astype(np.float64) - gold_wav_np[:min_len].astype(np.float64))
            logger.info("one-shot vs gold: max_diff=%.4e mean_diff=%.4e (len ref=%d gold=%d)",
                         diff.max(), diff.mean(), len(wav_ref_np), len(gold_wav_np))

        # Also decode with prototype tokenizer_model.decode() for comparison
        out = tokenizer_model.decode(codes_t.permute(0, 2, 1), return_dict=False)
        wavs_proto = out[0]
        sr_proto = getattr(tokenizer_model, "output_sample_rate", 24000)
        wav_proto_np = wavs_proto[0] if isinstance(wavs_proto[0], np.ndarray) else wavs_proto[0].detach().cpu().numpy()
        _write_wav(out_dir / "code2wav_prototype_decode.wav", wav_proto_np, int(sr_proto))
        logger.info("Saved code2wav_prototype_decode.wav (tokenizer_model.decode)")
        compare_wav(wav_ref_np, wav_proto_np[: wav_ref_np.shape[0]], "prototype_decode")

    logger.info("Done. Streaming vs one-shot: %s", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
