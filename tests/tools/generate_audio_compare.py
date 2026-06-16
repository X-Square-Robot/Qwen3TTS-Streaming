#!/usr/bin/env python3
"""
Three-way audio quality comparison: prototype (official high-level API) vs manual PyTorch
decode loop vs ORT (talker_unified.onnx) decode loop. All three produce codec tokens,
then decode to WAV via the same PyTorch speech_tokenizer.decode() for fair comparison.

For **listenability only** (official vs deployed fused path), prefer:
  `tests/tools/compare_official_vs_triton_audio.py`
which writes `proto.wav` + `triton.wav` only.

This script adds manual/ORT paths and optional Triton (`trt_bf16.wav`); Triton uses the
same orchestrator as production (fused engine when `talker_code2wav_fused` is deployed).

Key: VoiceDesign uses non_streaming_mode=True by default, which folds ALL text tokens
into the prefill. The official _build_assistant_text format also includes a trailing
`\\n<|im_start|>assistant\\n` suffix. Both must be replicated exactly in manual/ORT paths.

Usage (host, conda activate qwen3-tts):
  python tests/tools/generate_audio_compare.py --variant design-1.7b
  python tests/tools/generate_audio_compare.py --variant design-1.7b --text "你好，世界" --max-steps 300
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports(
    "repo",
    "scripts_python",
    "scripts",
    "scripts_export",
    "third_party_qwen",
    "tests_integration",
)
from tests.support.triton_streaming import build_variant_request_payload, infer_stream

from utils import (
    setup_logging,
    resolve_model_path,
    load_tts_model,
    resolve_device,
    DEFAULT_OUTPUT_DIR,
    has_model_weights,
)
from official_prefill import build_prefill_like_official
from verify_prototype_parity import run_manual_decode_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audio_compare")

DEFAULT_TEXT = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"

OFFICIAL_ASSISTANT_FMT = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def _load_talker_dims(model_dir: Path):
    cfg_path = model_dir / "weights" / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            c = json.load(f)
        H = int(c.get("talker_hidden_size", 2048))
        n_heads = int(c.get("talker_num_heads", 16))
        num_kv_heads = int(c.get("talker_num_kv_heads", 8))
        num_layers = int(c.get("talker_num_layers", 28))
        head_dim = H // n_heads
        return H, num_kv_heads, head_dim, num_layers
    return 2048, 8, 128, 28



def run_ort_decode_loop(
    session,
    output_names,
    inputs_embeds,
    trailing_text_hidden,
    pad_embed,
    seq_len,
    num_layers,
    num_kv_heads,
    head_dim,
    max_steps,
    codec_eos_id,
    codec_vocab_size=3072,
    logits_topk=50,
    cp_num_stages=15,
):
    """Run ORT talker_unified prefill + decode, return (codes [T, 16], eos_step)."""
    B, S = 1, seq_len
    position_ids_prefill = np.arange(0, S, dtype=np.int64)
    position_ids_prefill = np.broadcast_to(
        position_ids_prefill.reshape(1, 1, -1, 1), (B, 3, S, 1)
    )

    token_counts = np.zeros((B, codec_vocab_size), dtype=np.float32)
    gumbel_noise = np.zeros((B, logits_topk), dtype=np.float32)
    cp_gumbel_noise = np.zeros((B, cp_num_stages, logits_topk), dtype=np.float32)
    temperature = np.zeros((B, 1), dtype=np.float32)
    penalty = np.ones((B, 1), dtype=np.float32)

    feed = {
        "input_embeds": inputs_embeds,
        "position_ids": position_ids_prefill,
        "token_counts": token_counts,
        "gumbel_noise": gumbel_noise,
        "cp_gumbel_noise": cp_gumbel_noise,
        "temperature": temperature,
        "penalty": penalty,
    }
    for i in range(num_layers):
        feed[f"past_kv_{i}_k"] = np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        feed[f"past_kv_{i}_v"] = np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float32
        )
    outs = session.run(output_names, feed)
    out_map = dict(zip(output_names, outs))
    codec_sum = out_map["codec_sum"]
    full_codec = out_map["full_codec"]
    token_counts = out_map.get("updated_token_counts", token_counts).copy()
    past_kv = []
    for i in range(num_layers):
        past_kv.append(out_map[f"present_kv_{i}_k"].copy())
        past_kv.append(out_map[f"present_kv_{i}_v"].copy())
    n_trailing = (
        trailing_text_hidden.shape[0] if trailing_text_hidden is not None else 0
    )
    text_add_0 = (
        trailing_text_hidden[0]
        if (trailing_text_hidden is not None and n_trailing > 0)
        else pad_embed
    )
    if text_add_0 is not None:
        if text_add_0.ndim == 2:
            text_add_0 = text_add_0.reshape(1, 1, -1)
        current_codec_sum = (
            codec_sum.astype(np.float64) + text_add_0.astype(np.float64)
        ).astype(np.float32)
    else:
        current_codec_sum = codec_sum.copy()
    current_pos = S
    all_codes = [full_codec[0].astype(np.int64).copy()]
    eos_step = -1
    if full_codec[0, 0] == codec_eos_id:
        eos_step = 0
    for step in range(max_steps - 1):
        pos_step = np.full((B, 3, 1, 1), current_pos, dtype=np.int64)
        dec_feed = {
            "input_embeds": current_codec_sum,
            "position_ids": pos_step,
            "token_counts": token_counts,
            "gumbel_noise": gumbel_noise,
            "cp_gumbel_noise": cp_gumbel_noise,
            "temperature": temperature,
            "penalty": penalty,
        }
        for i in range(num_layers):
            dec_feed[f"past_kv_{i}_k"] = past_kv[2 * i]
            dec_feed[f"past_kv_{i}_v"] = past_kv[2 * i + 1]
        dec_outs = session.run(output_names, dec_feed)
        dec_map = dict(zip(output_names, dec_outs))
        codec_sum_step = dec_map["codec_sum"]
        token_counts = dec_map.get("updated_token_counts", token_counts).copy()
        next_text = (
            trailing_text_hidden[step + 1]
            if (
                trailing_text_hidden is not None
                and step + 1 < n_trailing
            )
            else pad_embed
        )
        if next_text is not None:
            if next_text.ndim == 2:
                next_text = next_text.reshape(1, 1, -1)
            current_codec_sum = (
                codec_sum_step.astype(np.float64)
                + next_text.astype(np.float64)
            ).astype(np.float32)
        else:
            current_codec_sum = codec_sum_step
        full_codec_step = dec_map["full_codec"]
        for i in range(num_layers):
            past_kv[2 * i] = dec_map[f"present_kv_{i}_k"].copy()
            past_kv[2 * i + 1] = dec_map[f"present_kv_{i}_v"].copy()
        current_pos += 1
        row = full_codec_step[0].astype(np.int64)
        all_codes.append(row)
        if eos_step < 0 and row[0] == codec_eos_id:
            eos_step = step + 1
        if (step + 1) % 20 == 0:
            logger.info("  ORT step %d: codec_0=%d", step + 1, row[0])
    codes = np.stack(all_codes, axis=0)
    return codes, eos_step


def decode_codes_to_wav(model, codes, device, codebook_size=2048):
    """codes: [T, 16] numpy. Returns (wav np.ndarray, sample_rate int)."""
    if hasattr(model, "speech_tokenizer") and model.speech_tokenizer is not None:
        tokenizer = model.speech_tokenizer
    else:
        raise RuntimeError(
            "Model has no speech_tokenizer; cannot decode codec to WAV. "
            "Load from a full TTS model dir that includes speech_tokenizer/."
        )
    if isinstance(codes, np.ndarray):
        codes = np.clip(codes, 0, codebook_size - 1).astype(np.int64)
        codes_t = torch.from_numpy(codes).long().to(device)
    else:
        codes_t = codes.long().clamp(0, codebook_size - 1).to(device)
    wavs, sr = tokenizer.decode([{"audio_codes": codes_t}])
    wav = wavs[0] if isinstance(wavs[0], np.ndarray) else wavs[0].cpu().numpy()
    return wav, sr


def _determine_non_streaming(variant):
    """VoiceDesign defaults to non_streaming_mode=True; base/custom to False."""
    return "design" in variant.lower()


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Three-way audio compare: prototype vs manual PyTorch vs ORT"
    )
    parser.add_argument("--variant", default="custom-1.7b", help="Model variant")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Input text")
    parser.add_argument("--instruct", default="", help="VoiceDesign instruction text")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--speaker", default="vivian")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument(
        "--do-sample", action="store_true", default=True,
        help="Use sampling (default True for TTS quality)",
    )
    parser.add_argument(
        "--greedy", action="store_true", default=False,
        help="Force greedy (do_sample=False) for all paths",
    )
    parser.add_argument(
        "--triton-url", default="",
        help="Triton gRPC URL for TRT path (empty=skip). Set to localhost:8001 to test Triton.",
    )
    args = parser.parse_args()

    do_sample = not args.greedy
    non_streaming = _determine_non_streaming(args.variant)

    device = resolve_device(args.device)
    path = resolve_model_path(args.variant, args.models_dir)
    if not has_model_weights(path):
        logger.error("No weights for variant %s", args.variant)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "workspace" / "audio_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model via official high-level API ----
    logger.info("Loading model via Qwen3TTSModel.from_pretrained on %s ...", device)
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as TTSModelWrapper
    wrapper = TTSModelWrapper.from_pretrained(str(path), device_map=str(device), dtype=torch.float32)
    model = wrapper.model
    processor = wrapper.processor

    if getattr(model, "speech_tokenizer", None) is None:
        logger.error("Model has no speech_tokenizer.")
        sys.exit(1)

    speaker = args.speaker or ""
    language = args.language
    max_steps = args.max_steps
    codec_eos_id = int(model.config.talker_config.codec_eos_token_id)

    # ---- (1) Prototype: official high-level API ----
    logger.info("(1) Prototype: official API generate (%s) ...",
                "sampling" if do_sample else "greedy")
    gen_kwargs = dict(do_sample=do_sample, max_new_tokens=max_steps)
    if not do_sample:
        gen_kwargs["repetition_penalty"] = 1.0
        gen_kwargs["subtalker_dosample"] = False
    is_design = "design" in args.variant.lower()
    with torch.no_grad():
        if is_design:
            wavs_proto, sr = wrapper.generate_voice_design(
                text=args.text, instruct=args.instruct, language=language,
                non_streaming_mode=non_streaming, **gen_kwargs,
            )
        else:
            wavs_proto, sr = wrapper.generate_custom_voice(
                text=args.text, speaker=speaker, language=language,
                instruct=args.instruct,
                non_streaming_mode=non_streaming, **gen_kwargs,
            )
    wav_proto = wavs_proto[0]
    logger.info("  Prototype: wav length=%d samples (%.2f s)", len(wav_proto), len(wav_proto) / sr)
    sf.write(str(out_dir / "proto.wav"), wav_proto, sr)

    # ---- Tokenize text using official format ----
    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=args.text)
    tok_out = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = tok_out["input_ids"].to(device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    logger.info("  input_ids shape: %s (official format)", input_ids.shape)
    instruct_ids = None
    if args.instruct:
        instruct_text = wrapper._build_instruct_text(args.instruct)
        instruct_tok = processor(text=instruct_text, return_tensors="pt", padding=True)
        instruct_ids = instruct_tok["input_ids"].to(device=device, dtype=torch.long)
        if instruct_ids.dim() == 1:
            instruct_ids = instruct_ids.unsqueeze(0)

    # ---- (2) Manual PyTorch decode loop ----
    logger.info("(2) Manual PyTorch decode loop (non_streaming=%s) ...", non_streaming)
    prefill_embeds, trailing_list = build_prefill_like_official(
        model, input_ids, language, speaker, device,
        instruct_ids=instruct_ids,
        non_streaming_mode=non_streaming,
    )
    tts_pad_token_id = getattr(model.config, "tts_pad_token_id", 0)
    with torch.no_grad():
        pad_id = torch.tensor(
            [[tts_pad_token_id]], device=device, dtype=torch.long
        )
        pad_embed = model.talker.text_projection(
            model.talker.model.text_embedding(pad_id)
        )
    codes_manual, _, manual_eos = run_manual_decode_loop(
        model,
        prefill_embeds,
        trailing_list,
        pad_embed,
        max_steps,
        codec_eos_id,
        device,
    )
    logger.info("  Manual: T=%d, eos_step=%d", codes_manual.shape[0], manual_eos)

    # ---- (3) ORT decode loop ----
    logger.info("(3) ORT (talker_unified.onnx) decode loop ...")
    onnx_path = DEFAULT_OUTPUT_DIR / args.variant / "talker_unified.onnx"
    model_dir = DEFAULT_OUTPUT_DIR / args.variant
    if not onnx_path.exists():
        logger.error("ONNX not found: %s", onnx_path)
        sys.exit(1)
    H, num_kv_heads, head_dim, num_layers = _load_talker_dims(model_dir)
    inputs_embeds_np = prefill_embeds.detach().cpu().float().numpy()
    pad_embed_np = pad_embed.detach().cpu().float().numpy()
    if pad_embed_np.ndim == 2:
        pad_embed_np = pad_embed_np.reshape(1, 1, -1)
    trailing_np = None
    if trailing_list:
        trailing_np = np.stack(
            [t.detach().cpu().float().numpy() for t in trailing_list], axis=0
        )
        if trailing_np.ndim == 3:
            trailing_np = trailing_np[:, np.newaxis, :, :]
    import onnxruntime as ort
    session = ort.InferenceSession(
        str(onnx_path),
        ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    output_names = [o.name for o in session.get_outputs()]
    codes_ort, ort_eos = run_ort_decode_loop(
        session,
        output_names,
        inputs_embeds_np,
        trailing_np,
        pad_embed_np,
        inputs_embeds_np.shape[1],
        num_layers,
        num_kv_heads,
        head_dim,
        max_steps,
        codec_eos_id,
    )
    logger.info("  ORT: T=%d, eos_step=%d", codes_ort.shape[0], ort_eos)

    def trim_at_eos(codes, eos_step):
        if eos_step >= 0:
            return codes[:eos_step]
        return codes

    # ---- (4) TRT via tts_orchestrator streaming (KV stays on GPU) ----
    codes_trt_trim = None
    trt_eos = None
    wav_trt = None
    if getattr(args, "triton_url", None):
        try:
            import tritonclient.grpc as grpcclient
            logger.info("(4) TRT via tts_orchestrator stream at %s ...", args.triton_url)
            triton_client = grpcclient.InferenceServerClient(url=args.triton_url)
            if not triton_client.is_server_ready():
                raise RuntimeError(f"Triton not ready at {args.triton_url}")
            req_dict = build_variant_request_payload(
                variant=args.variant,
                text=args.text,
                language=language,
                speaker=speaker or "serena",
                instruct=args.instruct or "",
            )
            stream = infer_stream(triton_client, grpcclient, req_dict, timeout=120)
            elapsed = stream.total_ms / 1000.0
            if stream.error:
                raise RuntimeError(f"Orchestrator stream error: {stream.error}")
            if stream.audio is not None and stream.audio.size:
                wav_trt = stream.audio.astype(np.float32)
                sample_rate = int(stream.metadata.get("audio_format", {}).get("sample_rate", sr) or sr)
                logger.info("  TRT (orchestrator): %d samples (%.2f s), elapsed=%.2fs",
                            len(wav_trt), len(wav_trt) / sample_rate, elapsed)
            else:
                logger.warning("  TRT: no audio chunks received")
        except Exception as e:
            logger.warning("TRT orchestrator failed: %s", e)
            wav_trt = None

    codes_manual_trim = trim_at_eos(codes_manual, manual_eos)
    codes_ort_trim = trim_at_eos(codes_ort, ort_eos)

    # Decode manual / ORT / TRT to WAV (shared speech_tokenizer)
    logger.info("Decoding to WAV ...")
    wav_manual, _ = decode_codes_to_wav(model, codes_manual_trim, device)
    sf.write(str(out_dir / "manual_pytorch.wav"), wav_manual, sr)
    wav_ort, _ = decode_codes_to_wav(model, codes_ort_trim, device)
    sf.write(str(out_dir / "ort_fp32.wav"), wav_ort, sr)
    if wav_trt is not None:
        sf.write(str(out_dir / "trt_bf16.wav"), wav_trt, sr)

    # Report
    report_lines = [
        "Audio compare report (proto / manual / ORT / TRT)",
        "==================================================",
        f"text: {args.text[:60]}...",
        f"variant: {args.variant}",
        f"non_streaming_mode: {non_streaming}",
        f"do_sample: {do_sample}",
        f"max_steps: {max_steps}",
        "",
        "Audio lengths:",
        f"  prototype:  {len(wav_proto)} samples ({len(wav_proto)/sr:.2f} s)",
        f"  manual:     {len(wav_manual)} samples ({len(wav_manual)/sr:.2f} s), eos_step={manual_eos}",
        f"  ORT:        {len(wav_ort)} samples ({len(wav_ort)/sr:.2f} s), eos_step={ort_eos}",
    ]
    if wav_trt is not None:
        report_lines.append(
            f"  TRT:        {len(wav_trt)} samples ({len(wav_trt)/sr:.2f} s), via orchestrator"
        )
    elif getattr(args, "triton_url", None):
        report_lines.append(
            "  TRT:        (skipped: Triton not available)"
        )
    report_lines.extend([
        "",
        "Output WAV files (24 kHz):",
        f"  proto.wav          - official API ({'sampling' if do_sample else 'greedy'})",
        "  manual_pytorch.wav - manual PyTorch decode loop (greedy)",
        "  ort_fp32.wav       - ORT talker_unified.onnx decode loop (greedy)",
    ])
    if wav_trt is not None:
        report_lines.append("  trt_bf16.wav       - TRT via tts_orchestrator streaming")
    report_lines.extend([
        "",
        "Note: prototype uses official API; manual/ORT/TRT use greedy decode with",
        "build_prefill_like_official for codec comparison.",
    ])
    report_path = out_dir / "compare_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    for line in report_lines:
        logger.info("  %s", line)

    logger.info("WAV and report saved to %s", out_dir)
    del model, wrapper
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
