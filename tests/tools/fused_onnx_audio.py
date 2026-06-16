#!/usr/bin/env python3
"""Test talker_code2wav_fused.onnx: run full decode loop and save WAV.

Builds prefill with PrefillBuilder (same weights as engine), runs fused ONNX
through ORT, collects wav chunks, saves to workspace/audio_compare/fused_ort.wav.

Usage (conda activate qwen3-tts):
  python tests/tools/fused_onnx_audio.py
  python tests/tools/fused_onnx_audio.py --text "你好" --max-steps 100
"""

import argparse
import json
import logging
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from qwen3tts_tools.audio import save_wav
from qwen3tts_tools.common import REPO_ROOT

from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fused_ort_test")

DEFAULT_TEXT = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"
SAMPLE_RATE = 24000
SLIDING_WINDOW = 72
def main():
    parser = argparse.ArgumentParser(description="Test fused ONNX model audio output")
    parser.add_argument("--variant", default="custom-1.7b")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--speaker", default="vivian")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    exported_dir = REPO_ROOT / "workspace" / "exported" / args.variant
    onnx_path = exported_dir / "talker_code2wav_fused.onnx"
    weights_dir = exported_dir / "weights"
    manifest_path = exported_dir / "triton_manifest.json"

    model_dir_map = {
        "custom-0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "custom-1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }
    model_dir = model_dir_map.get(args.variant)
    if model_dir is None:
        logger.error("Unsupported variant: %s", args.variant)
        sys.exit(1)
    tokenizer_dir = REPO_ROOT / "workspace" / "models" / model_dir

    if not onnx_path.exists():
        logger.error("ONNX not found: %s", onnx_path)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    c2w_in_names = manifest["code2wav_fused"]["c2w_state_input_names"]
    c2w_out_names = manifest["code2wav_fused"]["c2w_state_output_names"]
    init_shapes = [tuple(s) for s in manifest["code2wav_fused"]["initial_state_shapes"]]
    num_layers = manifest["talker"]["num_layers"]
    num_kv_heads = manifest["talker"]["num_kv_heads"]
    head_dim = manifest["talker"]["head_dim"]
    codec_vocab_size = manifest["talker"]["vocab_size"]

    logger.info("Loading weights and tokenizer...")
    weights = EmbeddingWeights(str(weights_dir), device_id=0)
    tokenizer = load_lightweight_tokenizer(str(tokenizer_dir))
    builder = PrefillBuilder(weights, tokenizer)

    task_type = TaskType.CUSTOM_VOICE
    plan = builder.build_plan(task_type, text=args.text, language=args.language, speaker=args.speaker)
    prefill_embeds = plan.prefill_embeds
    trailing = plan.trailing
    pad_embed = weights.tts_pad_embed
    codec_eos_id = int(weights.codec_eos_id)
    batch = 1

    logger.info("Prefill embeds: %s, trailing: %d, codec_eos_id: %d",
                prefill_embeds.shape, len(trailing), codec_eos_id)

    providers = ["CUDAExecutionProvider"] if args.provider == "cuda" else ["CPUExecutionProvider"]
    logger.info("Loading fused ONNX model with %s ...", providers[0])
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_names = {i.name for i in sess.get_inputs()}
    output_names = [o.name for o in sess.get_outputs()]

    c2w_states_np = [np.zeros(shape, dtype=np.float32) for shape in init_shapes]
    talker_past_kv = np.empty((batch, num_layers * 2, num_kv_heads, 0, head_dim), dtype=np.float32)
    c2w_past_kv = np.empty((batch, 16, 16, 0, 64), dtype=np.float32)

    seq = prefill_embeds.shape[1]
    inp_np = prefill_embeds.detach().cpu().float().numpy()
    pos_np = (
        torch.arange(seq, dtype=torch.int64)
        .reshape(1, 1, -1, 1)
        .expand(batch, 3, seq, 1)
        .numpy()
    )

    gumbel_noise = np.zeros((batch, 50), dtype=np.float32)
    cp_gumbel_noise = np.zeros((batch, 15, 50), dtype=np.float32)
    temperature = np.zeros((batch, 1), dtype=np.float32)
    penalty = np.ones((batch, 1), dtype=np.float32)
    token_counts = np.zeros((batch, codec_vocab_size), dtype=np.int64)

    def build_feed(inp, pos, cache_pos_val, past_kv_np, c2w_kv_np, c2w_st, tc):
        cur_seq = inp.shape[1]
        past_len = past_kv_np.shape[3]
        c2w_past_len = c2w_kv_np.shape[3]
        chunk_t = 1

        feed = {
            "input_embeds": inp.astype(np.float32),
            "position_ids": pos.astype(np.int64),
            "attention_bias": np.zeros((batch, 1, cur_seq, past_len + cur_seq), dtype=np.float32),
            "token_counts": tc.astype(np.int64),
            "gumbel_noise": gumbel_noise,
            "cp_gumbel_noise": cp_gumbel_noise,
            "temperature": temperature,
            "penalty": penalty,
            "cache_position": np.full((batch, chunk_t), cache_pos_val, dtype=np.float32),
            "c2w_attention_bias": np.zeros(
                (batch, 1, chunk_t, min(c2w_past_len + chunk_t, SLIDING_WINDOW)),
                dtype=np.float32,
            ),
            "talker_past_kv": past_kv_np.astype(np.float32),
            "c2w_past_kv": c2w_kv_np.astype(np.float32),
        }
        for name, st in zip(c2w_in_names, c2w_st):
            feed[name] = st.astype(np.float32)
        return {k: v for k, v in feed.items() if k in input_names}

    wav_chunks = []
    all_codec_0 = []
    eos_step = -1

    logger.info("Running prefill (seq=%d) ...", seq)
    t0 = time.perf_counter()

    feed = build_feed(inp_np, pos_np, 0, talker_past_kv, c2w_past_kv, c2w_states_np, token_counts)
    outs = dict(zip(output_names, sess.run(output_names, feed)))

    codec_sum = outs["codec_sum"]
    full_codec = outs["full_codec"]
    wav_chunk = outs["wav"]
    token_counts = outs["updated_token_counts"].copy()

    talker_past_kv = outs["talker_new_kv"].copy()
    c2w_past_kv = outs["c2w_new_kv"].copy()
    kv_max = max(1, SLIDING_WINDOW - 1)
    if c2w_past_kv.shape[3] > kv_max:
        c2w_past_kv = c2w_past_kv[:, :, :, -kv_max:, :].copy()
    c2w_states_np = [outs[name].copy() for name in c2w_out_names]

    if wav_chunk is not None and wav_chunk.size > 0:
        wav_chunks.append(wav_chunk.flatten())
    codec_0 = int(full_codec[0, 0])
    all_codec_0.append(codec_0)
    if codec_0 == codec_eos_id:
        eos_step = 0

    text_add = trailing[0].detach().cpu().float().numpy() if len(trailing) > 0 else pad_embed.detach().cpu().float().numpy()
    if text_add.ndim == 2:
        text_add = text_add.reshape(1, 1, -1)
    next_input = (codec_sum.astype(np.float64) + text_add.astype(np.float64)).astype(np.float32)

    logger.info("Prefill done: codec_0=%d, wav_samples=%d", codec_0, wav_chunk.flatten().size if wav_chunk is not None else 0)

    for step in range(1, args.max_steps):
        if eos_step >= 0:
            break

        pos_step = np.full((batch, 3, 1, 1), seq + step - 1, dtype=np.int64)
        feed = build_feed(next_input, pos_step, step, talker_past_kv, c2w_past_kv, c2w_states_np, token_counts)
        outs = dict(zip(output_names, sess.run(output_names, feed)))

        codec_sum = outs["codec_sum"]
        full_codec = outs["full_codec"]
        wav_chunk = outs["wav"]
        token_counts = outs["updated_token_counts"].copy()

        talker_past_kv = np.concatenate([talker_past_kv, outs["talker_new_kv"]], axis=3).copy()
        c2w_past_kv = np.concatenate([c2w_past_kv, outs["c2w_new_kv"]], axis=3).copy()
        if c2w_past_kv.shape[3] > kv_max:
            c2w_past_kv = c2w_past_kv[:, :, :, -kv_max:, :].copy()
        c2w_states_np = [outs[name].copy() for name in c2w_out_names]

        if wav_chunk is not None and wav_chunk.size > 0:
            wav_chunks.append(wav_chunk.flatten())

        codec_0 = int(full_codec[0, 0])
        all_codec_0.append(codec_0)
        if codec_0 == codec_eos_id:
            eos_step = step

        text_add = (
            trailing[step].detach().cpu().float().numpy()
            if step < len(trailing)
            else pad_embed.detach().cpu().float().numpy()
        )
        if text_add.ndim == 2:
            text_add = text_add.reshape(1, 1, -1)
        next_input = (codec_sum.astype(np.float64) + text_add.astype(np.float64)).astype(np.float32)

        if (step) % 10 == 0:
            logger.info("  step %d: codec_0=%d", step, codec_0)

    elapsed = time.perf_counter() - t0

    if eos_step >= 0:
        wav_chunks = wav_chunks[:eos_step]

    if wav_chunks:
        full_wav = np.concatenate(wav_chunks).astype(np.float32)
        duration = len(full_wav) / SAMPLE_RATE
    else:
        full_wav = np.array([], dtype=np.float32)
        duration = 0.0

    out_dir = REPO_ROOT / "workspace" / "audio_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fused_ort.wav"
    if full_wav.size > 0:
        save_wav(full_wav, out_path, sample_rate=SAMPLE_RATE)

    logger.info("=== RESULTS ===")
    logger.info("Text: %s", args.text)
    logger.info("Steps: %d, EOS step: %d", len(all_codec_0), eos_step)
    logger.info("WAV: %d samples, %.2f s", len(full_wav), duration)
    logger.info("Time: %.1f s", elapsed)
    logger.info("Saved: %s", out_path)
    logger.info("Compare with proto.wav / ort_fp32.wav in same directory")


if __name__ == "__main__":
    main()
