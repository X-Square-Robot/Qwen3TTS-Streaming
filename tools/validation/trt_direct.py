#!/usr/bin/env python3
"""Direct TRT engine test: bypass engine loop, call TRT directly.

Mirrors the fused ONNX test logic but uses TRT engine directly (same
engine file as the standalone engine). Handles TRT's minimum shape
constraints with dummy KV + masking (same approach as executor.py).

Usage (conda activate qwen3-tts):
  python tools/validation/trt_direct.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure the validation tools directory is on sys.path for _bootstrap
_validation_dir = str(Path(__file__).resolve().parent)
if _validation_dir not in sys.path:
    sys.path.insert(0, _validation_dir)

from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from audio import save_wav
from common import REPO_ROOT

from engine.backend.executor import TRTEngine
from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trt_direct")

TEXT = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"
SAMPLE_RATE = 24000
SLIDING_WINDOW = 72
_DUMMY_PAST_LEN = 1


def main():
    variant = "custom-1.7b"
    exported_dir = REPO_ROOT / "workspace" / "exported" / variant
    trt_path = exported_dir / "talker_code2wav_fused.engine"
    weights_dir = exported_dir / "weights"
    manifest = json.loads((exported_dir / "triton_manifest.json").read_text())
    tokenizer_dir = (
        REPO_ROOT / "workspace" / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    logger.info("Loading weights and tokenizer...")
    weights = EmbeddingWeights(str(weights_dir), device_id=0)
    tokenizer = load_lightweight_tokenizer(str(tokenizer_dir))
    builder = PrefillBuilder(weights, tokenizer)

    plan = builder.build_plan(
        TaskType.CUSTOM_VOICE, text=TEXT, language="auto", speaker="vivian"
    )
    prefill_embeds = plan.prefill_embeds.to(dtype)
    trailing = plan.trailing
    pad_embed = weights.tts_pad_embed
    codec_eos_id = int(weights.codec_eos_id)

    batch = 1
    seq = prefill_embeds.shape[1]
    num_layers = manifest["talker"]["num_layers"]
    num_kv_heads = manifest["talker"]["num_kv_heads"]
    head_dim = manifest["talker"]["head_dim"]
    codec_vocab_size = manifest["talker"]["vocab_size"]
    c2w_in_names = manifest["code2wav_fused"]["c2w_state_input_names"]
    c2w_out_names = manifest["code2wav_fused"]["c2w_state_output_names"]
    init_shapes = [tuple(s) for s in manifest["code2wav_fused"]["initial_state_shapes"]]

    n_c2w_layers = manifest["architecture"]["n_c2w_layers"]
    c2w_kv_heads = manifest["architecture"]["c2w_kv_heads"]
    c2w_head_dim = manifest["architecture"]["c2w_head_dim"]

    logger.info(
        "Prefill: shape=%s, trailing=%d, codec_eos=%d",
        prefill_embeds.shape,
        len(trailing),
        codec_eos_id,
    )

    logger.info("Loading TRT engine...")
    engine = TRTEngine(str(trt_path), device)
    engine.load()
    all_input_names, all_output_names = engine.get_io_names()
    input_names_set = set(all_input_names)

    logger.info("TRT inputs: %s", all_input_names)
    logger.info("TRT outputs: %s", all_output_names)

    stream = torch.cuda.Stream(device=device)

    c2w_states = {
        name: torch.zeros(shape, device=device, dtype=dtype)
        for name, shape in zip(c2w_in_names, init_shapes)
    }

    gumbel = torch.zeros(batch, 50, device=device, dtype=torch.float32)
    cp_gumbel = torch.zeros(batch, 15, 50, device=device, dtype=torch.float32)
    temperature = torch.ones(batch, 1, device=device, dtype=torch.float32)
    penalty = torch.ones(batch, 1, device=device, dtype=torch.float32)
    token_counts = torch.zeros(
        batch, codec_vocab_size, device=device, dtype=torch.int64
    )

    out_names = [n for n in all_output_names]

    def run_step(
        inp_emb,
        pos_ids,
        cache_pos_val,
        t_past_kv,
        c_past_kv,
        c_st,
        tc,
        use_dummy_kv=False,
        past_len=0,
    ):
        cur_seq = inp_emb.shape[1]
        t_past_len = t_past_kv.shape[3]
        c_past_len = c_past_kv.shape[3]
        chunk_t = 1

        if use_dummy_kv:
            attn_total = _DUMMY_PAST_LEN + cur_seq
            attn = torch.zeros(
                batch, 1, cur_seq, attn_total, device=device, dtype=dtype
            )
            attn[:, :, :, :_DUMMY_PAST_LEN] = float("-inf")
            if cur_seq > 1:
                causal = torch.triu(
                    torch.full(
                        (cur_seq, cur_seq), float("-inf"), device=device, dtype=dtype
                    ),
                    diagonal=1,
                )
                attn[:, :, :, _DUMMY_PAST_LEN : _DUMMY_PAST_LEN + cur_seq] += (
                    causal.unsqueeze(0).unsqueeze(0)
                )
        else:
            attn = torch.zeros(
                batch, 1, cur_seq, t_past_len + cur_seq, device=device, dtype=dtype
            )

        c2w_key_total = min(c_past_len + chunk_t, SLIDING_WINDOW)
        c2w_attn = torch.zeros(
            batch, 1, chunk_t, c2w_key_total, device=device, dtype=dtype
        )
        if use_dummy_kv and c_past_len == _DUMMY_PAST_LEN:
            c2w_attn[:, :, :, 0] = float("-inf")

        d = {
            "input_embeds": inp_emb.to(dtype).contiguous(),
            "position_ids": pos_ids.contiguous(),
            "attention_bias": attn.contiguous(),
            "token_counts": tc.contiguous(),
            "gumbel_noise": gumbel.contiguous(),
            "cp_gumbel_noise": cp_gumbel.contiguous(),
            "temperature": temperature.contiguous(),
            "penalty": penalty.contiguous(),
            "cache_position": torch.full(
                (batch, chunk_t), cache_pos_val, device=device, dtype=torch.float32
            ).contiguous(),
            "c2w_attention_bias": c2w_attn.contiguous(),
            "talker_past_kv": t_past_kv.contiguous(),
            "c2w_past_kv": c_past_kv.contiguous(),
        }
        for name, st in c_st.items():
            d[name] = st.to(dtype).contiguous()

        d = {k: v for k, v in d.items() if k in input_names_set}

        with torch.cuda.stream(stream):
            raw = engine.infer(d, out_names, stream)
        stream.synchronize()
        return raw

    max_steps = 200
    wav_chunks = []
    eos_step = -1

    logger.info("Running prefill (seq=%d) ...", seq)
    t0 = time.perf_counter()

    pos_prefill = (
        torch.arange(seq, device=device, dtype=torch.int64)
        .reshape(1, 1, -1, 1)
        .expand(batch, 3, seq, 1)
    )

    dummy_talker_kv = torch.zeros(
        batch,
        num_layers * 2,
        num_kv_heads,
        _DUMMY_PAST_LEN,
        head_dim,
        device=device,
        dtype=dtype,
    )
    dummy_c2w_kv = torch.zeros(
        batch,
        n_c2w_layers * 2,
        c2w_kv_heads,
        _DUMMY_PAST_LEN,
        c2w_head_dim,
        device=device,
        dtype=dtype,
    )

    raw = run_step(
        prefill_embeds,
        pos_prefill,
        0,
        dummy_talker_kv,
        dummy_c2w_kv,
        c2w_states,
        token_counts,
        use_dummy_kv=True,
        past_len=0,
    )

    codec_sum = raw["codec_sum"]
    full_codec = raw["full_codec"]
    wav_out = raw["wav"]
    token_counts = raw["updated_token_counts"].clone()

    talker_past_kv = raw["talker_new_kv"].contiguous()
    c2w_past_kv = raw["c2w_new_kv"].contiguous()

    c2w_states = {
        in_n: raw[out_n].clone() for in_n, out_n in zip(c2w_in_names, c2w_out_names)
    }

    if wav_out is not None and wav_out.numel() > 0:
        wav_chunks.append(wav_out[0].cpu().float().numpy().flatten())

    codec_0 = int(full_codec[0, 0].item())
    if codec_0 == codec_eos_id:
        eos_step = 0

    text_add = trailing[0] if len(trailing) > 0 else pad_embed
    next_input = (codec_sum.float() + text_add.to(device).float()).to(torch.float32)

    past_len = seq

    logger.info(
        "Prefill done: codec_0=%d, talker_kv=%s, c2w_kv=%s, wav=%d",
        codec_0,
        talker_past_kv.shape,
        c2w_past_kv.shape,
        wav_out.numel() if wav_out is not None else 0,
    )

    for step in range(1, max_steps):
        if eos_step >= 0:
            break

        pos_step = torch.full(
            (batch, 3, 1, 1), past_len, device=device, dtype=torch.int64
        )

        raw = run_step(
            next_input,
            pos_step,
            step,
            talker_past_kv,
            c2w_past_kv,
            c2w_states,
            token_counts,
            use_dummy_kv=False,
            past_len=past_len,
        )

        codec_sum = raw["codec_sum"]
        full_codec = raw["full_codec"]
        wav_out = raw["wav"]
        token_counts = raw["updated_token_counts"].clone()

        new_kv = raw["talker_new_kv"]
        talker_past_kv = torch.cat([talker_past_kv, new_kv], dim=3).contiguous()

        c2w_past_kv = torch.cat([c2w_past_kv, raw["c2w_new_kv"]], dim=3).contiguous()
        kv_max = max(1, SLIDING_WINDOW - 1)
        if c2w_past_kv.shape[3] > kv_max:
            c2w_past_kv = c2w_past_kv[:, :, :, -kv_max:, :].contiguous()

        c2w_states = {
            in_n: raw[out_n].clone() for in_n, out_n in zip(c2w_in_names, c2w_out_names)
        }

        if wav_out is not None and wav_out.numel() > 0:
            wav_chunks.append(wav_out[0].cpu().float().numpy().flatten())

        codec_0 = int(full_codec[0, 0].item())
        if codec_0 == codec_eos_id:
            eos_step = step

        text_add = trailing[step] if step < len(trailing) else pad_embed
        next_input = (codec_sum.float() + text_add.to(device).float()).to(torch.float32)

        past_len += 1

        if step % 10 == 0:
            logger.info(
                "  step %d: codec_0=%d, past_len=%d, c2w_kv=%s",
                step,
                codec_0,
                past_len,
                c2w_past_kv.shape,
            )

    elapsed = time.perf_counter() - t0

    if eos_step >= 0:
        wav_chunks = wav_chunks[:eos_step]

    if wav_chunks:
        full_wav = np.concatenate(wav_chunks).astype(np.float32)
    else:
        full_wav = np.array([], dtype=np.float32)

    out_dir = REPO_ROOT / "workspace" / "audio_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trt_direct.wav"
    if full_wav.size > 0:
        save_wav(full_wav, out_path, sample_rate=SAMPLE_RATE)

    logger.info("=== RESULTS ===")
    logger.info("Steps: %d, EOS: %d", step, eos_step)
    logger.info("WAV: %d samples, %.2f s", len(full_wav), len(full_wav) / SAMPLE_RATE)
    logger.info("Time: %.1f s", elapsed)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
