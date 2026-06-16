#!/usr/bin/env python3
"""
PyTorch bf16 streaming decode baseline — for diagnosing TRT hallucination.

Runs the talker model in bf16 with streaming mode (prefix + first_text prefilled,
remaining text as trailing tokens fed step-by-step).  This is the PyTorch-native
equivalent of our TRT engine's decode loop.

Usage:
    python scripts/python/pytorch_streaming_baseline.py \
        --text "人工智能正在深刻改变我们的世界。" \
        --speaker vivian \
        --output /tmp/baseline_out

    # Long story (matches the test suite's longtext-story test):
    python scripts/python/pytorch_streaming_baseline.py \
        --text-file tests/fixtures/story_text.txt \
        --speaker vivian \
        --max-steps 512 \
        --output /tmp/story_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "python"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "export"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Qwen3-TTS"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline")


def run_cp_stages_bf16(cp, past_hidden, codec_token_0, device):
    """Run unrolled CP (15 stages, no KV cache) in bf16.  Matches CodePredictorUnrolled."""
    embed_0 = cp.model.codec_embedding[0].weight.new_zeros(1)  # dummy — we use talker embed
    # Actually embed codec_0 through TALKER's embedding (same as official + our TRT)
    talker_embed = past_hidden  # will be replaced below
    # We need the talker codec_embedding — passed separately
    raise NotImplementedError("Use run_cp_stages_with_talker")


def run_cp_stages_with_talker(cp, talker_codec_embedding, past_hidden, codec_token_0, device):
    """Run unrolled CP in bf16.  Exact match to official forward() logic."""
    embed_0 = talker_codec_embedding(codec_token_0).unsqueeze(1)  # [1, 1, talker_H]
    seq = cp.small_to_mtp_projection(torch.cat([past_hidden, embed_0], dim=1))  # [1, 2, cp_H]

    tokens = []
    for stage in range(len(cp.lm_head)):
        B, S, D = seq.shape
        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        pos_embs = cp.model.rotary_emb(seq, pos_ids)
        causal = torch.triu(
            torch.full((S, S), float("-inf"), device=device, dtype=seq.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        h = seq
        for layer in cp.model.layers:
            h = layer(
                h,
                attention_mask=causal,
                position_ids=pos_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=torch.arange(S, device=device),
                position_embeddings=pos_embs,
            )[0]
        h = cp.model.norm(h)
        logits = cp.lm_head[stage](h[:, -1:, :])
        token = logits.argmax(dim=-1).squeeze(-1)  # greedy
        tokens.append(token.item())

        if stage < len(cp.lm_head) - 1:
            next_emb = cp.small_to_mtp_projection(
                cp.model.codec_embedding[stage](token).unsqueeze(1)
            )
            seq = torch.cat([seq, next_emb], dim=1)

    return tokens


def build_codec_sum_bf16(talker, cp, codec_token_0_id, cp_tokens, device):
    """Sum all 16 codec embeddings — matches official forward() line 1682-1687."""
    hidden_size = talker.model.config.hidden_size
    codec_sum = torch.zeros(1, 1, hidden_size, device=device, dtype=torch.bfloat16)

    all_ids = [codec_token_0_id] + cp_tokens
    for i, tid in enumerate(all_ids):
        t = torch.tensor([tid], device=device, dtype=torch.long)
        if i == 0:
            codec_sum += talker.model.codec_embedding(t).unsqueeze(1)
        elif i - 1 < len(cp.model.codec_embedding):
            codec_sum += cp.model.codec_embedding[i - 1](t).unsqueeze(1)

    return codec_sum


def main():
    parser = argparse.ArgumentParser(description="PyTorch bf16 streaming decode baseline")
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    parser.add_argument("--text", default=None, help="Input text")
    parser.add_argument("--text-file", default=None, help="Read text from file")
    parser.add_argument("--speaker", default="vivian")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--output", default="/tmp/pytorch_baseline")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text().strip()
    elif args.text:
        text = args.text
    else:
        text = "人工智能正在深刻改变我们的世界。从语音识别到自然语言处理，AI的应用已经渗透到生活的方方面面。"

    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model in bf16 ----
    logger.info("Loading model in bf16 ...")
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerForConditionalGeneration,
    )
    talker = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager",
    )
    talker.eval()
    cp = talker.code_predictor
    codec_eos_id = getattr(talker.config, "codec_eos_token_id", 2150)
    tts_pad_token_id = getattr(talker.config, "tts_pad_token_id", 0)
    logger.info(f"  codec_eos_id={codec_eos_id}, pad_id={tts_pad_token_id}")

    # ---- Build streaming prefill using our PrefillBuilder (production code path) ----
    logger.info("Building streaming prefill via PrefillBuilder ...")
    sys.path.insert(0, str(REPO_ROOT))
    from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType

    weights_dir = str(Path(args.model_dir).parent / "exported" / "custom-1.7b" / "weights")
    if not Path(weights_dir).exists():
        weights_dir = str(REPO_ROOT / "workspace" / "exported" / "custom-1.7b" / "weights")
    tokenizer = load_lightweight_tokenizer(args.model_dir)
    w = EmbeddingWeights(weights_dir, device_id=0)
    builder = PrefillBuilder(w, tokenizer)

    token_ids = builder._encode_text_ids(text)
    logger.info(f"  text tokens: {len(token_ids)}")

    plan = builder.build_plan_from_ids(
        task_type=TaskType.CUSTOM_VOICE,
        token_ids=token_ids,
        language=args.language,
        speaker=args.speaker,
        include_eos=True,
    )
    prefill_embeds = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    trailing_list = [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing]

    # Build pad embed in bf16
    pad_embed = w.tts_pad_embed.to(device=device, dtype=torch.bfloat16)

    logger.info(f"  prefill: {list(prefill_embeds.shape)}, trailing: {len(trailing_list)} tokens")

    # ---- Prefill ----
    logger.info("Running prefill ...")
    with torch.no_grad():
        prefill_out = talker.model(
            inputs_embeds=prefill_embeds,
            use_cache=True,
            return_dict=True,
        )
    kv = prefill_out.past_key_values
    past_hidden = prefill_out.last_hidden_state[:, -1:, :]
    prefill_logits = talker.codec_head(prefill_out.last_hidden_state)
    codec_0 = prefill_logits[:, -1, :].argmax(dim=-1)
    logger.info(f"  prefill codec_0 = {codec_0.item()}")

    # ---- Decode loop (bf16, streaming) ----
    logger.info(f"Running streaming decode (max_steps={args.max_steps}) ...")
    all_codecs = []
    eos_step = -1
    t0 = time.time()

    with torch.no_grad():
        for step in range(args.max_steps):
            # 1. CP: generate remaining 15 tokens
            cp_tokens = run_cp_stages_with_talker(
                cp, talker.model.codec_embedding, past_hidden, codec_0, device
            )

            # 2. Build codec_sum (sum of all 16 embeddings)
            full_codec = [codec_0.item()] + cp_tokens
            all_codecs.append(full_codec)
            codec_sum = build_codec_sum_bf16(talker, cp, codec_0.item(), cp_tokens, device)

            # 3. Add text token
            if step < len(trailing_list):
                text_add = trailing_list[step]
            else:
                text_add = pad_embed
            next_input = codec_sum + text_add  # bf16 + bf16 → bf16

            # 4. Talker forward
            step_out = talker.model(
                inputs_embeds=next_input,
                past_key_values=kv,
                use_cache=True,
                return_dict=True,
            )
            past_hidden = step_out.last_hidden_state[:, -1:, :]
            kv = step_out.past_key_values
            step_logits = talker.codec_head(past_hidden)
            codec_0 = step_logits[:, -1, :].argmax(dim=-1)

            # 5. Check EOS
            if codec_0.item() == codec_eos_id:
                if eos_step < 0:
                    eos_step = step
                    logger.info(f"  EOS at step {step}")
                break

            # Progress
            phase = "text" if step < len(trailing_list) else "pad"
            if step < 3 or step % 50 == 0:
                logger.info(f"  step {step}: codec_0={full_codec[0]}, phase={phase}")

    elapsed = time.time() - t0
    all_codecs = np.array(all_codecs, dtype=np.int64)
    logger.info(f"Decode done: {len(all_codecs)} steps in {elapsed:.1f}s, eos_step={eos_step}")

    # ---- Save codecs ----
    np.savez(out_dir / "codecs.npz", codecs=all_codecs, eos_step=eos_step)
    logger.info(f"Saved codecs to {out_dir / 'codecs.npz'}")

    # ---- Decode audio via speech_tokenizer ----
    logger.info("Decoding audio ...")
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration as FullModel
        full_model2 = FullModel.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, device_map=args.device,
            attn_implementation="eager",
        )
        full_model2.eval()
        codes_tensor = torch.from_numpy(all_codecs).to(device=device, dtype=torch.long)
        if codes_tensor.dim() == 2:
            codes_tensor = codes_tensor.unsqueeze(0)  # [1, T, 16]
        wavs, sr = full_model2.speech_tokenizer.decode(
            [{"audio_codes": codes_tensor[0]}]
        )
        import soundfile as sf
        wav_data = wavs[0]
        wav_np = wav_data.cpu().float().numpy() if isinstance(wav_data, torch.Tensor) else np.asarray(wav_data, dtype=np.float32)
        sf.write(str(out_dir / "audio.wav"), wav_np, sr)
        logger.info(f"Saved audio to {out_dir / 'audio.wav'} ({len(wav_np)/sr:.1f}s, sr={sr})")
        del full_model2
    except Exception as e:
        logger.error(f"Audio decode failed: {e}")
        logger.info("Codecs saved — you can decode audio separately.")

    # ---- Print codec_0 sequence for quick comparison ----
    codec_0_seq = all_codecs[:, 0].tolist()
    logger.info(f"First 20 codec_0: {codec_0_seq[:20]}")
    if eos_step >= 0:
        logger.info(f"Last 10 codec_0 before EOS: {codec_0_seq[max(0,eos_step-10):eos_step+1]}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
