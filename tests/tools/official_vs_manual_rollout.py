#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensLogitsProcessor,
)
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Qwen3-TTS"))
sys.path.insert(0, str(REPO_ROOT))

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
from engine.backend.prefill import EmbeddingWeights, OFFICIAL_ASSISTANT_FMT, PrefillBuilder, TaskType


def sample_talker_token(logits, history_ids, *, top_k, temperature, repetition_penalty, suppress_tokens):
    scores = logits.float().clone()
    if suppress_tokens:
        scores[..., suppress_tokens] = -1e9
    if repetition_penalty != 1.0:
        scores = RepetitionPenaltyLogitsProcessor(repetition_penalty)(history_ids, scores)
    if temperature != 1.0:
        scores = TemperatureLogitsWarper(temperature)(history_ids, scores)
    if top_k > 0:
        scores = TopKLogitsWarper(top_k)(history_ids, scores)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def build_prefill(text, speaker, language, device):
    tokenizer = load_lightweight_tokenizer(str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    weights_dir = str(REPO_ROOT / "workspace/exported/custom-1.7b/weights")
    w = EmbeddingWeights(weights_dir, device_id=device.index or 0)
    builder = PrefillBuilder(w, tokenizer)
    token_ids = builder._encode_text_ids(text)
    plan = builder.build_plan_from_ids(
        task_type=TaskType.CUSTOM_VOICE,
        token_ids=token_ids,
        language=language,
        speaker=speaker,
        include_eos=True,
    )
    return plan, w


def run_manual(model, text, speaker, language, seed, max_steps=512):
    device = next(model.parameters()).device
    talker = model.talker
    cp = talker.code_predictor
    plan, w = build_prefill(text, speaker, language, device)
    prefill_embeds = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    trailing = [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing]
    pad_embed = w.tts_pad_embed.to(device=device, dtype=torch.bfloat16)

    suppress_tokens = [
        i for i in range(talker.config.vocab_size - 1024, talker.config.vocab_size)
        if i not in (talker.config.codec_eos_token_id,)
    ]

    with torch.no_grad():
        out = talker.model(inputs_embeds=prefill_embeds, use_cache=True, return_dict=True)
    kv = out.past_key_values
    past_hidden = out.last_hidden_state[:, -1:, :]
    logits = talker.codec_head(out.last_hidden_state)[:, -1, :]

    torch.manual_seed(seed)
    history = torch.empty((1, 0), device=device, dtype=torch.long)
    codec0 = sample_talker_token(
        logits, history,
        top_k=50, temperature=0.9,
        repetition_penalty=1.05,
        suppress_tokens=suppress_tokens,
    )
    history = torch.cat([history, codec0.unsqueeze(1)], dim=1)

    seq = [codec0.item()]
    phases = ["prefill"]
    for step in range(max_steps - 1):
        with torch.no_grad():
            cp_input = torch.cat((past_hidden, talker.model.codec_embedding(codec0).unsqueeze(1)), dim=1)
            pred = cp.generate(
                inputs_embeds=cp_input,
                max_new_tokens=talker.config.num_code_groups - 1,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        cp_tokens = pred.sequences
        # Match official forward() exactly: use CP embedding tables, then sum in talker space.
        last_id_hidden = talker.get_input_embeddings()(codec0.unsqueeze(1))
        codec_hiddens = torch.cat(
            [last_id_hidden]
            + [talker.code_predictor.get_input_embeddings()[i](cp_tokens[..., i:i+1]) for i in range(talker.config.num_code_groups - 1)],
            dim=1,
        )
        next_input = codec_hiddens.sum(1, keepdim=True)
        phase = "text" if step < len(trailing) else "pad"
        next_input = next_input + (trailing[step] if step < len(trailing) else pad_embed)

        with torch.no_grad():
            step_out = talker.model(inputs_embeds=next_input, past_key_values=kv, use_cache=True, return_dict=True)
        kv = step_out.past_key_values
        past_hidden = step_out.last_hidden_state[:, -1:, :]
        logits = talker.codec_head(past_hidden)[:, -1, :]
        codec0 = sample_talker_token(
            logits, history,
            top_k=50, temperature=0.9,
            repetition_penalty=1.05,
            suppress_tokens=suppress_tokens,
        )
        history = torch.cat([history, codec0.unsqueeze(1)], dim=1)
        seq.append(codec0.item())
        phases.append(phase)
        if codec0.item() == talker.config.codec_eos_token_id:
            break
    return seq, phases, len(trailing)


def run_official(model, text, speaker, language, seed, max_steps=512):
    device = next(model.parameters()).device
    tokenizer = load_lightweight_tokenizer(str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=text)
    out = tokenizer(assistant_text, return_tensors="pt")
    input_ids = torch.as_tensor(out["input_ids"], device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    torch.manual_seed(seed)
    with torch.no_grad():
        codes_list, _ = model.generate(
            input_ids=[input_ids],
            languages=[language],
            speakers=[speaker],
            non_streaming_mode=False,
            do_sample=True,
            subtalker_dosample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            max_new_tokens=max_steps,
        )
    codes = codes_list[0]
    codes = codes.detach().cpu().tolist() if hasattr(codes, 'detach') else codes
    seq = [row[0] for row in codes]
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", default="vivian")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-steps", type=int, default=256)
    args = ap.parse_args()

    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
    model.eval()

    off = run_official(model, args.text, args.speaker, args.language, args.seed, args.max_steps)
    man, phases, trailing_len = run_manual(model, args.text, args.speaker, args.language, args.seed, args.max_steps)

    n = min(len(off), len(man))
    first_div = -1
    for i in range(n):
        if off[i] != man[i]:
            first_div = i
            break

    print('trailing_len:', trailing_len)
    print('official_len:', len(off), 'manual_len:', len(man))
    print('first_divergence:', first_div)
    print('official first40:', off[:40])
    print('manual   first40:', man[:40])
    if first_div >= 0:
        lo = max(0, first_div - 5)
        hi = min(n, first_div + 10)
        print('--- window around divergence ---')
        for i in range(lo, hi):
            phase = phases[i] if i < len(phases) else 'n/a'
            mark = '<<' if i == first_div else '  '
            print(f'{mark} step={i:03d} phase={phase:7s} official={off[i]:4d} manual={man[i]:4d}')
    else:
        print('Sequences match on common prefix.')


if __name__ == '__main__':
    main()
