#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'third_party' / 'Qwen3-TTS'))
sys.path.insert(0, str(REPO_ROOT))

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
from engine.backend.prefill import EmbeddingWeights, OFFICIAL_ASSISTANT_FMT, PrefillBuilder, TaskType


def run_official(model, text: str, max_new_tokens: int):
    device = next(model.parameters()).device
    tok = load_lightweight_tokenizer(str(REPO_ROOT / 'workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice'))
    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=text)
    out = tok(assistant_text, return_tensors='pt')
    input_ids = torch.as_tensor(out['input_ids'], device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    with torch.no_grad():
        codes_list, _ = model.generate(
            input_ids=[input_ids],
            languages=['auto'],
            speakers=['vivian'],
            non_streaming_mode=False,
            do_sample=False,
            subtalker_dosample=False,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            max_new_tokens=max_new_tokens,
        )
    codes = codes_list[0].detach().cpu().tolist()
    return [row[0] for row in codes], codes


def run_manual(model, text: str, max_steps: int):
    device = next(model.parameters()).device
    talker = model.talker
    cp = talker.code_predictor

    tokenizer = load_lightweight_tokenizer(str(REPO_ROOT / 'workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice'))
    weights_dir = str(REPO_ROOT / 'workspace/exported/custom-1.7b/weights')
    w = EmbeddingWeights(weights_dir, device_id=device.index or 0)
    builder = PrefillBuilder(w, tokenizer)
    token_ids = builder._encode_text_ids(text)
    plan = builder.build_plan_from_ids(
        task_type=TaskType.CUSTOM_VOICE,
        token_ids=token_ids,
        language='auto',
        speaker='vivian',
        include_eos=True,
    )
    prefill_embeds = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    trailing = [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing]
    pad_embed = w.tts_pad_embed.to(device=device, dtype=torch.bfloat16)
    suppress_tokens = [i for i in range(talker.config.vocab_size - 1024, talker.config.vocab_size) if i not in (talker.config.codec_eos_token_id,)]

    history = torch.empty((1, 0), device=device, dtype=torch.long)
    with torch.no_grad():
        out = talker.model(inputs_embeds=prefill_embeds, use_cache=True, return_dict=True)
    kv = out.past_key_values
    past_hidden = out.last_hidden_state[:, -1:, :]
    logits = talker.codec_head(out.last_hidden_state)[:, -1, :].float()
    logits[..., suppress_tokens] = -1e9
    logits = RepetitionPenaltyLogitsProcessor(1.05)(history, logits)
    codec0 = logits.argmax(dim=-1)
    history = torch.cat([history, codec0.unsqueeze(1)], dim=1)

    talker_seq = [codec0.item()]
    cp_seqs = []

    for step in range(max_steps - 1):
        with torch.no_grad():
            cp_input = torch.cat((past_hidden, talker.model.codec_embedding(codec0).unsqueeze(1)), dim=1)
            pred = cp.generate(
                inputs_embeds=cp_input,
                max_new_tokens=talker.config.num_code_groups - 1,
                do_sample=False,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        cp_tokens = pred.sequences[0].detach().cpu().tolist()
        cp_seqs.append(cp_tokens)

        last_id_hidden = talker.get_input_embeddings()(codec0.unsqueeze(1))
        codec_hiddens = torch.cat(
            [last_id_hidden] + [talker.code_predictor.get_input_embeddings()[i](pred.sequences[..., i:i+1]) for i in range(talker.config.num_code_groups - 1)],
            dim=1,
        )
        next_input = codec_hiddens.sum(1, keepdim=True) + (trailing[step] if step < len(trailing) else pad_embed)
        with torch.no_grad():
            step_out = talker.model(inputs_embeds=next_input, past_key_values=kv, use_cache=True, return_dict=True)
        kv = step_out.past_key_values
        past_hidden = step_out.last_hidden_state[:, -1:, :]
        logits = talker.codec_head(past_hidden)[:, -1, :].float()
        logits[..., suppress_tokens] = -1e9
        logits = RepetitionPenaltyLogitsProcessor(1.05)(history, logits)
        codec0 = logits.argmax(dim=-1)
        history = torch.cat([history, codec0.unsqueeze(1)], dim=1)
        talker_seq.append(codec0.item())
        if codec0.item() == talker.config.codec_eos_token_id:
            break

    return talker_seq, cp_seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', required=True)
    ap.add_argument('--max-steps', type=int, default=16)
    args = ap.parse_args()

    model_dir = str(REPO_ROOT / 'workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice')
    model = Qwen3TTSForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='eager')
    model.eval()

    off_talker, off_codes = run_official(model, args.text, args.max_steps)
    our_talker, our_cp = run_manual(model, args.text, args.max_steps)

    n = min(len(off_talker), len(our_talker))
    fd = -1
    for i in range(n):
        if off_talker[i] != our_talker[i]:
            fd = i
            break

    print('official talker:', off_talker[:n])
    print('our      talker:', our_talker[:n])
    print('first divergence:', fd)
    if fd >= 0:
        lo = max(0, fd - 3)
        hi = min(n, fd + 6)
        for i in range(lo, hi):
            print(f'step={i:02d} official={off_talker[i]:4d} our={our_talker[i]:4d}')


if __name__ == '__main__':
    main()
