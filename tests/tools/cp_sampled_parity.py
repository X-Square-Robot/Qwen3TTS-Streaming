#!/usr/bin/env python3
"""
Sampled parity experiments for official CP vs unrolled CP.

Goals:
1. Compare official cached CP generate vs our unrolled CP on real on-manifold inputs
2. Compare rollout divergence step-by-step on representative segments
3. Help decide whether official streaming+sample params are unstable or our pipeline diverges sharply
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Qwen3-TTS"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "export"))
sys.path.insert(0, str(REPO_ROOT))

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
from utils import CodePredictorUnrolled
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
from engine.backend.prefill import EmbeddingWeights, PrefillBuilder, TaskType


def build_on_manifold_state(talker, text: str, speaker: str, language: str, device: torch.device):
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
    prefill_embeds = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = talker.model(inputs_embeds=prefill_embeds, use_cache=True, return_dict=True)
    past_hidden = out.last_hidden_state[:, -1:, :]
    logits = talker.codec_head(out.last_hidden_state)
    codec0 = logits[:, -1, :].argmax(dim=-1)
    return past_hidden, codec0, plan, logits[:, -1, :]


def run_official_cp(cp, talker, past_hidden, codec0, do_sample: bool, top_k: int, top_p: float, temperature: float):
    with torch.no_grad():
        result = cp.generate(
            inputs_embeds=torch.cat((past_hidden, talker.model.codec_embedding(codec0).unsqueeze(1)), dim=1),
            max_new_tokens=talker.config.num_code_groups - 1,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    return result.sequences[0].detach().cpu().tolist()


def run_unrolled_cp(cp_unrolled, past_hidden, codec0, do_sample: bool, top_k: int, temperature: float, seed: int, device: torch.device):
    if do_sample:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        cp_gumbel = torch.rand(
            1, cp_unrolled.num_stages, top_k,
            device=device, dtype=torch.float32, generator=g,
        ).clamp(1e-8, 1.0)
        cp_gumbel = -torch.log(-torch.log(cp_gumbel))
    else:
        cp_gumbel = None
    with torch.no_grad():
        tokens = cp_unrolled(
            past_hidden,
            codec0,
            cp_gumbel_noise=cp_gumbel,
            temperature=torch.full((1, 1), temperature, device=device, dtype=torch.float32),
        )
    return tokens[0].detach().cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default="vivian")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")

    talker = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=args.device, attn_implementation="eager"
    )
    talker.eval()
    cp = talker.code_predictor
    cp_unrolled = CodePredictorUnrolled(cp, talker.model.codec_embedding, logits_topk=args.top_k).to(device).eval()

    past_hidden, codec0, plan, talker_logits = build_on_manifold_state(
        talker, args.text, args.speaker, args.language, device
    )

    print("=== On-manifold CP sampled parity ===")
    print("text:", args.text)
    print("codec0:", codec0.item())
    top5v, top5i = talker_logits.float().topk(5)
    print("talker top5:", list(zip(top5i[0].tolist(), [round(v, 4) for v in top5v[0].tolist()])))
    print("trailing_len:", len(plan.trailing))
    print()

    # Greedy sanity
    off_g = run_official_cp(cp, talker, past_hidden, codec0, False, args.top_k, args.top_p, args.temperature)
    unr_g = run_unrolled_cp(cp_unrolled, past_hidden, codec0, False, args.top_k, args.temperature, 0, device)
    print("Greedy official:", off_g)
    print("Greedy unrolled:", unr_g)
    print("Greedy match:", off_g == unr_g)
    print()

    # Sampled trials
    full_match = 0
    prefix_match_stats = []
    official_counter = [Counter() for _ in range(15)]
    unrolled_counter = [Counter() for _ in range(15)]

    for seed in range(args.trials):
        torch.manual_seed(seed)
        off = run_official_cp(cp, talker, past_hidden, codec0, True, args.top_k, args.top_p, args.temperature)
        unr = run_unrolled_cp(cp_unrolled, past_hidden, codec0, True, args.top_k, args.temperature, seed, device)
        if off == unr:
            full_match += 1
        # longest common prefix
        lcp = 0
        for a, b in zip(off, unr):
            if a == b:
                lcp += 1
            else:
                break
        prefix_match_stats.append(lcp)
        for i, tok in enumerate(off):
            official_counter[i][tok] += 1
        for i, tok in enumerate(unr):
            unrolled_counter[i][tok] += 1
        print(f"seed={seed:02d} lcp={lcp:2d} off={off} unr={unr}")

    print()
    print("=== Summary ===")
    print(f"full sequence exact match: {full_match}/{args.trials}")
    print(f"longest common prefix avg: {sum(prefix_match_stats)/len(prefix_match_stats):.2f}")
    print()
    print("Stage-wise top tokens (official vs unrolled):")
    for stage in range(15):
        off_top = official_counter[stage].most_common(5)
        unr_top = unrolled_counter[stage].most_common(5)
        print(f"stage {stage:02d} off={off_top} | unr={unr_top}")


if __name__ == "__main__":
    main()
