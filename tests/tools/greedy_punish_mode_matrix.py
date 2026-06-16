#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "python"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Qwen3-TTS"))
sys.path.insert(0, str(REPO_ROOT))

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
from engine.backend.prefill import EmbeddingWeights, OFFICIAL_ASSISTANT_FMT, PrefillBuilder, TaskType
from official_prefill import build_prefill_like_official


def apply_talker_processors(
    raw_logits: torch.Tensor,
    history_ids: torch.Tensor,
    suppress_tokens: list[int],
    repetition_penalty: float,
) -> torch.Tensor:
    scores = raw_logits.detach().float().clone()
    if suppress_tokens:
        scores[..., suppress_tokens] = -1e9
    if repetition_penalty != 1.0:
        scores = RepetitionPenaltyLogitsProcessor(repetition_penalty)(history_ids, scores)
    return scores


def find_first_divergence(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def build_official_prefill(model, text: str, device: torch.device):
    tokenizer = load_lightweight_tokenizer(str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=text)
    out = tokenizer(assistant_text, return_tensors="pt")
    input_ids = torch.as_tensor(out["input_ids"], device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    prefill_embeds, trailing_list = build_prefill_like_official(
        model, input_ids, "auto", "vivian", device
    )

    talker = model.talker
    pad_id = torch.tensor([[model.config.tts_pad_token_id]], device=device, dtype=torch.long)
    with torch.no_grad():
        pad_embed = talker.text_projection(talker.model.text_embedding(pad_id))

    return {
        "prefill_embeds": prefill_embeds.to(device=device, dtype=torch.bfloat16),
        "trailing_list": [t.to(device=device, dtype=torch.bfloat16) for t in trailing_list],
        "pad_embed": pad_embed.to(device=device, dtype=torch.bfloat16),
    }


def build_engine_prefill(text: str, device: torch.device):
    tokenizer = load_lightweight_tokenizer(str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    weights_dir = str(REPO_ROOT / "workspace/exported/custom-1.7b/weights")
    weights = EmbeddingWeights(weights_dir, device_id=device.index or 0)
    builder = PrefillBuilder(weights, tokenizer)
    token_ids = builder._encode_text_ids(text)
    plan = builder.build_plan_from_ids(
        task_type=TaskType.CUSTOM_VOICE,
        token_ids=token_ids,
        language="auto",
        speaker="vivian",
        include_eos=True,
    )
    return {
        "prefill_embeds": plan.prefill_embeds.to(device=device, dtype=torch.bfloat16),
        "trailing_list": [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing],
        "pad_embed": weights.tts_pad_embed.to(device=device, dtype=torch.bfloat16),
    }


def build_engine_prefill_with_live_specials(model, text: str, device: torch.device):
    tokenizer = load_lightweight_tokenizer(str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    weights_dir = str(REPO_ROOT / "workspace/exported/custom-1.7b/weights")
    weights = EmbeddingWeights(weights_dir, device_id=device.index or 0)
    talker = model.talker
    special_ids = torch.tensor(
        [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
        device=device,
        dtype=torch.long,
    )
    with torch.no_grad():
        live_bos, live_eos, live_pad = talker.text_projection(
            talker.get_text_embeddings()(special_ids)
        ).chunk(3, dim=1)
    weights.tts_bos_embed = live_bos.to(device=device, dtype=torch.bfloat16)
    weights.tts_eos_embed = live_eos.to(device=device, dtype=torch.bfloat16)
    weights.tts_pad_embed = live_pad.to(device=device, dtype=torch.bfloat16)

    builder = PrefillBuilder(weights, tokenizer)
    token_ids = builder._encode_text_ids(text)
    plan = builder.build_plan_from_ids(
        task_type=TaskType.CUSTOM_VOICE,
        token_ids=token_ids,
        language="auto",
        speaker="vivian",
        include_eos=True,
    )
    return {
        "prefill_embeds": plan.prefill_embeds.to(device=device, dtype=torch.bfloat16),
        "trailing_list": [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing],
        "pad_embed": weights.tts_pad_embed.to(device=device, dtype=torch.bfloat16),
    }


def run_official_generate(model, text: str, max_steps: int, repetition_penalty: float) -> list[int]:
    tokenizer = load_lightweight_tokenizer(str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=text)
    input_ids = tokenizer(assistant_text, return_tensors="pt")["input_ids"]
    input_ids = torch.as_tensor(input_ids, device=next(model.parameters()).device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    model.talker.rope_deltas = None
    with torch.no_grad():
        codes_list, _ = model.generate(
            input_ids=[input_ids],
            languages=["auto"],
            speakers=["vivian"],
            non_streaming_mode=False,
            do_sample=False,
            subtalker_dosample=False,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_steps,
        )
    codes = codes_list[0].detach().cpu().tolist()
    return [row[0] for row in codes]


def run_talker_forward_stepwise(
    model,
    *,
    prefill_embeds: torch.Tensor,
    trailing_list: list[torch.Tensor],
    pad_embed: torch.Tensor,
    max_steps: int,
    repetition_penalty: float,
) -> list[int]:
    device = next(model.parameters()).device
    talker = model.talker
    suppress_tokens = [
        i
        for i in range(talker.config.vocab_size - 1024, talker.config.vocab_size)
        if i not in (talker.config.codec_eos_token_id,)
    ]

    talker.rope_deltas = None
    trailing_text_hidden = (
        torch.cat(trailing_list, dim=1)
        if trailing_list
        else torch.zeros((1, 0, prefill_embeds.shape[-1]), device=device, dtype=torch.bfloat16)
    )

    original_len = prefill_embeds.shape[1]
    attention_mask = torch.ones(1, original_len, device=device, dtype=torch.long)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    cache_position = torch.arange(original_len, device=device, dtype=torch.long)

    history = torch.empty((1, 0), device=device, dtype=torch.long)
    tokens: list[int] = []

    with torch.no_grad():
        current = talker.forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=prefill_embeds,
            use_cache=True,
            cache_position=cache_position,
            generation_step=None,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=pad_embed,
            subtalker_dosample=False,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            return_dict=True,
        )
    scores = apply_talker_processors(
        current.logits[:, -1, :], history, suppress_tokens, repetition_penalty
    )
    token = scores.argmax(dim=-1)
    history = torch.cat([history, token.unsqueeze(1)], dim=1)
    tokens.append(int(token.item()))

    for step in range(1, max_steps):
        attention_mask = torch.ones(1, original_len + step, device=device, dtype=torch.long)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -1:].clone()
        cache_position = torch.tensor([original_len + step - 1], device=device, dtype=torch.long)

        with torch.no_grad():
            current = talker.forward(
                input_ids=token.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=current.past_key_values,
                use_cache=True,
                cache_position=cache_position,
                generation_step=current.generation_step,
                trailing_text_hidden=current.trailing_text_hidden,
                tts_pad_embed=current.tts_pad_embed,
                past_hidden=current.past_hidden,
                subtalker_dosample=False,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                subtalker_temperature=0.9,
                return_dict=True,
            )
        scores = apply_talker_processors(
            current.logits[:, -1, :], history, suppress_tokens, repetition_penalty
        )
        token = scores.argmax(dim=-1)
        history = torch.cat([history, token.unsqueeze(1)], dim=1)
        tokens.append(int(token.item()))
        if int(token.item()) == talker.config.codec_eos_token_id:
            break

    return tokens


def run_engine_manual(
    model,
    *,
    prefill_embeds: torch.Tensor,
    trailing_list: list[torch.Tensor],
    pad_embed: torch.Tensor,
    max_steps: int,
    repetition_penalty: float,
) -> list[int]:
    device = next(model.parameters()).device
    talker = model.talker
    cp = talker.code_predictor
    suppress_tokens = [
        i
        for i in range(talker.config.vocab_size - 1024, talker.config.vocab_size)
        if i not in (talker.config.codec_eos_token_id,)
    ]

    history = torch.empty((1, 0), device=device, dtype=torch.long)
    with torch.no_grad():
        out = talker.model(inputs_embeds=prefill_embeds, use_cache=True, return_dict=True)
    kv = out.past_key_values
    past_hidden = out.last_hidden_state[:, -1:, :]
    scores = apply_talker_processors(
        talker.codec_head(out.last_hidden_state)[:, -1, :], history, suppress_tokens, repetition_penalty
    )
    codec0 = scores.argmax(dim=-1)
    history = torch.cat([history, codec0.unsqueeze(1)], dim=1)

    tokens = [int(codec0.item())]
    for step in range(max_steps - 1):
        with torch.no_grad():
            pred = cp.generate(
                inputs_embeds=torch.cat((past_hidden, talker.model.codec_embedding(codec0).unsqueeze(1)), dim=1),
                max_new_tokens=talker.config.num_code_groups - 1,
                do_sample=False,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        last_id_hidden = talker.get_input_embeddings()(codec0.unsqueeze(1))
        codec_hiddens = torch.cat(
            [last_id_hidden]
            + [
                talker.code_predictor.get_input_embeddings()[i](pred.sequences[..., i : i + 1])
                for i in range(talker.config.num_code_groups - 1)
            ],
            dim=1,
        )
        next_input = codec_hiddens.sum(1, keepdim=True) + (
            trailing_list[step] if step < len(trailing_list) else pad_embed
        )
        with torch.no_grad():
            out = talker.model(inputs_embeds=next_input, past_key_values=kv, use_cache=True, return_dict=True)
        kv = out.past_key_values
        past_hidden = out.last_hidden_state[:, -1:, :]
        scores = apply_talker_processors(
            talker.codec_head(past_hidden)[:, -1, :], history, suppress_tokens, repetition_penalty
        )
        codec0 = scores.argmax(dim=-1)
        history = torch.cat([history, codec0.unsqueeze(1)], dim=1)
        tokens.append(int(codec0.item()))
        if int(codec0.item()) == talker.config.codec_eos_token_id:
            break

    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument(
        "--out-json",
        default=str(REPO_ROOT / "workspace" / "greedy_punish_mode_matrix.json"),
    )
    args = ap.parse_args()

    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager"
    )
    model.eval()

    device = next(model.parameters()).device
    official_prefill = build_official_prefill(model, args.text, device)
    engine_prefill = build_engine_prefill(args.text, device)
    engine_prefill_live_specials = build_engine_prefill_with_live_specials(model, args.text, device)

    results = {
        "official_generate": run_official_generate(
            model, args.text, args.max_steps, args.repetition_penalty
        ),
        "official_stepwise": run_talker_forward_stepwise(
            model,
            prefill_embeds=official_prefill["prefill_embeds"],
            trailing_list=official_prefill["trailing_list"],
            pad_embed=official_prefill["pad_embed"],
            max_steps=args.max_steps,
            repetition_penalty=args.repetition_penalty,
        ),
        "engine_stepwise": run_talker_forward_stepwise(
            model,
            prefill_embeds=engine_prefill["prefill_embeds"],
            trailing_list=engine_prefill["trailing_list"],
            pad_embed=engine_prefill["pad_embed"],
            max_steps=args.max_steps,
            repetition_penalty=args.repetition_penalty,
        ),
        "engine_stepwise_live_specials": run_talker_forward_stepwise(
            model,
            prefill_embeds=engine_prefill_live_specials["prefill_embeds"],
            trailing_list=engine_prefill_live_specials["trailing_list"],
            pad_embed=engine_prefill_live_specials["pad_embed"],
            max_steps=args.max_steps,
            repetition_penalty=args.repetition_penalty,
        ),
        "engine_manual": run_engine_manual(
            model,
            prefill_embeds=engine_prefill["prefill_embeds"],
            trailing_list=engine_prefill["trailing_list"],
            pad_embed=engine_prefill["pad_embed"],
            max_steps=args.max_steps,
            repetition_penalty=args.repetition_penalty,
        ),
        "engine_manual_live_specials": run_engine_manual(
            model,
            prefill_embeds=engine_prefill_live_specials["prefill_embeds"],
            trailing_list=engine_prefill_live_specials["trailing_list"],
            pad_embed=engine_prefill_live_specials["pad_embed"],
            max_steps=args.max_steps,
            repetition_penalty=args.repetition_penalty,
        ),
    }

    names = list(results.keys())
    comparisons = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lhs = names[i]
            rhs = names[j]
            comparisons.append(
                {
                    "lhs": lhs,
                    "rhs": rhs,
                    "first_divergence": find_first_divergence(results[lhs], results[rhs]),
                    "lhs_prefix": results[lhs][:12],
                    "rhs_prefix": results[rhs][:12],
                }
            )

    summary = {"results": results, "comparisons": comparisons}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    for name in names:
        print(f"{name}: {results[name][:12]}")
    print("--- comparisons ---")
    for item in comparisons:
        print(
            f"{item['lhs']} vs {item['rhs']}: first_divergence={item['first_divergence']}"
        )
    print("saved summary to:", out_path)


if __name__ == "__main__":
    main()
