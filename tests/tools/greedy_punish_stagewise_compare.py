#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Qwen3-TTS"))
sys.path.insert(0, str(REPO_ROOT))

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
from engine.backend.prefill import EmbeddingWeights, OFFICIAL_ASSISTANT_FMT, PrefillBuilder, TaskType


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    diff = (af - bf).abs()
    return {
        "max": float(diff.max().cpu()),
        "mean": float(diff.mean().cpu()),
        "cosine": float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()),
    }


def topk_summary(scores: torch.Tensor, k: int = 10) -> list[list[float]]:
    vals, idx = scores.detach().float().topk(k)
    return [[int(i), float(v)] for i, v in zip(idx[0].cpu().tolist(), vals[0].cpu().tolist())]


def top2_margin(scores: torch.Tensor) -> float:
    vals = scores.detach().float().topk(2).values
    return float((vals[0, 0] - vals[0, 1]).cpu())


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


def build_engine_plan(text: str, device: torch.device):
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
    return plan, weights, tokenizer


def run_official_trace(model, text: str, max_steps: int, repetition_penalty: float):
    device = next(model.parameters()).device
    tokenizer = load_lightweight_tokenizer(str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=text)
    out = tokenizer(assistant_text, return_tensors="pt")
    input_ids = torch.as_tensor(out["input_ids"], device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    talker = model.talker
    cp = talker.code_predictor
    orig_talker_forward = talker.forward
    orig_cp_forward = cp.forward
    talker_sig = inspect.signature(orig_talker_forward)
    cp_sig = inspect.signature(orig_cp_forward)

    cp_records: list[dict] = []
    talker_records: list[dict] = []
    state = {"cp_calls": 0, "talker_calls": 0}

    def cp_forward_wrapper(*f_args, **f_kwargs):
        state["cp_calls"] += 1
        out_cp = orig_cp_forward(*f_args, **f_kwargs)
        logits = out_cp.logits[:, -1, :].detach().float().cpu()
        cp_records.append(
            {
                "call": state["cp_calls"],
                "stage": int(out_cp.generation_steps) - 1,
                "input_ids": None if f_kwargs.get("input_ids") is None else f_kwargs["input_ids"].detach().cpu(),
                "raw_logits": logits,
                "token": int(logits.argmax(dim=-1).item()),
                "top10": topk_summary(logits),
                "top2_margin": top2_margin(logits),
            }
        )
        return out_cp

    def talker_forward_wrapper(*f_args, **f_kwargs):
        state["talker_calls"] += 1
        call_id = state["talker_calls"]
        cp_start = state["cp_calls"]
        out_t = orig_talker_forward(*f_args, **f_kwargs)
        cp_end = state["cp_calls"]
        raw_logits = out_t.logits[:, -1, :].detach().float().cpu()
        talker_records.append(
            {
                "call": call_id,
                "generation_step_in": f_kwargs.get("generation_step"),
                "input_ids": None if f_kwargs.get("input_ids") is None else f_kwargs["input_ids"].detach().cpu(),
                "past_hidden_in": None if f_kwargs.get("past_hidden") is None else f_kwargs["past_hidden"].detach().float().cpu(),
                "raw_logits": raw_logits,
                "past_hidden_out": out_t.past_hidden.detach().float().cpu(),
                "codec_ids": None if out_t.hidden_states[1] is None else out_t.hidden_states[1].detach().cpu(),
                "cp_records": cp_records[cp_start:cp_end],
            }
        )
        return out_t

    talker_forward_wrapper.__signature__ = talker_sig
    cp_forward_wrapper.__signature__ = cp_sig
    talker.forward = talker_forward_wrapper
    cp.forward = cp_forward_wrapper
    try:
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
    finally:
        talker.forward = orig_talker_forward
        cp.forward = orig_cp_forward

    codes = codes_list[0].detach().cpu()
    talker_tokens = [int(row[0].item()) for row in codes]
    if len(talker_records) != len(talker_tokens):
        print(
            f"warning: talker forward calls ({len(talker_records)}) != returned talker tokens ({len(talker_tokens)}); truncating to common prefix",
            file=sys.stderr,
        )
        keep = min(len(talker_records), len(talker_tokens))
        talker_records = talker_records[:keep]
        talker_tokens = talker_tokens[:keep]

    suppress_tokens = [
        i
        for i in range(talker.config.vocab_size - 1024, talker.config.vocab_size)
        if i not in (talker.config.codec_eos_token_id,)
    ]

    history = []
    for step, rec in enumerate(talker_records):
        hist = torch.tensor([history], dtype=torch.long)
        processed = apply_talker_processors(rec["raw_logits"], hist, suppress_tokens, repetition_penalty)
        rec["processed_logits"] = processed.cpu()
        rec["selected_token"] = talker_tokens[step]
        rec["processed_top10"] = topk_summary(processed)
        rec["processed_top2_margin"] = top2_margin(processed)
        history.append(talker_tokens[step])

    return {
        "talker_records": talker_records,
        "talker_tokens": talker_tokens,
        "codec_rows": codes,
        "suppress_tokens": suppress_tokens,
    }


def run_manual_trace(model, text: str, max_steps: int, repetition_penalty: float):
    device = next(model.parameters()).device
    talker = model.talker
    cp = talker.code_predictor
    plan, weights, _ = build_engine_plan(text, device)
    prefill_embeds = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    trailing = [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing]
    pad_embed = weights.tts_pad_embed.to(device=device, dtype=torch.bfloat16)

    suppress_tokens = [
        i
        for i in range(talker.config.vocab_size - 1024, talker.config.vocab_size)
        if i not in (talker.config.codec_eos_token_id,)
    ]

    talker_records: list[dict] = []
    history = torch.empty((1, 0), device=device, dtype=torch.long)

    with torch.no_grad():
        prefill_out = talker.model(inputs_embeds=prefill_embeds, use_cache=True, return_dict=True)
    kv = prefill_out.past_key_values
    past_hidden = prefill_out.last_hidden_state[:, -1:, :]
    raw_logits = talker.codec_head(prefill_out.last_hidden_state)[:, -1, :]
    processed = apply_talker_processors(raw_logits, history, suppress_tokens, repetition_penalty)
    codec0 = processed.argmax(dim=-1)
    history = torch.cat([history, codec0.unsqueeze(1)], dim=1)
    talker_records.append(
        {
            "call": 1,
            "phase": "prefill",
            "input_ids": None,
            "past_hidden_in": None,
            "past_hidden_out": past_hidden.detach().float().cpu(),
            "raw_logits": raw_logits.detach().float().cpu(),
            "processed_logits": processed.detach().float().cpu(),
            "selected_token": int(codec0.item()),
            "cp_records": [],
        }
    )

    for step in range(max_steps - 1):
        cp_records = []
        with torch.no_grad():
            cp_input = torch.cat((past_hidden, talker.model.codec_embedding(codec0).unsqueeze(1)), dim=1)
            cp_out = cp.forward(
                inputs_embeds=cp_input,
                use_cache=True,
                return_dict=True,
            )
        cp_logits = cp_out.logits[:, -1, :]
        cp_token = cp_logits.argmax(dim=-1)
        cp_records.append(
            {
                "stage": 0,
                "raw_logits": cp_logits.detach().float().cpu(),
                "token": int(cp_token.item()),
                "top10": topk_summary(cp_logits),
                "top2_margin": top2_margin(cp_logits),
            }
        )
        cp_kv = cp_out.past_key_values
        cp_generation_steps = cp_out.generation_steps
        cp_tokens = [cp_token]

        for stage in range(1, talker.config.num_code_groups - 1):
            with torch.no_grad():
                cp_out = cp.forward(
                    input_ids=cp_token.unsqueeze(1),
                    past_key_values=cp_kv,
                    generation_steps=cp_generation_steps,
                    use_cache=True,
                    return_dict=True,
                )
            cp_logits = cp_out.logits[:, -1, :]
            cp_token = cp_logits.argmax(dim=-1)
            cp_kv = cp_out.past_key_values
            cp_generation_steps = cp_out.generation_steps
            cp_tokens.append(cp_token)
            cp_records.append(
                {
                    "stage": stage,
                    "raw_logits": cp_logits.detach().float().cpu(),
                    "token": int(cp_token.item()),
                    "top10": topk_summary(cp_logits),
                    "top2_margin": top2_margin(cp_logits),
                }
            )

        last_id_hidden = talker.get_input_embeddings()(codec0.unsqueeze(1))
        codec_hiddens = torch.cat(
            [last_id_hidden]
            + [talker.code_predictor.get_input_embeddings()[i](tok.unsqueeze(1)) for i, tok in enumerate(cp_tokens)],
            dim=1,
        )
        next_input = codec_hiddens.sum(1, keepdim=True)
        phase = "text" if step < len(trailing) else "pad"
        next_input = next_input + (trailing[step] if step < len(trailing) else pad_embed)

        with torch.no_grad():
            talker_out = talker.model(inputs_embeds=next_input, past_key_values=kv, use_cache=True, return_dict=True)
        kv = talker_out.past_key_values
        past_hidden_in = past_hidden
        past_hidden = talker_out.last_hidden_state[:, -1:, :]
        raw_logits = talker.codec_head(past_hidden)[:, -1, :]
        processed = apply_talker_processors(raw_logits, history, suppress_tokens, repetition_penalty)
        codec0 = processed.argmax(dim=-1)
        history = torch.cat([history, codec0.unsqueeze(1)], dim=1)

        talker_records.append(
            {
                "call": step + 2,
                "phase": phase,
                "input_ids": history[:, -2:-1].detach().cpu(),
                "past_hidden_in": past_hidden_in.detach().float().cpu(),
                "past_hidden_out": past_hidden.detach().float().cpu(),
                "raw_logits": raw_logits.detach().float().cpu(),
                "processed_logits": processed.detach().float().cpu(),
                "selected_token": int(codec0.item()),
                "cp_records": cp_records,
            }
        )
        if int(codec0.item()) == talker.config.codec_eos_token_id:
            break

    return {
        "talker_records": talker_records,
        "talker_tokens": [rec["selected_token"] for rec in talker_records],
        "suppress_tokens": suppress_tokens,
    }


def stage_summary(official_stage: dict, manual_stage: dict) -> dict:
    return {
        "stage": official_stage["stage"],
        "official_token": official_stage["token"],
        "manual_token": manual_stage["token"],
        "logits_diff": tensor_diff(official_stage["raw_logits"], manual_stage["raw_logits"]),
        "official_top10": official_stage["top10"],
        "manual_top10": manual_stage["top10"],
        "official_top2_margin": official_stage["top2_margin"],
        "manual_top2_margin": manual_stage["top2_margin"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--out-json", default=str(REPO_ROOT / "workspace" / "greedy_punish_stagewise_summary.json"))
    args = ap.parse_args()

    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager"
    )
    model.eval()

    official = run_official_trace(model, args.text, args.max_steps, args.repetition_penalty)
    manual = run_manual_trace(model, args.text, args.max_steps, args.repetition_penalty)

    official_talker = official["talker_tokens"]
    manual_talker = manual["talker_tokens"]
    n = min(len(official_talker), len(manual_talker))
    first_div = -1
    for i in range(n):
        if official_talker[i] != manual_talker[i]:
            first_div = i
            break

    summary: dict[str, object] = {
        "official_talker": official_talker[:n],
        "manual_talker": manual_talker[:n],
        "first_talker_divergence": first_div,
        "talker_steps": [],
    }

    for step in range(n):
        off_rec = official["talker_records"][step]
        man_rec = manual["talker_records"][step]
        item = {
            "step": step,
            "official_token": off_rec["selected_token"],
            "manual_token": man_rec["selected_token"],
            "raw_logits_diff": tensor_diff(off_rec["raw_logits"], man_rec["raw_logits"]),
            "processed_logits_diff": tensor_diff(off_rec["processed_logits"], man_rec["processed_logits"]),
            "official_processed_top10": topk_summary(off_rec["processed_logits"]),
            "manual_processed_top10": topk_summary(man_rec["processed_logits"]),
            "official_processed_top2_margin": top2_margin(off_rec["processed_logits"]),
            "manual_processed_top2_margin": top2_margin(man_rec["processed_logits"]),
            "cp_stage_summaries": [],
        }
        if step > 0:
            item["past_hidden_in_diff"] = tensor_diff(off_rec["past_hidden_in"], man_rec["past_hidden_in"])
            talker = model.talker
            talker_device = next(talker.parameters()).device
            talker_dtype = next(talker.parameters()).dtype
            off_cp_entry = talker.code_predictor.small_to_mtp_projection(
                torch.cat(
                    (
                        off_rec["past_hidden_in"].to(talker_device, dtype=talker_dtype),
                        talker.model.codec_embedding(off_rec["input_ids"].to(talker_device).squeeze(-1)).unsqueeze(1),
                    ),
                    dim=1,
                )
            ).detach().float().cpu()
            man_cp_entry = talker.code_predictor.small_to_mtp_projection(
                torch.cat(
                    (
                        man_rec["past_hidden_in"].to(talker_device, dtype=talker_dtype),
                        talker.model.codec_embedding(man_rec["input_ids"].to(talker_device).squeeze(-1)).unsqueeze(1),
                    ),
                    dim=1,
                )
            ).detach().float().cpu()
            item["cp_entry_diff"] = tensor_diff(off_cp_entry, man_cp_entry)

            off_cp = off_rec["cp_records"]
            man_cp = man_rec["cp_records"]
            cp_first_div = -1
            for stage_idx, (off_stage, man_stage) in enumerate(zip(off_cp, man_cp)):
                stage_item = stage_summary(off_stage, man_stage)
                item["cp_stage_summaries"].append(stage_item)
                if cp_first_div < 0 and off_stage["token"] != man_stage["token"]:
                    cp_first_div = stage_idx
            item["first_cp_stage_divergence"] = cp_first_div

        summary["talker_steps"].append(item)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("official talker:", official_talker[:n])
    print("manual   talker:", manual_talker[:n])
    print("first_talker_divergence:", first_div)

    if first_div >= 0:
        off = summary["talker_steps"][first_div]
        print("talker step summary:", json.dumps({
            "step": off["step"],
            "official_token": off["official_token"],
            "manual_token": off["manual_token"],
            "raw_logits_diff": off["raw_logits_diff"],
            "processed_logits_diff": off["processed_logits_diff"],
            "official_processed_top10": off["official_processed_top10"][:5],
            "manual_processed_top10": off["manual_processed_top10"][:5],
            "first_cp_stage_divergence": off.get("first_cp_stage_divergence", -1),
            "cp_entry_diff": off.get("cp_entry_diff"),
        }, ensure_ascii=False, indent=2))
        cp_stage_div = off.get("first_cp_stage_divergence", -1)
        if cp_stage_div is not None and cp_stage_div >= 0:
            stage_item = off["cp_stage_summaries"][cp_stage_div]
            print("first divergent cp stage:", json.dumps(stage_item, ensure_ascii=False, indent=2))

    print("saved summary to:", out_path)


if __name__ == "__main__":
    main()
