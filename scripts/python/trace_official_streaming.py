#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports("repo", "third_party_qwen")

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
from engine.backend.prefill import OFFICIAL_ASSISTANT_FMT


def tensor_summary(x: torch.Tensor | None):
    if x is None:
        return None
    x = x.detach()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "device": str(x.device),
        "norm": float(x.float().norm().cpu()),
        "mean": float(x.float().mean().cpu()),
        "min": float(x.float().min().cpu()),
        "max": float(x.float().max().cpu()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", default="vivian")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--trace-start", type=int, default=6)
    ap.add_argument("--trace-end", type=int, default=8)
    args = ap.parse_args()

    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    device = "cuda:0"

    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device, attn_implementation="eager"
    )
    model.eval()

    tokenizer = load_lightweight_tokenizer(model_dir)
    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=args.text)
    out = tokenizer(assistant_text, return_tensors="pt")
    input_ids = torch.as_tensor(out["input_ids"], device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    trace = {
        "talker_forward": [],
        "cp_forward": [],
    }

    # We trace the high-level forward() methods by wrapping class methods on the instance.
    talker = model.talker
    cp = talker.code_predictor
    orig_talker_forward = talker.forward
    orig_cp_forward = cp.forward
    talker_sig = inspect.signature(orig_talker_forward)
    cp_sig = inspect.signature(orig_cp_forward)

    state = {"talker_call": 0, "cp_call": 0}

    def talker_forward_wrapper(*f_args, **f_kwargs):
        state["talker_call"] += 1
        cid = state["talker_call"]
        rec = {
            "call": cid,
            "input_ids": None if f_kwargs.get("input_ids") is None else f_kwargs["input_ids"].detach().cpu().tolist(),
            "inputs_embeds": tensor_summary(f_kwargs.get("inputs_embeds")),
            "past_hidden": tensor_summary(f_kwargs.get("past_hidden")),
            "generation_step": f_kwargs.get("generation_step"),
            "trailing_text_hidden": tensor_summary(f_kwargs.get("trailing_text_hidden")),
            "tts_pad_embed": tensor_summary(f_kwargs.get("tts_pad_embed")),
            "attention_mask": None if f_kwargs.get("attention_mask") is None else {
                "shape": list(f_kwargs["attention_mask"].shape),
                "sum": int(f_kwargs["attention_mask"].sum().item()),
                "last_row": f_kwargs["attention_mask"][0].detach().cpu().tolist()[-16:],
            },
            "position_ids": None if f_kwargs.get("position_ids") is None else {
                "shape": list(f_kwargs["position_ids"].shape),
                "last_values": f_kwargs["position_ids"][..., -min(8, f_kwargs["position_ids"].shape[-1]):].detach().cpu().tolist(),
            },
            "cache_position": None if f_kwargs.get("cache_position") is None else f_kwargs["cache_position"].detach().cpu().tolist(),
        }
        out = orig_talker_forward(*f_args, **f_kwargs)
        rec["out_generation_step"] = out.generation_step
        rec["out_logits"] = tensor_summary(out.logits)
        rec["out_past_hidden"] = tensor_summary(out.past_hidden)
        hs = out.hidden_states
        if isinstance(hs, tuple) and len(hs) == 2:
            rec["out_codec_ids"] = None if hs[1] is None else hs[1].detach().cpu().tolist()
        trace["talker_forward"].append(rec)
        return out

    def cp_forward_wrapper(*f_args, **f_kwargs):
        state["cp_call"] += 1
        cid = state["cp_call"]
        rec = {
            "call": cid,
            "input_ids": None if f_kwargs.get("input_ids") is None else f_kwargs["input_ids"].detach().cpu().tolist(),
            "inputs_embeds": tensor_summary(f_kwargs.get("inputs_embeds")),
            "generation_steps": f_kwargs.get("generation_steps"),
            "cache_position": None if f_kwargs.get("cache_position") is None else f_kwargs["cache_position"].detach().cpu().tolist(),
        }
        out = orig_cp_forward(*f_args, **f_kwargs)
        rec["out_generation_steps"] = out.generation_steps
        rec["out_logits"] = tensor_summary(out.logits)
        trace["cp_forward"].append(rec)
        return out

    talker_forward_wrapper.__signature__ = talker_sig
    cp_forward_wrapper.__signature__ = cp_sig
    talker.forward = talker_forward_wrapper
    cp.forward = cp_forward_wrapper

    torch.manual_seed(args.seed)
    with torch.no_grad():
        codes_list, _ = model.generate(
            input_ids=[input_ids],
            languages=[args.language],
            speakers=[args.speaker],
            non_streaming_mode=False,
            do_sample=True,
            subtalker_dosample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            max_new_tokens=args.max_steps,
        )

    out_path = REPO_ROOT / "workspace" / "official_trace.json"
    with open(out_path, "w") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    print(f"saved trace to: {out_path}")
    print(f"talker forward calls: {len(trace['talker_forward'])}")
    print(f"cp forward calls: {len(trace['cp_forward'])}")
    for rec in trace["talker_forward"][: min(12, len(trace['talker_forward']))]:
        cid = rec['call']
        print(f"call={cid} gen_step={rec['generation_step']} -> out_gen_step={rec['out_generation_step']} input_ids={rec['input_ids']} out_codec_ids={rec.get('out_codec_ids')}")


if __name__ == '__main__':
    main()
