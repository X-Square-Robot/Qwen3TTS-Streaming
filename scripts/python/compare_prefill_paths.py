#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports("repo", "scripts_python", "third_party_qwen")

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
from engine.backend.prefill import (
    EmbeddingWeights,
    OFFICIAL_ASSISTANT_FMT,
    PrefillBuilder,
    TaskType,
)
from official_prefill import build_prefill_like_official


def tensor_diff(a: torch.Tensor, b: torch.Tensor):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    diff = (af - bf).abs()
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "max": float(diff.max().cpu()),
        "mean": float(diff.mean().cpu()),
        "cosine": float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", default="vivian")
    ap.add_argument("--language", default="auto")
    ap.add_argument(
        "--prompt-mode",
        choices=("supported", "raw"),
        default="supported",
        help="Use the official wrapper prompt (supported) or the bare core prompt (raw).",
    )
    args = ap.parse_args()

    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    device = torch.device("cuda:0")

    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = load_lightweight_tokenizer(model_dir)
    if args.prompt_mode == "supported":
        assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=args.text)
    else:
        assistant_text = f"<|im_start|>assistant\n{args.text}<|im_end|>"
    tok_out = tokenizer(assistant_text, return_tensors="pt")
    input_ids = torch.as_tensor(tok_out["input_ids"], device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    official_prefill, official_trailing = build_prefill_like_official(
        model,
        input_ids,
        args.language,
        args.speaker,
        device,
    )

    weights_dir = str(REPO_ROOT / "workspace/exported/custom-1.7b/weights")
    weights = EmbeddingWeights(weights_dir, device_id=device.index or 0)
    builder = PrefillBuilder(weights, tokenizer)
    text_ids = builder._encode_text_ids(args.text)
    plan = builder.build_plan_from_ids(
        task_type=TaskType.CUSTOM_VOICE,
        token_ids=text_ids,
        language=args.language,
        speaker=args.speaker,
        include_eos=True,
    )
    engine_prefill = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    engine_trailing = [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing]

    official_streaming_text_ids = input_ids[:, 4:-5].detach().cpu().tolist()
    engine_streaming_text_ids = [text_ids[1:]]

    print("prompt_mode:", args.prompt_mode)
    print("assistant_input_ids:", input_ids.detach().cpu().tolist())
    print("builder_text_ids:", text_ids)
    print("official_prefill_len:", official_prefill.shape[1], "engine_prefill_len:", engine_prefill.shape[1])
    print("official_trailing_len:", len(official_trailing), "engine_trailing_len:", len(engine_trailing))
    print("official_streaming_text_ids:", official_streaming_text_ids)
    print("engine_streaming_text_ids:", engine_streaming_text_ids)
    print("prefill_diff:", json.dumps(tensor_diff(official_prefill, engine_prefill), ensure_ascii=False))

    n_common = min(len(official_trailing), len(engine_trailing))
    for i in range(n_common):
        print(
            f"trailing_diff[{i}]:",
            json.dumps(tensor_diff(official_trailing[i], engine_trailing[i]), ensure_ascii=False),
        )
    if len(official_trailing) != len(engine_trailing):
        print("trailing_len_mismatch:", len(official_trailing), len(engine_trailing))


if __name__ == "__main__":
    main()
