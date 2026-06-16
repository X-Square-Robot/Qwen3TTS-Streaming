#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from qwen3tts_tools.common import REPO_ROOT, bootstrap_project_imports

bootstrap_project_imports("repo", "scripts_python", "third_party_qwen")

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer
from engine.backend.prefill import EmbeddingWeights, OFFICIAL_ASSISTANT_FMT, PrefillBuilder, TaskType
from official_prefill import build_prefill_like_official


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | list[int]]:
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


def topk_summary(scores: torch.Tensor, k: int = 10) -> list[list[float]]:
    vals, idx = scores.detach().float().topk(k)
    return [[int(i), float(v)] for i, v in zip(idx[0].cpu().tolist(), vals[0].cpu().tolist())]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument(
        "--out-json",
        default=str(REPO_ROOT / "workspace" / "live_vs_exported_prefill.json"),
    )
    args = ap.parse_args()

    model_dir = str(REPO_ROOT / "workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    device = torch.device("cuda:0")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager"
    )
    model.eval()
    talker = model.talker

    tokenizer = load_lightweight_tokenizer(model_dir)
    weights = EmbeddingWeights(str(REPO_ROOT / "workspace/exported/custom-1.7b/weights"), device_id=0)
    builder = PrefillBuilder(weights, tokenizer)
    raw_special = torch.load(
        REPO_ROOT / "workspace/exported/custom-1.7b/weights/special_embeddings.pt",
        map_location=device,
        weights_only=True,
    )

    assistant_text = OFFICIAL_ASSISTANT_FMT.format(text=args.text)
    input_ids = tokenizer(assistant_text, return_tensors="pt")["input_ids"]
    input_ids = torch.as_tensor(input_ids, device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    text_ids = builder._encode_text_ids(args.text)
    text_ids_tensor = torch.tensor([text_ids], device=device, dtype=torch.long)

    official_prefill, official_trailing = build_prefill_like_official(
        model, input_ids, "auto", "vivian", device
    )
    plan = builder.build_plan_from_ids(
        task_type=TaskType.CUSTOM_VOICE,
        token_ids=text_ids,
        language="auto",
        speaker="vivian",
        include_eos=True,
    )
    engine_prefill = plan.prefill_embeds.to(device=device, dtype=torch.bfloat16)
    engine_trailing = [t.to(device=device, dtype=torch.bfloat16) for t in plan.trailing]

    assistant_role_ids = input_ids[:, :3]
    official_role = talker.text_projection(talker.get_text_embeddings()(assistant_role_ids))
    engine_role = weights.text_embed(assistant_role_ids)

    official_text = talker.text_projection(talker.get_text_embeddings()(text_ids_tensor))
    engine_text = weights.text_embed(text_ids_tensor)

    special_ids = torch.tensor(
        [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
        device=device,
        dtype=torch.long,
    )
    official_special = talker.text_projection(talker.get_text_embeddings()(special_ids))
    runtime_special = torch.cat([weights.tts_bos_embed, weights.tts_eos_embed, weights.tts_pad_embed], dim=1)
    exported_special = torch.cat(
        [
            raw_special["tts_bos_embed"].to(device=device, dtype=torch.bfloat16),
            raw_special["tts_eos_embed"].to(device=device, dtype=torch.bfloat16),
            raw_special["tts_pad_embed"].to(device=device, dtype=torch.bfloat16),
        ],
        dim=1,
    )

    tc = model.config.talker_config
    speaker_id = tc.spk_id["vivian"]
    codec_prefill_ids = torch.tensor(
        [[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id, speaker_id, tc.codec_pad_id, tc.codec_bos_id]],
        device=device,
        dtype=torch.long,
    )
    official_codec = talker.get_input_embeddings()(codec_prefill_ids)
    engine_codec = weights.codec_embed(codec_prefill_ids)

    suppress_tokens = [
        i
        for i in range(talker.config.vocab_size - 1024, talker.config.vocab_size)
        if i not in (talker.config.codec_eos_token_id,)
    ]
    history = torch.empty((1, 0), device=device, dtype=torch.long)

    with torch.no_grad():
        official_out = talker.model(inputs_embeds=official_prefill.to(device=device, dtype=torch.bfloat16), use_cache=True, return_dict=True)
        engine_out = talker.model(inputs_embeds=engine_prefill, use_cache=True, return_dict=True)
    official_raw = talker.codec_head(official_out.last_hidden_state)[:, -1, :]
    engine_raw = talker.codec_head(engine_out.last_hidden_state)[:, -1, :]
    official_processed = apply_talker_processors(official_raw, history, suppress_tokens, args.repetition_penalty)
    engine_processed = apply_talker_processors(engine_raw, history, suppress_tokens, args.repetition_penalty)

    summary = {
        "text_ids": text_ids,
        "component_diffs": {
            "assistant_role": tensor_diff(official_role, engine_role),
            "text_embed_full": tensor_diff(official_text, engine_text),
            "tts_specials_exported_file": tensor_diff(official_special, exported_special),
            "tts_specials_runtime": tensor_diff(official_special, runtime_special),
            "codec_prefill_stack": tensor_diff(official_codec, engine_codec),
        },
        "prefill_diff": tensor_diff(official_prefill, engine_prefill),
        "trailing_diffs": [tensor_diff(a, b) for a, b in zip(official_trailing, engine_trailing)],
        "prefill_forward": {
            "past_hidden_diff": tensor_diff(
                official_out.last_hidden_state[:, -1:, :], engine_out.last_hidden_state[:, -1:, :]
            ),
            "raw_logits_diff": tensor_diff(official_raw, engine_raw),
            "processed_logits_diff": tensor_diff(official_processed, engine_processed),
            "official_token0": int(official_processed.argmax(dim=-1).item()),
            "engine_token0": int(engine_processed.argmax(dim=-1).item()),
            "official_top10": topk_summary(official_processed),
            "engine_top10": topk_summary(engine_processed),
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("component assistant_role:", json.dumps(summary["component_diffs"]["assistant_role"], ensure_ascii=False))
    print("component text_embed_full:", json.dumps(summary["component_diffs"]["text_embed_full"], ensure_ascii=False))
    print("component tts_specials_exported_file:", json.dumps(summary["component_diffs"]["tts_specials_exported_file"], ensure_ascii=False))
    print("component tts_specials_runtime:", json.dumps(summary["component_diffs"]["tts_specials_runtime"], ensure_ascii=False))
    print("component codec_prefill_stack:", json.dumps(summary["component_diffs"]["codec_prefill_stack"], ensure_ascii=False))
    print("prefill_diff:", json.dumps(summary["prefill_diff"], ensure_ascii=False))
    print("prefill_forward past_hidden_diff:", json.dumps(summary["prefill_forward"]["past_hidden_diff"], ensure_ascii=False))
    print("prefill_forward processed_logits_diff:", json.dumps(summary["prefill_forward"]["processed_logits_diff"], ensure_ascii=False))
    print("prefill_forward token0:", summary["prefill_forward"]["official_token0"], summary["prefill_forward"]["engine_token0"])
    print("saved summary to:", out_path)


if __name__ == "__main__":
    main()
