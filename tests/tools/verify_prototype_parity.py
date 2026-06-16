#!/usr/bin/env python3
"""
Prototype parity verification: compare official model.generate(greedy) output with
manual decode loop (PrefillBuilder prefill + Talker→CP→codec_sum→Talker) step-by-step.

Ensures three-layer consistency: prototype == manual PyTorch == ONNX/TRT.

Usage (host, conda activate qwen3-tts):
  python tests/tools/verify_prototype_parity.py --variant design-1.7b --text "你好，世界"
  python tests/tools/verify_prototype_parity.py --variant design-1.7b --text "你好" --max-steps 100
"""

import argparse
import json
import logging
import sys

import numpy as np
import torch

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

REPO_ROOT = bootstrap_tool_imports()
from qwen3tts_tools.common import bootstrap_project_imports

bootstrap_project_imports("scripts_python", "scripts", "scripts_export", "third_party_qwen")

from utils import (
    setup_logging,
    resolve_model_path,
    load_tts_model,
    resolve_device,
    resolve_tokenizer_path,
    DEFAULT_OUTPUT_DIR,
    has_model_weights,
)
from official_prefill import build_prefill_like_official

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("proto_parity")


def run_prototype_generate(model, input_ids, language, speaker, max_new_tokens, device):
    """Run official model.generate() with greedy settings. Returns [T, 16] codec tokens."""
    input_ids = input_ids.to(device)
    with torch.no_grad():
        talker_codes_list, _ = model.generate(
            input_ids=[input_ids],
            languages=[language],
            speakers=[speaker] if speaker else [""],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
        )
    # talker_codes_list: list of [T_i, 16] per batch item
    codes = talker_codes_list[0]
    if isinstance(codes, torch.Tensor):
        return codes.cpu().numpy()
    return np.array(codes, dtype=np.int64)


def run_manual_decode_loop(
    model, prefill_embeds, trailing_list, pad_embed, max_steps, codec_eos_id, device
):
    """
    Run manual decode loop (same logic as verify_e2e_trt_ref) with prefill + trailing.
    Returns (all_codec_tokens [n_steps, 16], step_output_token_0 [n_steps], eos_step).
    """
    talker = model.talker
    cp = talker.code_predictor
    hidden_size = talker.model.config.hidden_size

    prefill_embeds = prefill_embeds.to(device).float()
    pad_embed = pad_embed.to(device).float()
    trailing_list = [t.to(device).float() for t in trailing_list]

    with torch.no_grad():
        out = talker.model(
            inputs_embeds=prefill_embeds,
            use_cache=True,
            return_dict=True,
        )
    past_kv = out.past_key_values
    past_hidden = out.last_hidden_state[:, -1:, :]
    prefill_logits = talker.codec_head(out.last_hidden_state)
    codec_token_0 = prefill_logits[:, -1, :].argmax(dim=-1)

    all_codec_tokens = []
    step_output_codec_token_0 = []
    eos_step = -1
    kv = past_kv
    current_past_hidden = past_hidden
    current_codec_token_0 = codec_token_0

    for step in range(max_steps):
        with torch.no_grad():
            embed_0 = talker.model.codec_embedding(current_codec_token_0).unsqueeze(1)
            seq = cp.small_to_mtp_projection(
                torch.cat([current_past_hidden, embed_0], dim=1)
            )
            pt_tokens = []
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
                token = logits.argmax(dim=-1).squeeze(-1)
                pt_tokens.append(token.item())
                if stage < len(cp.lm_head) - 1:
                    next_emb = cp.small_to_mtp_projection(
                        cp.model.codec_embedding[stage](token).unsqueeze(1)
                    )
                    seq = torch.cat([seq, next_emb], dim=1)

            full_codec = [current_codec_token_0.item()] + pt_tokens
            all_codec_tokens.append(full_codec)

            all_tokens = torch.tensor([full_codec], device=device, dtype=torch.long)
            codec_sum = torch.zeros(
                1, 1, hidden_size, device=device, dtype=current_past_hidden.dtype
            )
            for i in range(min(16, all_tokens.shape[1])):
                if i == 0:
                    codec_sum += talker.model.codec_embedding(all_tokens[:, i]).unsqueeze(1)
                elif i - 1 < len(cp.model.codec_embedding):
                    codec_sum += cp.model.codec_embedding[i - 1](
                        all_tokens[:, i]
                    ).unsqueeze(1)

            text_add = (
                trailing_list[step]
                if step < len(trailing_list)
                else pad_embed
            )
            next_input = (codec_sum + text_add).float()

            step_out = talker.model(
                inputs_embeds=next_input,
                past_key_values=kv,
                use_cache=True,
                return_dict=True,
            )
            current_past_hidden = step_out.last_hidden_state
            kv = step_out.past_key_values
            step_logits = talker.codec_head(current_past_hidden)
            current_codec_token_0 = step_logits[:, -1, :].argmax(dim=-1)
            step_output_codec_token_0.append(current_codec_token_0.item())
            if eos_step < 0 and current_codec_token_0.item() == codec_eos_id:
                eos_step = step

        if (step + 1) % 20 == 0 or step == 0:
            logger.info(f"  manual step {step}: codec_0={full_codec[0]}")

    all_codec_tokens = np.array(all_codec_tokens, dtype=np.int64)
    step_output_codec_token_0 = np.array(step_output_codec_token_0, dtype=np.int64)
    return all_codec_tokens, step_output_codec_token_0, eos_step


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Prototype parity: official generate vs manual decode loop"
    )
    parser.add_argument("--variant", default="design-1.7b", help="Model variant")
    parser.add_argument("--text", required=True, help="Input text (same for both paths)")
    parser.add_argument("--language", default="auto", help="Language (e.g. auto, Chinese)")
    parser.add_argument("--speaker", default="", help="Speaker (empty for voice_design)")
    parser.add_argument("--max-steps", type=int, default=200, help="Max decode steps")
    parser.add_argument("--models-dir", default=None, help="Models dir for tokenizer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None, help="Where to save prototype_ref.npz")
    args = parser.parse_args()

    device = resolve_device(args.device)
    variant = args.variant
    text = args.text
    language = args.language
    speaker = args.speaker or ""
    max_steps = args.max_steps

    path = resolve_model_path(variant, args.models_dir)
    if not has_model_weights(path):
        logger.error(f"No weights for {variant}")
        sys.exit(1)

    weights_dir = DEFAULT_OUTPUT_DIR / variant / "weights"
    if not weights_dir.exists():
        logger.error(
            f"Exported weights not found: {weights_dir}. Run export_models.sh first."
        )
        sys.exit(1)

    cfg = json.load(open(weights_dir / "config.json"))
    codec_eos_id = int(cfg.get("codec_eos_token_id", 4198))

    logger.info(f"Loading {variant} on {device} (FP32 for parity) ...")
    model = load_tts_model(path, device=device, dtype=torch.float32)
    talker = model.talker
    tts_pad_token_id = getattr(model.config, "tts_pad_token_id", 0)
    if hasattr(model.config, "talker_config") and getattr(
        model.config.talker_config, "codec_eos_token_id", None
    ):
        codec_eos_id = model.config.talker_config.codec_eos_token_id

    # Prefer tokenizer from model dir (vocab.json + merges.txt); fallback to dedicated tokenizer dir
    tokenizer_dir = str(path)  # model path has vocab.json + merges.txt for TTS models
    bootstrap_project_imports("repo")
    from engine.backend.prefill import EmbeddingWeights, PrefillBuilder
    from engine.backend.prefill import TaskType
    from engine.frontend.spliter.tokenizer import load_lightweight_tokenizer

    tokenizer = load_lightweight_tokenizer(tokenizer_dir)
    if tokenizer is None:
        alt_dir = str(resolve_tokenizer_path(args.models_dir))
        tokenizer = load_lightweight_tokenizer(alt_dir)
    if tokenizer is None:
        logger.error(
            "Failed to load tokenizer. Ensure model dir or tokenizer_dir has tokenizer.json or vocab.json+merges.txt."
        )
        sys.exit(1)

    assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>"
    out = tokenizer(assistant_text, return_tensors="pt")
    input_ids_np = out["input_ids"]
    if hasattr(input_ids_np, "numpy"):
        input_ids = torch.from_numpy(input_ids_np).to(device=device, dtype=torch.long)
    else:
        input_ids = torch.as_tensor(input_ids_np, device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    # --- (1) Prototype: official model.generate (greedy) ---
    logger.info("(1) Running official model.generate (greedy) ...")
    prototype_codec = run_prototype_generate(
        model, input_ids, language, speaker, max_steps, device
    )
    T_proto = prototype_codec.shape[0]
    prototype_first = prototype_codec[:, 0]
    proto_eos_step = -1
    for i in range(T_proto):
        if prototype_first[i] == codec_eos_id:
            proto_eos_step = i
            break
    logger.info(f"  Prototype: T={T_proto}, eos_step={proto_eos_step}")

    # --- (2) Manual: official-style prefill + decode loop ---
    logger.info("(2) Building prefill (official-style) + running manual decode loop ...")
    if getattr(model.config, "talker_config", None) is not None:
        prefill_embeds, trailing_list = build_prefill_like_official(
            model, input_ids, language, speaker, device
        )
    else:
        if isinstance(device, torch.device) and device.type == "cuda":
            device_id = device.index
        elif isinstance(device, str) and device.startswith("cuda:"):
            device_id = int(device.split(":")[-1])
        else:
            device_id = 0
        weights = EmbeddingWeights(str(weights_dir), device_id=device_id)
        prefill_builder = PrefillBuilder(weights, tokenizer)
        task_type = TaskType.VOICE_DESIGN if not speaker else TaskType.CUSTOM_VOICE
        prefill_embeds, trailing_list = prefill_builder.build(
            task_type, text, language=language, speaker=speaker or None
        )
        logger.warning("Model has no talker_config; used PrefillBuilder (may diverge from prototype).")

    with torch.no_grad():
        pad_id = torch.tensor([[tts_pad_token_id]], device=device, dtype=torch.long)
        pad_embed = talker.text_projection(talker.model.text_embedding(pad_id))

    manual_codec, manual_token_0, manual_eos_step = run_manual_decode_loop(
        model,
        prefill_embeds,
        trailing_list,
        pad_embed,
        max_steps,
        codec_eos_id,
        device,
    )
    T_manual = manual_codec.shape[0]
    logger.info(f"  Manual: T={T_manual}, eos_step={manual_eos_step}")

    # --- (3) Compare ---
    logger.info("(3) Comparing step-by-step ...")
    n_compare = min(T_proto, T_manual)
    token_0_matches = 0
    full_codec_matches = 0
    first_divergence = -1

    for step in range(n_compare):
        proto_row = prototype_codec[step]
        manual_row = manual_codec[step]
        if proto_row[0] == manual_row[0]:
            token_0_matches += 1
        else:
            if first_divergence < 0:
                first_divergence = step
        if np.array_equal(proto_row, manual_row):
            full_codec_matches += 1

    token_0_rate = token_0_matches / n_compare if n_compare > 0 else 0
    full_rate = full_codec_matches / n_compare if n_compare > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("  PROTOTYPE PARITY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  codec_token_0 match: {token_0_matches}/{n_compare} = {token_0_rate:.1%}")
    logger.info(f"  full_codec match:    {full_codec_matches}/{n_compare} = {full_rate:.1%}")
    logger.info(f"  EOS: prototype step={proto_eos_step}, manual step={manual_eos_step}")
    if first_divergence >= 0:
        logger.info(f"  First divergence at step: {first_divergence}")
        logger.info(
            f"    prototype codec_0={prototype_codec[first_divergence, 0]}, "
            f"manual codec_0={manual_codec[first_divergence, 0]}"
        )

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUTPUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / "prototype_ref.npz"
    np.savez(
        ref_path,
        prototype_codec_tokens=prototype_codec,
        prototype_n_steps=np.array(T_proto),
        prototype_first_codebook=prototype_first,
        prototype_eos_step=np.array(proto_eos_step),
    )
    logger.info(f"  Saved: {ref_path}")

    ok = (
        token_0_rate >= 0.99
        and full_rate >= 0.99
        and (proto_eos_step == manual_eos_step or (proto_eos_step < 0 and manual_eos_step < 0))
    )
    status = "PASS" if ok else "FAIL"
    logger.info(f"  Status: {status}")

    del model
    torch.cuda.empty_cache()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
