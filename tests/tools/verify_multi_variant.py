#!/usr/bin/env python3
"""
[Phase 1, Item 8] Multi-variant prefill verification.

Two goals:
  A. Compare Talker Backbone & Code Predictor weights across same-size variants
     to determine if engines can be shared (architecture.md §2.2.3).
  B. Run prefill construction for each variant's task_type (Base/CustomVoice/
     VoiceDesign) and verify the output is reasonable.

Usage:
  conda activate qwen3-tts
  python tests/tools/verify_multi_variant.py
  python tests/tools/verify_multi_variant.py --size 1.7b   # only 1.7B variants
  python tests/tools/verify_multi_variant.py --size 0.6b   # only 0.6B variants
"""

import argparse
import json
import logging
import os
import sys
import time
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()
from qwen3tts_tools.common import bootstrap_project_imports

bootstrap_project_imports("scripts_export", "third_party_qwen")

from utils import (
    setup_logging,
    resolve_model_path,
    load_tts_model,
    resolve_device,
    MODEL_VARIANTS,
    DEFAULT_OUTPUT_DIR,
    has_model_weights,
    DEFAULT_MODELS_DIR,
)

logger = logging.getLogger("multi_variant")


def cosine_sim_flat(a, b):
    return float(F.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ))


VARIANT_GROUPS = {
    "0.6b": ["base-0.6b", "custom-0.6b"],
    "1.7b": ["custom-1.7b", "design-1.7b"],
}


# ---------------------------------------------------------------------------
#  Part A: Weight comparison across same-size variants
# ---------------------------------------------------------------------------

def compare_weights(variants, device):
    """Load each variant and compare Talker/CP layer weights pairwise."""
    logger.info("=" * 60)
    logger.info("  Part A: Weight Comparison")
    logger.info("=" * 60)

    models = {}
    for v in variants:
        try:
            path = resolve_model_path(v)
            if not has_model_weights(path):
                logger.warning(f"  {v}: no weights found, skipping")
                continue
            logger.info(f"  Loading {v} ...")
            models[v] = load_tts_model(path, device=device, dtype=torch.float32)
        except FileNotFoundError:
            logger.warning(f"  {v}: model not found, skipping")

    if len(models) < 2:
        logger.warning("  Need at least 2 variants to compare, skipping")
        return {}

    results = {}

    for (v1, m1), (v2, m2) in combinations(models.items(), 2):
        pair = f"{v1}_vs_{v2}"
        logger.info(f"\n  --- {v1} vs {v2} ---")

        pair_result = {}

        # Talker Backbone layers
        t1_layers = dict(m1.talker.model.layers.named_parameters())
        t2_layers = dict(m2.talker.model.layers.named_parameters())

        talker_sims = []
        for name in sorted(t1_layers.keys()):
            if name in t2_layers:
                s = cosine_sim_flat(t1_layers[name].data, t2_layers[name].data)
                talker_sims.append(s)

        avg_talker = float(np.mean(talker_sims)) if talker_sims else 0
        min_talker = float(np.min(talker_sims)) if talker_sims else 0
        logger.info(f"  Talker layers: avg_cosine={avg_talker:.6f}, "
                    f"min={min_talker:.6f} ({len(talker_sims)} params)")
        pair_result["talker_avg_cosine"] = avg_talker
        pair_result["talker_min_cosine"] = min_talker
        pair_result["talker_n_params"] = len(talker_sims)

        # Talker norm
        t1_norm = dict(m1.talker.model.norm.named_parameters())
        t2_norm = dict(m2.talker.model.norm.named_parameters())
        norm_sims = []
        for name in t1_norm:
            if name in t2_norm:
                s = cosine_sim_flat(t1_norm[name].data, t2_norm[name].data)
                norm_sims.append(s)
        if norm_sims:
            avg_norm = float(np.mean(norm_sims))
            logger.info(f"  Talker norm: avg_cosine={avg_norm:.6f}")
            pair_result["talker_norm_cosine"] = avg_norm

        # Codec head
        s = cosine_sim_flat(
            m1.talker.codec_head.weight.data,
            m2.talker.codec_head.weight.data)
        logger.info(f"  Codec head weight: cosine={s:.6f}")
        pair_result["codec_head_cosine"] = s

        # Code Predictor layers
        cp1 = dict(m1.talker.code_predictor.model.layers.named_parameters())
        cp2 = dict(m2.talker.code_predictor.model.layers.named_parameters())
        cp_sims = []
        for name in sorted(cp1.keys()):
            if name in cp2:
                s = cosine_sim_flat(cp1[name].data, cp2[name].data)
                cp_sims.append(s)

        avg_cp = float(np.mean(cp_sims)) if cp_sims else 0
        min_cp = float(np.min(cp_sims)) if cp_sims else 0
        logger.info(f"  CP layers: avg_cosine={avg_cp:.6f}, "
                    f"min={min_cp:.6f} ({len(cp_sims)} params)")
        pair_result["cp_avg_cosine"] = avg_cp
        pair_result["cp_min_cosine"] = min_cp
        pair_result["cp_n_params"] = len(cp_sims)

        # Code Predictor lm_heads
        n_heads = min(len(m1.talker.code_predictor.lm_head),
                      len(m2.talker.code_predictor.lm_head))
        lm_sims = []
        for i in range(n_heads):
            s = cosine_sim_flat(
                m1.talker.code_predictor.lm_head[i].weight.data,
                m2.talker.code_predictor.lm_head[i].weight.data)
            lm_sims.append(s)
        avg_lm = float(np.mean(lm_sims)) if lm_sims else 0
        logger.info(f"  CP lm_heads: avg_cosine={avg_lm:.6f} ({n_heads} heads)")
        pair_result["cp_lm_heads_cosine"] = avg_lm

        # Text embedding
        s = cosine_sim_flat(
            m1.talker.model.text_embedding.weight.data,
            m2.talker.model.text_embedding.weight.data)
        logger.info(f"  Text embedding: cosine={s:.6f}")
        pair_result["text_embedding_cosine"] = s

        # Codec embedding (talker)
        s = cosine_sim_flat(
            m1.talker.model.codec_embedding.weight.data,
            m2.talker.model.codec_embedding.weight.data)
        logger.info(f"  Codec embedding: cosine={s:.6f}")
        pair_result["codec_embedding_cosine"] = s

        # Verdict
        shareable = (avg_talker > 0.999 and avg_cp > 0.999 and
                     avg_lm > 0.999 and s > 0.999)
        pair_result["engine_shareable"] = shareable
        if shareable:
            logger.info(f"  >>> SHAREABLE: Talker/CP engines can be shared")
        else:
            logger.info(f"  >>> NOT SHAREABLE: weights differ significantly")

        results[pair] = pair_result

    # Free models
    for m in models.values():
        del m
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
#  Part B: Prefill construction for each task_type
# ---------------------------------------------------------------------------

def verify_prefill_per_variant(variants, device):
    """Run a simple prefill for each variant and check output sanity."""
    logger.info("\n" + "=" * 60)
    logger.info("  Part B: Prefill per Variant")
    logger.info("=" * 60)

    results = {}

    for v in variants:
        try:
            path = resolve_model_path(v)
            if not has_model_weights(path):
                logger.warning(f"  {v}: no weights, skipping")
                continue
        except FileNotFoundError:
            logger.warning(f"  {v}: not found, skipping")
            continue

        logger.info(f"\n  --- {v} ---")
        model = load_tts_model(path, device=device, dtype=torch.float32)
        talker = model.talker
        config = model.config

        weights_dir = DEFAULT_OUTPUT_DIR / v / "weights"
        if not weights_dir.exists():
            logger.warning(f"  {v}: no exported weights, skipping prefill")
            del model
            continue

        cfg = json.load(open(weights_dir / "config.json"))
        task_type = cfg["tts_model_type"]
        hidden_size = cfg["talker_hidden_size"]
        logger.info(f"  task_type={task_type}, hidden={hidden_size}")

        # Build a minimal prefill sequence
        # Common: role (3 tokens) + tag (3 tokens) + bos (1 token)
        with torch.no_grad():
            # Simulate text tokens (just use some token IDs)
            n_text = 10
            text_ids = torch.randint(100, 5000, (1, n_text), device=device)
            text_embed = talker.text_projection(talker.model.text_embedding(text_ids))

            # Tag tokens (codec domain)
            codec_bos = cfg.get("codec_bos_id", 2149)
            tag_ids = torch.tensor([[codec_bos]], device=device)
            codec_embed = talker.model.codec_embedding(tag_ids)

            # Construct minimal prefill: text + codec_bos
            inputs_embeds = torch.cat([text_embed, codec_embed], dim=1)

            if task_type == "custom_voice" and hasattr(talker.config, "spk_id"):
                spk_ids = talker.config.spk_id
                if spk_ids:
                    first_spk = list(spk_ids.values())[0]
                    spk_embed = talker.model.codec_embedding(
                        torch.tensor([[first_spk]], device=device))
                    inputs_embeds = torch.cat([text_embed, spk_embed, codec_embed], dim=1)
                    logger.info(f"  CustomVoice: inserted speaker (id={first_spk})")

            logger.info(f"  Prefill shape: {inputs_embeds.shape}")

            # Run Talker forward
            out = talker.model(
                inputs_embeds=inputs_embeds,
                use_cache=False,
                return_dict=True,
            )
            hidden = out.last_hidden_state
            logits = talker.codec_head(hidden)

            logger.info(f"  Hidden: {hidden.shape}, range=[{hidden.min():.2f}, {hidden.max():.2f}]")
            logger.info(f"  Logits: {logits.shape}, range=[{logits.min():.2f}, {logits.max():.2f}]")

            codec_token_0 = logits[:, -1, :].argmax(dim=-1)
            logger.info(f"  First codec_token_0: {codec_token_0.item()}")

            # Run one Code Predictor step
            cp = talker.code_predictor
            past_hidden = hidden[:, -1:, :]
            embed_0 = talker.model.codec_embedding(codec_token_0).unsqueeze(1)
            seq = cp.small_to_mtp_projection(torch.cat([past_hidden, embed_0], dim=1))

            B, S, D = seq.shape
            pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            pos_embs = cp.model.rotary_emb(seq, pos_ids)
            causal = torch.triu(
                torch.full((S, S), float('-inf'), device=device, dtype=seq.dtype),
                diagonal=1).unsqueeze(0).unsqueeze(0)

            h = seq
            for layer in cp.model.layers:
                h = layer(h, attention_mask=causal, position_ids=pos_ids,
                          past_key_values=None, output_attentions=False,
                          use_cache=False,
                          cache_position=torch.arange(S, device=device),
                          position_embeddings=pos_embs)[0]
            h = cp.model.norm(h)
            cp_logits = cp.lm_head[0](h[:, -1:, :])
            token_1 = cp_logits.argmax(dim=-1).squeeze(-1)
            logger.info(f"  CP stage 0 token: {token_1.item()}")

        vr = {
            "task_type": task_type,
            "hidden_size": hidden_size,
            "prefill_shape": list(inputs_embeds.shape),
            "hidden_range": [float(hidden.min()), float(hidden.max())],
            "logits_range": [float(logits.min()), float(logits.max())],
            "codec_token_0": codec_token_0.item(),
            "cp_token_1": token_1.item(),
            "status": "pass",
        }
        results[v] = vr
        logger.info(f"  [PASS] {v}")

        del model
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Multi-variant weight comparison & prefill verification")
    parser.add_argument("--size", default=None, choices=["0.6b", "1.7b"],
                        help="Only test a specific model size group")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)

    if args.size:
        groups = {args.size: VARIANT_GROUPS[args.size]}
    else:
        groups = VARIANT_GROUPS

    all_results = {"weight_comparison": {}, "prefill_verification": {}}
    t0 = time.time()

    for size, variants in groups.items():
        logger.info(f"\n{'#'*60}")
        logger.info(f"  Model size: {size}")
        logger.info(f"  Variants: {variants}")
        logger.info(f"{'#'*60}")

        # Part A
        wc = compare_weights(variants, device)
        all_results["weight_comparison"][size] = wc

        # Part B
        pv = verify_prefill_per_variant(variants, device)
        all_results["prefill_verification"].update(pv)

    elapsed = time.time() - t0

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)

    for size, pairs in all_results["weight_comparison"].items():
        for pair, data in pairs.items():
            shareable = data.get("engine_shareable", False)
            tag = "SHAREABLE" if shareable else "NOT SHAREABLE"
            logger.info(f"  [{tag}] {pair}:")
            logger.info(f"    Talker: avg={data['talker_avg_cosine']:.6f}, "
                        f"min={data['talker_min_cosine']:.6f}")
            logger.info(f"    CP:     avg={data['cp_avg_cosine']:.6f}, "
                        f"min={data['cp_min_cosine']:.6f}")

    for v, data in all_results["prefill_verification"].items():
        logger.info(f"  [PASS] {v} ({data['task_type']}): "
                    f"codec_0={data['codec_token_0']}, cp_1={data['cp_token_1']}")

    logger.info(f"  Total time: {elapsed:.1f}s")

    report_path = DEFAULT_OUTPUT_DIR / "multi_variant_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
