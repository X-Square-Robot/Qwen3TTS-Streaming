#!/usr/bin/env python3
"""
[Step 01] Export Embedding weights for the TTS Orchestrator.

Exports .pt weights + numpy snapshots for in-process torch embedding in Orchestrator:
  - .pt: text_embedding, text_projection, codec_embeddings, codec_embeddings_3d,
         special_embeddings, codec_head, code_predictor_lm_heads, config.json
  - .npz: special_embeddings.npz, codec_embeddings_3d.npz (lightweight numpy for BLS)

Applies to: ALL variants
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow importing from scripts/python (for CodecEmbeddingSum)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from python.codec_embedding_sum import CodecEmbeddingSum
from engine.runtime.model_version import (
    MODEL_VERSION_FILENAME,
    load_model_version,
    validate_model_version,
)

from utils import (
    setup_logging,
    resolve_model_path,
    ensure_output_dir,
    prepare_native_cursor_head,
    load_tts_model,
    resolve_device,
    resolve_dtype,
    add_common_args,
    MODEL_VARIANTS,
    DEFAULT_DTYPE,
    DTYPE_NAMES,
)

logger = logging.getLogger("onnx_export")


def _cast_state_dict(state_dict: dict, dtype: torch.dtype) -> dict:
    """Cast all float tensors in a state_dict to the target dtype."""
    return {
        k: v.to(dtype) if v.is_floating_point() else v for k, v in state_dict.items()
    }


def export_embeddings(
    variant: str,
    models_dir: str = None,
    output_dir: str = None,
    device: str = "cpu",
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> Path:
    model_path = resolve_model_path(variant, models_dir)
    out_dir = ensure_output_dir(output_dir, variant) / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The optional model-owned cursor checkpoint is kept beside the exported
    # embedding weights so the assembled package is self-describing: its
    # presence opts the matching cursor-enabled export/runtime path in.
    prepare_native_cursor_head(model_path, out_dir)

    exported_version_path = out_dir.parent / MODEL_VERSION_FILENAME
    source_version_path = Path(model_path) / MODEL_VERSION_FILENAME
    # MODEL_VERSION is model-owned metadata: validate and copy it before the
    # expensive export so every successful artifact is unambiguously tagged.
    # During migration, an already-tagged export remains rerunnable even when
    # the external source-model mount is read-only and cannot yet be updated.
    release_override = os.environ.get("QWEN3_TTS_MODEL_RELEASE_VERSION", "").strip()
    if release_override:
        model_version = validate_model_version(
            release_override,
            source="QWEN3_TTS_MODEL_RELEASE_VERSION",
        )
        logger.info(
            "  %s: using explicit model release from autorun",
            MODEL_VERSION_FILENAME,
        )
    elif source_version_path.is_file():
        model_version = load_model_version(model_path)
    elif exported_version_path.is_file():
        model_version = validate_model_version(
            exported_version_path.read_text(encoding="utf-8"),
            source=str(exported_version_path),
        )
        logger.warning(
            "  %s is missing; preserving version from existing export",
            source_version_path,
        )
    else:
        model_version = load_model_version(model_path)
    if exported_version_path.exists():
        # The assembled package is immutable, but this intermediate export is
        # intentionally replaceable so Phase A can be rerun.
        exported_version_path.chmod(0o644)
    exported_version_path.write_text(f"{model_version}\n", encoding="utf-8")
    exported_version_path.chmod(0o444)
    logger.info(
        "  %s: %s (validated for export)",
        MODEL_VERSION_FILENAME,
        model_version,
    )

    logger.info(f"Loading model: {variant} from {model_path} (fp32 for precision)")
    logger.info(f"  Weights will be saved as {DTYPE_NAMES[dtype]} for inference")
    model = load_tts_model(model_path, device=device, dtype=torch.float32)

    talker = model.talker
    config = model.config

    # 1. Text Embedding
    text_emb = _cast_state_dict(talker.model.text_embedding.state_dict(), dtype)
    path = out_dir / "text_embedding.pt"
    torch.save(text_emb, path)
    n_params = sum(p.numel() for p in talker.model.text_embedding.parameters())
    logger.info(f"  text_embedding.pt: {n_params / 1e6:.1f}M params, saved to {path}")

    # 2. Text Projection (ResizeMLP)
    text_proj = _cast_state_dict(talker.text_projection.state_dict(), dtype)
    path = out_dir / "text_projection.pt"
    torch.save(text_proj, path)
    n_params = sum(p.numel() for p in talker.text_projection.parameters())
    logger.info(f"  text_projection.pt: {n_params / 1e6:.1f}M params, saved to {path}")

    # 3. Codec Embeddings
    codec_embs = {
        "talker_codec_embedding": _cast_state_dict(
            talker.model.codec_embedding.state_dict(), dtype
        ),
        "code_predictor_codec_embeddings": [
            _cast_state_dict(emb.state_dict(), dtype)
            for emb in talker.code_predictor.model.codec_embedding
        ],
    }
    path = out_dir / "codec_embeddings.pt"
    torch.save(codec_embs, path)
    n_talker = sum(p.numel() for p in talker.model.codec_embedding.parameters())
    n_cp = sum(
        sum(p.numel() for p in emb.parameters())
        for emb in talker.code_predictor.model.codec_embedding
    )
    logger.info(
        f"  codec_embeddings.pt: talker={n_talker / 1e6:.1f}M + "
        f"code_predictor={n_cp / 1e6:.1f}M params, saved to {path}"
    )

    # 3b. Pre-stacked 3D codec embedding (for optimized gather+sum, §2.1)
    stacked = CodecEmbeddingSum.stack_weights(
        talker.model.codec_embedding.weight,
        [emb.weight for emb in talker.code_predictor.model.codec_embedding],
        dtype=dtype,
    )
    path_3d = out_dir / "codec_embeddings_3d.pt"
    torch.save(stacked.cpu(), path_3d)
    logger.info(
        f"  codec_embeddings_3d.pt: shape={tuple(stacked.shape)}, saved to {path_3d}"
    )
    # Lightweight deploy: numpy format for BLS (cupy load)
    path_3d_npz = out_dir / "codec_embeddings_3d.npz"
    np.savez_compressed(
        path_3d_npz,
        data=stacked.float().detach().cpu().numpy(),
    )
    logger.info(f"  codec_embeddings_3d.npz: saved to {path_3d_npz}")

    # 4. Special Embeddings (tts_pad, tts_bos, tts_eos) — compute with target-dtype
    # modules so the saved tensors match runtime BF16 prefill exactly.
    with torch.no_grad():
        text_embedding_runtime = copy.deepcopy(talker.model.text_embedding).to(
            device=device, dtype=dtype
        )
        text_projection_runtime = copy.deepcopy(talker.text_projection).to(
            device=device, dtype=dtype
        )
        text_embedding_runtime.eval()
        text_projection_runtime.eval()
        special_ids = torch.tensor(
            [
                [
                    config.tts_pad_token_id,
                    config.tts_bos_token_id,
                    config.tts_eos_token_id,
                ]
            ],
            device=device,
        )
        special_text_embed = text_embedding_runtime(special_ids)
        special_projected = text_projection_runtime(special_text_embed)
        tts_pad_embed = special_projected[:, 0:1, :]
        tts_bos_embed = special_projected[:, 1:2, :]
        tts_eos_embed = special_projected[:, 2:3, :]

    special = {
        "tts_pad_embed": tts_pad_embed.cpu(),
        "tts_bos_embed": tts_bos_embed.cpu(),
        "tts_eos_embed": tts_eos_embed.cpu(),
        "tts_pad_token_id": config.tts_pad_token_id,
        "tts_bos_token_id": config.tts_bos_token_id,
        "tts_eos_token_id": config.tts_eos_token_id,
    }
    path = out_dir / "special_embeddings.pt"
    torch.save(special, path)
    logger.info(
        f"  special_embeddings.pt: shape={tts_pad_embed.shape}, saved to {path}"
    )
    # Lightweight deploy: numpy format for BLS (no torch)
    path_special_npz = out_dir / "special_embeddings.npz"
    np.savez(
        path_special_npz,
        tts_pad_embed=tts_pad_embed.float().cpu().numpy(),
        tts_bos_embed=tts_bos_embed.float().cpu().numpy(),
        tts_eos_embed=tts_eos_embed.float().cpu().numpy(),
    )
    logger.info(f"  special_embeddings.npz: saved to {path_special_npz}")

    # 5. Codec Head
    codec_head = _cast_state_dict(talker.codec_head.state_dict(), dtype)
    path = out_dir / "codec_head.pt"
    torch.save(codec_head, path)
    n_params = sum(p.numel() for p in talker.codec_head.parameters())
    logger.info(f"  codec_head.pt: {n_params / 1e6:.1f}M params, saved to {path}")

    # 6. Code Predictor lm_heads (num_code_groups-1 heads, needed for fallback mode)
    lm_heads = [
        _cast_state_dict(head.state_dict(), dtype)
        for head in talker.code_predictor.lm_head
    ]
    path = out_dir / "code_predictor_lm_heads.pt"
    torch.save(lm_heads, path)
    n_params = sum(
        sum(p.numel() for p in head.parameters())
        for head in talker.code_predictor.lm_head
    )
    logger.info(
        f"  code_predictor_lm_heads.pt: {n_params / 1e6:.1f}M params ({len(lm_heads)} heads)"
    )

    # 7. Model config metadata
    metadata = {
        "variant": variant,
        "target_dtype": DTYPE_NAMES[dtype],
        "tts_model_type": config.tts_model_type,
        "tts_model_size": config.tts_model_size,
        "tokenizer_type": config.tokenizer_type,
        "talker_hidden_size": talker.config.hidden_size,
        "talker_text_hidden_size": talker.config.text_hidden_size,
        "talker_vocab_size": talker.config.vocab_size,
        "talker_num_layers": talker.config.num_hidden_layers,
        "talker_num_heads": talker.config.num_attention_heads,
        "talker_num_kv_heads": talker.config.num_key_value_heads,
        "talker_head_dim": getattr(
            talker.config,
            "head_dim",
            talker.config.hidden_size // talker.config.num_attention_heads,
        ),
        "num_code_groups": talker.config.num_code_groups,
        "code_predictor_hidden_size": talker.config.code_predictor_config.hidden_size,
        "code_predictor_num_layers": talker.config.code_predictor_config.num_hidden_layers,
        "code_predictor_vocab_size": talker.config.code_predictor_config.vocab_size,
        "codec_eos_token_id": talker.config.codec_eos_token_id,
        "codec_bos_id": talker.config.codec_bos_id,
        "codec_pad_id": talker.config.codec_pad_id,
        "codec_nothink_id": getattr(talker.config, "codec_nothink_id", 2155),
        "codec_think_bos_id": getattr(talker.config, "codec_think_bos_id", 2156),
        "codec_think_eos_id": getattr(talker.config, "codec_think_eos_id", 2157),
        "codec_think_id": getattr(talker.config, "codec_think_id", 2154),
        "hidden_act": getattr(talker.config, "hidden_act", "silu"),
        "im_start_token_id": config.im_start_token_id,
        "im_end_token_id": config.im_end_token_id,
    }
    if hasattr(talker.config, "spk_id") and talker.config.spk_id:
        metadata["spk_id"] = talker.config.spk_id
    if hasattr(talker.config, "spk_is_dialect") and talker.config.spk_is_dialect:
        metadata["spk_is_dialect"] = talker.config.spk_is_dialect
    if hasattr(talker.config, "codec_language_id") and talker.config.codec_language_id:
        metadata["codec_language_id"] = talker.config.codec_language_id

    path = out_dir / "config.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  config.json: saved to {path}")

    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return out_dir


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Export Embedding weights")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model variant. Default: export all variants",
    )
    add_common_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    variants = [args.variant] if args.variant else list(MODEL_VARIANTS.keys())

    for variant in variants:
        if variant not in MODEL_VARIANTS:
            logger.error(f"Unknown variant: {variant}")
            continue
        try:
            path = export_embeddings(
                variant, args.models_dir, args.output_dir, device, dtype
            )
            logger.info(f"[{variant}] Embeddings exported to: {path}")
        except FileNotFoundError as e:
            logger.warning(f"[{variant}] Skipped: {e}")
        except Exception as e:
            logger.error(f"[{variant}] Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
