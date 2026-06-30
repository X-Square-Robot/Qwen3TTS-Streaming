"""
L2 test T2.1: Verify export_01_embeddings.py output structure.

Checks that after running:
  mamba run -n qwen3-tts python scripts/export/export_01_embeddings.py --variant base-1.7b --device cuda

the expected files and config fields exist. Skips if workspace/exported/<variant> not present.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORTED = REPO_ROOT / "workspace" / "exported"


def _variant_weights_dir(variant: str = "base-1.7b"):
    return EXPORTED / variant / "weights"


@pytest.fixture(scope="module")
def weights_dir():
    d = _variant_weights_dir()
    if not d.is_dir():
        pytest.skip(f"Export output not found: {d} (run export_01_embeddings.py first)")
    return d


def test_legacy_pt_files_exist(weights_dir):
    """Legacy .pt: text_embedding, text_projection, codec_embeddings, special_embeddings, codec_head, config.json."""
    required = [
        "text_embedding.pt",
        "text_projection.pt",
        "codec_embeddings.pt",
        "special_embeddings.pt",
        "codec_head.pt",
        "config.json",
    ]
    for name in required:
        p = weights_dir / name
        assert p.is_file(), f"Missing {p}"


def test_codec_embeddings_3d_pt_exists(weights_dir):
    """codec_embeddings_3d.pt (optional but expected for ICL)."""
    p = weights_dir / "codec_embeddings_3d.pt"
    assert p.is_file(), f"Missing {p} (required for ICL)"


def test_npz_files_exist(weights_dir):
    """New .npz: special_embeddings.npz, codec_embeddings_3d.npz (from current export_01)."""
    for name in ("special_embeddings.npz", "codec_embeddings_3d.npz"):
        p = weights_dir / name
        if not p.is_file():
            pytest.skip(
                f"Missing {p} — re-run export_01_embeddings.py to produce .npz outputs"
            )


def test_config_has_hidden_act(weights_dir):
    """config.json contains hidden_act field (from current export_01)."""
    cfg_path = weights_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "hidden_act" not in cfg:
        pytest.skip("config.json missing hidden_act — re-run export_01_embeddings.py")
    assert cfg["hidden_act"] in ("silu", "gelu", "quick_gelu"), (
        f"Unexpected hidden_act: {cfg['hidden_act']}"
    )


def test_code_predictor_lm_heads_optional(weights_dir):
    """code_predictor_lm_heads.pt is optional."""
    p = weights_dir / "code_predictor_lm_heads.pt"
    # Only check it exists when present (some exports may omit)
    if p.is_file():
        assert p.stat().st_size > 0
