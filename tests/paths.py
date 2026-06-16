"""Shared path setup and variant auto-discovery for all tests.

Variant resolution order:
  1. TEST_VARIANT env var (explicit override)
  2. First variant with weights in workspace/exported/

Usage:
    from tests.paths import TOKENIZER_DIR, WEIGHTS_DIR, VARIANT
"""

import os
from pathlib import Path

# Re-export REPO_ROOT from the canonical source (qwen3tts_tools.common).
# For environments where qwen3tts_tools is not on sys.path (e.g. bare pytest),
# fall back to a relative computation.
try:
    from qwen3tts_tools.common import REPO_ROOT
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[1]

EXPORTED_DIR = REPO_ROOT / "workspace" / "exported"
MODELS_DIR = REPO_ROOT / "workspace" / "models"

# variant name → HuggingFace model directory name
VARIANT_MODEL_MAP = {
    "design-1.7b": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom-1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "base-1.7b":   "Qwen3-TTS-12Hz-1.7B-Base",
    "custom-0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base-0.6b":   "Qwen3-TTS-12Hz-0.6B-Base",
}


def _discover_variant() -> str:
    """Auto-discover first exported variant with weights."""
    if not EXPORTED_DIR.is_dir():
        return ""
    for vdir in sorted(EXPORTED_DIR.iterdir()):
        if not vdir.is_dir() or vdir.name == "tokenizer":
            continue
        if (vdir / "weights").is_dir():
            return vdir.name
    return ""


VARIANT = os.environ.get("TEST_VARIANT", "") or _discover_variant()

TOKENIZER_DIR = MODELS_DIR / VARIANT_MODEL_MAP.get(VARIANT, "") if VARIANT else Path("")
WEIGHTS_DIR = EXPORTED_DIR / VARIANT / "weights" if VARIANT else Path("")
ONNX_DIR = EXPORTED_DIR / VARIANT if VARIANT else Path("")
ENGINE_DIR = EXPORTED_DIR / VARIANT / "engines" / "talker_code2wav_fused" if VARIANT else Path("")
SHARED_TOKENIZER_DIR = EXPORTED_DIR / "tokenizer"
