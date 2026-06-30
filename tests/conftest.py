"""
Pytest conftest: shared path setup and variant auto-discovery.

Path constants (REPO_ROOT, VARIANT, TOKENIZER_DIR, etc.) are defined here
and can be imported from any test module via ``from tests.conftest import ...``.
"""

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# REPO_ROOT — canonical source
# ---------------------------------------------------------------------------

# Put scripts/python on sys.path so the shared helpers (common/audio/layer_audit)
# are importable by tests and tools alike.
_SCRIPTS_PYTHON = Path(__file__).resolve().parents[1] / "scripts" / "python"
if str(_SCRIPTS_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_PYTHON))

try:
    from common import REPO_ROOT
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Variant auto-discovery
# ---------------------------------------------------------------------------

EXPORTED_DIR = REPO_ROOT / "workspace" / "exported"
MODELS_DIR = REPO_ROOT / "workspace" / "models"

VARIANT_MODEL_MAP = {
    "design-1.7b": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom-1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "base-1.7b": "Qwen3-TTS-12Hz-1.7B-Base",
    "custom-0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "base-0.6b": "Qwen3-TTS-12Hz-0.6B-Base",
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
ENGINE_DIR = (
    EXPORTED_DIR / VARIANT / "engines" / "talker_code2wav_fused"
    if VARIANT
    else Path("")
)
SHARED_TOKENIZER_DIR = EXPORTED_DIR / "tokenizer"


# ---------------------------------------------------------------------------
# Legacy sys.path setup for orchestrator source
# ---------------------------------------------------------------------------

ORCH_1 = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if ORCH_1.is_dir() and str(ORCH_1) not in sys.path:
    sys.path.insert(0, str(ORCH_1))
