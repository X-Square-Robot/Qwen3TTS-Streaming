#!/usr/bin/env python3
"""
Unified component verification tool for Qwen3-TTS.

Merges the following legacy scripts into a single entry point:
  - verify_code2wav_streaming.py       -> --component code2wav
  - verify_speech_tokenizer_encoder.py -> --component speech-tokenizer
  - verify_precision_ort.py            -> --component precision-ort
  - verify_prototype_parity.py         -> --component prototype-parity

Usage (from repo root, conda activate qwen3-tts):
  python tools/validation/verify_components.py --component code2wav [options]
  python tools/validation/verify_components.py --component speech-tokenizer [options]
  python tools/validation/verify_components.py --component precision-ort --onnx ... --ref ...
  python tools/validation/verify_components.py --component prototype-parity --text "你好" [options]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Ensure the validation tools directory is on sys.path for _bootstrap
_validation_dir = str(Path(__file__).resolve().parent)
if _validation_dir not in sys.path:
    sys.path.insert(0, _validation_dir)

from _bootstrap import bootstrap_tool_imports

bootstrap_tool_imports()

from qwen3_tts_protocol.audio import DEFAULT_SAMPLE_RATE  # noqa: F401 — protocol import per spec
from common import REPO_ROOT, bootstrap_project_imports

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPONENTS = ("code2wav", "speech-tokenizer", "precision-ort", "prototype-parity")

LEGACY_SCRIPTS = {
    "code2wav":          "verify_code2wav_streaming.py",
    "speech-tokenizer":  "verify_speech_tokenizer_encoder.py",
    "precision-ort":     "verify_precision_ort.py",
    "prototype-parity":  "verify_prototype_parity.py",
}

# Per-component arg mapping: (unified_flag, legacy_flag, pass_if_default)
# "pass_if_default" means: only forward this flag when the user explicitly set it.
COMPONENT_ARGS: dict[str, list[tuple[str, str, bool]]] = {
    "code2wav": [
        ("models_dir", "--models-dir", False),
        ("variant",    "--variant",     True),
        ("num_frames", "--num-frames",  True),
        ("device",     "--device",      True),
    ],
    "speech-tokenizer": [
        ("models_dir", "--models-dir", False),
        ("onnx",       "--onnx",       True),
        ("device",     "--device",     True),
        ("num_tests",  "--num-tests",  True),
    ],
    "precision-ort": [
        ("onnx",      "--onnx",      False),
        ("ref",       "--ref",       False),
        ("model_dir", "--model-dir", True),
        ("steps",     "--steps",     True),
        ("provider",  "--provider",  True),
    ],
    "prototype-parity": [
        ("variant",    "--variant",    True),
        ("text",       "--text",       False),
        ("language",   "--language",   True),
        ("speaker",    "--speaker",    True),
        ("max_steps",  "--max-steps",  True),
        ("models_dir", "--models-dir", True),
        ("device",     "--device",     True),
    ],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("verify_components")


def _print_deprecation(component: str) -> None:
    legacy = LEGACY_SCRIPTS[component]
    logger.warning(
        "[deprecation] %s is the legacy script for --component %s. "
        "Use verify_components.py --component %s instead.",
        legacy, component, component,
    )


def _build_legacy_cmd(component: str, args: argparse.Namespace) -> list[str]:
    """Build the command-line to invoke the legacy script for a given component."""
    script = REPO_ROOT / "tests" / "tools" / LEGACY_SCRIPTS[component]
    cmd = [sys.executable, str(script)]
    for attr, flag, pass_if_default in COMPONENT_ARGS[component]:
        val = getattr(args, attr, None)
        if val is None:
            continue
        # For pass_if_default flags, skip if the value matches the parser default.
        if pass_if_default:
            for action in _PARSER_ACTIONS:
                if action.dest == attr and val == action.default:
                    val = None
                    break
        if val is not None:
            cmd.extend([flag, str(val)])
    return cmd


def _run_component(component: str, args: argparse.Namespace) -> int:
    """Delegate to the legacy script and return its exit code."""
    cmd = _build_legacy_cmd(component, args)
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


# ===================================================================
# CLI
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_components",
        description="Unified component verification for Qwen3-TTS. "
                    "Each --component delegates to the corresponding legacy script.",
    )
    p.add_argument("--component", required=True, choices=COMPONENTS,
                   help="Component to verify: " + ", ".join(COMPONENTS))
    # Shared
    p.add_argument("--models-dir", default=None, help="Models root (default: workspace/models)")
    p.add_argument("--device", default=None, help="Torch device (default: auto)")
    # code2wav
    p.add_argument("--variant", default="design-1.7b", help="Model variant (code2wav / prototype-parity)")
    p.add_argument("--num-frames", type=int, default=24, help="Codec frames for code2wav fixture")
    # speech-tokenizer
    p.add_argument("--onnx", default=None, help="Path to ONNX model file")
    p.add_argument("--num-tests", type=int, default=5, help="Test durations count (speech-tokenizer)")
    # precision-ort
    p.add_argument("--ref", default=None, help="Path to e2e_trt_ref.npz (precision-ort)")
    p.add_argument("--model-dir", default=None, help="Exported variant dir (precision-ort)")
    p.add_argument("--steps", type=int, default=None, help="Max decode steps (precision-ort)")
    p.add_argument("--provider", default="CPUExecutionProvider", help="ORT provider (precision-ort)")
    # prototype-parity
    p.add_argument("--text", default=None, help="Input text (prototype-parity)")
    p.add_argument("--language", default="auto", help="Language (prototype-parity)")
    p.add_argument("--speaker", default="", help="Speaker (prototype-parity)")
    p.add_argument("--max-steps", type=int, default=200, help="Max decode steps (prototype-parity)")
    return p


# Stash parser actions so _build_legacy_cmd can check defaults.
_PARSER_ACTIONS: list = []


def main() -> None:
    global _PARSER_ACTIONS
    parser = build_parser()
    _PARSER_ACTIONS = parser._actions
    args = parser.parse_args()

    _print_deprecation(args.component)
    rc = _run_component(args.component, args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
