#!/usr/bin/env python3
"""
Unified engine verification tool.

Merges the functionality of eight separate verify_* scripts into a single
entry-point driven by --phase.  Each phase delegates to the original logic
while printing a deprecation notice pointing users at the new interface.

Deprecated scripts (use --phase instead):
  verify_e2e.py                  -> --phase e2e
  verify_e2e_trt.py              -> --phase e2e-trt
  verify_e2e_trt_ref.py          -> --phase e2e-trt-ref
  verify_fused_triton_backend.py -> --phase fused-triton
  verify_onnx_autoregressive.py  -> --phase onnx-autoregressive
  verify_code_predictor_trt.py   -> --phase code-predictor
  verify_trt_talker.py           -> --phase trt-talker
  verify_multi_variant.py        -> --phase multi-variant

Usage examples:
  python tools/validation/verify_engine.py --phase e2e --variant design-1.7b --steps 10
  python tools/validation/verify_engine.py --phase e2e-trt --model-dir workspace/exported/design-1.7b
  python tools/validation/verify_engine.py --phase e2e-trt-ref --variant design-1.7b --steps 50
  python tools/validation/verify_engine.py --phase fused-triton --variant custom-1.7b --text "Hello"
  python tools/validation/verify_engine.py --phase onnx-autoregressive --dump-dir /dumps --onnx model.onnx
  python tools/validation/verify_engine.py --phase code-predictor --variant-dir workspace/exported/custom-1.7b
  python tools/validation/verify_engine.py --phase trt-talker --variant design-1.7b
  python tools/validation/verify_engine.py --phase multi-variant --size 1.7b
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
#  Bootstrap — add repo root to sys.path so the common helper is importable
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

# Ensure the validation tools directory is on sys.path for _bootstrap
_validation_dir = str(Path(__file__).resolve().parent)
if _validation_dir not in sys.path:
    sys.path.insert(0, _validation_dir)

from _bootstrap import bootstrap_tool_imports

REPO_ROOT = bootstrap_tool_imports()

from common import bootstrap_project_imports

bootstrap_project_imports("scripts", "scripts_export", "third_party_qwen")

# Protocol & streaming support (imported for side-effect availability in phases)
try:
    import qwen3_tts_protocol  # noqa: F401
except ImportError:
    pass

try:
    from tests.support import triton_streaming  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger("verify_engine")

# ---------------------------------------------------------------------------
#  Phase registry
# ---------------------------------------------------------------------------

# Directory containing this script and the phase sub-scripts
_TOOLS_DIR = Path(__file__).resolve().parent

PHASE_REGISTRY: dict[str, tuple[str, str]] = {
    # phase key             (deprecated script,            filename)
    "e2e":                 ("verify_e2e.py",                  "verify_e2e.py"),
    "e2e-trt":             ("verify_e2e_trt.py",              "verify_e2e_trt.py"),
    "e2e-trt-ref":         ("verify_e2e_trt_ref.py",          "verify_e2e_trt_ref.py"),
    "fused-triton":        ("verify_fused_triton_backend.py",  "verify_fused_triton_backend.py"),
    "onnx-autoregressive": ("verify_onnx_autoregressive.py",   "verify_onnx_autoregressive.py"),
    "code-predictor":      ("verify_code_predictor_trt.py",    "verify_code_predictor_trt.py"),
    "trt-talker":          ("verify_trt_talker.py",            "verify_trt_talker.py"),
    "multi-variant":       ("verify_multi_variant.py",         "verify_multi_variant.py"),
}

PHASE_HELP: dict[str, str] = {
    "e2e":                 "Full decomposed pipeline: PyTorch vs exported weights + ONNX (Stages A-E)",
    "e2e-trt":             "TRT unified engine decode loop vs FP32 PyTorch reference (.npz)",
    "e2e-trt-ref":         "Generate FP32 PyTorch reference .npz for e2e-trt phase",
    "fused-triton":        "ORT fused ONNX vs Triton TRT fused backend parity",
    "onnx-autoregressive": "Replay TRT dump session through ONNX autoregressively",
    "code-predictor":      "ONNX Runtime vs TRT Code Predictor token parity",
    "trt-talker":          "Talker unified TRT engine load + shape verification",
    "multi-variant":       "Multi-variant weight comparison & prefill verification",
}

def _load_phase_module(phase: str):
    """Load a phase sub-script by file path using importlib."""
    _, filename = PHASE_REGISTRY[phase]
    filepath = _TOOLS_DIR / filename
    module_name = filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filepath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
#  Deprecation notice
# ---------------------------------------------------------------------------

def _print_deprecation(phase: str, old_script: str) -> None:
    """Print a deprecation notice referencing the old script name."""
    print(
        f"[DEPRECATION] {old_script} is deprecated. "
        f"Use: python tools/validation/verify_engine.py --phase {phase} ...",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
#  Phase-specific help printer
# ---------------------------------------------------------------------------

def _print_phase_help(phase: str) -> None:
    """Import the phase module, construct its argparse, and print its help."""
    try:
        mod = _load_phase_module(phase)
        # Most modules define argparse inside main().  We temporarily
        # intercept argparse.ArgumentParser.__init__ to capture the parser
        # without actually running main().
        _captured: list[argparse.ArgumentParser] = []
        _orig_init = argparse.ArgumentParser.__init__

        def _capturing_init(self, *a, **kw):
            _orig_init(self, *a, **kw)
            _captured.append(self)

        argparse.ArgumentParser.__init__ = _capturing_init  # type: ignore[assignment]
        try:
            saved_argv = sys.argv
            sys.argv = [PHASE_REGISTRY[phase][0], "--help"]
            try:
                mod.main()
            except SystemExit:
                pass
            finally:
                sys.argv = saved_argv
        finally:
            argparse.ArgumentParser.__init__ = _orig_init  # type: ignore[assignment]
    except Exception as exc:
        print(f"(Could not retrieve phase help: {exc})", file=sys.stderr)


# ---------------------------------------------------------------------------
#  Phase runners — import the sub-module, rewrite sys.argv, call main()
# ---------------------------------------------------------------------------

def _run_phase(phase: str, argv_tail: list[str]) -> int:
    """Import the phase module, forward remaining args, and invoke its main().

    The sub-module's own argparse handles all phase-specific flags.
    We only set sys.argv so that ``argparse.parse_args()`` inside the
    sub-module sees the flags it expects.
    """
    old_script, _ = PHASE_REGISTRY[phase]
    _print_deprecation(phase, old_script)

    # Handle --phase-help: print the sub-module's full --help, then exit 0.
    if "--phase-help" in argv_tail:
        _print_phase_help(phase)
        return 0

    mod = _load_phase_module(phase)
    # Rewrite sys.argv so the sub-module's own argparse sees its expected flags
    sys.argv = [old_script] + argv_tail

    main_fn = getattr(mod, "main", None)
    if main_fn is None:
        logger.error("Phase module %s has no main() function", old_script)
        return 1

    try:
        rc = main_fn()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1

    return rc if isinstance(rc, int) else 0


# ---------------------------------------------------------------------------
#  Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    phase_choices = sorted(PHASE_REGISTRY.keys())
    phase_desc = "\n".join(f"  {k:22s} {PHASE_HELP[k]}" for k in phase_choices)

    parser = argparse.ArgumentParser(
        prog="verify_engine",
        description=(
            "Unified engine verification tool.\n\n"
            "Select a verification phase with --phase.  All remaining arguments\n"
            "are forwarded to the phase-specific script (each has its own\n"
            "argparse).  Use --phase-help to print the sub-module's full help.\n\n"
            "Available phases:\n"
            + phase_desc
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--phase",
        required=True,
        choices=phase_choices,
        help="Verification phase to run",
    )

    parser.add_argument(
        "--phase-help",
        action="store_true",
        default=False,
        help="Print the full --help for the selected phase's sub-module, then exit",
    )

    return parser


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Parse just --phase (and optional --phase-help); pass everything else
    # through to the sub-module.
    parser = build_parser()
    args, argv_tail = parser.parse_known_args()

    phase = args.phase
    if args.phase_help:
        argv_tail = ["--phase-help"]

    return _run_phase(phase, argv_tail)


if __name__ == "__main__":
    sys.exit(main())
