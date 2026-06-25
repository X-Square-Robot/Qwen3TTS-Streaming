#!/bin/bash
# ===========================================================================
#  export_models.sh — Export Qwen3-TTS models (ONNX + PyTorch weights)
#
#  Runs the Python export pipeline for all (or selected) model variants.
#  Produces: ONNX models and embedding weights (.pt).
#
#  TRT engine compilation (trtexec) is NOT done here — run build_engines.sh
#  after this script completes to compile TRT engines inside an NGC container.
#
#  Prerequisites:
#    Activate the Python virtual environment created by setup_env.sh:
#      conda activate qwen3-tts    # or: source <venv-path>/bin/activate
#
#  Usage:
#    bash scripts/bash/export_models.sh                         # all variants, bf16, auto GPU
#    bash scripts/bash/export_models.sh --variant custom-1.7b   # one variant
#    bash scripts/bash/export_models.sh --dtype fp32            # fp32 precision
#    bash scripts/bash/export_models.sh --device cpu            # force CPU
#
#  Can be called standalone or from setup_env.sh (Step 9).
#  Automatically installs missing export dependencies.
#
#  Output: workspace/exported/<variant>/
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

EXPORT_DIR="${REPO_ROOT}/scripts/export"

# ---------------------------------------------------------------------------
log_step "Qwen3-TTS Model Export"
log_info "Export scripts: ${EXPORT_DIR}"
log_info "Arguments: $*"

# Ensure core dependencies
if ! python3 -c "import torch" 2>/dev/null; then
    log_error "PyTorch not found. Run 'autorun.sh setup' first or install manually."
    exit 1
fi

if ! python3 -c "import qwen_tts" 2>/dev/null; then
    log_error "qwen_tts package not found."
    log_error "Run 'autorun.sh setup' first or: pip install -e ${REPO_ROOT}/third_party/Qwen3-TTS"
    exit 1
fi

# Auto-install export dependencies (idempotent)
install_onnx_export_deps
install_safetensors

# Run the master export script
cd "${EXPORT_DIR}"
python3 export_all.py "$@"

log_info "Export complete. Check workspace/exported/ for results."
log_info "Next: run build_engines.sh to compile TRT engines (trtexec)."
