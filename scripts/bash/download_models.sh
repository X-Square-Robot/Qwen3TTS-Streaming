#!/bin/bash
# ===========================================================================
#  download_models.sh — Standalone Qwen3-TTS model downloader
#
#  Can be used independently of setup_env.sh / autorun.sh to download or update models.
#  Interactively asks which model variant to download when not specified.
#  Supports version pinning via model_versions.conf.
#
#  Usage:
#    bash scripts/bash/download_models.sh [target_dir] [variant] [source]
#    bash scripts/bash/download_models.sh --pin [target_dir]
#
#  Examples:
#    bash scripts/bash/download_models.sh                      # interactive
#    bash scripts/bash/download_models.sh ./models base-1.7b   # explicit
#    bash scripts/bash/download_models.sh ./models all hf      # all via HF
#    bash scripts/bash/download_models.sh --pin                # pin versions
#    MODEL_VARIANT=base-1.7b bash scripts/bash/download_models.sh
#
#  Environment variables:
#    MODEL_VARIANT    Model variant (if unset, interactive prompt)
#    MODEL_SOURCE     Download source (default: auto)
#    HF_MIRROR        HuggingFace mirror URL (for China users)
#    HF_ENDPOINT      HuggingFace API endpoint override
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

# Handle flags before positional args
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: $(basename "$0") [target_dir] [variant] [source]"
    echo "       $(basename "$0") --pin [target_dir]"
    echo ""
    list_model_variants
    echo ""
    echo "Sources: auto (default), hf (HuggingFace), modelscope"
    echo ""
    echo "Options:"
    echo "  --pin       Pin currently downloaded model revisions to"
    echo "              model_versions.conf for reproducibility"
    echo "  --list, -l  List available model variants"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_VARIANT    Model variant (interactive prompt if unset)"
    echo "  MODEL_SOURCE     Download source (default: auto)"
    echo "  HF_MIRROR        HuggingFace mirror URL"
    echo "  HF_ENDPOINT      HuggingFace API endpoint"
    exit 0
fi

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then
    list_model_variants
    exit 0
fi

if [ "${1:-}" = "--pin" ]; then
    TARGET_DIR="${2:-${REPO_ROOT}/workspace/models}"
    log_step "Pinning model revisions..."
    pin_model_versions "$TARGET_DIR"
    log_info "Revisions pinned to: ${SCRIPT_DIR}/model_versions.conf"
    log_info "Commit this file to lock model versions for reproducibility."
    exit 0
fi

TARGET_DIR="${1:-${REPO_ROOT}/workspace/models}"
VARIANT="${MODEL_VARIANT:-${2:-}}"
SOURCE="${MODEL_SOURCE:-${3:-auto}}"

# If variant not explicitly specified, ask the user
if [ -z "$VARIANT" ]; then
    VARIANT=$(select_model_variant)
fi
VARIANT="${VARIANT:-base-1.7b}"

# ---------------------------------------------------------------------------

echo ""
echo -e "${_CLR_BLUE}Qwen3-TTS Model Downloader${_CLR_RESET}"
echo ""
echo "  Target:    $TARGET_DIR"
echo "  Variant:   $VARIANT"
echo "  Source:     $SOURCE"
echo ""

ensure_git_lfs || exit 1

download_qwen3_tts_models "$TARGET_DIR" "$VARIANT" "$SOURCE"

log_info "Done. Models are at: $TARGET_DIR"
log_info "To pin these versions: $(basename "$0") --pin $TARGET_DIR"
