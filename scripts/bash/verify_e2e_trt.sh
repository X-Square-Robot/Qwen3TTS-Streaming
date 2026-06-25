#!/bin/bash
# ===========================================================================
#  verify_e2e_trt.sh — TRT E2E verification (talker_context + talker_decode_fused)
#
#  Two-step flow:
#    1. Host: Generate PyTorch reference (verify_e2e_trt_ref.py) → e2e_trt_ref.npz
#    2. Container: Run TRT engines vs reference (verify_e2e_trt.py) → e2e_trt_report.json
#
#  Prerequisites:
#    - TRT engines: workspace/exported/<variant>/talker_context.engine,
#      talker_decode_fused.engine (from build_engines.sh)
#    - For step 1: conda env qwen3-tts; workspace/exported/<variant>/weights/
#    - Docker with NVIDIA Container Toolkit
#
#  Usage:
#    bash scripts/bash/verify_e2e_trt.sh --variant design-1.7b
#    bash scripts/bash/verify_e2e_trt.sh --variant design-1.7b --steps 50
#    bash scripts/bash/verify_e2e_trt.sh --variant design-1.7b --skip-ref
#    bash scripts/bash/verify_e2e_trt.sh --gpu 1 --variant design-1.7b
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
SCRIPTS_PY="${REPO_ROOT}/scripts/python"
TESTS_TOOLS="${REPO_ROOT}/tests/tools"
VARIANT=""
N_STEPS=50
SKIP_REF=false
DRY_RUN=false
USER_IMAGE="${NGC_IMAGE:-}"
GPU_DEVICES="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)    VARIANT="$2"; shift 2 ;;
        --steps)      N_STEPS="$2"; shift 2 ;;
        --skip-ref)   SKIP_REF=true; shift ;;
        --image)      USER_IMAGE="$2"; shift 2 ;;
        --gpu)        GPU_DEVICES="device=$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --variant <name>   Model variant (e.g. design-1.7b)"
            echo "  --steps <N>        Decode steps for ref + TRT (default: 50)"
            echo "  --skip-ref        Skip host ref gen; only run container verify"
            echo "  --image <uri>     Override NGC container image"
            echo "  --gpu <id>        Use only GPU id (e.g. 1)"
            echo "  --dry-run         Show commands without executing"
            exit 0
            ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$VARIANT" ]; then
    log_error "Please specify --variant (e.g. design-1.7b)"
    exit 1
fi

VARIANT_DIR="$EXPORTED_DIR/$VARIANT"
if [ ! -d "$VARIANT_DIR" ]; then
    log_error "Variant dir not found: $VARIANT_DIR"
    exit 1
fi
if [ ! -f "$VARIANT_DIR/talker_context.engine" ] || [ ! -f "$VARIANT_DIR/talker_decode_fused.engine" ]; then
    log_error "TRT engines not found: $VARIANT_DIR/talker_context.engine and talker_decode_fused.engine. Run build_engines.sh first."
    exit 1
fi

# ── Step 1: Generate PyTorch reference on host ──
run_ref_gen() {
    if $SKIP_REF; then
        if [ ! -f "$VARIANT_DIR/e2e_trt_ref.npz" ]; then
            log_error "Reference missing: $VARIANT_DIR/e2e_trt_ref.npz (run without --skip-ref first)"
            return 1
        fi
        log_info "Using existing reference: $VARIANT_DIR/e2e_trt_ref.npz"
        return 0
    fi
    if [ ! -f "$VARIANT_DIR/weights/config.json" ]; then
        log_error "Weights missing: $VARIANT_DIR/weights/. Run export_models.sh first."
        return 1
    fi
    log_step "Generating E2E TRT reference: $VARIANT (steps=$N_STEPS)"
    if $DRY_RUN; then
        log_info "[DRY RUN] Would run: python3 tests/tools/verify_e2e_trt_ref.py --variant $VARIANT --steps $N_STEPS"
        return 0
    fi
    ( cd "${REPO_ROOT}" && python3 "${TESTS_TOOLS}/verify_e2e_trt_ref.py" --variant "$VARIANT" --steps "$N_STEPS" ) || return 1
    log_info "Reference saved: $VARIANT_DIR/e2e_trt_ref.npz"
    return 0
}

# ── Step 2: Run TRT E2E verification in container ──
run_container_verify() {
    if [ ! -f "$VARIANT_DIR/e2e_trt_ref.npz" ]; then
        log_error "Reference file missing: $VARIANT_DIR/e2e_trt_ref.npz"
        return 1
    fi
    log_step "Running TRT E2E verification: $VARIANT"
    local docker_cmd=(
        docker run --rm --gpus "$GPU_DEVICES"
        -v "$VARIANT_DIR:/mnt/model"
        -v "$TESTS_TOOLS:/mnt/tests:ro"
        -v "$SCRIPTS_PY:/mnt/scripts:ro"
        "$NGC_IMAGE"
        bash -c "pip install -q onnxruntime 2>/dev/null; python3 /mnt/tests/verify_e2e_trt.py --model-dir /mnt/model --ref-file /mnt/model/e2e_trt_ref.npz --variant $VARIANT"
    )
    if $DRY_RUN; then
        log_info "[DRY RUN] Would execute:"
        log_info "  ${docker_cmd[*]}"
        return 0
    fi
    if "${docker_cmd[@]}"; then
        log_info "E2E TRT verification PASSED: $VARIANT"
        return 0
    else
        log_error "E2E TRT verification FAILED: $VARIANT"
        return 1
    fi
}

# ── Resolve container image ──
log_step "TRT E2E Verification (Phase 2, Item 12)"

if ! $SKIP_REF; then
    log_info "Step 1: PyTorch reference (host); Step 2: TRT verify (container)"
else
    log_info "Step 2 only: TRT verify (container)"
fi

if [ -n "$USER_IMAGE" ]; then
    NGC_IMAGE="$USER_IMAGE"
    log_info "Using image: $NGC_IMAGE"
else
    log_info "Auto-detecting NGC container ..."
    _NGC_VERIFY_MANIFEST=1
    NGC_IMAGE=$(resolve_ngc_image_info) \
        || { log_error "Cannot determine container. Use --image to specify."; exit 1; }
fi

if ! $SKIP_REF || ! $DRY_RUN; then
    check_docker_gpu_ready || exit 1
    ensure_ngc_image "$NGC_IMAGE" || exit 1
fi

if run_ref_gen && run_container_verify; then
    log_info ""
    log_info "Report: $VARIANT_DIR/e2e_trt_report.json"
    exit 0
else
    exit 1
fi
