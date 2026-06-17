#!/bin/bash
# ===========================================================================
#  L2 test T2.2: Dockerfile.triton build and runtime checks.
#
#  Verifies the slim Triton runtime image:
#    - Build succeeds (qwen3-tts-triton:<tag> or custom BUILD_TAG)
#    - Container can import torch, tokenizers, tensorrt
#    - tritonserver --help works
#
#  Run from repo root:
#    bash tools/validation/dockerfile_triton_check.sh
#  Optional: BUILD_TAG=qwen3-tts-triton:26.01 bash tools/validation/dockerfile_triton_check.sh
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LIB_DIR="${REPO_ROOT}/scripts/bash/lib"
source "${LIB_DIR}/logging.sh"
source "${LIB_DIR}/utils.sh"
source "${LIB_DIR}/docker.sh"
source "${LIB_DIR}/triton.sh"

BUILD_TAG="${BUILD_TAG:-}"
NGC_TAG=""
if [ -z "$BUILD_TAG" ]; then
  NGC_TAG=$(resolve_ngc_tag) || { log_error "resolve_ngc_tag failed"; exit 1; }
  BUILD_TAG="${_DEPLOY_IMAGE_NAME}:${NGC_TAG}"
fi

log_step "T2.2 Dockerfile.triton test (image: $BUILD_TAG)"

if ! docker image inspect "$BUILD_TAG" &>/dev/null; then
  log_info "Building deploy image..."
  build_triton_deploy_image "${NGC_TAG:-}" || {
    log_error "build_triton_deploy_image failed"
    exit 1
  }
  if [ -n "$NGC_TAG" ]; then
    BUILD_TAG="${_DEPLOY_IMAGE_NAME}:${NGC_TAG}"
  fi
fi

log_info "Checking torch, tokenizers, tensorrt..."
docker run --rm "$BUILD_TAG" python3 -c "
import sys
import torch
print(f'torch {torch.__version__}', file=sys.stderr)
import tokenizers
print(f'tokenizers {tokenizers.__version__}', file=sys.stderr)
import tensorrt
print(f'tensorrt {tensorrt.__version__}', file=sys.stderr)
" || { log_error "torch/tokenizers/tensorrt check failed"; exit 1; }

log_info "Checking baked engine package under /opt/qwen3-tts..."
docker run --rm "$BUILD_TAG" python3 -c "
import sys
sys.path.insert(0, '/opt/qwen3-tts')
import engine
print('engine OK', file=sys.stderr)
" || { log_error "baked engine import failed"; exit 1; }

log_info "Checking tritonserver..."
# Triton may exit non-zero when no GPU is present; verify it runs and prints version/help
out=$(docker run --rm "$BUILD_TAG" tritonserver --help 2>&1) || true
if ! echo "$out" | grep -q "Triton Inference Server"; then
  log_error "tritonserver --help did not show expected output"
  exit 1
fi

log_info "T2.2 test_dockerfile_triton.sh passed"
