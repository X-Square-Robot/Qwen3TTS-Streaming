#!/bin/bash
# ===========================================================================
#  build_on_target.sh — Compile TensorRT engines from a build bundle
#
#  Bundle layout (created by make_engine_build_bundle):
#    .
#    ├── target_profile.json
#    ├── build_manifest.json
#    ├── scripts/                ← full repo scripts/ copy
#    ├── workspace/exported/<variant>/*.onnx
#    └── workspace/exported/tokenizer/*.onnx
#
#  This script:
#    1. Validates that the host SM matches the target_profile (and warns on
#       driver mismatch).
#    2. Resolves the trtexec runner (docker or host) and compiles all
#       variants via compile_engines_in_bundle.
#    3. Writes artifact_manifest.json with sha256 / TensorRT / SM / driver.
#    4. Packs engine_artifact_bundle.tar.zst.
#
#  Designed to run unattended on the production-like GPU host (which may
#  not have Docker — e.g. Aliyun DSW containers).  Set BUILD_RUNNER=host
#  to force host trtexec when the auto-detection picks the wrong path.
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || pwd)"
export REPO_ROOT

# Defensive tools.sh path resolution — usable both inside a bundle layout
# (./scripts/bash/tools.sh) and when copied next to scripts/.
_TOOLS_CANDIDATES=(
    "${SCRIPT_DIR}/scripts/bash/tools.sh"
    "${SCRIPT_DIR}/tools.sh"
    "${REPO_ROOT}/scripts/bash/tools.sh"
)
_TOOLS_PATH=""
for _t in "${_TOOLS_CANDIDATES[@]}"; do
    if [ -f "$_t" ]; then
        _TOOLS_PATH="$_t"
        break
    fi
done
if [ -z "$_TOOLS_PATH" ]; then
    echo "[ERROR] Cannot locate tools.sh." >&2
    echo "        Looked for:" >&2
    for _t in "${_TOOLS_CANDIDATES[@]}"; do echo "          $_t" >&2; done
    echo "        Did you extract the engine_build_bundle.tar.zst correctly?" >&2
    echo "        The bundle layout should be:" >&2
    echo "          .  ←  bundle root (you should run from here)" >&2
    echo "          ├── run.sh / build_on_target.sh" >&2
    echo "          ├── scripts/bash/tools.sh" >&2
    echo "          └── workspace/exported/<variant>/..." >&2
    exit 1
fi
source "$_TOOLS_PATH"

BUNDLE_ROOT="$SCRIPT_DIR"
BUILD_MANIFEST="$BUNDLE_ROOT/build_manifest.json"
TARGET_PROFILE="$BUNDLE_ROOT/target_profile.json"
OUT="${OUT:-$BUNDLE_ROOT/engine_artifact_bundle.tar.zst}"

[ -f "$BUILD_MANIFEST" ] || { log_error "Missing build_manifest.json"; exit 1; }
[ -f "$TARGET_PROFILE" ] || { log_error "Missing target_profile.json"; exit 1; }

log_step "Validating target host profile"
expected_driver=$(cross_host_json_value "$TARGET_PROFILE" driver_version)
expected_ngc_tag=$(cross_host_json_value "$BUILD_MANIFEST" ngc_tag)
build_device=$(cross_host_json_value "$BUILD_MANIFEST" build_gpu_device || echo auto)

if [ "$build_device" = "auto" ] || [ "$build_device" = "all" ] || [ -z "$build_device" ]; then
    probe_device=0
else
    probe_device="${build_device#cuda:}"
    probe_device="${probe_device#device=}"
fi

expected_sm=$(python3 - "$TARGET_PROFILE" "$probe_device" <<'PY'
import json
import sys
path, gpu_index = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as f:
    data = json.load(f)
gpus = data.get("gpus") or []
gpu = next((g for g in gpus if int(g.get("index", -1)) == gpu_index), gpus[0] if gpus else {})
print(gpu.get("sm", ""))
PY
)

actual_driver=$(detect_driver_version) || { log_error "Cannot detect target driver"; exit 1; }
if [ "$actual_driver" != "$expected_driver" ]; then
    log_warn "Driver differs from target_profile: expected=$expected_driver actual=$actual_driver"
fi

actual_cc=$(nvidia-smi --id="$probe_device" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -d ' ') \
    || { log_error "Cannot detect GPU compute capability"; exit 1; }
actual_sm="sm_${actual_cc//./}"
if [ -n "$expected_sm" ] && [ "$actual_sm" != "$expected_sm" ]; then
    log_error "GPU SM mismatch: expected=$expected_sm actual=$actual_sm"
    log_error "The engines compiled here will not load on the target host."
    exit 1
fi

# Resolve runner first so we know whether to expect docker or host trtexec.
runner=$(resolve_trtexec_runner) || {
    log_error "Cannot resolve trtexec runner on this host"
    log_error "Set BUILD_RUNNER=host with TRTEXEC_HOST=<path>, or install docker + NVIDIA toolkit"
    exit 1
}
log_info "Runner selected: $runner"

# Surface NGC_IMAGE so trtexec_runner_docker can pick it up if needed.
NGC_TAG="$expected_ngc_tag"
NGC_IMAGE=$(cross_host_json_value "$BUILD_MANIFEST" ngc_image)
export NGC_TAG NGC_IMAGE

log_step "Compiling TensorRT engines on target"
compile_engines_in_bundle "$BUNDLE_ROOT" || {
    log_error "Engine compilation failed inside bundle"
    exit 1
}

log_step "Writing artifact manifest"
write_artifact_manifest "$BUNDLE_ROOT" || {
    log_error "Failed to write artifact_manifest.json"
    exit 1
}

log_step "Packaging engine artifact bundle"
pack_engine_artifact_bundle "$BUNDLE_ROOT" "$OUT" || {
    log_error "Failed to package engine artifact bundle"
    exit 1
}

log_info "Done. Output: $OUT"
log_info "On the packaging machine, run:"
log_info "  bash scripts/bash/autorun.sh import-artifact $(basename "$OUT")"
