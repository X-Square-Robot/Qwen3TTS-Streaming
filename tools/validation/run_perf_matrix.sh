#!/bin/bash
# ===========================================================================
#  Serving performance benchmark matrix: protocols x concurrency levels.
#
#  Drives tools/validation/perf_matrix_sdk.py -- which talks to the engine
#  through the real qwen3tts client SDK, not a hand-rolled protocol client --
#  against already-running engine/Triton services. Saves one raw JSON file
#  per (target, level) combination under workspace/perf_matrix/<run_id>/,
#  plus a connection-time isolation pass (reuse vs. cold, uniform across all
#  three transports since perf_matrix_sdk.py's --connection-mode works the
#  same way for all of them). Raw output is consumed by
#  summarize_perf_matrix.py.
#
#  Does NOT deploy anything -- engine-grpc/engine-websocket and Triton must
#  already be up (see `compose.sh up`). This only issues client requests.
#
#  Run from repo root:
#    bash tools/validation/run_perf_matrix.sh
#
#  Env overrides (all optional):
#    TARGETS                default: engine-grpc,engine-websocket,triton-grpc
#    LEVELS                  default: 1,16,32,64,128
#    CONCURRENCY_SAMPLES      default: 8   (measured rounds per level)
#    CONCURRENCY_WARMUP       default: 2   (warmup rounds per level, excluded)
#    CONN_SAMPLES             default: 30  (measured rounds per connection mode)
#    CONN_WARMUP              default: 3
#    ENGINE_GRPC_ENDPOINT     default: localhost:50051
#    ENGINE_WS_ENDPOINT       default: ws://localhost:50052/v1/ws
#    TRITON_GRPC_ENDPOINT     default: localhost:8001
#    RUN_ID                   default: UTC timestamp
#    PYTHON_BIN               default: python
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LIB_DIR="${REPO_ROOT}/scripts/bash/lib"
source "${LIB_DIR}/logging.sh"

export PYTHONPATH="${REPO_ROOT}/client/src${PYTHONPATH:+:${PYTHONPATH}}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${REPO_ROOT}/workspace/perf_matrix/${RUN_ID}"
TARGETS="${TARGETS:-engine-grpc,engine-websocket,triton-grpc}"
LEVELS="${LEVELS:-1,16,32,64,128}"
CONCURRENCY_SAMPLES="${CONCURRENCY_SAMPLES:-8}"
CONCURRENCY_WARMUP="${CONCURRENCY_WARMUP:-2}"
CONN_SAMPLES="${CONN_SAMPLES:-30}"
CONN_WARMUP="${CONN_WARMUP:-3}"
ENGINE_GRPC_ENDPOINT="${ENGINE_GRPC_ENDPOINT:-localhost:50051}"
ENGINE_WS_ENDPOINT="${ENGINE_WS_ENDPOINT:-ws://localhost:50052/v1/ws}"
TRITON_GRPC_ENDPOINT="${TRITON_GRPC_ENDPOINT:-localhost:8001}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PERF_TOOL="${REPO_ROOT}/tools/validation/perf_matrix_sdk.py"

_endpoint_for() {
  case "$1" in
    engine-grpc) echo "$ENGINE_GRPC_ENDPOINT" ;;
    engine-websocket) echo "$ENGINE_WS_ENDPOINT" ;;
    triton-grpc) echo "$TRITON_GRPC_ENDPOINT" ;;
    *) log_error "unknown target: $1"; exit 1 ;;
  esac
}

mkdir -p "$OUT_DIR"
log_step "Perf matrix run_id=${RUN_ID} targets=${TARGETS} levels=${LEVELS} -> ${OUT_DIR}"

# ---- environment snapshot (raw; summarize_perf_matrix.py structures it) ----
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv \
  > "${OUT_DIR}/nvidia_smi.csv" 2>&1 || log_warn "nvidia-smi snapshot failed"
curl -sf http://localhost:8080/health > "${OUT_DIR}/engine_health.json" 2>/dev/null \
  || log_warn "engine health snapshot failed (is engine-grpc up on :8080?)"
docker inspect qwen3-engine --format '{{.Config.Image}} created={{.Created}}' \
  > "${OUT_DIR}/engine_image.txt" 2>/dev/null || true
docker inspect qwen3tts-streaming --format '{{.Config.Image}} created={{.Created}}' \
  > "${OUT_DIR}/triton_image.txt" 2>/dev/null || true
cp "${REPO_ROOT}/engine.yaml" "${OUT_DIR}/engine.yaml.snapshot" 2>/dev/null || true

run_case() {
  # run_case <out_basename> <perf_matrix_sdk.py args...>
  local out_base="$1"
  shift
  local out_file="${OUT_DIR}/${out_base}.json"
  local log_file="${OUT_DIR}/${out_base}.log"
  log_info "  -> ${out_base}"
  if ! "$PYTHON_BIN" "$PERF_TOOL" "$@" --json > "$out_file" 2> "$log_file"; then
    log_warn "     non-zero exit for ${out_base}; see ${log_file}"
  fi
}

IFS=',' read -ra TARGET_ARR <<< "$TARGETS"
IFS=',' read -ra LEVEL_ARR <<< "$LEVELS"

# ---- 1. concurrency sweep: protocol x level ----
for target in "${TARGET_ARR[@]}"; do
  endpoint="$(_endpoint_for "$target")"
  log_step "Concurrency sweep: ${target} (${endpoint})"
  for level in "${LEVEL_ARR[@]}"; do
    run_case "${target}_c${level}" \
      --endpoint "$endpoint" \
      --transport "$target" \
      --concurrency "$level" \
      --rounds "$CONCURRENCY_SAMPLES" \
      --warmup-rounds "$CONCURRENCY_WARMUP" \
      --connection-mode reuse
  done
done

# ---- 2. connection-time isolation: reuse vs. cold, uniform across all 3 ----
for target in "${TARGET_ARR[@]}"; do
  endpoint="$(_endpoint_for "$target")"
  log_step "Connection-time isolation: ${target} (reuse/cold)"
  run_case "${target}_conn-reuse" \
    --endpoint "$endpoint" \
    --transport "$target" \
    --concurrency 1 \
    --rounds "$CONN_SAMPLES" \
    --warmup-rounds "$CONN_WARMUP" \
    --connection-mode reuse
  run_case "${target}_conn-cold" \
    --endpoint "$endpoint" \
    --transport "$target" \
    --concurrency 1 \
    --rounds "$CONN_SAMPLES" \
    --warmup-rounds "$CONN_WARMUP" \
    --connection-mode cold
done

log_info "Perf matrix done. Raw JSON in ${OUT_DIR}"
log_info "Next: python tools/validation/summarize_perf_matrix.py ${OUT_DIR}"
