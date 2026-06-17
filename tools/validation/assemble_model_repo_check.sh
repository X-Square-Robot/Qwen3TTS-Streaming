#!/bin/bash
# ===========================================================================
#  L2 test T2.4: assemble_model_repo ONNX/TRT dual mode and validate.
#
#  Verifies:
#    - ONNX mode: fused runtime ONNX is copied under tts_orchestrator/1/runtime/,
#                 tokenizer assets and TTSEngine payload are present
#    - TRT mode:  fused runtime TRT plan is copied under tts_orchestrator/1/runtime/
#    - Both: orchestrator config carries model_package_dir/first_chunk_frames and no
#            top-level talker_code2wav_fused Triton model is exposed
#
#  Run from repo root:
#    bash tools/validation/assemble_model_repo_check.sh
#  Requires: workspace/exported/<variant>/ with ONNX (and optionally .engine for trt).
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
VARIANT="${VARIANT:-base-1.7b}"

# Source triton lib (and deps) without full tools.sh to avoid venv/network
LIB_DIR="${REPO_ROOT}/scripts/bash/lib"
source "${LIB_DIR}/logging.sh"
source "${LIB_DIR}/utils.sh"
source "${LIB_DIR}/docker.sh"
source "${LIB_DIR}/triton.sh"

TEST_REPO_ONNX="${TEST_REPO_ONNX:-${REPO_ROOT}/workspace/test_repo_onnx}"
TEST_REPO_TRT="${TEST_REPO_TRT:-${REPO_ROOT}/workspace/test_repo_trt}"
TEST_REPO_V2="${TEST_REPO_V2:-${REPO_ROOT}/workspace/test_repo_v2}"

log_step "T2.4 assemble_model_repo tests (variant=$VARIANT)"

if [ ! -d "${EXPORTED_DIR}/${VARIANT}" ]; then
  log_error "Variant dir not found: ${EXPORTED_DIR}/${VARIANT}. Run Phase A export first."
  exit 1
fi

# ---- ONNX mode ----
log_info "Assembling ONNX mode -> ${TEST_REPO_ONNX}"
rm -rf "${TEST_REPO_ONNX}"
assemble_model_repo "${EXPORTED_DIR}" "${VARIANT}" "${TEST_REPO_ONNX}" "onnx" || { log_error "assemble (onnx) failed"; exit 1; }
validate_model_repo "${TEST_REPO_ONNX}" || { log_error "validate (onnx) failed"; exit 1; }

# Assert ONNX-specific
ORCH_1="${TEST_REPO_ONNX}/tts_orchestrator/1"
ORCH_HTTP_1="${TEST_REPO_ONNX}/tts_orchestrator_http/1"
RUNTIME_ONNX="${ORCH_1}/runtime"
[ -f "${ORCH_1}/model.py" ] || { log_error "Missing ${ORCH_1}/model.py"; exit 1; }
[ -f "${ORCH_HTTP_1}/model.py" ] || { log_error "Missing ${ORCH_HTTP_1}/model.py"; exit 1; }
[ -f "${ORCH_1}/engine/server.py" ] || { log_error "Missing TTSEngine payload: engine/server.py"; exit 1; }
[ -f "${ORCH_1}/engine/frontend/interface.py" ] || { log_error "Missing TTSEngine payload: engine/frontend/interface.py"; exit 1; }
[ -f "${ORCH_1}/engine/backend/engine_loop.py" ] || { log_error "Missing TTSEngine payload: engine/backend/engine_loop.py"; exit 1; }
[ -f "${ORCH_1}/weights/config.json" ] || { log_error "Missing orchestrator weights/config.json"; exit 1; }
[ -f "${ORCH_1}/resources/speakers/liangwuque/ref.wav" ] || { log_error "Missing packaged speaker ref.wav"; exit 1; }
[ -f "${ORCH_1}/resources/speakers/liangwuque/ref.txt" ] || { log_error "Missing packaged speaker ref.txt"; exit 1; }
[ -f "${ORCH_1}/triton_manifest.json" ] || { log_error "Missing package-root manifest"; exit 1; }
[ -f "${RUNTIME_ONNX}/triton_manifest.json" ] || { log_error "Missing runtime manifest"; exit 1; }
[ -f "${RUNTIME_ONNX}/model.onnx" ] || { log_error "Missing runtime/model.onnx"; exit 1; }
[ ! -e "${TEST_REPO_ONNX}/talker_code2wav_fused" ] || { log_error "Unexpected top-level talker_code2wav_fused model in ONNX repo"; exit 1; }
for legacy in __pycache__ greedy_tokenizer.py batch_decode_scheduler.py text_segmenter.py \
              prefill_builder.py audio_utils.py lightweight_tokenizer.py \
              session_manager.py decode_fsm.py ratio_tracker.py mlfq_scheduler.py; do
  [ ! -e "${ORCH_1}/${legacy}" ] || { log_error "Unexpected legacy BLS payload remained: ${ORCH_1}/${legacy}"; exit 1; }
done
grep -q "first_chunk_frames" "${TEST_REPO_ONNX}/tts_orchestrator/config.pbtxt" || { log_error "Orchestrator config missing first_chunk_frames"; exit 1; }
grep -q 'key: "model_package_dir"' "${TEST_REPO_ONNX}/tts_orchestrator/config.pbtxt" || { log_error "Orchestrator config missing model_package_dir"; exit 1; }
grep -q '/models/tts_orchestrator/1' "${TEST_REPO_ONNX}/tts_orchestrator/config.pbtxt" || { log_error "Orchestrator config has wrong model_package_dir"; exit 1; }
grep -q 'name: "tts_orchestrator_http"' "${TEST_REPO_ONNX}/tts_orchestrator_http/config.pbtxt" || { log_error "HTTP orchestrator config missing model name"; exit 1; }
grep -q 'key: "target_model"' "${TEST_REPO_ONNX}/tts_orchestrator_http/config.pbtxt" || { log_error "HTTP orchestrator config missing target_model"; exit 1; }
if [ -d "${ORCH_1}/tokenizer" ]; then
  if [ -f "${ORCH_1}/tokenizer/tokenizer.json" ]; then
    log_info "  tokenizer.json present in orchestrator"
  fi
fi
log_info "ONNX assemble checks passed"

# ---- ONNX mode with alternate model version ----
log_info "Assembling ONNX mode with version 2 -> ${TEST_REPO_V2}"
rm -rf "${TEST_REPO_V2}"
assemble_model_repo "${EXPORTED_DIR}" "${VARIANT}" "${TEST_REPO_V2}" "onnx" "2" || { log_error "assemble (onnx, v2) failed"; exit 1; }
validate_model_repo "${TEST_REPO_V2}" || { log_error "validate (onnx, v2) failed"; exit 1; }

ORCH_2="${TEST_REPO_V2}/tts_orchestrator/2"
ORCH_HTTP_2="${TEST_REPO_V2}/tts_orchestrator_http/2"
RUNTIME_ONNX_V2="${ORCH_2}/runtime"
[ -f "${ORCH_2}/model.py" ] || { log_error "Missing ${ORCH_2}/model.py"; exit 1; }
[ -f "${ORCH_HTTP_2}/model.py" ] || { log_error "Missing ${ORCH_HTTP_2}/model.py"; exit 1; }
[ -f "${ORCH_2}/resources/speakers/liangwuque/ref.wav" ] || { log_error "Missing v2 packaged speaker ref.wav"; exit 1; }
[ -f "${ORCH_2}/resources/speakers/liangwuque/ref.txt" ] || { log_error "Missing v2 packaged speaker ref.txt"; exit 1; }
[ -f "${ORCH_2}/triton_manifest.json" ] || { log_error "Missing v2 package-root manifest"; exit 1; }
[ -f "${RUNTIME_ONNX_V2}/triton_manifest.json" ] || { log_error "Missing v2 runtime manifest"; exit 1; }
[ -f "${RUNTIME_ONNX_V2}/model.onnx" ] || { log_error "Missing v2 runtime/model.onnx"; exit 1; }
grep -q '/models/tts_orchestrator/2' "${TEST_REPO_V2}/tts_orchestrator/config.pbtxt" || { log_error "V2 orchestrator config has wrong model_package_dir"; exit 1; }
grep -q 'name: "tts_orchestrator_http"' "${TEST_REPO_V2}/tts_orchestrator_http/config.pbtxt" || { log_error "V2 HTTP orchestrator config missing model name"; exit 1; }
log_info "ONNX assemble checks for version 2 passed"

# ---- TRT mode (if .engine exists) ----
if [ -f "${EXPORTED_DIR}/${VARIANT}/talker_code2wav_fused.engine" ] || [ -f "${EXPORTED_DIR}/${VARIANT}/talker_code2wav_fused.plan" ]; then
  log_info "Assembling TRT mode -> ${TEST_REPO_TRT}"
  rm -rf "${TEST_REPO_TRT}"
  assemble_model_repo "${EXPORTED_DIR}" "${VARIANT}" "${TEST_REPO_TRT}" "trt" || { log_error "assemble (trt) failed"; exit 1; }
  validate_model_repo "${TEST_REPO_TRT}" || { log_error "validate (trt) failed"; exit 1; }
  RUNTIME_TRT="${TEST_REPO_TRT}/tts_orchestrator/1/runtime"
  [ -f "${TEST_REPO_TRT}/tts_orchestrator/1/triton_manifest.json" ] || { log_error "Missing TRT package-root manifest"; exit 1; }
  [ -f "${TEST_REPO_TRT}/tts_orchestrator/1/resources/speakers/liangwuque/ref.wav" ] || { log_error "Missing TRT packaged speaker ref.wav"; exit 1; }
  [ -f "${TEST_REPO_TRT}/tts_orchestrator/1/resources/speakers/liangwuque/ref.txt" ] || { log_error "Missing TRT packaged speaker ref.txt"; exit 1; }
  [ -f "${RUNTIME_TRT}/triton_manifest.json" ] || { log_error "Missing TRT runtime manifest"; exit 1; }
  [ -f "${RUNTIME_TRT}/model.plan" ] || { log_error "Missing TRT runtime/model.plan"; exit 1; }
  if [ -f "${EXPORTED_DIR}/tokenizer/code2wav_decoder.engine" ]; then
    [ -f "${TEST_REPO_TRT}/tts_orchestrator/1/tokenizer/code2wav_decoder.engine" ] \
      || { log_error "Missing TRT tokenizer/code2wav_decoder.engine for ICL warm state"; exit 1; }
  fi
  [ ! -e "${TEST_REPO_TRT}/talker_code2wav_fused" ] || { log_error "Unexpected top-level talker_code2wav_fused model in TRT repo"; exit 1; }
  grep -q 'key: "model_package_dir"' "${TEST_REPO_TRT}/tts_orchestrator/config.pbtxt" || { log_error "TRT orchestrator config missing model_package_dir"; exit 1; }
  grep -q '/models/tts_orchestrator/1' "${TEST_REPO_TRT}/tts_orchestrator/config.pbtxt" || { log_error "TRT orchestrator config has wrong model_package_dir"; exit 1; }
  [ -f "${TEST_REPO_TRT}/tts_orchestrator_http/1/model.py" ] || { log_error "Missing TRT HTTP orchestrator model.py"; exit 1; }
  log_info "TRT assemble checks passed"
else
  log_warn "No talker_code2wav_fused.engine/.plan found; skipping TRT assemble test"
fi

log_info "T2.4 test_assemble.sh passed"
