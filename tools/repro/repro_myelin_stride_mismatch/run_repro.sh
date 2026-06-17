#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-both}" # fail | pass | both
OUT_ROOT="${2:-/tmp/myelin_repro_case}"
IMAGE="${IMAGE:-nvcr.io/nvidia/tritonserver:26.02-py3}"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

if [[ "${MODE}" != "fail" && "${MODE}" != "pass" && "${MODE}" != "both" ]]; then
  echo "[error] MODE must be fail|pass|both, got: ${MODE}"
  exit 2
fi

generate_shapes() {
  local static_state="${1:-0}" # 0 dynamic, 1 static(batch=8)
  local max_b=8
  local min="codes:1x16x4,cache_position:1x4,c2w_attention_bias:1x1x4x5"
  local opt="codes:1x16x4,cache_position:1x4,c2w_attention_bias:1x1x4x8"
  local max="codes:${max_b}x16x4,cache_position:${max_b}x4,c2w_attention_bias:${max_b}x1x4x76"

  for i in {0..7}; do
    min+=",past_kv_${i}_k:1x16x1x64,past_kv_${i}_v:1x16x1x64"
    opt+=",past_kv_${i}_k:1x16x4x64,past_kv_${i}_v:1x16x4x64"
    max+=",past_kv_${i}_k:${max_b}x16x72x64,past_kv_${i}_v:${max_b}x16x72x64"
  done

  local states=(
    "conv_state_0:1x512x2" "conv_state_1:1x1024x6" "conv_state_2:1x1024x6" "conv_state_3:1x1024x6"
    "conv_state_4:1x768x6" "conv_state_5:1x768x18" "conv_state_6:1x768x54" "conv_state_7:1x384x6"
    "conv_state_8:1x384x18" "conv_state_9:1x384x54" "conv_state_10:1x192x6" "conv_state_11:1x192x18"
    "conv_state_12:1x192x54" "conv_state_13:1x96x6" "conv_state_14:1x96x18" "conv_state_15:1x96x54"
    "conv_state_16:1x96x6" "transconv_overlap_0:1x768x8" "transconv_overlap_1:1x384x5"
    "transconv_overlap_2:1x192x4" "transconv_overlap_3:1x96x3"
  )
  for kv in "${states[@]}"; do
    local name="${kv%%:*}"
    local shape="${kv#*:}"
    if [[ "${static_state}" == "1" ]]; then
      local fixed="8x${shape#1x}"
      min+=",${name}:${fixed}"
      opt+=",${name}:${fixed}"
      max+=",${name}:${fixed}"
    else
      min+=",${name}:${shape}"
      opt+=",${name}:${shape}"
      max+=",${name}:${max_b}x${shape#1x}"
    fi
  done

  local iin="int64:chw,fp32:chw"
  for _ in {1..38}; do
    iin+=",bf16:chw"
  done

  echo "MIN=${min}"
  echo "OPT=${opt}"
  echo "MAX=${max}"
  echo "IIN=${iin}"
}

run_case() {
  local kind="$1"       # fail|pass
  local static_state="$2" # 0|1
  local onnx_path
  local engine_path
  local log_path

  if [[ "${kind}" == "fail" ]]; then
    onnx_path="${OUT_ROOT}/dynamic_state/tokenizer/code2wav_decoder_wav_only.onnx"
    engine_path="${OUT_ROOT}/dynamic_state/tokenizer/code2wav_decoder_wav_only.engine"
    log_path="${LOG_DIR}/fail.log"
  else
    onnx_path="${OUT_ROOT}/static_state_8/tokenizer/code2wav_decoder_wav_only.onnx"
    engine_path="${OUT_ROOT}/static_state_8/tokenizer/code2wav_decoder_wav_only.engine"
    log_path="${LOG_DIR}/pass.log"
  fi

  if [[ ! -f "${onnx_path}" ]]; then
    echo "[error] Missing ONNX: ${onnx_path}"
    echo "[hint ] Run: python ${SCRIPT_DIR}/build_repro_onnx.py --out-root ${OUT_ROOT}"
    exit 3
  fi

  eval "$(generate_shapes "${static_state}")"
  local cmd=(
    docker run --rm --gpus all
    -v "$(dirname "$(dirname "${onnx_path}")"):/mnt/model"
    "${IMAGE}"
    "${TRTEXEC}"
    --onnx="/mnt/model/tokenizer/code2wav_decoder_wav_only.onnx"
    --saveEngine="/mnt/model/tokenizer/$(basename "${engine_path}")"
    --bf16
    --inputIOFormats="${IIN}"
    --outputIOFormats="bf16:chw"
    --minShapes="${MIN}"
    --optShapes="${OPT}"
    --maxShapes="${MAX}"
    --memPoolSize="workspace:${WORKSPACE_MB}"
  )

  echo "[run ] ${kind} -> ${log_path}"
  if [[ "${kind}" == "fail" ]]; then
    set +e
    "${cmd[@]}" 2>&1 | tee "${log_path}"
    local rc=$?
    set -e
    if grep -q "MyelinCheckException: tensor.cpp:852" "${log_path}"; then
      echo "[ok  ] fail case hit Myelin tensor.cpp:852 as expected"
    else
      echo "[warn] fail case did not contain expected Myelin tensor.cpp:852"
      return 1
    fi
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee "${log_path}"
  if grep -q "&&&& PASSED TensorRT.trtexec" "${log_path}"; then
    echo "[ok  ] pass case built successfully"
  else
    echo "[warn] pass case did not show PASSED marker"
    return 1
  fi
}

case "${MODE}" in
  fail) run_case "fail" 0 ;;
  pass) run_case "pass" 1 ;;
  both)
    run_case "fail" 0
    run_case "pass" 1
    ;;
esac

echo
echo "[done] logs:"
echo "  ${LOG_DIR}/fail.log"
echo "  ${LOG_DIR}/pass.log"
