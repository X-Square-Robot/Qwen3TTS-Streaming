#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-/tmp/myelin_repro_minimal}"
MODE="${2:-both}" # fail | pass | both

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${MODE}" != "fail" && "${MODE}" != "pass" && "${MODE}" != "both" ]]; then
  echo "[error] MODE must be fail|pass|both, got: ${MODE}"
  exit 2
fi

# Hard-clean known experiment toggles from shell environment.
for k in \
  C2W_STATIC_STATE_BATCH \
  C2W_DEBUG_BYPASS_QUANTIZER \
  C2W_DEBUG_BYPASS_TRANSFORMER \
  TRT_MYELIN_DISABLE \
  ENGINE_DTYPE \
  MAX_BATCH_SIZE \
  BUILD_VERIFICATION_ENGINES
do
  unset "${k}" || true
done

# Also drop any C2W_DEBUG_* custom key.
while IFS='=' read -r k _; do
  if [[ "${k}" == C2W_DEBUG_* ]]; then
    unset "${k}" || true
  fi
done < <(env)

echo "[env ] cleaned debug toggles"
echo "[path] OUT_ROOT=${OUT_ROOT}"
echo "[mode] ${MODE}"

conda run --no-capture-output -n qwen3-tts \
  python "${SCRIPT_DIR}/build_repro_onnx.py" --out-root "${OUT_ROOT}"

IMAGE="nvcr.io/nvidia/tritonserver:26.02-py3" \
WORKSPACE_MB="4096" \
bash "${SCRIPT_DIR}/run_repro.sh" "${MODE}" "${OUT_ROOT}"

echo
echo "[done] minimal repro pipeline finished."
echo "       fail log: ${OUT_ROOT}/logs/fail.log"
echo "       pass log: ${OUT_ROOT}/logs/pass.log"
