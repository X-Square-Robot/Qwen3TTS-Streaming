#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-/tmp/myelin_repro_frozen}"
MODE="${2:-both}" # fail | pass | both

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST_DIR="${OUT_ROOT}/manifest"
mkdir -p "${MANIFEST_DIR}"

if [[ "${MODE}" != "fail" && "${MODE}" != "pass" && "${MODE}" != "both" ]]; then
  echo "[error] MODE must be fail|pass|both, got: ${MODE}"
  exit 2
fi

CONDA_BIN="$(command -v conda || true)"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "[error] conda not found in PATH"
  exit 3
fi

DOCKER_BIN="$(command -v docker || true)"
if [[ -z "${DOCKER_BIN}" ]]; then
  echo "[error] docker not found in PATH"
  exit 3
fi

MINIMAL_CMD="cd ${REPO_ROOT} && bash ${SCRIPT_DIR}/minimal_pipeline.sh ${OUT_ROOT} ${MODE}"

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=${REPO_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "mode=${MODE}"
  echo "conda_bin=${CONDA_BIN}"
  echo "docker_bin=${DOCKER_BIN}"
  echo "minimal_cmd=${MINIMAL_CMD}"
} > "${MANIFEST_DIR}/cmd.txt"

env -i \
  HOME="${HOME}" \
  USER="${USER:-}" \
  LOGNAME="${LOGNAME:-${USER:-}}" \
  SHELL="${SHELL:-/bin/bash}" \
  PATH="${PATH}" \
  LANG="${LANG:-C.UTF-8}" \
  LC_ALL="${LC_ALL:-C.UTF-8}" \
  bash -lc "env | sort > '${MANIFEST_DIR}/env.txt'"

env -i \
  HOME="${HOME}" \
  USER="${USER:-}" \
  LOGNAME="${LOGNAME:-${USER:-}}" \
  SHELL="${SHELL:-/bin/bash}" \
  PATH="${PATH}" \
  LANG="${LANG:-C.UTF-8}" \
  LC_ALL="${LC_ALL:-C.UTF-8}" \
  bash -lc "${MINIMAL_CMD}"

{
  echo "# sha256 generated at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for f in \
    "${OUT_ROOT}/dynamic_state/tokenizer/code2wav_decoder_wav_only.onnx" \
    "${OUT_ROOT}/static_state_8/tokenizer/code2wav_decoder_wav_only.onnx" \
    "${OUT_ROOT}/logs/fail.log" \
    "${OUT_ROOT}/logs/pass.log"
  do
    if [[ -f "${f}" ]]; then
      sha256sum "${f}"
    else
      echo "MISSING  ${f}"
    fi
  done
} > "${MANIFEST_DIR}/sha256.txt"

echo
echo "[done] frozen pipeline finished."
echo "       manifest: ${MANIFEST_DIR}"
