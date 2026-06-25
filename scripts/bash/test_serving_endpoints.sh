#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="${QWEN3_TTS_ENV_NAME:-qwen3-tts}"

eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"

exec python tests/tools/serving_endpoints.py "$@"
