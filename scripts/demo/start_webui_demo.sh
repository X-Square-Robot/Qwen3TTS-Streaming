#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

VARIANT="${MODEL_VARIANT:-custom-1.7b}"
HOST="${QWEN_DEMO_HOST:-127.0.0.1}"
API_PORT="${QWEN_DEMO_PORT:-7860}"
WEBUI_PORT="${WEBUI_PORT:-5173}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_MAX_BATCH_SLOTS="${TRITON_MAX_BATCH_SLOTS:-128}"
TRITON_MAX_SESSIONS="${TRITON_MAX_SESSIONS:-128}"
ENGINE_WS_PORT="${ENGINE_WEBSOCKET_PORT:-50052}"
START_TRITON=true
START_ENGINE=false
LIVE_CONCURRENCY=true

usage() {
  cat <<EOF
Usage: scripts/demo/start_webui_demo.sh [options]

Starts the local demo stack for first-time evaluation:
  1. Triton server, unless --no-triton is set or port ${TRITON_GRPC_PORT} is already listening
  2. demo_api on ${HOST}:${API_PORT}
  3. Vite WebUI on 0.0.0.0:${WEBUI_PORT}

Options:
  --variant <name>          Model variant for Triton compose startup (default: ${VARIANT})
  --triton-slots <N>        Triton active decode slots / MAX_BATCH_SLOTS (default: ${TRITON_MAX_BATCH_SLOTS})
  --no-triton               Do not start Triton automatically
  --with-engine             Also start standalone engine container on WebSocket port ${ENGINE_WS_PORT}
  --live-concurrency        Enable real Triton concurrency jobs in demo_api (default)
  --simulated-concurrency   Use simulated concurrency metrics with no lane audio
  -h, --help                Show this help

Notes:
  All demo panels (LLM PK, Speak TRT, Concurrency) talk to the same Triton.
  --with-engine is optional and only needed when you also want to compare
  against a side-by-side standalone engine container; the WebUI doesn't
  need it.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --triton-slots)
      TRITON_MAX_BATCH_SLOTS="$2"
      shift 2
      ;;
    --no-triton)
      START_TRITON=false
      shift
      ;;
    --with-engine)
      START_ENGINE=true
      shift
      ;;
    --live-concurrency)
      LIVE_CONCURRENCY=true
      shift
      ;;
    --simulated-concurrency)
      LIVE_CONCURRENCY=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

port_listening() {
  local port="$1"
  ss -ltn "( sport = :${port} )" | tail -n +2 | grep -q .
}

choose_python() {
  if [[ -n "${QWEN_DEMO_PYTHON:-}" ]]; then
    printf '%s\n' "${QWEN_DEMO_PYTHON}"
    return
  fi
  if [[ -x "${HOME}/miniforge3/envs/qwen3-tts/bin/python" ]]; then
    printf '%s\n' "${HOME}/miniforge3/envs/qwen3-tts/bin/python"
    return
  fi
  if [[ -x "${HOME}/miniconda3/envs/qwen3-tts/bin/python" ]]; then
    printf '%s\n' "${HOME}/miniconda3/envs/qwen3-tts/bin/python"
    return
  fi
  printf '%s\n' "python3"
}

wait_http() {
  local url="$1"
  local name="$2"
  for _ in $(seq 1 80); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "${name} ready: ${url}"
      return 0
    fi
    sleep 0.25
  done
  echo "${name} did not become ready: ${url}" >&2
  return 1
}

require_demo_python_dependencies() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys
missing = [name for name in ("aiohttp", "numpy", "tritonclient") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing demo_api Python dependencies: " + ", ".join(missing), file=sys.stderr)
    print("Install with: python -m pip install -r demo_api/requirements.txt", file=sys.stderr)
    raise SystemExit(1)
PY
}

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "${API_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${WEBUI_PID:-}" ]]; then
    kill "${WEBUI_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
PYTHON_BIN="$(choose_python)"
require_demo_python_dependencies

export TRITON_MAX_BATCH_SLOTS
export TRITON_MAX_SESSIONS
DETECTED_TRITON_SLOTS=""

detect_running_triton_slots() {
  if command -v docker >/dev/null 2>&1; then
    docker inspect qwen3tts-streaming \
      --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
      | awk -F= '$1 == "MAX_BATCH_SLOTS" {print $2; exit}'
  fi
}

if ${START_TRITON}; then
  if port_listening "${TRITON_GRPC_PORT}"; then
    echo "Triton gRPC already listening on :${TRITON_GRPC_PORT}"
    DETECTED_TRITON_SLOTS="$(detect_running_triton_slots || true)"
    if [[ -n "${DETECTED_TRITON_SLOTS}" && "${DETECTED_TRITON_SLOTS}" != "${TRITON_MAX_BATCH_SLOTS}" ]]; then
      echo "Running Triton MAX_BATCH_SLOTS=${DETECTED_TRITON_SLOTS}; requested ${TRITON_MAX_BATCH_SLOTS}. Restart Triton to change active slots."
    fi
  else
    echo "Starting Triton for variant ${VARIANT} with MAX_BATCH_SLOTS=${TRITON_MAX_BATCH_SLOTS}..."
    bash scripts/bash/compose.sh up --gateway triton --variant "${VARIANT}" --prepare
  fi
fi

if ${START_ENGINE}; then
  if port_listening "${ENGINE_WS_PORT}"; then
    echo "Engine WebSocket already listening on :${ENGINE_WS_PORT}"
  else
    echo "Starting standalone engine for variant ${VARIANT}..."
    bash scripts/bash/compose.sh up --gateway engine --variant "${VARIANT}" --max-sessions "${ENGINE_MAX_SESSIONS:-16}"
  fi
fi

export QWEN_DEMO_TRITON_GRPC="${QWEN_DEMO_TRITON_GRPC:-localhost:${TRITON_GRPC_PORT}}"
export QWEN_DEMO_TRITON_MAX_BATCH_SLOTS="${DETECTED_TRITON_SLOTS:-${TRITON_MAX_BATCH_SLOTS}}"
export QWEN_DEMO_TRITON_MAX_SESSIONS="${TRITON_MAX_SESSIONS}"
export QWEN_DEMO_ENGINE_WS="${QWEN_DEMO_ENGINE_WS:-ws://localhost:${ENGINE_WS_PORT}/v1/ws}"
export QWEN_DEMO_ENGINE_CAPABILITIES="${QWEN_DEMO_ENGINE_CAPABILITIES:-http://localhost:${ENGINE_WS_PORT}/v1/capabilities}"
export QWEN_DEMO_ENABLE_LIVE_CONCURRENCY=$(${LIVE_CONCURRENCY} && echo 1 || echo 0)
ENGINE_STATUS="not started by launcher"
if ${START_ENGINE}; then
  ENGINE_STATUS="started or reused by launcher"
fi

if port_listening "${API_PORT}"; then
  echo "demo_api port already listening on :${API_PORT}"
else
  echo "Starting demo_api on ${HOST}:${API_PORT}..."
  "${PYTHON_BIN}" -m demo_api --host "${HOST}" --port "${API_PORT}" &
  API_PID=$!
fi
wait_http "http://${HOST}:${API_PORT}/healthz" "demo_api"

if [[ ! -d webui/node_modules ]]; then
  echo "Installing WebUI dependencies..."
  npm --prefix webui install
fi

if port_listening "${WEBUI_PORT}"; then
  echo "WebUI port already listening on :${WEBUI_PORT}"
else
  echo "Starting WebUI on http://localhost:${WEBUI_PORT} ..."
  npm --prefix webui run dev -- --host 0.0.0.0 --port "${WEBUI_PORT}" &
  WEBUI_PID=$!
fi
wait_http "http://127.0.0.1:${WEBUI_PORT}" "webui"

cat <<EOF

Demo stack is running.
  WebUI:    http://localhost:${WEBUI_PORT}
  Demo API: http://${HOST}:${API_PORT}
  Triton:   ${QWEN_DEMO_TRITON_GRPC}
  Engine:   ${QWEN_DEMO_ENGINE_WS} (${ENGINE_STATUS})

Press Ctrl-C to stop demo_api and WebUI. Triton is managed by docker compose and is left running.
EOF

wait
