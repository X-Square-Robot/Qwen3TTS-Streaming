#!/bin/bash

set -euo pipefail

config_path="${ENGINE_CONFIG:-/app/engine.yaml}"

model_repo="${ENGINE_MODEL_REPOSITORY:-/models}"
model_name="${ENGINE_MODEL_NAME:-tts_orchestrator}"
model_version="${ENGINE_MODEL_VERSION:-${MODEL_VERSION:-1}}"
model_package_dir="${ENGINE_MODEL_PACKAGE_DIR:-${model_repo}/${model_name}/${model_version}}"

# Engine code source. Default: the image's /app copy, so the image tag
# identifies the code that runs. ENGINE_CODE_FROM_PACKAGE=1 switches to the
# engine/ copy bundled in the model package — an emergency override to hotfix
# runtime code without an image rebuild. The bundled copy itself always ships
# in the package: Triton BLS loads it from the model repository regardless.
engine_code_root="/app"
if [[ "${ENGINE_CODE_FROM_PACKAGE:-0}" = "1" ]]; then
    if [[ ! -d "${model_package_dir}/engine" ]]; then
        echo "ENGINE_CODE_FROM_PACKAGE=1 but no engine/ code in model package: ${model_package_dir}" >&2
        echo "Re-assemble with: bash scripts/bash/compose.sh prepare --gateway engine --engine-mode trt, or unset ENGINE_CODE_FROM_PACKAGE" >&2
        exit 1
    fi
    engine_code_root="${model_package_dir}"
fi

# Run from the selected code root: python puts the CWD first on sys.path,
# so the cd — not PYTHONPATH order — decides which engine/ copy resolves
# the package paths.
resolved_paths="$(
    cd "$engine_code_root" && python3 - "$model_package_dir" <<'PY'
import sys
from engine.config import resolve_model_package_paths

p = resolve_model_package_paths(sys.argv[1])
print("\t".join([
    p.package_dir,
    p.engine_dir,
    p.weights_dir,
    p.tokenizer_dir,
    p.manifest_path,
    p.runtime_artifact_path,
    p.engine_mode,
]))
PY
)"
IFS=$'\t' read -r model_package_dir engine_dir weights_dir tokenizer_dir manifest_path runtime_artifact engine_mode <<< "$resolved_paths"

if [[ ! -d "$model_package_dir" ]]; then
    echo "Model package directory not found: $model_package_dir" >&2
    echo "Expected Triton-compatible model package: /models/tts_orchestrator/${model_version}/{runtime,weights,tokenizer}" >&2
    exit 1
fi
if [[ ! -d "$tokenizer_dir" ]]; then
    echo "Tokenizer directory not found: $tokenizer_dir" >&2
    exit 1
fi
if [[ ! -d "$weights_dir" ]]; then
    echo "Weights directory not found: $weights_dir" >&2
    exit 1
fi
if [[ ! -d "$engine_dir" ]]; then
    echo "Runtime directory not found: $engine_dir" >&2
    exit 1
fi
if [[ ! -f "$manifest_path" ]]; then
    echo "Model package manifest not found: $manifest_path" >&2
    exit 1
fi
if [[ "$engine_mode" != "trt" ]]; then
    echo "Engine Docker requires a TensorRT model package, got engine_mode=${engine_mode:-unknown}: $manifest_path" >&2
    echo "Re-assemble with: bash scripts/bash/compose.sh prepare --gateway engine --engine-mode trt --model-version ${model_version}" >&2
    exit 1
fi
if [[ ! -f "$runtime_artifact" ]]; then
    if [[ -f "${engine_dir}/model.onnx" ]]; then
        echo "Engine Docker requires a TensorRT model package, but found ONNX runtime only: ${engine_dir}/model.onnx" >&2
        echo "Re-assemble with: bash scripts/bash/compose.sh prepare --gateway engine --engine-mode trt --model-version ${model_version}" >&2
    else
        echo "TensorRT runtime artifact not found: ${runtime_artifact}" >&2
        echo "Run Phase B and assemble the shared model_repository in trt mode." >&2
    fi
    exit 1
fi
if [[ ! -f "$config_path" ]]; then
    echo "Config file not found: $config_path" >&2
    exit 1
fi

export ENGINE_SCHEDULER_MAX_BATCH_SIZE="${ENGINE_MAX_BATCH_SIZE:-48}"
# Container platforms commonly inject PORT as the only public service port.
# The gateway serves Demo, SDK, health, capabilities and both WebSocket
# protocols on that one port. HEALTH_PORT=0 disables the optional early-binding
# health listener; /health remains available on PORT after the gateway binds.
export ENGINE_SERVER_WEBSOCKET_PORT="${ENGINE_SERVER_WEBSOCKET_PORT:-${PORT:-${ENGINE_WEBSOCKET_PORT:-50052}}}"
export ENGINE_SERVER_HEALTH_PORT="${ENGINE_SERVER_HEALTH_PORT:-${HEALTH_PORT:-${ENGINE_HEALTH_PORT:-8080}}}"
tls_dir="${TLS_DIR:-/app/tls}"
tls_cert_file="${TLS_CERT_FILE:-}"
tls_key_file="${TLS_KEY_FILE:-}"
tls_auto_enable="${TLS_AUTO_ENABLE:-false}"
case "${tls_auto_enable,,}" in
    1|true|yes|on) tls_auto_enable="true" ;;
    0|false|no|off|"") tls_auto_enable="false" ;;
    *)
        echo "TLS_AUTO_ENABLE must be a boolean (true/false)." >&2
        exit 2
        ;;
esac
if [[ "$tls_auto_enable" = "true" && -z "$tls_cert_file" && -z "$tls_key_file" ]]; then
    if [[ ! -f "${tls_dir}/cert.local.pem" || ! -f "${tls_dir}/key.local.pem" ]]; then
        echo "TLS_AUTO_ENABLE=true but cert.local.pem/key.local.pem were not found in ${tls_dir}." >&2
        exit 2
    fi
    tls_cert_file="${tls_dir}/cert.local.pem"
    tls_key_file="${tls_dir}/key.local.pem"
    echo "TLS_AUTO_ENABLE=true: using development TLS certificate from ${tls_dir}."
fi
if [[ -n "$tls_cert_file" || -n "$tls_key_file" ]]; then
    if [[ -z "$tls_cert_file" || -z "$tls_key_file" ]]; then
        echo "TLS_CERT_FILE and TLS_KEY_FILE must be set together." >&2
        exit 2
    fi
    if [[ ! -r "$tls_cert_file" || ! -r "$tls_key_file" ]]; then
        echo "TLS certificate or private key is not readable." >&2
        exit 2
    fi
fi
export ENGINE_SERVER_TLS_CERT_FILE="$tls_cert_file"
export ENGINE_SERVER_TLS_KEY_FILE="$tls_key_file"
if [[ -n "${ENGINE_MAX_SEQ_LEN:-}" ]]; then
    export ENGINE_SCHEDULER_MAX_SEQ_LEN="${ENGINE_MAX_SEQ_LEN}"
fi
# Conditional (unlike the port exports above) so engine.yaml's
# server.health_probe_mode still applies when the env is unset.
if [[ -n "${ENGINE_HEALTH_PROBE_MODE:-}" ]]; then
    export ENGINE_SERVER_HEALTH_PROBE_MODE="${ENGINE_HEALTH_PROBE_MODE}"
fi

if [[ "$engine_code_root" = "/app" ]]; then
    export PYTHONPATH="${PYTHONPATH:-/app}"
else
    export PYTHONPATH="${engine_code_root}:${PYTHONPATH:-/app}"
fi

cmd=(
    python3 -m engine.server
    --config "$config_path"
    --model-package-dir "$model_package_dir"
    --device "${ENGINE_DEVICE:-0}"
    --max-batch "${ENGINE_MAX_BATCH_SIZE:-48}"
    --max-sessions "${ENGINE_MAX_SESSIONS:-128}"
    --port "${ENGINE_GRPC_PORT:-50051}"
)

# Lightweight fingerprint precheck.  Runs *before* Python startup so a
# wrong-GPU container fails in milliseconds with a clear error rather than
# 20-30s into engine init with a cryptic TRT log.  Only checks gpu_sm;
# full check (TRT version + driver) happens inside engine/server.py
# via engine.runtime.fingerprint.enforce_engine_fingerprint.
#
# Bypass: ENGINE_SKIP_FINGERPRINT_PRECHECK=1 (the Python guard inside
# engine/server.py still runs and is the authoritative check).
artifact_manifest=""
for candidate in \
    "$model_package_dir/artifact_manifest.json" \
    "${model_package_dir}/../artifact_manifest.json" \
    "${model_repo}/artifact_manifest.json"; do
    if [[ -f "$candidate" ]]; then
        artifact_manifest="$candidate"
        break
    fi
done

if [[ "${ENGINE_SKIP_FINGERPRINT_PRECHECK:-0}" != "1" ]] && [[ -n "$artifact_manifest" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[WARN] entrypoint: nvidia-smi unavailable; skipping SM precheck (Python guard will still run)" >&2
    else
        device_index="${ENGINE_DEVICE:-0}"
        actual_cc=$(nvidia-smi --id="$device_index" \
            --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | tr -d ' ' || true)
        if [[ -z "$actual_cc" ]]; then
            echo "[WARN] entrypoint: cannot read compute_cap for device $device_index; deferring to Python guard" >&2
        else
            actual_sm="sm_${actual_cc//./}"
            expected_sm=$(python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('gpu_sm','') or '')
except Exception:
    print('')
" "$artifact_manifest")
            if [[ -n "$expected_sm" ]] && [[ "$expected_sm" != "$actual_sm" ]]; then
                if [[ "${QWEN3_ALLOW_FINGERPRINT_MISMATCH:-0}" = "1" ]]; then
                    echo "[WARN] entrypoint: SM mismatch (expected=$expected_sm actual=$actual_sm) — bypassed via QWEN3_ALLOW_FINGERPRINT_MISMATCH=1" >&2
                else
                    echo "[ERROR] entrypoint: GPU SM mismatch — refusing to start" >&2
                    echo "        expected: $expected_sm (from $artifact_manifest)" >&2
                    echo "        actual:   $actual_sm (device $device_index)" >&2
                    echo "        The engine was compiled for a different GPU architecture." >&2
                    echo "        Either:" >&2
                    echo "          1. Run this image on a host with $expected_sm GPUs" >&2
                    echo "          2. Re-build the engine on the target hardware via autorun.sh build" >&2
                    echo "          3. Set QWEN3_ALLOW_FINGERPRINT_MISMATCH=1 (debugging only)" >&2
                    exit 1
                fi
            else
                echo "[INFO] entrypoint: SM precheck OK (sm=$actual_sm)" >&2
            fi
        fi
    fi
elif [[ -z "$artifact_manifest" ]]; then
    if [[ "${QWEN3_ALLOW_FINGERPRINT_MISMATCH:-0}" != "1" ]]; then
        echo "[WARN] entrypoint: no artifact_manifest.json found near $model_package_dir" >&2
        echo "        Python guard will fail-stop unless QWEN3_ALLOW_FINGERPRINT_MISMATCH=1" >&2
    fi
fi

echo "Starting engine from shared model package" >&2
if [[ "$engine_code_root" = "/app" ]]; then
    echo "  engine_code=image:/app" >&2
else
    echo "  engine_code=package:${engine_code_root} (ENGINE_CODE_FROM_PACKAGE=1)" >&2
fi
echo "  config=${config_path}" >&2
echo "  model_repo=${model_repo}" >&2
echo "  model_package=${model_package_dir}" >&2
echo "  tokenizer=${tokenizer_dir}" >&2
echo "  weights=${weights_dir}" >&2
echo "  engine_dir=${engine_dir}" >&2
echo "  manifest=${manifest_path}" >&2
echo "  artifact_manifest=${artifact_manifest:-<not found>}" >&2
echo "  runtime_artifact=${runtime_artifact}" >&2
echo "  engine_mode=${engine_mode}" >&2
echo "  pythonpath=${PYTHONPATH}" >&2

# python3 -m prepends the CWD to sys.path ahead of PYTHONPATH, so this cd —
# not the exports above — is what actually selects the engine/ copy that runs.
cd "$engine_code_root"
exec "${cmd[@]}"
