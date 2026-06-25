#!/bin/bash
# ===========================================================================
#  engine.sh — Standalone TTS Engine lifecycle helpers
#
#  Functions: resolve_engine_package_paths, engine_start, engine_stop,
#             engine_status, engine_health_check
#  Depends:   lib/logging.sh, lib/utils.sh
#
#  Manages the standalone TTS engine server (python -m engine.server)
#  as a background process with PID file tracking.
# ===========================================================================

[[ -n "${_LIB_ENGINE_LOADED:-}" ]] && return 0
_LIB_ENGINE_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"

ENGINE_GRPC_PORT="${ENGINE_GRPC_PORT:-50051}"
ENGINE_WEBSOCKET_PORT="${ENGINE_WEBSOCKET_PORT:-50052}"
ENGINE_HEALTH_PORT="${ENGINE_HEALTH_PORT:-8080}"
ENGINE_IMAGE="${ENGINE_IMAGE:-qwen3-engine:26.02}"
ENGINE_CONTAINER_NAME="${ENGINE_CONTAINER_NAME:-qwen3-engine}"
# Standalone engine: interpreter selection (Phase A conda env is not auto-activated for deploy).
#   ENGINE_PYTHON        If set, must be an executable python with torch + deps.
#   QWEN3_TTS_ENV_NAME   Conda env name to look up (default: qwen3-tts).

# ---------------------------------------------------------------------------
#  resolve_engine_python_bin <repo_root>
#  Prints a python executable path for engine.server. Prefers ENGINE_PYTHON,
#  then <repo>/.venv/bin/python, then conda env QWEN3_TTS_ENV_NAME, else python3.
# ---------------------------------------------------------------------------
resolve_engine_python_bin() {
    local repo_root="$1"
    local env_name="${QWEN3_TTS_ENV_NAME:-qwen3-tts}"

    if [ -n "${ENGINE_PYTHON:-}" ] && [ -x "$ENGINE_PYTHON" ]; then
        printf '%s\n' "$ENGINE_PYTHON"
        return 0
    fi
    if [ -x "$repo_root/.venv/bin/python" ]; then
        printf '%s\n' "$repo_root/.venv/bin/python"
        return 0
    fi
    local list_cmd=""
    command -v conda &>/dev/null && list_cmd=conda
    [ -z "$list_cmd" ] && command -v mamba &>/dev/null && list_cmd=mamba
    if [ -n "$list_cmd" ]; then
        local prefix
        prefix=$($list_cmd env list 2>/dev/null | awk -v n="$env_name" '$1 == n { print $NF; exit }')
        if [ -n "$prefix" ] && [ -x "$prefix/bin/python" ]; then
            printf '%s\n' "$prefix/bin/python"
            return 0
        fi
    fi
    if command -v python3 &>/dev/null; then
        command -v python3
        return 0
    fi
    if command -v python &>/dev/null; then
        command -v python
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
#  resolve_engine_package_paths <repo_root>
#  Resolves the shared Triton-compatible package consumed by host standalone,
#  engine Docker, and Triton. Sets _ENGINE_MODEL_PACKAGE_DIR,
#  _ENGINE_TOKENIZER_DIR, _ENGINE_WEIGHTS_DIR, _ENGINE_DIR, and
#  _ENGINE_RUNTIME_ARTIFACT.
# ---------------------------------------------------------------------------
resolve_engine_package_paths() {
    local repo_root="$1"
    local model_repo="${MODEL_REPO_DIR:-$repo_root/workspace/model_repository}"
    local model_version
    model_version=$(resolve_model_version) || return 1
    local model_package="${ENGINE_MODEL_PACKAGE_DIR:-}"
    if [ -z "$model_package" ] || { [[ "$model_package" == /models/* ]] && [ ! -d "$model_package" ]; }; then
        model_package="$model_repo/tts_orchestrator/$model_version"
    fi

    local resolved_paths
    resolved_paths=$(PYTHONPATH="$repo_root" python3 - "$model_package" <<'PY' 2>/dev/null || true
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
)
    if [ -z "$resolved_paths" ]; then
        log_error "Failed to resolve shared model package: $model_package"
        log_error "Run: bash scripts/bash/compose.sh prepare --gateway engine --engine-mode trt --model-version $model_version"
        return 1
    fi

    local manifest_path engine_mode
    IFS=$'\t' read -r \
        _ENGINE_MODEL_PACKAGE_DIR \
        _ENGINE_DIR \
        _ENGINE_WEIGHTS_DIR \
        _ENGINE_TOKENIZER_DIR \
        manifest_path \
        _ENGINE_RUNTIME_ARTIFACT \
        engine_mode <<< "$resolved_paths"

    if [ ! -d "$_ENGINE_MODEL_PACKAGE_DIR" ]; then
        log_error "Shared model package not found: $_ENGINE_MODEL_PACKAGE_DIR"
        log_error "Expected: model_repository/tts_orchestrator/$model_version/{runtime,weights,tokenizer}"
        log_error "Run: bash scripts/bash/compose.sh prepare --gateway engine --engine-mode trt --model-version $model_version"
        return 1
    fi
    if [ ! -d "$_ENGINE_DIR" ]; then
        log_error "Runtime directory not found: $_ENGINE_DIR"
        return 1
    fi
    if [ ! -d "$_ENGINE_WEIGHTS_DIR" ]; then
        log_error "Weights directory not found: $_ENGINE_WEIGHTS_DIR"
        return 1
    fi
    if [ ! -d "$_ENGINE_TOKENIZER_DIR" ]; then
        log_error "Tokenizer directory not found: $_ENGINE_TOKENIZER_DIR"
        return 1
    fi
    if [ ! -f "$manifest_path" ]; then
        log_error "Model package manifest not found: $manifest_path"
        return 1
    fi
    if [ "$engine_mode" != "trt" ]; then
        log_error "Standalone engine requires a TensorRT model package, got engine_mode=${engine_mode:-unknown}: $manifest_path"
        log_error "Use --engine-mode trt for engine/standalone, or choose Triton for ONNX packages."
        return 1
    fi
    if [ ! -f "$_ENGINE_RUNTIME_ARTIFACT" ]; then
        log_error "TensorRT runtime artifact not found: $_ENGINE_RUNTIME_ARTIFACT"
        log_error "Run Phase B and assemble the shared model_repository in trt mode."
        return 1
    fi

    return 0
}

# ---------------------------------------------------------------------------
#  engine_pid_file <repo_root>
#  Returns the PID file path for the standalone engine.
# ---------------------------------------------------------------------------
engine_pid_file() {
    echo "$1/workspace/.engine.pid"
}

# ---------------------------------------------------------------------------
#  engine_log_file <repo_root>
#  Returns the log file path for the standalone engine.
# ---------------------------------------------------------------------------
engine_log_file() {
    echo "$1/workspace/engine.log"
}

# ---------------------------------------------------------------------------
#  engine_start <repo_root> <variant> [options...]
#
#  Starts the standalone TTS engine server as a background process.
#  Options:
#    --port <N>            gRPC port (default: $ENGINE_GRPC_PORT)
#    --ws-port <N>         WebSocket port (default: $ENGINE_WEBSOCKET_PORT)
#    --device <N>          GPU device (default: 0)
#    --max-batch <N>       Max batch size (default: 48)
#    --max-seq-len <N>     Max scheduler sequence length (optional)
#    --max-sessions <N>    Max concurrent sessions (default: 128)
#    --foreground          Run in foreground (don't daemonize)
# ---------------------------------------------------------------------------
engine_start() {
    local repo_root="$1"
    local variant="$2"
    shift 2

    local port="$ENGINE_GRPC_PORT"
    local ws_port="$ENGINE_WEBSOCKET_PORT"
    local device=0
    local max_batch=48
    local max_sessions=128
    local max_seq_len=""
    local foreground=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port)          port="$2"; shift 2 ;;
            --ws-port)       ws_port="$2"; shift 2 ;;
            --device)        device="$2"; shift 2 ;;
            --max-batch)     max_batch="$2"; shift 2 ;;
            --max-seq-len)   max_seq_len="$2"; shift 2 ;;
            --max-sessions)  max_sessions="$2"; shift 2 ;;
            --foreground)    foreground=true; shift ;;
            *)               shift ;;
        esac
    done

    local pid_file
    pid_file=$(engine_pid_file "$repo_root")
    local log_file
    log_file=$(engine_log_file "$repo_root")

    # Check if already running
    if [ -f "$pid_file" ]; then
        local old_pid
        old_pid=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            log_warn "Engine already running (PID $old_pid)"
            log_info "Stop first: deploy.sh stop"
            return 1
        fi
        rm -f "$pid_file"
    fi

    resolve_engine_package_paths "$repo_root" || return 1

    local pybin
    pybin=$(resolve_engine_python_bin "$repo_root") || {
        log_error "No python interpreter found for standalone engine"
        return 1
    }
    if ! "$pybin" -c "import torch" 2>/dev/null; then
        log_error "Selected Python cannot import torch: $pybin"
        log_error "Activate Phase A environment: conda activate ${QWEN3_TTS_ENV_NAME:-qwen3-tts}"
        log_error "Or set ENGINE_PYTHON to a Python that has PyTorch (and project deps) installed."
        return 1
    fi

    log_step "Starting Standalone TTS Engine"
    log_info "  Python:       $pybin"
    log_info "  Variant:      $variant"
    log_info "  Model package:$_ENGINE_MODEL_PACKAGE_DIR"
    log_info "  Tokenizer:    $_ENGINE_TOKENIZER_DIR"
    log_info "  Weights:      $_ENGINE_WEIGHTS_DIR"
    log_info "  Runtime:      $_ENGINE_DIR"
    log_info "  TRT Engine:   $_ENGINE_RUNTIME_ARTIFACT"
    log_info "  GPU Device:   $device"
    log_info "  Max Batch:    $max_batch"
    log_info "  Max Seq Len:  ${max_seq_len:-auto}"
    log_info "  Max Sessions: $max_sessions"
    log_info "  gRPC Port:    $port"
    log_info "  WS Port:      $ws_port"

    local cmd=(
        "$pybin" -m engine.server
        --model-package-dir "$_ENGINE_MODEL_PACKAGE_DIR"
        --device "$device"
        --max-batch "$max_batch"
        --max-sessions "$max_sessions"
        --port "$port"
        --ws-port "$ws_port"
    )
    if [ -n "$max_seq_len" ]; then
        cmd+=(--max-seq-len "$max_seq_len")
        export ENGINE_SCHEDULER_MAX_SEQ_LEN="$max_seq_len"
    fi

    if $foreground; then
        log_info "Running in foreground (Ctrl+C to stop)..."
        cd "$repo_root" && exec "${cmd[@]}"
    fi

    mkdir -p "$(dirname "$log_file")"
    cd "$repo_root" && nohup "${cmd[@]}" > "$log_file" 2>&1 &
    local engine_pid=$!

    echo "$engine_pid" > "$pid_file"
    log_info "Engine started (PID $engine_pid)"
    log_info "  Log file: $log_file"
    log_info "  PID file: $pid_file"

    return 0
}

# ---------------------------------------------------------------------------
#  engine_stop <repo_root>
#  Stops the standalone TTS engine gracefully (SIGTERM, then SIGKILL).
# ---------------------------------------------------------------------------
engine_stop() {
    local repo_root="$1"
    local pid_file
    pid_file=$(engine_pid_file "$repo_root")

    if [ ! -f "$pid_file" ]; then
        log_info "No engine PID file found (not running?)"
        return 0
    fi

    local pid
    pid=$(cat "$pid_file" 2>/dev/null)
    if [ -z "$pid" ]; then
        rm -f "$pid_file"
        log_info "Empty PID file, cleaned up"
        return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        log_info "Engine process (PID $pid) not running, cleaned up PID file"
        return 0
    fi

    log_info "Stopping engine (PID $pid)..."
    kill -TERM "$pid" 2>/dev/null

    local elapsed=0
    while [ "$elapsed" -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$pid_file"
            log_info "Engine stopped gracefully"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    log_warn "Engine did not stop within 10s, sending SIGKILL..."
    kill -9 "$pid" 2>/dev/null
    sleep 1
    rm -f "$pid_file"
    log_info "Engine killed"
    return 0
}

# ---------------------------------------------------------------------------
#  engine_status <repo_root>
#  Returns status: "none" | "running" | "healthy"
# ---------------------------------------------------------------------------
engine_status() {
    local repo_root="$1"
    local pid_file
    pid_file=$(engine_pid_file "$repo_root")

    if [ ! -f "$pid_file" ]; then
        echo "none"
        return 0
    fi

    local pid
    pid=$(cat "$pid_file" 2>/dev/null)
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        echo "none"
        return 0
    fi

    echo "running"
    return 0
}

# ---------------------------------------------------------------------------
#  engine_health_check <port> [timeout_sec]
#  Waits for the engine gRPC port to become reachable.
# ---------------------------------------------------------------------------
engine_health_check() {
    local port="${1:-$ENGINE_GRPC_PORT}"
    local timeout="${2:-60}"
    local elapsed=0
    local interval=2

    while [ "$elapsed" -lt "$timeout" ]; do
        if command -v grpc_health_probe &>/dev/null; then
            if grpc_health_probe -addr "localhost:${port}" -connect-timeout 1s &>/dev/null; then
                log_info "Engine healthy on port $port"
                return 0
            fi
        else
            # Fallback: check if port is open
            if (echo >/dev/tcp/localhost/"$port") 2>/dev/null; then
                log_info "Engine port $port is open (gRPC health probe not installed)"
                return 0
            fi
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    log_error "Engine health check timed out after ${timeout}s"
    return 1
}

# ---------------------------------------------------------------------------
#  engine_docker_image_has_app <image_tag>
#  Returns 0 if the image was built from Dockerfile.engine (bundled engine/ under /app).
#  Returns 1 if the tag points at a wrong image (e.g. base TensorRT retagged as qwen3-engine).
# ---------------------------------------------------------------------------
engine_docker_image_has_app() {
    local image="$1"
    docker run --rm --entrypoint "" "$image" \
        python3 -c "import engine.server" &>/dev/null
}

# ---------------------------------------------------------------------------
#  engine_docker_image_supports_model_package_engine <image_tag>
#  Returns 0 if the image contains the entrypoint/code needed to run engine.server
#  from the assembled model package payload.
# ---------------------------------------------------------------------------
engine_docker_image_supports_model_package_engine() {
    local image="$1"
    docker run --rm --entrypoint "" "$image" sh -lc '
        grep -q "Prefer the engine/ package copied into the assembled model package" /app/scripts/compose/engine-entrypoint.sh &&
        python3 - <<'"'"'PY'"'"'
from engine.config import EngineConfig

raise SystemExit(0 if hasattr(EngineConfig(), "references") else 1)
PY
    ' &>/dev/null
}

# ---------------------------------------------------------------------------
#  engine_docker_image_tensorrt_release <image_tag>
#  Echoes NVIDIA_TENSORRT_VERSION from the image (e.g. 25.03, 26.02).
# ---------------------------------------------------------------------------
engine_docker_image_tensorrt_release() {
    local image="$1"
    docker image inspect "$image" \
        --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
        | awk -F= '$1 == "NVIDIA_TENSORRT_VERSION" { print $2; exit }'
}

# ---------------------------------------------------------------------------
#  engine_docker_image_matches_release <image_tag> <ngc_tag>
#  Returns 0 when the existing image base matches the expected NGC release tag.
# ---------------------------------------------------------------------------
engine_docker_image_matches_release() {
    local image="$1"
    local expected="$2"
    local actual
    actual=$(engine_docker_image_tensorrt_release "$image" || true)
    [ -n "$actual" ] && [ "$actual" = "$expected" ]
}

# ---------------------------------------------------------------------------
#  engine_docker_image_torch_cuda_tag <image_tag>
#  Echoes the CUDA wheel tag from torch.version.cuda (e.g. cu128, cu130).
# ---------------------------------------------------------------------------
engine_docker_image_torch_cuda_tag() {
    local image="$1"
    docker run --rm --entrypoint python3 "$image" -c '
import torch
cuda = torch.version.cuda or ""
parts = cuda.split(".")
if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
    print(f"cu{parts[0]}{parts[1]}")
' 2>/dev/null
}

# ---------------------------------------------------------------------------
#  engine_docker_image_matches_torch_cuda <image_tag> <cuda_tag>
#  Returns 0 when the image's torch wheel matches the expected CUDA tag.
# ---------------------------------------------------------------------------
engine_docker_image_matches_torch_cuda() {
    local image="$1"
    local expected="$2"
    local actual
    actual=$(engine_docker_image_torch_cuda_tag "$image" || true)
    [ -n "$actual" ] && [ "$actual" = "$expected" ]
}

# ---------------------------------------------------------------------------
#  engine_build_image <repo_root> [image_tag]
#  Builds the Docker image for the standalone engine.
# ---------------------------------------------------------------------------
engine_build_image() {
    local repo_root="$1"
    local image_tag="${2:-$ENGINE_IMAGE}"
    local dockerfile="$repo_root/Dockerfile.engine"

    if [ ! -f "$dockerfile" ]; then
        log_error "Dockerfile not found: $dockerfile"
        return 1
    fi

    log_step "Building engine Docker image: $image_tag"
    local image_tag_release=""
    if [[ "$image_tag" =~ :([0-9]{2}\.[0-9]{2})$ ]]; then
        image_tag_release="${BASH_REMATCH[1]}"
    fi
    local base_image="${ENGINE_BASE_IMAGE:-}"
    if [ -z "$base_image" ]; then
        if [ -n "$image_tag_release" ]; then
            base_image="nvcr.io/nvidia/tensorrt:${image_tag_release}-py3"
        else
            base_image="nvcr.io/nvidia/tensorrt:26.02-py3"
        fi
    fi
    local pytorch_cuda_tag="${ENGINE_PYTORCH_CUDA_TAG:-${PYTORCH_CUDA_TAG:-}}"
    if [ -z "$pytorch_cuda_tag" ]; then
        if [ -n "$image_tag_release" ] && declare -F resolve_ngc_torch_index_tag >/dev/null; then
            pytorch_cuda_tag=$(resolve_ngc_torch_index_tag "$image_tag_release" 2>/dev/null || true)
        fi
        pytorch_cuda_tag="${pytorch_cuda_tag:-cu130}"
    fi
    local base_release=""
    if [[ "$base_image" =~ :([0-9]{2}\.[0-9]{2})-py3$ ]]; then
        base_release="${BASH_REMATCH[1]}"
    fi
    if [ -n "$image_tag_release" ] && [ -n "$base_release" ] && [ "$image_tag_release" != "$base_release" ]; then
        log_error "Engine image tag/base image mismatch: image=$image_tag implies $image_tag_release but BASE_IMAGE=$base_image"
        return 1
    fi
    # BuildKit: enables RUN --mount cache for pip (faster rebuilds; see Dockerfile.engine).
    DOCKER_BUILDKIT=1 docker build \
        --build-arg "BASE_IMAGE=$base_image" \
        --build-arg "PYTORCH_CUDA_TAG=$pytorch_cuda_tag" \
        -t "$image_tag" \
        -f "$dockerfile" \
        "$repo_root" \
        || { log_error "Docker build failed"; return 1; }

    local expected_release="${image_tag_release:-$base_release}"
    if [ -n "$expected_release" ] && ! engine_docker_image_matches_release "$image_tag" "$expected_release"; then
        local actual_release
        actual_release=$(engine_docker_image_tensorrt_release "$image_tag" || true)
        log_error "Built image TensorRT release mismatch: image=$image_tag expected=$expected_release actual=${actual_release:-unknown}"
        log_error "Base image was: $base_image"
        return 1
    fi
    if [ -n "$pytorch_cuda_tag" ] && ! engine_docker_image_matches_torch_cuda "$image_tag" "$pytorch_cuda_tag"; then
        local actual_torch_cuda_tag
        actual_torch_cuda_tag=$(engine_docker_image_torch_cuda_tag "$image_tag" || true)
        log_error "Built image PyTorch CUDA wheel mismatch: image=$image_tag expected=$pytorch_cuda_tag actual=${actual_torch_cuda_tag:-unknown}"
        return 1
    fi
    log_info "Image built: $image_tag"
}

# ---------------------------------------------------------------------------
#  engine_start_docker <repo_root> <variant> [options...]
#
#  Starts the standalone TTS engine server inside a Docker container.
#  Options: same as engine_start, plus --image <tag>
# ---------------------------------------------------------------------------
engine_start_docker() {
    local repo_root="$1"
    local variant="$2"
    shift 2

    local port="$ENGINE_GRPC_PORT"
    local ws_port="$ENGINE_WEBSOCKET_PORT"
    local health_port="$ENGINE_HEALTH_PORT"
    local device=0
    local max_batch=48
    local max_sessions=128
    local max_seq_len=""
    local image="$ENGINE_IMAGE"
    local container_name="$ENGINE_CONTAINER_NAME"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port)          port="$2"; shift 2 ;;
            --ws-port)       ws_port="$2"; shift 2 ;;
            --device)        device="$2"; shift 2 ;;
            --max-batch)     max_batch="$2"; shift 2 ;;
            --max-sessions)  max_sessions="$2"; shift 2 ;;
            --max-seq-len)   max_seq_len="$2"; shift 2 ;;
            --image)         image="$2"; shift 2 ;;
            --name)          container_name="$2"; shift 2 ;;
            *)               shift ;;
        esac
    done

    if docker inspect "$container_name" &>/dev/null; then
        log_warn "Container '$container_name' already exists"
        if docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null | grep -q true; then
            log_warn "Container is running. Stop first: deploy.sh stop"
            return 1
        fi
        docker rm "$container_name" >/dev/null 2>&1
    fi

    resolve_engine_package_paths "$repo_root" || return 1
    local model_repo_host="${MODEL_REPO_DIR:-$repo_root/workspace/model_repository}"
    local model_version
    model_version=$(resolve_model_version) || return 1
    local model_package_container="/models/tts_orchestrator/$model_version"

    log_step "Starting Engine (Docker: $image)"
    log_info "  Container:    $container_name"
    log_info "  Variant:      $variant"
    log_info "  Model repo:   $model_repo_host"
    log_info "  Model package:$model_package_container"
    log_info "  TRT Engine:   $_ENGINE_RUNTIME_ARTIFACT"
    log_info "  GPU Device:   $device"
    log_info "  Max Batch:    $max_batch"
    log_info "  Max Sessions: $max_sessions"
    log_info "  Max Seq Len:  ${max_seq_len:-auto}"
    log_info "  gRPC Port:    $port"
    log_info "  WS Port:      $ws_port"

    local -a run_cmd=(
        python3 -m engine.server
        --config /app/engine.yaml
        --model-package-dir "$model_package_container"
    )
    run_cmd+=(
        --device "$device"
        --max-batch "$max_batch"
        --max-sessions "$max_sessions"
        --port "$port"
        --ws-port "$ws_port"
    )

    local -a env_args=(
        -e "ENGINE_SCHEDULER_MAX_BATCH_SIZE=$max_batch"
        -e "ENGINE_SERVER_WEBSOCKET_PORT=$ws_port"
        -e "ENGINE_SERVER_HEALTH_PORT=$health_port"
    )
    if [[ -n "$max_seq_len" ]]; then
        env_args+=( -e "ENGINE_SCHEDULER_MAX_SEQ_LEN=$max_seq_len" )
    fi

    docker run --gpus all -d \
        --name "$container_name" \
        -w "$model_package_container" \
        -e "PYTHONPATH=$model_package_container:/app" \
        -v "$model_repo_host:/models:ro" \
        -p "${port}:${port}" \
        -p "${ws_port}:${ws_port}" \
        -p "${health_port}:${health_port}" \
        --shm-size=4g \
        "${env_args[@]}" \
        "$image" \
        "${run_cmd[@]}" \
        || { log_error "Docker run failed"; return 1; }

    log_info "Container started: $container_name"
    log_info "  Logs: docker logs -f $container_name"
}

# ---------------------------------------------------------------------------
#  engine_stop_docker [container_name]
#  Stops the engine Docker container.
# ---------------------------------------------------------------------------
engine_stop_docker() {
    local container_name="${1:-$ENGINE_CONTAINER_NAME}"
    if ! docker inspect "$container_name" &>/dev/null; then
        log_info "No container '$container_name' found"
        return 0
    fi
    log_info "Stopping container '$container_name'..."
    docker stop "$container_name" >/dev/null 2>&1
    docker rm "$container_name" >/dev/null 2>&1
    log_info "Container stopped and removed"
}

# ---------------------------------------------------------------------------
#  engine_show_status <repo_root>
#  Prints detailed status info (for human consumption).
# ---------------------------------------------------------------------------
engine_show_status() {
    local repo_root="$1"
    local pid_file
    pid_file=$(engine_pid_file "$repo_root")
    local log_file
    log_file=$(engine_log_file "$repo_root")
    local status
    status=$(engine_status "$repo_root")

    case "$status" in
        none)
            log_info "Standalone engine: NOT RUNNING"
            ;;
        running|healthy)
            local pid
            pid=$(cat "$pid_file" 2>/dev/null)
            log_info "Standalone engine: RUNNING (PID $pid)"
            if [ -f "$log_file" ]; then
                log_info "  Log file: $log_file"
                log_info "  Last 5 lines:"
                tail -5 "$log_file" 2>/dev/null | while IFS= read -r line; do
                    echo "    $line"
                done
            fi
            ;;
    esac
}
