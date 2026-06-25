#!/bin/bash
# ===========================================================================
#  trtexec_runner.sh — Abstract trtexec invocation across docker / host
#
#  Two runners with identical semantics:
#    docker: docker run --rm --gpus … -v <dir>:/mnt/model NGC_IMAGE trtexec …
#    host:   $TRTEXEC_HOST … (direct invocation, no container)
#
#  The runner is chosen by resolve_trtexec_runner() which honors the
#  BUILD_RUNNER env var (auto|docker|host).  Auto detection prefers
#  docker when docker + GPU runtime are available, falls back to host
#  trtexec (e.g. inside an Aliyun DSW container that has trtexec but
#  cannot run nested docker).
#
#  Public interface:
#    resolve_trtexec_runner          → echoes "docker" or "host"
#    _trtexec_run <mount_dir> <onnx_relpath> <engine_relpath> -- <trtexec args…>
#    probe_host_trtexec_path         → echoes resolved trtexec binary path
#    probe_host_trtexec_version      → echoes TensorRT version from host trtexec
#    runner_supports_nested_docker   → returns 0 when runner=docker (useful
#                                       for callers that need image probing)
#
#  Env vars:
#    BUILD_RUNNER       auto | docker | host        (default: auto)
#    TRTEXEC_HOST       path to host trtexec        (default: auto-detected)
#    NGC_IMAGE          NGC container image URI     (used by docker runner)
#    DOCKER_GPU_ARGS    bash array, e.g. (--gpus all)  (set by caller)
#
#  Depends: lib/logging.sh, lib/docker.sh (check_docker_gpu_ready)
# ===========================================================================

[[ -n "${_LIB_TRTEXEC_RUNNER_LOADED:-}" ]] && return 0
_LIB_TRTEXEC_RUNNER_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/docker.sh"

# Cache so repeated callers don't redo the smoke test in the same process.
_QWEN3_RUNNER_CACHED=""

# ---------------------------------------------------------------------------
#  probe_host_trtexec_path
#  Finds the first available trtexec binary on the host.
#  Checks $TRTEXEC_HOST, then PATH, then common NGC/DSW locations.
#  Echoes the path or returns 1 if none found.
# ---------------------------------------------------------------------------
probe_host_trtexec_path() {
    local candidates=()
    if [ -n "${TRTEXEC_HOST:-}" ]; then
        candidates+=("$TRTEXEC_HOST")
    fi
    candidates+=(
        "/usr/src/tensorrt/bin/trtexec"
        "/opt/tritonserver/bin/trtexec"
        "/usr/local/tensorrt/bin/trtexec"
    )

    local path
    for path in "${candidates[@]}"; do
        if [ -x "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    # Fallback: PATH lookup
    if command -v trtexec &>/dev/null; then
        command -v trtexec
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
#  probe_host_trtexec_version
#  Runs the resolved host trtexec and extracts the TensorRT version string.
#  Echoes "10.5.0.18" (or similar) on success, empty on failure.
# ---------------------------------------------------------------------------
probe_host_trtexec_version() {
    local bin
    bin=$(probe_host_trtexec_path) || return 1
    # trtexec --version prints lines like "TensorRT.trtexec [TensorRT v100500] …"
    # or "&&&& RUNNING TensorRT.trtexec … # /usr/src/tensorrt/bin/trtexec --version"
    # We accept either the v<digits> form or a dotted version on the same line.
    local raw
    raw=$("$bin" --version 2>&1 | head -20)
    local ver
    ver=$(echo "$raw" | sed -nE 's/.*\[TensorRT v([0-9]+)\].*/\1/p' | head -1)
    if [ -n "$ver" ]; then
        # v100500 → 10.05.00 → 10.5.0
        if [ "${#ver}" -ge 6 ]; then
            local major="${ver:0:2}"
            local minor="${ver:2:2}"
            local patch="${ver:4:2}"
            major=$((10#$major))
            minor=$((10#$minor))
            patch=$((10#$patch))
            echo "${major}.${minor}.${patch}"
            return 0
        fi
    fi
    # Fallback: dotted version on any line
    ver=$(echo "$raw" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
    if [ -n "$ver" ]; then
        echo "$ver"
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
#  _docker_runner_available
#  Internal: returns 0 if docker + NVIDIA runtime work.  Quieter than
#  check_docker_gpu_ready (which logs errors) — for auto-detection.
# ---------------------------------------------------------------------------
_docker_runner_available() {
    command -v docker &>/dev/null || return 1
    docker info &>/dev/null || return 1
    # Don't require the user to have selected an NGC image yet; just check
    # the NVIDIA runtime exists on this host.
    if docker info 2>/dev/null | grep -qi 'nvidia'; then
        return 0
    fi
    if command -v nvidia-container-cli &>/dev/null; then
        return 0
    fi
    if [ -f /etc/nvidia-container-runtime/config.toml ]; then
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
#  _host_runner_available
#  Internal: returns 0 if host trtexec is present and nvidia-smi works.
# ---------------------------------------------------------------------------
_host_runner_available() {
    probe_host_trtexec_path >/dev/null || return 1
    command -v nvidia-smi &>/dev/null || return 1
    return 0
}

# ---------------------------------------------------------------------------
#  resolve_trtexec_runner
#  Picks the runner based on BUILD_RUNNER (auto|docker|host).
#  Caches the decision in _QWEN3_RUNNER_CACHED for the rest of the process.
#  Echoes "docker" or "host" on success; non-zero exit on failure.
# ---------------------------------------------------------------------------
resolve_trtexec_runner() {
    if [ -n "$_QWEN3_RUNNER_CACHED" ]; then
        echo "$_QWEN3_RUNNER_CACHED"
        return 0
    fi

    local choice="${BUILD_RUNNER:-auto}"
    case "$choice" in
        docker)
            if ! _docker_runner_available; then
                log_error "BUILD_RUNNER=docker but docker + NVIDIA runtime not available"
                return 1
            fi
            _QWEN3_RUNNER_CACHED="docker"
            ;;
        host)
            if ! _host_runner_available; then
                log_error "BUILD_RUNNER=host but no host trtexec found"
                log_error "Set TRTEXEC_HOST to your trtexec binary or install TensorRT"
                return 1
            fi
            _QWEN3_RUNNER_CACHED="host"
            ;;
        auto)
            if _docker_runner_available; then
                _QWEN3_RUNNER_CACHED="docker"
            elif _host_runner_available; then
                _QWEN3_RUNNER_CACHED="host"
            else
                log_error "Cannot resolve trtexec runner: neither docker+NGC nor host trtexec is usable"
                log_error "  - For docker mode: install Docker + NVIDIA Container Toolkit"
                log_error "  - For host mode:   set TRTEXEC_HOST to your trtexec binary"
                return 1
            fi
            ;;
        *)
            log_error "Unknown BUILD_RUNNER value: $choice (use auto|docker|host)"
            return 1
            ;;
    esac

    echo "$_QWEN3_RUNNER_CACHED"
}

# ---------------------------------------------------------------------------
#  runner_supports_nested_docker
#  Returns 0 when runner=docker.  Used by callers that need to invoke
#  additional `docker run` commands (e.g. trtexec --version probe).
# ---------------------------------------------------------------------------
runner_supports_nested_docker() {
    local runner
    runner=$(resolve_trtexec_runner) || return 1
    [ "$runner" = "docker" ]
}

# ---------------------------------------------------------------------------
#  _trtexec_run_docker <mount_dir> <onnx_relpath> <engine_relpath> -- args…
#  Runs trtexec inside an NGC container with the directory mounted at
#  /mnt/model.  Requires NGC_IMAGE and DOCKER_GPU_ARGS to be set by caller.
# ---------------------------------------------------------------------------
_trtexec_run_docker() {
    local mount_dir="$1" onnx="$2" engine="$3"
    shift 3
    # Consume the "--" separator if present.
    if [ "${1:-}" = "--" ]; then
        shift
    fi

    if [ -z "${NGC_IMAGE:-}" ]; then
        log_error "_trtexec_run_docker: NGC_IMAGE is not set"
        return 1
    fi

    local trtexec_in_container="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
    local docker_gpu_args=()
    if [ -n "${DOCKER_GPU_ARGS+x}" ]; then
        docker_gpu_args=("${DOCKER_GPU_ARGS[@]}")
    else
        docker_gpu_args=(--gpus all)
    fi

    docker run --rm "${docker_gpu_args[@]}" \
        -v "$mount_dir:/mnt/model" \
        "$NGC_IMAGE" \
        "$trtexec_in_container" \
        --onnx="/mnt/model/$onnx" \
        --saveEngine="/mnt/model/$engine" \
        "$@"
}

# ---------------------------------------------------------------------------
#  _trtexec_run_host <mount_dir> <onnx_relpath> <engine_relpath> -- args…
#  Runs the host trtexec directly with absolute paths derived from
#  mount_dir.  No container is started.
# ---------------------------------------------------------------------------
_trtexec_run_host() {
    local mount_dir="$1" onnx="$2" engine="$3"
    shift 3
    if [ "${1:-}" = "--" ]; then
        shift
    fi

    local bin
    bin=$(probe_host_trtexec_path) || {
        log_error "_trtexec_run_host: no trtexec found on host"
        return 1
    }

    "$bin" \
        --onnx="$mount_dir/$onnx" \
        --saveEngine="$mount_dir/$engine" \
        "$@"
}

# ---------------------------------------------------------------------------
#  _trtexec_run <mount_dir> <onnx_relpath> <engine_relpath> -- <trtexec args…>
#  Dispatches to the resolved runner.  The "--" separator between the
#  positional triple and the trailing trtexec args is conventional but
#  optional (the helpers strip it).
# ---------------------------------------------------------------------------
_trtexec_run() {
    local runner
    runner=$(resolve_trtexec_runner) || return 1
    "_trtexec_run_${runner}" "$@"
}
