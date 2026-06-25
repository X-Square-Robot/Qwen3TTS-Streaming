#!/bin/bash
# ===========================================================================
#  utils.sh — General-purpose shell utilities
#
#  Functions: retry, require_cmd, ensure_workdir, check_gpu,
#             normalize_gpu_device, select_best_gpu_index,
#             resolve_gpu_device_index, gpu_total_memory_mb
#  Depends:   lib/logging.sh
# ===========================================================================

[[ -n "${_LIB_UTILS_LOADED:-}" ]] && return 0
_LIB_UTILS_LOADED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logging.sh"

# ---------------------------------------------------------------------------
#  retry <max_attempts> <initial_delay_sec> <command...>
#  Retries a command with exponential back-off.
# ---------------------------------------------------------------------------
retry() {
    local max_attempts="${1:-3}"
    local delay="${2:-5}"
    shift 2
    local attempt=1
    while [ "$attempt" -le "$max_attempts" ]; do
        if "$@"; then
            return 0
        fi
        log_warn "Attempt $attempt/$max_attempts failed: $*"
        if [ "$attempt" -lt "$max_attempts" ]; then
            log_info "Retrying in ${delay}s..."
            sleep "$delay"
            delay=$((delay * 2))
        fi
        attempt=$((attempt + 1))
    done
    log_error "All $max_attempts attempts failed: $*"
    return 1
}

# ---------------------------------------------------------------------------
#  require_cmd <command>
#  Exits with error if the command is not available.
# ---------------------------------------------------------------------------
require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" &>/dev/null; then
        log_error "Required command not found: $cmd"
        return 1
    fi
}

# ---------------------------------------------------------------------------
#  resolve_model_version [raw]
#  Normalizes the shared Triton model version number.
#  Precedence: explicit arg > MODEL_VERSION > ENGINE_MODEL_VERSION > 1.
# ---------------------------------------------------------------------------
resolve_model_version() {
    local raw="${1:-${MODEL_VERSION:-${ENGINE_MODEL_VERSION:-1}}}"
    raw="${raw//[[:space:]]/}"

    if [[ "$raw" =~ ^[1-9][0-9]*$ ]]; then
        echo "$raw"
        return 0
    fi

    log_error "Invalid model version: ${raw:-<empty>}"
    log_error "Use a positive integer like 1 or 2."
    return 1
}

# ---------------------------------------------------------------------------
#  ensure_workdir <path>
#  Creates the directory if needed, then cd into it.
# ---------------------------------------------------------------------------
ensure_workdir() {
    local WORKDIR="$1"
    if [ -z "$WORKDIR" ]; then
        log_error "Workdir is not set"
        return 1
    fi
    if [ ! -d "$WORKDIR" ]; then
        log_info "Workdir($WORKDIR) not found, creating..."
        mkdir -p "$WORKDIR" || { log_error "Failed to create $WORKDIR"; return 1; }
    fi
    cd "$WORKDIR" || { log_error "Failed to cd into $WORKDIR"; return 1; }
    log_info "Working directory: $(pwd)"
}

# ---------------------------------------------------------------------------
#  check_gpu
#  Detects NVIDIA GPUs via nvidia-smi; returns 1 if none found.
# ---------------------------------------------------------------------------
check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        log_warn "nvidia-smi not found — no NVIDIA GPU driver detected"
        return 1
    fi
    local gpu_count
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$gpu_count" -eq 0 ]; then
        log_warn "No NVIDIA GPU detected"
        return 1
    fi
    log_info "Detected $gpu_count GPU(s):"
    nvidia-smi -L
}

# ---------------------------------------------------------------------------
#  normalize_gpu_device <raw>
#  Normalizes user-facing GPU selectors:
#    auto | "" -> auto
#    all       -> all
#    1         -> 1
#    cuda:1    -> 1
#    device=1  -> 1
# ---------------------------------------------------------------------------
normalize_gpu_device() {
    local raw="${1:-auto}"
    raw="${raw,,}"
    raw="${raw//[[:space:]]/}"

    case "$raw" in
        ""|auto)
            echo "auto"
            return 0
            ;;
        all)
            echo "all"
            return 0
            ;;
        cuda:*) raw="${raw#cuda:}" ;;
        device=*) raw="${raw#device=}" ;;
    esac

    if [[ "$raw" =~ ^[0-9]+$ ]]; then
        echo "$raw"
        return 0
    fi

    log_error "Invalid GPU device selector: ${1:-}"
    log_error "Use auto, all, a numeric id (e.g. 1), cuda:1, or device=1."
    return 1
}

# ---------------------------------------------------------------------------
#  select_best_gpu_index
#  Prints the GPU index with the most currently free memory.
# ---------------------------------------------------------------------------
select_best_gpu_index() {
    if ! command -v nvidia-smi &>/dev/null; then
        log_warn "nvidia-smi not found; falling back to GPU 0"
        echo "0"
        return 0
    fi

    local best
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, '
            {
                idx=$1; free=$2;
                gsub(/^[ \t]+|[ \t]+$/, "", idx);
                gsub(/^[ \t]+|[ \t]+$/, "", free);
                if (free + 0 > best_free + 0 || NR == 1) {
                    best_idx=idx;
                    best_free=free + 0;
                }
            }
            END {
                if (best_idx != "") print best_idx;
            }
        ') || true

    if [ -z "$best" ]; then
        log_warn "Could not query GPU free memory; falling back to GPU 0"
        best="0"
    fi
    echo "$best"
}

# ---------------------------------------------------------------------------
#  resolve_gpu_device_index [raw]
#  Resolves auto/cuda:N/N into a concrete numeric GPU id.
# ---------------------------------------------------------------------------
resolve_gpu_device_index() {
    local raw="${1:-auto}"
    local norm
    norm=$(normalize_gpu_device "$raw") || return 1
    if [ "$norm" = "all" ]; then
        log_error "GPU selector 'all' is not valid where a single device id is required"
        return 1
    fi
    if [ "$norm" = "auto" ]; then
        select_best_gpu_index
    else
        echo "$norm"
    fi
}

# ---------------------------------------------------------------------------
#  gpu_total_memory_mb <gpu_index>
#  Prints total memory in MiB for a GPU id; prints 0 if unavailable.
# ---------------------------------------------------------------------------
gpu_total_memory_mb() {
    local gpu_id="${1:-0}"
    if ! command -v nvidia-smi &>/dev/null; then
        echo "0"
        return 0
    fi
    nvidia-smi --id="$gpu_id" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -1 \
        | awk '{print int($1)}' \
        || echo "0"
}

# ---------------------------------------------------------------------------
#  gpu_free_memory_mb <gpu_index>
#  Prints free memory in MiB for a GPU id; prints 0 if unavailable.
# ---------------------------------------------------------------------------
gpu_free_memory_mb() {
    local gpu_id="${1:-0}"
    if ! command -v nvidia-smi &>/dev/null; then
        echo "0"
        return 0
    fi
    nvidia-smi --id="$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -1 \
        | awk '{print int($1)}' \
        || echo "0"
}

# ---------------------------------------------------------------------------
#  log_gpu_selection <label> <gpu_index>
# ---------------------------------------------------------------------------
log_gpu_selection() {
    local label="$1"
    local gpu_id="$2"
    if command -v nvidia-smi &>/dev/null; then
        local row
        row=$(nvidia-smi --id="$gpu_id" --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null \
            | head -1 || true)
        if [ -n "$row" ]; then
            log_info "$label GPU: $row (index, uuid, name, total MiB, free MiB)"
            return 0
        fi
    fi
    log_info "$label GPU: $gpu_id"
}
