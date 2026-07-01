#!/bin/bash
# ===========================================================================
#  compose.sh — Unified Docker Compose workflow for engine / Triton deploy
#
#  Wraps compose.yaml with project-specific defaults:
#    - engine: standalone engine.server in Docker with mounted model_repository
#    - triton: Triton Inference Server with the same mounted model_repository
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/tools.sh"

COMPOSE_FILE="${REPO_ROOT}/infra/docker/compose.yaml"
COMPOSE_DEV_FILE="${REPO_ROOT}/infra/docker/compose.dev.yaml"
EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
MODEL_REPO_DIR="${MODEL_REPO_DIR:-${REPO_ROOT}/workspace/model_repository}"
MODEL_VERSION="${MODEL_VERSION:-${ENGINE_MODEL_VERSION:-1}}"

COMMAND=""
GATEWAY="all"
VARIANT=""
ENGINE_MODE="${ENGINE_MODE:-trt}"
DRY_RUN=false
NO_HEALTH_CHECK=false
FORCE_PREPARE=false
FOLLOW=false
BUILD_BEFORE_UP=false
USE_DEV_OVERLAY=false
WATCH_NO_UP=false
WATCH_QUIET=false
WATCH_NO_PRUNE=false
IMAGE_OVERRIDE=""
CONTAINER_OVERRIDE=""

ENGINE_PORT="${ENGINE_GRPC_PORT:-50051}"
ENGINE_WEBSOCKET="${ENGINE_WEBSOCKET_PORT:-50052}"
ENGINE_HEALTH="${ENGINE_HEALTH_PORT:-8080}"
ENGINE_DEVICE="${ENGINE_DEVICE:-${RUNTIME_GPU_DEVICE:-auto}}"
ENGINE_MAX_BATCH="${ENGINE_MAX_BATCH_SIZE:-${RUNTIME_MAX_BATCH_SIZE:-}}"
ENGINE_MAX_SESSIONS="${ENGINE_MAX_SESSIONS:-128}"
ENGINE_MAX_SEQ_LEN="${ENGINE_MAX_SEQ_LEN:-${RUNTIME_MAX_SEQ_LEN:-}}"

TRITON_HTTP="${TRITON_HTTP_PORT:-8000}"
TRITON_GRPC="${TRITON_GRPC_PORT:-8001}"
TRITON_METRICS="${TRITON_METRICS_PORT:-8002}"
TRITON_GPU_DEVICE="${TRITON_GPU_DEVICE:-$ENGINE_DEVICE}"
TRITON_MAX_BATCH="${TRITON_MAX_BATCH_SLOTS:-${RUNTIME_MAX_BATCH_SIZE:-}}"
TRITON_MAX_SEQ_LEN="${TRITON_MAX_SEQ_LEN:-${RUNTIME_MAX_SEQ_LEN:-}}"

usage() {
    cat <<'EOF'
Usage: compose.sh <command> [options]

Commands:
  build                  Build compose image(s)
  prepare                Assemble shared model_repository
  up                     Start compose service(s)
  watch                  Watch files and auto refresh/rebuild service(s)
  down                   Stop/remove compose service(s)
  logs                   Show service logs
  ps                     Show compose status
  config                 Render resolved compose config

Options:
  --gateway <mode>       engine | triton | all (default: all)
  --variant <name>       Model variant (auto-discover when possible)
  --engine-mode <mode>   trt | onnx for model_repository assembly (default: trt)
  --repo-dir <path>      Shared model_repository path
  --model-version <N>    Triton model version directory (default: 1)
  --image <tag>          Override service image tag for selected gateway
  --container <name>     Override container name for selected gateway
  --port <N>             Engine gRPC port
  --ws-port <N>          Engine WebSocket port
  --health-port <N>      Engine health port
  --grpc-port <N>        Triton gRPC port
  --http-port <N>        Triton HTTP port
  --metrics-port <N>     Triton metrics port
  --device <N|auto>      Runtime CUDA device (engine and Triton)
  --triton-device <N>    Triton CUDA device override
  --max-batch <N>        Runtime max batch size
  --max-sessions <N>     Engine max sessions
  --max-seq-len <N>      Optional engine scheduler max seq len override
  --build                Build before `up`
  --dev                  Enable compose.dev.yaml source bind mounts
  --prepare              Force Triton repo assembly before `up`
  --no-up                For `watch`, do not start services before watching
  --quiet                For `watch`, hide build output
  --no-prune             For `watch`, keep dangling images after rebuild
  --no-health-check      Skip readiness wait after `up`
  --follow               Follow logs (for `logs`)
  --dry-run              Print the compose command instead of executing
  -h, --help             Show this help
EOF
}

discover_first_variant() {
    local vdir
    for vdir in "$EXPORTED_DIR"/*/; do
        local vname
        vname=$(basename "$vdir")
        [[ "$vname" == "tokenizer" ]] && continue
        if [[ -d "$vdir/weights" ]]; then
            printf '%s\n' "$vname"
            return 0
        fi
    done
    return 1
}

resolve_variant_if_needed() {
    if [[ -n "$VARIANT" ]]; then
        if [[ ! -d "$EXPORTED_DIR/$VARIANT" ]]; then
            log_error "Variant not found: $EXPORTED_DIR/$VARIANT"
            exit 1
        fi
        return 0
    fi
    VARIANT=$(discover_first_variant) || {
        log_error "No exported variants found in $EXPORTED_DIR"
        exit 1
    }
    log_info "Auto-discovered variant: $VARIANT"
}

require_docker_compose() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker is required"
        exit 1
    fi
    if ! docker compose version &>/dev/null; then
        log_error "docker compose is required"
        exit 1
    fi
}

require_docker_compose_if_needed() {
    $DRY_RUN && return 0
    require_docker_compose
}

compose_cmd() {
    local profile_args=()
    case "$GATEWAY" in
        engine) profile_args=(--profile engine) ;;
        triton) profile_args=(--profile triton) ;;
        all) profile_args=(--profile engine --profile triton) ;;
        *)
            log_error "Unknown gateway: $GATEWAY"
            exit 1
            ;;
    esac

    local compose_files=(-f "$COMPOSE_FILE")
    if $USE_DEV_OVERLAY && [[ -f "$COMPOSE_DEV_FILE" ]]; then
        compose_files+=(-f "$COMPOSE_DEV_FILE")
    fi

    local cmd=(docker compose "${compose_files[@]}" "${profile_args[@]}")
    cmd+=("$@")

    if $DRY_RUN; then
        printf '[DRY RUN] '
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    "${cmd[@]}"
}

export_compose_env() {
    resolve_compose_runtime_controls

    export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-qwen3-tts}"
    export MODEL_VARIANT="$VARIANT"
    export MODEL_REPO_DIR="$MODEL_REPO_DIR"
    MODEL_VERSION=$(resolve_model_version "$MODEL_VERSION") || exit 1
    export MODEL_VERSION
    export ENGINE_MODEL_VERSION="$MODEL_VERSION"
    export ENGINE_MODEL_PACKAGE_DIR="/models/tts_orchestrator/$MODEL_VERSION"

    export ENGINE_GRPC_PORT="$ENGINE_PORT"
    export ENGINE_WEBSOCKET_PORT="$ENGINE_WEBSOCKET"
    export ENGINE_HEALTH_PORT="$ENGINE_HEALTH"
    export ENGINE_DEVICE="$ENGINE_DEVICE"
    export ENGINE_MAX_BATCH_SIZE="$ENGINE_MAX_BATCH"
    export ENGINE_MAX_SESSIONS="$ENGINE_MAX_SESSIONS"
    export ENGINE_MAX_SEQ_LEN="$ENGINE_MAX_SEQ_LEN"

    export TRITON_HTTP_PORT="$TRITON_HTTP"
    export TRITON_GRPC_PORT="$TRITON_GRPC"
    export TRITON_METRICS_PORT="$TRITON_METRICS"
    export TRITON_GPU_DEVICE="$TRITON_GPU_DEVICE"
    export TRITON_MAX_BATCH_SLOTS="$TRITON_MAX_BATCH"
    export TRITON_MAX_SEQ_LEN="$TRITON_MAX_SEQ_LEN"
    export ENGINE_MAX_DECODE_LEN="$TRITON_MAX_SEQ_LEN"
}

_compose_manifest_ngc_tag() {
    local tag=""
    tag=$(resolve_manifest_ngc_tag "$MODEL_REPO_DIR" "$MODEL_VERSION" 2>/dev/null || true)
    if [[ -z "$tag" && -n "$VARIANT" && -f "$EXPORTED_DIR/$VARIANT/triton_manifest.json" ]]; then
        tag=$(resolve_manifest_ngc_tag "$EXPORTED_DIR/$VARIANT/triton_manifest.json" "$MODEL_VERSION" 2>/dev/null || true)
    fi
    if [[ -z "$tag" ]]; then
        tag=$(resolve_ngc_tag 2>/dev/null || true)
    fi
    printf '%s\n' "$tag"
}

_compose_ngc_tag_from_image() {
    local image="${1:-}"
    local tag="${image##*:}"
    if [[ "$tag" =~ ^[0-9]{2}\.[0-9]{2}$ ]]; then
        printf '%s\n' "$tag"
    fi
}

_compose_apply_engine_ngc_defaults() {
    local ngc_tag="$1"
    export ENGINE_BASE_IMAGE="${ENGINE_BASE_IMAGE:-nvcr.io/nvidia/tensorrt:${ngc_tag}-py3}"
    if [[ -z "${ENGINE_PYTORCH_CUDA_TAG:-}" ]]; then
        if [[ -n "${PYTORCH_CUDA_TAG:-}" ]]; then
            export ENGINE_PYTORCH_CUDA_TAG="$PYTORCH_CUDA_TAG"
        else
            local torch_cuda_tag
            torch_cuda_tag=$(resolve_ngc_torch_index_tag "$ngc_tag" 2>/dev/null || true)
            [[ -n "$torch_cuda_tag" ]] && export ENGINE_PYTORCH_CUDA_TAG="$torch_cuda_tag"
        fi
    fi
}

_compose_apply_triton_ngc_defaults() {
    local ngc_tag="$1"
    export TRITON_BASE_IMAGE="${TRITON_BASE_IMAGE:-nvcr.io/nvidia/tritonserver:${ngc_tag}-py3}"
    if [[ -z "${TRITON_PYTORCH_CUDA_TAG:-}" ]]; then
        if [[ -n "${PYTORCH_CUDA_TAG:-}" ]]; then
            export TRITON_PYTORCH_CUDA_TAG="$PYTORCH_CUDA_TAG"
        else
            local torch_cuda_tag
            torch_cuda_tag=$(resolve_ngc_torch_index_tag "$ngc_tag" 2>/dev/null || true)
            [[ -n "$torch_cuda_tag" ]] && export TRITON_PYTORCH_CUDA_TAG="$torch_cuda_tag"
        fi
    fi
    # TensorRT Python version must match the NGC tag's TRT (and the Phase B plan);
    # resolve from the matrix so Dockerfile.triton doesn't fall back to its default
    # (which is the newest tag's version and mismatches an older base's plan).
    if [[ -z "${TRITON_TENSORRT_PYTHON_VERSION:-}" ]]; then
        local trt_ver
        trt_ver=$(resolve_ngc_tag_tensorrt_version "$ngc_tag" 2>/dev/null || true)
        [[ -n "$trt_ver" ]] && export TRITON_TENSORRT_PYTHON_VERSION="$trt_ver"
    fi
}

resolve_compose_image_defaults() {
    local manifest_ngc_tag=""

    if [[ "$GATEWAY" == "engine" || "$GATEWAY" == "all" ]]; then
        local engine_ngc_tag=""
        if [[ -z "${IMAGE_OVERRIDE:-}" && ( -z "${ENGINE_IMAGE:-}" || "${ENGINE_IMAGE:-}" == "qwen3-engine:26.02" ) ]]; then
            manifest_ngc_tag="$(_compose_manifest_ngc_tag)"
            engine_ngc_tag="$manifest_ngc_tag"
            if [[ -n "$engine_ngc_tag" ]]; then
                export ENGINE_IMAGE="qwen3-engine:${engine_ngc_tag}"
                log_info "Using engine image from Phase B manifest: $ENGINE_IMAGE"
            fi
        elif [[ -n "${ENGINE_IMAGE:-}" ]]; then
            engine_ngc_tag="$(_compose_ngc_tag_from_image "$ENGINE_IMAGE")"
            if [[ -n "$engine_ngc_tag" ]]; then
                log_info "Using engine build base from image tag: $ENGINE_IMAGE -> NGC $engine_ngc_tag"
            fi
        fi
        if [[ -n "$engine_ngc_tag" ]]; then
            _compose_apply_engine_ngc_defaults "$engine_ngc_tag"
        fi
    fi

    if [[ "$GATEWAY" == "triton" || "$GATEWAY" == "all" ]]; then
        local triton_ngc_tag=""
        if [[ -z "${IMAGE_OVERRIDE:-}" && ( -z "${TRITON_IMAGE:-}" || "${TRITON_IMAGE:-}" == "qwen3tts-streaming:26.02" ) ]]; then
            manifest_ngc_tag="${manifest_ngc_tag:-$(_compose_manifest_ngc_tag)}"
            triton_ngc_tag="$manifest_ngc_tag"
            if [[ -n "$triton_ngc_tag" ]]; then
                export TRITON_IMAGE="qwen3tts-streaming:${triton_ngc_tag}"
                log_info "Using Triton image from Phase B manifest: $TRITON_IMAGE"
            fi
        elif [[ -n "${TRITON_IMAGE:-}" ]]; then
            triton_ngc_tag="$(_compose_ngc_tag_from_image "$TRITON_IMAGE")"
            if [[ -n "$triton_ngc_tag" ]]; then
                log_info "Using Triton build base from image tag: $TRITON_IMAGE -> NGC $triton_ngc_tag"
            fi
        fi
        if [[ -n "$triton_ngc_tag" ]]; then
            _compose_apply_triton_ngc_defaults "$triton_ngc_tag"
        fi
    fi
}

log_compose_runtime_summary() {
    log_info "[DRY RUN] Resolved compose runtime:"
    log_info "  Gateway:          $GATEWAY"
    [ -n "$VARIANT" ] && log_info "  Variant:          $VARIANT"
    log_info "  Model version:    $MODEL_VERSION"
    log_info "  Model repo:       $MODEL_REPO_DIR"
    [[ "$GATEWAY" == "engine" || "$GATEWAY" == "all" ]] && log_info "  Engine image:     ${ENGINE_IMAGE:-compose default}"
    [[ "$GATEWAY" == "triton" || "$GATEWAY" == "all" ]] && log_info "  Triton image:     ${TRITON_IMAGE:-compose default}"
    log_info "  Engine device:    $ENGINE_DEVICE"
    log_info "  Engine max batch: $ENGINE_MAX_BATCH"
    log_info "  Engine max seq:   $ENGINE_MAX_SEQ_LEN"
    log_info "  Triton device:    $TRITON_GPU_DEVICE"
    log_info "  Triton max batch: $TRITON_MAX_BATCH"
    log_info "  Triton max seq:   $TRITON_MAX_SEQ_LEN"
}

verify_compose_engine_image() {
    $DRY_RUN && return 0

    local image="${ENGINE_IMAGE:-qwen3-engine:26.02}"
    local image_tag_release=""
    image_tag_release="$(_compose_ngc_tag_from_image "$image")"
    local base_release=""
    if [[ -n "${ENGINE_BASE_IMAGE:-}" && "$ENGINE_BASE_IMAGE" =~ :([0-9]{2}\.[0-9]{2})-py3$ ]]; then
        base_release="${BASH_REMATCH[1]}"
    fi
    if [[ -n "$image_tag_release" && -n "$base_release" && "$image_tag_release" != "$base_release" ]]; then
        log_error "Engine image tag/base image mismatch: image=$image implies $image_tag_release but ENGINE_BASE_IMAGE=$ENGINE_BASE_IMAGE"
        return 1
    fi
    local expected_release="${image_tag_release:-$base_release}"

    if [[ -n "$expected_release" ]] && ! engine_docker_image_matches_release "$image" "$expected_release"; then
        local actual_release
        actual_release=$(engine_docker_image_tensorrt_release "$image" || true)
        log_error "Built engine image TensorRT release mismatch: image=$image expected=$expected_release actual=${actual_release:-unknown}"
        log_error "Check ENGINE_BASE_IMAGE / --image; Docker tags do not guarantee image contents."
        return 1
    fi

    local expected_torch_cuda_tag="${ENGINE_PYTORCH_CUDA_TAG:-}"
    if [[ -n "$expected_torch_cuda_tag" ]] && ! engine_docker_image_matches_torch_cuda "$image" "$expected_torch_cuda_tag"; then
        local actual_torch_cuda_tag
        actual_torch_cuda_tag=$(engine_docker_image_torch_cuda_tag "$image" || true)
        log_error "Built engine image PyTorch CUDA wheel mismatch: image=$image expected=$expected_torch_cuda_tag actual=${actual_torch_cuda_tag:-unknown}"
        return 1
    fi
}

compose_manifest_profile_value() {
    local key="$1"
    local manifest="$EXPORTED_DIR/$VARIANT/triton_manifest.json"
    if [[ -z "$VARIANT" || ! -f "$manifest" ]]; then
        return 0
    fi
    python3 - "$manifest" "$key" <<'PY' 2>/dev/null || true
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
value = data.get("engine_profile", {}).get(key, "")
if value not in ("", None):
    print(value)
PY
}

compose_manifest_tts_model_type() {
    local manifest="$EXPORTED_DIR/$VARIANT/triton_manifest.json"
    if [[ -z "$VARIANT" || ! -f "$manifest" ]]; then
        return 0
    fi
    python3 - "$manifest" <<'PY' 2>/dev/null || true
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(
    data.get("tts_model_type")
    or data.get("orchestrator", {}).get("tts_model_type")
    or ""
)
PY
}

resolve_compose_runtime_controls() {
    if [[ "${ENGINE_DEVICE:-auto}" = "auto" || -z "${ENGINE_DEVICE:-}" ]]; then
        ENGINE_DEVICE=$(resolve_gpu_device_index "${ENGINE_DEVICE:-auto}") || exit 1
    else
        ENGINE_DEVICE=$(resolve_gpu_device_index "$ENGINE_DEVICE") || exit 1
    fi
    if [[ "${TRITON_GPU_DEVICE:-auto}" = "auto" || -z "${TRITON_GPU_DEVICE:-}" ]]; then
        TRITON_GPU_DEVICE="$ENGINE_DEVICE"
    else
        TRITON_GPU_DEVICE=$(resolve_gpu_device_index "$TRITON_GPU_DEVICE") || exit 1
    fi

    if [[ -z "${ENGINE_MAX_BATCH:-}" ]]; then
        local profile_max_batch model_type default_base_batch
        profile_max_batch=$(compose_manifest_profile_value max_batch_size)
        model_type="$(compose_manifest_tts_model_type | tr '[:upper:]' '[:lower:]')"
        if [[ "$GATEWAY" == "engine" || "$GATEWAY" == "all" ]] \
            && [[ "$model_type" == "base" || "$model_type" == "icl" || "$VARIANT" == base-* || "$VARIANT" == icl-* ]]; then
            default_base_batch="${ENGINE_BASE_ICL_MAX_BATCH:-8}"
            ENGINE_MAX_BATCH="$default_base_batch"
            if [[ "$ENGINE_MAX_BATCH" =~ ^[1-9][0-9]*$ && "$profile_max_batch" =~ ^[1-9][0-9]*$ ]] \
                && (( ENGINE_MAX_BATCH > profile_max_batch )); then
                ENGINE_MAX_BATCH="$profile_max_batch"
            fi
            log_info "Using conservative Base/ICL engine max batch: $ENGINE_MAX_BATCH"
        else
            ENGINE_MAX_BATCH="$profile_max_batch"
            ENGINE_MAX_BATCH="${ENGINE_MAX_BATCH:-128}"
        fi
    fi
    if [[ -z "${ENGINE_MAX_SEQ_LEN:-}" ]]; then
        ENGINE_MAX_SEQ_LEN=$(compose_manifest_profile_value max_seq_len)
        ENGINE_MAX_SEQ_LEN="${ENGINE_MAX_SEQ_LEN:-512}"
    fi
    if [[ -z "${TRITON_MAX_BATCH:-}" ]]; then
        TRITON_MAX_BATCH="$ENGINE_MAX_BATCH"
    fi
    if [[ -z "${TRITON_MAX_SEQ_LEN:-}" ]]; then
        TRITON_MAX_SEQ_LEN="$ENGINE_MAX_SEQ_LEN"
    fi

    local name value
    for name in ENGINE_MAX_BATCH ENGINE_MAX_SEQ_LEN TRITON_MAX_BATCH TRITON_MAX_SEQ_LEN; do
        value="${!name}"
        if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
            log_error "$name must be a positive integer, got: $value"
            exit 1
        fi
    done
}

compose_service_container_name() {
    local service="$1"
    case "$service" in
        engine) printf '%s\n' "${ENGINE_CONTAINER_NAME:-qwen3-engine}" ;;
        triton) printf '%s\n' "${TRITON_CONTAINER_NAME:-qwen3tts-streaming}" ;;
        *)
            log_error "Unknown compose service: $service"
            return 1
            ;;
    esac
}

compose_container_label() {
    local container="$1"
    local label="$2"
    docker inspect -f "{{ index .Config.Labels \"$label\" }}" "$container" 2>/dev/null || true
}

compose_container_is_managed_service() {
    local container="$1"
    local service="$2"
    local project="${COMPOSE_PROJECT_NAME:-qwen3-tts}"
    local cproject
    local cservice
    cproject=$(compose_container_label "$container" "com.docker.compose.project")
    cservice=$(compose_container_label "$container" "com.docker.compose.service")
    [[ "$cproject" == "$project" && "$cservice" == "$service" ]]
}

compose_container_network_count() {
    local container="$1"
    docker inspect -f '{{len .NetworkSettings.Networks}}' "$container" 2>/dev/null || printf '0\n'
}

compose_container_published_count() {
    local container="$1"
    local target_port="$2"
    docker inspect -f "{{with index .NetworkSettings.Ports \"${target_port}/tcp\"}}{{len .}}{{else}}0{{end}}" \
        "$container" 2>/dev/null || printf '0\n'
}

compose_remove_stale_container_if_needed() {
    local service="$1"
    local container="$2"
    shift 2

    docker inspect "$container" &>/dev/null || return 0

    if ! compose_container_is_managed_service "$container" "$service"; then
        log_warn "Container '$container' already exists but is not managed by this compose project/service."
        log_warn "If compose fails with a name conflict, stop or rename that container first."
        return 0
    fi

    local network_count
    network_count=$(compose_container_network_count "$container")
    local stale_reason=""
    if [[ "$network_count" == "0" ]]; then
        stale_reason="no Docker network attachment"
    else
        local port
        local published_count
        for port in "$@"; do
            published_count=$(compose_container_published_count "$container" "$port")
            if [[ "$published_count" == "0" ]]; then
                stale_reason="missing published port ${port}/tcp"
                break
            fi
        done
    fi

    if [[ -n "$stale_reason" ]]; then
        log_warn "Removing stale compose container '$container' ($stale_reason)."
        log_warn "Docker Compose will recreate it with fresh network and port bindings."
        docker rm -f "$container" >/dev/null
    fi
}

compose_preflight_service() {
    local service="$1"
    local container
    container=$(compose_service_container_name "$service")
    case "$service" in
        engine)
            compose_remove_stale_container_if_needed \
                "$service" "$container" "$ENGINE_PORT" "$ENGINE_WEBSOCKET" "$ENGINE_HEALTH"
            ;;
        triton)
            compose_remove_stale_container_if_needed "$service" "$container" 8000 8001 8002
            ;;
    esac
}

compose_assert_service_network() {
    local service="$1"
    local container
    container=$(compose_service_container_name "$service")
    shift

    if ! docker inspect "$container" &>/dev/null; then
        log_error "Compose service '$service' did not create container '$container'"
        return 1
    fi

    local status
    status=$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true)
    if [[ "$status" != "running" ]]; then
        log_error "Container '$container' is not running (status=$status)"
        compose_diagnose_service "$service"
        return 1
    fi

    local network_count
    network_count=$(compose_container_network_count "$container")
    if [[ "$network_count" == "0" ]]; then
        log_error "Container '$container' has no Docker network attachment"
        compose_diagnose_service "$service"
        return 1
    fi

    local port
    local published_count
    for port in "$@"; do
        published_count=$(compose_container_published_count "$container" "$port")
        if [[ "$published_count" == "0" ]]; then
            log_error "Container '$container' is missing published port ${port}/tcp"
            compose_diagnose_service "$service"
            return 1
        fi
    done
}

compose_diagnose_service() {
    local service="$1"
    local container
    container=$(compose_service_container_name "$service")

    if ! docker inspect "$container" &>/dev/null; then
        return 0
    fi

    log_info "Docker inspect summary for '$container':"
    docker inspect "$container" \
        --format '  status={{.State.Status}} restarting={{.State.Restarting}} exit={{.State.ExitCode}} network_mode={{.HostConfig.NetworkMode}} ports={{json .NetworkSettings.Ports}} networks={{json .NetworkSettings.Networks}}' \
        2>/dev/null || true
    log_info "Recent logs for '$container':"
    docker logs --tail 80 "$container" 2>&1 || true
}

compose_wait_engine_http_health() {
    local port="${1:-$ENGINE_HEALTH}"
    local timeout="${2:-90}"
    local elapsed=0
    local interval=2
    local url="http://localhost:${port}/health"

    if ! command -v curl &>/dev/null; then
        engine_health_check "$ENGINE_PORT" "$timeout"
        return $?
    fi

    while [ "$elapsed" -lt "$timeout" ]; do
        local body
        body=$(curl -fsS "$url" 2>/dev/null || true)
        if printf '%s\n' "$body" | grep -q '"running"[[:space:]]*:[[:space:]]*true'; then
            log_info "Engine HTTP health ready at localhost:${port}"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    log_error "Engine HTTP health check timed out after ${timeout}s"
    return 1
}

compose_wait_engine_ready() {
    compose_assert_service_network engine "$ENGINE_PORT" "$ENGINE_WEBSOCKET" "$ENGINE_HEALTH" || return 1
    if compose_wait_engine_http_health "$ENGINE_HEALTH" 90; then
        return 0
    fi
    compose_diagnose_service engine
    return 1
}

compose_wait_triton_ready() {
    compose_assert_service_network triton 8000 8001 8002 || return 1
    if triton_health_check "localhost" "$TRITON_HTTP" 120; then
        return 0
    fi
    compose_diagnose_service triton
    return 1
}

gateway_uses_engine_container() {
    [[ "$GATEWAY" == "engine" || "$GATEWAY" == "all" ]]
}

prepare_model_repo() {
    resolve_variant_if_needed
    if gateway_uses_engine_container && [[ "$ENGINE_MODE" != "trt" ]]; then
        log_error "Engine Docker consumes the shared model_repository, but requires engine-mode=trt."
        log_error "Use --engine-mode trt for --gateway engine/all, or choose --gateway triton for ONNX."
        exit 1
    fi
    assemble_model_repo "$EXPORTED_DIR" "$VARIANT" "$MODEL_REPO_DIR" "$ENGINE_MODE" "$MODEL_VERSION"
    validate_model_repo "$MODEL_REPO_DIR" "$MODEL_VERSION"
}

ensure_model_repo() {
    resolve_variant_if_needed
    if $FORCE_PREPARE; then
        prepare_model_repo
        return 0
    fi
    if [[ ! -d "$MODEL_REPO_DIR" ]] || [[ -z "$(ls -A "$MODEL_REPO_DIR" 2>/dev/null)" ]]; then
        log_info "Shared model_repository missing, assembling it first"
        prepare_model_repo
        return 0
    fi

    local repo_info
    repo_info=$(PYTHONPATH="$REPO_ROOT" python3 - "$MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path
from engine.config import resolve_model_package_paths

p = resolve_model_package_paths(sys.argv[1])
variant = ""
if Path(p.manifest_path).is_file():
    with open(p.manifest_path, encoding="utf-8") as f:
        variant = str(json.load(f).get("variant") or "")
print("\t".join([
    p.manifest_path,
    p.runtime_artifact_path,
    p.weights_dir,
    p.tokenizer_dir,
    variant,
    p.engine_mode,
]))
PY
)
    IFS=$'\t' read -r manifest artifact weights_dir tokenizer_dir repo_variant repo_mode <<< "$repo_info"
    if [[ ! -f "$manifest" || ! -f "$artifact" || ! -d "$weights_dir" || ! -d "$tokenizer_dir" ]]; then
        log_warn "Shared model_repository is incomplete for engine_mode=$ENGINE_MODE, re-assembling"
        prepare_model_repo
        return 0
    fi
    if model_package_engine_payload_stale "$REPO_ROOT" "$MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION"; then
        log_warn "Shared model_repository engine payload is missing or stale, re-assembling"
        prepare_model_repo
        return 0
    fi
    if model_package_resources_stale "$REPO_ROOT" "$MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION"; then
        log_warn "Shared model_repository resources are missing or stale, re-assembling"
        prepare_model_repo
        return 0
    fi
    local runtime_dir
    runtime_dir="$(dirname "$artifact")"
    local src_runtime_asset dst_runtime_asset
    local runtime_asset_candidates=()
    if [[ "$ENGINE_MODE" == "trt" ]]; then
        runtime_asset_candidates=(
            "$EXPORTED_DIR/$VARIANT/speaker_encoder.engine"
            "$EXPORTED_DIR/$VARIANT/speech_tokenizer_codec_fused.engine"
        )
    else
        runtime_asset_candidates=(
            "$EXPORTED_DIR/$VARIANT/speaker_encoder.onnx"
            "$EXPORTED_DIR/$VARIANT/speech_tokenizer_codec_fused.onnx"
        )
    fi
    for src_runtime_asset in "${runtime_asset_candidates[@]}"; do
        [[ -f "$src_runtime_asset" ]] || continue
        dst_runtime_asset="$runtime_dir/$(basename "$src_runtime_asset")"
        if [[ ! -f "$dst_runtime_asset" || "$src_runtime_asset" -nt "$dst_runtime_asset" ]]; then
            log_warn "Shared model_repository runtime support asset is missing or stale: $(basename "$src_runtime_asset")"
            prepare_model_repo
            return 0
        fi
    done

    if [[ "$repo_variant" != "$VARIANT" || "$repo_mode" != "$ENGINE_MODE" ]]; then
        log_warn "Shared model_repository is for variant=${repo_variant:-unknown}, engine_mode=${repo_mode:-unknown}; requested variant=$VARIANT, engine_mode=$ENGINE_MODE"
        prepare_model_repo
    fi
}

cmd_build() {
    require_docker_compose_if_needed
    resolve_variant_if_needed
    export_compose_env
    resolve_compose_image_defaults
    case "$GATEWAY" in
        engine)
            compose_cmd build engine
            verify_compose_engine_image
            ;;
        triton) compose_cmd build triton ;;
        all)
            compose_cmd build engine triton
            verify_compose_engine_image
            ;;
    esac
}

cmd_prepare() {
    resolve_variant_if_needed
    if $DRY_RUN; then
        local model_version
        model_version=$(resolve_model_version "$MODEL_VERSION") || exit 1
        log_info "[DRY RUN] Would assemble shared model_repository:"
        log_info "  Gateway:     $GATEWAY"
        log_info "  Variant:     $VARIANT"
        log_info "  Engine mode: $ENGINE_MODE"
        log_info "  Source:      $EXPORTED_DIR/$VARIANT"
        log_info "  Target:      $MODEL_REPO_DIR"
        log_info "  Version:     $model_version"
        return 0
    fi
    prepare_model_repo
}

# Run `compose up` for the given services, honoring --build.
_compose_up_exec() {
    if $BUILD_BEFORE_UP; then
        compose_cmd up --build -d "$@"
    else
        compose_cmd up -d "$@"
    fi
}

cmd_up() {
    require_docker_compose_if_needed

    if $DRY_RUN; then
        resolve_variant_if_needed
        export_compose_env
        resolve_compose_image_defaults
        log_compose_runtime_summary
        case "$GATEWAY" in
            engine) _compose_up_exec engine ;;
            triton) _compose_up_exec triton ;;
            all)    _compose_up_exec engine triton ;;
        esac
        return 0
    fi

    resolve_variant_if_needed
    export_compose_env

    case "$GATEWAY" in
        engine)
            ensure_model_repo
            export_compose_env
            resolve_compose_image_defaults
            compose_preflight_service engine
            _compose_up_exec engine
            if ! $NO_HEALTH_CHECK; then
                compose_wait_engine_ready
            fi
            ;;
        triton)
            ensure_model_repo
            export_compose_env
            resolve_compose_image_defaults
            compose_preflight_service triton
            _compose_up_exec triton
            if ! $NO_HEALTH_CHECK; then
                compose_wait_triton_ready
            fi
            ;;
        all)
            ensure_model_repo
            export_compose_env
            resolve_compose_image_defaults
            compose_preflight_service engine
            compose_preflight_service triton
            _compose_up_exec engine triton
            if ! $NO_HEALTH_CHECK; then
                compose_wait_engine_ready
                compose_wait_triton_ready
            fi
            ;;
    esac
}

cmd_watch() {
    require_docker_compose_if_needed

    if $USE_DEV_OVERLAY; then
        log_error "'watch' cannot be combined with --dev; use either bind mounts or compose watch"
        exit 1
    fi

    case "$GATEWAY" in
        engine|triton|all) ensure_model_repo ;;
    esac

    export_compose_env
    resolve_compose_image_defaults

    local args=(watch)
    $WATCH_NO_UP && args+=(--no-up)
    $WATCH_QUIET && args+=(--quiet)
    $WATCH_NO_PRUNE && args+=(--prune=false)

    case "$GATEWAY" in
        engine) args+=(engine) ;;
        triton) args+=(triton) ;;
        all) args+=(engine triton) ;;
    esac

    compose_cmd "${args[@]}"
}

cmd_down() {
    require_docker_compose_if_needed
    export_compose_env
    case "$GATEWAY" in
        engine)
            compose_cmd stop engine
            compose_cmd rm -sf engine
            ;;
        triton)
            compose_cmd stop triton
            compose_cmd rm -sf triton
            ;;
        all)
            compose_cmd down --remove-orphans
            ;;
    esac
}

cmd_logs() {
    require_docker_compose_if_needed
    export_compose_env
    local args=(logs)
    $FOLLOW && args+=(-f)
    case "$GATEWAY" in
        engine) args+=(engine) ;;
        triton) args+=(triton) ;;
        all) ;;
    esac
    compose_cmd "${args[@]}"
}

cmd_ps() {
    require_docker_compose_if_needed
    export_compose_env
    compose_cmd ps
}

cmd_config() {
    require_docker_compose_if_needed
    resolve_variant_if_needed
    export_compose_env
    resolve_compose_image_defaults
    compose_cmd config
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gateway) GATEWAY="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --engine-mode) ENGINE_MODE="$2"; shift 2 ;;
        --repo-dir) MODEL_REPO_DIR="$2"; shift 2 ;;
        --model-version) MODEL_VERSION="$2"; shift 2 ;;
        --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
        --container) CONTAINER_OVERRIDE="$2"; shift 2 ;;
        --port) ENGINE_PORT="$2"; shift 2 ;;
        --ws-port) ENGINE_WEBSOCKET="$2"; shift 2 ;;
        --health-port) ENGINE_HEALTH="$2"; shift 2 ;;
        --grpc-port) TRITON_GRPC="$2"; shift 2 ;;
        --http-port) TRITON_HTTP="$2"; shift 2 ;;
        --metrics-port) TRITON_METRICS="$2"; shift 2 ;;
        --device) ENGINE_DEVICE="$2"; TRITON_GPU_DEVICE="$2"; shift 2 ;;
        --triton-device) TRITON_GPU_DEVICE="$2"; shift 2 ;;
        --max-batch|--runtime-max-batch-size|--runtime-max-batch)
            ENGINE_MAX_BATCH="$2"; TRITON_MAX_BATCH="$2"; shift 2 ;;
        --max-sessions) ENGINE_MAX_SESSIONS="$2"; shift 2 ;;
        --max-seq-len|--runtime-max-seq-len|--runtime-max-seq)
            ENGINE_MAX_SEQ_LEN="$2"; TRITON_MAX_SEQ_LEN="$2"; shift 2 ;;
        --build) BUILD_BEFORE_UP=true; shift ;;
        --dev) USE_DEV_OVERLAY=true; shift ;;
        --prepare) FORCE_PREPARE=true; shift ;;
        --no-up) WATCH_NO_UP=true; shift ;;
        --quiet) WATCH_QUIET=true; shift ;;
        --no-prune) WATCH_NO_PRUNE=true; shift ;;
        --no-health-check) NO_HEALTH_CHECK=true; shift ;;
        --follow) FOLLOW=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
        ;;
    esac
done

if [[ -n "$IMAGE_OVERRIDE" ]]; then
    case "$GATEWAY" in
        engine) export ENGINE_IMAGE="$IMAGE_OVERRIDE" ;;
        triton) export TRITON_IMAGE="$IMAGE_OVERRIDE" ;;
        *)
            log_error "--image requires --gateway engine or --gateway triton"
            exit 1
            ;;
    esac
fi

if [[ -n "$CONTAINER_OVERRIDE" ]]; then
    case "$GATEWAY" in
        engine) export ENGINE_CONTAINER_NAME="$CONTAINER_OVERRIDE" ;;
        triton) export TRITON_CONTAINER_NAME="$CONTAINER_OVERRIDE" ;;
        *)
            log_error "--container requires --gateway engine or --gateway triton"
            exit 1
            ;;
    esac
fi

case "$COMMAND" in
    build) cmd_build ;;
    prepare) cmd_prepare ;;
    up) cmd_up ;;
    watch) cmd_watch ;;
    down) cmd_down ;;
    logs) cmd_logs ;;
    ps) cmd_ps ;;
    config) cmd_config ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
