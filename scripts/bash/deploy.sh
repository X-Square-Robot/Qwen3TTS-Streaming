#!/bin/bash
# ===========================================================================
#  deploy.sh — Phase C: Deploy TTS service (standalone engine or Triton)
#
#  Unified entry point for packaging or deploying the TTS service. Supports three gateway modes:
#
#  A) Standalone gRPC Server (--gateway standalone)
#     Runs `python -m engine.server` on the host (see ENGINE_PYTHON / conda).
#     Best for: development, single-model production, low-latency.
#
#  B) Triton Backend (--gateway triton)
#     Assembles model_repository + starts Triton via docker compose.
#     Best for: multi-model serving, K8s, enterprise infrastructure.
#
#  C) Engine Docker (--gateway engine-docker)
#     Builds (if missing) Dockerfile.engine and runs engine.server via docker compose;
#     mounts the shared model_repository read-only. No host PyTorch required.
#     Best for: portable deploy, matching TRT base with Phase B.
#
#  Usage:
#    bash scripts/bash/deploy.sh package --gateway engine-docker    # assemble model repo + build engine image, do not run
#    bash scripts/bash/deploy.sh run                               # standalone (default)
#    bash scripts/bash/deploy.sh run --gateway triton               # Triton mode
#    bash scripts/bash/deploy.sh run --gateway engine-docker      # Engine image + container
#    bash scripts/bash/deploy.sh run --foreground                   # don't daemonize
#    bash scripts/bash/deploy.sh stop                               # stop service
#    bash scripts/bash/deploy.sh status                             # show status
#
#  Triton-specific commands (forwarded to build_triton.sh):
#    bash scripts/bash/deploy.sh assemble [--engine-mode onnx|trt]
#    bash scripts/bash/deploy.sh pull
#    bash scripts/bash/deploy.sh build-image
#    bash scripts/bash/deploy.sh build [--tag <image:tag>]
#
#  Environment variables:
#    GATEWAY_MODE            Override gateway: standalone | triton | engine-docker
#    MODEL_REPO_DIR          Shared model_repository path (default: workspace/model_repository)
#    ENGINE_GRPC_PORT        Standalone gRPC port (default: 50051)
#    ENGINE_WEBSOCKET_PORT   Standalone WebSocket port (default: 50052)
#    RUNTIME_GPU_DEVICE      Runtime GPU device (auto | N | cuda:N; default: auto)
#    RUNTIME_MAX_BATCH_SIZE  Runtime scheduler batch limit (default: manifest profile)
#    RUNTIME_MAX_SEQ_LEN     Runtime scheduler seq limit (default: manifest profile)
#    ENGINE_PYTHON           Python binary for standalone engine (default: conda env qwen3-tts, else PATH)
#    QWEN3_TTS_ENV_NAME      Conda env name for auto-resolve (default: qwen3-tts)
#    TRITON_GRPC_PORT        Triton gRPC port (default: 8001)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../.." && pwd))"
source "${SCRIPT_DIR}/tools.sh"

# ── Defaults ──
EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
MODEL_REPO_DIR="${MODEL_REPO_DIR:-${REPO_ROOT}/workspace/model_repository}"
GATEWAY_MODE="${GATEWAY_MODE:-standalone}"
VARIANT=""
MODEL_VERSION="${MODEL_VERSION:-${ENGINE_MODEL_VERSION:-1}}"
DRY_RUN=false
FORCE_IMAGE_BUILD=false

# Standalone options
ENGINE_PORT="${ENGINE_GRPC_PORT:-50051}"
ENGINE_WS_PORT="${ENGINE_WEBSOCKET_PORT:-50052}"
GPU_DEVICE="${RUNTIME_GPU_DEVICE:-auto}"
MAX_BATCH="${RUNTIME_MAX_BATCH_SIZE:-}"
MAX_SESSIONS=128
MAX_SEQ_LEN="${RUNTIME_MAX_SEQ_LEN:-}"
FOREGROUND=false

# Triton forwarding
TRITON_ARGS=()

# engine-docker: use --engine-image for an explicit image.  Without it, Phase C
# derives qwen3-engine:<tag> from the model manifest's Phase B builder_image.
ENGINE_IMAGE_EXPLICIT=false

# ── Help ──

usage() {
    cat << 'EOF'
Usage: deploy.sh <command> [options]

Commands:
  package                Assemble deployment artifacts; optionally build image, do not start service
  run                    Start the TTS service
  stop                   Stop the TTS service
  status                 Show service status

  Triton-specific (forwarded to build_triton.sh):
    assemble             Assemble Triton model_repository
    pull                 Pull NGC Triton container image
    build-image          Build deploy image
    build                Build self-contained Docker image

Options:
  --gateway <mode>       Gateway: standalone | triton | engine-docker (default: standalone)
  --engine-image <tag>   Image tag for engine-docker (default: Phase B NGC tag)
  --variant <name>       Model variant (default: auto-discover)
  --model-version <N>    Triton model version directory (default: 1)
  --build, --rebuild-image
                          Rebuild runtime image from current code before use
  --dry-run              Show what would be done

  Standalone options:
    --port <N>           gRPC port (default: 50051)
    --ws-port <N>        WebSocket port (default: 50052)
    --device <N|auto>    Runtime GPU device (default: auto)
    --max-batch <N>      Runtime max batch size (default: manifest profile, else 128)
    --max-seq-len <N>    Runtime max sequence length (default: manifest profile, else 512)
    --max-sessions <N>   Max concurrent sessions (default: 128)
    --foreground         Run in foreground (don't daemonize)

  Triton options (forwarded to build_triton.sh):
    --engine-mode <mode> onnx | trt (default: trt)
    --image <uri>        Override NGC container image
    --container <name>   Container name
    --no-health-check    Skip health check

Examples:
  deploy.sh run                                  # standalone, auto-discover variant
  deploy.sh run --variant custom-1.7b            # standalone, specific variant
  deploy.sh package --gateway engine-docker --variant custom-1.7b
  deploy.sh run --gateway triton                 # Triton mode
  deploy.sh run --gateway engine-docker          # Engine Dockerfile + container
  deploy.sh run --foreground                     # standalone, foreground
  deploy.sh run --model-version 2                # use tts_orchestrator/2 package
  deploy.sh stop                                 # stop whatever is running
  deploy.sh status                               # show status for both modes
  deploy.sh assemble --engine-mode trt           # Triton: assemble model repo
EOF
}

# ── Discover first available variant ──
discover_first_variant() {
    for vdir in "$EXPORTED_DIR"/*/; do
        local vname
        vname=$(basename "$vdir")
        [[ "$vname" == "tokenizer" ]] && continue
        if [ -d "$vdir/weights" ]; then
            echo "$vname"
            return 0
        fi
    done
    return 1
}

resolve_variant() {
    if [ -n "$VARIANT" ]; then
        if [ ! -d "$EXPORTED_DIR/$VARIANT" ]; then
            log_error "Variant not found: $EXPORTED_DIR/$VARIANT"
            exit 1
        fi
        return 0
    fi

    VARIANT=$(discover_first_variant) \
        || { log_error "No exported variants found in $EXPORTED_DIR/"; exit 1; }
    log_info "Auto-discovered variant: $VARIANT"
}

# ── Argument parsing ──
COMMAND=""

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gateway)        GATEWAY_MODE="$2"; shift 2 ;;
        --variant)        VARIANT="$2"; shift 2 ;;
        --model-version)  MODEL_VERSION="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --build|--rebuild-image) FORCE_IMAGE_BUILD=true; shift ;;
        --engine-image)   ENGINE_IMAGE="$2"; ENGINE_IMAGE_EXPLICIT=true; shift 2 ;;
        --help|-h)        usage; exit 0 ;;

        # Standalone options
        --port)           ENGINE_PORT="$2"; shift 2 ;;
        --ws-port)        ENGINE_WS_PORT="$2"; shift 2 ;;
        --device|--runtime-device) GPU_DEVICE="$2"; shift 2 ;;
        --max-batch|--runtime-max-batch-size|--runtime-max-batch) MAX_BATCH="$2"; shift 2 ;;
        --max-seq-len|--runtime-max-seq-len|--runtime-max-seq) MAX_SEQ_LEN="$2"; shift 2 ;;
        --max-sessions)   MAX_SESSIONS="$2"; shift 2 ;;
        --foreground)     FOREGROUND=true; shift ;;

        # Everything else is forwarded to build_triton.sh
        *)
            TRITON_ARGS+=("$1")
            # Options with arguments: consume next arg too
            case "$1" in
                --engine-mode|--image|--repo-dir|--container|--tag)
                    if [[ $# -gt 1 ]]; then
                        TRITON_ARGS+=("$2")
                        shift
                    fi
                    ;;
            esac
            shift
            ;;
    esac
done

MODEL_VERSION=$(resolve_model_version "$MODEL_VERSION") || exit 1
export MODEL_VERSION
export ENGINE_MODEL_VERSION="$MODEL_VERSION"

# Validate gateway mode
case "$GATEWAY_MODE" in
    standalone|triton|engine-docker) ;;
    *)
        log_error "Unknown gateway mode: $GATEWAY_MODE (expected: standalone | triton | engine-docker)"
        exit 1
        ;;
esac

# ── Runtime defaults ──

_manifest_profile_value() {
    local key="$1"
    local manifest="$EXPORTED_DIR/$VARIANT/triton_manifest.json"
    if [ ! -f "$manifest" ]; then
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

resolve_runtime_controls() {
    local raw_device="$GPU_DEVICE"
    GPU_DEVICE=$(resolve_gpu_device_index "$GPU_DEVICE") || exit 1
    if [ "$raw_device" = "auto" ] || [ -z "$raw_device" ]; then
        log_gpu_selection "Runtime" "$GPU_DEVICE"
    fi

    if [ -z "$MAX_BATCH" ]; then
        MAX_BATCH=$(_manifest_profile_value max_batch_size)
        MAX_BATCH="${MAX_BATCH:-128}"
    fi
    if [ -z "$MAX_SEQ_LEN" ]; then
        MAX_SEQ_LEN=$(_manifest_profile_value max_seq_len)
        MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
    fi
    if ! [[ "$MAX_BATCH" =~ ^[1-9][0-9]*$ ]]; then
        log_error "Runtime max batch must be a positive integer, got: $MAX_BATCH"
        exit 1
    fi
    if ! [[ "$MAX_SEQ_LEN" =~ ^[1-9][0-9]*$ ]]; then
        log_error "Runtime max seq len must be a positive integer, got: $MAX_SEQ_LEN"
        exit 1
    fi

    export RUNTIME_GPU_DEVICE="$GPU_DEVICE"
    export RUNTIME_MAX_BATCH_SIZE="$MAX_BATCH"
    export RUNTIME_MAX_SEQ_LEN="$MAX_SEQ_LEN"
}

check_engine_artifact_ready() {
    local _amf="${REPO_ROOT}/workspace/exported/artifact_manifest.json"
    local _exp="${REPO_ROOT}/workspace/exported"
    local _tp="${TARGET_PROFILE:-${REPO_ROOT}/workspace/target_profile.json}"
    local _tp_arg=""
    [ -f "$_tp" ] && _tp_arg="$_tp"
    if [ ! -f "$_amf" ]; then
        if [ "${ALLOW_FINGERPRINT_MISMATCH:-}" = "1" ]; then
            log_warn "Missing artifact_manifest.json — ignored (ALLOW_FINGERPRINT_MISMATCH=1)"
            return 0
        fi
        log_error "未发现 artifact_manifest.json: $_amf"
        log_error "  Strict 模式要求 package/run 前必须有 build 写入的 artifact_manifest。"
        log_error "  请先执行:"
        log_error "    bash scripts/bash/autorun.sh build"
        log_error "    或 bash scripts/bash/autorun.sh import-artifact <bundle>"
        log_error "  调试可临时使用: ALLOW_FINGERPRINT_MISMATCH=1"
        return 1
    fi
    if engine_fingerprint_check "$_amf" "$_exp" "$_tp_arg"; then
        return 0
    fi
    if [ "${ALLOW_FINGERPRINT_MISMATCH:-}" = "1" ]; then
        log_warn "Fingerprint mismatch — ignored (ALLOW_FINGERPRINT_MISMATCH=1)"
        return 0
    fi
    log_error "  请重新 build 或 import-artifact 以更新 engines。"
    return 1
}

EXPECTED_ENGINE_RELEASE=""
EXPECTED_ENGINE_TORCH_CUDA_TAG=""

resolve_standalone_tensorrt_version() {
    if [ -n "${STANDALONE_ENGINE_TENSORRT_PIP_VERSION:-}" ]; then
        printf '%s\n' "$STANDALONE_ENGINE_TENSORRT_PIP_VERSION"
        return 0
    fi

    local manifest_tag="${NGC_TAG:-}"
    if [ -z "$manifest_tag" ]; then
        manifest_tag=$(resolve_manifest_ngc_tag "$EXPORTED_DIR/$VARIANT/triton_manifest.json" 2>/dev/null || true)
    fi
    if [ -z "$manifest_tag" ]; then
        manifest_tag=$(resolve_manifest_ngc_tag "$MODEL_REPO_DIR" "$MODEL_VERSION" 2>/dev/null || true)
    fi
    if [ -n "$manifest_tag" ]; then
        local trt_version
        trt_version=$(resolve_ngc_tag_tensorrt_version "$manifest_tag" 2>/dev/null || true)
        if [ -n "$trt_version" ]; then
            printf '%s\n' "$trt_version"
            return 0
        fi
    fi

    printf '%s\n' "10.15.1.29"
}

resolve_engine_docker_image() {
    local img="${ENGINE_IMAGE:-qwen3-engine:26.02}"
    local expected_release=""
    local manifest_tag=""
    if ! $ENGINE_IMAGE_EXPLICIT && [[ "$img" == qwen3-engine:* ]]; then
        if [[ "$img" =~ ^qwen3-engine:([0-9]+\.[0-9]+)$ ]]; then
            expected_release="${BASH_REMATCH[1]}"
        fi
        if [ -z "$expected_release" ] || [ "$expected_release" = "26.02" ]; then
            manifest_tag="${NGC_TAG:-}"
            local tag_source="manifest"
            if [ -n "$manifest_tag" ]; then
                tag_source="ngc_tag"
            fi
            if [ -z "$manifest_tag" ]; then
                manifest_tag=$(resolve_manifest_ngc_tag "$EXPORTED_DIR/$VARIANT/triton_manifest.json" 2>/dev/null || true)
            fi
            if [ -z "$manifest_tag" ]; then
                manifest_tag=$(resolve_manifest_ngc_tag "$MODEL_REPO_DIR" "$MODEL_VERSION" 2>/dev/null || true)
            fi
            if [ -n "$manifest_tag" ]; then
                img="qwen3-engine:${manifest_tag}"
                expected_release="$manifest_tag"
                if [ "$tag_source" = "manifest" ]; then
                    log_info "Using engine Docker image from Phase B manifest: $img"
                elif [ "$tag_source" = "ngc_tag" ]; then
                    log_info "Using engine Docker image from NGC_TAG: $img"
                fi
            else
                log_error "Cannot determine engine Docker image from Phase B manifest."
                log_error "Run Phase B with a target profile first, or pass --engine-image explicitly."
                log_error "Typical flow: probe_target.sh on production GPU -> build_engines.sh make-bundle/remote-build -> import-artifact."
                return 1
            fi
        fi
    fi

    if [ -n "$expected_release" ] && [ -z "${ENGINE_BASE_IMAGE:-}" ]; then
        export ENGINE_BASE_IMAGE="nvcr.io/nvidia/tensorrt:${expected_release}-py3"
    fi
    if [ -n "$expected_release" ] && [ -z "${ENGINE_PYTORCH_CUDA_TAG:-${PYTORCH_CUDA_TAG:-}}" ]; then
        local torch_cuda_tag
        torch_cuda_tag=$(resolve_ngc_torch_index_tag "$expected_release" 2>/dev/null || true)
        if [ -n "$torch_cuda_tag" ]; then
            export ENGINE_PYTORCH_CUDA_TAG="$torch_cuda_tag"
        fi
    fi

    EXPECTED_ENGINE_RELEASE="$expected_release"
    EXPECTED_ENGINE_TORCH_CUDA_TAG="${ENGINE_PYTORCH_CUDA_TAG:-${PYTORCH_CUDA_TAG:-}}"
    echo "$img"
}

ensure_engine_docker_image_current() {
    local img="$1"
    local need_build=false
    if $FORCE_IMAGE_BUILD; then
        log_info "Rebuilding engine image from current code (--build/--rebuild-image)."
        need_build=true
    elif ! docker image inspect "$img" &>/dev/null; then
        need_build=true
    elif ! engine_docker_image_has_app "$img"; then
        log_warn "镜像 $img 存在但未包含 /app 下的 engine 包（常见于把 TensorRT 基础镜像误打成同名 tag）。"
        log_info "将按 Dockerfile.engine 重新构建..."
        need_build=true
    elif ! engine_docker_image_supports_model_package_engine "$img"; then
        log_warn "镜像 $img 的 engine 代码或启动脚本较旧，无法优先使用模型包内的 engine/。"
        log_info "将按 Dockerfile.engine 重新构建..."
        need_build=true
    elif [ -n "$EXPECTED_ENGINE_RELEASE" ] && ! engine_docker_image_matches_release "$img" "$EXPECTED_ENGINE_RELEASE"; then
        local actual_release
        actual_release=$(engine_docker_image_tensorrt_release "$img" || true)
        log_warn "镜像 $img 的 TensorRT 版本是 ${actual_release:-unknown}，但 Phase B manifest 对应 $EXPECTED_ENGINE_RELEASE。"
        log_info "将按 Dockerfile.engine 使用 ENGINE_BASE_IMAGE=$ENGINE_BASE_IMAGE 重新构建..."
        need_build=true
    elif [ -n "$EXPECTED_ENGINE_TORCH_CUDA_TAG" ] && ! engine_docker_image_matches_torch_cuda "$img" "$EXPECTED_ENGINE_TORCH_CUDA_TAG"; then
        local actual_torch_cuda_tag
        actual_torch_cuda_tag=$(engine_docker_image_torch_cuda_tag "$img" || true)
        log_warn "镜像 $img 的 PyTorch CUDA wheel 是 ${actual_torch_cuda_tag:-unknown}，但目标应为 $EXPECTED_ENGINE_TORCH_CUDA_TAG。"
        log_info "将按 Dockerfile.engine 使用 ENGINE_PYTORCH_CUDA_TAG=$EXPECTED_ENGINE_TORCH_CUDA_TAG 重新构建..."
        need_build=true
    fi
    if $need_build; then
        bash "${SCRIPT_DIR}/compose.sh" build --gateway engine --image "$img" || return 1
    fi
}

# ── Commands ──

cmd_package() {
    resolve_variant
    $DRY_RUN || check_engine_artifact_ready || return 1

    case "$GATEWAY_MODE" in
        standalone)
            local prepare_args=(
                prepare
                --gateway engine
                --variant "$VARIANT"
                --engine-mode trt
                --repo-dir "$MODEL_REPO_DIR"
                --model-version "$MODEL_VERSION"
            )
            $DRY_RUN && prepare_args+=(--dry-run)
            MODEL_REPO_DIR="$MODEL_REPO_DIR" bash "${SCRIPT_DIR}/compose.sh" "${prepare_args[@]}"
            log_info "Model package ready: $MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION"
            ;;
        engine-docker)
            local prepare_args=(
                prepare
                --gateway engine
                --variant "$VARIANT"
                --engine-mode trt
                --repo-dir "$MODEL_REPO_DIR"
                --model-version "$MODEL_VERSION"
            )
            $DRY_RUN && prepare_args+=(--dry-run)
            MODEL_REPO_DIR="$MODEL_REPO_DIR" bash "${SCRIPT_DIR}/compose.sh" "${prepare_args[@]}"
            local img
            img=$(resolve_engine_docker_image) || return 1
            if ! $DRY_RUN; then
                # Packaging should always refresh image code. Docker cache keeps
                # dependency layers, but COPY engine/ and scripts/compose reflect
                # the current checkout.
                FORCE_IMAGE_BUILD=true
                ensure_engine_docker_image_current "$img" || return 1
            else
                log_info "[DRY RUN] Would build engine Docker image from current code: $img"
            fi
            log_step "Engine package ready"
            log_info "  Engine image:  $img"
            log_info "  Model package: $MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION"
            log_info "  Production run mounts the model package under /models."
            ;;
        triton)
            local build_args=(
                --variant "$VARIANT"
                --model-version "$MODEL_VERSION"
                --engine-mode trt
            )
            $DRY_RUN && build_args+=(--dry-run)
            build_args+=("${TRITON_ARGS[@]}")
            bash "${SCRIPT_DIR}/build_triton.sh" build "${build_args[@]}"
            ;;
    esac
}

cmd_run() {
    resolve_variant
    resolve_runtime_controls

    # Strict pre-flight: refuse to deploy engines without a valid
    # artifact_manifest.json from the unified build pipeline.  Mirrors the
    # check in autorun.sh::run_phase_c so direct `deploy.sh run` callers
    # are protected too.  Bypass via ALLOW_FINGERPRINT_MISMATCH=1.
    $DRY_RUN || check_engine_artifact_ready || return 1

    case "$GATEWAY_MODE" in
        standalone)
            cmd_run_standalone
            ;;
        triton)
            cmd_run_triton
            ;;
        engine-docker)
            cmd_run_engine_docker
            ;;
    esac
}

cmd_run_standalone() {
    if $DRY_RUN; then
        log_info "[DRY RUN] Would start standalone engine:"
        log_info "  Variant:    $VARIANT"
        log_info "  Model repo: $MODEL_REPO_DIR"
        log_info "  Package:    $MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION"
        log_info "  Port:       $ENGINE_PORT"
        log_info "  WS Port:    $ENGINE_WS_PORT"
        log_info "  Device:     $GPU_DEVICE"
        log_info "  Max Batch:  $MAX_BATCH"
        log_info "  Max Seq:    ${MAX_SEQ_LEN:-auto}"
        log_info "  Max Sess:   $MAX_SESSIONS"
        log_info "  Foreground: $FOREGROUND"
        return 0
    fi

    MODEL_REPO_DIR="$MODEL_REPO_DIR" bash "${SCRIPT_DIR}/compose.sh" \
        prepare \
        --gateway engine \
        --variant "$VARIANT" \
        --engine-mode trt \
        --repo-dir "$MODEL_REPO_DIR" \
        --model-version "$MODEL_VERSION"

    ENGINE_MODEL_PACKAGE_DIR="$MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION"

    local pybin
    pybin=$(resolve_engine_python_bin "$REPO_ROOT") || exit 1

    local standalone_trt_version
    standalone_trt_version=$(resolve_standalone_tensorrt_version)
    export STANDALONE_ENGINE_TENSORRT_PIP_VERSION="$standalone_trt_version"
    install_tensorrt_for_python "$pybin" "$standalone_trt_version" || exit 1

    local start_args=(
        --port "$ENGINE_PORT"
        --ws-port "$ENGINE_WS_PORT"
        --device "$GPU_DEVICE"
        --max-batch "$MAX_BATCH"
        --max-sessions "$MAX_SESSIONS"
    )
    if [ -n "$MAX_SEQ_LEN" ]; then
        start_args+=(--max-seq-len "$MAX_SEQ_LEN")
    fi
    if $FOREGROUND; then
        start_args+=(--foreground)
    fi

    engine_start "$REPO_ROOT" "$VARIANT" "${start_args[@]}" || exit 1

    if ! $FOREGROUND; then
        echo ""
        if engine_health_check "$ENGINE_PORT" 30; then
            echo ""
            log_step "Standalone TTS Engine Running"
            log_info "  gRPC endpoint:  localhost:${ENGINE_PORT}"
            log_info "  WebSocket:      ws://localhost:${ENGINE_WS_PORT}/v1/ws"
            log_info "  Variant:        $VARIANT"
            log_info "  Model package:  $MODEL_REPO_DIR/tts_orchestrator/$MODEL_VERSION"
            log_info "  Log file:       $(engine_log_file "$REPO_ROOT")"
            echo ""
            log_info "Stop: bash scripts/bash/deploy.sh stop"
        else
            log_warn "Engine started but port not yet reachable"
            log_info "Check logs: tail -f $(engine_log_file "$REPO_ROOT")"
        fi
    fi
}

cmd_run_triton() {
    local has_image_override=false
    local arg
    for arg in "${TRITON_ARGS[@]}"; do
        if [ "$arg" = "--image" ]; then
            has_image_override=true
            break
        fi
    done

    local compose_args=(
        up
        --gateway triton
        --prepare
        --device "$GPU_DEVICE"
        --max-batch "$MAX_BATCH"
        --max-seq-len "$MAX_SEQ_LEN"
        --model-version "$MODEL_VERSION"
    )
    if ! $has_image_override; then
        local triton_image="${TRITON_IMAGE:-}"
        if [ -z "$triton_image" ]; then
            if [ -n "${NGC_TAG:-}" ]; then
                triton_image=$(resolve_triton_deploy_image) || exit 1
            fi
        fi
        if [ -z "$triton_image" ]; then
            local manifest_tag
            manifest_tag=$(resolve_manifest_ngc_tag "$EXPORTED_DIR/$VARIANT/triton_manifest.json" 2>/dev/null || true)
            if [ -z "$manifest_tag" ]; then
                manifest_tag=$(resolve_manifest_ngc_tag "$MODEL_REPO_DIR" "$MODEL_VERSION" 2>/dev/null || true)
            fi
            if [ -n "$manifest_tag" ]; then
                triton_image="qwen3-tts-triton:${manifest_tag}"
                log_info "Using Triton image from Phase B manifest: $triton_image"
            fi
        fi
        if [ -z "$triton_image" ]; then
            triton_image=$(resolve_triton_deploy_image) || exit 1
        fi
        compose_args+=(--image "$triton_image")
    fi
    [ -n "$VARIANT" ] && compose_args+=(--variant "$VARIANT")
    $DRY_RUN && compose_args+=(--dry-run)
    compose_args+=("${TRITON_ARGS[@]}")

    bash "${SCRIPT_DIR}/compose.sh" "${compose_args[@]}"
}

cmd_run_engine_docker() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker is required for --gateway engine-docker"
        exit 1
    fi

    local img
    img=$(resolve_engine_docker_image) || exit 1

    if $DRY_RUN; then
        log_info "[DRY RUN] Would start engine Docker container (Dockerfile.engine)"
        log_info "  Variant:     $VARIANT"
        log_info "  Image:       $img"
        log_info "  Port:        $ENGINE_PORT"
        log_info "  WS Port:     $ENGINE_WS_PORT"
        log_info "  Device:      $GPU_DEVICE"
        log_info "  Max batch:   $MAX_BATCH"
        log_info "  Max seq len: ${MAX_SEQ_LEN:-auto}"
        log_info "  Max sess:    $MAX_SESSIONS"
        $FORCE_IMAGE_BUILD && log_info "  Rebuild:     yes (--build/--rebuild-image)"
        return 0
    fi

    ensure_engine_docker_image_current "$img" || exit 1

    local compose_args=(
        up
        --gateway engine
        --prepare
        --variant "$VARIANT"
        --image "$img"
        --port "$ENGINE_PORT"
        --ws-port "$ENGINE_WS_PORT"
        --device "$GPU_DEVICE"
        --max-batch "$MAX_BATCH"
        --max-sessions "$MAX_SESSIONS"
        --model-version "$MODEL_VERSION"
    )
    if [ -n "$MAX_SEQ_LEN" ]; then
        compose_args+=(--max-seq-len "$MAX_SEQ_LEN")
    fi
    $DRY_RUN && compose_args+=(--dry-run)

    bash "${SCRIPT_DIR}/compose.sh" "${compose_args[@]}" || exit 1

    echo ""
    log_step "Engine Docker Running"
    log_info "  gRPC endpoint:  localhost:${ENGINE_PORT}"
    log_info "  WebSocket:      ws://localhost:${ENGINE_WS_PORT}/v1/ws"
    log_info "  Variant:        $VARIANT"
    log_info "  Image:          $img"
    log_info "  Container:      ${ENGINE_CONTAINER_NAME:-qwen3-engine}"
    echo ""
    log_info "Logs: bash scripts/bash/compose.sh logs --gateway engine --follow"
    log_info "Stop: bash scripts/bash/deploy.sh stop"
}

cmd_stop() {
    local stopped=false

    # Stop standalone engine (if running)
    local engine_st
    engine_st=$(engine_status "$REPO_ROOT")
    if [ "$engine_st" != "none" ]; then
        engine_stop "$REPO_ROOT"
        stopped=true
    fi

    # Stop compose-managed Docker services
    if command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
        if docker compose -f "${REPO_ROOT}/compose.yaml" ps -q 2>/dev/null | grep -q .; then
            bash "${SCRIPT_DIR}/compose.sh" down --gateway all
            stopped=true
        fi
    fi

    if ! $stopped; then
        log_info "No running TTS service found"
    fi
}

cmd_status() {
    log_step "TTS Service Status"
    echo ""

    # Standalone engine status
    engine_show_status "$REPO_ROOT"
    echo ""

    if command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
        bash "${SCRIPT_DIR}/compose.sh" ps || true
    else
        log_info "Docker Compose: not available"
    fi
}

# ── Triton-specific commands: forward to build_triton.sh ──

cmd_forward_triton() {
    local subcmd="$1"
    local forward_args=()
    [ -n "$VARIANT" ] && forward_args+=(--variant "$VARIANT")
    forward_args+=(--model-version "$MODEL_VERSION")
    $DRY_RUN && forward_args+=(--dry-run)
    forward_args+=("${TRITON_ARGS[@]}")

    bash "${SCRIPT_DIR}/build_triton.sh" "$subcmd" "${forward_args[@]}"
}

# ── Main dispatch ──

case "$COMMAND" in
    package)     cmd_package ;;
    run)         cmd_run ;;
    stop)        cmd_stop ;;
    status)      cmd_status ;;
    assemble)    cmd_forward_triton "assemble" ;;
    pull)        cmd_forward_triton "pull" ;;
    build-image) cmd_forward_triton "build-image" ;;
    build)       cmd_forward_triton "build" ;;
    -h|--help)   usage ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
