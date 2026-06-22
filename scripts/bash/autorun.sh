#!/bin/bash
# ===========================================================================
#  autorun.sh — Intelligent launcher for Qwen3-TTS Triton pipeline
#
#  Smart entry point that orchestrates the build/deploy phases:
#    Phase A (setup):   Environment + model export       → setup_env.sh
#    Phase B (build):   TensorRT engine compilation       → build_engines.sh
#    Phase C (package): Model/image package assembly      → deploy.sh package
#    Phase C (run):     Current-machine service startup   → deploy.sh run
#
#  Usage:
#    bash scripts/bash/autorun.sh                    # interactive mode
#    bash scripts/bash/autorun.sh base-1.7b          # full pipeline for variant
#    bash scripts/bash/autorun.sh all [options]       # full pipeline (explicit)
#    bash scripts/bash/autorun.sh setup [options]     # Phase A only
#    bash scripts/bash/autorun.sh build [options]     # Phase B only
#    bash scripts/bash/autorun.sh package [options]   # Phase C package only
#    bash scripts/bash/autorun.sh deploy [options]    # Phase C run on current machine
#    bash scripts/bash/autorun.sh status              # show pipeline status
#    bash scripts/bash/autorun.sh stop                # stop TTS service
#    bash scripts/bash/autorun.sh list-ngc            # list container versions
#    bash scripts/bash/autorun.sh update-matrix       # update NGC compat matrix
#
#  Options:
#    --variant, -m <name>    Model variant (base-1.7b, custom-1.7b, ...)
#    --model-version <N>     Triton model version directory (default: 1)
#    --ngc-tag <tag>         Select NGC tritonserver tag (e.g. 25.03)
#    --target-driver <ver>   Target NVIDIA driver for NGC container selection
#    --yes, -y               Skip confirmations (non-interactive)
#    --dry-run               Show what would be done
#    -h, --help              Show full help
#
#    Phase A (forwarded to setup_env.sh):
#      --python <ver>        Python version (default: from NGC matrix)
#      --env-name <name>     Virtual env name (default: qwen3-tts)
#      --source <src>        Model source (auto|hf|modelscope)
#      --skip-models         Skip model download
#      --skip-deps           Skip dependency installation
#      --skip-export         Skip model export
#
#    Phase B (forwarded to build_engines.sh):
#      --max-batch-size <N>  TRT max batch (default: suggested by exported model + target GPU memory)
#      --max-input-len <N>   TRT max prefill/input token length
#      --max-seq-len <N>     TRT max KV sequence length
#      --image <uri>         Override NGC container image
#      --dtype <type>        Engine precision: bf16|fp16|fp32 (default: bf16)
#      --build-device <dev>  GPU for TensorRT build (auto|all|N|cuda:N)
#
#    Phase C (forwarded to deploy.sh):
#      --gateway <mode>      Gateway: standalone | triton | engine-docker
#      --engine-mode <mode>  Triton assemble mode: trt | onnx
#      --runtime-max-batch-size <N> Runtime scheduler max batch
#      --runtime-max-seq-len <N> Runtime scheduler max sequence length
#      --runtime-device <N>  GPU for serving runtime (auto|N|cuda:N)
#      --engine-image <tag>  engine-docker image tag (optional)
#      --port <port>         Standalone gRPC port (default: 50051)
#      --grpc-port <port>    Triton gRPC port (default: 8001)
#      --http-port <port>    Triton HTTP port (default: 8000)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../.." && pwd))"
source "${SCRIPT_DIR}/tools.sh"

# ── Known model variant names (for auto-detection of positional args) ──
_KNOWN_VARIANTS="base-1.7b custom-1.7b design-1.7b base-0.6b custom-0.6b all-1.7b all"
_KNOWN_COMMANDS="all setup build package deploy status stop list-ngc update-matrix probe-target discover-target make-bundle import-artifact remote-build help"

_is_variant() { [[ " $_KNOWN_VARIANTS " == *" $1 "* ]]; }
_is_command() { [[ " $_KNOWN_COMMANDS " == *" $1 "* ]]; }
_is_enabled() { [[ "${1:-}" == "true" || "${1:-}" == "1" ]]; }
_env_or_empty() { printenv "$1" 2>/dev/null || true; }

# ── Defaults ──
COMMAND=""
VARIANT="${MODEL_VARIANT:-}"
MODEL_VERSION="${MODEL_VERSION:-${ENGINE_MODEL_VERSION:-1}}"
DRY_RUN=false
YES_MODE=false
REBUILD_IMAGE=false
TARGET_DRIVER="${TARGET_DRIVER:-}"
TARGET_PROFILE="${TARGET_PROFILE:-}"
NGC_TAG="${NGC_TAG:-}"
GLOBAL_DEVICE="${QWEN3_TTS_GPU_DEVICE:-${GPU_DEVICE:-}}"
EXPORT_DEVICE="${EXPORT_DEVICE:-}"
BUILD_GPU_DEVICE="${BUILD_GPU_DEVICE:-}"
RUNTIME_GPU_DEVICE="${RUNTIME_GPU_DEVICE:-}"

# Phase A forwarding
SETUP_ARGS=()
PYTHON_VERSION="${PYTHON_VERSION:-}"
ENV_NAME="${ENV_NAME:-}"
MODEL_SOURCE="${MODEL_SOURCE:-}"
SKIP_MODELS=false
SKIP_DEPS=false
SKIP_EXPORT=false

# Phase B forwarding
BUILD_ARGS=()
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
BUILD_IMAGE="${NGC_IMAGE:-}"
BUILD_IMAGE_EXPLICIT=false
ENGINE_DTYPE="${ENGINE_DTYPE:-}"
TRITON_IO_FLOAT_DTYPE="${TRITON_IO_FLOAT_DTYPE:-}"
BUNDLE_OUT="${BUNDLE_OUT:-${REPO_ROOT}/workspace/engine_build_bundle.tar.zst}"
ARTIFACT_IN=""
REMOTE_HOST=""
REMOTE_WORKDIR="/tmp/qwen3-tts-engine-build"

# discover-target options (mutually exclusive); resolved in cmd_discover_target.
DISCOVER_MODE=""        # local | remote | paste
DISCOVER_OUT="${REPO_ROOT}/workspace/target_profile.json"

# Phase C forwarding
DEPLOY_ARGS=()
GATEWAY_MODE="${GATEWAY_MODE:-}"
ENGINE_MODE="${ENGINE_MODE:-}"
ENGINE_PORT="$(_env_or_empty ENGINE_GRPC_PORT)"
ENGINE_DOCKER_IMAGE="$(_env_or_empty ENGINE_IMAGE)"
ENGINE_DOCKER_IMAGE_EXPLICIT=false
GRPC_PORT="$(_env_or_empty TRITON_GRPC_PORT)"
HTTP_PORT="$(_env_or_empty TRITON_HTTP_PORT)"
RUNTIME_MAX_BATCH_SIZE="${RUNTIME_MAX_BATCH_SIZE:-}"
RUNTIME_MAX_SEQ_LEN="${RUNTIME_MAX_SEQ_LEN:-}"

# ── Help ──

usage() {
    cat << 'EOF'
Usage: autorun.sh [command] [model_variant] [options]

Commands:
  all               Local full pipeline: setup → build → package → run (default)
  setup             Phase A only (environment + model export)
  build             Phase B only (TensorRT engine compilation)
  package           Phase C package only (assemble model repo / build image; do not run)
  deploy            Phase C run only (start service on the current machine)
  status            Show pipeline status
  stop              Stop TTS service
  probe-target      Capture target_profile.json on the current host (local)
  discover-target   Acquire target_profile.json from the production target
                    (--local / --remote-host <h> / --paste)
  make-bundle       Create an offline TensorRT engine build bundle
  import-artifact   Import engine artifact bundle returned from target host
  remote-build      Build engines on an SSH target and import the result
  update-matrix     Update NGC compatibility matrix from NVIDIA website
  list-ngc          List selectable NGC container versions
  help              Show this help

If no command is given, launches interactive mode (Phase C: standalone | triton | engine-docker).
If a model variant name is given without a command, runs the full pipeline.

Options:
  --variant, -m <name>    Model variant (base-1.7b, custom-1.7b, design-1.7b,
                          base-0.6b, custom-0.6b, all-1.7b, all)
  --model-version <N>     Triton model version directory (default: 1)
  --ngc-tag <tag>         Select NGC tritonserver:<tag>-py3 for Phase B/C
                          (default: newest compatible tag for current driver).
  --target-driver <ver>   Target NVIDIA driver version for NGC container
                          selection (e.g. 575.57 for production machines).
                          Overrides local driver detection.
  --target-profile <json> Target profile from probe_target.sh; this becomes
                          the single source of truth for NGC tag selection.
  --device <dev>          Use this GPU for export, build, and runtime
                          (auto | N | cuda:N; default: auto).
  --export-device <dev>   Override Phase A export device only (auto | cpu | N | cuda:N).
  --build-device <dev>    Override Phase B TensorRT build GPU only (auto | all | N | cuda:N).
  --runtime-device <dev>  Override Phase C serving GPU only (auto | N | cuda:N).
  --yes, -y               Skip confirmations (non-interactive)
  --build, --rebuild-image
                          Rebuild runtime image from current code before package/deploy
  --dry-run               Show what would be done without executing
  --out <path>            Bundle/profile output path for probe-target/make-bundle
  --artifact <path>       Engine artifact bundle for import-artifact
  --remote-host <host>    SSH host for remote-build
  --remote-workdir <dir>  Remote work directory for remote-build
  -h, --help              Show this help

Phase A options (forwarded to setup_env.sh):
  --python <version>      Python version (default: from NGC matrix)
  --env-name <name>       Virtual env name (default: qwen3-tts)
  --source <source>       Model download source (auto|hf|modelscope)
  --skip-models           Skip model download
  --skip-deps             Skip dependency installation
  --skip-export           Skip model export

Phase B options (forwarded to build_engines.sh):
  --max-batch-size <N>    TensorRT profile max batch (default: suggested by exported model + target GPU memory)
  --max-input-len <N>     TensorRT profile max input/prefill length
  --max-seq-len <N>       TensorRT profile max sequence/KV length
  --image <uri>           Override NGC container image
  --dtype <type>          Engine precision: bf16|fp16|fp32|fp8 (default: bf16)
  --triton-io-float-dtype <type>
                          Float I/O dtype for generated TRT/Triton configs
                          Default: same as --dtype. Use fp32 if Triton reports dtype mismatch
                          (e.g. TYPE_FP32 vs TYPE_BF16).

Phase C options (forwarded to deploy.sh):
  --gateway <mode>        Gateway: standalone | triton | engine-docker (default: standalone)
  --engine-mode <mode>    Triton assemble mode: trt | onnx (default: trt)
  --runtime-max-batch-size <N>
                          Runtime scheduler max batch size
  --runtime-max-seq-len <N>
                          Runtime scheduler max sequence length
  --engine-image <tag>    Image for engine-docker (default: qwen3-engine:<ngc-tag>, else qwen3-engine:26.02)
  --port <port>           Standalone / engine-docker gRPC port (default: 50051)
  --grpc-port <port>      Triton gRPC port (default: 8001)
  --http-port <port>      Triton HTTP port (default: 8000)

Examples:
  autorun.sh                          # interactive guided setup
  autorun.sh base-1.7b                # full pipeline for base-1.7b
  autorun.sh all -m custom-1.7b       # full pipeline for custom-1.7b
  autorun.sh all -m custom-1.7b --device auto
  autorun.sh setup --skip-export      # Phase A without export
  autorun.sh custom-1.7b --dtype fp32 # Full pipeline, FP32 engines (fixes dtype mismatch)
  autorun.sh build -m custom-1.7b --dtype fp32    # Phase B with fp32
  autorun.sh build -m custom-1.7b --dtype fp16    # Phase B with fp16
  autorun.sh build -m custom-1.7b --max-batch-size 64 --max-input-len 128 --max-seq-len 512
  autorun.sh build -m custom-1.7b --build-device 1
  autorun.sh build -m custom-1.7b --ngc-tag 25.03
  autorun.sh list-ngc                  # list available container tags
  autorun.sh build --target-driver 575.57   # build for production driver
  autorun.sh package --gateway engine-docker -m custom-1.7b
  autorun.sh package --gateway engine-docker -m custom-1.7b --build
  autorun.sh deploy                   # Phase C run on current machine (standalone engine)
  autorun.sh deploy --gateway triton         # Phase C (Triton)
  autorun.sh deploy --gateway engine-docker  # Phase C (engine container image)
  autorun.sh make-bundle -m custom-1.7b --target-profile target_profile.json --out workspace/engine_build_bundle.tar.zst
  autorun.sh import-artifact workspace/engine_artifact_bundle.tar.zst
  autorun.sh remote-build -m custom-1.7b --target-profile target_profile.json --remote-host user@prod-gpu-host
  autorun.sh deploy --runtime-max-batch-size 32 --runtime-max-seq-len 512
  autorun.sh status                   # show pipeline status
  autorun.sh update-matrix            # fetch latest NGC compat data
EOF
}

# ── Argument parsing ──
# Handles: autorun.sh [command] [variant] [--options...]
# Smart detection: first positional arg can be a command or variant name.

parse_args() {
    # First pass: detect command and variant from positional args
    local positionals=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --variant|-m)       VARIANT="$2"; shift 2 ;;
            --model-version)    MODEL_VERSION="$2"; shift 2 ;;
            --ngc-tag|--container-tag) NGC_TAG="$2"; shift 2 ;;
            --yes|-y)           YES_MODE=true; shift ;;
            --build|--rebuild-image) REBUILD_IMAGE=true; shift ;;
            --dry-run)          DRY_RUN=true; shift ;;
            --target-driver)    TARGET_DRIVER="$2"; shift 2 ;;
            --target-profile)   TARGET_PROFILE="$2"; shift 2 ;;
            --device)           GLOBAL_DEVICE="$2"; shift 2 ;;
            --export-device)    EXPORT_DEVICE="$2"; shift 2 ;;
            --build-device)     BUILD_GPU_DEVICE="$2"; shift 2 ;;
            --runtime-device)   RUNTIME_GPU_DEVICE="$2"; shift 2 ;;
            --help|-h)          usage; exit 0 ;;

            # Phase A
            --python)           PYTHON_VERSION="$2"; shift 2 ;;
            --env-name)         ENV_NAME="$2"; shift 2 ;;
            --source)           MODEL_SOURCE="$2"; shift 2 ;;
            --skip-models)      SKIP_MODELS=true; shift ;;
            --skip-deps)        SKIP_DEPS=true; shift ;;
            --skip-export)      SKIP_EXPORT=true; shift ;;

            # Phase B
            --max-batch-size)   MAX_BATCH_SIZE="$2"; shift 2 ;;
            --max-input-len)    MAX_INPUT_LEN="$2"; shift 2 ;;
            --max-seq-len)      MAX_SEQ_LEN="$2"; shift 2 ;;
            --image)            BUILD_IMAGE="$2"; BUILD_IMAGE_EXPLICIT=true; shift 2 ;;
            --dtype)            ENGINE_DTYPE="$2"; shift 2 ;;
            --triton-io-float-dtype) TRITON_IO_FLOAT_DTYPE="$2"; shift 2 ;;
            --out)              BUNDLE_OUT="$2"; DISCOVER_OUT="$2"; shift 2 ;;
            --artifact)         ARTIFACT_IN="$2"; shift 2 ;;
            --remote-host)      REMOTE_HOST="$2"; DISCOVER_MODE="${DISCOVER_MODE:-remote}"; shift 2 ;;
            --remote-workdir)   REMOTE_WORKDIR="$2"; shift 2 ;;
            --local)            DISCOVER_MODE="local"; shift ;;
            --paste)            DISCOVER_MODE="paste"; shift ;;

            # Phase C
            --gateway)          GATEWAY_MODE="$2"; shift 2 ;;
            --engine-mode)      ENGINE_MODE="$2"; shift 2 ;;
            --runtime-max-batch-size|--runtime-max-batch) RUNTIME_MAX_BATCH_SIZE="$2"; shift 2 ;;
            --runtime-max-seq-len|--runtime-max-seq) RUNTIME_MAX_SEQ_LEN="$2"; shift 2 ;;
            --engine-image)      ENGINE_DOCKER_IMAGE="$2"; ENGINE_DOCKER_IMAGE_EXPLICIT=true; shift 2 ;;
            --port)             ENGINE_PORT="$2"; shift 2 ;;
            --grpc-port)        GRPC_PORT="$2"; shift 2 ;;
            --http-port)        HTTP_PORT="$2"; shift 2 ;;

            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                positionals+=("$1")
                shift
                ;;
        esac
    done

    # Resolve positional arguments
    for pos in "${positionals[@]}"; do
        if [ -z "$COMMAND" ] && _is_command "$pos"; then
            COMMAND="$pos"
        elif [ -z "$VARIANT" ] && _is_variant "$pos"; then
            VARIANT="$pos"
            # Bare variant without command implies "all"
            [ -z "$COMMAND" ] && COMMAND="all"
        elif [ "$COMMAND" = "import-artifact" ] && [ -z "$ARTIFACT_IN" ]; then
            ARTIFACT_IN="$pos"
        else
            log_error "Unexpected argument: $pos"
            usage
            exit 1
        fi
    done
}

# ── Build forwarding arg arrays ──

build_forward_args() {
    # Target driver override (propagated via env var to all phases)
    if [ -n "$TARGET_PROFILE" ]; then
        export TARGET_PROFILE
        if [ -z "$TARGET_DRIVER" ]; then
            TARGET_DRIVER=$(cross_host_json_value "$TARGET_PROFILE" driver_version || true)
            [ -n "$TARGET_DRIVER" ] && export TARGET_DRIVER
        fi
        NGC_TAG=$(resolve_ngc_tag_from_profile "$TARGET_PROFILE") || exit 1
        export NGC_TAG
    fi
    if [ -n "$TARGET_DRIVER" ]; then
        export TARGET_DRIVER
    fi
    MODEL_VERSION=$(resolve_model_version "$MODEL_VERSION") || exit 1

    if [ -z "$NGC_TAG" ] && ! $BUILD_IMAGE_EXPLICIT; then
        case "${COMMAND:-all}" in
            all|setup|build)
                NGC_TAG=$(resolve_ngc_tag "${TARGET_DRIVER:-}" 2>/dev/null || true)
                ;;
            deploy)
                # A deploy-only invocation must not re-derive NGC_TAG from the
                # host driver here.  deploy.sh resolves the runtime image from
                # the current Phase B manifest, then forces a fresh assemble.
                :
                ;;
        esac
    fi
    if [ -n "$NGC_TAG" ]; then
        export NGC_TAG
        if ! resolve_ngc_entry_by_tag "$NGC_TAG" >/dev/null; then
            exit 1
        fi
        local _ngc_image
        _ngc_image=$(resolve_ngc_image_from_tag "$NGC_TAG") || exit 1
        local _torch_tag
        _torch_tag=$(resolve_ngc_torch_index_tag "$NGC_TAG") || exit 1
        local _trt_python_version
        _trt_python_version=$(resolve_ngc_tag_tensorrt_version "$NGC_TAG") || exit 1
        local _trt_cuda_version
        _trt_cuda_version=$(resolve_ngc_tag_cuda_version "$NGC_TAG") || exit 1
        if ! $BUILD_IMAGE_EXPLICIT; then
            BUILD_IMAGE="$_ngc_image"
        fi
        export TRITON_BASE_IMAGE="${TRITON_BASE_IMAGE:-$_ngc_image}"
        export ENGINE_BASE_IMAGE="${ENGINE_BASE_IMAGE:-nvcr.io/nvidia/tensorrt:${NGC_TAG}-py3}"
        export TRITON_IMAGE="${TRITON_IMAGE:-qwen3-tts-triton:${NGC_TAG}}"
        if ! $ENGINE_DOCKER_IMAGE_EXPLICIT; then
            ENGINE_DOCKER_IMAGE="qwen3-engine:${NGC_TAG}"
            export ENGINE_IMAGE="$ENGINE_DOCKER_IMAGE"
        fi
        export PYTORCH_CUDA_TAG="${PYTORCH_CUDA_TAG:-$_torch_tag}"
        export ENGINE_PYTORCH_CUDA_TAG="${ENGINE_PYTORCH_CUDA_TAG:-$_torch_tag}"
        export TRITON_PYTORCH_CUDA_TAG="${TRITON_PYTORCH_CUDA_TAG:-$_torch_tag}"
        export TRITON_TENSORRT_PYTHON_VERSION="${TRITON_TENSORRT_PYTHON_VERSION:-$_trt_python_version}"
        export TRITON_TENSORRT_PIP_VERSION="${TRITON_TENSORRT_PIP_VERSION:-$_trt_python_version}"
        export STANDALONE_ENGINE_TENSORRT_PIP_VERSION="${STANDALONE_ENGINE_TENSORRT_PIP_VERSION:-$_trt_python_version}"
        case "${_trt_cuda_version%%.*}" in
            13) export STANDALONE_ENGINE_TENSORRT_PIP_PACKAGE="${STANDALONE_ENGINE_TENSORRT_PIP_PACKAGE:-tensorrt-cu13}" ;;
            12) export STANDALONE_ENGINE_TENSORRT_PIP_PACKAGE="${STANDALONE_ENGINE_TENSORRT_PIP_PACKAGE:-tensorrt-cu12}" ;;
            *)  export STANDALONE_ENGINE_TENSORRT_PIP_PACKAGE="${STANDALONE_ENGINE_TENSORRT_PIP_PACKAGE:-tensorrt}" ;;
        esac
    fi
    if [ -z "$ENGINE_DOCKER_IMAGE" ]; then
        ENGINE_DOCKER_IMAGE="$(_env_or_empty ENGINE_IMAGE)"
    fi

    # Phase A args
    SETUP_ARGS=()
    if [ -n "$VARIANT" ]; then
        # Pass variant via env var (setup_env.sh reads MODEL_VARIANT)
        export MODEL_VARIANT="$VARIANT"
    fi
    export AUTHORUN_COMMAND="${COMMAND:-}"
    if [ -n "$GATEWAY_MODE" ]; then
        export GATEWAY_MODE
    fi
    export MODEL_VERSION
    export ENGINE_MODEL_VERSION="$MODEL_VERSION"
    if [ -n "$PYTHON_VERSION" ]; then
        export PYTHON_VERSION
    fi
    if [ -n "$MODEL_SOURCE" ]; then
        export MODEL_SOURCE="$MODEL_SOURCE"
    fi
    local effective_export_device="${EXPORT_DEVICE:-$GLOBAL_DEVICE}"
    local effective_build_device="${BUILD_GPU_DEVICE:-$GLOBAL_DEVICE}"
    local effective_runtime_device="${RUNTIME_GPU_DEVICE:-$GLOBAL_DEVICE}"

    if [ -n "$effective_export_device" ]; then
        export EXPORT_DEVICE="$effective_export_device"
    fi
    if $SKIP_MODELS; then export SKIP_MODELS=1; fi
    if $SKIP_DEPS; then export SKIP_DEPS=1; fi
    if $SKIP_EXPORT; then export SKIP_EXPORT=1; fi
    if [ -n "$ENV_NAME" ]; then export ENV_NAME="$ENV_NAME"; fi

    # Phase B args — "all"/"all-1.7b" are meta-variants; omit --variant so
    # build_engines.sh discovers all exported checkpoints automatically.
    BUILD_ARGS=()
    if [ -n "$VARIANT" ] && [[ "$VARIANT" != all* ]]; then
        BUILD_ARGS+=(--variant "$VARIANT")
    fi
    if [ -n "$TARGET_DRIVER" ]; then BUILD_ARGS+=(--target-driver "$TARGET_DRIVER"); fi
    if [ -n "$TARGET_PROFILE" ]; then BUILD_ARGS+=(--target-profile "$TARGET_PROFILE"); fi
    if [ -n "$effective_build_device" ]; then BUILD_ARGS+=(--device "$effective_build_device"); fi
    if [ -n "$MAX_BATCH_SIZE" ]; then BUILD_ARGS+=(--max-batch-size "$MAX_BATCH_SIZE"); fi
    if [ -n "$MAX_INPUT_LEN" ]; then BUILD_ARGS+=(--max-input-len "$MAX_INPUT_LEN"); fi
    if [ -n "$MAX_SEQ_LEN" ]; then BUILD_ARGS+=(--max-seq-len "$MAX_SEQ_LEN"); fi
    if [ -n "$BUILD_IMAGE" ]; then BUILD_ARGS+=(--image "$BUILD_IMAGE"); fi
    if [ -n "$ENGINE_DTYPE" ]; then BUILD_ARGS+=(--dtype "$ENGINE_DTYPE"); fi
    if [ -n "$TRITON_IO_FLOAT_DTYPE" ]; then BUILD_ARGS+=(--triton-io-float-dtype "$TRITON_IO_FLOAT_DTYPE"); fi
    if [ -n "$BUNDLE_OUT" ]; then BUILD_ARGS+=(--out "$BUNDLE_OUT"); fi
    if [ -n "$REMOTE_HOST" ]; then BUILD_ARGS+=(--remote-host "$REMOTE_HOST"); fi
    if [ -n "$REMOTE_WORKDIR" ]; then BUILD_ARGS+=(--remote-workdir "$REMOTE_WORKDIR"); fi
    if $DRY_RUN; then BUILD_ARGS+=(--dry-run); fi

    # Phase C args (forwarded to deploy.sh)
    DEPLOY_ARGS=()
    if [ -n "$VARIANT" ] && [[ "$VARIANT" != all* ]]; then
        DEPLOY_ARGS+=(--variant "$VARIANT")
    fi
    if [ -n "$GATEWAY_MODE" ]; then DEPLOY_ARGS+=(--gateway "$GATEWAY_MODE"); fi
    if [ -n "$ENGINE_MODE" ]; then DEPLOY_ARGS+=(--engine-mode "$ENGINE_MODE"); fi
    if [ -n "$effective_runtime_device" ]; then DEPLOY_ARGS+=(--device "$effective_runtime_device"); fi
    if [ -n "$RUNTIME_MAX_BATCH_SIZE" ]; then DEPLOY_ARGS+=(--max-batch "$RUNTIME_MAX_BATCH_SIZE"); fi
    if [ -n "$RUNTIME_MAX_SEQ_LEN" ]; then DEPLOY_ARGS+=(--max-seq-len "$RUNTIME_MAX_SEQ_LEN"); fi
    if $REBUILD_IMAGE; then DEPLOY_ARGS+=(--build); fi
    if $ENGINE_DOCKER_IMAGE_EXPLICIT; then
        if [ -n "$ENGINE_DOCKER_IMAGE" ]; then DEPLOY_ARGS+=(--engine-image "$ENGINE_DOCKER_IMAGE"); fi
    fi
    if [ -n "$ENGINE_PORT" ]; then DEPLOY_ARGS+=(--port "$ENGINE_PORT"); fi
    if [ -n "$MODEL_VERSION" ]; then DEPLOY_ARGS+=(--model-version "$MODEL_VERSION"); fi
    if $BUILD_IMAGE_EXPLICIT; then
        if [ -n "$BUILD_IMAGE" ]; then DEPLOY_ARGS+=(--image "$BUILD_IMAGE"); fi
    fi
    if [ -n "$GRPC_PORT" ]; then
        export TRITON_GRPC_PORT="$GRPC_PORT"
    fi
    if [ -n "$HTTP_PORT" ]; then
        export TRITON_HTTP_PORT="$HTTP_PORT"
    fi
    if $DRY_RUN; then DEPLOY_ARGS+=(--dry-run); fi
}

# ── Phase runners ──

run_phase_a() {
    log_step "阶段 A: 环境配置与模型导出"
    echo ""

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run: bash setup_env.sh"
        [ -n "$VARIANT" ] && log_info "  MODEL_VARIANT=$VARIANT"
        [ -n "$PYTHON_VERSION" ] && log_info "  PYTHON_VERSION=$PYTHON_VERSION"
        [ -n "$ENV_NAME" ] && log_info "  ENV_NAME=$ENV_NAME"
        [ -n "$MODEL_SOURCE" ] && log_info "  MODEL_SOURCE=$MODEL_SOURCE"
        [ -n "${EXPORT_DEVICE:-$GLOBAL_DEVICE}" ] && log_info "  EXPORT_DEVICE=${EXPORT_DEVICE:-$GLOBAL_DEVICE}"
        _is_enabled "$SKIP_MODELS" && log_info "  SKIP_MODELS=1"
        _is_enabled "$SKIP_DEPS" && log_info "  SKIP_DEPS=1"
        _is_enabled "$SKIP_EXPORT" && log_info "  SKIP_EXPORT=1"
        return 0
    fi

    bash "${SCRIPT_DIR}/setup_env.sh" "${SETUP_ARGS[@]}" || {
        log_error "Phase A 失败。"
        log_info  "请修正问题后重新执行: bash scripts/bash/autorun.sh setup"
        return 1
    }
}

run_phase_b() {
    log_step "阶段 B: TensorRT 引擎编译 (trtexec)"
    echo ""

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run unified bundle compile: variants=${VARIANT:-all} dtype=${ENGINE_DTYPE:-bf16} io_dtype=${TRITON_IO_FLOAT_DTYPE:-${ENGINE_DTYPE:-bf16}}"
        return 0
    fi

    # Unified path: all builds (local / cross-host / SSH) go through the
    # same bundle primitive.  Local mode synthesizes a bundle workspace
    # via hardlinks so we pay no extra disk cost, then runs
    # compile_engines_in_bundle + write_artifact_manifest +
    # engine_fingerprint_check, finally folding the engines back into
    # workspace/exported/.
    local exported_dir="${REPO_ROOT}/workspace/exported"
    local target_profile_path="${TARGET_PROFILE:-${REPO_ROOT}/workspace/target_profile.json}"

    if [ ! -f "$target_profile_path" ]; then
        if [ -t 0 ]; then
            log_warn "未发现 target_profile.json: $target_profile_path"
            log_warn "本机编译需要先 discover-target；现在用 --local 自动采集本机指纹（仅当本机就是目标硬件时合法）"
            DISCOVER_MODE="local"
            DISCOVER_OUT="$target_profile_path"
            cmd_discover_target || return 1
        else
            log_error "未发现 target_profile.json: $target_profile_path"
            log_error "请先运行: bash scripts/bash/autorun.sh discover-target --local"
            log_error "  或:    bash scripts/bash/autorun.sh discover-target --remote-host <user@dsw>"
            log_error "  或:    bash scripts/bash/autorun.sh discover-target --paste"
            return 1
        fi
    fi

    # Discover variants if none provided (mirrors build_engines.sh discover_trt_variants).
    local -a variants_arr=()
    if [ -n "${VARIANT:-}" ] && [[ "$VARIANT" != all* ]]; then
        variants_arr=("$VARIANT")
    else
        local v_dir
        for v_dir in "$exported_dir"/*/; do
            [ -d "$v_dir" ] || continue
            if [ -f "${v_dir}talker_code2wav_fused.onnx" ]; then
                variants_arr+=("$(basename "$v_dir")")
            fi
        done
        if [ ${#variants_arr[@]} -eq 0 ]; then
            log_error "No talker_code2wav_fused.onnx in $exported_dir"
            log_error "Run Phase A first: bash scripts/bash/autorun.sh setup"
            return 1
        fi
    fi
    local variant_csv
    variant_csv=$(IFS=,; echo "${variants_arr[*]}")

    local engine_dtype="${ENGINE_DTYPE:-bf16}"
    local io_dtype="${TRITON_IO_FLOAT_DTYPE:-$engine_dtype}"
    local build_device="${BUILD_GPU_DEVICE:-${GLOBAL_DEVICE:-auto}}"

    # Resolve build profile defaults from target GPU memory.  Same logic
    # as build_engines.sh::_resolve_build_profile_defaults so all paths
    # produce consistent engine profiles.
    local _bp_gpu_index=0
    if [[ "$build_device" =~ ^[0-9]+$ ]]; then
        _bp_gpu_index="$build_device"
    elif [[ "$build_device" =~ ^cuda:([0-9]+)$ ]]; then
        _bp_gpu_index="${BASH_REMATCH[1]}"
    fi
    local _bp_suggested
    _bp_suggested=$(resolve_build_profile_for_target \
        "$target_profile_path" "$_bp_gpu_index" \
        "$exported_dir" "$variant_csv" \
        "$engine_dtype" "${MAX_INPUT_LEN:-128}" "${MAX_SEQ_LEN:-512}")
    local _bp_arr
    read -r -a _bp_arr <<< "$_bp_suggested"
    local max_batch="${MAX_BATCH_SIZE:-${_bp_arr[0]:-32}}"
    local max_input="${MAX_INPUT_LEN:-${_bp_arr[1]:-128}}"
    local max_seq="${MAX_SEQ_LEN:-${_bp_arr[2]:-512}}"
    local _bp_summary
    _bp_summary=$(summarize_build_profile_from_exports \
        "$(target_profile_memory_mb "$target_profile_path" "$_bp_gpu_index")" \
        "$exported_dir" "$variant_csv" "$engine_dtype" "$max_input" "$max_seq")

    local cache_root="${REPO_ROOT}/workspace/.local-build-cache"
    log_info "Phase B (unified bundle): variants=$variant_csv dtype=$engine_dtype"
    [ -n "$_bp_summary" ] && log_info "  Export-aware sizing: $_bp_summary"
    log_info "  bundle workspace: $cache_root"

    # Build the local bundle workspace via hardlinks (fast).  scripts/ is
    # symlinked to repo so build_on_target/build_engines see the same code
    # without paying a multi-MB copy.
    prepare_local_bundle_workspace \
        "$REPO_ROOT" \
        "$exported_dir" \
        "$target_profile_path" \
        "$cache_root" \
        "$variant_csv" \
        "$engine_dtype" \
        "$io_dtype" \
        "$max_batch" \
        "$max_input" \
        "$max_seq" \
        "$build_device" \
        || { log_error "prepare_local_bundle_workspace failed"; return 1; }

    compile_engines_in_bundle "$cache_root" || {
        log_error "compile_engines_in_bundle failed"
        return 1
    }

    write_artifact_manifest "$cache_root" || {
        log_error "write_artifact_manifest failed"
        return 1
    }

    # Strict fingerprint validation before we copy engines back to the
    # canonical workspace/exported/ tree.  Cross-validates against
    # target_profile so a mis-built engine cannot pollute downstream.
    if ! engine_fingerprint_check \
        "$cache_root/artifact_manifest.json" \
        "$cache_root/workspace/exported" \
        "$target_profile_path"; then
        if [ "${ALLOW_FINGERPRINT_MISMATCH:-}" = "1" ]; then
            log_warn "Fingerprint mismatch ignored (ALLOW_FINGERPRINT_MISMATCH=1)"
        else
            log_error "Engine fingerprint check failed; not importing into workspace/exported/"
            log_error "Set ALLOW_FINGERPRINT_MISMATCH=1 to override (debugging only)"
            return 1
        fi
    fi

    collect_local_engines_to_workspace "$cache_root" "$exported_dir" || {
        log_error "collect_local_engines_to_workspace failed"
        return 1
    }

    echo "$engine_dtype" > "$exported_dir/.engine_dtype"
    log_info "Phase B 完成。engines + artifact_manifest 已落到 $exported_dir"
}

run_phase_c() {
    log_step "阶段 C: 当前机器启动 TTS 服务"
    echo ""

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run: bash deploy.sh run ${DEPLOY_ARGS[*]}"
        return 0
    fi

    # Strict pre-deploy fingerprint check.  Refuses to deploy engines
    # whose hardware/runtime fingerprint doesn't match the target host's
    # triton_manifest.json (and target_profile.json when present).  The
    # check is bypassable via ALLOW_FINGERPRINT_MISMATCH=1 only for
    # explicit debugging.
    local artifact_manifest="${REPO_ROOT}/workspace/exported/artifact_manifest.json"
    local exported_dir="${REPO_ROOT}/workspace/exported"
    local target_profile_path="${TARGET_PROFILE:-${REPO_ROOT}/workspace/target_profile.json}"
    if [ ! -f "$artifact_manifest" ]; then
        log_error "未发现 artifact_manifest.json: $artifact_manifest"
        log_error "  Strict 模式要求 deploy 前必须有 build 写入的 artifact_manifest。"
        log_error "  请先执行:"
        log_error "    bash scripts/bash/autorun.sh build       # 本机/SSH 路径"
        log_error "    bash scripts/bash/autorun.sh import-artifact <bundle>  # 跨机路径"
        log_error "  调试可临时使用: ALLOW_FINGERPRINT_MISMATCH=1 autorun.sh deploy"
        if [ "${ALLOW_FINGERPRINT_MISMATCH:-}" != "1" ]; then
            return 1
        fi
        log_warn "ALLOW_FINGERPRINT_MISMATCH=1 — 跳过指纹校验（仅用于调试）"
    else
        local tp_arg=""
        [ -f "$target_profile_path" ] && tp_arg="$target_profile_path"
        if ! engine_fingerprint_check "$artifact_manifest" "$exported_dir" "$tp_arg"; then
            if [ "${ALLOW_FINGERPRINT_MISMATCH:-}" = "1" ]; then
                log_warn "Fingerprint mismatch ignored (ALLOW_FINGERPRINT_MISMATCH=1)"
            else
                log_error "  请重新 build 或 import-artifact 以更新 engines。"
                return 1
            fi
        fi
    fi

    bash "${SCRIPT_DIR}/deploy.sh" run "${DEPLOY_ARGS[@]}" || {
        log_error "Phase C 失败。"
        log_info  "请修正问题后重新执行: bash scripts/bash/autorun.sh deploy"
        return 1
    }
}

run_phase_c_package() {
    log_step "阶段 C: 组装/打包部署产物"
    echo ""

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run: bash deploy.sh package ${DEPLOY_ARGS[*]}"
        return 0
    fi

    local artifact_manifest="${REPO_ROOT}/workspace/exported/artifact_manifest.json"
    local exported_dir="${REPO_ROOT}/workspace/exported"
    local target_profile_path="${TARGET_PROFILE:-${REPO_ROOT}/workspace/target_profile.json}"
    if [ ! -f "$artifact_manifest" ]; then
        log_error "未发现 artifact_manifest.json: $artifact_manifest"
        log_error "  打包前必须先完成 Phase B 或 import-artifact。"
        return 1
    fi
    local tp_arg=""
    [ -f "$target_profile_path" ] && tp_arg="$target_profile_path"
    if ! engine_fingerprint_check "$artifact_manifest" "$exported_dir" "$tp_arg"; then
        if [ "${ALLOW_FINGERPRINT_MISMATCH:-}" = "1" ]; then
            log_warn "Fingerprint mismatch ignored (ALLOW_FINGERPRINT_MISMATCH=1)"
        else
            log_error "  请重新 build 或 import-artifact 以更新 engines。"
            return 1
        fi
    fi

    bash "${SCRIPT_DIR}/deploy.sh" package "${DEPLOY_ARGS[@]}" || {
        log_error "Phase C package 失败。"
        log_info  "请修正问题后重新执行: bash scripts/bash/autorun.sh package"
        return 1
    }
}

# ── Command handlers ──

cmd_all() {
    show_run_banner "完整本机流程" "A → B → package → run"

    run_phase_a || exit 1
    echo ""
    run_phase_b || exit 1
    echo ""
    run_phase_c_package || exit 1
    echo ""
    run_phase_c || exit 1

    echo ""
    if $DRY_RUN; then
        echo -e "${_CLR_GREEN}预演完成：未实际组装产物或启动服务。${_CLR_RESET}"
    else
        echo -e "${_CLR_GREEN}全部阶段完成！部署产物已组装，TTS 服务已在当前机器运行。${_CLR_RESET}"
    fi
    echo ""
}

cmd_setup() {
    show_run_banner "阶段 A" "环境 + 导出"
    run_phase_a || exit 1
}

cmd_build() {
    show_run_banner "阶段 B" "TensorRT 引擎"
    run_phase_b || exit 1
}

cmd_package() {
    show_run_banner "阶段 C" "组装/打包产物（不启动服务）"
    run_phase_c_package || exit 1
}

cmd_probe_target() {
    local out="${BUNDLE_OUT:-${REPO_ROOT}/workspace/target_profile.json}"
    bash "${SCRIPT_DIR}/probe_target.sh" --out "$out"
}

# ---------------------------------------------------------------------------
#  cmd_discover_target — acquire target_profile.json from the production
#  target via one of three modes.  Picks the right mode based on:
#    --local                       run probe on this host
#    --remote-host <user@host>     ssh + probe + scp back
#    --paste                       print probe script, accept pasted JSON
#  Default (no flag): --local (kept for compatibility with autorun.sh
#  probe-target behaviour) but warns the user that the result reflects
#  the local host, not the production target.
# ---------------------------------------------------------------------------
cmd_discover_target() {
    local mode="${DISCOVER_MODE:-}"
    local out="$DISCOVER_OUT"
    mkdir -p "$(dirname "$out")"

    if [ -z "$mode" ]; then
        if [ -n "$REMOTE_HOST" ]; then
            mode="remote"
        else
            log_warn "No --local / --remote-host / --paste flag; defaulting to --local"
            mode="local"
        fi
    fi

    case "$mode" in
        local)
            log_step "Discovering target profile (local host)"
            bash "${SCRIPT_DIR}/probe_target.sh" --out "$out"
            ;;
        remote)
            [ -n "$REMOTE_HOST" ] || { log_error "--remote-host is required for remote discovery"; exit 1; }
            log_step "Discovering target profile via SSH: $REMOTE_HOST"
            local remote_dir="${REMOTE_WORKDIR:-/tmp/qwen3-tts-engine-build}"
            ssh "$REMOTE_HOST" "mkdir -p '$remote_dir'" || { log_error "ssh to $REMOTE_HOST failed"; exit 1; }
            # Ship the entire scripts/ directory so probe_target.sh can source tools.sh.
            scp -r "${REPO_ROOT}/scripts" "$REMOTE_HOST:$remote_dir/" \
                || { log_error "scp scripts to $REMOTE_HOST failed"; exit 1; }
            ssh "$REMOTE_HOST" "bash '$remote_dir/scripts/bash/probe_target.sh' --out '$remote_dir/target_profile.json'" \
                || { log_error "remote probe_target.sh failed"; exit 1; }
            scp "$REMOTE_HOST:$remote_dir/target_profile.json" "$out" \
                || { log_error "scp target_profile.json from $REMOTE_HOST failed"; exit 1; }
            log_info "Target profile retrieved from $REMOTE_HOST → $out"
            ;;
        paste)
            log_step "Paste-mode target discovery"
            local probe_script="${SCRIPT_DIR}/_probe_standalone.py"
            if [ ! -f "$probe_script" ]; then
                log_error "Standalone probe missing: $probe_script"
                exit 1
            fi
            echo ""
            echo "  Paste mode (适用于 DSW 没 SSH 的场景):"
            echo "    1) 在目标机器（DSW / 容器）打开终端"
            echo "    2) 把下面分隔符之间的 Python 探测脚本完整粘贴并保存为 probe.py"
            echo "       (脚本是 self-contained，只依赖 python3 + nvidia-smi)"
            echo "    3) 运行: python3 probe.py --out /tmp/target_profile.json"
            echo "    4) cat /tmp/target_profile.json 内容粘贴回这里，Ctrl-D 结束"
            echo ""
            echo "  一行复制命令（在 DSW 上执行——把分隔符内整段粘贴到 heredoc 之间）:"
            echo "    cat > probe.py <<'PROBE_EOF'"
            echo "    ... (粘贴下方脚本) ..."
            echo "    PROBE_EOF"
            echo "    python3 probe.py --out /tmp/target_profile.json"
            echo ""
            echo "============================== BEGIN probe.py =============================="
            cat "$probe_script"
            echo "=============================== END probe.py ==============================="
            echo ""
            log_info "Paste the JSON output from DSW (Ctrl-D to finish):"
            local pasted=""
            pasted=$(cat)
            if [ -z "$pasted" ]; then
                log_error "Empty paste; aborting"
                exit 1
            fi
            printf '%s\n' "$pasted" > "$out"
            # Validate it's actually a JSON profile, not a paste of probe.py itself.
            if ! python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$out" 2>/dev/null; then
                log_error "Pasted content is not valid JSON: $out"
                log_error "  Expected: the JSON output of 'python3 probe.py --out /tmp/...'"
                log_error "  Got first line: $(head -1 "$out")"
                exit 1
            fi
            log_info "Target profile saved: $out"
            ;;
        *)
            log_error "Unknown discover-target mode: $mode"
            exit 1
            ;;
    esac

    # Sanity check the resulting JSON.
    if ! cross_host_json_value "$out" driver_version >/dev/null 2>&1; then
        log_error "$out does not look like a target_profile.json (missing driver_version)"
        exit 1
    fi
    local _drv _ngc _sm
    _drv=$(cross_host_json_value "$out" driver_version)
    _ngc=$(cross_host_json_value "$out" recommended_ngc_tag)
    _sm=$(cross_host_json_value "$out" "gpus[0].sm" 2>/dev/null || true)
    log_info "  driver: $_drv  ngc: $_ngc  sm: ${_sm:-?}"
}

cmd_make_bundle() {
    show_run_banner "跨机 Phase B" "生成 engine build bundle"
    bash "${SCRIPT_DIR}/build_engines.sh" make-bundle "${BUILD_ARGS[@]}" || exit 1
}

cmd_import_artifact() {
    [ -n "$ARTIFACT_IN" ] || { log_error "Missing --artifact <engine_artifact_bundle.tar.zst>"; exit 1; }
    bash "${SCRIPT_DIR}/build_engines.sh" import-artifact "$ARTIFACT_IN" || exit 1
}

cmd_remote_build() {
    show_run_banner "跨机 Phase B" "remote-ssh engine build"
    bash "${SCRIPT_DIR}/build_engines.sh" remote-build "${BUILD_ARGS[@]}" || exit 1
}

cmd_deploy() {
    show_run_banner "阶段 C" "当前机器启动 TTS 服务"
    run_phase_c || exit 1
}

cmd_status() {
    print_status_summary "$REPO_ROOT" "$VARIANT"
}

cmd_stop() {
    local stop_args=()
    [ -n "$VARIANT" ] && stop_args+=(--variant "$VARIANT")
    bash "${SCRIPT_DIR}/deploy.sh" stop "${stop_args[@]}"
}

cmd_update_matrix() {
    source "${SCRIPT_DIR}/lib/ngc_updater.sh"
    update_ngc_matrix "${SCRIPT_DIR}/ngc_matrix.conf"
}

cmd_list_ngc() {
    list_ngc_matrix "${TARGET_DRIVER:-}"
}

# ── Cross-host interactive guide ──

_prompt_with_default() {
    local prompt="$1"
    local default_value="$2"
    local timeout="${3:-30}"
    local value=""
    if [ -t 0 ]; then
        read -rp "$prompt [$default_value] (${timeout}s 后自动): " -t "$timeout" value || true
    fi
    echo "${value:-$default_value}"
}

_default_triton_io_dtype() {
    case "${1:-bf16}" in
        bf16|fp16|fp32) echo "$1" ;;
        *) echo "fp32" ;;
    esac
}

_prompt_build_dtypes() {
    local engine_default="${ENGINE_DTYPE:-bf16}"
    local io_default="${TRITON_IO_FLOAT_DTYPE:-$(_default_triton_io_dtype "$engine_default")}"

    echo ""
    echo "  精度配置 (阶段 B)"
    echo "    引擎精度: TensorRT builder/compute precision"
    echo "    I/O 精度: fused TensorRT/Triton float binding dtype，默认跟随引擎精度"

    ENGINE_DTYPE=$(_prompt_with_default "  引擎精度 bf16|fp16|fp32|fp8" "$engine_default")
    io_default="${TRITON_IO_FLOAT_DTYPE:-$(_default_triton_io_dtype "$ENGINE_DTYPE")}"
    TRITON_IO_FLOAT_DTYPE=$(_prompt_with_default "  浮点 I/O 精度 bf16|fp16|fp32" "$io_default")
}

interactive_cross_host_guide() {
    echo ""
    echo "  跨机 Engine 编译引导"
    echo ""
    echo "  适用场景：当前机器负责导图/打包，但 TensorRT engine 要在生产同构 GPU 上编译。"
    echo ""
    echo "  [0] discover-target — 获取目标机器 target_profile.json"
    echo "      支持本地 / SSH 远端 / 复制粘贴 三种方式 (DSW 推荐 SSH 或粘贴)"
    echo "  [1] 在当前机器采集 target_profile.json (= [0] --local)"
    echo "  [2] 生成离线 engine_build_bundle.tar.zst"
    echo "      当前机器已有 ONNX + 目标机器 target_profile.json。"
    echo "  [3] 导入目标机器返回的 engine_artifact_bundle.tar.zst"
    echo "      导入 .engine 并执行严格指纹校验。"
    echo "  [4] SSH 远端编译并自动导入"
    echo "      当前机器能 SSH 到生产同构 GPU。"
    echo "  [5] 打印完整流程命令"
    echo "  [q] 返回/退出"
    echo ""

    local xchoice=""
    if [ ! -t 0 ]; then
        log_info "非交互模式，打印跨机流程说明"
        xchoice="5"
    else
        read -rp "  请选择 [0-5/q] (默认: 5): " -t 30 xchoice || true
        xchoice="${xchoice:-5}"
    fi

    case "$xchoice" in
        0)
            DISCOVER_OUT=$(_prompt_with_default "  target_profile.json 输出路径" "${REPO_ROOT}/workspace/target_profile.json")
            echo ""
            echo "  选择获取方式:"
            echo "    [a] --local    当前主机就是生产目标"
            echo "    [b] --remote   SSH 到生产机器"
            echo "    [c] --paste    打印探测脚本并粘贴回 JSON"
            local _dchoice=""
            read -rp "  请选择 [a/b/c] (默认: a): " -t 30 _dchoice || true
            _dchoice="${_dchoice:-a}"
            case "$_dchoice" in
                a) DISCOVER_MODE="local" ;;
                b)
                    DISCOVER_MODE="remote"
                    REMOTE_HOST=$(_prompt_with_default "  SSH 目标主机 user@host" "${REMOTE_HOST:-user@prod-gpu-host}")
                    REMOTE_WORKDIR=$(_prompt_with_default "  远端工作目录" "$REMOTE_WORKDIR")
                    ;;
                c) DISCOVER_MODE="paste" ;;
                *) DISCOVER_MODE="local" ;;
            esac
            cmd_discover_target
            ;;
        1)
            BUNDLE_OUT=$(_prompt_with_default "  target_profile.json 输出路径" "${REPO_ROOT}/workspace/target_profile.json")
            cmd_probe_target
            ;;
        2)
            if [ -z "$VARIANT" ]; then
                VARIANT=$(select_model_variant)
                VARIANT="${VARIANT:-base-1.7b}"
                export MODEL_VARIANT="$VARIANT"
            fi
            TARGET_PROFILE=$(_prompt_with_default "  目标机器 target_profile.json 路径" "${REPO_ROOT}/workspace/target_profile.json")
            BUNDLE_OUT=$(_prompt_with_default "  engine build bundle 输出路径" "${REPO_ROOT}/workspace/engine_build_bundle.tar.zst")
            _prompt_build_dtypes
            build_forward_args
            cmd_make_bundle
            ;;
        3)
            ARTIFACT_IN=$(_prompt_with_default "  engine artifact bundle 路径" "${REPO_ROOT}/workspace/engine_artifact_bundle.tar.zst")
            cmd_import_artifact
            ;;
        4)
            if [ -z "$VARIANT" ]; then
                VARIANT=$(select_model_variant)
                VARIANT="${VARIANT:-base-1.7b}"
                export MODEL_VARIANT="$VARIANT"
            fi
            TARGET_PROFILE=$(_prompt_with_default "  目标机器 target_profile.json 路径" "${REPO_ROOT}/workspace/target_profile.json")
            REMOTE_HOST=$(_prompt_with_default "  SSH 目标主机 user@host" "${REMOTE_HOST:-user@prod-gpu-host}")
            REMOTE_WORKDIR=$(_prompt_with_default "  远端工作目录" "$REMOTE_WORKDIR")
            _prompt_build_dtypes
            build_forward_args
            cmd_remote_build
            ;;
        5)
            cat <<EOF

  离线推荐流程：

    # 1. 在生产同构 GPU 机器上：
    bash scripts/bash/probe_target.sh --out target_profile.json

    # 2. 把 target_profile.json 拷回当前导图/打包机器，生成构建包：
    bash scripts/bash/autorun.sh make-bundle -m custom-1.7b \\
      --target-profile target_profile.json \\
      --dtype bf16 \\
      --triton-io-float-dtype bf16 \\
      --out workspace/engine_build_bundle.tar.zst

    # 3. 把 engine_build_bundle.tar.zst 拷到目标机器：
    mkdir -p /tmp/qwen3-engine-build
    tar --zstd -xf engine_build_bundle.tar.zst -C /tmp/qwen3-engine-build
    cd /tmp/qwen3-engine-build
    bash build_on_target.sh

    # 4. 把 engine_artifact_bundle.tar.zst 拷回当前机器并导入：
    bash scripts/bash/autorun.sh import-artifact workspace/engine_artifact_bundle.tar.zst

    # 5. 回到打包机：组装模型包并用当前代码构建 engine 镜像；不启动服务：
    bash scripts/bash/autorun.sh package -m custom-1.7b --gateway engine-docker --build

    # 6. 只有当前机器就是生产服务机/本机验证机时，才启动服务：
    bash scripts/bash/autorun.sh deploy -m custom-1.7b --gateway engine-docker

  SSH 自动流程：

    bash scripts/bash/autorun.sh remote-build -m custom-1.7b \\
      --target-profile target_profile.json \\
      --dtype bf16 \\
      --triton-io-float-dtype bf16 \\
      --remote-host user@prod-gpu-host \\
      --remote-workdir /tmp/qwen3-engine-build

EOF
            ;;
        q|Q)
            echo "  已退出跨机引导。"
            ;;
        *)
            log_error "无效选择: $xchoice"
            exit 1
            ;;
    esac
}

# ── Banner ──

show_run_banner() {
    local phase_name="$1"
    local description="$2"

    echo ""
    echo -e "${_CLR_BLUE}╔══════════════════════════════════════════════════════════╗${_CLR_RESET}"
    echo -e "${_CLR_BLUE}║     Qwen3-TTS Triton — ${phase_name}${_CLR_RESET}"
    echo -e "${_CLR_BLUE}╚══════════════════════════════════════════════════════════╝${_CLR_RESET}"
    echo ""
    echo "  模式:      $description"
    [ -n "$VARIANT" ] && echo "  变体:      $VARIANT"
    [ -n "$MODEL_VERSION" ] && echo "  版本:      $MODEL_VERSION"
    [ -n "${GATEWAY_MODE:-}" ] && echo "  阶段 C:    $GATEWAY_MODE"
    [ -n "$ENGINE_DTYPE" ] && echo "  引擎精度:  $ENGINE_DTYPE"
    if [ -n "$TRITON_IO_FLOAT_DTYPE" ]; then
        echo "  I/O 精度:  $TRITON_IO_FLOAT_DTYPE"
    elif [ -n "$ENGINE_DTYPE" ]; then
        echo "  I/O 精度:  $ENGINE_DTYPE (默认跟随引擎精度)"
    fi
    [ -n "${EXPORT_DEVICE:-$GLOBAL_DEVICE}" ] && echo "  导出 GPU:  ${EXPORT_DEVICE:-$GLOBAL_DEVICE}"
    [ -n "${BUILD_GPU_DEVICE:-$GLOBAL_DEVICE}" ] && echo "  构建 GPU:  ${BUILD_GPU_DEVICE:-$GLOBAL_DEVICE}"
    [ -n "${RUNTIME_GPU_DEVICE:-$GLOBAL_DEVICE}" ] && echo "  运行 GPU:  ${RUNTIME_GPU_DEVICE:-$GLOBAL_DEVICE}"
    [ -n "$MAX_BATCH_SIZE" ] && echo "  构建 batch: $MAX_BATCH_SIZE"
    [ -n "$MAX_INPUT_LEN" ] && echo "  构建 input: $MAX_INPUT_LEN"
    [ -n "$MAX_SEQ_LEN" ] && echo "  构建 seq:   $MAX_SEQ_LEN"
    [ -n "$RUNTIME_MAX_BATCH_SIZE" ] && echo "  运行 batch: $RUNTIME_MAX_BATCH_SIZE"
    [ -n "$RUNTIME_MAX_SEQ_LEN" ] && echo "  运行 seq:   $RUNTIME_MAX_SEQ_LEN"
    [ -n "$TARGET_DRIVER" ] && echo "  目标驱动:  $TARGET_DRIVER"
    [ -n "$NGC_TAG" ] && echo "  NGC tag:   $NGC_TAG"
    [ -n "$BUILD_IMAGE" ] && echo "  Phase B镜像: $BUILD_IMAGE"
    [ -n "${TRITON_BASE_IMAGE:-}" ] && echo "  Triton base: ${TRITON_BASE_IMAGE}"
    [ -n "${ENGINE_BASE_IMAGE:-}" ] && echo "  Engine base: ${ENGINE_BASE_IMAGE}"
    $DRY_RUN && echo "  预演:      是"
    echo ""
}

# ── Interactive mode ──

interactive_mode() {
    print_status_summary "$REPO_ROOT" ""

    # Determine where we are in the pipeline
    local resume_point
    resume_point=$(detect_resume_point "$REPO_ROOT" "")
    local resume_label
    case "$resume_point" in
        setup)   resume_label="setup" ;;
        build)   resume_label="build" ;;
        package) resume_label="package" ;;
        deploy)  resume_label="run" ;;
        done)    resume_label="done" ;;
        *)       resume_label="$resume_point" ;;
    esac

    echo "  要执行什么操作？"
    echo ""
    echo "  [1] 完整本机流程  (setup → build → package → run)"
    echo "  [2] 环境配置      (阶段 A)"
    echo "  [3] 构建引擎      (阶段 B)"
    echo "  [4] 组装/打包产物 (不启动服务)"
    echo "  [5] 当前机器启动服务 (run，不做跨机部署)"
    if [ "$resume_point" != "done" ]; then
        echo "  [6] 从 $resume_label 恢复"
    else
        echo "  [6] 已完成/查看当前状态"
    fi
    echo "  [7] 查看详细状态"
    echo "  [8] 跨机编译引导"
    echo "  [q] 退出"
    echo ""

    local choice
    if [ ! -t 0 ]; then
        log_info "非交互模式，默认执行完整本机流程"
        choice="1"
    else
        read -rp "  请选择 [1-8/q] (默认: 1, 30s 后自动执行完整本机流程): " -t 30 choice || true
        choice="${choice:-1}"
    fi

    case "$choice" in
        7)
            print_status_summary "$REPO_ROOT" "$VARIANT"
            return
            ;;
        8)
            interactive_cross_host_guide
            return
            ;;
        q|Q)
            echo "  已退出。"
            exit 0
            ;;
        1|2|3|4|5|6) ;;
        *)
            log_error "无效选择: $choice"
            exit 1
            ;;
    esac

    # Ask for variant if not already set
    if [ -z "$VARIANT" ]; then
        echo ""
        VARIANT=$(select_model_variant)
        VARIANT="${VARIANT:-base-1.7b}"
        export MODEL_VARIANT="$VARIANT"
    fi

    local needs_setup=false
    case "${choice:-1}" in
        1|2) needs_setup=true ;;
        6) [[ "$resume_point" == "setup" ]] && needs_setup=true ;;
    esac

    # Ask for engine dtype and profile when Phase B will run.
    local needs_build=false
    case "${choice:-1}" in
        1|3) needs_build=true ;;
        6) [[ "$resume_point" == "setup" || "$resume_point" == "build" ]] && needs_build=true ;;
    esac

    # Phase C package gateway: standalone | Triton | engine Docker image
    local will_package=false
    case "${choice:-1}" in
        1|4) will_package=true ;;
        6) [[ "$resume_point" == "setup" || "$resume_point" == "build" || "$resume_point" == "package" ]] && will_package=true ;;
    esac

    # Phase C run gateway: current machine only
    local will_deploy=false
    case "${choice:-1}" in
        1|5) will_deploy=true ;;
        6) [[ "$resume_point" != "done" ]] && will_deploy=true ;;
    esac

    if [ -t 0 ] && { $needs_setup || $needs_build || $will_package || $will_deploy; }; then
        echo ""
        echo "  模型版本 (Triton model version)"
        local _mv=""
        read -rp "  版本号 [${MODEL_VERSION}] (30s 后自动): " -t 30 _mv || true
        if [ -n "$_mv" ]; then
            MODEL_VERSION="$_mv"
        fi
    fi

    if [ -t 0 ] && { $needs_setup || $needs_build || $will_deploy; } && \
        [ -z "$GLOBAL_DEVICE" ] && [ -z "$EXPORT_DEVICE" ] && [ -z "$BUILD_GPU_DEVICE" ] && [ -z "$RUNTIME_GPU_DEVICE" ]; then
        echo ""
        echo "  GPU 选择 (默认: auto，自动选择当前空闲显存最多的 GPU)"
        if command -v nvidia-smi &>/dev/null; then
            nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null \
                | sed 's/^/    /' || true
        fi
        local _gpu=""
        read -rp "  GPU [auto] (可填 0 / 1 / cuda:1，30s 后自动): " -t 30 _gpu || true
        if [ -n "$_gpu" ]; then
            GLOBAL_DEVICE="$_gpu"
        fi
    fi

    if $needs_build && [ -t 0 ]; then
        _prompt_build_dtypes
    fi

    if $needs_build && [ -z "$NGC_TAG" ] && ! $BUILD_IMAGE_EXPLICIT && [ -t 0 ]; then
        local _selected_ngc_tag=""
        _selected_ngc_tag=$(select_ngc_tag_interactive "${TARGET_DRIVER:-}") || exit 1
        if [ -n "$_selected_ngc_tag" ]; then
            NGC_TAG="$_selected_ngc_tag"
        fi
    fi

    if $needs_build && [ -t 0 ] && \
        { [ -z "$MAX_BATCH_SIZE" ] || [ -z "$MAX_INPUT_LEN" ] || [ -z "$MAX_SEQ_LEN" ]; }; then
        echo ""
        echo "  TensorRT 构建 profile (留空则 Phase B 按构建 GPU 显存给建议默认值)"
        local _mb="" _mi="" _ms=""
        [ -z "$MAX_BATCH_SIZE" ] && read -rp "  max-batch-size [auto]: " -t 30 _mb || true
        [ -z "$MAX_INPUT_LEN" ] && read -rp "  max-input-len [auto]: " -t 30 _mi || true
        [ -z "$MAX_SEQ_LEN" ] && read -rp "  max-seq-len [auto]: " -t 30 _ms || true
        [ -n "$_mb" ] && MAX_BATCH_SIZE="$_mb"
        [ -n "$_mi" ] && MAX_INPUT_LEN="$_mi"
        [ -n "$_ms" ] && MAX_SEQ_LEN="$_ms"
    fi

    if { $will_package || $will_deploy; } && [ -z "$GATEWAY_MODE" ] && [ -t 0 ]; then
        echo ""
        echo "  阶段 C — 产物/运行方式:"
        echo "    [1] standalone    — 组装模型包；run 时本机 Python 运行 engine"
        echo "    [2] triton        — 组装 model_repository；package 可构建自包含 Triton 镜像"
        echo "    [3] engine-docker — 组装模型包并构建 engine 镜像；run 时挂载模型包启动容器"
        echo ""
        local gwch=""
        read -rp "  请选择 [1-3] (默认: 1 standalone, 30s 后选默认): " -t 30 gwch || true
        gwch="${gwch:-1}"
        case "$gwch" in
            2) GATEWAY_MODE="triton" ;;
            3) GATEWAY_MODE="engine-docker" ;;
            *) GATEWAY_MODE="standalone" ;;
        esac
        log_info "已选择 Phase C 方式: $GATEWAY_MODE"
    fi
    if { $will_package || $will_deploy; } && [ -z "$GATEWAY_MODE" ]; then
        GATEWAY_MODE="standalone"
    fi

    if $will_deploy && ! $will_package && [ "$GATEWAY_MODE" = "engine-docker" ] && ! $REBUILD_IMAGE && [ -t 0 ]; then
        echo ""
        echo "  engine-docker 镜像代码更新"
        echo "  run 只负责在当前机器启动服务；如果 engine/ 或 scripts/compose 有改动，需要重建镜像。"
        local _rebuild=""
        read -rp "  重新构建 engine 镜像以包含当前代码？[y/N] (30s 后选 N): " -t 30 _rebuild || true
        case "$_rebuild" in
            y|Y|yes|YES) REBUILD_IMAGE=true ;;
        esac
    fi

    if $will_deploy && [ -t 0 ] && \
        { [ -z "$RUNTIME_MAX_BATCH_SIZE" ] || [ -z "$RUNTIME_MAX_SEQ_LEN" ]; }; then
        echo ""
        echo "  Runtime 上限 (留空则读取 manifest engine_profile；不能超过 Phase B profile)"
        local _rb="" _rs=""
        [ -z "$RUNTIME_MAX_BATCH_SIZE" ] && read -rp "  runtime-max-batch-size [manifest]: " -t 30 _rb || true
        [ -z "$RUNTIME_MAX_SEQ_LEN" ] && read -rp "  runtime-max-seq-len [manifest]: " -t 30 _rs || true
        [ -n "$_rb" ] && RUNTIME_MAX_BATCH_SIZE="$_rb"
        [ -n "$_rs" ] && RUNTIME_MAX_SEQ_LEN="$_rs"
    fi

    case "${choice:-1}" in
        1) COMMAND="all" ;;
        2) COMMAND="setup" ;;
        3) COMMAND="build" ;;
        4) COMMAND="package" ;;
        5) COMMAND="deploy" ;;
        6)
            case "$resume_point" in
                setup)   COMMAND="all" ;;
                build)   COMMAND="build" ;;
                package) COMMAND="package" ;;
                deploy)  COMMAND="deploy" ;;
                done)    COMMAND="status" ;;
                *)       COMMAND="all" ;;
            esac
            ;;
    esac

    build_forward_args

    case "${choice:-1}" in
        1) cmd_all ;;
        2) cmd_setup ;;
        3) cmd_build ;;
        4) cmd_package ;;
        5) cmd_deploy ;;
        6)
            case "$resume_point" in
                setup)   cmd_all ;;
                build)   cmd_build; echo ""; cmd_package; echo ""; cmd_deploy ;;
                package) cmd_package; echo ""; cmd_deploy ;;
                deploy)  cmd_deploy ;;
                done)    cmd_status ;;
                *)      cmd_all ;;
            esac
            ;;
    esac
}

# ── Main ──

main() {
    parse_args "$@"
    MODEL_VERSION=$(resolve_model_version "$MODEL_VERSION") || exit 1
    export MODEL_VERSION
    export ENGINE_MODEL_VERSION="$MODEL_VERSION"

    # No command → interactive mode
    if [ -z "$COMMAND" ]; then
        interactive_mode
        return
    fi

    # Resolve variant interactively if needed for phases that require it
    if [ -z "$VARIANT" ] && [[ "$COMMAND" =~ ^(all|setup)$ ]] && ! $YES_MODE; then
        if [ -t 0 ]; then
            VARIANT=$(select_model_variant)
            VARIANT="${VARIANT:-base-1.7b}"
        else
            VARIANT="base-1.7b"
        fi
    fi

    build_forward_args

    case "$COMMAND" in
        all)            cmd_all ;;
        setup)          cmd_setup ;;
        build)          cmd_build ;;
        package)        cmd_package ;;
        probe-target)   cmd_probe_target ;;
        discover-target) cmd_discover_target ;;
        make-bundle)    cmd_make_bundle ;;
        import-artifact) cmd_import_artifact ;;
        remote-build)   cmd_remote_build ;;
        deploy)         cmd_deploy ;;
        status)         cmd_status ;;
        stop)           cmd_stop ;;
        list-ngc)       cmd_list_ngc ;;
        update-matrix)  cmd_update_matrix ;;
        help)           usage ;;
        *)              log_error "Unknown command: $COMMAND"; usage; exit 1 ;;
    esac
}

main "$@"
