#!/bin/bash
# ===========================================================================
#  status.sh — Pipeline status detection for all three phases
#
#  Functions: detect_phase_a_status, detect_phase_b_status,
#             detect_phase_c_package_status, detect_phase_c_status,
#             detect_available_variants,
#             print_status_summary, detect_resume_point
#  Depends:   lib/logging.sh, lib/utils.sh
#
#  Used by autorun.sh to provide an interactive status dashboard and
#  determine which pipeline stages still need to run.
# ===========================================================================

[[ -n "${_LIB_STATUS_LOADED:-}" ]] && return 0
_LIB_STATUS_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"

# Status is communicated via stdout (echo), not exit codes.
# All detect_* functions return 0 to stay compatible with set -e.

# ---------------------------------------------------------------------------
#  detect_phase_a_status <exported_dir> [variant]
#
#  Checks Phase A artifacts under exported_dir:
#    - tokenizer/ with ONNX models (shared)
#    - <variant>/ with ONNX + checkpoint + weights
#
#  Outputs one of: none | partial | complete
# ---------------------------------------------------------------------------
detect_phase_a_status() {
    local exported_dir="$1"
    local variant="${2:-}"

    if [ ! -d "$exported_dir" ]; then
        echo "none"
        return 0
    fi

    local has_tokenizer=false
    local has_variant=false
    local has_weights=false
    local has_onnx=false

    # Shared tokenizer models
    if [ -f "$exported_dir/tokenizer/speech_tokenizer_encoder.onnx" ] \
        && [ -f "$exported_dir/tokenizer/code2wav_decoder.onnx" ]; then
        has_tokenizer=true
    fi

    # Variant-specific artifacts
    if [ -n "$variant" ]; then
        local vdir="$exported_dir/$variant"
        if [ -d "$vdir" ]; then
            has_variant=true
            [ -d "$vdir/weights" ] && has_weights=true
            ls "$vdir"/*.onnx &>/dev/null 2>&1 && has_onnx=true
        fi
    else
        # Auto-detect any variant
        for vdir in "$exported_dir"/*/; do
            local vname
            vname=$(basename "$vdir")
            [[ "$vname" == "tokenizer" ]] && continue
            has_variant=true
            [ -d "$vdir/weights" ] && has_weights=true
            ls "$vdir"/*.onnx &>/dev/null 2>&1 && has_onnx=true
            break
        done
    fi

    if $has_tokenizer && $has_variant && $has_weights && $has_onnx; then
        echo "complete"
    elif $has_tokenizer || $has_variant; then
        echo "partial"
    else
        echo "none"
    fi
    return 0
}

# ---------------------------------------------------------------------------
#  detect_phase_b_status <exported_dir> [variant]
#
#  Checks for compiled TensorRT engines (.engine files alongside ONNX).
#  Outputs: none | complete
# ---------------------------------------------------------------------------
detect_phase_b_status() {
    local exported_dir="$1"
    local variant="${2:-}"

    if [ ! -d "$exported_dir" ]; then
        echo "none"
        return 0
    fi

    local check_dirs=()
    if [ -n "$variant" ]; then
        check_dirs=("$exported_dir/$variant")
    else
        for vdir in "$exported_dir"/*/; do
            local vname
            vname=$(basename "$vdir")
            [[ "$vname" == "tokenizer" ]] && continue
            check_dirs+=("$vdir")
        done
    fi

    for vdir in "${check_dirs[@]}"; do
        if [ -f "$vdir/talker_code2wav_fused.engine" ] || [ -f "$vdir/talker_unified.engine" ]; then
            echo "complete"
            return 0
        fi
    done

    echo "none"
    return 0
}

# ---------------------------------------------------------------------------
#  detect_phase_c_package_status <repo_root> [variant] [model_version]
#
#  Checks whether the shared deployment model package exists. This is the
#  Phase C package artifact consumed by standalone, engine-docker, and Triton.
#  Outputs: none | partial | complete
# ---------------------------------------------------------------------------
detect_phase_c_package_status() {
    local repo_root="$1"
    local variant="${2:-}"
    local model_version
    model_version=$(resolve_model_version "${3:-${MODEL_VERSION:-${ENGINE_MODEL_VERSION:-1}}}" 2>/dev/null || echo "1")
    local package_dir="$repo_root/workspace/model_repository/tts_orchestrator/$model_version"
    local runtime_dir="$package_dir/runtime"

    if [ ! -d "$package_dir" ]; then
        echo "none"
        return 0
    fi

    local has_runtime=false
    local has_python=false
    local has_manifest=false
    local has_payload=false
    [ -f "$runtime_dir/model.plan" ] || [ -f "$runtime_dir/model.onnx" ] && has_runtime=true
    [ -f "$package_dir/model.py" ] && [ -d "$package_dir/engine" ] && has_python=true
    [ -f "$runtime_dir/triton_manifest.json" ] || [ -f "$package_dir/triton_manifest.json" ] && has_manifest=true

    if [ -n "$variant" ]; then
        local exported_manifest="$repo_root/workspace/exported/$variant/triton_manifest.json"
        if [ -f "$exported_manifest" ] && [ -f "$runtime_dir/triton_manifest.json" ]; then
            local repo_variant
            repo_variant=$(python3 - "$runtime_dir/triton_manifest.json" <<'PY' 2>/dev/null || true
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(data.get("model_variant") or data.get("variant") or "")
PY
)
            if [ -z "$repo_variant" ] || [ "$repo_variant" = "$variant" ]; then
                has_payload=true
            fi
        else
            has_payload=true
        fi
    else
        has_payload=true
    fi

    if $has_runtime && $has_python && $has_manifest && $has_payload; then
        echo "complete"
    else
        echo "partial"
    fi
    return 0
}

# ---------------------------------------------------------------------------
#  detect_phase_c_status [repo_root] [container_name]
#
#  Checks if TTS service is running (standalone engine, engine container, or Triton container).
#  Outputs: none | running | healthy
#
#  Checks standalone engine first (PID file), then engine/Triton containers.
# ---------------------------------------------------------------------------
detect_phase_c_status() {
    local repo_root="${1:-$(git rev-parse --show-toplevel 2>/dev/null || echo ".")}"
    local container_name="${2:-qwen3-tts-triton}"

    # Check standalone engine first
    local engine_st
    engine_st=$(engine_status "$repo_root" 2>/dev/null || echo "none")
    if [ "$engine_st" != "none" ]; then
        echo "$engine_st"
        return 0
    fi

    # Check engine Docker container
    if command -v docker &>/dev/null; then
        local engine_container="${ENGINE_CONTAINER_NAME:-qwen3-engine}"
        if docker ps -q --filter "name=$engine_container" 2>/dev/null | grep -q .; then
            local health_port="${ENGINE_HEALTH_PORT:-8080}"
            if curl -sf "http://localhost:${health_port}/health" &>/dev/null; then
                echo "healthy"
            else
                echo "running"
            fi
            return 0
        fi

        if docker ps -q --filter "name=$container_name" 2>/dev/null | grep -q .; then
            local http_port="${TRITON_HTTP_PORT:-8000}"
            if curl -sf "http://localhost:${http_port}/v2/health/ready" &>/dev/null; then
                echo "healthy"
            else
                echo "running"
            fi
            return 0
        fi
    fi

    echo "none"
    return 0
}

# ---------------------------------------------------------------------------
#  detect_available_variants <exported_dir> <models_dir>
#
#  Lists variants with their pipeline stage (downloaded / exported / engine / deployed).
#  Output: one line per variant, format "variant:stage"
#    stage = downloaded | exported | engine_ready
# ---------------------------------------------------------------------------
detect_available_variants() {
    local exported_dir="$1"
    local models_dir="$2"

    # Exported variants (Phase A output)
    if [ -d "$exported_dir" ]; then
        for vdir in "$exported_dir"/*/; do
            [ ! -d "$vdir" ] && continue
            local vname
            vname=$(basename "$vdir")
            [[ "$vname" == "tokenizer" ]] && continue

            if [ -f "$vdir/talker_code2wav_fused.engine" ] || [ -f "$vdir/talker_unified.engine" ]; then
                echo "${vname}:engine_ready"
            elif ls "$vdir"/*.onnx &>/dev/null 2>&1; then
                echo "${vname}:exported"
            fi
        done
    fi

    # Downloaded-only variants (not yet exported)
    if [ -d "$models_dir" ]; then
        for mdir in "$models_dir"/*/; do
            [ ! -d "$mdir" ] && continue
            local mname
            mname=$(basename "$mdir")

            # Map model directory names back to variant keys
            local vkey=""
            case "$mname" in
                Qwen3-TTS-12Hz-1.7B-Base)        vkey="base-1.7b" ;;
                Qwen3-TTS-12Hz-1.7B-CustomVoice) vkey="custom-1.7b" ;;
                Qwen3-TTS-12Hz-1.7B-VoiceDesign) vkey="design-1.7b" ;;
                Qwen3-TTS-12Hz-0.6B-Base)        vkey="base-0.6b" ;;
                Qwen3-TTS-12Hz-0.6B-CustomVoice) vkey="custom-0.6b" ;;
            esac

            [ -z "$vkey" ] && continue

            # Only report if not already exported
            if [ ! -d "$exported_dir/$vkey" ]; then
                echo "${vkey}:downloaded"
            fi
        done
    fi
}

# ---------------------------------------------------------------------------
#  _status_icon <status_string>
#  Maps status to a display character.
# ---------------------------------------------------------------------------
_status_icon() {
    case "$1" in
        complete|healthy|engine_ready) echo -e "${_CLR_GREEN}OK${_CLR_RESET}" ;;
        partial|running|exported)      echo -e "${_CLR_YELLOW}partial${_CLR_RESET}" ;;
        downloaded)                    echo -e "${_CLR_YELLOW}downloaded${_CLR_RESET}" ;;
        none)                          echo -e "${_CLR_RED}--${_CLR_RESET}" ;;
        *)                             echo "$1" ;;
    esac
}

# ---------------------------------------------------------------------------
#  print_status_summary <repo_root> [variant]
#
#  Prints a formatted status panel showing all three phase states,
#  detected GPU info, and available variants.
# ---------------------------------------------------------------------------
print_status_summary() {
    local repo_root="$1"
    local variant="${2:-}"
    local exported_dir="$repo_root/workspace/exported"
    local models_dir="$repo_root/workspace/models"

    # Detect phase status
    local phase_a phase_b phase_c_package phase_c
    phase_a=$(detect_phase_a_status "$exported_dir" "$variant")
    phase_b=$(detect_phase_b_status "$exported_dir" "$variant")
    phase_c_package=$(detect_phase_c_package_status "$repo_root" "$variant")
    phase_c=$(detect_phase_c_status "$repo_root")

    local pa_icon pb_icon pcp_icon pc_icon
    pa_icon=$(_status_icon "$phase_a")
    pb_icon=$(_status_icon "$phase_b")
    pcp_icon=$(_status_icon "$phase_c_package")
    pc_icon=$(_status_icon "$phase_c")

    # Detect GPU (best effort)
    local gpu_info="(not detected)"
    if command -v nvidia-smi &>/dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | sed 's/,/ (driver /' | sed 's/$/)/' ) || gpu_info="(detection failed)"
    fi

    # Detect Docker
    local docker_info="not installed"
    if command -v docker &>/dev/null; then
        if docker info &>/dev/null 2>&1; then
            docker_info="OK"
        else
            docker_info="installed (daemon not running)"
        fi
    fi

    echo ""
    echo -e "${_CLR_BLUE}╔══════════════════════════════════════════════════════════╗${_CLR_RESET}"
    echo -e "${_CLR_BLUE}║       Qwen3-TTS — Pipeline Status                        ║${_CLR_RESET}"
    echo -e "${_CLR_BLUE}╠══════════════════════════════════════════════════════════╣${_CLR_RESET}"
    echo -e "${_CLR_BLUE}║${_CLR_RESET}  GPU:    $gpu_info"
    echo -e "${_CLR_BLUE}║${_CLR_RESET}  Docker: $docker_info"
    echo -e "${_CLR_BLUE}╠══════════════════════════════════════════════════════════╣${_CLR_RESET}"
    echo -e "${_CLR_BLUE}║${_CLR_RESET}  Phase A (setup + export):     $pa_icon"
    echo -e "${_CLR_BLUE}║${_CLR_RESET}  Phase B (TRT engines):        $pb_icon"
    echo -e "${_CLR_BLUE}║${_CLR_RESET}  Phase C1 (package):           $pcp_icon"
    echo -e "${_CLR_BLUE}║${_CLR_RESET}  Phase C2 (current run):       $pc_icon"
    echo -e "${_CLR_BLUE}╠══════════════════════════════════════════════════════════╣${_CLR_RESET}"

    # List available variants
    local variants
    variants=$(detect_available_variants "$exported_dir" "$models_dir")

    if [ -n "$variants" ]; then
        echo -e "${_CLR_BLUE}║${_CLR_RESET}  Available variants:"
        while IFS=: read -r vname vstage; do
            local vicon
            vicon=$(_status_icon "$vstage")
            echo -e "${_CLR_BLUE}║${_CLR_RESET}    $vname  ($vicon)"
        done <<< "$variants"
    else
        echo -e "${_CLR_BLUE}║${_CLR_RESET}  No variants found (run setup first)"
    fi

    echo -e "${_CLR_BLUE}╚══════════════════════════════════════════════════════════╝${_CLR_RESET}"
    echo ""
}

# ---------------------------------------------------------------------------
#  detect_resume_point <repo_root> [variant]
#
#  Determines which phase to resume from based on current state.
#  Outputs: setup | build | package | deploy | done
# ---------------------------------------------------------------------------
detect_resume_point() {
    local repo_root="$1"
    local variant="${2:-}"
    local exported_dir="$repo_root/workspace/exported"

    local phase_a phase_b phase_c_package phase_c
    phase_a=$(detect_phase_a_status "$exported_dir" "$variant")
    phase_b=$(detect_phase_b_status "$exported_dir" "$variant")
    phase_c_package=$(detect_phase_c_package_status "$repo_root" "$variant")
    phase_c=$(detect_phase_c_status "$repo_root")

    if [ "$phase_c" = "healthy" ]; then
        echo "done"
    elif [ "$phase_c_package" = "complete" ]; then
        echo "deploy"
    elif [ "$phase_b" = "complete" ]; then
        echo "package"
    elif [ "$phase_a" = "complete" ]; then
        echo "build"
    else
        echo "setup"
    fi
}
