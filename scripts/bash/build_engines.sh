#!/bin/bash
# ===========================================================================
#  build_engines.sh — Phase B: Compile TensorRT engines from ONNX (trtexec)
#
#  Reads ONNX files produced by Phase A (export_models.sh / export_all.py) and
#  compiles them into TensorRT engines using trtexec inside an NGC container
#  (tritonserver:xx.yy-py3, trtexec at /usr/src/tensorrt/bin/trtexec).
#
#  Default production: speaker_encoder.engine, speech_tokenizer_codec_fused.engine (base),
#  talker_code2wav_fused.engine (per variant).
#  Set BUILD_VERIFICATION_ENGINES=1 to also build talker_unified, code2wav_decoder,
#  speech_tokenizer_encoder (tokenizer dir).
#
#  Prerequisites:
#    - NVIDIA GPU with driver >= 550.54
#    - Docker with NVIDIA Container Toolkit
#    - ONNX at workspace/exported/<variant>/ and workspace/exported/tokenizer/
#
#  Usage:
#    bash scripts/bash/build_engines.sh                    # all variants
#    bash scripts/bash/build_engines.sh --variant base-1.7b
#    bash scripts/bash/build_engines.sh --max-batch-size 4
#    bash scripts/bash/build_engines.sh --dry-run
#    bash scripts/bash/build_engines.sh --pull-only
#
#  Environment variables:
#    NGC_IMAGE        Docker image override (default: auto-detect from driver)
#    MAX_BATCH_SIZE   Max batch (default: auto by exported model + target GPU memory)
#    MAX_INPUT_LEN    Prefill len (default: auto by exported model + target GPU memory)
#    MAX_SEQ_LEN      Total seq len (default: auto by exported model + target GPU memory)
#    ENGINE_DTYPE     bfloat16|float16|float32|fp8 (default: bfloat16)
#    BUILD_GPU_DEVICE GPU for TRT build (auto | all | N | cuda:N; default: auto)
#
#  Output: workspace/exported/<variant>/*.engine, workspace/exported/tokenizer/*.engine
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../.." && pwd))"
source "${SCRIPT_DIR}/tools.sh"

# trtexec path inside NGC tritonserver image; exported so trtexec_runner
# uses the same constant when constructing docker run commands.
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
export TRTEXEC
DOCKER_GPU_ARGS=(--gpus all)
# QWEN3_IN_BUNDLE_ROOT lets compile_engines_in_bundle (build_pipeline.sh)
# redirect EXPORTED_DIR onto a bundle workspace so the same compile code
# runs unmodified for local / cross-host / SSH paths.

# ── Engine build defaults are resolved after GPU selection. ──
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
ENGINE_DTYPE="${ENGINE_DTYPE:-bfloat16}"
TRITON_IO_FLOAT_DTYPE="${TRITON_IO_FLOAT_DTYPE:-}"
# Per-submodule compute precision for the fused engine.
#
# CP (empty = follow ENGINE_DTYPE, i.e. bf16): CP once defaulted to fp32 while
# hunting the 0601-checkpoint-era streaming hallucination (bf16 flips near-tie
# argmax in the CP tail; see streaming_hallucination.md Finding #15). That
# hallucination turned out to be the damaged 0601 weights, not precision: with
# the 0701 retrain, a full-bf16 engine (cp=bf16) measures 0/100 hallucinations
# on the same deterministic seed set as the cp=fp32 baseline (2026-07-06), and
# cp=fp32 costs real decode latency (CP was ~45% of kernel time under fp32). CP
# now follows ENGINE_DTYPE; set CP_PRECISION=fp32 to reproduce the historical
# mixed-precision build or to debug numerical parity against ORT/PyTorch.
#
# Code2Wav (default fp16): TRT 10.13 on sm120 has no optimized bf16 conv kernel
# (microbench fp16 2.5-3.4x faster; conv768 k7 0.509->0.148ms), so a bf16 c2w
# vocoder dominates the decode step. c2w=fp16 pins only /code2wav/* to fp16 (the
# vocoder does not flow back into the talker sampling path — verified halluprobe
# 0/100 + normal audio level/spectrum, commit 23127b7) and cuts that cost hard.
# The mixed build auto-selects --precisionConstraints=prefer (see below) so the
# c2w Pad/Slice glue that lacks an fp16 kernel falls back instead of failing.
# Set CODE2WAV_PRECISION=bf16 for a uniform-bf16 numerical-parity build.
BACKBONE_PRECISION="${BACKBONE_PRECISION:-}"
CP_PRECISION="${CP_PRECISION:-}"
CODE2WAV_PRECISION="${CODE2WAV_PRECISION:-fp16}"
# Exported so the make-bundle manifest writer (a python heredoc subprocess) can
# persist these into build_manifest.json for the cross-host build to honor.
export BACKBONE_PRECISION CP_PRECISION CODE2WAV_PRECISION
BUILD_GPU_DEVICE="${BUILD_GPU_DEVICE:-auto}"
RESOLVED_BUILD_GPU_DEVICE=""

if [ -n "${QWEN3_IN_BUNDLE_ROOT:-}" ]; then
    EXPORTED_DIR="${QWEN3_IN_BUNDLE_ROOT}/workspace/exported"
else
    EXPORTED_DIR="${REPO_ROOT}/workspace/exported"
fi
TOKENIZER_DIR="${EXPORTED_DIR}/tokenizer"
VARIANT=""
DRY_RUN=false
PULL_ONLY=false
USER_IMAGE="${NGC_IMAGE:-}"
TARGET_DRIVER="${TARGET_DRIVER:-}"
COMMAND="build"
BUILD_MODE="${BUILD_MODE:-local}"
TARGET_PROFILE="${TARGET_PROFILE:-}"
BUNDLE_OUT="${BUNDLE_OUT:-${REPO_ROOT}/workspace/engine_build_bundle.tar.zst}"
ARTIFACT_IN=""
REMOTE_HOST=""
REMOTE_WORKDIR="/tmp/qwen3-tts-engine-build"
ALLOW_FINGERPRINT_MISMATCH=false

_normalize_dtype_value() {
    local value="${1,,}"
    case "$value" in
        bf16|bfloat16) echo "bf16" ;;
        fp16|float16)  echo "fp16" ;;
        fp32|float32)  echo "fp32" ;;
        fp8|float8)    echo "fp8" ;;
        *) return 1 ;;
    esac
}

normalize_build_dtypes() {
    ENGINE_DTYPE="$(_normalize_dtype_value "$ENGINE_DTYPE")" || {
        log_error "Unknown ENGINE_DTYPE: $ENGINE_DTYPE (use bf16|fp16|fp32|fp8)"
        exit 1
    }
    if [ -n "$TRITON_IO_FLOAT_DTYPE" ]; then
        TRITON_IO_FLOAT_DTYPE="$(_normalize_dtype_value "$TRITON_IO_FLOAT_DTYPE")" || {
            log_error "Unknown TRITON_IO_FLOAT_DTYPE: $TRITON_IO_FLOAT_DTYPE (use bf16|fp16|fp32|fp8)"
            exit 1
        }
    else
        TRITON_IO_FLOAT_DTYPE="$ENGINE_DTYPE"
    fi
}

normalize_build_dtypes

# _trtexec_precision_flags: echo trtexec precision flags for current ENGINE_DTYPE
_trtexec_precision_flags() {
    case "$ENGINE_DTYPE" in
        bf16) echo "--bf16" ;;
        fp16) echo "--fp16" ;;
        fp32) echo "" ;;
        fp8)  echo "--fp8" ;;
        *)    echo "" ;;
    esac
}

# _trtexec_io_format: echo IO format string for float tensors (e.g. bf16:chw)
_trtexec_io_format() {
    case "$ENGINE_DTYPE" in
        bf16) echo "bf16:chw" ;;
        fp16) echo "fp16:chw" ;;
        fp32) echo "fp32:chw" ;;
        fp8)  echo "fp8:chw" ;;
        *)    echo "fp32:chw" ;;
    esac
}

_validate_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        log_error "$name must be a positive integer, got: $value"
        exit 1
    fi
}

_resolve_build_gpu_device() {
    local norm
    norm=$(normalize_gpu_device "$BUILD_GPU_DEVICE") || exit 1
    if [ "$norm" = "auto" ]; then
        RESOLVED_BUILD_GPU_DEVICE=$(select_best_gpu_index)
        log_gpu_selection "Phase B build" "$RESOLVED_BUILD_GPU_DEVICE"
    elif [ "$norm" = "all" ]; then
        RESOLVED_BUILD_GPU_DEVICE="all"
        log_info "Phase B build GPU: all visible GPUs (trtexec will choose its default device)"
    else
        RESOLVED_BUILD_GPU_DEVICE="$norm"
        log_gpu_selection "Phase B build" "$RESOLVED_BUILD_GPU_DEVICE"
    fi
}

_suggest_build_profile_for_memory() {
    # Single source of truth for the coarse tiers lives in
    # lib/build_pipeline.sh (sourced via tools.sh); this wrapper only
    # survives for call-site compatibility.
    suggest_build_profile_from_memory "$@"
}

_resolve_build_profile_defaults() {
    local variant_csv="${1:-}"
    local mem_mb=0
    local mem_label="unknown"
    if [ -n "$TARGET_PROFILE" ]; then
        local profile_gpu=0
        if [[ "$RESOLVED_BUILD_GPU_DEVICE" =~ ^[0-9]+$ ]]; then
            profile_gpu="$RESOLVED_BUILD_GPU_DEVICE"
        fi
        mem_mb=$(target_profile_memory_mb "$TARGET_PROFILE" "$profile_gpu")
        [ -n "$mem_mb" ] || mem_mb=0
        mem_label="${mem_mb} MiB from target profile GPU ${profile_gpu}"
    elif [ "$RESOLVED_BUILD_GPU_DEVICE" != "all" ]; then
        mem_mb=$(gpu_total_memory_mb "$RESOLVED_BUILD_GPU_DEVICE")
        [ -n "$mem_mb" ] || mem_mb=0
        mem_label="${mem_mb} MiB on GPU ${RESOLVED_BUILD_GPU_DEVICE}"
    fi

    local suggested
    if [ -n "$variant_csv" ] && [ -d "$EXPORTED_DIR" ]; then
        local profile_input="${MAX_INPUT_LEN:-128}"
        local profile_seq="${MAX_SEQ_LEN:-512}"
        suggested=($(suggest_build_profile_from_exports \
            "$mem_mb" "$EXPORTED_DIR" "$variant_csv" \
            "$ENGINE_DTYPE" "$profile_input" "$profile_seq"))
        local profile_summary
        profile_summary=$(summarize_build_profile_from_exports \
            "$mem_mb" "$EXPORTED_DIR" "$variant_csv" \
            "$ENGINE_DTYPE" "$profile_input" "$profile_seq")
        [ -n "$profile_summary" ] && log_info "  Export-aware sizing: $profile_summary"
    else
        suggested=($(_suggest_build_profile_for_memory "$mem_mb"))
    fi
    [ -n "$MAX_BATCH_SIZE" ] || MAX_BATCH_SIZE="${suggested[0]}"
    [ -n "$MAX_INPUT_LEN" ] || MAX_INPUT_LEN="${suggested[1]}"
    [ -n "$MAX_SEQ_LEN" ] || MAX_SEQ_LEN="${suggested[2]}"

    _validate_positive_int "MAX_BATCH_SIZE" "$MAX_BATCH_SIZE"
    _validate_positive_int "MAX_INPUT_LEN" "$MAX_INPUT_LEN"
    _validate_positive_int "MAX_SEQ_LEN" "$MAX_SEQ_LEN"

    log_info "Build profile: max_batch=${MAX_BATCH_SIZE}, max_input=${MAX_INPUT_LEN}, max_seq=${MAX_SEQ_LEN}"
    log_info "  Default profile source: selected build GPU memory (${mem_label}) + exported model sizing; override with --max-batch-size/--max-input-len/--max-seq-len"
}

_detect_docker_gpu_args() {
    local image="$1"
    local err=""
    local docker_gpu_arg="all"
    local visible_devices="all"

    if [ -n "$RESOLVED_BUILD_GPU_DEVICE" ] && [ "$RESOLVED_BUILD_GPU_DEVICE" != "all" ]; then
        docker_gpu_arg="device=${RESOLVED_BUILD_GPU_DEVICE}"
        visible_devices="$RESOLVED_BUILD_GPU_DEVICE"
    fi

    if docker run --rm --gpus "$docker_gpu_arg" "$image" /bin/true >/dev/null 2>&1; then
        DOCKER_GPU_ARGS=(--gpus "$docker_gpu_arg")
        log_info "Docker GPU launch mode: --gpus ${docker_gpu_arg}"
        return 0
    fi

    err=$(docker run --rm \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES="$visible_devices" \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        "$image" /bin/true 2>&1) && {
        DOCKER_GPU_ARGS=(
            --runtime=nvidia
            -e NVIDIA_VISIBLE_DEVICES="$visible_devices"
            -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
        )
        log_warn "Docker GPU launch fallback enabled: --runtime=nvidia (NVIDIA_VISIBLE_DEVICES=${visible_devices})"
        return 0
    }

    log_error "Docker GPU smoke test failed for image: $image"
    if [ -n "$err" ]; then
        echo "$err" >&2
    fi
    return 1
}

# Talker dimensions for trtexec. Read from model config.json, else fallback.
# Output: H num_kv_heads head_dim num_layers
_get_talker_dims() {
    local variant="$1"

    # Variant name → model directory name
    local -A _VARIANT_DIR=(
        [base-0.6b]=Qwen3-TTS-12Hz-0.6B-Base
        [custom-0.6b]=Qwen3-TTS-12Hz-0.6B-CustomVoice
        [base-1.7b]=Qwen3-TTS-12Hz-1.7B-Base
        [custom-1.7b]=Qwen3-TTS-12Hz-1.7B-CustomVoice
        [design-1.7b]=Qwen3-TTS-12Hz-1.7B-VoiceDesign
    )
    local model_dir="${REPO_ROOT}/workspace/models/${_VARIANT_DIR[$variant]:-}"
    local cfg="$model_dir/config.json"

    if [ -f "$cfg" ]; then
        python3 -c "
import json, sys
c = json.load(open('$cfg'))
tc = c.get('talker_config', c)
h = tc.get('hidden_size', 1024)
nkv = tc.get('num_key_value_heads', 8)
head_dim = tc.get('head_dim', h // tc.get('num_attention_heads', 16))
nlayers = tc.get('num_hidden_layers', 28)
print(h, nkv, head_dim, nlayers)
" 2>/dev/null && return
    fi

    # Fallback: both 0.6b and 1.7b share head_dim=128, kv_heads=8, layers=28
    case "$variant" in
        base-0.6b|custom-0.6b)   echo "1024 8 128 28" ;;
        base-1.7b|custom-1.7b|design-1.7b) echo "2048 8 128 28" ;;
        *) echo "1024 8 128 28" ;;
    esac
}

# Build single Talker unified TRT engine (prefill + decode) from ONNX via trtexec.
# S_past min=0: prefill passes empty past_kv; decode uses real KV cache.
build_talker_unified_trt() {
    local variant="$1"
    local variant_dir="$EXPORTED_DIR/$variant"
    local unified_onnx="$variant_dir/talker_unified.onnx"

    if [ ! -f "$unified_onnx" ]; then
        log_error "Unified TRT: missing ONNX for $variant (need talker_unified.onnx)"
        log_error "  Run: python scripts/export/export_09_talker_code2wav_fused.py --variant $variant"
        return 1
    fi

    local dims
    dims=($(_get_talker_dims "$variant"))
    local H="${dims[0]:-1024}" KV_HEADS="${dims[1]:-2}" HEAD_DIM="${dims[2]:-64}" NUM_LAYERS="${dims[3]:-28}"

    log_step "Building Talker unified TRT engine: $variant (H=$H, kv_heads=$KV_HEADS, head_dim=$HEAD_DIM, layers=$NUM_LAYERS)"
    log_info "  Unified ONNX: $unified_onnx"
    log_info "  Image:       $NGC_IMAGE"

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run trtexec for talker_unified.engine"
        return 0
    fi

    # Single engine: min/opt/max for input_embeds, position_ids, and all past_kv_{i}_k/v
    # All tensors sharing the "batch" dim must have the same batch value per profile.
    # ONNX export shape: input_embeds [B,S,H], position_ids [B,3,S,1] (dim1=3 is the three token-type streams)
    # min: batch=1, S=1, S_past=0 (prefill, no history); opt: batch=1, S=1, S_past=128 (decode hot path); max: batch=B, S=512, S_past=4096
    local OPT_BATCH=1 OPT_S_PAST=128
    local unif_min="input_embeds:1x1x${H},position_ids:1x3x1x1"
    local unif_opt="input_embeds:${OPT_BATCH}x1x${H},position_ids:${OPT_BATCH}x3x1x1"
    local unif_max="input_embeds:${MAX_BATCH_SIZE}x${MAX_INPUT_LEN}x${H},position_ids:${MAX_BATCH_SIZE}x3x${MAX_INPUT_LEN}x1"
    local i=0
    while [ "$i" -lt "$NUM_LAYERS" ]; do
        unif_min="$unif_min,past_kv_${i}_k:1x${KV_HEADS}x0x${HEAD_DIM},past_kv_${i}_v:1x${KV_HEADS}x0x${HEAD_DIM}"
        unif_opt="$unif_opt,past_kv_${i}_k:${OPT_BATCH}x${KV_HEADS}x${OPT_S_PAST}x${HEAD_DIM},past_kv_${i}_v:${OPT_BATCH}x${KV_HEADS}x${OPT_S_PAST}x${HEAD_DIM}"
        unif_max="$unif_max,past_kv_${i}_k:${MAX_BATCH_SIZE}x${KV_HEADS}x${MAX_SEQ_LEN}x${HEAD_DIM},past_kv_${i}_v:${MAX_BATCH_SIZE}x${KV_HEADS}x${MAX_SEQ_LEN}x${HEAD_DIM}"
        i=$((i + 1))
    done

    # I/O formats: input_embeds, position_ids=int64, all KV (58 inputs, 60 outputs)
    local io_fmt
    io_fmt=$(_trtexec_io_format)
    local io_in="${io_fmt},int64:chw"
    local io_out="${io_fmt},int64:chw,${io_fmt},${io_fmt}"
    i=0
    while [ "$i" -lt "$NUM_LAYERS" ]; do
        io_in="$io_in,${io_fmt},${io_fmt}"
        io_out="$io_out,${io_fmt},${io_fmt}"
        i=$((i + 1))
    done

    local prec_flag
    prec_flag=$(_trtexec_precision_flags)
    log_info "Building talker_unified.engine (trtexec, ${ENGINE_DTYPE^^} I/O) ..."
    if ! _trtexec_run "$variant_dir" talker_unified.onnx talker_unified.engine -- \
            $prec_flag \
            --memPoolSize=workspace:8192 \
            --minShapes="$unif_min" \
            --optShapes="$unif_opt" \
            --maxShapes="$unif_max" \
            --inputIOFormats="$io_in" \
            --outputIOFormats="$io_out"; then
        log_error "trtexec talker_unified engine failed for $variant"
        return 1
    fi

    log_info "Talker unified TRT engine built: $variant_dir/talker_unified.engine"
    return 0
}

# Production: talker + code2wav fused ONNX (step 09).
build_talker_code2wav_fused_trt() {
    local variant="$1"
    local variant_dir="$EXPORTED_DIR/$variant"
    local onnx="$variant_dir/talker_code2wav_fused.onnx"
    if [ ! -f "$onnx" ]; then
        log_error "Missing $onnx — run: python scripts/export/export_09_talker_code2wav_fused.py --variant $variant"
        return 1
    fi
    local dims
    dims=($(_get_talker_dims "$variant"))
    local H="${dims[0]:-1024}" KV_HEADS="${dims[1]:-8}" HEAD_DIM="${dims[2]:-128}" NUM_LAYERS="${dims[3]:-28}"

    local profile_py="${REPO_ROOT}/scripts/python/trt_fused_talk_c2w_profiles.py"
    if [ ! -f "$profile_py" ]; then
        log_error "Missing $profile_py"
        return 1
    fi
    local n_c2w=8
    local n_cp=15
    if [ -f "$variant_dir/triton_manifest.json" ]; then
        n_c2w=$(python3 -c "import json; d=json.load(open('$variant_dir/triton_manifest.json')); print(int(d['code2wav_fused']['num_code2wav_hidden_layers']))")
        n_cp=$(python3 -c "import json; d=json.load(open('$variant_dir/triton_manifest.json')); print(int(d.get('architecture', {}).get('cp_num_stages', 15)))")
    fi
    local fused_min fused_opt fused_max fused_dec_min fused_dec_opt fused_dec_max
    local profile_out
    profile_out=$(python3 "$profile_py" "$H" "$KV_HEADS" "$HEAD_DIM" "$NUM_LAYERS" "$MAX_BATCH_SIZE" "$MAX_INPUT_LEN" "$MAX_SEQ_LEN" "$n_c2w" "$n_cp")
    fused_min=$(sed -n '1p' <<<"$profile_out")
    fused_opt=$(sed -n '2p' <<<"$profile_out")
    fused_max=$(sed -n '3p' <<<"$profile_out")
    # Profile 1 (decode-only, seq=1): small activation scratch so the runtime
    # CUDA-graph decode path can afford a dedicated execution context.
    fused_dec_min=$(sed -n '4p' <<<"$profile_out")
    fused_dec_opt=$(sed -n '5p' <<<"$profile_out")
    fused_dec_max=$(sed -n '6p' <<<"$profile_out")

    log_step "Building talker_code2wav_fused.engine: $variant"
    if $DRY_RUN; then
        log_info "[DRY RUN] trtexec talker_code2wav_fused"
        return 0
    fi
    local prec_flag
    local fused_io_py="${REPO_ROOT}/scripts/python/trt_fused_io_formats.py"
    local mf="$variant_dir/triton_manifest.json"
    local fused_io_in="" fused_io_out=""
    # Mixed-precision: when backbone/cp/code2wav differ, the helper emits a
    # --layerPrecisions wildcard spec (empty for a uniform engine).
    local -a mixed_args=()
    if [ -f "$mf" ] && [ -f "$fused_io_py" ]; then
        fused_io_in=$(python3 "$fused_io_py" "$mf" --emit input)
        fused_io_out=$(python3 "$fused_io_py" "$mf" --emit output)
        prec_flag=$(python3 "$fused_io_py" "$mf" --emit prec)
        local layer_prec
        layer_prec=$(python3 "$fused_io_py" "$mf" --emit layer-precisions)
        if [ -n "$layer_prec" ]; then
            # Constraint mode is auto-derived from the manifest precisions:
            #   prefer — when a speed pin (fp16/fp8 below a higher-precision
            #     backbone, e.g. code2wav=fp16) is present: its Pad/Slice glue
            #     has no fp16 Myelin kernel and must fall back instead of the
            #     build hard-failing.
            #   obey — uniform / higher-precision repro pins (e.g. cp=fp32),
            #     where every pinned layer has a conforming kernel and a
            #     missing one should hard-fail loudly.
            # PRECISION_CONSTRAINTS overrides the auto choice when set.
            local auto_constraints
            auto_constraints=$(python3 "$fused_io_py" "$mf" --emit constraints)
            local prec_constraints="${PRECISION_CONSTRAINTS:-${auto_constraints:-obey}}"
            mixed_args=(--precisionConstraints="$prec_constraints" --layerPrecisions="$layer_prec")
            log_info "  mixed precision: global=[$prec_flag] constraints=$prec_constraints layerPrecisions=$layer_prec"
        fi
        log_info "  triton_manifest: engine_dtype + triton_io_float_dtype drive trtexec precision and I/O formats"
    else
        if [ ! -f "$mf" ]; then
            log_warn "  No triton_manifest.json — using ENGINE_DTYPE for precision only; fused I/O formats omitted"
        else
            log_warn "  Missing $fused_io_py — using ENGINE_DTYPE only"
        fi
        prec_flag=$(_trtexec_precision_flags)
    fi
    # Profile 0: prefill + decode (as before).  Profile 1: decode-only —
    # the runtime CUDA-graph path binds its dedicated execution context to
    # it because its activation scratch is ~6x smaller than profile 0's.
    local -a profile_args=(
        --profile=0
        --minShapes="$fused_min"
        --optShapes="$fused_opt"
        --maxShapes="$fused_max"
    )
    if [ -n "$fused_dec_min" ]; then
        profile_args+=(
            --profile=1
            --minShapes="$fused_dec_min"
            --optShapes="$fused_dec_opt"
            --maxShapes="$fused_dec_max"
        )
    fi
    if [ -n "$fused_io_in" ] && [ -n "$fused_io_out" ]; then
        if ! _trtexec_run "$variant_dir" talker_code2wav_fused.onnx talker_code2wav_fused.engine -- \
            $prec_flag \
            "${mixed_args[@]}" \
            --inputIOFormats="$fused_io_in" \
            --outputIOFormats="$fused_io_out" \
            --memPoolSize=workspace:8192 \
            "${profile_args[@]}"; then
            log_error "trtexec talker_code2wav_fused failed for $variant"
            return 1
        fi
    else
        if ! _trtexec_run "$variant_dir" talker_code2wav_fused.onnx talker_code2wav_fused.engine -- \
            $prec_flag \
            "${mixed_args[@]}" \
            --memPoolSize=workspace:8192 \
            "${profile_args[@]}"; then
            log_error "trtexec talker_code2wav_fused failed for $variant"
            return 1
        fi
    fi
    log_info "talker_code2wav_fused.engine built: $variant_dir"
    return 0
}

# Production (base ICL): speech_tokenizer_codec_fused in variant dir.
build_speech_tokenizer_codec_fused_trt() {
    local variant="$1"
    local variant_dir="$EXPORTED_DIR/$variant"
    local onnx="$variant_dir/speech_tokenizer_codec_fused.onnx"
    [ -f "$onnx" ] || return 0
    log_step "Building speech_tokenizer_codec_fused.engine: $variant"
    if $DRY_RUN; then
        log_info "[DRY RUN] trtexec speech_tokenizer_codec_fused"
        return 0
    fi
    if ! _trtexec_run "$variant_dir" speech_tokenizer_codec_fused.onnx speech_tokenizer_codec_fused.engine -- \
        --minShapes=waveform:1x1x960 \
        --optShapes=waveform:1x1x48000 \
        --maxShapes=waveform:1x1x192000 \
        --memPoolSize=workspace:6144; then
        log_error "trtexec speech_tokenizer_codec_fused failed for $variant"
        return 1
    fi
    log_info "speech_tokenizer_codec_fused.engine built: $variant_dir"
    return 0
}

# Build peripheral TRT engines (speaker_encoder, speech_tokenizer_encoder, code2wav_decoder)
build_peripheral_engines() {
    if $DRY_RUN; then
        log_info "[DRY RUN] Would run trtexec for speech_tokenizer_encoder, code2wav_decoder"
        return 0
    fi
    local image="$1"
    local failed=0

    # Speech tokenizer encoder (shared, in tokenizer dir)
    # Min waveform = 960 samples (1 codec frame at stride 8×6×5×4=960, 24kHz → 40ms).
    # Max waveform = 192000 samples (8s @ 24kHz). Smaller values produce 0-length intermediates that TRT cannot handle.
    # speech_tokenizer_encoder: output audio_codes is Int64 (discrete codes). Must NOT use
    # --outputIOFormats/--bf16 which would override it. Use default fp32 for this small model.
    # maxShapes 192000 = 8s @24kHz. With 8s cap, workspace ~1.7GB; 6GB leaves margin.
    if [ -f "$TOKENIZER_DIR/speech_tokenizer_encoder.onnx" ]; then
        log_info "Building speech_tokenizer_encoder.engine ..."
        if ! _trtexec_run "$TOKENIZER_DIR" speech_tokenizer_encoder.onnx speech_tokenizer_encoder.engine -- \
            --minShapes=waveform:1x1x960 \
            --optShapes=waveform:1x1x48000 \
            --maxShapes=waveform:1x1x192000 \
            --memPoolSize=workspace:6144; then
            log_error "speech_tokenizer_encoder trtexec failed"
            failed=$((failed + 1))
        fi
    else
        log_info "speech_tokenizer_encoder.onnx not found, skipping"
    fi

    # Code2Wav decoder (stateful streaming, chunk_T=4 fixed; 39 inputs, 38 outputs)
    if [ -f "$TOKENIZER_DIR/code2wav_decoder.onnx" ]; then
        log_info "Building code2wav_decoder.engine (streaming, chunk_T=4) ..."
        C2W_BATCH="${MAX_BATCH_SIZE:-8}"
        # Dynamic: codes (batch), cache_position [B,4], c2w_attention_bias [B,1,4,past+4],
        # past_kv_* (batch + past_len 1..72), conv/transconv states (batch).
        C2W_MIN="codes:1x16x4,cache_position:1x4,c2w_attention_bias:1x1x4x5"
        C2W_OPT="codes:1x16x4,cache_position:1x4,c2w_attention_bias:1x1x4x8"
        C2W_MAX="codes:${C2W_BATCH}x16x4,cache_position:${C2W_BATCH}x4,c2w_attention_bias:${C2W_BATCH}x1x4x76"
        for i in 0 1 2 3 4 5 6 7; do
            C2W_MIN="${C2W_MIN},past_kv_${i}_k:1x16x1x64,past_kv_${i}_v:1x16x1x64"
            C2W_OPT="${C2W_OPT},past_kv_${i}_k:1x16x4x64,past_kv_${i}_v:1x16x4x64"
            C2W_MAX="${C2W_MAX},past_kv_${i}_k:${C2W_BATCH}x16x72x64,past_kv_${i}_v:${C2W_BATCH}x16x72x64"
        done
        # Conv/transconv states: batch dynamic, fixed time dims (from plan)
        for name in conv_state_0:1x512x2 conv_state_1:1x1024x6 conv_state_2:1x1024x6 conv_state_3:1x1024x6 \
            conv_state_4:1x768x6 conv_state_5:1x768x18 conv_state_6:1x768x54 conv_state_7:1x384x6 conv_state_8:1x384x18 conv_state_9:1x384x54 \
            conv_state_10:1x192x6 conv_state_11:1x192x18 conv_state_12:1x192x54 conv_state_13:1x96x6 conv_state_14:1x96x18 conv_state_15:1x96x54 conv_state_16:1x96x6 \
            transconv_overlap_0:1x768x8 transconv_overlap_1:1x384x5 transconv_overlap_2:1x192x4 transconv_overlap_3:1x96x3; do
            n="${name%%:*}"
            s="${name#*:}"
            C2W_MIN="${C2W_MIN},${n}:${s}"
            C2W_OPT="${C2W_OPT},${n}:${s}"
            C2W_MAX="${C2W_MAX},${n}:${C2W_BATCH}x${s#1x}"
        done
        # Input order: codes (int64), cache_position (fp32), c2w_attention_bias (float), then 37 state float tensors.
        # Output order: 38 float tensors (wav + present_kv + new_conv_state + new_transconv_overlap).
        # Do NOT use generic bf16:chw for all inputs — codes stays int64 and cache_position stays fp32.
        local io_fmt
        io_fmt=$(_trtexec_io_format)
        local c2w_io_in="int64:chw,fp32:chw,${io_fmt}"
        local i=0
        while [ $i -lt 37 ]; do c2w_io_in="${c2w_io_in},${io_fmt}"; i=$((i+1)); done
        local c2w_io_out=""
        i=0
        while [ $i -lt 38 ]; do
            [ $i -gt 0 ] && c2w_io_out="${c2w_io_out},"
            c2w_io_out="${c2w_io_out}${io_fmt}"
            i=$((i+1))
        done
        # Force I/O type to match engine precision; Triton config must match.
        if ! _trtexec_run "$TOKENIZER_DIR" code2wav_decoder.onnx code2wav_decoder.engine -- \
            $(_trtexec_precision_flags) \
            --inputIOFormats="$c2w_io_in" \
            --outputIOFormats="$c2w_io_out" \
            --minShapes="$C2W_MIN" \
            --optShapes="$C2W_OPT" \
            --maxShapes="$C2W_MAX" \
            --memPoolSize=workspace:4096; then
            log_error "code2wav_decoder trtexec failed"
            failed=$((failed + 1))
        fi
    else
        log_warn "code2wav_decoder.onnx not found, skipping (export step 06 / verification)"
    fi

    [ "$failed" -eq 0 ]
}

# Discover variants that have production fused talker+code2wav ONNX
discover_trt_variants() {
    local found=()
    for vdir in "$EXPORTED_DIR"/*/; do
        [ -d "$vdir" ] || continue
        if [ -f "${vdir}talker_code2wav_fused.onnx" ]; then
            found+=("$(basename "$vdir")")
        fi
    done
    echo "${found[@]}"
}

update_variant_manifest_profile() {
    local variant="$1"
    local mark_built="${2:-false}"
    local manifest="$EXPORTED_DIR/$variant/triton_manifest.json"
    local update_py="${REPO_ROOT}/scripts/python/update_triton_manifest_profile.py"

    if [ ! -f "$manifest" ]; then
        log_warn "No triton_manifest.json for $variant; cannot record engine profile"
        return 0
    fi
    if [ ! -f "$update_py" ]; then
        log_warn "Missing update helper: $update_py"
        return 0
    fi

    local args=(
        --manifest "$manifest"
        --engine-mode trt
        --engine-dtype "$ENGINE_DTYPE"
        --triton-io-float-dtype "$TRITON_IO_FLOAT_DTYPE"
        --max-batch-size "$MAX_BATCH_SIZE"
    )
    [ -n "$BACKBONE_PRECISION" ] && args+=(--backbone-precision "$BACKBONE_PRECISION")
    [ -n "$CP_PRECISION" ] && args+=(--cp-precision "$CP_PRECISION")
    [ -n "$CODE2WAV_PRECISION" ] && args+=(--code2wav-precision "$CODE2WAV_PRECISION")
    args+=(
        --max-input-len "$MAX_INPUT_LEN"
        --max-seq-len "$MAX_SEQ_LEN"
        --builder-image "$NGC_IMAGE"
        --target-driver "$TARGET_DRIVER"
    )
    if [ -n "${NGC_TAG:-}" ]; then
        args+=(--ngc-tag "$NGC_TAG")
        local _trt_ver
        _trt_ver=$(resolve_ngc_tag_tensorrt_version "$NGC_TAG" 2>/dev/null || true)
        [ -n "$_trt_ver" ] && args+=(--tensorrt-version "$_trt_ver")
    fi
    if [ -n "$TARGET_PROFILE" ]; then
        args+=(--target-profile "$TARGET_PROFILE")
    fi
    if [ "$mark_built" != "true" ]; then
        args+=(--skip-built-at)
    fi

    python3 "$update_py" "${args[@]}"
}

# Speaker engines for every variant directory that has speaker_encoder.onnx
build_speaker_encoders_all() {
    local image="$1"
    if $DRY_RUN; then
        log_info "[DRY RUN] Would build speaker_encoder engines"
        return 0
    fi
    local failed=0
    for vdir in "$EXPORTED_DIR"/*/; do
        [ -d "$vdir" ] || continue
        local onnx="${vdir}speaker_encoder.onnx"
        [ -f "$onnx" ] || continue
        local variant_name
        variant_name=$(basename "$vdir")
        log_info "Building speaker_encoder.engine for $variant_name ..."
        if ! _trtexec_run "$vdir" speaker_encoder.onnx speaker_encoder.engine -- \
            $(_trtexec_precision_flags) \
            --inputIOFormats=$(_trtexec_io_format) \
            --outputIOFormats=$(_trtexec_io_format) \
            --minShapes=mel:1x1x128 \
            --optShapes=mel:1x300x128 \
            --maxShapes=mel:${MAX_BATCH_SIZE}x1000x128 \
            --memPoolSize=workspace:1024; then
            log_error "speaker_encoder trtexec failed for $variant_name"
            failed=$((failed + 1))
        fi
    done
    [ "$failed" -eq 0 ]
}

# ── Argument parsing ──
if [[ $# -gt 0 ]]; then
    case "$1" in
        make-bundle|import-artifact|remote-build|build)
            COMMAND="$1"
            shift
            ;;
    esac
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)        VARIANT="$2"; shift 2 ;;
        --mode)           BUILD_MODE="$2"; shift 2 ;;
        --image)          USER_IMAGE="$2"; shift 2 ;;
        --device|--build-device) BUILD_GPU_DEVICE="$2"; shift 2 ;;
        --target-profile) TARGET_PROFILE="$2"; shift 2 ;;
        --out)            BUNDLE_OUT="$2"; shift 2 ;;
        --remote-host)    REMOTE_HOST="$2"; shift 2 ;;
        --remote-workdir) REMOTE_WORKDIR="$2"; shift 2 ;;
        --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max-input-len)  MAX_INPUT_LEN="$2"; shift 2 ;;
        --max-seq-len)    MAX_SEQ_LEN="$2"; shift 2 ;;
        --dtype)          ENGINE_DTYPE="$2"; shift 2 ;;
        --triton-io-float-dtype) TRITON_IO_FLOAT_DTYPE="$2"; shift 2 ;;
        --backbone-precision) BACKBONE_PRECISION="$2"; shift 2 ;;
        --cp-precision)   CP_PRECISION="$2"; shift 2 ;;
        --code2wav-precision) CODE2WAV_PRECISION="$2"; shift 2 ;;
        --target-driver)  TARGET_DRIVER="$2"; export TARGET_DRIVER; shift 2 ;;
        --allow-fingerprint-mismatch) ALLOW_FINGERPRINT_MISMATCH=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --pull-only)      PULL_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [build|make-bundle|remote-build|import-artifact] [options]"
            echo ""
            echo "Options:"
            echo "  --variant <name>       Build for a specific model variant"
            echo "  --mode local|bundle|remote-ssh  Cross-host mode (default: local)"
            echo "  --target-profile <json> Target machine profile from probe_target.sh"
            echo "  --out <bundle>         Output build bundle for make-bundle"
            echo "  --remote-host <host>   SSH host for remote-build / --mode remote-ssh"
            echo "  --remote-workdir <dir> Remote work dir (default: /tmp/qwen3-tts-engine-build)"
            echo "  --image <uri>          Override NGC container image (default: auto-detect)"
            echo "  --target-driver <ver>  Target NVIDIA driver for NGC container selection"
            echo "  --device N|auto|all    Build GPU device (default: auto; aliases: --build-device)"
            echo "  --max-batch-size N     Max batch size (default: auto by exported model + target GPU memory)"
            echo "  --max-input-len N      Max input length for prefill (default: auto by exported model + target GPU memory)"
            echo "  --max-seq-len N        Max sequence length incl. KV cache (default: auto by exported model + target GPU memory)"
            echo "  --dtype bf16|fp16|fp32|fp8  Engine precision (default: bfloat16)"
            echo "  --backbone-precision T      Talker backbone precision (default: follow --dtype)"
            echo "  --cp-precision T            Code Predictor precision (default: follow --dtype, i.e. bf16)"
            echo "  --code2wav-precision T      Code2Wav precision (default: fp16 — faster conv; bf16 for uniform build)"
            echo "  --triton-io-float-dtype T   Float I/O dtype (default: same as --dtype)"
            echo "  --dry-run              Show docker commands without executing"
            echo "  --pull-only            Pull the container image and exit"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *)
            if [ "$COMMAND" = "import-artifact" ] && [ -z "$ARTIFACT_IN" ]; then
                ARTIFACT_IN="$1"; shift
            else
                log_error "Unknown argument: $1"; exit 1
            fi
            ;;
    esac
done

normalize_build_dtypes

case "$BUILD_MODE" in
    local|bundle|remote-ssh) ;;
    *) log_error "Unknown build mode: $BUILD_MODE"; exit 1 ;;
esac
if [ "$BUILD_MODE" = "bundle" ]; then
    COMMAND="make-bundle"
elif [ "$BUILD_MODE" = "remote-ssh" ]; then
    COMMAND="remote-build"
fi

# ===========================================================================
#  Phase B: TRT engine build (trtexec from ONNX)
# ===========================================================================

log_step "Phase B: TensorRT Engine Build (trtexec)"

if [ "$COMMAND" = "import-artifact" ]; then
    [ -n "$ARTIFACT_IN" ] || { log_error "Missing artifact bundle path"; exit 1; }
    strict=true
    $ALLOW_FINGERPRINT_MISMATCH && strict=false
    extract_engine_artifact_bundle "$REPO_ROOT" "$EXPORTED_DIR" "$ARTIFACT_IN" "$strict"
    exit 0
fi

if [ -n "$TARGET_PROFILE" ] && [ -z "$TARGET_DRIVER" ]; then
    TARGET_DRIVER=$(cross_host_json_value "$TARGET_PROFILE" driver_version || true)
    [ -n "$TARGET_DRIVER" ] && export TARGET_DRIVER
fi

if [ "$COMMAND" = "make-bundle" ] || [ "$COMMAND" = "remote-build" ]; then
    [ -n "$TARGET_PROFILE" ] || { log_error "--target-profile is required for $COMMAND"; exit 1; }
    norm_device=$(normalize_gpu_device "$BUILD_GPU_DEVICE") || exit 1
    if [ "$norm_device" = "auto" ] || [ "$norm_device" = "all" ]; then
        RESOLVED_BUILD_GPU_DEVICE="0"
    else
        RESOLVED_BUILD_GPU_DEVICE="$norm_device"
    fi
    log_info "Target build GPU from profile: ${RESOLVED_BUILD_GPU_DEVICE}"
else
    # Build mode: resolve runner first so we know whether to expect docker.
    _resolved_runner=$(resolve_trtexec_runner) || exit 1
    log_info "Engine build runner: $_resolved_runner"
    if [ "$_resolved_runner" = "docker" ]; then
        check_docker_gpu_ready || exit 1
    fi
    _resolve_build_gpu_device
fi

if [ -n "$TARGET_PROFILE" ] && [ -z "$USER_IMAGE" ]; then
    NGC_TAG="$(resolve_ngc_tag_from_profile "$TARGET_PROFILE")" || exit 1
    export NGC_TAG
    NGC_IMAGE=$(resolve_ngc_image_from_tag "$NGC_TAG") || exit 1
    log_info "Using NGC image from target profile: $NGC_IMAGE"
elif [ -n "$USER_IMAGE" ]; then
    NGC_IMAGE="$USER_IMAGE"
    log_info "Using user-specified image: $NGC_IMAGE"
else
    # Sync NGC compatibility matrix from NVIDIA website (best-effort; skip if offline)
    if $DRY_RUN; then
        log_info "[DRY RUN] Skipping NGC matrix auto-update"
    elif [[ -z "${NGC_SKIP_MATRIX_UPDATE:-}" ]]; then
        source "${SCRIPT_DIR}/lib/ngc_updater.sh" 2>/dev/null || true
        update_ngc_matrix "${SCRIPT_DIR}/ngc_matrix.conf" 2>/dev/null || true
    fi
    # docker manifest probing only makes sense when docker is the runner.
    if [ "${_resolved_runner:-docker}" = "docker" ]; then
        _NGC_VERIFY_MANIFEST=1
    fi
    NGC_IMAGE=$(resolve_ngc_image_info) \
        || { log_error "Cannot determine NGC container. Use --image."; exit 1; }
fi
export NGC_IMAGE
# Derive NGC_TAG from the resolved image (e.g. nvcr.io/...:25.10-py3 -> 25.10)
# so the manifest records engine_profile.ngc_tag.  Deploy (compose.sh) reads it
# to pick a TRT-matched engine base image; without it the engine container would
# default to a possibly-mismatched TensorRT release and fail to load the plan.
if [ -z "${NGC_TAG:-}" ] && [[ "$NGC_IMAGE" =~ :([0-9]{2}\.[0-9]{2})-py3 ]]; then
    NGC_TAG="${BASH_REMATCH[1]}"
    export NGC_TAG
fi
if [ "$COMMAND" != "make-bundle" ] && [ "$COMMAND" != "remote-build" ]; then
    if [ "${_resolved_runner:-docker}" = "docker" ]; then
        ensure_ngc_image "$NGC_IMAGE" || exit 1
        _detect_docker_gpu_args "$NGC_IMAGE" || exit 1
    else
        log_info "Skipping docker image pull (runner=$_resolved_runner; using host trtexec)"
    fi
fi

if $PULL_ONLY; then
    log_info "Image ready. Re-run without --pull-only to build engines."
    exit 0
fi

if [ ! -d "$EXPORTED_DIR" ]; then
    log_error "No exported models at: $EXPORTED_DIR"
    log_error "Run Phase A first: autorun.sh setup or export_all.py"
    exit 1
fi

if [[ -n "$VARIANT" && "$VARIANT" != all* ]]; then
    if [ ! -f "$EXPORTED_DIR/$VARIANT/talker_code2wav_fused.onnx" ]; then
        log_error "Missing talker_code2wav_fused.onnx for $VARIANT. Run export_09_talker_code2wav_fused.py"
        exit 1
    fi
    VARIANTS=("$VARIANT")
else
    read -ra VARIANTS <<< "$(discover_trt_variants)"
    if [ ${#VARIANTS[@]} -eq 0 ]; then
        log_error "No talker_code2wav_fused.onnx in $EXPORTED_DIR (run export_all.py or export_09 per variant)"
        exit 1
    fi
    log_info "Discovered variants: ${VARIANTS[*]}"
fi

variant_csv=$(IFS=,; echo "${VARIANTS[*]}")
_resolve_build_profile_defaults "$variant_csv"

if [ "$COMMAND" = "make-bundle" ]; then
    if $DRY_RUN; then
        log_info "[DRY RUN] Would create build bundle: $BUNDLE_OUT"
        log_info "  Variants: $variant_csv"
        log_info "  Target profile: $TARGET_PROFILE"
        exit 0
    fi
    for variant in "${VARIANTS[@]}"; do
        update_variant_manifest_profile "$variant" false
    done
    make_engine_build_bundle "$REPO_ROOT" "$EXPORTED_DIR" "$BUNDLE_OUT" "$TARGET_PROFILE" \
        "$variant_csv" "$ENGINE_DTYPE" "$TRITON_IO_FLOAT_DTYPE" \
        "$MAX_BATCH_SIZE" "$MAX_INPUT_LEN" "$MAX_SEQ_LEN" "$RESOLVED_BUILD_GPU_DEVICE"
    exit 0
fi

if [ "$COMMAND" = "remote-build" ]; then
    [ -n "$REMOTE_HOST" ] || { log_error "--remote-host is required for remote-build"; exit 1; }
    tmp_bundle="${REPO_ROOT}/workspace/engine_build_bundle.remote.tar.zst"
    artifact_local="${REPO_ROOT}/workspace/engine_artifact_bundle.remote.tar.zst"
    for variant in "${VARIANTS[@]}"; do
        update_variant_manifest_profile "$variant" false
    done
    make_engine_build_bundle "$REPO_ROOT" "$EXPORTED_DIR" "$tmp_bundle" "$TARGET_PROFILE" \
        "$variant_csv" "$ENGINE_DTYPE" "$TRITON_IO_FLOAT_DTYPE" \
        "$MAX_BATCH_SIZE" "$MAX_INPUT_LEN" "$MAX_SEQ_LEN" "$RESOLVED_BUILD_GPU_DEVICE"
    if $DRY_RUN; then
        log_info "[DRY RUN] Would upload/run/download via SSH host: $REMOTE_HOST"
        exit 0
    fi
    ssh "$REMOTE_HOST" "rm -rf '$REMOTE_WORKDIR' && mkdir -p '$REMOTE_WORKDIR'"
    scp "$tmp_bundle" "$REMOTE_HOST:$REMOTE_WORKDIR/engine_build_bundle.tar.zst"
    ssh "$REMOTE_HOST" "cd '$REMOTE_WORKDIR' && (tar --zstd -xf engine_build_bundle.tar.zst 2>/dev/null || tar -z -xf engine_build_bundle.tar.zst) && bash build_on_target.sh"
    scp "$REMOTE_HOST:$REMOTE_WORKDIR/engine_artifact_bundle.tar.zst" "$artifact_local"
    extract_engine_artifact_bundle "$REPO_ROOT" "$EXPORTED_DIR" "$artifact_local" "true"
    exit 0
fi

FAILED=0
SUCCEEDED=0

if ! $DRY_RUN; then
    for variant in "${VARIANTS[@]}"; do
        update_variant_manifest_profile "$variant" false
    done
fi

if ! build_speaker_encoders_all "$NGC_IMAGE"; then
    FAILED=$((FAILED + 1))
fi

for variant in "${VARIANTS[@]}"; do
    if [[ "$variant" == base-* ]] && [ -f "$EXPORTED_DIR/$variant/speech_tokenizer_codec_fused.onnx" ]; then
        build_speech_tokenizer_codec_fused_trt "$variant" || FAILED=$((FAILED + 1))
    fi
done

for variant in "${VARIANTS[@]}"; do
    if build_talker_code2wav_fused_trt "$variant"; then
        if ! $DRY_RUN; then
            update_variant_manifest_profile "$variant" true
        fi
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

if [ "${BUILD_VERIFICATION_ENGINES:-0}" = "1" ]; then
    log_step "BUILD_VERIFICATION_ENGINES=1: talker_unified + tokenizer ONNX engines"
    for variant in "${VARIANTS[@]}"; do
        build_talker_unified_trt "$variant" || FAILED=$((FAILED + 1))
    done
    if ! build_peripheral_engines "$NGC_IMAGE"; then
        FAILED=$((FAILED + 1))
    fi
fi

echo ""
log_step "Engine Build Summary"
log_info "  Image:     $NGC_IMAGE"
log_info "  Succeeded: $SUCCEEDED (talker_code2wav_fused variants)"
[ "$FAILED" -gt 0 ] && log_error "  Failed: $FAILED"

if [ "$SUCCEEDED" -gt 0 ] && ! $DRY_RUN; then
    echo "$ENGINE_DTYPE" > "$EXPORTED_DIR/.engine_dtype"
    log_info "Saved ENGINE_DTYPE=$ENGINE_DTYPE to $EXPORTED_DIR/.engine_dtype"
    echo ""
    log_info "Engines: $EXPORTED_DIR/<variant>/*.engine, $TOKENIZER_DIR/*.engine"
    log_info "Next: bash scripts/bash/build_triton.sh assemble --engine-mode trt && build_triton.sh run"
fi

exit "$FAILED"
