#!/bin/bash
# ===========================================================================
#  triton.sh — Triton Inference Server deployment helpers (Phase C)
#
#  Functions: assemble_model_repo, validate_model_repo,
#             build_triton_image, resolve_triton_deploy_image,
#             triton_run, triton_health_check
#  Depends:   lib/logging.sh, lib/utils.sh, lib/docker.sh
#
#  Assembles Phase A/B exported artifacts into a Triton model_repository
#  and manages the deployment container lifecycle.
# ===========================================================================

[[ -n "${_LIB_TRITON_LOADED:-}" ]] && return 0
_LIB_TRITON_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"
source "${_LIB_DIR}/docker.sh"

# Default production layout: only the Python orchestrator is exposed to Triton.
# Runtime assets (fused engine, optional voice-clone ONNX exports, manifest) live
# under tts_orchestrator/<version>/runtime. ASSEMBLE_VERIFICATION_MODELS=1 can still add
# standalone verification models such as talker_unified/code2wav.
_TRITON_REQUIRED_MODELS=(
    "tts_orchestrator"
    "tts_orchestrator_http"
)

# Default Triton gRPC / HTTP / metrics ports
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8002}"

# ---------------------------------------------------------------------------
#  resolve_triton_deploy_image [driver_version]
#  Returns the Triton deployment image for Phase C.
#  Uses the deploy image (py3 + torch/tokenizers) built by
#  build_triton_deploy_image().  Falls back to raw -py3 if deploy
#  image is not yet built.
#  Echoes the full image tag to stdout.
# ---------------------------------------------------------------------------
resolve_triton_deploy_image() {
    local ngc_tag
    if [ -n "${NGC_TAG:-}" ]; then
        ngc_tag=$(resolve_ngc_tag "$@") \
            || { log_error "Cannot determine NGC tag"; return 1; }
    else
        ngc_tag=$(resolve_manifest_ngc_tag "${MODEL_REPO_DIR:-}" "${MODEL_VERSION:-${ENGINE_MODEL_VERSION:-1}}" 2>/dev/null || true)
        if [ -z "$ngc_tag" ]; then
            ngc_tag=$(resolve_ngc_tag "$@") \
                || { log_error "Cannot determine NGC tag"; return 1; }
        else
            log_info "Using Triton image from model manifest (builder_image tag): $ngc_tag"
        fi
    fi

    local deploy_tag="${_DEPLOY_IMAGE_NAME}:${ngc_tag}"
    if docker image inspect "$deploy_tag" &>/dev/null; then
        log_info "Using Triton deploy image: $deploy_tag"
        echo "$deploy_tag"
        return 0
    fi

    local image="${_NGC_TRITON_BASE}:${ngc_tag}${_NGC_PY3_SUFFIX}"
    log_warn "Deploy image not found ($deploy_tag), falling back to: $image"
    log_warn "Build it first: bash scripts/bash/build_triton.sh build-image"
    echo "$image"
}

# ---------------------------------------------------------------------------
#  _link_or_copy <src> <dst>
#  Copies src to dst.  Always uses cp (not symlinks) because the assembled
#  model_repository is mounted into Docker containers where host-absolute
#  symlinks would be dangling.
# ---------------------------------------------------------------------------
_link_or_copy() {
    local src="$1" dst="$2"
    cp -a "$src" "$dst"
}

# ---------------------------------------------------------------------------
#  model_package_resources_stale <repo_root> <model_package_dir>
#  Returns 0 when repo_root/resources should be re-copied into the package.
# ---------------------------------------------------------------------------
model_package_resources_stale() {
    local repo_root="$1"
    local package_dir="$2"
    python3 - "$repo_root/resources" "$package_dir/resources" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

if not src.exists():
    sys.exit(0 if dst.exists() else 1)
if not src.is_dir():
    sys.exit(1)
if not dst.is_dir():
    sys.exit(0)

for path in src.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(src)
    other = dst / rel
    if not other.is_file():
        sys.exit(0)
    try:
        src_stat = path.stat()
        dst_stat = other.stat()
    except OSError:
        sys.exit(0)
    if src_stat.st_size != dst_stat.st_size or src_stat.st_mtime_ns > dst_stat.st_mtime_ns:
        sys.exit(0)

for path in dst.rglob("*"):
    if path.is_file() and not (src / path.relative_to(dst)).is_file():
        sys.exit(0)

sys.exit(1)
PY
}

# ---------------------------------------------------------------------------
#  model_package_engine_payload_stale <repo_root> <model_package_dir>
#  Returns 0 when repo_root/engine should be re-copied into the package.
# ---------------------------------------------------------------------------
model_package_engine_payload_stale() {
    local repo_root="$1"
    local package_dir="$2"
    python3 - "$repo_root/engine" "$package_dir/engine" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

def ignored(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}

if not src.is_dir():
    sys.exit(1)
if not dst.is_dir():
    sys.exit(0)

for path in src.rglob("*"):
    if not path.is_file() or ignored(path):
        continue
    rel = path.relative_to(src)
    other = dst / rel
    if not other.is_file():
        sys.exit(0)
    try:
        src_stat = path.stat()
        dst_stat = other.stat()
    except OSError:
        sys.exit(0)
    if src_stat.st_size != dst_stat.st_size or src_stat.st_mtime_ns > dst_stat.st_mtime_ns:
        sys.exit(0)

for path in dst.rglob("*"):
    if ignored(path):
        continue
    if path.is_file() and not (src / path.relative_to(dst)).is_file():
        sys.exit(0)

sys.exit(1)
PY
}

# ---------------------------------------------------------------------------
#  assemble_model_repo <exported_dir> <variant> <model_repo_dir> [engine_mode] [model_version]
#
#  engine_mode: trt (default) | onnx
#  TRT mode:  copies .engine → model.plan, backend tensorrt
#  ONNX mode: copies .onnx → model.onnx, backend onnxruntime
#
#  Default layout: only tts_orchestrator is exposed to Triton. The fused runtime
#  engine and optional voice-clone ONNX exports are copied into
#  tts_orchestrator/<version>/runtime/. ASSEMBLE_VERIFICATION_MODELS=1 adds standalone
#  speech_tokenizer_encoder, talker_unified, code2wav models for debugging.
# ---------------------------------------------------------------------------
assemble_model_repo() {
    local exported_dir="$1"
    local variant="$2"
    local repo_dir="$3"
    local engine_mode="${4:-trt}"
    local model_version
    model_version=$(resolve_model_version "${5:-}") || return 1

    local variant_dir="$exported_dir/$variant"
    local tokenizer_dir="$exported_dir/tokenizer"
    local orch_model_dir="$repo_dir/tts_orchestrator/$model_version"
    local orch_http_model_dir="$repo_dir/tts_orchestrator_http/$model_version"
    local runtime_dir="$orch_model_dir/runtime"
    # Read engine dtype from build_engines.sh output; default bf16
    local engine_dtype="bf16"
    if [ -f "$exported_dir/.engine_dtype" ]; then
        engine_dtype=$(cat "$exported_dir/.engine_dtype" 2>/dev/null | tr -d '\n' || echo "bf16")
    fi
    engine_dtype="${engine_dtype:-bf16}"

    log_step "Assembling Triton model repository (engine_mode=$engine_mode, dtype=$engine_dtype, version=$model_version)"
    log_info "  Variant:    $variant"
    log_info "  Source:     $variant_dir"
    log_info "  Repository: $repo_dir"

    if [ ! -d "$variant_dir" ]; then
        log_error "Variant directory not found: $variant_dir"
        log_error "Run Phase A first: autorun.sh setup or export_all.py"
        return 1
    fi
    if [ ! -f "$variant_dir/triton_manifest.json" ]; then
        log_error "Missing $variant_dir/triton_manifest.json — run export_09 (writes manifest + fused ONNX)"
        return 1
    fi

    mkdir -p "$repo_dir"
    rm -rf \
        "$repo_dir/talker_code2wav_fused" \
        "$repo_dir/speaker_encoder" \
        "$repo_dir/speech_tokenizer_codec_fused" \
        "$repo_dir/speech_tokenizer_encoder" \
        "$repo_dir/talker_unified" \
        "$repo_dir/code2wav" \
        "$repo_dir/tts_orchestrator" \
        "$repo_dir/tts_orchestrator_http" \
        "$repo_dir/triton_manifest.json" \
        "$repo_dir/artifact_manifest.json"

    # Helper: copy engine/onnx only; config.pbtxt comes from generate_triton_configs.py
    _place_model() {
        local name="$1"
        local src="$2"
        local model_dir="$repo_dir/$name/$model_version"
        mkdir -p "$model_dir"
        if [ "$engine_mode" = "trt" ]; then
            _link_or_copy "$src" "$model_dir/model.plan"
        else
            _link_or_copy "$src" "$model_dir/model.onnx"
            # Copy ONNX external data file if present (large models use .onnx.data).
            # Keep the original filename because the ONNX proto references it internally.
            if [ -f "${src}.data" ]; then
                _link_or_copy "${src}.data" "$model_dir/$(basename "${src}.data")"
                log_info "    + copied external data: $(basename "${src}.data")"
            fi
        fi
    }

    # _resolve_model_src <base_path> selects .onnx or .engine based on engine_mode,
    # verifying that the chosen file actually exists.
    # Returns: 0 + prints path on success, 1 on not found.
    _resolve_model_src() {
        local base="$1"
        local model_name
        model_name="$(basename "$base")"
        local candidates=()
        if [ "$engine_mode" = "trt" ]; then
            candidates=(
                "${base}.engine"
                "${base}/model.plan"
                "${base}/${model_name}.engine"
            )
        else
            candidates=(
                "${base}.onnx"
                "${base}/model.onnx"
                "${base}/${model_name}.onnx"
            )
        fi
        local candidate
        for candidate in "${candidates[@]}"; do
            if [ -f "$candidate" ]; then
                echo "$candidate"; return 0
            fi
        done
        return 1
    }

    _has_onnx_model_src() {
        local base="$1"
        local model_name
        model_name="$(basename "$base")"
        local candidates=(
            "${base}.onnx"
            "${base}/model.onnx"
            "${base}/${model_name}.onnx"
        )
        local candidate
        for candidate in "${candidates[@]}"; do
            if [ -f "$candidate" ]; then
                return 0
            fi
        done
        return 1
    }

    _copy_onnx_external_data_if_present() {
        local src="$1"
        local dst_dir="$2"
        if [ -f "${src}.data" ]; then
            _link_or_copy "${src}.data" "$dst_dir/$(basename "${src}.data")"
        fi
    }

    mkdir -p "$runtime_dir"

    # ── 1. Optional voice-clone runtime assets ──
    # Standalone TRT preprocessing loads these through TRTEngine, so TRT mode
    # must package .engine files instead of ONNX fallbacks.
    local speaker_src
    if speaker_src="$(_resolve_model_src "$variant_dir/speaker_encoder")"; then
        if [ "$engine_mode" = "trt" ]; then
            _link_or_copy "$speaker_src" "$runtime_dir/speaker_encoder.engine"
            log_info "  runtime/speaker_encoder.engine: OK"
        else
            _link_or_copy "$speaker_src" "$runtime_dir/speaker_encoder.onnx"
            _copy_onnx_external_data_if_present "$speaker_src" "$runtime_dir"
            log_info "  runtime/speaker_encoder.onnx: OK"
        fi
    elif [ "$engine_mode" = "trt" ] && _has_onnx_model_src "$variant_dir/speaker_encoder"; then
        log_error "  speaker_encoder.engine: MISSING for TRT voice-clone preprocessing. Run Phase B."
        return 1
    else
        log_warn "  runtime/speaker_encoder: SKIPPED (only needed for voice clone)"
    fi

    local speech_codec_src
    if speech_codec_src="$(_resolve_model_src "$variant_dir/speech_tokenizer_codec_fused")"; then
        if [ "$engine_mode" = "trt" ]; then
            _link_or_copy "$speech_codec_src" "$runtime_dir/speech_tokenizer_codec_fused.engine"
            log_info "  runtime/speech_tokenizer_codec_fused.engine: OK"
        else
            _link_or_copy "$speech_codec_src" "$runtime_dir/speech_tokenizer_codec_fused.onnx"
            _copy_onnx_external_data_if_present "$speech_codec_src" "$runtime_dir"
            log_info "  runtime/speech_tokenizer_codec_fused.onnx: OK"
        fi
    elif [ "$engine_mode" = "trt" ] && _has_onnx_model_src "$variant_dir/speech_tokenizer_codec_fused"; then
        log_error "  speech_tokenizer_codec_fused.engine: MISSING for TRT ICL preprocessing. Run Phase B."
        return 1
    else
        log_warn "  runtime/speech_tokenizer_codec_fused: SKIPPED (not required for non-base / non-ICL)"
    fi

    # ── 2. Talker + Code2Wav fused (required — production single runtime engine) ──
    local fused_src
    if fused_src="$(_resolve_model_src "$variant_dir/talker_code2wav_fused")"; then
        if [ "$engine_mode" = "trt" ]; then
            _link_or_copy "$fused_src" "$runtime_dir/model.plan"
            log_info "  runtime/model.plan: OK"
        else
            _link_or_copy "$fused_src" "$runtime_dir/model.onnx"
            if [ -f "${fused_src}.data" ]; then
                _link_or_copy "${fused_src}.data" "$runtime_dir/$(basename "${fused_src}.data")"
                log_info "    + copied runtime external data: $(basename "${fused_src}.data")"
            fi
            log_info "  runtime/model.onnx: OK"
        fi
    else
        log_error "  talker_code2wav_fused: MISSING ${engine_mode} file (required). Run export_09 + Phase B."
        return 1
    fi

    # ── 3. Verification-only models (optional) ──
    if [ "${ASSEMBLE_VERIFICATION_MODELS:-0}" = "1" ]; then
        local stoken_src
        if stoken_src="$(_resolve_model_src "$tokenizer_dir/speech_tokenizer_encoder")"; then
            _place_model "speech_tokenizer_encoder" "$stoken_src"
            log_info "  speech_tokenizer_encoder: OK (verification)"
        fi
        local talker_src
        if talker_src="$(_resolve_model_src "$variant_dir/talker_unified")"; then
            _place_model "talker_unified" "$talker_src"
            log_info "  talker_unified: OK (verification)"
        fi
        local c2w_src
        if c2w_src="$(_resolve_model_src "$tokenizer_dir/code2wav_decoder")"; then
            _place_model "code2wav" "$c2w_src"
            log_info "  code2wav: OK (verification)"
        fi
    fi

    # ── 5. TTS Orchestrator (Python BLS backend) ──
    local weights_dir="$variant_dir/weights"
    mkdir -p "$orch_model_dir/weights"
    if [ -d "$weights_dir" ]; then
        for f in "$weights_dir"/*; do
            _link_or_copy "$f" "$orch_model_dir/weights/$(basename "$f")"
        done
        log_info "  tts_orchestrator/weights: OK"
    else
        log_warn "  tts_orchestrator/weights: MISSING (no embedding weights found)"
    fi

    # Copy orchestrator adapter + new engine package (paths relative to repo root)
    local repo_root
    repo_root="$(cd "${_LIB_DIR}/../../.." && pwd)"
    local orch_py_dir="$repo_root/model_repository/tts_orchestrator/1"
    if [ -d "$orch_py_dir" ]; then
        cp "$orch_py_dir/model.py" "$orch_model_dir/model.py"

        rm -rf "$orch_model_dir/engine"
        cp -R "$repo_root/engine" "$orch_model_dir/engine"

        log_info "  tts_orchestrator/python: OK (copied model.py + engine/ package)"
    else
        log_warn "  tts_orchestrator/python: source dir not found, using stub"
    fi

    local orch_http_py_dir="$repo_root/model_repository/tts_orchestrator_http/1"
    mkdir -p "$orch_http_model_dir"
    if [ -d "$orch_http_py_dir" ] && [ -f "$orch_http_py_dir/model.py" ]; then
        cp "$orch_http_py_dir/model.py" "$orch_http_model_dir/model.py"
        log_info "  tts_orchestrator_http/python: OK (copied offline HTTP aggregator)"
    else
        log_error "  tts_orchestrator_http/model.py missing in source tree"
        return 1
    fi

    # Copy text tokenizer files for orchestrator
    local model_base_dir
    model_base_dir="$(cd "$(dirname "$exported_dir")" && pwd)/models"
    local tok_dir=""
    case "$variant" in
        design-1.7b) tok_dir="$model_base_dir/Qwen3-TTS-12Hz-1.7B-VoiceDesign" ;;
        custom-1.7b) tok_dir="$model_base_dir/Qwen3-TTS-12Hz-1.7B-CustomVoice" ;;
        base-1.7b)   tok_dir="$model_base_dir/Qwen3-TTS-12Hz-1.7B-Base" ;;
        custom-0.6b) tok_dir="$model_base_dir/Qwen3-TTS-12Hz-0.6B-CustomVoice" ;;
        base-0.6b)   tok_dir="$model_base_dir/Qwen3-TTS-12Hz-0.6B-Base" ;;
    esac
    if [ -n "$tok_dir" ] && [ -d "$tok_dir" ]; then
        mkdir -p "$orch_model_dir/tokenizer"
        for tf in tokenizer.json tokenizer_config.json vocab.json merges.txt \
                  config.json generation_config.json; do
            [ -f "$tok_dir/$tf" ] && \
                _link_or_copy "$tok_dir/$tf" "$orch_model_dir/tokenizer/$tf"
        done
        if [[ "$variant" == base-* || "$variant" == icl-* ]] \
            && [ "$engine_mode" = "trt" ] \
            && [ -f "$tokenizer_dir/code2wav_decoder.engine" ]; then
            _link_or_copy "$tokenizer_dir/code2wav_decoder.engine" \
                "$orch_model_dir/tokenizer/code2wav_decoder.engine"
            log_info "  tts_orchestrator/tokenizer/code2wav_decoder.engine: OK (ICL warm state)"
        elif [[ "$variant" == base-* || "$variant" == icl-* ]] \
            && [ "$engine_mode" = "trt" ] \
            && [ -f "$tokenizer_dir/code2wav_decoder.onnx" ]; then
            log_warn "  tts_orchestrator/tokenizer/code2wav_decoder.engine: SKIPPED (run Phase B to enable ICL warm state)"
        fi
        log_info "  tts_orchestrator/tokenizer: OK"
    else
        log_warn "  tts_orchestrator/tokenizer: SKIPPED (model dir not found)"
    fi

    local resources_dir="$repo_root/resources"
    if [ -d "$resources_dir" ]; then
        rm -rf "$orch_model_dir/resources"
        _link_or_copy "$resources_dir" "$orch_model_dir/resources"
        log_info "  tts_orchestrator/resources: OK"
    else
        log_warn "  tts_orchestrator/resources: SKIPPED (resources/ not found)"
    fi

    _write_tts_orchestrator_stub_if_missing "$repo_dir" "$model_version"

    if ! PYTHONPATH="$repo_root/scripts/python" python3 - \
        "$variant_dir/triton_manifest.json" \
        "$engine_mode" \
        "$model_version" \
        "$repo_dir" \
        "$orch_model_dir/triton_manifest.json" \
        "$runtime_dir/triton_manifest.json" <<'PY'; then
import json
import sys
from pathlib import Path

from triton_manifest_io import load_manifest

src = Path(sys.argv[1])
engine_mode = sys.argv[2]
model_version = sys.argv[3]
output_repo = Path(sys.argv[4])
targets = [Path(p) for p in sys.argv[5:]]
model_package_dir = f"/models/tts_orchestrator/{model_version}"

manifest = load_manifest(src, output_repo=output_repo, model_package_dir=model_package_dir)
manifest["engine_mode"] = engine_mode
manifest.setdefault("package", {})["model_package_dir"] = model_package_dir
manifest.setdefault("orchestrator", {})["model_package_dir"] = model_package_dir
optional_assets = manifest.setdefault("package", {}).setdefault("optional_assets", {})
if isinstance(optional_assets, dict):
    # speech_tokenizer_encoder is verification-only. Production runtime uses
    # speech_tokenizer_codec_fused for ICL preprocessing.
    optional_assets.pop("speech_tokenizer_encoder", None)
    if str(manifest.get("variant", "")).startswith(("base-", "icl-")):
        if engine_mode == "trt":
            optional_assets["speaker_encoder"] = "runtime/speaker_encoder.engine"
            optional_assets["speech_tokenizer_codec_fused"] = "runtime/speech_tokenizer_codec_fused.engine"
        else:
            optional_assets["speaker_encoder"] = "runtime/speaker_encoder.onnx"
            optional_assets["speech_tokenizer_codec_fused"] = "runtime/speech_tokenizer_codec_fused.onnx"
    else:
        optional_assets.pop("speaker_encoder", None)
        optional_assets.pop("speech_tokenizer_codec_fused", None)

for target in targets:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
        log_error "  failed to write package triton_manifest.json"
        return 1
    fi
    log_info "  triton_manifest.json: copied (tts_orchestrator/$model_version + runtime)"

    # Propagate artifact_manifest.json (written by build_pipeline.sh) into
    # the model package so the runtime fingerprint guard can locate it from
    # standalone and Triton deployments. Keep runtime/ as a sibling copy for
    # sha256 verification tools.
    if [ -f "$exported_dir/artifact_manifest.json" ]; then
        cp -f "$exported_dir/artifact_manifest.json" "$orch_model_dir/artifact_manifest.json"
        cp -f "$exported_dir/artifact_manifest.json" "$runtime_dir/artifact_manifest.json"
        log_info "  artifact_manifest.json: copied (tts_orchestrator/$model_version + runtime)"
    else
        log_warn "  artifact_manifest.json: NOT FOUND in $exported_dir"
        log_warn "    Runtime fingerprint guard will fail-stop unless QWEN3_ALLOW_FINGERPRINT_MISMATCH=1"
        log_warn "    Run: bash scripts/bash/autorun.sh build"
        log_warn "      or: bash scripts/bash/autorun.sh import-artifact <bundle>"
    fi

    # Keep the Triton Python backend payload minimal.  Only the new adapter,
    # engine package, tokenizer / weights, and manifest should enter the container.
    find "$orch_model_dir" -mindepth 1 -maxdepth 1 \
        ! -name "model.py" \
        ! -name "engine" \
        ! -name "tokenizer" \
        ! -name "weights" \
        ! -name "runtime" \
        ! -name "resources" \
        ! -name "triton_manifest.json" \
        ! -name "artifact_manifest.json" \
        -exec rm -rf {} +
    find "$orch_model_dir" -type d -name "__pycache__" -prune -exec rm -rf {} +
    log_info "  tts_orchestrator/python: pruned legacy payload"

	    if ! python3 "$repo_root/scripts/python/generate_triton_configs.py" \
	        --manifest "$runtime_dir/triton_manifest.json" \
	        --output-repo "$repo_dir" \
        --engine-mode "$engine_mode" \
        --engine-dtype "$engine_dtype"; then
        log_error "  generate_triton_configs.py failed"
        return 1
    fi
    log_info "  Triton config.pbtxt: generated from triton_manifest.json"

    echo ""
    log_info "Model repository assembled: $repo_dir"
    return 0
}

# ---------------------------------------------------------------------------
#  _resolve_repo_manifest <repo_dir> [model_version]
#  Finds the manifest that describes an assembled model package.  The package
#  copies are authoritative; the repo-root manifest is kept only as a backward
#  compatibility fallback for older repositories.
# ---------------------------------------------------------------------------
_resolve_repo_manifest() {
    local repo_dir="$1"
    local model_version="${2:-}"
    local candidate

    if [ -n "$model_version" ]; then
        candidate="$repo_dir/tts_orchestrator/$model_version/runtime/triton_manifest.json"
        [ -f "$candidate" ] && { echo "$candidate"; return 0; }
        candidate="$repo_dir/tts_orchestrator/$model_version/triton_manifest.json"
        [ -f "$candidate" ] && { echo "$candidate"; return 0; }
    fi

    if [ -d "$repo_dir/tts_orchestrator" ]; then
        candidate=$(find "$repo_dir/tts_orchestrator" \
            -path "*/runtime/triton_manifest.json" -type f 2>/dev/null | sort | head -n 1)
        [ -n "$candidate" ] && { echo "$candidate"; return 0; }
        candidate=$(find "$repo_dir/tts_orchestrator" \
            -mindepth 2 -maxdepth 2 -name "triton_manifest.json" -type f 2>/dev/null | sort | head -n 1)
        [ -n "$candidate" ] && { echo "$candidate"; return 0; }
    fi

    candidate="$repo_dir/triton_manifest.json"
    [ -f "$candidate" ] && { echo "$candidate"; return 0; }
    return 1
}

_infer_model_version_from_repo() {
    local repo_dir="$1"
    local manifest
    manifest=$(_resolve_repo_manifest "$repo_dir" "" 2>/dev/null || true)
    [ -n "$manifest" ] || return 1
    python3 - "$manifest" <<'PY' 2>/dev/null
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    manifest = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    manifest = {}

package = manifest.get("package") or {}
package_dir = ""
if isinstance(package, dict):
    package_dir = str(package.get("model_package_dir") or "")
if not package_dir:
    orch = manifest.get("orchestrator") or {}
    if isinstance(orch, dict):
        package_dir = str(orch.get("model_package_dir") or "")
if package_dir:
    print(Path(package_dir).name)
elif path.parent.name == "runtime":
    print(path.parent.parent.name)
else:
    print(path.parent.name)
PY
}

# ---------------------------------------------------------------------------
#  sync_trt_configs <repo_dir> <engine_mode> <exported_dir>
#  When engine_mode=trt, ensure optional models (speaker_encoder,
#  speech_tokenizer_encoder) that exist in the repo have full TRT config
#  (explicit input/output). Fixes "failed to specify dimensions" when using
#  an existing repo assembled for a different variant (e.g. base had speaker_encoder).
# ---------------------------------------------------------------------------
sync_trt_configs() {
    local repo_dir="$1"
    local engine_mode="$2"
    local exported_dir="${3:-}"
    [ "$engine_mode" != "trt" ] && return 0

    local engine_dtype="bf16"
    if [ -n "$exported_dir" ] && [ -f "$exported_dir/.engine_dtype" ]; then
        engine_dtype=$(cat "$exported_dir/.engine_dtype" 2>/dev/null | tr -d '\n' || echo "bf16")
    fi
    engine_dtype="${engine_dtype:-bf16}"

    local manifest_path
    manifest_path=$(_resolve_repo_manifest "$repo_dir" "${MODEL_VERSION:-${ENGINE_MODEL_VERSION:-}}" 2>/dev/null || true)
    if [ -z "$manifest_path" ]; then
        log_warn "  sync_trt_configs: no triton_manifest.json in model package (skip)"
        return 0
    fi
    local repo_root
    repo_root="$(cd "${_LIB_DIR}/../../.." && pwd)"
    if python3 "$repo_root/scripts/python/generate_triton_configs.py" \
        --manifest "$manifest_path" \
        --output-repo "$repo_dir" \
        --engine-mode "$engine_mode" \
        --engine-dtype "$engine_dtype"; then
        log_info "  Triton configs: regenerated from $manifest_path (sync_trt_configs)"
        return 0
    fi
    log_error "  generate_triton_configs.py failed"
    return 1
}

# ---------------------------------------------------------------------------
#  validate_model_repo <model_repo_dir>
#  Quick sanity check for the thin Python backend layout.
#  Returns 0 if all critical assets are present, 1 otherwise.
# ---------------------------------------------------------------------------
validate_model_repo() {
    local repo_dir="$1"
    local model_version="${2:-}"
    local missing=0
    local warned=0

    if [ -z "$model_version" ]; then
        model_version=$(_infer_model_version_from_repo "$repo_dir" 2>/dev/null || true)
    fi
    model_version=$(resolve_model_version "${model_version:-}") || return 1

    log_step "Validating model repository: $repo_dir"

    for model in "${_TRITON_REQUIRED_MODELS[@]}"; do
        local model_dir="$repo_dir/$model"
        if [ ! -f "$model_dir/config.pbtxt" ]; then
            log_error "  $model: no config.pbtxt (required)"
            missing=$((missing + 1))
            continue
        fi

        if [ ! -d "$model_dir/$model_version" ]; then
            log_error "  $model: no version directory ($model_version/)"
            missing=$((missing + 1))
            continue
        fi

        log_info "  $model: OK"
    done

    local orch_dir="$repo_dir/tts_orchestrator/$model_version"
    local repo_root
    repo_root="$(cd "${_LIB_DIR}/../../.." && pwd)"
    local package_info
    package_info=$(PYTHONPATH="$repo_root" python3 - "$orch_dir" <<'PY' 2>/dev/null || true
import sys
from engine.config import resolve_model_package_paths

p = resolve_model_package_paths(sys.argv[1])
print("\t".join([p.engine_dir, p.manifest_path, p.runtime_artifact_path]))
PY
)
    local runtime_dir manifest_path runtime_engine
    IFS=$'\t' read -r runtime_dir manifest_path runtime_engine <<< "$package_info"
    runtime_dir="${runtime_dir:-$orch_dir/runtime}"
    manifest_path="${manifest_path:-$runtime_dir/triton_manifest.json}"
    runtime_engine="${runtime_engine:-$runtime_dir/model.plan}"

    if [ ! -f "$orch_dir/model.py" ]; then
        log_error "  tts_orchestrator/$model_version/model.py: missing"
        missing=$((missing + 1))
    else
        log_info "  tts_orchestrator/$model_version/model.py: OK"
    fi
    if [ ! -d "$orch_dir/engine" ]; then
        log_error "  tts_orchestrator/$model_version/engine/: missing"
        missing=$((missing + 1))
    else
        log_info "  tts_orchestrator/$model_version/engine/: OK"
    fi
    if [ ! -f "$manifest_path" ]; then
        log_error "  model package manifest: missing ($manifest_path)"
        missing=$((missing + 1))
    else
        log_info "  model package manifest: OK ($manifest_path)"
    fi
    if [ ! -f "$orch_dir/triton_manifest.json" ]; then
        log_error "  tts_orchestrator/$model_version/triton_manifest.json: missing"
        missing=$((missing + 1))
    else
        log_info "  tts_orchestrator/$model_version/triton_manifest.json: OK"
    fi
    if [ ! -f "$orch_dir/artifact_manifest.json" ]; then
        log_warn "  tts_orchestrator/$model_version/artifact_manifest.json: missing"
        log_warn "    Runtime fingerprint guard will fail-stop unless QWEN3_ALLOW_FINGERPRINT_MISMATCH=1"
        warned=$((warned + 1))
    else
        log_info "  tts_orchestrator/$model_version/artifact_manifest.json: OK"
    fi
    if [ -d "$orch_dir/resources" ]; then
        log_info "  tts_orchestrator/$model_version/resources/: OK"
    elif [ -d "$repo_root/resources" ]; then
        log_warn "  tts_orchestrator/$model_version/resources/: missing (references using resources/... may fail)"
        warned=$((warned + 1))
    fi

    if [ ! -f "$runtime_engine" ]; then
        log_error "  runtime artifact: missing ($runtime_engine)"
        missing=$((missing + 1))
    else
        log_info "  runtime engine: OK ($(basename "$runtime_engine"))"
    fi

    if [ -d "$repo_dir/talker_code2wav_fused" ]; then
        log_error "  legacy top-level model detected: $repo_dir/talker_code2wav_fused"
        log_error "  re-run assemble so the fused engine lives under tts_orchestrator/$model_version/runtime/"
        missing=$((missing + 1))
    fi

    local orch_http_dir="$repo_dir/tts_orchestrator_http/$model_version"
    if [ ! -f "$orch_http_dir/model.py" ]; then
        log_error "  tts_orchestrator_http/$model_version/model.py: missing"
        missing=$((missing + 1))
    else
        log_info "  tts_orchestrator_http/$model_version/model.py: OK"
    fi

    if [ "$missing" -gt 0 ]; then
        log_error "Validation failed: $missing required model(s) missing"
        return 1
    fi

    if [ "$warned" -gt 0 ]; then
        log_warn "Validation passed with $warned optional model(s) missing"
    else
        log_info "Validation passed: all models present"
    fi
    return 0
}

# ---------------------------------------------------------------------------
#  build_triton_image <repo_root> <image_tag> [base_image]
#  Builds a self-contained deployment image with all model artifacts baked in.
#  Uses the Dockerfile at <repo_root>/Dockerfile.triton.
#
#  For development, prefer triton_run() with volume mounts instead.
# ---------------------------------------------------------------------------
build_triton_image() {
    local repo_root="$1"
    local image_tag="$2"
    local base_image="${3:-}"
    local trt_python_version="${TRITON_TENSORRT_PIP_VERSION:-10.15.1.29}"
    local pytorch_cuda_tag="${TRITON_PYTORCH_CUDA_TAG:-${PYTORCH_CUDA_TAG:-cu130}}"

    if [ -z "$base_image" ]; then
        base_image=$(resolve_triton_deploy_image) \
            || { log_error "Cannot determine base image"; return 1; }
    fi

    local model_repo="$repo_root/workspace/model_repository"
    local model_version="${MODEL_VERSION:-${ENGINE_MODEL_VERSION:-1}}"
    local inferred_version
    inferred_version=$(_infer_model_version_from_repo "$model_repo" 2>/dev/null || true)
    if [ -n "$inferred_version" ]; then
        model_version="$inferred_version"
    fi
    model_version=$(resolve_model_version "${model_version:-}") || return 1
    local model_dir="$model_repo/tts_orchestrator/$model_version"
    local runtime_dir="$model_dir/runtime"
    local fused_artifact=""
    if [ -f "$runtime_dir/model.plan" ]; then
        fused_artifact="$runtime_dir/model.plan"
    elif [ -f "$runtime_dir/model.onnx" ]; then
        fused_artifact="$runtime_dir/model.onnx"
    elif [ -f "$runtime_dir/talker_code2wav_fused.engine" ]; then
        fused_artifact="$runtime_dir/talker_code2wav_fused.engine"
    fi

    if [ ! -f "$model_dir/model.py" ] \
        || [ ! -d "$model_dir/engine" ] \
        || [ -z "$fused_artifact" ]; then
        log_error "Assembled model repository is incomplete for image build"
        log_error "Expected:"
        log_error "  $model_dir/model.py"
        log_error "  $model_dir/engine/"
        log_error "  $runtime_dir/model.plan or model.onnx"
        return 1
    fi

    local build_dir
    build_dir=$(mktemp -d)
    local dockerfile="$build_dir/Dockerfile"
    mkdir -p "$build_dir/model_repository"
    if command -v rsync &>/dev/null; then
        rsync -a "$model_repo/" "$build_dir/model_repository/"
    else
        cp -a "$model_repo/." "$build_dir/model_repository/"
    fi
    cat > "$dockerfile" <<DOCKERFILE
ARG BASE_IMAGE=${base_image}
ARG TENSORRT_PYTHON_VERSION=${trt_python_version}
ARG PYTORCH_CUDA_TAG=${pytorch_cuda_tag}
FROM \${BASE_IMAGE}
ARG TENSORRT_PYTHON_VERSION
ARG PYTORCH_CUDA_TAG

ENV TRITON_PYTORCH_CUDA_TAG=\${PYTORCH_CUDA_TAG}

RUN python3 -m pip install --no-cache-dir \
    -i https://mirrors.bfsu.edu.cn/pypi/web/simple \
    --trusted-host mirrors.bfsu.edu.cn \
    filelock \
    fsspec \
    jinja2 \
    mpmath \
    networkx \
    sympy \
    tokenizers \
    "tensorrt==\${TENSORRT_PYTHON_VERSION}" \
    && python3 -m pip install --no-cache-dir \
    --timeout 120 \
    --retries 10 \
    --index-url https://download.pytorch.org/whl/\${PYTORCH_CUDA_TAG} \
    torch

COPY model_repository /models

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

EXPOSE 8000 8001 8002

ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/models", "--strict-model-config=false", "--disable-auto-complete-config", "--log-verbose=1"]
DOCKERFILE

    log_step "Building Triton deployment image: $image_tag"
    log_info "  Base image:  $base_image"
    log_info "  Runtime:     /models/tts_orchestrator/$model_version/model.py + engine/ + $(basename "$fused_artifact")"

    docker build \
        --build-arg "BASE_IMAGE=$base_image" \
        --build-arg "TENSORRT_PYTHON_VERSION=$trt_python_version" \
        --build-arg "PYTORCH_CUDA_TAG=$pytorch_cuda_tag" \
        -t "$image_tag" \
        -f "$dockerfile" \
        "$build_dir" \
        || { rm -rf "$build_dir"; log_error "Docker build failed"; return 1; }

    rm -rf "$build_dir"

    log_info "Image built: $image_tag"
}

# ---------------------------------------------------------------------------
#  triton_resolve_container_name <model_repo_dir> <container_name_arg> [variant_cli]
#  Must stay in sync with triton_run naming: default base name + model_variant
#  from assembled tts_orchestrator/config.pbtxt, or explicit --variant on stop.
#  Precedence: non-default container_name_arg; else variant_cli; else pbtxt.
# ---------------------------------------------------------------------------
triton_resolve_container_name() {
    local repo_dir="$1"
    local cname="${2:-qwen3-tts-triton}"
    local variant_cli="${3:-}"

    if [ "$cname" != "qwen3-tts-triton" ]; then
        printf '%s\n' "$cname"
        return 0
    fi
    if [ -n "$variant_cli" ]; then
        printf '%s\n' "qwen3-tts-triton-${variant_cli}"
        return 0
    fi
    local config_file="$repo_dir/tts_orchestrator/config.pbtxt"
    if [ -f "$config_file" ]; then
        local var_val
        var_val=$(grep -A 1 'key: "model_variant"' "$config_file" | grep 'string_value' | cut -d'"' -f2 || true)
        if [ -n "$var_val" ]; then
            printf '%s\n' "qwen3-tts-triton-${var_val}"
            return 0
        fi
    fi
    printf '%s\n' "qwen3-tts-triton"
}

# ---------------------------------------------------------------------------
#  triton_run <model_repo_dir> [image] [container_name] [extra_docker_args...]
#  Starts a Triton server container with the model repository volume-mounted.
#  Runs in detached mode; use triton_health_check to verify readiness.
# ---------------------------------------------------------------------------
triton_run() {
    local repo_dir="$1"
    local image="${2:-}"
    local cname_arg="${3:-qwen3-tts-triton}"
    shift 3 2>/dev/null || true

    if [ -z "$image" ]; then
        image=$(resolve_triton_deploy_image) \
            || { log_error "Cannot determine Triton image"; return 1; }
    fi

    repo_dir="$(cd "$repo_dir" && pwd)"

    local container_name
    container_name=$(triton_resolve_container_name "$repo_dir" "$cname_arg" "")

    # Remove existing container (running or exited) so we can start fresh
    if docker container inspect "$container_name" &>/dev/null; then
        log_warn "Removing existing container: $container_name"
        docker rm -f "$container_name" &>/dev/null || true
    fi

    log_step "Starting Triton Inference Server"
    log_info "  Image:      $image"
    log_info "  Repository: $repo_dir"
    log_info "  Container:  $container_name"
    log_info "  Ports:      gRPC=$TRITON_GRPC_PORT HTTP=$TRITON_HTTP_PORT metrics=$TRITON_METRICS_PORT"

    local gpu_device="${TRITON_GPU_DEVICE:-0}"
    local max_batch_slots="${TRITON_MAX_BATCH_SLOTS:-${RUNTIME_MAX_BATCH_SIZE:-128}}"
    local max_seq_len="${TRITON_MAX_SEQ_LEN:-${RUNTIME_MAX_SEQ_LEN:-512}}"
    local variant_label=""
    local type_label=""
    
    # Try to extract variant from config.pbtxt if available
    local config_file="$repo_dir/tts_orchestrator/config.pbtxt"
    if [ -f "$config_file" ]; then
        local var_val=$(grep -A 1 'key: "model_variant"' "$config_file" | grep 'string_value' | cut -d'"' -f2 || true)
        local type_val=$(grep -A 1 'key: "tts_model_type"' "$config_file" | grep 'string_value' | cut -d'"' -f2 || true)
        if [ -n "$var_val" ]; then variant_label="--label tts.variant=${var_val}"; fi
        if [ -n "$type_val" ]; then type_label="--label tts.model_type=${type_val}"; fi
    fi

    # When code2wav uses TensorRT (BF16), pass override so orchestrator sends correct dtype
    local code2wav_bf16_env=""
    if [ -f "$repo_dir/code2wav/config.pbtxt" ] && grep -q "tensorrt" "$repo_dir/code2wav/config.pbtxt" 2>/dev/null; then
        code2wav_bf16_env="-e OVERRIDE_CODE2WAV_BF16=1"
    fi

    docker run -d --gpus "\"device=${gpu_device}\"" \
        --name "$container_name" \
        --shm-size=1g \
        --ulimit memlock=-1 \
        $variant_label $type_label \
        $code2wav_bf16_env \
        -e "MAX_BATCH_SLOTS=${max_batch_slots}" \
        -e "ENGINE_MAX_DECODE_LEN=${max_seq_len}" \
        -p "${TRITON_HTTP_PORT}:8000" \
        -p "${TRITON_GRPC_PORT}:8001" \
        -p "${TRITON_METRICS_PORT}:8002" \
        -v "$repo_dir:/models" \
        "$@" \
        "$image" \
        tritonserver \
            --model-repository=/models \
            --log-verbose=1 \
            --strict-model-config=false \
            --disable-auto-complete-config \
        || { log_error "Failed to start Triton container"; return 1; }

    log_info "Container started: $container_name"
    log_info "Waiting for server to be ready ..."
}

# ---------------------------------------------------------------------------
#  triton_health_check [host] [port] [timeout_sec]
#  Polls the Triton health endpoint until ready or timeout.
# ---------------------------------------------------------------------------
triton_health_check() {
    local host="${1:-localhost}"
    local port="${2:-$TRITON_HTTP_PORT}"
    local timeout="${3:-60}"

    local url="http://${host}:${port}/v2/health/ready"
    local elapsed=0
    local interval=3

    while [ "$elapsed" -lt "$timeout" ]; do
        if curl -sf "$url" &>/dev/null; then
            log_info "Triton server ready at ${host}:${port}"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    log_error "Triton health check timed out after ${timeout}s"
    log_error "Check logs: docker logs <container_name>"
    return 1
}

# ---------------------------------------------------------------------------
#  triton_stop [container_name]
#  Stops and removes a running Triton container.
# ---------------------------------------------------------------------------
triton_stop() {
    local container_name="${1:-qwen3-tts-triton}"

    # Exact name only (docker ps --filter name= is substring match and caused false positives)
    if ! docker container inspect "$container_name" &>/dev/null; then
        log_info "Container not found: $container_name"
        return 0
    fi
    log_info "Stopping and removing Triton container: $container_name"
    docker rm -f "$container_name" \
        || { log_error "Failed to remove container: $container_name"; return 1; }
    log_info "Container removed: $container_name"
}


# ---------------------------------------------------------------------------
#  _write_tts_orchestrator_stub_if_missing <repo_dir> [model_version]
#  Minimal Python stub when model_repository sources were not copied.
# ---------------------------------------------------------------------------
_write_tts_orchestrator_stub_if_missing() {
    local repo_dir="$1"
    local model_version
    model_version=$(resolve_model_version "${2:-}") || return 1
    local orch_dir="$repo_dir/tts_orchestrator/$model_version"
    mkdir -p "$orch_dir"
    if [ -f "$orch_dir/model.py" ]; then
        return 0
    fi
    cat > "$orch_dir/model.py" << 'PYEOF'
import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        print("[TTS Orchestrator] Stub initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            audio = np.array([b""], dtype=object)
            is_final = np.array([True], dtype=bool)
            response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("audio_chunk", audio),
                                pb_utils.Tensor("event_type", np.array(["error"], dtype=object)),
                                pb_utils.Tensor("event_json", np.array([json.dumps({"type":"error","message":"stub"})], dtype=object)),
                                pb_utils.Tensor("is_final", is_final)])
            responses.append(response)
        return responses

    def finalize(self):
        print("[TTS Orchestrator] Finalized")
PYEOF
    log_warn "  tts_orchestrator/model.py: stub created (no source tree)"
}
