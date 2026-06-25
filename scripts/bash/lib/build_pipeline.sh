#!/bin/bash
# ===========================================================================
#  build_pipeline.sh — Orchestration helpers for the unified engine pipeline
#
#  This module is the bridge between dispatch (autorun.sh / build_engines.sh
#  entrypoint) and the runner (lib/trtexec_runner.sh).  Local builds, manual
#  cross-host builds and SSH-driven remote builds all share the same primitive
#  here: take a "bundle root" directory that contains ONNX + build_manifest +
#  target_profile, compile every variant via _trtexec_run, then write
#  artifact_manifest.json with sha256 + tensorrt_version + gpu_sm so the
#  deploy-side fingerprint check has something to validate against.
#
#  Bundle layout (identical for local / cross-host / SSH paths):
#      <bundle_root>/
#        build_manifest.json
#        target_profile.json
#        scripts/                         (optional; populated for ssh/manual)
#        workspace/exported/<variant>/    ONNX inputs → engines after build
#        workspace/exported/tokenizer/    shared tokenizer engines
#        artifact_manifest.json           (written after compile)
#
#  Public interface:
#    prepare_local_bundle_workspace   <repo_root> <target_profile> <bundle_root>
#                                      <variant_csv> <engine_dtype> <io_dtype>
#                                      <max_batch> <max_input> <max_seq>
#                                      <build_gpu_device>
#    compile_engines_in_bundle        <bundle_root>
#    write_artifact_manifest          <bundle_root>
#    collect_local_engines_to_workspace <bundle_root> <exported_dir>
#    pack_engine_artifact_bundle      <bundle_root> <out_tar_zst>
#
#  Depends: lib/logging.sh, lib/cross_host.sh, lib/trtexec_runner.sh,
#           lib/docker.sh
# ===========================================================================

[[ -n "${_LIB_BUILD_PIPELINE_LOADED:-}" ]] && return 0
_LIB_BUILD_PIPELINE_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"
source "${_LIB_DIR}/docker.sh"
source "${_LIB_DIR}/cross_host.sh"
source "${_LIB_DIR}/trtexec_runner.sh"

# ---------------------------------------------------------------------------
#  suggest_build_profile_from_memory <memory_mib>
#  Echoes "max_batch max_input max_seq" suggested for the given GPU memory.
#  Shared by autorun.sh (pre-bundle defaults) and build_engines.sh.
# ---------------------------------------------------------------------------
suggest_build_profile_from_memory() {
    local mem_mb="${1:-0}"
    # nvidia-smi reports usable MiB, not marketing GB.  Some 48 GB class
    # cards report around 46,000 MiB, so keep tier cutoffs below the nominal
    # decimal values used in docs.
    if [ "$mem_mb" -ge 76000 ]; then
        echo "128 128 512"
    elif [ "$mem_mb" -ge 45000 ]; then
        echo "64 128 512"
    elif [ "$mem_mb" -ge 29000 ]; then
        echo "32 128 512"
    else
        echo "16 96 384"
    fi
}

# ---------------------------------------------------------------------------
#  suggest_build_profile_from_exports <memory_mib> <exported_dir> <variant_csv>
#                                     <engine_dtype> <max_input> <max_seq>
#  Echoes "max_batch max_input max_seq" using exported manifest/weights to
#  estimate fixed TRT memory plus per-lane KV/state peak.  Falls back to the
#  coarse memory tier when manifests are unavailable.
# ---------------------------------------------------------------------------
suggest_build_profile_from_exports() {
    local mem_mb="${1:-0}"
    local exported_dir="${2:-}"
    local variant_csv="${3:-}"
    local engine_dtype="${4:-bf16}"
    local max_input="${5:-128}"
    local max_seq="${6:-512}"
    local helper="${_LIB_DIR}/../../python/suggest_engine_profile.py"

    if [ -f "$helper" ] && [ -d "$exported_dir" ]; then
        python3 "$helper" \
            --memory-mib "$mem_mb" \
            --exported-dir "$exported_dir" \
            --variants "$variant_csv" \
            --engine-dtype "$engine_dtype" \
            --max-input-len "$max_input" \
            --max-seq-len "$max_seq" \
            2>/dev/null && return 0
    fi
    suggest_build_profile_from_memory "$mem_mb"
}

summarize_build_profile_from_exports() {
    local mem_mb="${1:-0}"
    local exported_dir="${2:-}"
    local variant_csv="${3:-}"
    local engine_dtype="${4:-bf16}"
    local max_input="${5:-128}"
    local max_seq="${6:-512}"
    local helper="${_LIB_DIR}/../../python/suggest_engine_profile.py"

    if [ -f "$helper" ] && [ -d "$exported_dir" ]; then
        python3 "$helper" \
            --memory-mib "$mem_mb" \
            --exported-dir "$exported_dir" \
            --variants "$variant_csv" \
            --engine-dtype "$engine_dtype" \
            --max-input-len "$max_input" \
            --max-seq-len "$max_seq" \
            --format summary \
            2>/dev/null || true
    fi
}

# ---------------------------------------------------------------------------
#  resolve_build_profile_for_target <target_profile> <gpu_index>
#                                   [exported_dir] [variant_csv]
#                                   [engine_dtype] [max_input] [max_seq]
#  Echoes "max_batch max_input max_seq" using the target_profile's GPU
#  memory plus export-aware model sizing when exported_dir/variant_csv are
#  provided.  Falls back to current host's nvidia-smi if the profile lacks the
#  requested index, then to coarse memory tiers if manifests are unavailable.
# ---------------------------------------------------------------------------
resolve_build_profile_for_target() {
    local target_profile="$1"
    local gpu_index="${2:-0}"
    local exported_dir="${3:-}"
    local variant_csv="${4:-}"
    local engine_dtype="${5:-bf16}"
    local max_input="${6:-128}"
    local max_seq="${7:-512}"
    local mem_mb=0
    if [ -f "$target_profile" ]; then
        mem_mb=$(target_profile_memory_mb "$target_profile" "$gpu_index")
    fi
    if [ -z "$mem_mb" ] || [ "$mem_mb" = "0" ]; then
        mem_mb=$(gpu_total_memory_mb "$gpu_index" 2>/dev/null || echo 0)
    fi
    [ -n "$mem_mb" ] || mem_mb=0
    if [ -n "$exported_dir" ]; then
        suggest_build_profile_from_exports \
            "$mem_mb" "$exported_dir" "$variant_csv" \
            "$engine_dtype" "$max_input" "$max_seq"
    else
        suggest_build_profile_from_memory "$mem_mb"
    fi
}

# ---------------------------------------------------------------------------
#  _link_or_copy_onnx <src_dir> <dst_dir> <glob>
#  Used by prepare_local_bundle_workspace to avoid duplicating multi-GB
#  ONNX files when building locally.  Tries hardlink first (fastest,
#  works on most local filesystems), falls back to cp.
# ---------------------------------------------------------------------------
_link_or_copy_onnx() {
    local src_dir="$1"
    local dst_dir="$2"
    local pattern="$3"
    mkdir -p "$dst_dir"
    local file
    shopt -s nullglob
    for file in "$src_dir"/$pattern; do
        local base
        base=$(basename "$file")
        if [ -e "$dst_dir/$base" ]; then
            rm -f "$dst_dir/$base"
        fi
        if ln "$file" "$dst_dir/$base" 2>/dev/null; then
            continue
        fi
        # Fallback: real copy
        cp -a "$file" "$dst_dir/$base"
    done
    shopt -u nullglob
}

# ---------------------------------------------------------------------------
#  prepare_local_bundle_workspace
#  Creates a fresh bundle-shaped directory for local builds, materialized
#  via hardlinks from $exported_dir so the local path reuses the same
#  compile_engines_in_bundle primitive as the cross-host path without
#  paying multi-GB copy cost.
#
#  Args:
#    <repo_root>          repository root (for sourcing helper scripts)
#    <exported_dir>       source workspace/exported/ with ONNX inputs
#    <target_profile>     path to target_profile.json
#    <bundle_root>        destination directory (created/cleaned)
#    <variant_csv>        comma-separated variant names to include
#    <engine_dtype>       bf16|fp16|fp32|fp8
#    <io_dtype>           triton IO float dtype
#    <max_batch>          int
#    <max_input>          int
#    <max_seq>            int
#    <build_gpu_device>   auto|all|N|cuda:N
# ---------------------------------------------------------------------------
prepare_local_bundle_workspace() {
    local repo_root="$1"
    local exported_dir="$2"
    local target_profile="$3"
    local bundle_root="$4"
    local variant_csv="$5"
    local engine_dtype="$6"
    local io_dtype="$7"
    local max_batch="$8"
    local max_input="$9"
    local max_seq="${10}"
    local build_gpu_device="${11:-auto}"

    [ -f "$target_profile" ] || { log_error "target_profile.json missing: $target_profile"; return 1; }
    [ -d "$exported_dir" ] || { log_error "exported dir missing: $exported_dir"; return 1; }

    rm -rf "$bundle_root"
    mkdir -p "$bundle_root/workspace/exported"

    cp "$target_profile" "$bundle_root/target_profile.json"

    # Resolve recommended NGC tag/image from target_profile so the manifest
    # is consistent with what cross-host builds would record.
    local ngc_tag ngc_image
    ngc_tag=$(resolve_ngc_tag_from_profile "$target_profile") || return 1
    ngc_image=$(resolve_ngc_image_from_tag "$ngc_tag") || return 1

    # Materialize ONNX inputs via hardlinks (fast).
    local IFS_BACKUP="$IFS"
    IFS=','
    local variant
    for variant in $variant_csv; do
        [ -n "$variant" ] || continue
        if [ ! -d "$exported_dir/$variant" ]; then
            log_error "Variant export dir not found: $exported_dir/$variant"
            IFS="$IFS_BACKUP"
            return 1
        fi
        local dst="$bundle_root/workspace/exported/$variant"
        mkdir -p "$dst"
        _link_or_copy_onnx "$exported_dir/$variant" "$dst" "*.onnx"
        _link_or_copy_onnx "$exported_dir/$variant" "$dst" "*.onnx.data"
        if [ -f "$exported_dir/$variant/triton_manifest.json" ]; then
            cp -a "$exported_dir/$variant/triton_manifest.json" "$dst/"
        fi
        if [ -d "$exported_dir/$variant/weights" ]; then
            cp -aR "$exported_dir/$variant/weights" "$dst/weights"
        fi
    done
    IFS="$IFS_BACKUP"

    if [ -d "$exported_dir/tokenizer" ]; then
        local tok_dst="$bundle_root/workspace/exported/tokenizer"
        mkdir -p "$tok_dst"
        _link_or_copy_onnx "$exported_dir/tokenizer" "$tok_dst" "*.onnx"
        _link_or_copy_onnx "$exported_dir/tokenizer" "$tok_dst" "*.onnx.data"
    fi

    # Reference repo scripts via symlink for local builds; cross-host
    # builds populate a real copy through make_engine_build_bundle.
    if [ ! -e "$bundle_root/scripts" ]; then
        ln -s "$repo_root/scripts" "$bundle_root/scripts"
    fi

    # build_manifest.json — same schema as make_engine_build_bundle.
    python3 - "$bundle_root/build_manifest.json" \
        "$variant_csv" "$engine_dtype" "$io_dtype" \
        "$max_batch" "$max_input" "$max_seq" \
        "$ngc_tag" "$ngc_image" "$build_gpu_device" <<'PY'
import datetime as dt
import json
import sys

path, variants, dtype, io_dtype, mb, mi, ms, tag, image, device = sys.argv[1:11]
manifest = {
    "bundle_schema_version": 1,
    "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    "variants": [v for v in variants.split(",") if v],
    "engine_dtype": dtype,
    "triton_io_float_dtype": io_dtype or dtype,
    "max_batch_size": int(mb),
    "max_input_len": int(mi),
    "max_seq_len": int(ms),
    "ngc_tag": tag,
    "ngc_image": image,
    "build_gpu_device": device,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

    log_info "Local bundle workspace ready: $bundle_root"
}

# ---------------------------------------------------------------------------
#  compile_engines_in_bundle <bundle_root>
#
#  Reads bundle_root/build_manifest.json and bundle_root/target_profile.json,
#  then invokes build_engines.sh with --in-bundle so it operates against
#  bundle_root/workspace/exported/.  Uses the resolved trtexec runner under
#  the hood.
#
#  Returns non-zero if compilation fails.
# ---------------------------------------------------------------------------
compile_engines_in_bundle() {
    local bundle_root="$1"
    [ -d "$bundle_root" ] || { log_error "Bundle root not found: $bundle_root"; return 1; }

    local manifest="$bundle_root/build_manifest.json"
    local target_profile="$bundle_root/target_profile.json"
    [ -f "$manifest" ] || { log_error "Missing build_manifest.json in $bundle_root"; return 1; }
    [ -f "$target_profile" ] || { log_error "Missing target_profile.json in $bundle_root"; return 1; }

    local engine_dtype io_dtype max_batch max_input max_seq ngc_tag ngc_image build_device
    engine_dtype=$(cross_host_json_value "$manifest" engine_dtype) || engine_dtype=""
    io_dtype=$(cross_host_json_value "$manifest" triton_io_float_dtype) || io_dtype=""
    max_batch=$(cross_host_json_value "$manifest" max_batch_size) || max_batch=""
    max_input=$(cross_host_json_value "$manifest" max_input_len) || max_input=""
    max_seq=$(cross_host_json_value "$manifest" max_seq_len) || max_seq=""
    ngc_tag=$(cross_host_json_value "$manifest" ngc_tag) || ngc_tag=""
    ngc_image=$(cross_host_json_value "$manifest" ngc_image) || ngc_image=""
    build_device=$(cross_host_json_value "$manifest" build_gpu_device) || build_device="auto"

    local variants_json
    variants_json=$(cross_host_json_value "$manifest" variants) || variants_json="[]"
    local variants
    variants=$(python3 -c "import json,sys; print(' '.join(json.loads(sys.argv[1])))" "$variants_json")

    local runner
    runner=$(resolve_trtexec_runner) || return 1
    log_info "compile_engines_in_bundle: runner=$runner"

    # The bundle ships scripts/ either as a real copy (cross-host) or as a
    # symlink to the repo (local).  Either way invoke build_engines.sh from
    # there so trtexec runner picks up the bundle workspace.
    local scripts_dir="$bundle_root/scripts/bash"
    [ -f "$scripts_dir/build_engines.sh" ] || {
        log_error "Bundle scripts missing: $scripts_dir/build_engines.sh"
        return 1
    }

    export NGC_TAG="$ngc_tag"
    export NGC_IMAGE="$ngc_image"
    export BUILD_GPU_DEVICE="$build_device"

    local variant
    local failed=0
    for variant in $variants; do
        log_step "Bundle compile: variant=$variant (runner=$runner)"
        QWEN3_IN_BUNDLE_ROOT="$bundle_root" \
        BUILD_RUNNER="$runner" \
        bash "$scripts_dir/build_engines.sh" build \
            --variant "$variant" \
            --image "$ngc_image" \
            --device "$build_device" \
            --dtype "$engine_dtype" \
            --triton-io-float-dtype "$io_dtype" \
            --max-batch-size "$max_batch" \
            --max-input-len "$max_input" \
            --max-seq-len "$max_seq" \
            || failed=$((failed + 1))
    done

    if [ "$failed" -gt 0 ]; then
        log_error "compile_engines_in_bundle: $failed variant(s) failed"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
#  write_artifact_manifest <bundle_root>
#
#  Walks the engines in bundle_root/workspace/exported/, computes sha256
#  for each, captures runtime metadata (tensorrt_version, driver, sm, cuda),
#  and writes bundle_root/artifact_manifest.json that engine_fingerprint_check
#  can later validate.
# ---------------------------------------------------------------------------
write_artifact_manifest() {
    local bundle_root="$1"
    [ -d "$bundle_root" ] || { log_error "Bundle root not found: $bundle_root"; return 1; }

    local build_manifest="$bundle_root/build_manifest.json"
    local target_profile="$bundle_root/target_profile.json"
    [ -f "$build_manifest" ] || { log_error "Missing build_manifest.json"; return 1; }
    [ -f "$target_profile" ] || { log_error "Missing target_profile.json"; return 1; }

    local ngc_tag
    ngc_tag=$(cross_host_json_value "$build_manifest" ngc_tag)
    local ngc_image
    ngc_image=$(cross_host_json_value "$build_manifest" ngc_image)
    local build_device
    build_device=$(cross_host_json_value "$build_manifest" build_gpu_device) || build_device="auto"

    # Resolve actual TensorRT / driver / SM / CUDA on the executing host.
    local tensorrt_version=""
    if probe_host_trtexec_path >/dev/null 2>&1; then
        tensorrt_version=$(probe_host_trtexec_version 2>/dev/null || true)
    fi
    if [ -z "$tensorrt_version" ] && runner_supports_nested_docker; then
        tensorrt_version=$(docker run --rm "$ngc_image" /bin/bash -lc \
            '/usr/src/tensorrt/bin/trtexec --version 2>/dev/null | head -20' \
            2>/dev/null | sed -nE 's/.*\[TensorRT v([0-9]+)\].*/\1/p' | head -1 || true)
        if [ -n "$tensorrt_version" ] && [ "${#tensorrt_version}" -ge 6 ]; then
            local mj="${tensorrt_version:0:2}" mn="${tensorrt_version:2:2}" pt="${tensorrt_version:4:2}"
            tensorrt_version="$((10#$mj)).$((10#$mn)).$((10#$pt))"
        fi
    fi
    if [ -z "$tensorrt_version" ]; then
        tensorrt_version=$(resolve_ngc_tag_tensorrt_version "$ngc_tag" 2>/dev/null || true)
    fi

    local actual_driver=""
    actual_driver=$(detect_driver_version 2>/dev/null || true)

    # Probe device index for SM detection: from manifest, defaulting to 0 if auto/all.
    local probe_device="$build_device"
    if [ -z "$probe_device" ] || [ "$probe_device" = "auto" ] || [ "$probe_device" = "all" ]; then
        probe_device="0"
    fi
    probe_device="${probe_device#cuda:}"
    probe_device="${probe_device#device=}"

    local actual_cc=""
    if command -v nvidia-smi &>/dev/null; then
        actual_cc=$(nvidia-smi --id="$probe_device" --query-gpu=compute_cap \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    fi
    local actual_sm=""
    if [ -n "$actual_cc" ]; then
        actual_sm="sm_${actual_cc//./}"
    fi

    local cuda_version
    cuda_version=$(cross_host_json_value "$target_profile" cuda_runtime 2>/dev/null || true)

    python3 - "$bundle_root" "$build_manifest" "$target_profile" \
        "$tensorrt_version" "$actual_driver" "$actual_sm" "$cuda_version" <<'PY'
import datetime as dt
import hashlib
import json
import socket
import sys
from pathlib import Path

root = Path(sys.argv[1])
build = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
target = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
tensorrt_version, driver, sm, cuda_version = sys.argv[4:8]

engines = {}
for engine in sorted((root / "workspace" / "exported").glob("**/*.engine")):
    rel = engine.relative_to(root / "workspace" / "exported").as_posix()
    h = hashlib.sha256()
    with engine.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    engines[rel] = {"sha256": h.hexdigest(), "size": engine.stat().st_size}

gpus = target.get("gpus") or []
gpu = gpus[0] if gpus else {}
manifest = {
    "artifact_schema_version": 1,
    "built_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    "build_host": socket.gethostname(),
    "ngc_tag": build["ngc_tag"],
    "ngc_image": build["ngc_image"],
    "tensorrt_version": tensorrt_version,
    "cuda_version": cuda_version,
    "gpu_sm": sm,
    "gpu_name": gpu.get("name", ""),
    "driver_version": driver,
    "engine_dtype": build["engine_dtype"],
    "triton_io_float_dtype": build["triton_io_float_dtype"],
    "max_batch_size": build["max_batch_size"],
    "max_input_len": build["max_input_len"],
    "max_seq_len": build["max_seq_len"],
    "engines": engines,
}
(root / "artifact_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
    log_info "Artifact manifest written: $bundle_root/artifact_manifest.json"
}

# ---------------------------------------------------------------------------
#  collect_local_engines_to_workspace <bundle_root> <exported_dir>
#  Copies *.engine files and the artifact_manifest.json from the bundle
#  back into the canonical workspace/exported/ tree.  Used by the local
#  build path which works in a temp/cache directory.
# ---------------------------------------------------------------------------
collect_local_engines_to_workspace() {
    local bundle_root="$1"
    local exported_dir="$2"
    [ -d "$bundle_root/workspace/exported" ] || {
        log_error "Bundle workspace/exported missing: $bundle_root"
        return 1
    }
    mkdir -p "$exported_dir"

    # rsync if available for clean delta copy; fall back to cp -a
    if command -v rsync &>/dev/null; then
        rsync -a --include='*/' --include='*.engine' --include='triton_manifest.json' \
            --include='.engine_dtype' --exclude='*' \
            "$bundle_root/workspace/exported/" "$exported_dir/"
    else
        local d
        for d in "$bundle_root/workspace/exported"/*/; do
            [ -d "$d" ] || continue
            local name
            name=$(basename "$d")
            mkdir -p "$exported_dir/$name"
            cp -a "$d"*.engine "$exported_dir/$name/" 2>/dev/null || true
            if [ -f "$d/triton_manifest.json" ]; then
                cp -a "$d/triton_manifest.json" "$exported_dir/$name/"
            fi
        done
    fi

    if [ -f "$bundle_root/artifact_manifest.json" ]; then
        cp "$bundle_root/artifact_manifest.json" "$exported_dir/artifact_manifest.json"
    fi
    if [ -f "$bundle_root/workspace/exported/.engine_dtype" ]; then
        cp "$bundle_root/workspace/exported/.engine_dtype" "$exported_dir/.engine_dtype"
    fi
    log_info "Engines collected into: $exported_dir"
}

# ---------------------------------------------------------------------------
#  pack_engine_artifact_bundle <bundle_root> <out_tar_zst>
#  Used by build_on_target.sh (cross-host path).  Writes a tarball that
#  contains:
#    artifact_manifest.json
#    exported/<variant>/*.engine + triton_manifest.json
#    exported/tokenizer/*.engine
#    exported/.engine_dtype
# ---------------------------------------------------------------------------
pack_engine_artifact_bundle() {
    local bundle_root="$1"
    local out="$2"
    [ -f "$bundle_root/artifact_manifest.json" ] || {
        log_error "artifact_manifest.json missing in bundle root: $bundle_root"
        return 1
    }

    local tmp
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' RETURN
    mkdir -p "$tmp/exported"

    cp "$bundle_root/artifact_manifest.json" "$tmp/artifact_manifest.json"

    # Collect engines per variant
    local engine_dtype
    engine_dtype=$(cross_host_json_value "$bundle_root/build_manifest.json" engine_dtype || echo "bf16")
    local v_dir
    for v_dir in "$bundle_root/workspace/exported"/*/; do
        [ -d "$v_dir" ] || continue
        local v
        v=$(basename "$v_dir")
        mkdir -p "$tmp/exported/$v"
        cp -a "$v_dir"*.engine "$tmp/exported/$v/" 2>/dev/null || true
        if [ -f "$v_dir/triton_manifest.json" ]; then
            cp -a "$v_dir/triton_manifest.json" "$tmp/exported/$v/"
        fi
    done

    echo "$engine_dtype" > "$tmp/exported/.engine_dtype"

    mkdir -p "$(dirname "$out")"
    local tar_cmd
    tar_cmd=$(_cross_host_tar_cmd)
    (cd "$tmp" && $tar_cmd -cf "$out" .)
    log_info "Engine artifact bundle written: $out"
}
