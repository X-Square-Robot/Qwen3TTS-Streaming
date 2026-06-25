#!/bin/bash
# ===========================================================================
#  cross_host.sh — Cross-host TensorRT engine build helpers
# ===========================================================================

[[ -n "${_LIB_CROSS_HOST_LOADED:-}" ]] && return 0
_LIB_CROSS_HOST_LOADED=1

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/logging.sh"
source "${_LIB_DIR}/utils.sh"
source "${_LIB_DIR}/docker.sh"

cross_host_json_value() {
    local path="$1"
    local expr="$2"
    python3 - "$path" "$expr" <<'PY' 2>/dev/null
import json
import sys

path, expr = sys.argv[1:3]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
cur = data
for part in expr.split("."):
    if not part:
        continue
    if part.endswith("]") and "[" in part:
        name, idx = part[:-1].split("[", 1)
        if name:
            cur = cur[name]
        cur = cur[int(idx)]
    else:
        cur = cur[part]
if cur is None:
    sys.exit(1)
if isinstance(cur, (dict, list)):
    print(json.dumps(cur, ensure_ascii=False))
else:
    print(cur)
PY
}

resolve_ngc_tag_from_profile() {
    local profile="$1"
    if [ ! -f "$profile" ]; then
        log_error "Target profile not found: $profile"
        return 1
    fi
    local tag
    tag=$(cross_host_json_value "$profile" "recommended_ngc_tag" || true)
    if [ -z "$tag" ]; then
        log_error "target_profile.json has no recommended_ngc_tag: $profile"
        return 1
    fi
    if ! resolve_ngc_entry_by_tag "$tag" >/dev/null; then
        return 1
    fi
    echo "$tag"
}

target_profile_memory_mb() {
    local profile="$1"
    local gpu_index="${2:-0}"
    python3 - "$profile" "$gpu_index" <<'PY' 2>/dev/null || echo "0"
import json
import sys

path, gpu_index = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as f:
    data = json.load(f)
gpus = data.get("gpus") or []
if not gpus:
    print(0)
    raise SystemExit
selected = next((g for g in gpus if int(g.get("index", -1)) == gpu_index), gpus[0])
print(int(selected.get("memory_total_mib") or 0))
PY
}

_cross_host_tar_cmd() {
    if command -v tar &>/dev/null && tar --help 2>/dev/null | grep -q -- "--zstd"; then
        echo "tar --zstd"
    else
        echo "tar -z"
    fi
}

_cross_host_bundle_ext_note() {
    if tar --help 2>/dev/null | grep -q -- "--zstd"; then
        echo "zstd"
    else
        echo "gzip"
    fi
}

make_engine_build_bundle() {
    local repo_root="$1"
    local exported_dir="$2"
    local out="$3"
    local target_profile="$4"
    local variant_csv="$5"
    local engine_dtype="$6"
    local triton_io_float_dtype="$7"
    local max_batch_size="$8"
    local max_input_len="$9"
    local max_seq_len="${10}"
    local build_gpu_device="${11:-auto}"

    if [ ! -f "$target_profile" ]; then
        log_error "Target profile not found: $target_profile"
        return 1
    fi
    [ -d "$exported_dir" ] || { log_error "Exported dir not found: $exported_dir"; return 1; }

    local ngc_tag ngc_image
    ngc_tag=$(resolve_ngc_tag_from_profile "$target_profile") || return 1
    ngc_image=$(resolve_ngc_image_from_tag "$ngc_tag") || return 1

    local tmp
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' RETURN
    mkdir -p "$tmp/bundle/workspace/exported"

    cp "$target_profile" "$tmp/bundle/target_profile.json"
    cp "$repo_root/scripts/bash/build_on_target.sh" "$tmp/bundle/build_on_target.sh"
    chmod +x "$tmp/bundle/build_on_target.sh"
    # When CROSS_HOST_BUNDLE_SCRIPTS=symlink, link instead of copy — used by
    # the local build path (prepare_local_bundle_workspace) to avoid a
    # multi-MB scripts/ copy on every build.  Cross-host / SSH paths leave
    # this unset so the bundle is fully self-contained.
    if [ "${CROSS_HOST_BUNDLE_SCRIPTS:-copy}" = "symlink" ]; then
        ln -s "$repo_root/scripts" "$tmp/bundle/scripts"
    else
        cp -R "$repo_root/scripts" "$tmp/bundle/scripts"
    fi

    IFS=',' read -r -a variants <<< "$variant_csv"
    local variant
    for variant in "${variants[@]}"; do
        [ -n "$variant" ] || continue
        if [ ! -d "$exported_dir/$variant" ]; then
            log_error "Variant export dir not found: $exported_dir/$variant"
            return 1
        fi
        mkdir -p "$tmp/bundle/workspace/exported/$variant"
        cp -a "$exported_dir/$variant"/*.onnx "$tmp/bundle/workspace/exported/$variant/" 2>/dev/null || true
        cp -a "$exported_dir/$variant"/*.onnx.data "$tmp/bundle/workspace/exported/$variant/" 2>/dev/null || true
        cp -a "$exported_dir/$variant/triton_manifest.json" "$tmp/bundle/workspace/exported/$variant/" 2>/dev/null || true
        [ -d "$exported_dir/$variant/weights" ] && cp -a "$exported_dir/$variant/weights" "$tmp/bundle/workspace/exported/$variant/weights"
    done
    if [ -d "$exported_dir/tokenizer" ]; then
        mkdir -p "$tmp/bundle/workspace/exported/tokenizer"
        cp -a "$exported_dir/tokenizer"/*.onnx "$tmp/bundle/workspace/exported/tokenizer/" 2>/dev/null || true
        cp -a "$exported_dir/tokenizer"/*.onnx.data "$tmp/bundle/workspace/exported/tokenizer/" 2>/dev/null || true
    fi

    python3 - "$tmp/bundle/build_manifest.json" "$variant_csv" "$engine_dtype" "$triton_io_float_dtype" \
        "$max_batch_size" "$max_input_len" "$max_seq_len" "$ngc_tag" "$ngc_image" "$build_gpu_device" <<'PY'
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

    # Friendly DSW / cross-host entry point.  Wraps build_on_target.sh
    # with auto-runner resolution (BUILD_RUNNER=auto by default), so DSW
    # users that have host trtexec but no docker can just run "bash run.sh".
    cat > "$tmp/bundle/run.sh" <<'EOF'
#!/bin/bash
# ===========================================================================
#  run.sh — One-shot entry point for engine_build_bundle on a target host.
#  Usage:
#    bash run.sh                       # auto-detect runner (docker|host)
#    BUILD_RUNNER=host bash run.sh     # force host trtexec (DSW etc.)
#    BUILD_RUNNER=docker bash run.sh   # force docker NGC trtexec
#    TRTEXEC_HOST=/path/to/trtexec bash run.sh
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/build_on_target.sh" "$@"
EOF
    chmod +x "$tmp/bundle/run.sh"

    cat > "$tmp/bundle/README.md" <<EOF
# Qwen3-TTS TensorRT Engine Build Bundle

Run on the target production-like GPU host:

\`\`\`bash
bash run.sh
\`\`\`

This is a thin wrapper around \`build_on_target.sh\` that auto-detects
whether to compile via docker (NGC container) or directly with host
trtexec.  On Aliyun DSW or any container without docker, the bundle
will use the host trtexec found at \`/usr/src/tensorrt/bin/trtexec\`
(override with \`TRTEXEC_HOST=\`).

The script writes \`engine_artifact_bundle.tar.zst\` in the current directory.
Copy that artifact back to the packaging machine and run:

\`\`\`bash
bash scripts/bash/autorun.sh import-artifact engine_artifact_bundle.tar.zst
\`\`\`
EOF

    mkdir -p "$(dirname "$out")"
    local tar_cmd
    tar_cmd=$(_cross_host_tar_cmd)
    (cd "$tmp/bundle" && $tar_cmd -cf "$out" .)
    log_info "Engine build bundle written: $out ($(_cross_host_bundle_ext_note))"
}

extract_engine_artifact_bundle() {
    local repo_root="$1"
    local exported_dir="$2"
    local artifact="$3"
    # Strict by default — callers must explicitly pass "false" (and acknowledge
    # ALLOW_FINGERPRINT_MISMATCH=1 elsewhere) to bypass the fingerprint check.
    local strict="${4:-true}"

    [ -f "$artifact" ] || { log_error "Engine artifact bundle not found: $artifact"; return 1; }
    local tmp
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' RETURN
    local tar_cmd
    tar_cmd=$(_cross_host_tar_cmd)
    $tar_cmd -xf "$artifact" -C "$tmp"

    local manifest="$tmp/artifact_manifest.json"
    [ -f "$manifest" ] || { log_error "artifact_manifest.json missing in artifact bundle"; return 1; }

    if [ "$strict" = "true" ]; then
        engine_fingerprint_check "$manifest" "$tmp/exported" || return 1
    fi

    mkdir -p "$exported_dir"
    if [ -d "$tmp/exported" ]; then
        cp -a "$tmp/exported/." "$exported_dir/"
    else
        log_error "exported/ missing in artifact bundle"
        return 1
    fi
    cp "$manifest" "$exported_dir/artifact_manifest.json"
    local dtype
    dtype=$(cross_host_json_value "$manifest" "engine_dtype" || echo "bf16")
    echo "${dtype:-bf16}" > "$exported_dir/.engine_dtype"
    log_info "Imported TensorRT engine artifact into: $exported_dir"
}

engine_fingerprint_check() {
    local artifact_manifest="$1"
    local exported_dir_or_manifest="$2"
    # Optional: a target_profile.json to cross-validate driver / SM / trtexec.
    local target_profile="${3:-}"

    if [ ! -f "$artifact_manifest" ]; then
        log_error "Engine artifact manifest missing: $artifact_manifest"
        log_error "  Run 'autorun.sh build' or 'autorun.sh import-artifact <bundle>' first."
        return 1
    fi

    python3 - "$artifact_manifest" "$exported_dir_or_manifest" "${target_profile:-}" <<'PY'
import json
import sys
from pathlib import Path

artifact_path = Path(sys.argv[1])
target = Path(sys.argv[2])
target_profile_arg = sys.argv[3] if len(sys.argv) > 3 else ""
artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)

def mm(v):
    parts = str(v or "").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(v or "")

manifests = []
if target.is_file():
    manifests = [target]
elif target.is_dir():
    manifests = sorted(target.glob("*/triton_manifest.json"))

errors = []
if not manifests:
    # No triton_manifest.json to cross-check against — that is fine for
    # the cross-host import path which validates against target_profile
    # instead.  Just warn so callers know we relied on the artifact alone.
    print(f"NOTE: no triton_manifest.json found under {target} (artifact-only check)", file=sys.stderr)

for manifest_path in manifests:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = data.get("engine_profile") or {}
    expected = {
        "ngc_tag": profile.get("ngc_tag") or "",
        "engine_dtype": profile.get("engine_dtype") or data.get("engine_dtype") or "",
        "max_batch_size": profile.get("max_batch_size"),
        "max_input_len": profile.get("max_input_len"),
        "max_seq_len": profile.get("max_seq_len"),
    }
    for key in ("engine_dtype", "max_batch_size", "max_input_len", "max_seq_len"):
        if str(expected.get(key) or "") and str(expected.get(key)) != str(artifact.get(key)):
            errors.append(f"{manifest_path}: {key} expected={expected.get(key)} artifact={artifact.get(key)}")
    if expected["ngc_tag"] and expected["ngc_tag"] != str(artifact.get("ngc_tag") or ""):
        errors.append(f"{manifest_path}: ngc_tag expected={expected['ngc_tag']} artifact={artifact.get('ngc_tag')}")
    expected_trt = profile.get("tensorrt_version") or ""
    if expected_trt and mm(expected_trt) != mm(artifact.get("tensorrt_version")):
        errors.append(f"{manifest_path}: tensorrt_version expected={expected_trt} artifact={artifact.get('tensorrt_version')}")
    expected_sm = profile.get("gpu_sm") or ""
    if expected_sm and expected_sm != str(artifact.get("gpu_sm") or ""):
        errors.append(f"{manifest_path}: gpu_sm expected={expected_sm} artifact={artifact.get('gpu_sm')}")

# Cross-validate against target_profile.json when provided.  This catches
# the "engines compiled on the wrong host (wrong SM / trtexec)" case.
if target_profile_arg:
    tp = Path(target_profile_arg)
    if tp.is_file():
        tp_data = json.loads(tp.read_text(encoding="utf-8"))
        gpus = tp_data.get("gpus") or []
        tp_sm = (gpus[0] if gpus else {}).get("sm") or ""
        if tp_sm and tp_sm != str(artifact.get("gpu_sm") or ""):
            errors.append(f"target_profile: gpu_sm expected={tp_sm} artifact={artifact.get('gpu_sm')}")
        host_env = tp_data.get("host_environment") or {}
        tp_trt = host_env.get("trtexec_version") or ""
        if tp_trt and mm(tp_trt) != mm(artifact.get("tensorrt_version")):
            errors.append(
                f"target_profile: trtexec_version expected={tp_trt} artifact={artifact.get('tensorrt_version')}"
            )

if errors:
    print("Engine fingerprint mismatch:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    raise SystemExit(1)
print("Engine fingerprint check: OK")
PY
}
