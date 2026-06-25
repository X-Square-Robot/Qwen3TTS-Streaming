#!/bin/bash
# ===========================================================================
#  probe_target.sh — Capture target GPU/driver profile for cross-host builds
#
#  Now also collects host_environment info (trtexec / cuda / cudnn / nvcc /
#  driver module / docker availability / which runner will be selected by
#  the unified build pipeline) so the deploy-side fingerprint check can
#  validate the actual compile environment.
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../.." && pwd))"
source "${SCRIPT_DIR}/tools.sh"

OUT=""

usage() {
    cat << 'EOF'
Usage: probe_target.sh [--out target_profile.json]

Captures the production/target machine GPU and driver profile.  Use the JSON
on the export/package machine as --target-profile when creating an engine build
bundle or running remote-ssh builds.

The output JSON includes a host_environment section detailing the trtexec
binary, TensorRT/CUDA/cuDNN versions, NVIDIA driver module version, docker
availability and which runner the unified build pipeline would select on
this host.  These fields are used by the deploy-side fingerprint check.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) log_error "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if ! command -v nvidia-smi &>/dev/null; then
    log_error "nvidia-smi not found; cannot probe target GPU profile"
    log_error "Run this command on the production-like GPU host, then copy target_profile.json back."
    exit 1
fi

driver_version=$(detect_driver_version) || {
    log_error "Cannot detect a valid NVIDIA driver version from nvidia-smi."
    log_error "Run this command on the production-like GPU host. If this is that host, fix NVML/driver first."
    exit 1
}
recommended_ngc_tag=$(resolve_ngc_tag "$driver_version") || exit 1

# Resolve host environment up front so probe is fully self-contained.
host_trtexec_path=""
host_trtexec_version=""
if host_trtexec_path=$(probe_host_trtexec_path 2>/dev/null); then
    host_trtexec_version=$(probe_host_trtexec_version 2>/dev/null || true)
fi

has_docker="false"
docker_has_gpu="false"
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    has_docker="true"
    if docker info 2>/dev/null | grep -qi 'nvidia'; then
        docker_has_gpu="true"
    elif command -v nvidia-container-cli &>/dev/null; then
        docker_has_gpu="true"
    elif [ -f /etc/nvidia-container-runtime/config.toml ]; then
        docker_has_gpu="true"
    fi
fi

selected_runner=""
if selected_runner=$(BUILD_RUNNER=auto resolve_trtexec_runner 2>/dev/null); then
    :
else
    selected_runner="none"
fi

nvcc_version=""
if command -v nvcc &>/dev/null; then
    nvcc_version=$(nvcc --version 2>/dev/null \
        | grep -oE 'release [0-9]+\.[0-9]+(\.[0-9]+)?' \
        | head -1 | awk '{print $2}' || true)
fi

cudnn_version=""
for h in /usr/include/cudnn_version.h /usr/include/x86_64-linux-gnu/cudnn_version.h; do
    if [ -f "$h" ]; then
        cudnn_major=$(awk '/CUDNN_MAJOR/{print $3; exit}' "$h" 2>/dev/null || true)
        cudnn_minor=$(awk '/CUDNN_MINOR/{print $3; exit}' "$h" 2>/dev/null || true)
        cudnn_patch=$(awk '/CUDNN_PATCHLEVEL/{print $3; exit}' "$h" 2>/dev/null || true)
        if [ -n "$cudnn_major" ]; then
            cudnn_version="${cudnn_major}.${cudnn_minor:-0}.${cudnn_patch:-0}"
            break
        fi
    fi
done
if [ -z "$cudnn_version" ] && command -v dpkg &>/dev/null; then
    cudnn_version=$(dpkg -l 2>/dev/null | awk '/libcudnn[0-9]+ +/{print $3; exit}' | cut -d- -f1 || true)
fi

nvidia_module_version=""
if [ -f /proc/driver/nvidia/version ]; then
    nvidia_module_version=$(grep -oE 'Module +[0-9]+(\.[0-9]+)+' /proc/driver/nvidia/version \
        | head -1 | awk '{print $2}' || true)
fi

python3 - "$driver_version" "$recommended_ngc_tag" "$OUT" \
    "$host_trtexec_path" "$host_trtexec_version" \
    "$has_docker" "$docker_has_gpu" "$selected_runner" \
    "$nvcc_version" "$cudnn_version" "$nvidia_module_version" <<'PY'
import csv
import datetime as dt
import json
import os
import socket
import subprocess
import sys

(driver_version, recommended_ngc_tag, out_path,
 host_trtexec_path, host_trtexec_version,
 has_docker, docker_has_gpu, selected_runner,
 nvcc_version, cudnn_version, nvidia_module_version) = sys.argv[1:12]

def run(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()

cuda_runtime = ""
try:
    text = run(["nvidia-smi"])
    marker = "CUDA Version:"
    if marker in text:
        cuda_runtime = text.split(marker, 1)[1].split("|", 1)[0].strip()
except Exception:
    pass

rows = []
try:
    raw = run([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,compute_cap,memory.total",
        "--format=csv,noheader,nounits",
    ])
    for row in csv.reader(raw.splitlines()):
        if len(row) < 5:
            continue
        index, uuid, name, cc, memory = [item.strip() for item in row[:5]]
        cc_norm = cc.replace(".", "")
        rows.append({
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "compute_capability": cc,
            "sm": f"sm_{cc_norm}",
            "memory_total_mib": int(float(memory)),
        })
except Exception as exc:
    print(f"failed to query GPU profile: {exc}", file=sys.stderr)
    sys.exit(1)

profile = {
    "profile_schema_version": 2,
    "host": socket.gethostname(),
    "captured_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    "driver_version": driver_version,
    "cuda_runtime": cuda_runtime,
    "recommended_ngc_tag": recommended_ngc_tag,
    "ngc_compatible_tags": [recommended_ngc_tag],
    "gpus": rows,
    "host_environment": {
        "trtexec_path": host_trtexec_path or "",
        "trtexec_version": host_trtexec_version or "",
        "cuda_runtime": cuda_runtime,
        "cudnn_version": cudnn_version or "",
        "nvcc_version": nvcc_version or "",
        "nvidia_module_version": nvidia_module_version or "",
        "has_docker": has_docker == "true",
        "docker_has_gpu": docker_has_gpu == "true",
        "selected_runner": selected_runner or "",
    },
}

payload = json.dumps(profile, indent=2, ensure_ascii=False) + "\n"
if out_path:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload)
else:
    print(payload, end="")
PY

if [ -n "$OUT" ]; then
    log_info "Target profile written: $OUT"
    log_info "  driver:         $driver_version"
    log_info "  recommended NGC: $recommended_ngc_tag"
    log_info "  host trtexec:   ${host_trtexec_path:-<not found>}${host_trtexec_version:+ ($host_trtexec_version)}"
    log_info "  has docker:     $has_docker (with GPU runtime: $docker_has_gpu)"
    log_info "  runner choice:  $selected_runner"
fi
