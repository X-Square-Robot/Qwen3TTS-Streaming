#!/usr/bin/env python3
# ===========================================================================
#  _probe_standalone.py — Self-contained target profile probe
#
#  Designed to be copy-pasted into a target host that does NOT have access
#  to the qwen3-tts-triton repo (e.g. an Aliyun DSW container without SSH).
#  Only depends on python3 and nvidia-smi.
#
#  Outputs the SAME schema as scripts/bash/probe_target.sh, so the result
#  is interchangeable with the SSH-based and local probe paths.
#
#  Usage on the target host:
#    python3 _probe_standalone.py --out /tmp/target_profile.json
#    cat /tmp/target_profile.json   # then paste back to the packaging machine
#
#  Or (no file write, dumps to stdout):
#    python3 _probe_standalone.py
# ===========================================================================
import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# NGC compatibility matrix.  Mirrors scripts/bash/ngc_matrix.conf format:
# "tag  min_driver  tensorrt_version  cuda_version  python_version  size_gb"
# Update by running `bash scripts/bash/autorun.sh update-matrix` on the
# packaging machine and re-generating the bundle / paste payload.
NGC_MATRIX = [
    "25.11  590.44   10.14.1.48  13.1  3.12  -",
    "25.05  575.51   10.10.0.31  12.9  3.12  -",
    "25.03  570.124  10.9.0.34   12.8  3.12  -",
    "24.07  555.42   10.2.0.19   12.5  3.10  -",
]


def run(args, **kw):
    """Run a subprocess, return stripped stdout (str). Empty on failure."""
    try:
        out = subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, **kw)
        return out.strip()
    except Exception:
        return ""


def parse_version(s):
    """Parse 'major.minor[.patch]'; missing components default to 0."""
    parts = [int(x) for x in re.findall(r"\d+", s or "")]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def driver_ge(installed, required):
    """Compare major.minor numerically (ignores patch)."""
    if not installed or not required:
        return False
    i = parse_version(installed)
    r = parse_version(required)
    return (i[0], i[1]) >= (r[0], r[1])


def detect_driver_version():
    """Echo NVIDIA driver version from nvidia-smi, or empty string."""
    if not shutil.which("nvidia-smi"):
        return ""
    out = run([
        "nvidia-smi", "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return ""
    ver = out.splitlines()[0].strip()
    if re.match(r"^\d+(\.\d+){1,2}$", ver):
        return ver
    return ""


def resolve_ngc_tag(driver_ver):
    """Return the newest matrix tag compatible with this driver."""
    for entry in NGC_MATRIX:
        parts = entry.split()
        if len(parts) < 2:
            continue
        tag, min_drv = parts[0], parts[1]
        if driver_ge(driver_ver, min_drv):
            return tag
    return ""


def probe_host_trtexec_path():
    """Find the first available trtexec binary on the host."""
    candidates = []
    env_override = os.environ.get("TRTEXEC_HOST")
    if env_override:
        candidates.append(env_override)
    candidates.extend([
        "/usr/src/tensorrt/bin/trtexec",
        "/opt/tritonserver/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
    ])
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    found = shutil.which("trtexec")
    if found:
        return found
    return ""


def probe_host_trtexec_version(bin_path):
    """Run `<bin> --version` and parse TensorRT version like 10.5.0."""
    if not bin_path:
        return ""
    try:
        raw = subprocess.check_output(
            [bin_path, "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except Exception:
        return ""
    # Match "[TensorRT v100500]" → 10.5.0
    m = re.search(r"\[TensorRT v(\d+)\]", raw)
    if m:
        v = m.group(1)
        if len(v) >= 6:
            return f"{int(v[:2])}.{int(v[2:4])}.{int(v[4:6])}"
    # Fallback: any dotted version
    m = re.search(r"\b(\d+\.\d+\.\d+(?:\.\d+)?)\b", raw)
    return m.group(1) if m else ""


def probe_nvcc_version():
    out = run(["nvcc", "--version"])
    m = re.search(r"release\s+(\d+\.\d+(?:\.\d+)?)", out or "")
    return m.group(1) if m else ""


def probe_cudnn_version():
    for header in (
        "/usr/include/cudnn_version.h",
        "/usr/include/x86_64-linux-gnu/cudnn_version.h",
    ):
        p = Path(header)
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        def pick(name, default="0"):
            m = re.search(rf"#define\s+{name}\s+(\d+)", text)
            return m.group(1) if m else default
        maj = pick("CUDNN_MAJOR", "")
        if maj:
            return f"{maj}.{pick('CUDNN_MINOR', '0')}.{pick('CUDNN_PATCHLEVEL', '0')}"
    out = run(["dpkg", "-l"])
    m = re.search(r"libcudnn\d+\s+(\S+)", out)
    return m.group(1).split("-", 1)[0] if m else ""


def probe_nvidia_module_version():
    p = Path("/proc/driver/nvidia/version")
    if not p.is_file():
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    m = re.search(r"Module\s+(\d+(?:\.\d+)+)", text)
    return m.group(1) if m else ""


def probe_docker():
    if not shutil.which("docker"):
        return False, False
    try:
        info = subprocess.check_output(
            ["docker", "info"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return False, False
    has_gpu = bool(re.search(r"nvidia", info, re.IGNORECASE))
    if not has_gpu and shutil.which("nvidia-container-cli"):
        has_gpu = True
    if not has_gpu and os.path.isfile("/etc/nvidia-container-runtime/config.toml"):
        has_gpu = True
    return True, has_gpu


def resolve_selected_runner(has_docker, docker_has_gpu, trtexec_path):
    """Mirror lib/trtexec_runner.sh::resolve_trtexec_runner (BUILD_RUNNER=auto)."""
    if has_docker and docker_has_gpu:
        return "docker"
    if trtexec_path and shutil.which("nvidia-smi"):
        return "host"
    return "none"


def collect_gpus():
    rows = []
    raw = run([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,compute_cap,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return rows
    for row in csv.reader(io.StringIO(raw)):
        if len(row) < 5:
            continue
        index, uuid, name, cc, memory = [x.strip() for x in row[:5]]
        try:
            rows.append({
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "compute_capability": cc,
                "sm": f"sm_{cc.replace('.', '')}",
                "memory_total_mib": int(float(memory)),
            })
        except Exception:
            continue
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Standalone target profile probe (paste-friendly).",
    )
    parser.add_argument("--out", default="", help="Write JSON to this path")
    args = parser.parse_args()

    if not shutil.which("nvidia-smi"):
        sys.stderr.write("[ERROR] nvidia-smi not found; run this on the GPU host\n")
        return 1

    driver = detect_driver_version()
    if not driver:
        sys.stderr.write("[ERROR] Cannot detect NVIDIA driver from nvidia-smi\n")
        return 1
    ngc_tag = resolve_ngc_tag(driver)
    if not ngc_tag:
        sys.stderr.write(
            f"[ERROR] Driver {driver} too old — no compatible NGC entry\n"
        )
        return 1

    gpus = collect_gpus()
    if not gpus:
        sys.stderr.write("[ERROR] nvidia-smi returned no GPUs\n")
        return 1

    trtexec_path = probe_host_trtexec_path()
    trtexec_version = probe_host_trtexec_version(trtexec_path) if trtexec_path else ""
    nvcc_version = probe_nvcc_version()
    cudnn_version = probe_cudnn_version()
    nvidia_module_version = probe_nvidia_module_version()
    has_docker, docker_has_gpu = probe_docker()
    selected_runner = resolve_selected_runner(has_docker, docker_has_gpu, trtexec_path)

    # CUDA runtime from nvidia-smi
    smi_text = run(["nvidia-smi"])
    cuda_runtime = ""
    m = re.search(r"CUDA Version:\s*([0-9.]+)", smi_text or "")
    if m:
        cuda_runtime = m.group(1)

    profile = {
        "profile_schema_version": 2,
        "host": socket.gethostname(),
        "captured_at_utc": dt.datetime.now(dt.timezone.utc)
                          .replace(microsecond=0).isoformat(),
        "driver_version": driver,
        "cuda_runtime": cuda_runtime,
        "recommended_ngc_tag": ngc_tag,
        "ngc_compatible_tags": [ngc_tag],
        "gpus": gpus,
        "host_environment": {
            "trtexec_path": trtexec_path,
            "trtexec_version": trtexec_version,
            "cuda_runtime": cuda_runtime,
            "cudnn_version": cudnn_version,
            "nvcc_version": nvcc_version,
            "nvidia_module_version": nvidia_module_version,
            "has_docker": has_docker,
            "docker_has_gpu": docker_has_gpu,
            "selected_runner": selected_runner,
        },
    }

    payload = json.dumps(profile, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload)
        sys.stderr.write(f"[INFO] Target profile written: {out_path}\n")
        sys.stderr.write(f"[INFO]   driver:         {driver}\n")
        sys.stderr.write(f"[INFO]   recommended NGC: {ngc_tag}\n")
        sys.stderr.write(
            f"[INFO]   host trtexec:    "
            f"{trtexec_path or '<not found>'}"
            f"{f' ({trtexec_version})' if trtexec_version else ''}\n"
        )
        sys.stderr.write(
            f"[INFO]   has docker:      {has_docker} "
            f"(with GPU runtime: {docker_has_gpu})\n"
        )
        sys.stderr.write(f"[INFO]   runner choice:   {selected_runner}\n")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
