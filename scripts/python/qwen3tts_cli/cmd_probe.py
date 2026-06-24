"""Probe subcommand — capture target GPU/driver profile."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _detect_driver_version() -> str:
    """Detect NVIDIA driver version via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _detect_gpus() -> list[dict]:
    """Detect GPU info via nvidia-smi."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,uuid,name,compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    cc = parts[3].replace(".", "")
                    gpus.append({
                        "index": int(parts[0]),
                        "uuid": parts[1],
                        "name": parts[2],
                        "compute_capability": parts[3],
                        "sm": f"sm_{cc}",
                        "memory_total_mib": int(float(parts[4])),
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return gpus


def _detect_docker() -> dict:
    """Detect Docker availability."""
    info = {"available": False, "nvidia_runtime": False}
    if not shutil.which("docker"):
        return info
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        info["available"] = result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        pass
    return info


def run_probe(args: argparse.Namespace) -> int:
    """Probe target system and write target_profile.json."""
    from qwen3tts_tools.ngc_matrix import resolve_ngc_tag

    driver_version = _detect_driver_version()
    if not driver_version:
        print("Error: Cannot detect NVIDIA driver version. Is nvidia-smi available?", file=sys.stderr)
        return 1

    gpus = _detect_gpus()
    if not gpus:
        print("Error: No GPUs detected.", file=sys.stderr)
        return 1

    docker_info = _detect_docker()

    # Resolve recommended NGC tag
    ngc_tag = ""
    try:
        ngc_tag = resolve_ngc_tag(driver_version) or ""
    except Exception:
        pass

    profile = {
        "schema_version": 2,
        "driver_version": driver_version,
        "recommended_ngc_tag": ngc_tag,
        "gpus": gpus,
        "host_environment": {
            "docker_available": docker_info["available"],
            "docker_nvidia_runtime": docker_info["nvidia_runtime"],
        },
    }

    output_path = Path(getattr(args, "output", "workspace/target_profile.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False))

    print(f"Target profile written to: {output_path}")
    print(f"  Driver: {driver_version}")
    print(f"  GPUs:   {len(gpus)}")
    for gpu in gpus:
        print(f"    [{gpu['index']}] {gpu['name']} (SM {gpu['compute_capability']}, {gpu['memory_total_mib']} MiB)")
    print(f"  NGC:    {ngc_tag or '(no compatible tag)'}")
    return 0
