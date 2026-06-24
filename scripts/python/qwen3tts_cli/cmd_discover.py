"""Discover-target subcommand — acquire target_profile.json from production target.

Supports three modes:
  --local        Run probe on the current host
  --remote-host  SSH to the target machine and run probe there
  --paste        Print a standalone probe script for the user to run on DSW
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_discover_target(args: argparse.Namespace) -> int:
    """Dispatch discover-target based on mode."""
    mode = getattr(args, "discover_mode", "") or "local"
    output = Path(getattr(args, "output", "workspace/target_profile.json"))

    dispatch = {
        "local": _discover_local,
        "remote": _discover_remote,
        "paste": _discover_paste,
    }
    handler = dispatch.get(mode)
    if handler is None:
        print(f"Error: Unknown discover-target mode: {mode}", file=sys.stderr)
        return 1
    return handler(output, args)


def _discover_local(output: Path, args: object) -> int:
    """Run probe on the local host."""
    from qwen3tts_cli.cmd_probe import run_probe

    # Construct a probe args namespace
    probe_args = argparse.Namespace(output=str(output))
    return run_probe(probe_args)


def _discover_remote(output: Path, args: object) -> int:
    """SSH to the remote host and run the probe there."""
    remote_host = getattr(args, "remote_host", "")
    if not remote_host:
        print("Error: --remote-host is required for remote discovery.", file=sys.stderr)
        return 1

    remote_workdir = getattr(args, "remote_workdir", "/tmp/qwen3-tts-engine-build")

    print(f"Discovering target profile via SSH: {remote_host}")

    # Prepare remote directory
    result = subprocess.run(
        ["ssh", remote_host, f"mkdir -p '{remote_workdir}'"],
    )
    if result.returncode != 0:
        print(f"Error: Failed to SSH to {remote_host}.", file=sys.stderr)
        return 1

    # Create a standalone probe script on the remote host
    probe_script = _generate_standalone_probe_script()
    remote_probe_path = f"{remote_workdir}/_probe_standalone.py"

    # Upload the probe script
    result = subprocess.run(
        ["ssh", remote_host, f"cat > '{remote_probe_path}' << 'PROBE_EOF'\n{probe_script}\nPROBE_EOF"],
    )
    if result.returncode != 0:
        print(f"Error: Failed to upload probe script to {remote_host}.", file=sys.stderr)
        return 1

    # Run the standalone probe on the remote
    result = subprocess.run(
        ["ssh", remote_host, f"python3 '{remote_probe_path}' --out '{remote_workdir}/target_profile.json'"],
    )
    if result.returncode != 0:
            print(f"Error: Remote probe script failed.", file=sys.stderr)
            return 1

    # Download the result
    result = subprocess.run(
        ["scp", f"{remote_host}:{remote_workdir}/target_profile.json", str(output)],
    )
    if result.returncode != 0:
        print(f"Error: Failed to download target_profile.json from {remote_host}.", file=sys.stderr)
        return 1

    # Validate the downloaded profile
    try:
        with open(output, encoding="utf-8") as f:
            profile = json.load(f)
        if "driver_version" not in profile:
            print(f"Warning: {output} does not contain driver_version", file=sys.stderr)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: Invalid profile downloaded: {e}", file=sys.stderr)
        return 1

    print(f"Target profile retrieved from {remote_host} → {output}")
    _print_profile_summary(output)
    return 0


def _discover_paste(output: Path, args: object) -> int:
    """Print a standalone probe script and accept pasted JSON output."""
    probe_script = _generate_standalone_probe_script()

    print()
    print("  Paste mode (适用于 DSW 没 SSH 的场景):")
    print("    1) 在目标机器（DSW / 容器）打开终端")
    print("    2) 把下方 Python 探测脚本完整粘贴并保存为 probe.py")
    print("    3) 运行: python3 probe.py --out /tmp/target_profile.json")
    print("    4) cat /tmp/target_profile.json 内容粘贴回这里，Ctrl-D 结束")
    print()
    print("============================== BEGIN probe.py ==============================")
    print(probe_script)
    print("=============================== END probe.py ===============================")
    print()

    print("Paste the JSON output from DSW (Ctrl-D to finish):")
    try:
        pasted = sys.stdin.read()
    except KeyboardInterrupt:
        print()
        return 1

    if not pasted.strip():
        print("Error: Empty paste; aborting.", file=sys.stderr)
        return 1

    # Validate it's JSON, not the probe script itself
    try:
        data = json.loads(pasted)
        if not isinstance(data, dict) or "driver_version" not in data:
            raise ValueError("Missing driver_version key")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: Pasted content is not a valid target_profile JSON: {e}", file=sys.stderr)
        print("  Expected: the JSON output of 'python3 probe.py --out /tmp/...'", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(pasted, encoding="utf-8")
    print(f"Target profile saved: {output}")
    _print_profile_summary(output)
    return 0


def _generate_standalone_probe_script() -> str:
    """Generate a self-contained Python probe script for paste/remote mode."""
    return '''#!/usr/bin/env python3
"""Standalone GPU/driver probe — no project imports needed."""
import json
import subprocess
import sys
from pathlib import Path

def detect_driver():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip().split("\\n")[0].strip()
    except Exception:
        pass
    return ""

def detect_gpus():
    gpus = []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split("\\n"):
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
    except Exception:
        pass
    return gpus

def detect_docker():
    info = {"available": False, "nvidia_runtime": False}
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=5)
        info["available"] = r.returncode == 0
    except Exception:
        pass
    return info

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/target_profile.json")
    args = parser.parse_args()

    driver = detect_driver()
    if not driver:
        print("Error: Cannot detect NVIDIA driver.", file=sys.stderr)
        sys.exit(1)

    gpus = detect_gpus()
    if not gpus:
        print("Error: No GPUs detected.", file=sys.stderr)
        sys.exit(1)

    docker_info = detect_docker()

    profile = {
        "schema_version": 2,
        "driver_version": driver,
        "recommended_ngc_tag": "",
        "gpus": gpus,
        "host_environment": {
            "docker_available": docker_info["available"],
            "docker_nvidia_runtime": docker_info["nvidia_runtime"],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    print(f"Target profile written to: {out}")
    print(f"  Driver: {driver}")
    print(f"  GPUs:   {len(gpus)}")
    for gpu in gpus:
        print(f"    [{gpu[\\'index\\']}] {gpu[\\'name\\']} (SM {gpu[\\'compute_capability\\']}, {gpu[\\'memory_total_mib\\']} MiB)")

if __name__ == "__main__":
    main()
'''


def _print_profile_summary(profile_path: Path) -> None:
    """Print a brief summary of the target profile."""
    try:
        with open(profile_path, encoding="utf-8") as f:
            profile = json.load(f)
        driver = profile.get("driver_version", "?")
        ngc = profile.get("recommended_ngc_tag", "?")
        gpus = profile.get("gpus", [])
        sm = gpus[0].get("sm", "?") if gpus else "?"
        print(f"  driver: {driver}  ngc: {ngc}  sm: {sm}")
    except (OSError, json.JSONDecodeError):
        pass
