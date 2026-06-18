"""Project status diagnostics — Python replacement for ``scripts/bash/lib/status.sh``.

Provides functions to inspect the current state of the Qwen3-TTS project:
exported models, built engines, model repository, Docker images, and
running containers/services.

Usage from CLI::

    python -m qwen3tts_tools.status
    python -m qwen3tts_tools.status --json

Usage from Python::

    from qwen3tts_tools.status import check_all, format_status
    report = check_all()
    print(format_status(report))
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """Return the project repository root."""
    # Walk up from this file: scripts/python/qwen3tts_tools/status.py → repo root
    return Path(__file__).resolve().parents[3]


@dataclass
class VariantInfo:
    """Status of a single model variant."""
    variant: str
    exported: bool = False
    onnx_files: int = 0
    trt_engines: int = 0
    triton_manifest: bool = False
    model_repo: bool = False


@dataclass
class DockerInfo:
    """Docker-related status."""
    docker_available: bool = False
    triton_images: list[str] = field(default_factory=list)
    engine_images: list[str] = field(default_factory=list)
    running_containers: list[str] = field(default_factory=list)


@dataclass
class ServiceInfo:
    """Running service status."""
    triton_running: bool = False
    engine_standalone_running: bool = False
    ports: dict[str, int] = field(default_factory=dict)


@dataclass
class ProjectStatus:
    """Complete project status report."""
    repo_root: str = ""
    variants: list[VariantInfo] = field(default_factory=list)
    docker: DockerInfo = field(default_factory=DockerInfo)
    services: ServiceInfo = field(default_factory=ServiceInfo)
    errors: list[str] = field(default_factory=list)


def _check_variant(variant_dir: Path, model_repo_dir: Path, model_version: int) -> VariantInfo:
    """Check the status of a single variant directory."""
    info = VariantInfo(variant=variant_dir.name)
    if not variant_dir.is_dir():
        return info

    info.exported = True

    # Count ONNX files
    onnx_dir = variant_dir / "onnx"
    if onnx_dir.is_dir():
        info.onnx_files = sum(1 for f in onnx_dir.rglob("*.onnx"))

    # Count TRT engine files
    trt_count = 0
    for plan_dir in variant_dir.rglob("*.plan"):
        trt_count += 1
    # Also check trt/ subdirectory pattern
    trt_dir = variant_dir / "trt"
    if trt_dir.is_dir():
        trt_count += sum(1 for f in trt_dir.rglob("*.plan"))
    info.trt_engines = trt_count

    # Check for triton manifest
    manifest_path = variant_dir / "triton_manifest.json"
    info.triton_manifest = manifest_path.is_file()

    # Check model repository for this version
    version_dir = model_repo_dir / variant_dir.name / str(model_version)
    info.model_repo = version_dir.is_dir()

    return info


def _check_docker() -> DockerInfo:
    """Check Docker status and images."""
    info = DockerInfo()

    if not shutil.which("docker"):
        return info

    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        info.docker_available = result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return info

    if not info.docker_available:
        return info

    # List Triton images
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if "triton" in line.lower() or "qwen3" in line.lower():
                    if "triton" in line.lower():
                        info.triton_images.append(line)
                    if "qwen3" in line.lower():
                        info.engine_images.append(line)
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Check running containers
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info.running_containers = [
                name for name in result.stdout.strip().split("\n") if name
            ]
    except (subprocess.TimeoutExpired, OSError):
        pass

    return info


def _check_services() -> ServiceInfo:
    """Check if services are running."""
    info = ServiceInfo()

    # Check for running Triton container
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=triton", "--filter", "name=qwen3",
             "--format", "{{.Names}} {{.Ports}}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                if "triton" in line.lower():
                    info.triton_running = True
                if "qwen3" in line.lower() and "engine" in line.lower():
                    info.engine_standalone_running = True
                # Extract ports
                if "->" in line:
                    for part in line.split():
                        if ":" in part and "->" in part:
                            try:
                                host_port = int(part.split(":")[-1].split("->")[0])
                                container_port = int(part.split("->")[-1].split("/")[0])
                                info.ports[f"container_{container_port}"] = host_port
                            except (ValueError, IndexError):
                                pass
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Check for standalone engine process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "engine.server"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            info.engine_standalone_running = True
    except (subprocess.TimeoutExpired, OSError):
        pass

    return info


def check_all(
    exported_dir: Path | None = None,
    model_repo_dir: Path | None = None,
    model_version: int = 1,
) -> ProjectStatus:
    """Run all status checks and return a complete report.

    Args:
        exported_dir: Path to ``workspace/exported/``. Auto-detected if None.
        model_repo_dir: Path to ``workspace/model_repository/``. Auto-detected if None.
        model_version: Model version number to check (default: 1).
    """
    root = _repo_root()
    if exported_dir is None:
        exported_dir = root / "workspace" / "exported"
    if model_repo_dir is None:
        model_repo_dir = root / "workspace" / "model_repository"

    status = ProjectStatus(repo_root=str(root))

    # Check each variant in exported/
    if exported_dir.is_dir():
        for variant_path in sorted(exported_dir.iterdir()):
            if variant_path.is_dir() and not variant_path.name.startswith("."):
                status.variants.append(
                    _check_variant(variant_path, model_repo_dir, model_version)
                )

    # Check Docker
    status.docker = _check_docker()

    # Check running services
    status.services = _check_services()

    return status


def format_status(status: ProjectStatus) -> str:
    """Format a ProjectStatus as a human-readable string."""
    lines: list[str] = []

    lines.append(f"Repository: {status.repo_root}")
    lines.append("")

    # Variant status
    if status.variants:
        lines.append("Variants:")
        for v in status.variants:
            exported = "✓" if v.exported else "✗"
            trt = f"trt={v.trt_engines}" if v.trt_engines else ""
            onnx = f"onnx={v.onnx_files}" if v.onnx_files else ""
            manifest = "manifest" if v.triton_manifest else ""
            repo = "repo" if v.model_repo else ""
            details = ", ".join(filter(None, [onnx, trt, manifest, repo]))
            lines.append(f"  {v.variant}: {exported} {details}")
    else:
        lines.append("No variants found in workspace/exported/")

    lines.append("")

    # Docker status
    if status.docker.docker_available:
        lines.append("Docker: available")
        if status.docker.triton_images:
            lines.append("  Triton images:")
            for img in status.docker.triton_images:
                lines.append(f"    {img}")
        if status.docker.engine_images:
            lines.append("  Engine images:")
            for img in status.docker.engine_images:
                lines.append(f"    {img}")
    else:
        lines.append("Docker: not available")

    lines.append("")

    # Services
    if status.services.triton_running:
        lines.append("Triton: running")
    else:
        lines.append("Triton: not running")

    if status.services.engine_standalone_running:
        lines.append("Engine (standalone): running")
    else:
        lines.append("Engine (standalone): not running")

    if status.services.ports:
        lines.append("  Ports:")
        for name, port in status.services.ports.items():
            lines.append(f"    {name}: {port}")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-TTS project status diagnostics")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--exported-dir", type=Path, help="Path to workspace/exported/")
    parser.add_argument("--model-repo-dir", type=Path, help="Path to workspace/model_repository/")
    parser.add_argument("--model-version", type=int, default=1, help="Model version (default: 1)")
    args = parser.parse_args()

    status = check_all(
        exported_dir=args.exported_dir,
        model_repo_dir=args.model_repo_dir,
        model_version=args.model_version,
    )

    if args.json:
        print(json.dumps(asdict(status), indent=2))
    else:
        print(format_status(status))


if __name__ == "__main__":
    main()
