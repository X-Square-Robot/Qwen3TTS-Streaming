"""Docker / NGC container utilities — Python replacement for ``scripts/bash/lib/docker.sh``.

Provides functions to detect GPU/driver info, check Docker GPU readiness,
pull and build images, and inspect running containers.  The NGC compatibility
matrix is loaded via :mod:`qwen3tts_tools.ngc_matrix`.

Usage from CLI::

    python -m qwen3tts_tools.docker
    python -m qwen3tts_tools.docker check-gpu
    python -m qwen3tts_tools.docker resolve-image --driver 570.86
    python -m qwen3tts_tools.docker ensure-image nvcr.io/nvidia/tritonserver:25.03-py3

Usage from Python::

    from qwen3tts_tools.docker import NgcMatrix, detect_driver_version
    matrix = NgcMatrix()
    driver = detect_driver_version()
    image = matrix.resolve_image(driver)
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from qwen3tts_tools.ngc_matrix import (
    NGC_PY3_SUFFIX,
    NGC_TRITON_BASE,
    QWEN3_MIN_DRIVER,
    NgcEntry,
    _driver_ge,
    load_ngc_matrix,
    resolve_ngc_entry,
    resolve_ngc_entry_by_tag,
    resolve_ngc_image,
    resolve_ngc_tag,
)


# ---------------------------------------------------------------------------
#  Structured data
# ---------------------------------------------------------------------------

@dataclass
class GpuInfo:
    """Detected GPU information.

    Attributes:
        driver_version: NVIDIA driver version (e.g. ``"570.86.10"``).
        compute_cap: GPU compute capability (e.g. ``"8.9"``).
    """

    driver_version: str = ""
    compute_cap: str = ""


@dataclass
class DockerGpuCheck:
    """Result of Docker + GPU readiness check.

    Attributes:
        docker_available: Docker CLI is installed and daemon is running.
        nvidia_toolkit: NVIDIA Container Toolkit is detected.
        gpu_accessible: GPU can be accessed from within a container.
        errors: List of error messages for failed checks.
        warnings: List of non-fatal warning messages.
    """

    docker_available: bool = False
    nvidia_toolkit: bool = False
    gpu_accessible: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """``True`` if Docker with GPU support is fully operational."""
        return self.docker_available and self.nvidia_toolkit


# ---------------------------------------------------------------------------
#  Subprocess helpers
# ---------------------------------------------------------------------------

def _run(
    cmd: list[str],
    *,
    timeout: int = 30,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with reasonable defaults.

    Args:
        cmd: Command and arguments.
        timeout: Timeout in seconds.
        check: If True, raise on non-zero return code.

    Returns:
        CompletedProcess instance with captured stdout/stderr.
    """
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


# ---------------------------------------------------------------------------
#  Detection functions
# ---------------------------------------------------------------------------

def detect_driver_version() -> str | None:
    """Detect the NVIDIA driver version via ``nvidia-smi``.

    Returns:
        Driver version string (e.g. ``"570.86.10"``), or ``None`` if
        ``nvidia-smi`` is unavailable or returns an unexpected format.
    """
    if not shutil.which("nvidia-smi"):
        return None

    try:
        result = _run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader,nounits"],
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    version = result.stdout.strip().split("\n")[0].strip()
    # Validate format: major.minor or major.minor.patch
    if not _is_valid_driver_version(version):
        return None

    return version


def detect_gpu_compute_cap() -> str | None:
    """Detect the compute capability of the first GPU via ``nvidia-smi``.

    Returns:
        Compute capability string (e.g. ``"8.9"``), or ``None`` if
        ``nvidia-smi`` is unavailable or no GPU is found.
    """
    if not shutil.which("nvidia-smi"):
        return None

    try:
        result = _run(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader,nounits"],
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    cap = result.stdout.strip().split("\n")[0].strip()
    return cap if cap else None


def detect_gpu_info() -> GpuInfo:
    """Detect both driver version and compute capability.

    Returns:
        :class:`GpuInfo` with as much information as could be gathered.
    """
    return GpuInfo(
        driver_version=detect_driver_version() or "",
        compute_cap=detect_gpu_compute_cap() or "",
    )


def detect_gpu_free_memory_mb(gpu_index: int = 0) -> int:
    """Detect free GPU memory in MiB via ``nvidia-smi``.

    Args:
        gpu_index: GPU device index (default 0).

    Returns:
        Free memory in MiB, or 0 if detection fails.
    """
    if not shutil.which("nvidia-smi"):
        return 0

    try:
        result = _run(
            ["nvidia-smi", f"--id={gpu_index}",
             "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return 0

    if result.returncode != 0:
        return 0

    try:
        return int(result.stdout.strip().split("\n")[0].strip())
    except (ValueError, IndexError):
        return 0


def detect_gpu_total_memory_mb(gpu_index: int = 0) -> int:
    """Detect total GPU memory in MiB via ``nvidia-smi``.

    Args:
        gpu_index: GPU device index (default 0).

    Returns:
        Total memory in MiB, or 0 if detection fails.
    """
    if not shutil.which("nvidia-smi"):
        return 0

    try:
        result = _run(
            ["nvidia-smi", f"--id={gpu_index}",
             "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return 0

    if result.returncode != 0:
        return 0

    try:
        return int(result.stdout.strip().split("\n")[0].strip())
    except (ValueError, IndexError):
        return 0


def detect_docker_gpu_args(image: str, gpu_device: str = "auto") -> list[str]:
    """Detect Docker GPU arguments for container passthrough.

    Tries ``--gpus device=N`` first, then falls back to
    ``--runtime=nvidia`` with environment variables.

    Args:
        image: Docker image to use for the smoke test.
        gpu_device: GPU device (auto|all|N|cuda:N).

    Returns:
        List of Docker run arguments for GPU passthrough.

    Raises:
        RuntimeError: If both GPU passthrough methods fail.
    """
    docker_gpu_arg = "all"
    visible_devices = "all"

    if gpu_device not in ("auto", "all", ""):
        norm = gpu_device.removeprefix("cuda:")
        docker_gpu_arg = f"device={norm}"
        visible_devices = norm

    # Try --gpus first
    try:
        proc = _run(
            ["docker", "run", "--rm", "--gpus", docker_gpu_arg,
             image, "/bin/true"],
            timeout=30,
        )
        if proc.returncode == 0:
            return ["--gpus", docker_gpu_arg]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: --runtime=nvidia
    try:
        proc = _run(
            [
                "docker", "run", "--rm",
                "--runtime=nvidia",
                "-e", f"NVIDIA_VISIBLE_DEVICES={visible_devices}",
                "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                image, "/bin/true",
            ],
            timeout=30,
        )
        if proc.returncode == 0:
            return [
                "--runtime=nvidia",
                "-e", f"NVIDIA_VISIBLE_DEVICES={visible_devices}",
                "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise RuntimeError(f"Docker GPU smoke test failed for image: {image}")


def _is_valid_driver_version(version: str) -> bool:
    """Check whether *version* looks like a valid NVIDIA driver version."""
    import re as _re
    return bool(_re.match(r"^\d+(\.\d+){1,2}$", version))


# ---------------------------------------------------------------------------
#  Docker checks
# ---------------------------------------------------------------------------

def check_docker_gpu_ready() -> DockerGpuCheck:
    """Verify Docker + NVIDIA Container Toolkit + GPU access.

    Performs the following checks in order:

    1. ``docker`` CLI is installed.
    2. Docker daemon is running and accessible.
    3. NVIDIA Container Toolkit is present (via ``docker info``,
       ``nvidia-container-cli``, or config file).
    4. (Optional) GPU is accessible inside a container.

    Returns:
        :class:`DockerGpuCheck` with results and any error/warning messages.
    """
    result = DockerGpuCheck()

    # 1. Docker CLI
    if not shutil.which("docker"):
        result.errors.append(
            "Docker not found. Install: https://docs.docker.com/engine/install/"
        )
        return result

    # 2. Docker daemon
    try:
        proc = _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        result.errors.append(
            "Docker daemon not running or insufficient permissions. "
            "Try: sudo systemctl start docker && sudo usermod -aG docker $USER"
        )
        return result

    if proc.returncode != 0:
        result.errors.append(
            "Docker daemon not running or insufficient permissions. "
            "Try: sudo systemctl start docker && sudo usermod -aG docker $USER"
        )
        return result

    result.docker_available = True

    # 3. NVIDIA Container Toolkit
    nvidia_detected = False

    # Check docker info for nvidia runtime
    try:
        info_proc = _run(["docker", "info"], timeout=10)
        if info_proc.returncode == 0 and "nvidia" in info_proc.stdout.lower():
            nvidia_detected = True
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Check nvidia-container-cli
    if not nvidia_detected and shutil.which("nvidia-container-cli"):
        nvidia_detected = True

    # Check config file
    if not nvidia_detected and Path("/etc/nvidia-container-runtime/config.toml").is_file():
        nvidia_detected = True

    if not nvidia_detected:
        result.errors.append(
            "NVIDIA Container Toolkit not detected. "
            "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        )
        return result

    result.nvidia_toolkit = True

    # 4. Try a quick GPU access test (non-fatal)
    try:
        test_proc = _run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.4.0-base-ubuntu22.04", "nvidia-smi", "-L"],
            timeout=60,
        )
        if test_proc.returncode == 0:
            result.gpu_accessible = True
        else:
            result.warnings.append(
                "GPU access test container failed — GPU may not be accessible from Docker"
            )
    except (subprocess.TimeoutExpired, OSError):
        result.warnings.append(
            "GPU access test timed out or failed — GPU accessibility unconfirmed"
        )

    return result


# ---------------------------------------------------------------------------
#  Docker image operations
# ---------------------------------------------------------------------------

def ensure_image(image_uri: str, *, retries: int = 2, retry_delay: int = 10) -> bool:
    """Pull a Docker image if not already present locally.

    Args:
        image_uri: Full image URI (e.g. ``"nvcr.io/nvidia/tritonserver:25.03-py3"``).
        retries: Number of pull retries on failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        ``True`` if the image is available locally (already present or
        successfully pulled), ``False`` otherwise.
    """
    # Check if image already exists locally
    try:
        proc = _run(["docker", "image", "inspect", image_uri], timeout=10)
        if proc.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Pull the image
    for attempt in range(1, retries + 1):
        try:
            proc = _run(["docker", "pull", image_uri], timeout=600)
            if proc.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
        except OSError:
            return False

        if attempt < retries:
            import time
            time.sleep(retry_delay)

    return False


def build_image(
    tag: str,
    dockerfile: str | Path,
    build_args: dict[str, str] | None = None,
    context: str | Path | None = None,
) -> bool:
    """Build a Docker image.

    Args:
        tag: Image tag (e.g. ``"qwen3-tts-triton:25.03"``).
        dockerfile: Path to the Dockerfile.
        build_args: Optional build arguments as key-value pairs.
        context: Build context directory. Defaults to the Dockerfile's
                 parent directory.

    Returns:
        ``True`` if the build succeeded, ``False`` otherwise.
    """
    dockerfile = Path(dockerfile)
    if context is None:
        context = dockerfile.parent
    context = Path(context)

    cmd: list[str] = ["docker", "build"]

    if build_args:
        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])

    cmd.extend(["-t", tag, "-f", str(dockerfile), str(context)])

    try:
        proc = _run(cmd, timeout=1800)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def container_is_running(name: str) -> bool:
    """Check whether a Docker container with the given name is running.

    Args:
        name: Container name or partial name to match.

    Returns:
        ``True`` if a running container matching *name* is found.
    """
    try:
        proc = _run(
            ["docker", "ps", "--filter", f"name={name}",
             "--format", "{{.Names}}"],
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False

    if proc.returncode != 0:
        return False

    return bool(proc.stdout.strip())


def list_images(filter_pattern: str = "") -> list[str]:
    """List Docker images, optionally filtered by a pattern.

    Args:
        filter_pattern: Substring to filter image names by (case-insensitive).
                        Empty string returns all images.

    Returns:
        List of image strings in ``repository:tag`` format.
    """
    try:
        proc = _run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []

    if proc.returncode != 0:
        return []

    images = [line for line in proc.stdout.strip().split("\n") if line]

    if filter_pattern:
        pattern_lower = filter_pattern.lower()
        images = [img for img in images if pattern_lower in img.lower()]

    return images


# ---------------------------------------------------------------------------
#  NgcMatrix — high-level wrapper
# ---------------------------------------------------------------------------

class NgcMatrix:
    """High-level wrapper around the NGC compatibility matrix and Docker operations.

    Loads the matrix once on construction and provides convenience methods
    that combine matrix lookups with Docker operations.

    Args:
        matrix_path: Path to ``ngc_matrix.conf``. Auto-detected if None.

    Example::

        matrix = NgcMatrix()
        driver = detect_driver_version()
        tag = matrix.resolve_tag(driver)
        image = matrix.resolve_image_uri(tag)
        matrix.ensure_base_image(driver)
    """

    def __init__(self, matrix_path: Path | str | None = None) -> None:
        self._entries: list[NgcEntry] = load_ngc_matrix(matrix_path)

    @property
    def entries(self) -> list[NgcEntry]:
        """All matrix entries, newest-first."""
        return list(self._entries)

    def resolve_tag(self, driver_version: str) -> str | None:
        """Resolve the best NGC tag for a driver version.

        Args:
            driver_version: Installed NVIDIA driver version.

        Returns:
            Compatible NGC tag, or ``None``.
        """
        return resolve_ngc_tag(driver_version, self._entries)

    def resolve_entry(self, driver_version: str) -> NgcEntry | None:
        """Resolve the full matrix entry for a driver version.

        Args:
            driver_version: Installed NVIDIA driver version.

        Returns:
            Best compatible :class:`NgcEntry`, or ``None``.
        """
        return resolve_ngc_entry(driver_version, self._entries)

    def resolve_entry_by_tag(self, ngc_tag: str) -> NgcEntry | None:
        """Look up a matrix entry by NGC tag.

        Args:
            ngc_tag: NGC container tag (e.g. ``"25.03"``).

        Returns:
            Matching :class:`NgcEntry`, or ``None``.
        """
        return resolve_ngc_entry_by_tag(ngc_tag, self._entries)

    def resolve_image_uri(self, ngc_tag: str) -> str | None:
        """Return the full nvcr.io image URI for a tag.

        Args:
            ngc_tag: NGC container tag.

        Returns:
            Image URI (e.g. ``"nvcr.io/nvidia/tritonserver:25.03-py3"``),
            or ``None`` if the tag is unknown.
        """
        return resolve_ngc_image(ngc_tag, self._entries)

    def resolve_image_for_driver(self, driver_version: str) -> str | None:
        """Resolve the full NGC image URI for a driver version.

        Combines :meth:`resolve_tag` and :meth:`resolve_image_uri`.

        Args:
            driver_version: Installed NVIDIA driver version.

        Returns:
            Full image URI, or ``None`` if no compatible entry.
        """
        tag = self.resolve_tag(driver_version)
        if tag is None:
            return None
        return self.resolve_image_uri(tag)

    def ensure_base_image(self, driver_version: str) -> str | None:
        """Pull the best compatible NGC base image if not present.

        Args:
            driver_version: Installed NVIDIA driver version.

        Returns:
            The image URI if available (pulled or already present),
            or ``None`` on failure.
        """
        image = self.resolve_image_for_driver(driver_version)
        if image is None:
            return None
        if ensure_image(image):
            return image
        return None

    def format_table(self, driver_version: str | None = None) -> str:
        """Format the matrix as a human-readable table.

        Args:
            driver_version: If provided, mark compatibility status.

        Returns:
            Formatted table string.
        """
        from qwen3tts_tools.ngc_matrix import format_matrix_table
        return format_matrix_table(self._entries, driver_version)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Docker / NGC container utilities for Qwen3-TTS",
    )
    sub = parser.add_subparsers(dest="command")

    # check-gpu
    sub.add_parser("check-gpu", help="Check Docker + NVIDIA GPU readiness")

    # detect-driver
    sub.add_parser("detect-driver", help="Print detected NVIDIA driver version")

    # detect-compute-cap
    sub.add_parser("detect-compute-cap", help="Print GPU compute capability")

    # resolve-image
    resolve_img = sub.add_parser("resolve-image", help="Resolve NGC image for driver")
    resolve_img.add_argument("--driver", type=str, default=None,
                             help="NVIDIA driver version (auto-detect if omitted)")

    # resolve-tag
    resolve_tag_p = sub.add_parser("resolve-tag", help="Resolve NGC tag for driver")
    resolve_tag_p.add_argument("--driver", type=str, default=None,
                               help="NVIDIA driver version (auto-detect if omitted)")

    # ensure-image
    ensure_img = sub.add_parser("ensure-image", help="Pull Docker image if not present")
    ensure_img.add_argument("image_uri", help="Docker image URI")

    # list-images
    list_img = sub.add_parser("list-images", help="List Docker images")
    list_img.add_argument("--filter", type=str, default="",
                          help="Filter pattern (substring match)")

    # list-matrix
    sub.add_parser("list-matrix", help="List NGC compatibility matrix")

    # running
    running_p = sub.add_parser("running", help="Check if a container is running")
    running_p.add_argument("name", help="Container name")

    args = parser.parse_args()

    if args.command == "check-gpu":
        result = check_docker_gpu_ready()
        if result.ready:
            print("Docker + NVIDIA GPU runtime: OK")
            if result.gpu_accessible:
                print("GPU access from container: OK")
            for w in result.warnings:
                print(f"WARNING: {w}")
        else:
            for err in result.errors:
                print(f"ERROR: {err}")
            raise SystemExit(1)

    elif args.command == "detect-driver":
        driver = detect_driver_version()
        if driver:
            print(driver)
        else:
            print("Cannot detect NVIDIA driver version")
            raise SystemExit(1)

    elif args.command == "detect-compute-cap":
        cap = detect_gpu_compute_cap()
        if cap:
            print(cap)
        else:
            print("Cannot detect GPU compute capability")
            raise SystemExit(1)

    elif args.command == "resolve-image":
        driver = args.driver or detect_driver_version()
        if not driver:
            print("Cannot determine driver version")
            raise SystemExit(1)
        matrix = NgcMatrix()
        image = matrix.resolve_image_for_driver(driver)
        if image:
            print(image)
        else:
            print(f"No compatible NGC container for driver {driver}")
            print(f"Minimum driver for Qwen3-TTS: >= {QWEN3_MIN_DRIVER}")
            raise SystemExit(1)

    elif args.command == "resolve-tag":
        driver = args.driver or detect_driver_version()
        if not driver:
            print("Cannot determine driver version")
            raise SystemExit(1)
        matrix = NgcMatrix()
        tag = matrix.resolve_tag(driver)
        if tag:
            print(tag)
        else:
            print(f"No compatible NGC container for driver {driver}")
            print(f"Minimum driver for Qwen3-TTS: >= {QWEN3_MIN_DRIVER}")
            raise SystemExit(1)

    elif args.command == "ensure-image":
        ok = ensure_image(args.image_uri)
        if ok:
            print(f"Image available: {args.image_uri}")
        else:
            print(f"Failed to ensure image: {args.image_uri}")
            raise SystemExit(1)

    elif args.command == "list-images":
        images = list_images(args.filter)
        for img in images:
            print(img)
        if not images:
            print("No matching images found")

    elif args.command == "list-matrix":
        driver = detect_driver_version()
        matrix = NgcMatrix()
        print(matrix.format_table(driver))

    elif args.command == "running":
        if container_is_running(args.name):
            print(f"Container '{args.name}' is running")
        else:
            print(f"Container '{args.name}' is NOT running")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
