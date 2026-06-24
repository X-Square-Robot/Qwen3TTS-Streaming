"""Docker Compose orchestration — Python wrapper.

Replaces scripts/bash/compose.sh core logic.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class ComposeManager:
    """Manage Docker Compose lifecycle for the TTS service."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or _repo_root()
        self.compose_dir = self.repo_root / "infra" / "docker"

    def _compose_base(self) -> list[str]:
        """Build the base docker compose command."""
        return [
            "docker", "compose",
            "-f", str(self.compose_dir / "compose.yaml"),
            "-f", str(self.compose_dir / "compose.dev.yaml"),
        ]

    def prepare(
        self,
        *,
        gateway: str = "engine",
        variant: str = "",
        engine_mode: str = "trt",
        repo_dir: str = "",
        model_version: int = 1,
        dry_run: bool = False,
    ) -> int:
        """Prepare model repository before starting services."""
        from qwen3tts_tools.triton import TritonManager

        if not repo_dir:
            repo_dir = str(self.repo_root / "workspace" / "model_repository")

        mgr = TritonManager(repo_root=self.repo_root)
        if variant and not dry_run:
            logger.info("Assembling model repository for variant: %s, engine_mode: %s", variant, engine_mode)
            result = mgr.assemble_model_repo(
                exported_dir=self.repo_root / "workspace" / "exported",
                variant=variant,
                model_repo_dir=Path(repo_dir),
                engine_mode=engine_mode,
                model_version=model_version,
            )
            if not result:
                logger.error("Model repository assembly failed.")
                return 1
        elif dry_run:
            logger.info("[DRY RUN] Would assemble model repository.")
        return 0

    def up(
        self,
        *,
        gateway: str = "engine",
        variant: str = "",
        engine_mode: str = "trt",
        image: str = "",
        device: str = "auto",
        max_batch: int = 0,
        max_seq_len: int = 0,
        model_version: int = 1,
        port: int = 50051,
        ws_port: int = 50052,
        container: str = "",
        no_health_check: bool = False,
        dry_run: bool = False,
    ) -> int:
        """Start Docker Compose services."""
        from qwen3tts_tools.common import REPO_ROOT
        from qwen3tts_tools.docker import detect_driver_version
        from qwen3tts_tools.ngc_matrix import resolve_ngc_image

        # Resolve device
        if device == "auto":
            from qwen3tts_tools.docker import detect_gpu_compute_cap
            device = "0"

        # Prepare model repository
        repo_dir = str(REPO_ROOT / "workspace" / "model_repository")
        result = self.prepare(
            gateway=gateway,
            variant=variant,
            engine_mode=engine_mode,
            repo_dir=repo_dir,
            model_version=model_version,
            dry_run=dry_run,
        )
        if result != 0:
            return result

        # Resolve image
        if not image:
            try:
                driver = detect_driver_version()
                image = resolve_ngc_image(driver) if driver else ""
            except Exception:
                image = ""
            if not image:
                image = os.environ.get("NGC_IMAGE", "nvcr.io/nvidia/tritonserver:25.05-py3")

        # Build docker compose up command
        cmd = self._compose_base() + [
            "up", "--detach", "--build",
        ]

        # Add profile
        if gateway == "engine":
            cmd.extend(["--profile", "engine"])
        elif gateway == "triton":
            cmd.extend(["--profile", "triton"])

        # Add environment variables
        env = os.environ.copy()
        env["MODEL_REPO_DIR"] = repo_dir
        env["ENGINE_MODE"] = engine_mode
        if variant:
            env["VARIANT"] = variant
        if device and device != "auto":
            env["RUNTIME_GPU_DEVICE"] = device
        if max_batch:
            env["RUNTIME_MAX_BATCH_SIZE"] = str(max_batch)
        if max_seq_len:
            env["RUNTIME_MAX_SEQ_LEN"] = str(max_seq_len)
        if model_version != 1:
            env["MODEL_VERSION"] = str(model_version)
        if image:
            env["ENGINE_IMAGE"] = image
        if port != 50051:
            env["ENGINE_GRPC_PORT"] = str(port)
        if ws_port != 50052:
            env["ENGINE_WEBSOCKET_PORT"] = str(ws_port)
        if container:
            env["CONTAINER_NAME"] = container
        if no_health_check:
            env["SKIP_HEALTH_CHECK"] = "1"

        if dry_run:
            logger.info("[DRY RUN] Would run: %s", " ".join(cmd))
            return 0

        logger.info("Starting %s service...", gateway)
        result = subprocess.call(cmd, env=env, cwd=str(self.repo_root))
        if result != 0:
            return result

        # Health check
        if not no_health_check:
            self._wait_healthy(gateway, port if gateway == "engine" else 8001)
        return 0

    def down(self, *, gateway: str = "all") -> int:
        """Stop and remove Docker Compose services."""
        cmd = self._compose_base() + ["down", "--remove-orphans"]

        if gateway == "engine":
            cmd.extend(["--profile", "engine"])
        elif gateway == "triton":
            cmd.extend(["--profile", "triton"])

        logger.info("Stopping %s service(s)...", gateway)
        return subprocess.call(cmd, cwd=str(self.repo_root))

    def logs(self, *, gateway: str = "engine", follow: bool = False) -> int:
        """Show service logs."""
        cmd = self._compose_base() + ["logs"]
        if follow:
            cmd.append("--follow")
        if gateway == "engine":
            cmd.append("engine")
        elif gateway == "triton":
            cmd.append("triton")
        return subprocess.call(cmd, cwd=str(self.repo_root))

    def ps(self) -> int:
        """Show service status."""
        cmd = self._compose_base() + ["ps"]
        return subprocess.call(cmd, cwd=str(self.repo_root))

    def _wait_healthy(self, gateway: str, port: int, timeout: int = 120) -> bool:
        """Wait for service health check."""
        import time
        import urllib.request
        import urllib.error

        if gateway == "engine":
            url = f"http://localhost:{port}/health"
        else:
            url = f"http://localhost:{port}/v2/health/ready"

        logger.info("Waiting for %s service on port %d (timeout %ds)...", gateway, port, timeout)
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                urllib.request.urlopen(url, timeout=5)
                logger.info("Service is healthy!")
                return True
            except (urllib.error.URLError, OSError):
                time.sleep(2)

        logger.warning("Health check timed out after %ds", timeout)
        return False
