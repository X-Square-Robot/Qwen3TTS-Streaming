"""Engine process manager for Qwen3-TTS-Triton.

This module provides a Python replacement for ``scripts/bash/lib/engine.sh``,
managing the engine.server lifecycle (start / stop / status / health-check)
with PID-file bookkeeping and structured logging.

Deprecation notice
------------------
The Bash library ``scripts/bash/lib/engine.sh`` is superseded by this module.
The shell version remains available for backward compatibility but should be
considered deprecated.  New code should import from
:mod:`qwen3tts_tools.engine` directly.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # scripts/python/qwen3tts_tools -> root


# ---------------------------------------------------------------------------
# Structured data
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Configuration for an engine server invocation."""

    config_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "engine.yaml")
    variant: str = "custom-1.7b"
    port: int = 8000
    ws_port: int = 8001
    device: str = "cuda:0"
    max_batch: int = 4
    max_sessions: int = 16
    max_seq_len: int = 2048
    foreground: bool = False


@dataclass
class EngineStatus:
    """Snapshot of the engine process status."""

    state: str  # "none" | "running" | "healthy"
    pid: Optional[int] = None
    uptime_s: Optional[float] = None


# ---------------------------------------------------------------------------
# EngineManager
# ---------------------------------------------------------------------------

class EngineManager:
    """Manage the ``engine.server`` process lifecycle.

    Responsibilities:
    * Locate a suitable Python interpreter (conda ``qwen3-tts``, mamba, or
      system Python).
    * Start ``python -m engine.server`` as a background process with a PID
      file and redirected log output.
    * Stop the engine via SIGTERM (with SIGKILL fallback).
    * Report process status and perform HTTP health checks.

    Example::

        mgr = EngineManager()
        mgr.start(config_path=Path("engine.yaml"), variant="custom-1.7b", port=8000)
        print(mgr.status())
        mgr.stop()
    """

    def __init__(self, workspace: Optional[Path] = None) -> None:
        self._workspace = workspace or _PROJECT_ROOT / "workspace"
        self._process: Optional[subprocess.Popen[bytes]] = None

    # -- Path helpers -------------------------------------------------------

    def pid_file(self) -> Path:
        """Return the PID file path (``workspace/engine.pid``)."""
        return self._workspace / "engine.pid"

    def log_file(self) -> Path:
        """Return the log file path (``workspace/engine.log``)."""
        return self._workspace / "engine.log"

    # -- Python interpreter discovery ---------------------------------------

    @staticmethod
    def find_python_bin() -> str:
        """Find a suitable Python interpreter.

        Search order:
        1. Conda environment ``qwen3-tts`` (``conda run``).
        2. Mamba environment ``qwen3-tts`` (``mamba run``).
        3. System ``python3`` / ``python``.

        Returns
        -------
        str
            The Python executable name or ``conda/mamba run`` prefix command
            suitable for :func:`subprocess.Popen`.
        """
        # 1. Try conda
        try:
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and "qwen3-tts" in result.stdout:
                logger.info("Found conda environment 'qwen3-tts'")
                return "conda run -n qwen3-tts python"
        except FileNotFoundError:
            logger.debug("conda not found on PATH")
        except subprocess.TimeoutExpired:
            logger.warning("conda env list timed out")

        # 2. Try mamba
        try:
            result = subprocess.run(
                ["mamba", "env", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and "qwen3-tts" in result.stdout:
                logger.info("Found mamba environment 'qwen3-tts'")
                return "mamba run -n qwen3-tts python"
        except FileNotFoundError:
            logger.debug("mamba not found on PATH")
        except subprocess.TimeoutExpired:
            logger.warning("mamba env list timed out")

        # 3. Fallback to system python
        for candidate in ("python3", "python"):
            try:
                subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    timeout=10,
                )
                logger.info("Using system Python: %s", candidate)
                return candidate
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                continue

        logger.error("No Python interpreter found")
        return "python"

    # -- Process lifecycle --------------------------------------------------

    def start(
        self,
        config_path: Optional[Path] = None,
        variant: str = "custom-1.7b",
        port: int = 8000,
        ws_port: int = 8001,
        device: str = "cuda:0",
        max_batch: int = 4,
        max_sessions: int = 16,
        max_seq_len: int = 2048,
        foreground: bool = False,
    ) -> EngineStatus:
        """Start ``engine.server`` as a background process.

        Parameters
        ----------
        config_path : Path or None
            Path to ``engine.yaml``.  Defaults to ``<project_root>/engine.yaml``.
        variant : str
            Model variant identifier (e.g. ``"custom-1.7b"``).
        port : int
            HTTP listening port.
        ws_port : int
            WebSocket listening port.
        device : str
            CUDA device string (e.g. ``"cuda:0"``).
        max_batch : int
            Maximum batch size.
        max_sessions : int
            Maximum concurrent sessions.
        max_seq_len : int
            Maximum sequence length.
        foreground : bool
            If ``True``, run in the foreground (block until exit).

        Returns
        -------
        EngineStatus
            The status snapshot after the start attempt.
        """
        if self.status().state != "none":
            logger.warning("Engine already running (PID %s)", self.status().pid)
            return self.status()

        cfg = config_path or _PROJECT_ROOT / "engine.yaml"
        python_cmd = self.find_python_bin()

        # Build the command list.  When python_cmd is a multi-word prefix
        # (e.g. "conda run -n qwen3-tts python"), split it; otherwise it is
        # just the interpreter name.
        cmd_parts = python_cmd.split() if " " in python_cmd else [python_cmd]
        cmd_parts += [
            "-m", "engine.server",
            "--config", str(cfg),
            "--variant", variant,
            "--port", str(port),
            "--ws-port", str(ws_port),
            "--device", device,
            "--max-batch", str(max_batch),
            "--max-sessions", str(max_sessions),
            "--max-seq-len", str(max_seq_len),
        ]

        # Ensure workspace directory exists
        self._workspace.mkdir(parents=True, exist_ok=True)

        log_path = self.log_file()
        pid_path = self.pid_file()

        if foreground:
            logger.info("Starting engine.server in foreground")
            proc = subprocess.Popen(cmd_parts, cwd=str(_PROJECT_ROOT))
            proc.wait()
            return EngineStatus(state="none")
        else:
            logger.info("Starting engine.server in background (log=%s)", log_path)
            log_fh = open(log_path, "a")
            proc = subprocess.Popen(
                cmd_parts,
                cwd=str(_PROJECT_ROOT),
                stdout=log_fh,
                stderr=log_fh,
            )
            self._process = proc

            # Write PID file
            pid_path.write_text(str(proc.pid))
            logger.info("Engine started with PID %d", proc.pid)

            return EngineStatus(state="running", pid=proc.pid)

    def stop(self, timeout: float = 10.0) -> EngineStatus:
        """Stop the engine process.

        Sends ``SIGTERM`` first.  If the process does not exit within
        *timeout* seconds, sends ``SIGKILL``.

        Parameters
        ----------
        timeout : float
            Seconds to wait after SIGTERM before escalating to SIGKILL.

        Returns
        -------
        EngineStatus
            The status snapshot after the stop attempt.
        """
        current = self.status()

        if current.state == "none":
            logger.info("Engine is not running — nothing to stop")
            return current

        pid = current.pid
        if pid is None:
            logger.error("Cannot stop engine: PID unknown")
            return EngineStatus(state="none")

        logger.info("Stopping engine (PID %d) with SIGTERM", pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            logger.warning("Process %d already gone", pid)
            self._cleanup_files()
            return EngineStatus(state="none")

        # Wait for graceful shutdown
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)  # probe — raises if gone
            except ProcessLookupError:
                logger.info("Engine process %d has exited", pid)
                self._cleanup_files()
                return EngineStatus(state="none")
            time.sleep(0.25)

        # Escalate to SIGKILL
        logger.warning("Engine did not exit within %.1fs — sending SIGKILL", timeout)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        self._cleanup_files()
        return EngineStatus(state="none")

    # -- Status / health ----------------------------------------------------

    def status(self) -> EngineStatus:
        """Check whether the engine process is running.

        Returns
        -------
        EngineStatus
            * ``state="none"``     — no PID file or process not alive.
            * ``state="running"``  — process is alive but health unknown.
            * ``state="healthy"``  — process is alive **and** health check
              passes.
        """
        pid_path = self.pid_file()
        if not pid_path.exists():
            return EngineStatus(state="none")

        try:
            pid = int(pid_path.read_text().strip())
        except (ValueError, OSError):
            logger.warning("Corrupt PID file %s — removing", pid_path)
            pid_path.unlink(missing_ok=True)
            return EngineStatus(state="none")

        # Check if the process is alive
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            logger.info("Stale PID file (PID %d no longer exists)", pid)
            self._cleanup_files()
            return EngineStatus(state="none")
        except PermissionError:
            logger.warning("No permission to signal PID %d", pid)
            return EngineStatus(state="running", pid=pid)

        # Process is alive — try a lightweight health probe
        # (We do not know the port here, so we just report "running".
        #  Use health_check() for a full HTTP probe.)
        return EngineStatus(state="running", pid=pid)

    @staticmethod
    def health_check(port: int = 8000, timeout: float = 5.0) -> bool:
        """Perform an HTTP GET health check against the engine.

        Parameters
        ----------
        port : int
            The HTTP port the engine is listening on.
        timeout : float
            Request timeout in seconds.

        Returns
        -------
        bool
            ``True`` if the server responded with a 2xx status.
        """
        import urllib.request
        import urllib.error

        url = f"http://localhost:{port}/health"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                healthy = 200 <= resp.status < 300
                if healthy:
                    logger.debug("Health check passed (status=%d)", resp.status)
                else:
                    logger.warning("Health check returned status %d", resp.status)
                return healthy
        except urllib.error.URLError as exc:
            logger.debug("Health check failed: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.debug("Health check error: %s", exc)
            return False

    # -- Internal helpers ---------------------------------------------------

    def _cleanup_files(self) -> None:
        """Remove PID file and clear internal process reference."""
        self.pid_file().unlink(missing_ok=True)
        self._process = None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> None:
    """Minimal CLI for ad-hoc usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage the Qwen3-TTS engine server process.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = sub.add_parser("start", help="Start the engine server")
    p_start.add_argument("--config", type=Path, default=None, help="Path to engine.yaml")
    p_start.add_argument("--variant", default="custom-1.7b", help="Model variant")
    p_start.add_argument("--port", type=int, default=8000, help="HTTP port")
    p_start.add_argument("--ws-port", type=int, default=8001, help="WebSocket port")
    p_start.add_argument("--device", default="cuda:0", help="CUDA device")
    p_start.add_argument("--max-batch", type=int, default=4, help="Max batch size")
    p_start.add_argument("--max-sessions", type=int, default=16, help="Max concurrent sessions")
    p_start.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    p_start.add_argument("--foreground", action="store_true", help="Run in foreground")

    # stop
    sub.add_parser("stop", help="Stop the engine server")

    # status
    p_status = sub.add_parser("status", help="Show engine status")
    p_status.add_argument("--port", type=int, default=8000, help="Port for health check")

    # health
    p_health = sub.add_parser("health", help="HTTP health check")
    p_health.add_argument("--port", type=int, default=8000, help="HTTP port")
    p_health.add_argument("--timeout", type=float, default=5.0, help="Timeout in seconds")

    # find-python
    sub.add_parser("find-python", help="Locate a suitable Python interpreter")

    args = parser.parse_args()
    mgr = EngineManager()

    if args.command == "start":
        result = mgr.start(
            config_path=args.config,
            variant=args.variant,
            port=args.port,
            ws_port=args.ws_port,
            device=args.device,
            max_batch=args.max_batch,
            max_sessions=args.max_sessions,
            max_seq_len=args.max_seq_len,
            foreground=args.foreground,
        )
        print(f"Engine status: {result.state} (pid={result.pid})")

    elif args.command == "stop":
        result = mgr.stop()
        print(f"Engine status: {result.state}")

    elif args.command == "status":
        result = mgr.status()
        # Augment with health check if running
        if result.state == "running":
            if EngineManager.health_check(port=args.port):
                result = EngineStatus(state="healthy", pid=result.pid)
        print(f"Engine status: {result.state} (pid={result.pid})")

    elif args.command == "health":
        ok = EngineManager.health_check(port=args.port, timeout=args.timeout)
        print(f"Health check: {'OK' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    elif args.command == "find-python":
        python_bin = EngineManager.find_python_bin()
        print(f"Python interpreter: {python_bin}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _parse_args()
