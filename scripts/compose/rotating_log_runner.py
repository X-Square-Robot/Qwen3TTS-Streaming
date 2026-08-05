#!/usr/bin/env python3
"""Run a service while mirroring and rotating its combined output.

This is deliberately a small, standard-library-only container entrypoint.  It
keeps logs available inside the container without making the service depend on
the Docker daemon's logging configuration.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import BinaryIO, Sequence


DEFAULT_LOG_DIR = "/var/log/qwen3tts"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 10
READ_SIZE = 64 * 1024
MAX_READS_PER_CYCLE = 16
MAX_REAPS_PER_CYCLE = 64
# A descendant may accidentally inherit the output pipe after the service has
# exited.  Do not let that orphan keep PID 1 alive forever.
POST_EXIT_DRAIN_SECONDS = 0.5
PR_SET_CHILD_SUBREAPER = 36


def _warn(message: str) -> None:
    """Best-effort warning that is safe even when the output fds are broken."""

    data = f"[rotating-log-runner] WARNING: {message}\n".encode(
        "utf-8", errors="backslashreplace"
    )
    try:
        os.set_blocking(2, False)
        os.write(2, data)
    except OSError:
        pass


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
    except ValueError:
        _warn(f"invalid {name}={raw!r}; using {default}")
        return default
    return value


def _nonnegative_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value < 0:
            raise ValueError
    except ValueError:
        _warn(f"invalid {name}={raw!r}; using {default}")
        return default
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    _warn(f"invalid {name}={raw!r}; using {int(default)}")
    return default


class _StdoutMirror:
    """Best-effort nonblocking stdout mirror.

    Container stdout is an observability aid, not part of the service's data
    path.  Backpressure from a logging driver or an attached client must never
    prevent the durable in-container log from being written.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = False
        self._nonblocking = False
        if enabled:
            self.force_enable()

    def force_enable(self) -> None:
        if not self._nonblocking:
            try:
                os.set_blocking(1, False)
            except OSError as exc:
                _warn(f"cannot make stdout nonblocking; disabling mirror: {exc}")
                self.enabled = False
                return
            self._nonblocking = True
        self.enabled = True

    def write(self, data: bytes) -> None:
        if not self.enabled:
            return

        view = memoryview(data)
        while view:
            try:
                written = os.write(1, view)
            except BlockingIOError:
                # Drop only the stdout copy.  The complete chunk has already
                # been written to the rotating file by _OutputSink.
                return
            except InterruptedError:
                continue
            except OSError as exc:
                if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                    return
                if exc.errno not in {errno.EPIPE, errno.EBADF}:
                    _warn(f"cannot mirror service output to stdout: {exc}")
                # A closed Docker/attach pipe must never take down the service.
                self.enabled = False
                return
            if written <= 0:
                self.enabled = False
                return
            view = view[written:]


class _RotatingFile:
    """A byte-counted rotating file with ``name.log.N`` backups."""

    def __init__(self, path: Path, max_bytes: int, backup_count: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._prune_excess_backups()
        self._file: BinaryIO = path.open("ab", buffering=0)
        self._size = os.fstat(self._file.fileno()).st_size

    def close(self) -> None:
        self._file.close()

    def _backup_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index}")

    def _prune_excess_backups(self) -> None:
        """Remove numbered backups left behind by a larger old retention limit."""

        prefix = f"{self.path.name}."
        for candidate in self.path.parent.iterdir():
            if not candidate.name.startswith(prefix):
                continue
            suffix = candidate.name[len(prefix) :]
            if suffix.isdigit() and int(suffix) > self.backup_count:
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass

    def _rotate(self) -> None:
        self._file.close()
        if self.backup_count == 0:
            self._file = self.path.open("wb", buffering=0)
            self._size = 0
            return

        oldest = self._backup_path(self.backup_count)
        try:
            oldest.unlink()
        except FileNotFoundError:
            pass

        for index in range(self.backup_count - 1, 0, -1):
            source = self._backup_path(index)
            if source.exists():
                os.replace(source, self._backup_path(index + 1))
        if self.path.exists():
            os.replace(self.path, self._backup_path(1))

        self._file = self.path.open("wb", buffering=0)
        self._size = 0

    def write(self, data: bytes) -> None:
        """Write all bytes, splitting oversized chunks at rotation boundaries."""

        view = memoryview(data)
        while view:
            if self._size >= self.max_bytes:
                self._rotate()

            limit = min(len(view), self.max_bytes - self._size)
            pending = view[:limit]
            while pending:
                written = self._file.write(pending)
                if written is None:
                    written = len(pending)
                if written <= 0:
                    raise OSError("log file write returned no progress")
                self._size += written
                pending = pending[written:]
            view = view[limit:]


class _OutputSink:
    def __init__(
        self,
        service: str,
        log_dir: Path,
        max_bytes: int,
        backup_count: int,
        mirror_stdout: bool,
    ) -> None:
        self._mirror = _StdoutMirror(mirror_stdout)
        self._log: _RotatingFile | None = None
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log = _RotatingFile(
                log_dir / f"{service}.log", max_bytes, backup_count
            )
        except OSError as exc:
            _warn(
                f"cannot open log directory {str(log_dir)!r}: {exc}; "
                "continuing with stdout only"
            )
            # Fail open even if QWEN_LOG_STDOUT=0: output must remain observable.
            self._mirror.force_enable()

    def write(self, data: bytes) -> None:
        # Persist first: a slow or abandoned container stdout consumer must not
        # delay the in-container log or apply backpressure to the service.
        if self._log is not None:
            try:
                self._log.write(data)
            except OSError as exc:
                try:
                    self._log.close()
                except OSError:
                    pass
                self._log = None
                self._mirror.force_enable()
                self._mirror.write(data)
                _warn(f"cannot write rotating log: {exc}; continuing with stdout only")
                return

        self._mirror.write(data)

    def close(self) -> None:
        if self._log is not None:
            try:
                self._log.close()
            except OSError as exc:
                _warn(f"cannot close rotating log: {exc}")


def _service_name(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise argparse.ArgumentTypeError(
            "service name must be a single file-name component"
        )
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="run a command with combined stdout/stderr log rotation"
    )
    parser.add_argument("--service", required=True, type=_service_name)
    parser.add_argument("command", nargs=argparse.REMAINDER, metavar="-- COMMAND...")
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def _forward_signal(process: subprocess.Popen[bytes], signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass
    except OSError as exc:
        _warn(f"cannot forward signal {signum} to child process group: {exc}")


def _enable_child_subreaper() -> bool:
    """Ask Linux to reparent orphaned descendants to this runner."""

    if not sys.platform.startswith("linux"):
        return False
    waitid_api = ("waitid", "P_ALL", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    if any(not hasattr(os, name) for name in waitid_api):
        return False

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        prctl.restype = ctypes.c_int
        result = prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (AttributeError, OSError) as exc:
        _warn(f"cannot enable child subreaper: {exc}")
        return False

    if result != 0:
        error_number = ctypes.get_errno()
        _warn(
            "cannot enable child subreaper: "
            f"{os.strerror(error_number)} (errno {error_number})"
        )
        return False
    return True


def _reap_exited_descendants(process: subprocess.Popen[bytes]) -> None:
    """Reap adopted descendants without consuming the direct child's status."""

    peek_flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    reap_flags = os.WEXITED | os.WNOHANG
    for _ in range(MAX_REAPS_PER_CYCLE):
        try:
            child = os.waitid(os.P_ALL, 0, peek_flags)
        except ChildProcessError:
            return
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            _warn(f"cannot inspect exited descendants: {exc}")
            return

        if child is None or child.si_pid == 0:
            return
        if child.si_pid == process.pid:
            # Popen owns this status.  poll() caches it so the final wait()
            # preserves both ordinary exit codes and signal termination.
            process.poll()
            if process.returncode is None:
                return
            continue

        try:
            os.waitid(os.P_PID, child.si_pid, reap_flags)
        except ChildProcessError:
            continue
        except OSError as exc:
            if exc.errno != errno.ECHILD:
                _warn(f"cannot reap exited descendant {child.si_pid}: {exc}")


def _drain_output(
    process: subprocess.Popen[bytes], sink: _OutputSink, reap_descendants: bool
) -> None:
    pipe = process.stdout
    if pipe is None:  # Defensive: Popen below always requests a pipe.
        return

    fd = pipe.fileno()
    os.set_blocking(fd, False)
    selector = selectors.DefaultSelector()
    selector.register(fd, selectors.EVENT_READ)
    exited_at: float | None = None
    eof = False
    try:
        while not eof:
            events = selector.select(timeout=0.1)
            if events:
                # Yield back to lifecycle checks even when a noisy orphan keeps
                # the pipe permanently readable.
                for _ in range(MAX_READS_PER_CYCLE):
                    try:
                        chunk = os.read(fd, READ_SIZE)
                    except BlockingIOError:
                        break
                    except InterruptedError:
                        continue
                    except OSError as exc:
                        _warn(f"cannot read service output: {exc}")
                        eof = True
                        break
                    if not chunk:
                        eof = True
                        break
                    sink.write(chunk)

            if reap_descendants:
                _reap_exited_descendants(process)
            if process.poll() is not None:
                if exited_at is None:
                    exited_at = time.monotonic()
                elif time.monotonic() - exited_at >= POST_EXIT_DRAIN_SECONDS:
                    # The direct child is gone but an orphan still owns the
                    # write end.  Everything currently buffered was drained.
                    break
    finally:
        selector.close()
        pipe.close()


def run(service: str, command: Sequence[str]) -> int:
    max_bytes = _positive_env("QWEN_LOG_MAX_BYTES", DEFAULT_MAX_BYTES)
    backup_count = _nonnegative_env("QWEN_LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT)
    mirror_stdout = _bool_env("QWEN_LOG_STDOUT", True)
    log_dir = Path(os.environ.get("QWEN_LOG_DIR", DEFAULT_LOG_DIR))
    sink = _OutputSink(service, log_dir, max_bytes, backup_count, mirror_stdout)
    reap_descendants = _enable_child_subreaper()

    process_holder: list[subprocess.Popen[bytes] | None] = [None]
    pending_signals: list[int] = []
    previous_handlers: dict[int, signal.Handlers] = {}

    def handle_signal(signum: int, _frame: object) -> None:
        process = process_holder[0]
        if process is None:
            pending_signals.append(signum)
        else:
            _forward_signal(process, signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    try:
        try:
            process = subprocess.Popen(
                list(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                start_new_session=True,
            )
        except OSError as exc:
            _warn(f"cannot start {command[0]!r}: {exc}")
            return 127 if exc.errno == errno.ENOENT else 126

        process_holder[0] = process
        for signum in pending_signals:
            _forward_signal(process, signum)
        pending_signals.clear()

        _drain_output(process, sink, reap_descendants)
        returncode = process.wait()
        if reap_descendants:
            _reap_exited_descendants(process)
        # Shell-compatible representation for a child terminated by a signal.
        return returncode if returncode >= 0 else 128 - returncode
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        sink.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args.service, args.command)


if __name__ == "__main__":
    raise SystemExit(main())
