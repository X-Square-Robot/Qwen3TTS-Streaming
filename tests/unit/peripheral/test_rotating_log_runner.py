from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "compose" / "rotating_log_runner.py"


def _runner_env(log_dir: Path, **overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "QWEN_LOG_DIR": str(log_dir),
            "QWEN_LOG_MAX_BYTES": "52428800",
            "QWEN_LOG_BACKUP_COUNT": "10",
            "QWEN_LOG_STDOUT": "1",
            **overrides,
        }
    )
    return env


def _invoke(
    log_dir: Path,
    child_code: str,
    *,
    service: str = "test-service",
    **env_overrides: str,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--service",
            service,
            "--",
            sys.executable,
            "-c",
            child_code,
        ],
        env=_runner_env(log_dir, **env_overrides),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_combines_binary_stdout_and_stderr_and_rotates_exactly(tmp_path: Path):
    result = _invoke(
        tmp_path,
        "import os; os.write(1, b'abc'); os.write(2, b'defghi')",
        QWEN_LOG_MAX_BYTES="5",
        QWEN_LOG_BACKUP_COUNT="2",
    )

    assert result.returncode == 0
    assert result.stdout == b"abcdefghi"
    assert (tmp_path / "test-service.log.1").read_bytes() == b"abcde"
    assert (tmp_path / "test-service.log").read_bytes() == b"fghi"


def test_retains_only_requested_number_of_backups(tmp_path: Path):
    result = _invoke(
        tmp_path,
        "import os; os.write(1, b'abcdefghijklm')",
        QWEN_LOG_MAX_BYTES="4",
        QWEN_LOG_BACKUP_COUNT="2",
        QWEN_LOG_STDOUT="0",
    )

    assert result.returncode == 0
    assert result.stdout == b""
    assert (tmp_path / "test-service.log.2").read_bytes() == b"efgh"
    assert (tmp_path / "test-service.log.1").read_bytes() == b"ijkl"
    assert (tmp_path / "test-service.log").read_bytes() == b"m"
    assert not (tmp_path / "test-service.log.3").exists()


def test_prunes_backups_above_a_reduced_retention_limit(tmp_path: Path):
    for index in range(1, 5):
        (tmp_path / f"test-service.log.{index}").write_bytes(str(index).encode())

    result = _invoke(
        tmp_path,
        "pass",
        QWEN_LOG_BACKUP_COUNT="2",
        QWEN_LOG_STDOUT="0",
    )

    assert result.returncode == 0
    assert (tmp_path / "test-service.log.1").read_bytes() == b"1"
    assert (tmp_path / "test-service.log.2").read_bytes() == b"2"
    assert not (tmp_path / "test-service.log.3").exists()
    assert not (tmp_path / "test-service.log.4").exists()


def test_log_directory_failure_falls_back_to_stdout(tmp_path: Path):
    not_a_directory = tmp_path / "regular-file"
    not_a_directory.write_text("occupied", encoding="utf-8")

    result = _invoke(
        not_a_directory,
        "import os, sys; os.write(2, b'still-running'); sys.exit(7)",
        QWEN_LOG_STDOUT="0",
    )

    assert result.returncode == 7
    assert result.stdout == b"still-running"
    assert b"continuing with stdout only" in result.stderr


def test_closed_parent_stdout_does_not_change_child_exit_code(tmp_path: Path):
    payload = b"x" * (256 * 1024)
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "--service",
            "broken-pipe",
            "--",
            sys.executable,
            "-c",
            "import os, sys; os.write(1, b'x' * (256 * 1024)); sys.exit(9)",
        ],
        env=_runner_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    process.stdout.close()

    assert process.wait(timeout=5) == 9
    assert (tmp_path / "broken-pipe.log").read_bytes() == payload


def test_unconsumed_stdout_does_not_block_logging_or_child_exit(tmp_path: Path):
    payload_size = 4 * 1024 * 1024
    child_code = (
        "import os,sys\n"
        f"remaining = memoryview(b'x' * {payload_size})\n"
        "while remaining:\n"
        "    remaining = remaining[os.write(1, remaining):]\n"
        "sys.exit(9)\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "--service",
            "backpressure",
            "--",
            sys.executable,
            "-c",
            child_code,
        ],
        env=_runner_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Deliberately do not consume either runner output pipe before wait().
        assert process.wait(timeout=5) == 9
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    assert (tmp_path / "backpressure.log").read_bytes() == b"x" * payload_size


def test_sigterm_is_forwarded_to_the_child_process_group(tmp_path: Path):
    marker = tmp_path / "signals"
    grandchild = (
        "import os,signal,time; p=os.environ['MARKER']; "
        "signal.signal(signal.SIGTERM, lambda *_: "
        "(open(p,'ab').write(b'G'), os._exit(0))); "
        "open(p+'.ready','wb').close(); time.sleep(30)"
    )
    child = (
        "import os,signal,subprocess,sys,time; p=os.environ['MARKER']; "
        f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        "deadline=time.time()+5; "
        "\nwhile not os.path.exists(p+'.ready') and time.time()<deadline: time.sleep(.01)\n"
        "signal.signal(signal.SIGTERM, lambda *_: "
        "(open(p,'ab').write(b'P'), os._exit(23))); "
        "os.write(1,b'READY\\n'); time.sleep(30)"
    )
    env = _runner_env(tmp_path)
    env["MARKER"] = str(marker)
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "--service",
            "signals",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stdout.readline() == b"READY\n"

    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=5) == 23
    assert set(marker.read_bytes()) == {ord("P"), ord("G")}


def test_inherited_pipe_does_not_deadlock_after_direct_child_exits(tmp_path: Path):
    pid_file = tmp_path / "orphan.pid"
    grandchild = "import time; time.sleep(30)"
    child = (
        "import pathlib,subprocess,sys; "
        f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid))"
    )
    started = time.monotonic()
    result = _invoke(tmp_path, child)
    elapsed = time.monotonic() - started

    orphan_pid = int(pid_file.read_text(encoding="utf-8"))
    try:
        os.kill(orphan_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    assert result.returncode == 0
    assert elapsed < 2


def test_continuously_writing_orphan_cannot_extend_post_exit_drain(tmp_path: Path):
    pid_file = tmp_path / "noisy-orphan.pid"
    grandchild = (
        "import os,signal\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGPIPE, signal.SIG_DFL)\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "payload = b'x' * 65536\n"
        "while True:\n"
        "    os.write(1, payload)\n"
    )
    child = (
        "import subprocess,sys\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "--service",
            "noisy-orphan",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        env=_runner_env(
            tmp_path,
            QWEN_LOG_MAX_BYTES="65536",
            QWEN_LOG_BACKUP_COUNT="0",
            QWEN_LOG_STDOUT="0",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    started = time.monotonic()

    try:
        assert process.wait(timeout=5) == 0
        assert time.monotonic() - started < 2
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if pid_file.exists():
            orphan_pid = int(pid_file.read_text(encoding="utf-8"))
            cmdline_path = Path(f"/proc/{orphan_pid}/cmdline")
            try:
                cmdline = cmdline_path.read_bytes()
            except FileNotFoundError:
                cmdline = b""
            if str(pid_file).encode() in cmdline:
                try:
                    os.kill(orphan_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    assert (tmp_path / "noisy-orphan.log").stat().st_size <= 65536


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or not hasattr(os, "waitid")
    or not Path("/proc").is_dir(),
    reason="Linux subreaper test requires waitid and procfs",
)
def test_reaps_adopted_orphan_without_stealing_direct_exit_status(tmp_path: Path):
    orphan_pid_file = tmp_path / "adopted.pid"
    orphan_stop_file = tmp_path / "adopted.stop"
    service_ready_file = tmp_path / "service.ready"
    grandchild = (
        "import os,time\n"
        "from pathlib import Path\n"
        f"pid_file = Path({str(orphan_pid_file)!r})\n"
        f"stop_file = Path({str(orphan_stop_file)!r})\n"
        "pid_file.write_text(str(os.getpid()))\n"
        "while not stop_file.exists():\n"
        "    time.sleep(0.01)\n"
    )
    intermediary = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"pid_file = Path({str(orphan_pid_file)!r})\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        "deadline = time.monotonic() + 5\n"
        "while not pid_file.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "if not pid_file.exists():\n"
        "    raise SystemExit(2)\n"
    )
    child = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"subprocess.run([sys.executable, '-c', {intermediary!r}], check=True)\n"
        f"Path({str(service_ready_file)!r}).touch()\n"
        "time.sleep(30)\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "--service",
            "subreaper",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        env=_runner_env(tmp_path, QWEN_LOG_STDOUT="0"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        deadline = time.monotonic() + 5
        while not service_ready_file.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.01)
        assert service_ready_file.exists()

        orphan_pid = int(orphan_pid_file.read_text(encoding="utf-8"))
        status_path = Path(f"/proc/{orphan_pid}/status")
        status = status_path.read_text(encoding="utf-8")
        parent_line = next(
            line for line in status.splitlines() if line.startswith("PPid:")
        )
        assert int(parent_line.split()[1]) == process.pid

        orphan_stop_file.touch()
        deadline = time.monotonic() + 3
        while status_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not status_path.exists()

        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 128 + signal.SIGTERM
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def test_requires_a_command(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--service", "empty", "--"],
        env=_runner_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert b"a command is required" in result.stderr
