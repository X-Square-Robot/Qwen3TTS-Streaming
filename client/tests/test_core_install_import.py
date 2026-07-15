"""Core install (no extras) must be importable.

The base wheel only depends on `requests`; grpcio/protobuf/tritonclient/numpy
arrive via extras. These tests simulate that environment in a subprocess with
an import blocker, guarding against any module-level import of an optional
dependency sneaking back into the eager `import qwen3tts` chain
(the original bug: engine_grpc.py importing the generated pb2 modules at
module level, which require protobuf/grpcio).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CLIENT_SRC = Path(__file__).resolve().parents[1] / "src"

# Top-level module names that only extras provide.
_BLOCKED = ("grpc", "google", "tritonclient", "numpy")

_CORE_ONLY_SCRIPT = f"""
import sys

BLOCKED = {_BLOCKED!r}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in BLOCKED:
            raise ImportError(f"blocked optional dependency: {{name}}")
        return None


sys.meta_path.insert(0, _Blocker())

import qwen3tts  # noqa: F401  (must not touch any blocked module)
from qwen3tts import AsyncTTSClient, TTSClient  # noqa: F401
from qwen3tts.exceptions import DependencyMissingError
from qwen3tts._adapters import engine_grpc

try:
    engine_grpc._require_grpc()
except DependencyMissingError:
    print("GUARD_OK")
else:
    raise AssertionError("_require_grpc() should raise without the grpc extra")
"""


def _run_in_core_only_env(script: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CLIENT_SRC)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_import_qwen3tts_without_optional_dependencies():
    result = _run_in_core_only_env(_CORE_ONLY_SCRIPT)
    assert result.returncode == 0, result.stderr
    assert "GUARD_OK" in result.stdout
