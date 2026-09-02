from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
AUTORUN = REPO_ROOT / "scripts" / "bash" / "autorun.sh"


def _run_autorun(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(AUTORUN), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_autorun_dry_run_accepts_release_and_package_metadata_without_writing():
    model_sidecar = (
        REPO_ROOT / "workspace" / "exported" / "custom-1.7b" / "MODEL_VERSION"
    )
    before = model_sidecar.read_bytes()

    result = _run_autorun(
        "package",
        "-m",
        "custom-1.7b",
        "--gateway",
        "standalone",
        "--model-version",
        "2",
        "--model-release-version",
        "zehan@20260820",
        "--packager",
        "packager-test",
        "--package-date",
        "2026-08-20",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "模型版本:  zehan@20260820" in output
    assert "引擎编译:  Phase B 自动生成" in output
    assert "打包人:    packager-test" in output
    assert "打包日期:  2026-08-20" in output
    assert "Would write MODEL_VERSION=zehan@20260820" in output
    assert "引擎编译版本" not in output
    assert model_sidecar.read_bytes() == before


def test_autorun_rejects_package_timestamp_instead_of_day_precision():
    result = _run_autorun(
        "package",
        "-m",
        "custom-1.7b",
        "--package-date",
        "2026-08-20T16:20:02+08:00",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "YYYY-MM-DD" in result.stderr


def test_autorun_rejects_one_model_release_for_combined_variant():
    result = _run_autorun(
        "package",
        "-m",
        "all-1.7b",
        "--model-release-version",
        "researcher@20260820",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "Cannot apply one model release version" in result.stderr
