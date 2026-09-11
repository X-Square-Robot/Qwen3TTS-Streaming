"""Contract tests for idempotent candidate promotion helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/bash/promote_container_image.sh"


def _write_fake_docker(tmp_path: Path) -> Path:
    docker = tmp_path / "docker"
    docker.write_text(
        """#!/bin/sh
set -eu

printf '%s\\n' "$*" >> "$DOCKER_CALL_LOG"

case "${1:-}" in
  pull)
    exit 0
    ;;
  image)
    image=
    for arg in "$@"; do image="$arg"; done
    if [ "$image" = "$CANDIDATE_IMAGE" ]; then
      printf 'registry.example/candidate@%s\\n' "$CANDIDATE_DIGEST"
    else
      printf 'registry.example/release@%s\\n' "$RELEASE_DIGEST"
    fi
    ;;
  manifest)
    case "$DOCKER_RELEASE_STATE" in
      missing)
        echo 'no such manifest' >&2
        exit 1
        ;;
      unknown)
        echo 'denied: requested access' >&2
        exit 1
        ;;
      same|different)
        exit 0
        ;;
      *)
        echo "unknown test state" >&2
        exit 2
        ;;
    esac
    ;;
  tag|push)
    exit 0
    ;;
  *)
    echo "unexpected docker command: $*" >&2
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return docker


def _run(
    tmp_path: Path, state: str, release_digest: str
) -> subprocess.CompletedProcess[str]:
    _write_fake_docker(tmp_path)
    candidate = "registry.example/candidate:candidate-1"
    release = "registry.example/candidate:v1.2.3"
    return subprocess.run(
        ["bash", str(SCRIPT), candidate, release, "sha256:candidate"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "DOCKER_CALL_LOG": str(tmp_path / "docker.log"),
            "DOCKER_RELEASE_STATE": state,
            "CANDIDATE_IMAGE": candidate,
            "RELEASE_IMAGE": release,
            "CANDIDATE_DIGEST": "sha256:candidate",
            "RELEASE_DIGEST": release_digest,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_missing_release_image_is_promoted(tmp_path: Path):
    result = _run(tmp_path, "missing", "sha256:candidate")

    assert result.returncode == 0
    calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert (
        "tag registry.example/candidate:candidate-1 registry.example/candidate:v1.2.3"
        in calls
    )
    assert "push registry.example/candidate:v1.2.3" in calls


def test_matching_release_image_is_left_untouched(tmp_path: Path):
    result = _run(tmp_path, "same", "sha256:candidate")

    assert result.returncode == 0
    calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert "tag " not in calls
    assert "push " not in calls


def test_different_release_image_is_rejected(tmp_path: Path):
    result = _run(tmp_path, "different", "sha256:other")

    assert result.returncode != 0
    assert "already contains different bytes" in result.stderr
    calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert "tag " not in calls
    assert "push " not in calls


def test_registry_access_error_is_not_treated_as_missing(tmp_path: Path):
    result = _run(tmp_path, "unknown", "sha256:release")

    assert result.returncode != 0
    assert "Could not determine whether release image exists" in result.stderr
