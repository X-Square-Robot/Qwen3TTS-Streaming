"""Contract tests for the local-only release candidate tag."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/bash/prepare_release_checkout.sh"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _candidate_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Release Test")
    (repo / "README").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "candidate")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_prepare_script_creates_only_a_local_tag(tmp_path: Path):
    repo, commit = _candidate_repo(tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), "v1.2.3", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "RELEASE_VERSION=v1.2.3" in result.stdout
    assert f"RELEASE_COMMIT_SHA={commit}" in result.stdout
    assert _git(repo, "rev-parse", "refs/tags/v1.2.3") == commit
    assert _git(repo, "branch", "--show-current") == ""
    assert _git(repo, "status", "--porcelain") == ""


def test_prepare_script_refuses_a_conflicting_existing_tag(tmp_path: Path):
    repo, first_commit = _candidate_repo(tmp_path)
    _git(repo, "tag", "v1.2.3", first_commit)
    (repo / "README").write_text("second candidate\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "second candidate")

    result = subprocess.run(
        ["bash", str(SCRIPT), "v1.2.3", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "already points to" in result.stderr


def test_prepare_script_rejects_an_unresolvable_source_ref(tmp_path: Path):
    repo, _ = _candidate_repo(tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), "v1.2.4", "missing-ref"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Could not resolve release source ref: missing-ref" in result.stderr
