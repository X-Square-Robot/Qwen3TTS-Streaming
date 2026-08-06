"""Static guards for the wheel-first GitHub and GitLab release paths.

The release contract is deliberately simple: tag CI builds one wheel without
submodules, publishes it, and the image job downloads those same bytes.  These
checks catch accidental reintroduction of a source/VCS install or a second
wheel build in the image job.
"""

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _job(ci: str, name: str, next_name: str) -> str:
    return ci.split(f"\n{name}:\n", 1)[1].split(f"\n{next_name}:\n", 1)[0]


def test_tag_pipeline_builds_one_wheel_without_submodules():
    ci = _read(".gitlab-ci.yml")
    github = _read(".github/workflows/release.yml")

    assert 'GIT_SUBMODULE_STRATEGY: "none"' in ci
    assert 'GIT_DEPTH: "0"' in ci
    assert ci.count("scripts/bash/release_client_wheel.sh") == 1
    assert "submodules: false" in github
    assert "fetch-depth: 0" in github
    assert github.count("scripts/bash/release_client_wheel.sh") == 1

    build_job = _job(ci, "build-client-wheel", "publish-client-wheel")
    assert "git submodule status" in build_job
    assert "Expected exactly one client wheel" in build_job


def test_image_consumes_published_wheel_instead_of_rebuilding_it():
    ci = _read(".gitlab-ci.yml")
    github = _read(".github/workflows/release.yml")
    publish_job = _job(ci, "publish-client-wheel", "build-engine-image")
    image_job = _job(ci, "build-engine-image", "create-release")
    github_image_job = github.split("\n  build-engine-image:\n", 1)[1].split(
        "\n  finalize-release:\n", 1
    )[0]

    assert "/packages/pypi" in publish_job
    assert "python -m twine upload" in publish_job
    assert "WHEEL_REGISTRY_URL" in image_job
    assert "WHEEL_SHA256" in image_job
    assert '--output "client/dist/$WHEEL_FILENAME"' in image_job
    assert '--build-arg "CLIENT_WHEEL_FILENAME=$WHEEL_FILENAME"' in image_job
    assert '--build-arg "CLIENT_WHEEL_SHA256=$WHEEL_SHA256"' in image_job
    assert "release_client_wheel.sh" not in image_job
    assert "pip wheel" not in image_job
    assert "compose.sh" not in image_job
    assert "gh release download" in github_image_job
    assert "--dir client/dist" in github_image_job
    assert "WHEEL_SHA256" in github_image_job
    assert '--build-arg "CLIENT_WHEEL_FILENAME=$WHEEL_FILENAME"' in github_image_job
    assert '--build-arg "CLIENT_WHEEL_SHA256=$WHEEL_SHA256"' in github_image_job
    assert "release_client_wheel.sh" not in github_image_job
    assert "pip wheel" not in github_image_job
    assert "compose.sh" not in github_image_job


def test_engine_image_fails_closed_on_wrong_wheel_bytes():
    dockerfile = _read("infra/docker/Dockerfile.engine")
    compose = _read("infra/docker/compose.yaml")

    assert 'ARG CLIENT_WHEEL_FILENAME=""' in dockerfile
    assert 'ARG CLIENT_WHEEL_SHA256=""' in dockerfile
    assert (
        '[ -n "${CLIENT_WHEEL_FILENAME}" ] || [ -n "${CLIENT_WHEEL_SHA256}" ]'
        in dockerfile
    )
    assert "Expected exactly one staged client wheel" in _read(".gitlab-ci.yml")
    assert "Expected exactly one staged client wheel" in _read(
        ".github/workflows/release.yml"
    )
    assert "find /app/sdk -maxdepth 1" in dockerfile
    assert "sha256sum -c -" in dockerfile
    assert "CLIENT_WHEEL_FILENAME: ${CLIENT_WHEEL_FILENAME:-}" in compose
    assert "CLIENT_WHEEL_SHA256: ${CLIENT_WHEEL_SHA256:-}" in compose


def test_release_wheels_remain_artifacts_not_git_sources():
    gitignore = _read(".gitignore")
    dockerignore = _read(".dockerignore")
    dockerignore_rules = {
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "client/dist/*" in gitignore
    assert "!client/dist/.gitkeep" in gitignore
    assert ".cache/" in gitignore
    assert "client/dist" not in {rule.rstrip("/") for rule in dockerignore_rules}
    assert "third_party" in {rule.rstrip("/") for rule in dockerignore_rules}
    tracked_wheels = subprocess.run(
        ["git", "ls-files", "client/dist/*.whl"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert tracked_wheels == ""


def test_release_links_are_durable_and_docs_never_use_vcs_installs():
    gitlab = _read(".gitlab-ci.yml")
    github = _read(".github/workflows/release.yml")
    markdown = "\n".join(
        path.read_text(encoding="utf-8")
        for path in REPO_ROOT.rglob("*.md")
        if "third_party" not in path.parts and "workspace" not in path.parts
    )

    assert '--repo "$CI_PROJECT_URL"' in gitlab
    assert '"url=$WHEEL_REGISTRY_URL"' in gitlab
    assert '"direct_asset_path=/client-sdk/$WHEEL_FILENAME"' in gitlab
    assert "$WHEEL_REGISTRY_URL" in gitlab
    assert "$WHEEL_RELEASE_URL" in gitlab
    assert "artifacts/raw" not in gitlab
    assert "gh release upload" in github
    vcs_prefix = "git" + "+"
    client_subdirectory = "subdirectory=" + "client"
    assert not (vcs_prefix in markdown and client_subdirectory in markdown)
