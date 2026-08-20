"""Static guards for the wheel-first GitHub and GitLab release paths.

The release contract is deliberately simple: tag CI builds one wheel without
submodules, publishes it, and the image job downloads those same bytes.  These
checks catch accidental reintroduction of a source/VCS install or a second
wheel build in the image job.
"""

import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _job(ci: str, name: str, next_name: str) -> str:
    return ci.split(f"\n{name}:\n", 1)[1].split(f"\n{next_name}:\n", 1)[0]


def _assert_gitlab_command_strings(value, *, depth: int = 0) -> None:
    """Mirror GitLab's string-or-nested-string-array command contract."""
    assert depth <= 10
    if isinstance(value, str):
        return
    assert isinstance(value, list)
    for item in value:
        _assert_gitlab_command_strings(item, depth=depth + 1)


def test_gitlab_shell_commands_parse_as_strings():
    config = yaml.safe_load(_read(".gitlab-ci.yml"))

    for job_name, job in config.items():
        if not isinstance(job, dict):
            continue
        for key in ("before_script", "script", "after_script"):
            if key in job:
                try:
                    _assert_gitlab_command_strings(job[key])
                except AssertionError as exc:
                    raise AssertionError(f"{job_name}.{key} contains a non-string") from exc


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


def test_runner_downloads_default_to_overridable_china_mirrors():
    ci = _read(".gitlab-ci.yml")
    github_ci = _read(".github/workflows/ci.yml")
    github_release = _read(".github/workflows/release.yml")
    dockerfile = _read("infra/docker/Dockerfile.engine")

    for workflow in (ci, github_ci, github_release):
        assert "https://mirrors.bfsu.edu.cn/pypi/web/simple" in workflow

    assert "https://mirrors.nju.edu.cn/pytorch/whl/cpu" in github_ci
    assert 'pip install --index-url "$PYTORCH_CPU_INDEX" torch' in github_ci
    assert "m.daocloud.io/docker.io/library/python:3.11-slim" in ci
    assert "m.daocloud.io/docker.io/library/docker:27.4.1-dind" in ci
    assert "m.daocloud.io/nvcr.io/nvidia/pytorch:25.10-py3" in ci
    assert "m.daocloud.io/nvcr.io/nvidia/pytorch:26.02-py3" in github_release
    assert "${DEBIAN_MIRROR}" in ci
    assert "${ALPINE_MIRROR}" in ci

    assert "ARG PIP_INDEX_URL=" in dockerfile
    assert "ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3" in dockerfile
    assert "PYTORCH_INDEX_BASE" not in dockerfile
    assert dockerfile.count("    pip install \\") == 1
    assert "PYTORCH_CUDA_TAG" not in dockerfile
    assert "torch_cuda == base_cuda" in dockerfile
    for workflow in (ci, github_release):
        assert '--build-arg "BASE_IMAGE=$ENGINE_BASE_IMAGE"' in workflow
        assert '--build-arg "PIP_INDEX_URL=$PIP_INDEX_URL"' in workflow


def test_engine_release_build_reuses_registry_layers_and_has_a_timeout():
    ci = _read(".gitlab-ci.yml")
    github = _read(".github/workflows/release.yml")
    image_job = _job(ci, "build-engine-image", "create-release")
    github_image_job = github.split("\n  build-engine-image:\n", 1)[1].split(
        "\n  finalize-release:\n", 1
    )[0]

    assert "timeout: 3h" in image_job
    assert "resource_group: engine-image-cache" in image_job
    assert "ENGINE_BUILD_CACHE_IMAGE" in ci
    assert 'DOCKER_BUILDKIT: "1"' in github_image_job
    for job in (image_job, github_image_job):
        assert "--cache-from" in job
        assert 'BUILDKIT_INLINE_CACHE=1' in job
        assert ":buildcache" in job or "ENGINE_BUILD_CACHE_IMAGE" in job


def test_gitlab_engine_image_uses_the_x2robot_registry():
    ci = _read(".gitlab-ci.yml")
    image_job = _job(ci, "build-engine-image", "create-release")
    release_job = ci.split("\ncreate-release:\n", 1)[1]

    assert 'X2ROBOT_REGISTRY: "cr.x2robot.cn"' in ci
    assert 'X2ROBOT_IMAGE: "cr.x2robot.cn/audio/qwen3tt-streaming"' in ci
    assert 'X2ROBOT_IMAGE_TAG_PREFIX: "trt25.10_580_cu13_"' in ci
    assert "${X2ROBOT_IMAGE}:${X2ROBOT_IMAGE_TAG_PREFIX}${CI_COMMIT_TAG}" in image_job
    assert "${X2ROBOT_IMAGE}:${X2ROBOT_IMAGE_TAG_PREFIX}buildcache" in image_job
    assert "X2ROBOT_REGISTRY_USER is required" in image_job
    assert "X2ROBOT_REGISTRY_PASSWORD is required" in image_job
    assert '--password-stdin "$X2ROBOT_REGISTRY"' in image_job
    assert "$CI_REGISTRY" not in image_job
    assert "ENGINE_RELEASE_IMAGE" in release_job


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
    gitlab_release = _read("scripts/bash/create_gitlab_release.sh")
    github = _read(".github/workflows/release.yml")
    markdown = "\n".join(
        path.read_text(encoding="utf-8")
        for path in REPO_ROOT.rglob("*.md")
        if "third_party" not in path.parts and "workspace" not in path.parts
    )

    assert "gitlab-org/cli:v1.112.0" in gitlab
    assert "sh scripts/bash/create_gitlab_release.sh" in gitlab
    assert (
        'glab config set api_protocol "$CI_SERVER_PROTOCOL" '
        '--host "$CI_SERVER_FQDN"' in gitlab_release
    )
    assert 'glab api --hostname "$CI_SERVER_FQDN" job --silent' in gitlab_release
    assert '--repo "$CI_PROJECT_PATH"' in gitlab_release
    assert '--hostname "$CI_SERVER_FQDN"' in gitlab_release
    assert '--form "name=$release_link_name"' in gitlab_release
    assert '--form "url=$release_link_url"' in gitlab_release
    assert '--form "direct_asset_path=$release_link_path"' in gitlab_release
    assert "$WHEEL_REGISTRY_URL" in gitlab_release
    assert "$WHEEL_RELEASE_URL" in gitlab_release
    assert "require_release_environment" in gitlab_release
    assert "artifacts/raw" not in gitlab
    assert "gh release upload" in github
    vcs_prefix = "git" + "+"
    client_subdirectory = "subdirectory=" + "client"
    assert not (vcs_prefix in markdown and client_subdirectory in markdown)
