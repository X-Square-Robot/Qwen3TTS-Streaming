from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_triton_profile_starts_openai_realtime_sidecar_with_internal_grpc():
    compose = yaml.safe_load(_read("infra/docker/compose.yaml"))
    service = compose["services"]["realtime-gateway"]
    triton_build_args = compose["services"]["triton"]["build"]["args"]

    assert service["profiles"] == ["triton"]
    assert service["depends_on"] == ["triton"]
    assert service["environment"]["TRITON_GRPC_ENDPOINT"] == "triton:8001"
    assert service["environment"]["TRITON_REALTIME_PORT"] == 50052
    assert service["ports"] == ["${TRITON_REALTIME_HOST_PORT:-50053}:50052"]
    assert "gpus" not in service
    assert any("workspace/realtime_usage" in volume for volume in service["volumes"])
    assert triton_build_args["CLIENT_WHEEL_FILENAME"] == "${CLIENT_WHEEL_FILENAME:-}"
    assert triton_build_args["CLIENT_WHEEL_SHA256"] == "${CLIENT_WHEEL_SHA256:-}"
    assert triton_build_args["TRITON_RUNTIME_BASE_IMAGE"] == (
        "${TRITON_RUNTIME_BASE_IMAGE:-triton-deps}"
    )
    assert service["command"][-3:] == [
        "python3",
        "-m",
        "engine.gateway.triton_realtime_server",
    ]


def test_triton_image_contains_sidecar_runtime_dependencies_and_protocol():
    dockerfile = _read("infra/docker/Dockerfile.triton")

    dependency_runs = dockerfile.split("RUN --mount=type=cache,target=/root/.cache/pip")
    assert len(dependency_runs) == 3
    assert "    attrs \\\n" not in dependency_runs[1]
    assert "    attrs \\\n" in dependency_runs[2]
    assert "    attrs \\\n" in dockerfile
    assert "aiohttp" in dockerfile
    assert "import aiohttp, attr" in dockerfile
    assert "ARG TRITON_RUNTIME_BASE_IMAGE=triton-deps" in dockerfile
    assert "FROM ${TRITON_RUNTIME_BASE_IMAGE} AS triton-runtime" in dockerfile
    assert '"tritonclient[grpc]>=2.54.0"' in dockerfile
    assert (
        "COPY client/src/qwen3tts_protocol/ /opt/qwen3-tts/qwen3tts_protocol/"
        in dockerfile
    )
    assert 'ARG CLIENT_WHEEL_FILENAME=""' in dockerfile
    assert 'ARG CLIENT_WHEEL_SHA256=""' in dockerfile
    assert "COPY client/dist/ /app/sdk/" in dockerfile
    assert "find /app/sdk -maxdepth 1" in dockerfile
    assert "sha256sum -c -" in dockerfile


def test_compose_wrapper_manages_sidecar_with_triton_lifecycle():
    script = _read("scripts/bash/compose.sh")
    deploy = _read("scripts/bash/deploy.sh")
    autorun = _read("scripts/bash/autorun.sh")

    assert "_compose_up_exec triton realtime-gateway" in script
    assert "compose_wait_triton_realtime_ready" in script
    assert "args+=(triton realtime-gateway)" in script
    assert "compose_cmd stop realtime-gateway triton" in script
    assert '--realtime-port) TRITON_REALTIME="$2"' in script
    assert '--realtime-port)  COMPOSE_EXTRA+=(--realtime-port "$2")' in deploy
    assert '--realtime-port)    REALTIME_PORT="$2"' in autorun
    assert "append_optarg DEPLOY_ARGS --realtime-port" in autorun


def test_release_pipelines_build_both_version_matched_runtime_images():
    github = _read(".github/workflows/release.yml")
    gitlab = _read(".gitlab-ci.yml")

    for pipeline in (github, gitlab):
        assert "--file infra/docker/Dockerfile.engine" in pipeline
        assert "--file infra/docker/Dockerfile.triton" in pipeline
        assert "BROWSER_SDK_VERSION" in pipeline
        assert "CLIENT_WHEEL_SHA256" in pipeline
        assert "/app/demo/index.html" in pipeline
        assert "engine.distribution.container_smoke --runtime standalone" in pipeline
        assert "engine.distribution.container_smoke --runtime triton" in pipeline
        assert 'TRITON_RUNTIME_BASE_IMAGE=' in pipeline
        assert '--build-arg "TRITON_RUNTIME_BASE_IMAGE=' in pipeline

    assert "triton_image" in github
    assert "TRITON_RELEASE_IMAGE" in gitlab


def test_release_uses_an_immutable_prebuilt_triton_dependency_base():
    github_release = _read(".github/workflows/release.yml")
    github_base = _read(".github/workflows/triton-runtime-base.yml")
    gitlab = _read(".gitlab-ci.yml")
    publisher = _read("scripts/bash/publish_triton_runtime_base.sh")

    assert 'BUILD_TRITON_RUNTIME_BASE == "1"' in gitlab
    assert "build-triton-runtime-base:" in gitlab
    assert "Required Triton runtime base is missing" in gitlab
    assert "Required Triton runtime base is missing" in github_release
    assert "workflow_dispatch:" in github_base
    assert "scripts/bash/publish_triton_runtime_base.sh" in gitlab
    assert "scripts/bash/publish_triton_runtime_base.sh" in github_base
    assert "--target triton-deps" in publisher
    assert 'docker push "$runtime_base_image"' in publisher
    assert "docker manifest inspect" in publisher


def test_web_release_stamps_only_the_publishable_workspace_without_registry_resolution():
    script = _read("scripts/bash/release_web.sh")

    assert "--workspace @xmultimodalinteraction/qwen3tts-browser" in script
    assert "--workspaces-update=false" in script
    assert "--workspace @xmultimodalinteraction/qwen3tts-demo" not in script
    assert script.index("npm run build") < script.index('npm version "$semver"')
    assert script.index('npm version "$semver"') < script.index("npm run pack:browser")


def test_local_compose_uses_reproducible_node_builder_for_web_artifacts():
    compose_script = _read("scripts/bash/compose.sh")
    dockerfile = _read("infra/docker/Dockerfile.web-builder")

    assert "Dockerfile.web-builder" in compose_script
    assert "--target web-artifacts" in compose_script
    assert '--output "type=local,dest=$staged_web"' in compose_script
    assert "FROM ${NODE_IMAGE} AS web-builder" in dockerfile
    assert "npm --prefix web ci" in dockerfile
    assert "npm --prefix web run build" in dockerfile
    assert "npm --prefix web run pack:browser" in dockerfile
    assert "/out/demo/downloads" in dockerfile
    assert "browser-sdk-tarball.txt" in dockerfile
    assert "FROM scratch AS web-artifacts" in dockerfile


def test_runtime_images_fail_closed_when_demo_artifact_is_missing():
    for path in (
        "infra/docker/Dockerfile.engine",
        "infra/docker/Dockerfile.triton",
    ):
        dockerfile = _read(path)
        assert "COPY web/packages/demo/dist/ /app/demo/" in dockerfile
        assert "test -s /app/demo/index.html" in dockerfile
        assert "xmultimodalinteraction-qwen3tts-browser-*.tgz" in dockerfile
        assert "Demo artifact missing" in dockerfile
