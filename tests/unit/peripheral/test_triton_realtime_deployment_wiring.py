from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_triton_profile_starts_openai_realtime_sidecar_with_internal_grpc():
    compose = yaml.safe_load(_read("infra/docker/compose.yaml"))
    service = compose["services"]["realtime-gateway"]

    assert service["profiles"] == ["triton"]
    assert service["depends_on"] == ["triton"]
    assert service["environment"]["TRITON_GRPC_ENDPOINT"] == "triton:8001"
    assert service["environment"]["TRITON_REALTIME_PORT"] == 50052
    assert service["ports"] == ["${TRITON_REALTIME_HOST_PORT:-50053}:50052"]
    assert "gpus" not in service
    assert any("workspace/realtime_usage" in volume for volume in service["volumes"])
    assert service["command"][-3:] == [
        "python3",
        "-m",
        "engine.gateway.triton_realtime_server",
    ]


def test_triton_image_contains_sidecar_runtime_dependencies_and_protocol():
    dockerfile = _read("infra/docker/Dockerfile.triton")

    assert "aiohttp" in dockerfile
    assert '"tritonclient[grpc]>=2.54.0"' in dockerfile
    assert (
        "COPY client/src/qwen3tts_protocol/ /opt/qwen3-tts/qwen3tts_protocol/"
        in dockerfile
    )


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
