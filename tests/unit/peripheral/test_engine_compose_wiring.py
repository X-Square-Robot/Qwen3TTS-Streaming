from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_engine_compose_publishes_demo_on_websocket_port_only():
    compose = yaml.safe_load(_read("infra/docker/compose.yaml"))
    engine = compose["services"]["engine"]

    assert engine["ports"] == [
        "${ENGINE_GRPC_PORT:-50051}:${ENGINE_GRPC_PORT:-50051}",
        "${ENGINE_WEBSOCKET_PORT:-50052}:${ENGINE_WEBSOCKET_PORT:-50052}",
    ]
    assert engine["environment"]["ENGINE_HEALTH_PORT"] == "${ENGINE_HEALTH_PORT:-8080}"
    assert "ENGINE_HEALTH_PORT" in engine["healthcheck"]["test"][1]


def test_compose_wrapper_uses_container_health_without_publishing_health_port():
    script = _read("scripts/bash/compose.sh")

    assert '"$service" "$container" "$ENGINE_PORT" "$ENGINE_WEBSOCKET"' in script
    assert 'compose_assert_service_network engine "$ENGINE_PORT" "$ENGINE_WEBSOCKET"' in script
    assert "compose_wait_engine_container_health 90" in script
    assert "Engine container health ready; public /health is on" in script
