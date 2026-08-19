from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.gateway.capabilities import RuntimeType, build_gateway_capabilities


REPO_ROOT = Path(__file__).resolve().parents[3]


def _build(runtime_type: RuntimeType):
    return build_gateway_capabilities(
        {
            "declared_supported_task_types": ["custom_voice"],
            "supported_audio_formats": [
                {"encoding": "pcm_s16le", "sample_rate": 24000, "channels": 1}
            ],
            "ref_audio_available": False,
            "ref_audio_reason": "not loaded",
        },
        runtime_type=runtime_type,
        backend="test",
        native_path="/v1/ws",
        resume_grace_ms=30000,
        resume_max_buffer_bytes=1024,
    )


def test_runtime_capabilities_share_the_same_public_contract():
    standalone = _build(RuntimeType.STANDALONE)
    triton = _build(RuntimeType.TRITON)

    for field in (
        "schema_version",
        "supported_api_protocols",
        "protocols",
        "tasks",
        "task_status",
        "audio_formats",
        "output_policy",
        "reference",
    ):
        assert standalone[field] == triton[field]
    assert standalone["runtime"]["type"] == "standalone"
    assert triton["runtime"]["type"] == "triton"


def test_capabilities_match_cross_sdk_golden_contract():
    golden = json.loads(
        (REPO_ROOT / "protocol/contracts/golden/capabilities-core.json").read_text()
    )
    capabilities = _build(RuntimeType.STANDALONE)

    assert capabilities["schema_version"] == golden["schema_version"]
    assert capabilities["supported_api_protocols"] == golden["supported_api_protocols"]
    assert capabilities["openai_realtime_path"] == golden["openai_realtime_path"]
    assert capabilities["native_websocket_path"] == golden["native_websocket_path"]
    assert (
        capabilities["protocols"]["openai_realtime"]["supported_extensions"]
        == golden["required_realtime_extensions"]
    )


def test_icl_internal_model_type_maps_to_public_voice_clone_task():
    capabilities = build_gateway_capabilities(
        {
            "loaded_model_type": "icl",
            "declared_supported_task_types": ["icl"],
            "ref_audio_available": True,
            "supported_audio_formats": [
                {"encoding": "pcm_s16le", "sample_rate": 24000, "channels": 1}
            ],
        },
        runtime_type=RuntimeType.STANDALONE,
        backend="test",
        native_path="/v1/ws",
        resume_grace_ms=30_000,
        resume_max_buffer_bytes=1024,
    )
    assert capabilities["tasks"] == ["voice_clone"]
    assert capabilities["task_status"] == [
        {"task": "voice_clone", "available": True, "stability": "experimental"}
    ]


def test_voice_clone_is_hidden_when_reference_runtime_is_unavailable():
    capabilities = build_gateway_capabilities(
        {
            "loaded_model_type": "icl",
            "declared_supported_task_types": ["icl"],
            "ref_audio_available": False,
        },
        runtime_type=RuntimeType.TRITON,
        backend="test",
        native_path="/v1/ws",
        resume_grace_ms=30_000,
        resume_max_buffer_bytes=1024,
    )

    assert capabilities["tasks"] == []
    assert capabilities["task_status"] == []


@pytest.mark.parametrize("runtime_type", list(RuntimeType))
def test_capabilities_validate_against_json_schema(runtime_type):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (REPO_ROOT / "protocol/contracts/capabilities.schema.json").read_text()
    )
    jsonschema.validate(_build(runtime_type), schema)
