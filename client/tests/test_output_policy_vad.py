"""Client-side VAD policy coverage.

The client SDK can *request* non-default VAD strategies (``energy`` / ``tenvad``)
and must serialize/send them faithfully across every transport. The actual VAD
*effect* (audio trimming) is server-side and is covered by
``tests/unit/peripheral/test_vad_processor.py`` — these tests only verify the
client correctly puts the requested policy on the wire.
"""

from __future__ import annotations

import pytest

from qwen3tts_protocol import (
    OutputPolicy,
    VADPolicy,
    parse_output_policy,
    serialize_output_policy,
)


def _vad_policy(strategy: str) -> OutputPolicy:
    return OutputPolicy(
        vad=VADPolicy(
            enabled=True,
            strategy=strategy,
            implementation="onnx",
            config={"model": "ten_vad"},
            chunk_ms=16,
            begin_threshold=0.6,
            begin_count=5,
            end_threshold=0.35,
            end_count=31,
            start_margin_ms=20,
        )
    )


class TestSerializeOutputPolicyVad:
    @pytest.mark.parametrize("strategy", ["energy", "tenvad"])
    def test_non_disabled_strategy_carries_tuning_fields(self, strategy):
        vad = serialize_output_policy(_vad_policy(strategy))["vad_policy"]

        assert vad["enabled"] is True
        assert vad["strategy"] == strategy
        assert vad["implementation"] == "onnx"
        assert vad["config"] == {"model": "ten_vad"}
        # The non-disabled branch must add the VAD tuning parameters.
        assert vad["chunk_ms"] == 16
        assert vad["begin_threshold"] == 0.6
        assert vad["begin_count"] == 5
        assert vad["end_threshold"] == 0.35
        assert vad["end_count"] == 31
        assert vad["start_margin_ms"] == 20

    def test_disabled_strategy_omits_tuning_fields(self):
        # Default OutputPolicy() has strategy == "disabled".
        vad = serialize_output_policy(OutputPolicy())["vad_policy"]

        assert vad["strategy"] == "disabled"
        for field_name in (
            "chunk_ms",
            "begin_threshold",
            "begin_count",
            "end_threshold",
            "end_count",
            "start_margin_ms",
        ):
            assert field_name not in vad

    def test_tenvad_round_trips_through_parse(self):
        restored = parse_output_policy(serialize_output_policy(_vad_policy("tenvad")))

        assert restored.vad.strategy == "tenvad"
        assert restored.vad.enabled is True
        assert restored.vad.implementation == "onnx"
        assert restored.vad.config == {"model": "ten_vad"}
        assert restored.vad.begin_threshold == 0.6
        assert restored.vad.end_count == 31
        assert restored.vad.start_margin_ms == 20


class TestEngineGrpcVadMapping:
    def test_output_policy_to_proto_maps_tenvad(self):
        pytest.importorskip("grpc")
        from qwen3tts._adapters.engine_grpc import _output_policy_to_proto

        proto = _output_policy_to_proto(
            OutputPolicy(vad=VADPolicy(enabled=True, strategy="tenvad", config={"k": "v"}))
        )

        assert proto.vad_policy.strategy == "tenvad"
        assert proto.vad_policy.enabled is True
        assert dict(proto.vad_policy.config) == {"k": "v"}
