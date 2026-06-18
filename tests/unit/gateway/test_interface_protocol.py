from __future__ import annotations

import json

from engine.core.types import AudioConfig, AudioEncoding, SessionConfig
from engine.interface import (
    OutputPipeline,
    SessionStartRequest,
    build_done_event,
    build_start_event,
    normalize_capabilities,
    parse_output_policy,
    parse_timing_context,
    serialize_output_policy,
    serialize_stream_event,
    serialize_timing_context,
    to_core_output_policy,
    to_core_timing_context,
)
from engine.interface.output import TIMING_CONTRACT
from engine.interface.protocol import PROTOCOL_VERSION


def test_output_policy_and_timing_round_trip_preserves_fields():
    output_policy = parse_output_policy(
        {
            "vad_policy": {
                "enabled": True,
                "strategy": "tail_guard",
                "implementation": "tenvad",
                "config": {"max_non_speech_frames": 8},
            },
            "chunk_ms": 16,
            "packet_format": "raw_pcm",
            "emit_text_events": False,
            "config": {"transport": "websocket"},
        }
    )
    timing = parse_timing_context(
        {
            "request_id": "req-1",
            "turn_id": "turn-1",
            "client_request_ts_ms": 1710000000000,
            "client_text_ts_ms": 1710000000100,
            "client_end_ts_ms": 1710000000200,
            "extra": {"network_rtt_ms": "12"},
        }
    )

    serialized_policy = serialize_output_policy(output_policy)
    serialized_timing = serialize_timing_context(timing)
    core_policy = to_core_output_policy(output_policy)
    core_timing = to_core_timing_context(timing)

    assert serialized_policy["vad_policy"]["enabled"] is True
    assert serialized_policy["vad_policy"]["strategy"] == "tail_guard"
    assert serialized_policy["emit_text_events"] is False
    assert serialized_timing["request_id"] == "req-1"
    assert serialized_timing["extra"]["network_rtt_ms"] == "12"
    assert core_policy.vad.implementation == "tenvad"
    assert core_policy.vad.config["max_non_speech_frames"] == 8
    assert core_timing.turn_id == "turn-1"


def test_default_vad_policy_is_disabled_and_noop_contract_only():
    policy = parse_output_policy({})
    timing = parse_timing_context({})

    cfg = SessionConfig(audio=AudioConfig(sample_rate=24000, encoding=AudioEncoding.PCM_F32))
    start = SessionStartRequest(
        session_id="sid",
        config=cfg,
        output_policy=policy,
        timing=timing,
    )
    event = serialize_stream_event(build_start_event("sid", start))

    assert policy.vad.enabled is False
    assert policy.vad.strategy == "disabled"
    assert event["meta"]["vad_enabled"] == "false"
    assert event["meta"]["vad_strategy"] == "disabled"
    assert json.loads(event["meta"]["output_policy_json"])["vad_policy"]["enabled"] is False


def test_output_pipeline_emits_server_timing_contract():
    timing = parse_timing_context(
        {
            "request_id": "req-2",
            "turn_id": "turn-2",
            "client_request_ts_ms": 1710000000000,
        }
    )
    cfg = SessionConfig(audio=AudioConfig(sample_rate=24000, encoding=AudioEncoding.PCM_S16LE))
    start = SessionStartRequest(
        session_id="sid",
        config=cfg,
        output_policy=parse_output_policy({}),
        timing=timing,
    )
    pipeline = OutputPipeline(start, request_received_monotonic=1.0, request_received_epoch_ms=1000)

    frame = pipeline.convert_audio_chunk(bytes([0, 0, 0, 0]))
    done = serialize_stream_event(build_done_event("sid", {}, pipeline))

    assert frame.first_chunk is True
    assert frame.meta["timing_contract"] == TIMING_CONTRACT
    assert frame.meta["first_audio_chunk"] == "true"
    assert frame.audio.encoding == AudioEncoding.PCM_S16LE
    assert done["type"] == "done"
    assert done["meta"]["request_id"] == "req-2"
    assert done["meta"]["turn_id"] == "turn-2"
    assert done["meta"]["client_request_ts_ms"] == "1710000000000"
    assert done["meta"]["audio_chunk_count"] == "1"


def test_normalize_capabilities_adds_interface_contract_fields():
    caps = normalize_capabilities({"variant": "custom-1.7b"})

    assert caps["protocol_version"] == PROTOCOL_VERSION
    assert "vad_policy" in caps["supported_output_policy_features"]
    assert "prefix_trim" in caps["supported_vad_strategies"]
    assert "server_ttft_ms" in caps["supported_timing_fields"]
