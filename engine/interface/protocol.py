from __future__ import annotations

import json

from ..core.types import OutputPolicyConfig, TimingConfig, VADConfig
from .types import OutputPolicy, StreamEvent, TimingContext

# 再导出枢纽：从共享协议层 qwen3tts_protocol 拉取并对 engine.interface 其余模块再暴露。
# 这些名字在本文件内不直接使用，但被 output.py / __init__.py / gateway 等 import，
# 属有意再导出（per-file-ignore F401 见 pyproject.toml）。
from qwen3tts_protocol.protocol import (
    PROTOCOL_VERSION,
    SUPPORTED_OUTPUT_POLICY_FEATURES,
    SUPPORTED_TIMING_FIELDS,
    SUPPORTED_VAD_STRATEGIES,
    normalize_capabilities,
    parse_output_policy,
    parse_timing_context,
    serialize_output_policy,
    serialize_timing_context,
)


def serialize_stream_event(event: StreamEvent) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": event.type,
        "session_id": event.session_id,
        "segment_id": int(event.segment_id),
        "text": event.text,
        "message": event.message,
        "meta": dict(event.meta or {}),
    }
    if event.audio is not None:
        payload["audio"] = {
            "encoding": event.audio.encoding.value,
            "sample_rate": int(event.audio.sample_rate),
            "channels": int(event.audio.channels),
        }
    return payload


def output_policy_json(policy: OutputPolicy) -> str:
    return json.dumps(
        serialize_output_policy(policy), ensure_ascii=False, sort_keys=True
    )


def timing_context_json(timing: TimingContext) -> str:
    return json.dumps(
        serialize_timing_context(timing), ensure_ascii=False, sort_keys=True
    )


def to_core_output_policy(policy: OutputPolicy) -> OutputPolicyConfig:
    return OutputPolicyConfig(
        vad=VADConfig(
            enabled=bool(policy.vad.enabled),
            strategy=str(policy.vad.strategy or "disabled"),
            implementation=str(policy.vad.implementation or ""),
            config={str(k): v for k, v in dict(policy.vad.config or {}).items()},
            chunk_ms=int(policy.vad.chunk_ms or 16),
            begin_threshold=float(policy.vad.begin_threshold or 0.6),
            begin_count=int(policy.vad.begin_count or 5),
            end_threshold=float(policy.vad.end_threshold or 0.35),
            end_count=int(policy.vad.end_count or 31),
            start_margin_ms=int(policy.vad.start_margin_ms or 20),
        ),
        chunk_ms=int(policy.chunk_ms or 0),
        packet_format=str(policy.packet_format or "raw_pcm"),
        emit_text_events=bool(policy.emit_text_events),
        config={str(k): v for k, v in dict(policy.config or {}).items()},
    )


def to_core_timing_context(timing: TimingContext) -> TimingConfig:
    return TimingConfig(
        request_id=str(timing.request_id or ""),
        turn_id=str(timing.turn_id or ""),
        client_request_ts_ms=int(timing.client_request_ts_ms or 0),
        client_text_ts_ms=int(timing.client_text_ts_ms or 0),
        client_end_ts_ms=int(timing.client_end_ts_ms or 0),
        extra={str(k): v for k, v in dict(timing.extra or {}).items()},
    )
