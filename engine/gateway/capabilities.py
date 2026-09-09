"""Canonical capability discovery shared by all public HTTP gateways."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from ..interface import normalize_capabilities


CAPABILITIES_SCHEMA_VERSION = "qwen.tts.capabilities.v1"
NATIVE_WEBSOCKET_PROTOCOL = "tts-session-v2alpha1"
OPENAI_REALTIME_PATH = "/v1/realtime"
OPENAI_REALTIME_PROTOCOL = "openai-realtime-v1"
QWEN_REALTIME_EXTENSION_PROTOCOL = "qwen-realtime-v1"
QWEN_TEXT_BUFFER_EXTENSION = "qwen.input_text_buffer.v1"
QWEN_TEXT_PROGRESS_EXTENSION = "qwen.text_progress.v1"
QWEN_PLAYBACK_ACK_EXTENSION = "qwen.playback_ack.v1"
QWEN_RESPONSE_RESUME_EXTENSION = "qwen.response_resume.v1"
REFERENCE_AUDIO_MAX_BYTES = 4 * 1024 * 1024
REFERENCE_AUDIO_MIME_TYPES = ("audio/wav", "audio/x-wav")
REALTIME_MAX_MESSAGE_BYTES = 8 * 1024 * 1024

_PUBLIC_TASKS = {"base", "voice_clone", "custom_voice", "voice_design"}
_PUBLIC_AUDIO_ENCODINGS = {"pcm_f32", "pcm_s16le"}
_PUBLIC_PROGRESS_MODES = ("ema", "disabled")


def _public_task(value: Any) -> str:
    normalized = str(value or "").strip()
    return "voice_clone" if normalized == "icl" else normalized


def _public_native_cursor_capability(declared: Mapping[str, Any]) -> dict[str, Any]:
    """Expose graph admission separately from the progress publisher route.

    A cursor-enabled TRT artifact can be loaded before the TN-to-Label-Plan
    bridge and CPU projector are wired into production.  Advertising the
    artifact as native progress support in that state would make clients pick
    a route that the engine cannot actually serve.
    """

    raw = declared.get("native_cursor")
    if not isinstance(raw, Mapping):
        raw = {}
    # Public capability projection is a trust boundary.  A native route is
    # meaningful only when the fused graph is actually loaded, and malformed
    # string values must not become truthy through Python's bool conversion.
    graph_enabled = raw.get("enabled") is True
    progress_available = graph_enabled and raw.get("progress_available") is True
    modes = list(_PUBLIC_PROGRESS_MODES)
    if progress_available:
        modes.insert(0, "native")
    result: dict[str, Any] = {
        "graph_enabled": graph_enabled,
        "progress_available": progress_available,
        "supported_progress_modes": modes,
    }
    if graph_enabled and not progress_available:
        result["reason"] = str(
            raw.get(
                "reason",
                "cursor graph is loaded but TN/Label Plan and progress publishing are not connected",
            )
        )
    for key in (
        "max_labels",
        "vocab_size",
        "cursor_head_sha256",
        "cursor_vocab_sha256",
        "cursor_rules_sha256",
        "model_fingerprint",
    ):
        if key in raw:
            result[key] = raw[key]
    return result


def _public_speech_state_capability(declared: Any) -> dict[str, Any] | None:
    """Project backend state details onto the small public wire contract.

    Standalone engines internally expose typed handle/operation metadata, but
    clients only negotiate whether handoff is available and why it is not.
    Triton already publishes this reduced shape; applying it here keeps both
    runtime paths byte-for-byte compatible.
    """

    if not isinstance(declared, Mapping):
        return None
    supported = declared.get("supported", declared.get("enabled", False))
    result: dict[str, Any] = {"supported": supported is True}
    if "reason" in declared:
        result["reason"] = str(declared.get("reason", "") or "")
    return result


class RuntimeType(str, Enum):
    STANDALONE = "standalone"
    TRITON = "triton"


def build_gateway_capabilities(
    base: Mapping[str, Any] | None,
    *,
    runtime_type: RuntimeType,
    backend: str,
    native_path: str,
    resume_grace_ms: int,
    resume_max_buffer_bytes: int,
    model_name: str = "",
    model_version: str = "",
) -> dict[str, Any]:
    """Build the versioned public contract without inferring model features."""

    declared = dict(base or {})
    capabilities = normalize_capabilities(declared)
    public_speech_state = _public_speech_state_capability(
        capabilities.get("speech_state")
    )
    if public_speech_state is not None:
        capabilities["speech_state"] = public_speech_state
    public_vad_strategies = [
        str(value)
        for value in declared.get("supported_vad_strategies", ()) or ()
        if str(value) in {"disabled", "energy", "tenvad"}
    ]
    native_features = [
        "persistent_sessions_v1",
        "stream_resume_v1",
        "playback_progress_v1",
    ]
    realtime_extensions = [
        QWEN_TEXT_BUFFER_EXTENSION,
        QWEN_TEXT_PROGRESS_EXTENSION,
        QWEN_PLAYBACK_ACK_EXTENSION,
        QWEN_RESPONSE_RESUME_EXTENSION,
    ]
    progress_features = [
        "text_progress_anchor_v1",
        "playback_progress_v1",
        QWEN_TEXT_PROGRESS_EXTENSION,
    ]
    public_tasks = [
        _public_task(value)
        for value in capabilities.get("declared_supported_task_types") or ()
        if _public_task(value) in _PUBLIC_TASKS
    ]
    public_tasks = list(dict.fromkeys(public_tasks))
    loaded_task = _public_task(capabilities.get("loaded_model_type"))
    if loaded_task in _PUBLIC_TASKS:
        public_tasks = [task for task in public_tasks if task == loaded_task] or [
            loaded_task
        ]
    if not capabilities.get("ref_audio_available", False):
        public_tasks = [task for task in public_tasks if task != "voice_clone"]
    public_audio_formats = [
        dict(value)
        for value in capabilities.get("supported_audio_formats") or ()
        if isinstance(value, Mapping)
        and value.get("encoding") in _PUBLIC_AUDIO_ENCODINGS
        and isinstance(value.get("sample_rate"), int)
        and not isinstance(value.get("sample_rate"), bool)
        and int(value["sample_rate"]) > 0
        and value.get("channels") == 1
    ]
    profile = capabilities.get("engine_profile")
    max_input_tokens = (
        int(profile.get("max_input_len", 0) or 0) if isinstance(profile, Mapping) else 0
    )
    capabilities.update(
        {
            "schema_version": CAPABILITIES_SCHEMA_VERSION,
            "runtime": {"type": runtime_type.value, "backend": backend},
            "backend": backend,
            "model": model_name or str(capabilities.get("variant", "")),
            "model_version": model_version,
            "supported_api_protocols": [
                NATIVE_WEBSOCKET_PROTOCOL,
                OPENAI_REALTIME_PROTOCOL,
            ],
            "openai_realtime_path": OPENAI_REALTIME_PATH,
            "native_websocket_path": native_path,
            "supported_websocket_features": native_features[:-1],
            "supported_realtime_extensions": realtime_extensions,
            "supported_progress_features": progress_features,
            "stream_resume_grace_ms": int(resume_grace_ms),
            "stream_resume_max_buffer_bytes": int(resume_max_buffer_bytes),
            "protocols": {
                "native_websocket": {
                    "path": native_path,
                    "current": NATIVE_WEBSOCKET_PROTOCOL,
                    "supported": [NATIVE_WEBSOCKET_PROTOCOL],
                    "features": native_features,
                    "audio_formats": ["pcm_f32", "pcm_s16le"],
                },
                "openai_realtime": {
                    "path": OPENAI_REALTIME_PATH,
                    "base": OPENAI_REALTIME_PROTOCOL,
                    "extension_protocol": QWEN_REALTIME_EXTENSION_PROTOCOL,
                    "supported_extensions": realtime_extensions,
                    "features": [
                        "base64_pcm16",
                        "full_duplex",
                        "serial_responses",
                        "active_response_resume",
                        "playback_ack",
                    ],
                    "audio_formats": ["pcm_s16le"],
                },
            },
            "tasks": public_tasks,
            "task_status": [
                {
                    "task": task,
                    "available": True,
                    "stability": "stable" if task == "custom_voice" else "experimental",
                }
                for task in public_tasks
            ],
            "speakers": list(capabilities.get("supported_speakers") or []),
            "languages": list(capabilities.get("supported_languages") or []),
            "input_modes": list(capabilities.get("supported_input_modes") or []),
            "native_cursor": _public_native_cursor_capability(capabilities),
            "audio_formats": public_audio_formats,
            "limits": {
                "max_input_tokens": max(0, max_input_tokens),
                "max_realtime_message_bytes": REALTIME_MAX_MESSAGE_BYTES,
            },
            "output_policy": {
                "features": list(
                    capabilities.get("supported_output_policy_features") or []
                ),
                "vad_strategies": list(public_vad_strategies),
            },
            "reference": {
                "available": bool(capabilities.get("ref_audio_available", False)),
                "max_duration_sec": float(
                    capabilities.get("ref_audio_max_duration_sec", 0.0) or 0.0
                ),
                "max_bytes": REFERENCE_AUDIO_MAX_BYTES,
                "mime_types": list(REFERENCE_AUDIO_MIME_TYPES),
                "speaker_encoder_available": bool(
                    capabilities.get("speaker_encoder_available", False)
                ),
                "ref_codec_available": bool(
                    capabilities.get("ref_codec_available", False)
                ),
                "icl_available": bool(capabilities.get("icl_available", False)),
                "reason": str(capabilities.get("ref_audio_reason", "") or ""),
            },
        }
    )
    return capabilities


__all__ = [
    "CAPABILITIES_SCHEMA_VERSION",
    "REALTIME_MAX_MESSAGE_BYTES",
    "REFERENCE_AUDIO_MAX_BYTES",
    "REFERENCE_AUDIO_MIME_TYPES",
    "RuntimeType",
    "build_gateway_capabilities",
]
