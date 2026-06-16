"""Triton-specific request types and payload builders.

These are the wire-format types for talking to the Triton orchestrator
backend.  They live in the protocol layer so that both the demo_api and
tests/tools can share them without depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Triton request type
# ---------------------------------------------------------------------------

@dataclass
class TtsRequest:
    """High-level TTS request parameters for the Triton orchestrator."""
    text: str
    speaker: str = "Serena"
    language: str = "auto"
    cache_mode: str = "hit"
    task_type: str = "custom_voice"
    audio_encoding: str = "pcm_f32"
    sample_rate: int = 24000
    input_mode: str | None = None
    group_policy: str | None = None


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def build_payload(request: TtsRequest) -> dict[str, Any]:
    """Build a Triton ``request`` payload dict from a TtsRequest.

    This is the canonical payload for a ``synthesize`` or single-shot
    Triton infer call.
    """
    payload: dict[str, Any] = {
        "text": request.text,
        "task_type": request.task_type,
        "speaker": request.speaker,
        "language": request.language,
        "cache_mode": request.cache_mode,
        "audio": {
            "encoding": request.audio_encoding,
            "sample_rate": request.sample_rate,
            "channels": 1,
        },
    }
    if request.input_mode:
        payload["input_mode"] = request.input_mode
    if request.group_policy:
        payload["group_policy"] = request.group_policy
    return payload


def build_action_payload(
    action: str,
    session_id: str,
    *,
    text: str = "",
    request: TtsRequest | None = None,
) -> dict[str, Any]:
    """Build a payload for one ``tts_orchestrator`` action.

    ``init`` / ``synthesize`` carry the full session config (speaker,
    language, audio format).  ``append_text`` carries ``session_id`` +
    ``text``.  ``text_complete`` and ``cancel`` only carry ``session_id``.
    """
    payload: dict[str, Any] = {"action": action, "session_id": session_id}
    if action in ("init", "start", "synthesize"):
        if request is None:
            raise ValueError(f"action {action!r} requires a TtsRequest")
        payload.update(
            {
                "task_type": request.task_type,
                "speaker": request.speaker,
                "language": request.language,
                "cache_mode": request.cache_mode,
                "audio": {
                    "encoding": request.audio_encoding,
                    "sample_rate": request.sample_rate,
                    "channels": 1,
                },
            }
        )
        if request.input_mode:
            payload["input_mode"] = request.input_mode
        if request.group_policy:
            payload["group_policy"] = request.group_policy
        if action == "synthesize":
            payload["text"] = text or request.text
    elif action in ("append_text", "append"):
        payload["text"] = text
    return payload


def build_request_payload(
    *,
    text: str,
    task_type: str = "",
    language: str = "auto",
    speaker: str = "",
    instruct: str = "",
    action: str = "",
    session_id: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Low-level request builder (compat with tests/support/triton_streaming).

    Prefer :func:`build_payload` for new code; this function exists to
    ease migration from the old ``tests.support.triton_streaming`` module.
    """
    payload: dict[str, Any] = {
        "text": text,
        "language": language,
    }
    if task_type:
        payload["task_type"] = task_type
    if speaker:
        payload["speaker"] = speaker
    if instruct:
        payload["instruct"] = instruct
    if action:
        payload["action"] = action
    if session_id:
        payload["session_id"] = session_id
    if extra:
        payload.update(extra)
    return payload


def build_variant_request_payload(
    *,
    variant: str,
    text: str,
    language: str = "auto",
    speaker: str = "",
    instruct: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a request payload auto-detecting task_type from variant name."""
    normalized = variant.lower()
    if normalized.startswith("design"):
        return build_request_payload(
            text=text,
            task_type="voice_design",
            language=language,
            instruct=instruct,
            extra=extra,
        )
    if normalized.startswith("custom"):
        return build_request_payload(
            text=text,
            task_type="custom_voice",
            language=language,
            speaker=speaker,
            instruct=instruct,
            extra=extra,
        )
    raise ValueError(
        f"Variant {variant!r} is not supported by this helper; use custom-* or design-*."
    )


def build_text_stream_requests(
    init_req: dict[str, Any],
    text_chunks: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Build a sequence of Triton streaming request dicts.

    Given an init request dict and a list of text chunks, returns a list
    of: [init_req, append_text×N, text_complete].
    """
    session_id = init_req.get("session_id")
    if not session_id:
        raise ValueError("init_req must include a session_id for streaming text input")

    requests = [dict(init_req)]
    requests.extend(
        {
            "action": "append_text",
            "session_id": session_id,
            "text": chunk_text,
        }
        for chunk_text in text_chunks
    )
    requests.append(
        {
            "action": "text_complete",
            "session_id": session_id,
        }
    )
    return requests


__all__ = [
    "TtsRequest",
    "build_action_payload",
    "build_payload",
    "build_request_payload",
    "build_text_stream_requests",
    "build_variant_request_payload",
]
