"""Protocol constants and utility functions for the TTS streaming protocol.

This module defines the protocol version, supported feature sets, and
capability normalization logic that is shared between the engine and client.

Serialization/parsing functions are re-exported from the package's __init__
for backward compatibility with import paths like
``from qwen3tts_protocol.protocol import parse_output_policy``.
"""

from __future__ import annotations

from typing import Any

# Re-export serialization/parsing functions from the package
from qwen3tts_protocol import (  # noqa: F401
    parse_output_policy,
    parse_timing_context,
    serialize_output_policy,
    serialize_timing_context,
)

# ---------------------------------------------------------------------------
# Protocol version
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "tts-session-v2alpha1"

# ---------------------------------------------------------------------------
# Supported feature / strategy sets
# ---------------------------------------------------------------------------

SUPPORTED_OUTPUT_POLICY_FEATURES = frozenset(
    {
        "vad_policy",
        "chunk_ms",
        "packet_format",
        "emit_text_events",
        # Guarded delivery is server-default. Clients can tune the lead with
        # output_policy.config["delivery_window_ms"] or explicitly request
        # legacy pass-through with {"delivery": "firehose"}.
        "guarded_delivery",
        # Legacy fields (backward compatibility)
        "request_context",
        "timing_context",
    }
)

SUPPORTED_TIMING_FIELDS = frozenset(
    {
        "request_id",
        "turn_id",
        "client_request_ts_ms",
        "client_text_ts_ms",
        "client_end_ts_ms",
        # Server-side timing fields (backward compatibility)
        "server_ttft_ms",
        "server_first_audio_epoch_ms",
        "server_session_create_to_first_raw_audio_ms",
    }
)

SUPPORTED_VAD_STRATEGIES = frozenset(
    {
        "disabled",
        "energy",
        "tenvad",
        # Legacy aliases (backward compatibility)
        "prefix_trim",
        "two_stage",
        "silence_aware",
    }
)

# ---------------------------------------------------------------------------
# Capabilities normalization
# ---------------------------------------------------------------------------


def normalize_capabilities(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw capabilities dict into the canonical protocol format.

    Ensures all required fields are present and that unsupported
    features/strategies are filtered out.
    """
    if not isinstance(raw, dict):
        raw = {}

    cap = dict(raw)

    # Protocol version
    if "protocol_version" not in cap:
        cap["protocol_version"] = PROTOCOL_VERSION

    # Supported features
    if "supported_output_policy_features" not in cap:
        cap["supported_output_policy_features"] = sorted(
            SUPPORTED_OUTPUT_POLICY_FEATURES
        )
    else:
        existing = cap["supported_output_policy_features"]
        if isinstance(existing, (list, set, frozenset)):
            cap["supported_output_policy_features"] = sorted(
                {f for f in existing if f in SUPPORTED_OUTPUT_POLICY_FEATURES}
            )

    # Supported VAD strategies
    if "supported_vad_strategies" not in cap:
        cap["supported_vad_strategies"] = sorted(SUPPORTED_VAD_STRATEGIES)
    else:
        existing = cap["supported_vad_strategies"]
        if isinstance(existing, (list, set, frozenset)):
            cap["supported_vad_strategies"] = sorted(
                {s for s in existing if s in SUPPORTED_VAD_STRATEGIES}
            )

    # Supported timing fields
    if "supported_timing_fields" not in cap:
        cap["supported_timing_fields"] = sorted(SUPPORTED_TIMING_FIELDS)
    else:
        existing = cap["supported_timing_fields"]
        if isinstance(existing, (list, set, frozenset)):
            cap["supported_timing_fields"] = sorted(
                {f for f in existing if f in SUPPORTED_TIMING_FIELDS}
            )

    return cap
