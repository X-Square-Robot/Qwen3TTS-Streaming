"""Connect-time pairing guard: server protocol generation must match the SDK.

Engine and client wheels are built from the same git tag; this guard turns a
mispaired install from silent protocol breakage into an immediate, actionable
error. The funnel is ``capabilities_from_payload`` (all four transports'
``get_capabilities``) plus the auto-detect probes.
"""

from __future__ import annotations

import pytest

from qwen3tts._internal.utils import capabilities_from_payload, check_protocol_version
from qwen3tts.exceptions import ProtocolVersionMismatchError
from qwen3tts_protocol.protocol import PROTOCOL_VERSION


def test_matching_version_passes():
    check_protocol_version(PROTOCOL_VERSION)


def test_missing_or_empty_version_tolerated():
    check_protocol_version(None)
    check_protocol_version("")
    check_protocol_version("   ")


def test_mismatch_raises_with_actionable_message():
    with pytest.raises(ProtocolVersionMismatchError) as excinfo:
        check_protocol_version("tts-session-v99")
    message = str(excinfo.value)
    assert "tts-session-v99" in message
    assert PROTOCOL_VERSION in message
    assert "/sdk/" in message
    assert "GitHub/GitLab Release or Package Registry" in message
    assert "capabilities.engine_version" in message


def test_env_escape_hatch_downgrades_to_warning(monkeypatch):
    monkeypatch.setenv("QWEN3TTS_SKIP_PROTOCOL_CHECK", "1")
    with pytest.warns(RuntimeWarning):
        check_protocol_version("tts-session-v99")


def test_capabilities_payload_funnel_enforces_guard():
    with pytest.raises(ProtocolVersionMismatchError):
        capabilities_from_payload(
            {"loaded_model_type": "custom", "protocol_version": "tts-session-v99"}
        )


def test_capabilities_payload_matching_version_parses():
    caps = capabilities_from_payload(
        {"loaded_model_type": "custom", "protocol_version": PROTOCOL_VERSION}
    )
    assert caps.protocol_version == PROTOCOL_VERSION
