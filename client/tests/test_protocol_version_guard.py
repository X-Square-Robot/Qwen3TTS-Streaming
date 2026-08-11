"""Connect-time guard: server and SDK protocol majors must be compatible.

Revisions in one protocol family and major are compatible; family or major
skew fails fast. The funnel is ``capabilities_from_payload`` (all four
transports' ``get_capabilities``) plus the auto-detect probes.
"""

from __future__ import annotations

import pytest

from qwen3tts._internal.utils import capabilities_from_payload, check_protocol_version
from qwen3tts.exceptions import ProtocolVersionMismatchError
from qwen3tts_protocol.protocol import (
    PROTOCOL_VERSION,
    protocol_compatibility_key,
    protocol_versions_compatible,
)


def test_matching_version_passes():
    check_protocol_version(PROTOCOL_VERSION)


@pytest.mark.parametrize(
    "server_version",
    ["tts-session-v2alpha2", "tts-session-v2beta1", "tts-session-v2.1"],
)
def test_same_protocol_major_passes(server_version):
    check_protocol_version(server_version)


def test_protocol_compatibility_key_includes_family_and_major():
    assert protocol_compatibility_key("tts-session-v2alpha1") == ("tts-session", 2)
    assert protocol_versions_compatible("tts-session-v2alpha1", "tts-session-v2alpha9")
    assert not protocol_versions_compatible("other-session-v2", PROTOCOL_VERSION)
    assert protocol_compatibility_key("unversioned") is None


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


def test_same_numeric_major_in_different_family_raises():
    with pytest.raises(ProtocolVersionMismatchError):
        check_protocol_version("other-session-v2alpha1")


def test_unknown_nonmatching_version_format_raises():
    with pytest.raises(ProtocolVersionMismatchError):
        check_protocol_version("legacy-protocol")


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
