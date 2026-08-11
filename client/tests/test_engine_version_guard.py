"""Connect-time SDK/engine release-skew diagnostics, via capabilities.

The engine advertises its release stamp in the ``engine_version`` capability.
It is useful diagnostic metadata, but only ``protocol_version`` defines wire
compatibility. Release skew must warn without rejecting a compatible client.
"""

from __future__ import annotations

import pytest

import qwen3tts
from qwen3tts._internal import utils as u
from qwen3tts._internal.utils import (
    capabilities_from_payload,
    check_capabilities_pairing,
    check_engine_version,
)
from qwen3tts.exceptions import ProtocolVersionMismatchError
from qwen3tts_protocol import capabilities_from_mapping
from qwen3tts_protocol.protocol import PROTOCOL_VERSION


# ── version normalization ────────────────────────────────────────────────


def test_normalize_release_strips_leading_v():
    assert u._normalize_release("v0.2.0") == "0.2.0"
    assert u._normalize_release(" 0.2.0 ") == "0.2.0"
    assert u._normalize_release(None) == ""


# ── check_engine_version ─────────────────────────────────────────────────


def test_matched_release_passes_despite_v_prefix(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    check_engine_version("v0.2.0")  # engine git-describe stamp vs SDK PEP 440


def test_empty_engine_version_tolerated(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    check_engine_version("")  # pre-versioning engine build
    check_engine_version(None)


def test_release_mismatch_warns_and_continues(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    with pytest.warns(RuntimeWarning) as warning_list:
        check_engine_version("v0.3.0")
    msg = str(warning_list[0].message)
    assert "0.2.0" in msg and "0.3.0" in msg
    assert "/sdk/" in msg
    assert "Continuing" in msg


def test_dev_build_skew_warns(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.1.dev5+gabc123")
    with pytest.warns(RuntimeWarning):
        check_engine_version("v0.3.0")


# ── check_capabilities_pairing: compatibility plus diagnostics ──────────


def test_pairing_checks_protocol_first(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    with pytest.raises(ProtocolVersionMismatchError):
        check_capabilities_pairing(
            {"protocol_version": "tts-session-v99", "engine_version": "v0.2.0"}
        )


def test_pairing_warns_for_engine_skew_when_protocol_ok(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    with pytest.warns(RuntimeWarning):
        check_capabilities_pairing(
            {"protocol_version": PROTOCOL_VERSION, "engine_version": "v0.3.0"}
        )


def test_pairing_passes_when_both_match(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    check_capabilities_pairing(
        {"protocol_version": PROTOCOL_VERSION, "engine_version": "v0.2.0"}
    )


# ── capabilities_from_payload is the transport funnel ────────────────────


def test_capabilities_payload_allows_engine_skew(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    with pytest.warns(RuntimeWarning):
        caps = capabilities_from_payload(
            {
                "loaded_model_type": "custom",
                "protocol_version": PROTOCOL_VERSION,
                "engine_version": "v0.3.0",
            }
        )
    assert caps.engine_version == "v0.3.0"


def test_capabilities_payload_matching_versions_parse(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    caps = capabilities_from_payload(
        {
            "loaded_model_type": "custom",
            "protocol_version": PROTOCOL_VERSION,
            "engine_version": "v0.2.0",
        }
    )
    assert caps.engine_version == "v0.2.0"


# ── engine_version rides the Capabilities dataclass ──────────────────────


def test_capabilities_mapping_carries_engine_version():
    caps = capabilities_from_mapping(
        {"loaded_model_type": "custom", "engine_version": "v0.2.0"}
    )
    assert caps.engine_version == "v0.2.0"
    # explicit field, not swept into extra
    assert "engine_version" not in caps.extra
