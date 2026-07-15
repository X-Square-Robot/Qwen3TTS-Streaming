"""Connect-time SDK<->engine *release* pairing guard, via capabilities.

The engine image and the client wheel are cut 1:1 from the same git tag. The
engine advertises its release stamp in the ``engine_version`` capability
(alongside ``protocol_version``); the client compares it to its own
``__version__`` at connect. This lives on the *capabilities* axis — distinct
from ``/health``, which is a pure liveness probe. The funnel is
``check_capabilities_pairing`` (auto-detect probes) and
``capabilities_from_payload`` (every transport's ``get_capabilities``).
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
from qwen3tts.exceptions import (
    EngineVersionMismatchError,
    ProtocolVersionMismatchError,
)
from qwen3tts_protocol import capabilities_from_mapping
from qwen3tts_protocol.protocol import PROTOCOL_VERSION


# ── version normalization / classification ───────────────────────────────


def test_normalize_release_strips_leading_v():
    assert u._normalize_release("v0.2.0") == "0.2.0"
    assert u._normalize_release(" 0.2.0 ") == "0.2.0"
    assert u._normalize_release(None) == ""


@pytest.mark.parametrize(
    "version,is_release",
    [
        ("v0.2.0", True),
        ("0.2.0", True),
        ("0.2.0b1", True),
        ("0.2.0rc2", True),
        ("v0.2.0-5-gabc123", False),  # git-describe distance
        ("0.2.0-5-gabc123-dirty", False),
        ("0.2.1.dev5+gabc123", False),  # hatch-vcs dev build
        ("0.0.0", False),  # source-tree fallback
        ("unknown", False),
        ("", False),
    ],
)
def test_is_clean_release(version, is_release):
    assert u._is_clean_release(version) is is_release


# ── check_engine_version ─────────────────────────────────────────────────


def test_matched_release_passes_despite_v_prefix(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    check_engine_version("v0.2.0")  # engine git-describe stamp vs SDK PEP 440


def test_empty_engine_version_tolerated(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    check_engine_version("")  # pre-versioning engine build
    check_engine_version(None)


def test_confirmed_release_mismatch_raises_actionable(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    with pytest.raises(EngineVersionMismatchError) as excinfo:
        check_engine_version("v0.3.0")
    msg = str(excinfo.value)
    assert "0.2.0" in msg and "0.3.0" in msg
    assert "/sdk/" in msg


def test_dev_build_either_side_only_warns(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.1.dev5+gabc123")
    with pytest.warns(RuntimeWarning):
        check_engine_version("v0.3.0")


def test_env_escape_hatch_downgrades_to_warning(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    monkeypatch.setenv("QWEN3TTS_SKIP_PROTOCOL_CHECK", "1")
    with pytest.warns(RuntimeWarning):
        check_engine_version("v0.3.0")


# ── check_capabilities_pairing: both axes, from one mapping ──────────────


def test_pairing_checks_protocol_first(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    with pytest.raises(ProtocolVersionMismatchError):
        check_capabilities_pairing(
            {"protocol_version": "tts-session-v99", "engine_version": "v0.2.0"}
        )


def test_pairing_checks_engine_when_protocol_ok(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    with pytest.raises(EngineVersionMismatchError):
        check_capabilities_pairing(
            {"protocol_version": PROTOCOL_VERSION, "engine_version": "v0.3.0"}
        )


def test_pairing_passes_when_both_match(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    check_capabilities_pairing(
        {"protocol_version": PROTOCOL_VERSION, "engine_version": "v0.2.0"}
    )


# ── capabilities_from_payload is the transport funnel ────────────────────


def test_capabilities_payload_enforces_engine_pairing(monkeypatch):
    monkeypatch.setattr(qwen3tts, "__version__", "0.2.0")
    with pytest.raises(EngineVersionMismatchError):
        capabilities_from_payload(
            {
                "loaded_model_type": "custom",
                "protocol_version": PROTOCOL_VERSION,
                "engine_version": "v0.3.0",
            }
        )


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
