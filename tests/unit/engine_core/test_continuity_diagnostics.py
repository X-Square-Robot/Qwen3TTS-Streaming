import queue

from engine.backend.engine_loop import EngineLoop, EngineSessionGroup
from engine.core.extensions import EngineExtensions
from engine.core.types import EngineRequest, RequestType, SessionConfig


def _group(session_id="session", *, factory=None):
    request = EngineRequest(
        type=RequestType.NEW_SESSION,
        session_id=session_id,
        session_config=SessionConfig(),
    )
    return EngineSessionGroup(
        session_id,
        request,
        extensions=EngineExtensions(continuity_factory=factory),
    )


def test_continuity_factory_failure_keeps_stable_disable_reason():
    def broken_factory(_session_id, _config):
        raise RuntimeError("method package unavailable")

    group = _group(factory=broken_factory)

    assert group.extension_continuity is None
    assert group.extension_continuity_disabled is True
    assert group.extension_continuity_disabled_reason == "continuity_factory_failed"


def test_policy_failure_reason_is_recorded_once():
    group = _group(factory=lambda *_: object())
    loop = object.__new__(EngineLoop)

    loop._disable_extension(group, "c2w_restore_failed")
    loop._disable_extension(group, "later_failure")

    assert group.extension_continuity_disabled is True
    assert group.extension_continuity_disabled_reason == "c2w_restore_failed"
