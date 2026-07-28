from engine.core.types import SessionConfig
from engine.gateway.session_identity import GatewaySessionIdentity


def test_gateway_identity_separates_client_and_engine_ids():
    first = GatewaySessionIdentity.create("client-request")
    second = GatewaySessionIdentity.create("client-request")

    assert first.client_session_id == second.client_session_id == "client-request"
    assert first.internal_session_id != second.internal_session_id
    assert "client-request" not in {
        first.internal_session_id,
        second.internal_session_id,
    }


def test_gateway_identity_generates_public_id_and_binds_stable_sampling_identity():
    identity = GatewaySessionIdentity.create(None)
    config = SessionConfig()
    config.timing.extra["_sampling_identity"] = "untrusted-client-value"

    identity.bind_engine_config(config)

    assert identity.client_session_id
    assert identity.internal_session_id
    assert identity.client_session_id != identity.internal_session_id
    assert config.timing.extra["_sampling_identity"] == identity.client_session_id
