"""Gateway-owned identities for one logical synthesis request.

The protocol ``session_id`` is client-controlled correlation data.  It must not
be used as the engine registry key: two clients are allowed to choose the same
value without sharing or replacing engine state.
"""

from __future__ import annotations

from dataclasses import dataclass
import uuid

from ..core.types import SessionConfig


_SAMPLING_IDENTITY_KEY = "_sampling_identity"


@dataclass(frozen=True)
class GatewaySessionIdentity:
    """Public correlation ID paired with a private engine execution ID."""

    client_session_id: str
    internal_session_id: str

    @classmethod
    def create(cls, client_session_id: str | None) -> "GatewaySessionIdentity":
        external = str(client_session_id or uuid.uuid4())
        return cls(
            client_session_id=external,
            internal_session_id=str(uuid.uuid4()),
        )

    def bind_engine_config(self, config: SessionConfig) -> None:
        """Attach stable sampling identity without exposing it on the wire.

        Historically the engine's random seed was derived from the public
        ``session_id``.  The engine now receives a random private UUID, so the
        original public value is carried separately to keep the same sampling
        trajectory for an otherwise identical request.
        """

        config.timing.extra[_SAMPLING_IDENTITY_KEY] = self.client_session_id


__all__ = [
    "GatewaySessionIdentity",
    "_SAMPLING_IDENTITY_KEY",
]
