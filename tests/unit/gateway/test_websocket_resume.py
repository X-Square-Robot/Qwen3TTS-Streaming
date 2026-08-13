from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.gateway.session_identity import GatewaySessionIdentity
from engine.gateway.websocket_resume import (
    ResumeProtocolError,
    ResumableSessionRegistry,
)


class _Engine:
    async def cancel(self, _session_id):
        return None


@pytest.mark.asyncio
async def test_playback_feedback_is_telemetry_only_and_validated():
    registry = ResumableSessionRegistry(
        _Engine(), grace_seconds=30.0, max_buffer_bytes=1024 * 1024
    )
    try:
        session, created = await registry.claim_start(
            token="resume-token",
            identity=GatewaySessionIdentity.create("client"),
            start_request=SimpleNamespace(),
            config_fingerprint="fingerprint",
        )
        assert created is True
        attachment = await session.attach(last_delivery_seq=0, audio_through_sample=0)

        await session.put(
            {
                "type": "audio",
                "audio": {
                    "pcm_data": b"\0" * 200,
                    "sample_rate": 24000,
                    "encoding": "pcm_s16le",
                    "channels": 1,
                    "meta": {},
                },
            }
        )
        await session.record_playback_progress(
            attachment.generation,
            played_through_sample=50,
            buffered_through_sample=100,
        )
        assert session.played_through_sample == 50
        assert session.buffered_through_sample == 100

        # Delayed feedback is idempotently ignored.
        await session.record_playback_progress(
            attachment.generation,
            played_through_sample=40,
            buffered_through_sample=40,
        )
        assert session.played_through_sample == 50

        with pytest.raises(ResumeProtocolError, match="buffered_through_sample"):
            await session.record_playback_progress(
                attachment.generation,
                played_through_sample=60,
                buffered_through_sample=50,
            )
        with pytest.raises(ResumeProtocolError, match="ahead of server output"):
            await session.record_playback_progress(
                attachment.generation,
                played_through_sample=50,
                buffered_through_sample=101,
            )

        # Once delivery 1 is trimmed, an exact replay of its old telemetry is
        # still harmless; a partially newer report must identify a retained
        # delivery and cannot use the expired sequence as RB.
        await session.acknowledge(
            attachment.generation,
            through_delivery_seq=1,
            audio_through_sample=100,
        )
        await session.record_playback_progress(
            attachment.generation,
            played_through_sample=50,
            buffered_through_sample=100,
            observed_delivery_seq=0,
        )
        with pytest.raises(ResumeProtocolError, match="no longer in the replay ledger"):
            await session.record_playback_progress(
                attachment.generation,
                played_through_sample=60,
                buffered_through_sample=100,
                observed_delivery_seq=0,
            )
    finally:
        await registry.close()
