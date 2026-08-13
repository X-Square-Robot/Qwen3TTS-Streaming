from __future__ import annotations

import pytest

from engine.session import (
    AudioFormat,
    AudioOutput,
    AttachmentFence,
    DeliveryLedger,
    EventOutput,
    LedgerError,
    StartedOutput,
    TerminalOutput,
    TerminalStatus,
)


@pytest.mark.asyncio
async def test_typed_delivery_ledger_replays_and_fences_attachments():
    ledger = DeliveryLedger(
        token="secret",
        config_fingerprint="fingerprint",
        protocol="openai-realtime-v1",
    )
    await ledger.publish(
        StartedOutput("sid", AudioFormat("pcm_s16le", 24000, 1))
    )
    await ledger.publish(
        AudioOutput(
            "sid",
            b"\0\0" * 1200,
            AudioFormat("pcm_s16le", 24000, 1),
            0,
            1200,
        )
    )
    first = await ledger.attach(last_delivery_seq=0, audio_through_sample=0)
    first_items = [first.queue.get_nowait(), first.queue.get_nowait()]
    assert [item.delivery_seq for item in first_items] == [1, 2]

    second = await ledger.attach(last_delivery_seq=1, audio_through_sample=0)
    fenced = first.queue.get_nowait()
    assert isinstance(fenced, AttachmentFence)
    replay = second.queue.get_nowait()
    assert replay.delivery_seq == 2

    await ledger.acknowledge(
        second.generation,
        through_delivery_seq=2,
        audio_through_sample=1200,
    )
    assert ledger.retained_bytes == 0


@pytest.mark.asyncio
async def test_typed_delivery_ledger_rejects_invalid_cursors_and_counts_terminal_once():
    ledger = DeliveryLedger(
        token="secret",
        config_fingerprint="fingerprint",
        protocol="tts-session-v2alpha1",
    )
    attachment = await ledger.attach()
    with pytest.raises(LedgerError, match="ahead"):
        await ledger.acknowledge(
            attachment.generation,
            through_delivery_seq=1,
            audio_through_sample=0,
        )

    await ledger.publish(
        TerminalOutput("sid", TerminalStatus.COMPLETED, usage={"total_tokens": 1})
    )
    with pytest.raises(LedgerError, match="terminal"):
        await ledger.publish(EventOutput("sid", "late"))
    await ledger.terminal_ack(attachment.generation)
    assert ledger.terminal is not None
