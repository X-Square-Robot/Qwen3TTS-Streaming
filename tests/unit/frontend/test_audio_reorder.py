from __future__ import annotations

import asyncio

import pytest

from engine.core.session import SegmentOrderMeta
from engine.core.types import EngineResult, ResultType, SessionConfig
from engine.frontend.interface import FrontendInterface
from engine.frontend.spliter.reorder import AudioReorder


def test_reorder_stall_is_measured_per_blocked_playhead() -> None:
    clock = [100.0]
    reorder = AudioReorder(stall_timeout_sec=1.0, time_fn=lambda: clock[0])

    assert reorder.push(0, 1, b"later") == []
    assert reorder.check_stall() is None
    clock[0] += 0.99
    assert reorder.check_stall() is None
    clock[0] += 0.02

    stall = reorder.check_stall()
    assert stall is not None
    assert stall.blocked_segment == (0, 0)
    assert stall.buffered_chunks == 1

    # Completing the predecessor releases the buffered segment and clears the
    # watchdog state; no implicit out-of-order drop occurs.
    assert reorder.mark_done(0, 0, group_final=False) == [b"later"]
    assert reorder.check_stall() is None


def test_reorder_stall_timer_resets_when_playhead_advances() -> None:
    clock = [10.0]
    reorder = AudioReorder(stall_timeout_sec=1.0, time_fn=lambda: clock[0])

    reorder.push(0, 1, b"second")
    clock[0] += 0.8
    assert reorder.mark_done(0, 0, group_final=False) == [b"second"]
    # The playhead is now (0, 1); a later segment starts a fresh wait budget.
    reorder.push(0, 2, b"third")
    clock[0] += 0.3
    assert reorder.check_stall() is None
    clock[0] += 0.8
    stall = reorder.check_stall()
    assert stall is not None
    assert stall.blocked_segment == (0, 1)


def test_reorder_zero_timeout_keeps_diagnostics_without_timeout_verdict() -> None:
    clock = [0.0]
    reorder = AudioReorder(stall_timeout_sec=0.0, time_fn=lambda: clock[0])
    reorder.push(1, 0, b"later")
    clock[0] += 100.0

    assert reorder.check_stall() is None
    state = reorder.pending_state()
    assert state["stall_key"] == [0, 0]
    assert state["stall_timeout_ms"] == 0.0


@pytest.mark.parametrize("value", [-1.0, float("inf"), "bad"])
def test_reorder_rejects_invalid_timeout(value) -> None:
    with pytest.raises(ValueError):
        AudioReorder(stall_timeout_sec=value)


class _Tokenizer:
    def encode_with_text(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return list(range(len(text))), list(text)


@pytest.mark.asyncio
async def test_frontend_watchdog_cancels_stalled_reorder_with_reason() -> None:
    inbox = asyncio.Queue()
    interface = FrontendInterface(
        engine_inbox=inbox,
        tokenizer=_Tokenizer(),
        reorder_stall_timeout_ms=20,
    )
    events: list[dict] = []
    done: asyncio.Future = asyncio.get_running_loop().create_future()

    async def on_done(_session_id: str, metrics: dict) -> None:
        done.set_result(metrics)

    async def on_event(_session_id: str, event: dict) -> None:
        events.append(event)

    session = await interface.create_session(
        "reorder-stall",
        config=SessionConfig(),
        on_done=on_done,
        on_event=on_event,
    )
    session.segment_order[1] = SegmentOrderMeta(1, 0, True)

    # Drop NEW_SESSION; the test stands in for the engine thread and waits for
    # the watchdog's internal cancellation request.
    while not inbox.empty():
        await inbox.get()
    await session.result_queue.put(
        EngineResult(
            type=ResultType.AUDIO_CHUNK,
            session_id=session.session_id,
            segment_idx=1,
            audio_bytes=b"later",
        )
    )
    cancel = None
    for _ in range(20):
        request = await asyncio.wait_for(inbox.get(), timeout=0.1)
        if request.type.name == "CANCEL_SESSION":
            cancel = request
            break
    assert cancel is not None
    assert cancel.cancel_reason == "audio_reorder_stall_timeout"
    assert any(
        event.get("type") == "warning"
        and event.get("meta", {}).get("reason") == "audio_reorder_stall_timeout"
        for event in events
    )

    await session.result_queue.put(
        EngineResult(
            type=ResultType.SESSION_DONE,
            session_id=session.session_id,
            metrics={"cancelled": True, "cancel_reason": cancel.cancel_reason},
        )
    )
    metrics = await asyncio.wait_for(done, timeout=1.0)
    assert metrics["cancel_reason"] == "audio_reorder_stall_timeout"
