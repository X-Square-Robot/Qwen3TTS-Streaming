"""Tests for guarded delivery (V3) and segment-retry frontend handling (V2):

- AudioReorder.discard: dropping a hallucinated lookahead attempt's buffer
- DeliveryHoldWindow: playhead-relative release / EOS flush / abort discard
- FrontendInterface._guarded_hold_for: output_policy opt-in parsing
- _consume_results end-to-end: held garbage tail discarded on loop_abort,
  flushed on codec_eos; EMA skips aborted segments
"""

import asyncio
from types import SimpleNamespace

import pytest

from engine.core.session import Session, SegmentOrderMeta
from engine.core.types import (
    EngineResult,
    OutputPolicyConfig,
    ResultType,
    SessionConfig,
)
from engine.frontend.hold_window import DeliveryHoldWindow
from engine.frontend.interface import FrontendInterface
from engine.frontend.spliter.reorder import AudioReorder


class TestReorderDiscard:
    def test_discard_drops_buffered_lookahead_chunks(self):
        reorder = AudioReorder()
        assert reorder.push(0, 0, b"a") == [b"a"]  # playhead: emitted
        assert reorder.push(0, 1, b"garbage1") == []  # lookahead: buffered
        assert reorder.push(0, 1, b"garbage2") == []

        assert reorder.discard(0, 1) == 2
        assert reorder.push(0, 1, b"fresh") == []  # rerun audio re-buffers

        # Finishing the playhead segment drains only the rerun audio.
        assert reorder.mark_done(0, 0) == [b"fresh"]

    def test_discard_missing_key_is_noop(self):
        reorder = AudioReorder()
        assert reorder.discard(3, 7) == 0


class TestDeliveryHoldWindow:
    def _window(self, window_sec=1.5, bps=100.0, start=1000.0):
        clock = SimpleNamespace(now=start)
        win = DeliveryHoldWindow(window_sec, bps, time_fn=lambda: clock.now)
        return win, clock

    def test_release_respects_playhead_window(self):
        win, clock = self._window()
        win.push([b"x" * 100, b"y" * 100, b"z" * 100])  # 1s of audio each

        # t=0: budget = 1.5s -> releases chunks until >=150 bytes released
        # (whole-chunk granularity overshoots by at most one chunk).
        out = win.release_due()
        assert [len(c) for c in out] == [100, 100]
        assert win.held_chunks == 1

        out = win.release_due()  # no time passed, budget unchanged
        assert out == []

        clock.now += 1.0  # budget 250 -> third chunk due
        assert [len(c) for c in win.release_due()] == [100]
        assert win.held_bytes == 0

    def test_clock_starts_at_first_release_not_construction(self):
        win, clock = self._window()
        clock.now += 100.0  # idle before first audio must not inflate budget
        win.push([b"x" * 100, b"y" * 100, b"z" * 100])
        assert [len(c) for c in win.release_due()] == [100, 100]

    def test_flush_releases_everything(self):
        win, _ = self._window(window_sec=0.5)
        win.push([b"a" * 100, b"b" * 100, b"c" * 100])
        win.release_due()
        flushed = win.flush()
        assert len(flushed) >= 1
        assert win.held_bytes == 0
        assert win.release_due() == []

    def test_discard_drops_held_without_crediting_release(self):
        win, clock = self._window()
        win.push([b"a" * 100, b"b" * 100, b"c" * 100, b"d" * 100])
        released = win.release_due()
        assert len(released) == 2
        assert win.discard() == 2
        assert win.held_bytes == 0
        # Later audio is still paced by the same playback clock.
        win.push([b"e" * 100])
        assert win.release_due() == []
        clock.now += 1.0
        assert [len(c) for c in win.release_due()] == [100]

    def test_first_chunk_is_immediate_then_rtf_lead_becomes_held(self):
        # 10 bytes = 100ms of audio. At RTF=2 a chunk arrives every 50ms.
        win, clock = self._window(window_sec=0.1, bps=100.0)

        win.push([b"a" * 10])
        assert win.release_due() == [b"a" * 10]  # no confirmation delay

        clock.now += 0.05
        win.push([b"b" * 10])
        assert win.release_due() == [b"b" * 10]

        clock.now += 0.05
        win.push([b"c" * 10])
        assert win.release_due() == []
        assert win.held_chunks == 1


class TestGuardedHoldPolicy:
    @staticmethod
    def _frontend(*, default=True, window_ms=100):
        return SimpleNamespace(
            _guarded_delivery_default=default,
            _guarded_delivery_window_ms=float(window_ms),
        )

    def _session(self, config: dict) -> Session:
        return Session(
            session_id="s1",
            config=SessionConfig(output_policy=OutputPolicyConfig(config=config)),
        )

    def test_enabled_by_default_and_explicit_firehose_disables(self):
        frontend = self._frontend()
        assert (
            FrontendInterface._guarded_hold_for(frontend, self._session({})) is not None
        )
        assert (
            FrontendInterface._guarded_hold_for(
                frontend, self._session({"delivery": "firehose"})
            )
            is None
        )

    def test_guarded_optin_with_window(self):
        hold = FrontendInterface._guarded_hold_for(
            self._frontend(),
            self._session({"delivery": "guarded", "delivery_window_ms": "800"}),
        )
        assert isinstance(hold, DeliveryHoldWindow)
        assert hold._window_sec == pytest.approx(0.8)

    def test_window_defaults_and_clamps(self):
        frontend = self._frontend()
        hold = FrontendInterface._guarded_hold_for(
            frontend, self._session({"delivery": "guarded"})
        )
        assert hold._window_sec == pytest.approx(0.1)
        hold = FrontendInterface._guarded_hold_for(
            frontend, self._session({"delivery": "GUARDED", "delivery_window_ms": "5"})
        )
        assert hold._window_sec == pytest.approx(0.1)
        hold = FrontendInterface._guarded_hold_for(
            frontend,
            self._session({"delivery": "guarded", "delivery_window_ms": "99999"}),
        )
        assert hold._window_sec == pytest.approx(10.0)
        hold = FrontendInterface._guarded_hold_for(
            frontend,
            self._session({"delivery": "guarded", "delivery_window_ms": "bogus"}),
        )
        assert hold._window_sec == pytest.approx(0.1)


class _FakeSpliter:
    def __init__(self):
        self.ratio_updates = []

    def update_ratio(self, audio_steps, text_tokens, *, overflow=False):
        self.ratio_updates.append((audio_steps, text_tokens, overflow))

    def on_segment_done(self, seg_idx):
        return []


def _fake_interface_self():
    async def _noop_async(*args, **kwargs):
        return None

    fake = SimpleNamespace(
        _dispatcher=SimpleNamespace(
            maybe_send_session_tokens_done=_noop_async,
        ),
        _dispatch_segment_actions=_noop_async,
        _cleanup_session=lambda *a, **kw: None,
        _emit_session_summary=lambda *a, **kw: None,
    )
    # Consumer tests exercise ordering and verdict settlement using compact
    # synthetic chunks.
    fake._guarded_hold_for = lambda session: (
        None
        if str(
            (session.config.output_policy.config or {}).get("delivery", "guarded")
        ).lower()
        == "firehose"
        else DeliveryHoldWindow(0.1, 24000 * 4)
    )
    return fake


def _guarded_session():
    session = Session(
        session_id="s1",
        config=SessionConfig(
            output_policy=OutputPolicyConfig(
                # ~100ms window: exactly one 1-second chunk fits before the
                # window closes (whole-chunk overshoot), the rest is held for
                # the duration of the test.
                config={"delivery": "guarded", "delivery_window_ms": "100"}
            )
        ),
    )
    session.spliter = _FakeSpliter()
    session.reorder = AudioReorder()
    session.segment_order[0] = SegmentOrderMeta(0, 0, True)
    return session


ONE_SEC_CHUNK = 24000 * 4  # 1s of f32 mono engine audio


def _run_consume(session, results, events=None, interface=None, callback_order=None):
    delivered = []
    done_meta = {}

    async def on_audio(session_id, chunk):
        delivered.append(chunk)

    async def on_done(session_id, meta):
        done_meta.update(meta)
        if callback_order is not None:
            callback_order.append("done")

    async def on_event(session_id, event):
        events.append(event)

    async def main():
        for r in results:
            session.result_queue.put_nowait(r)
        await FrontendInterface._consume_results(
            interface or _fake_interface_self(),
            session,
            on_audio=on_audio,
            on_done=on_done,
            on_event=on_event if events is not None else None,
        )

    # Own-loop pattern (not asyncio.run): asyncio.run clears the thread's
    # current event loop on exit, breaking legacy get_event_loop() callers
    # in tests that run after this module.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
    return delivered, done_meta


def _single_segment_results(eos_reason, extra_metrics=None):
    metrics = {
        "audio_steps": 25,
        "text_tokens": 5,
        "overflow": False,
        "eos_reason": eos_reason,
    }
    metrics.update(extra_metrics or {})
    return [
        EngineResult(
            type=ResultType.AUDIO_CHUNK,
            session_id="s1",
            segment_idx=0,
            audio_bytes=b"\x01" * ONE_SEC_CHUNK,
        ),
        EngineResult(
            type=ResultType.AUDIO_CHUNK,
            session_id="s1",
            segment_idx=0,
            audio_bytes=b"\x02" * ONE_SEC_CHUNK,
        ),
        EngineResult(
            type=ResultType.SEGMENT_END,
            session_id="s1",
            segment_idx=0,
            metrics=metrics,
        ),
        EngineResult(type=ResultType.SESSION_DONE, session_id="s1"),
    ]


FRAME_BYTES = 24000 * 4 * 80 // 1000  # one 80ms engine frame = 7680 bytes


class TestConsumeResultsGuarded:
    def test_session_summary_runs_after_output_policy_finalization(self):
        session = _guarded_session()
        callback_order = []
        interface = _fake_interface_self()
        interface._emit_session_summary = lambda *args, **kwargs: callback_order.append(
            "summary"
        )

        _run_consume(
            session,
            _single_segment_results("codec_eos"),
            interface=interface,
            callback_order=callback_order,
        )

        assert callback_order == ["done", "summary"]

    def test_loop_abort_discards_condemned_tail_and_skips_ema(self):
        session = _guarded_session()
        # 25 frames total, last 15 condemned -> discard covers the held chunk
        # (96000 bytes = 12.5 frames < 15 frames' worth of bytes).
        delivered, done_meta = _run_consume(
            session,
            _single_segment_results("loop_abort", {"abort_tail_frames": 15}),
        )
        # First chunk fit the 100ms window; the held garbage tail was dropped.
        assert delivered == [b"\x01" * ONE_SEC_CHUNK]
        assert session.spliter.ratio_updates == []
        assert done_meta  # SESSION_DONE reached the client

    def test_loop_abort_flushes_legit_audio_before_condemned_tail(self):
        # Three frame-aligned chunks of 12 frames each (36 frames total); only
        # the last 12 frames are condemned -> the middle chunk is legitimate
        # held speech and must be flushed, not discarded.
        chunk = FRAME_BYTES * 12
        session = _guarded_session()
        results = [
            EngineResult(
                type=ResultType.AUDIO_CHUNK,
                session_id="s1",
                segment_idx=0,
                audio_bytes=marker * chunk,
            )
            for marker in (b"\x01", b"\x02", b"\x03")
        ] + [
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="s1",
                segment_idx=0,
                metrics={
                    "audio_steps": 36,
                    "text_tokens": 5,
                    "overflow": False,
                    "eos_reason": "loop_abort",
                    "abort_tail_frames": 12,
                },
            ),
            EngineResult(type=ResultType.SESSION_DONE, session_id="s1"),
        ]
        delivered, _ = _run_consume(session, results)
        assert delivered == [b"\x01" * chunk, b"\x02" * chunk]

    def test_abort_keep_point_honors_ema_expected_length(self):
        # EMA says ~10 frames expected; even though the condemned run is only
        # the last 12 of 36 frames, the keep point clamps to the expected
        # sentence end and the rambling beyond it is dropped too.
        chunk = FRAME_BYTES * 12
        session = _guarded_session()
        session.spliter._ema_ratio = 2.0  # expected = 2.0 * 5 * 1.15 + 2 = 13
        results = [
            EngineResult(
                type=ResultType.AUDIO_CHUNK,
                session_id="s1",
                segment_idx=0,
                audio_bytes=marker * chunk,
            )
            for marker in (b"\x01", b"\x02", b"\x03")
        ] + [
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="s1",
                segment_idx=0,
                metrics={
                    "audio_steps": 36,
                    "text_tokens": 5,
                    "overflow": False,
                    "eos_reason": "loop_abort",
                    "abort_tail_frames": 12,
                },
            ),
            EngineResult(type=ResultType.SESSION_DONE, session_id="s1"),
        ]
        delivered, _ = _run_consume(session, results)
        # keep = min(36-12, 13) = 13 frames -> discard (36-13)*7680 = 23
        # frames' worth from the tail = both held chunks (24 frames).
        assert delivered == [b"\x01" * chunk]

    def test_codec_eos_flushes_held_tail_and_updates_ema(self):
        session = _guarded_session()
        delivered, _ = _run_consume(session, _single_segment_results("codec_eos"))
        assert delivered == [b"\x01" * ONE_SEC_CHUNK, b"\x02" * ONE_SEC_CHUNK]
        assert session.spliter.ratio_updates == [(25, 5, False)]

    def test_wholly_silent_abort_discards_only_retractable_hold(self):
        session = _guarded_session()
        results = _single_segment_results(
            "silent_audio_abort", {"discard_all_audio": True}
        )
        for result in results:
            if result.type == ResultType.AUDIO_CHUNK:
                result.audio_bytes = b"\x00" * ONE_SEC_CHUNK
        delivered, done_meta = _run_consume(session, results)
        # The first chunk has already gone out to minimize TTFT; only the
        # synthesized-ahead chunk that is still held can be retracted.
        assert delivered == [b"\x00" * ONE_SEC_CHUNK]
        assert done_meta

    def test_explicit_firehose_passes_everything_through(self):
        session = _guarded_session()
        session.config.output_policy.config = {"delivery": "firehose"}
        delivered, _ = _run_consume(
            session,
            _single_segment_results("loop_abort", {"abort_tail_frames": 15}),
        )
        # Without guarded delivery both chunks stream out (V1 behavior),
        # but the EMA skip still applies.
        assert len(delivered) == 2
        assert session.spliter.ratio_updates == []

    def test_segment_retry_discards_buffered_attempt(self):
        session = _guarded_session()
        session.config.output_policy.config = {"delivery": "firehose"}
        # Two segments in one group: seg0 is the playhead, seg1 the lookahead.
        session.segment_order[0] = SegmentOrderMeta(0, 0, False)
        session.segment_order[1] = SegmentOrderMeta(0, 1, True)
        results = [
            # Playhead segment 0 streams normally.
            EngineResult(
                type=ResultType.AUDIO_CHUNK,
                session_id="s1",
                segment_idx=0,
                audio_bytes=b"\x01" * 8,
            ),
            # Lookahead segment 1: garbage attempt, then engine retry.
            EngineResult(
                type=ResultType.AUDIO_CHUNK,
                session_id="s1",
                segment_idx=1,
                audio_bytes=b"\xbb" * 8,
            ),
            EngineResult(
                type=ResultType.SEGMENT_RETRY,
                session_id="s1",
                segment_idx=1,
                metrics={"retry_idx": 1, "retry_reason": "loop"},
            ),
            # Rerun audio for segment 1, then both segments finish.
            EngineResult(
                type=ResultType.AUDIO_CHUNK,
                session_id="s1",
                segment_idx=1,
                audio_bytes=b"\x02" * 8,
            ),
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="s1",
                segment_idx=0,
                metrics={
                    "audio_steps": 10,
                    "text_tokens": 5,
                    "overflow": False,
                    "eos_reason": "codec_eos",
                },
            ),
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="s1",
                segment_idx=1,
                metrics={
                    "audio_steps": 12,
                    "text_tokens": 6,
                    "overflow": False,
                    "eos_reason": "codec_eos",
                },
            ),
            EngineResult(type=ResultType.SESSION_DONE, session_id="s1"),
        ]
        delivered, _ = _run_consume(session, results)
        assert delivered == [b"\x01" * 8, b"\x02" * 8]


def _seg_end(seg_idx, eos_reason, audio_steps, tail=0):
    metrics = {
        "audio_steps": audio_steps,
        "text_tokens": 5,
        "overflow": False,
        "eos_reason": eos_reason,
    }
    if tail:
        metrics["abort_tail_frames"] = tail
    return EngineResult(
        type=ResultType.SEGMENT_END,
        session_id="s1",
        segment_idx=seg_idx,
        metrics=metrics,
    )


def _chunk(seg_idx, marker, frames=12):
    return EngineResult(
        type=ResultType.AUDIO_CHUNK,
        session_id="s1",
        segment_idx=seg_idx,
        audio_bytes=marker * (FRAME_BYTES * frames),
    )


def _two_segment_session():
    session = _guarded_session()
    session.segment_order[0] = SegmentOrderMeta(0, 0, False)
    session.segment_order[1] = SegmentOrderMeta(0, 1, True)
    return session


class TestReorderMarkDoneEx:
    def test_grouped_drain_preserves_segment_attribution(self):
        reorder = AudioReorder()
        reorder.push(0, 1, b"b1")  # lookahead, buffered
        reorder.push(0, 1, b"b2")
        reorder.push(0, 2, b"c1")  # further lookahead, buffered
        # seg (0,1) and (0,2) finish before the playhead (0,0).
        assert reorder.mark_done_ex(0, 1) == []
        assert reorder.mark_done_ex(0, 2, group_final=True) == []
        # Playhead finishes -> one chained drain, attributed per segment.
        groups = reorder.mark_done_ex(0, 0)
        assert groups == [
            ((0, 0), [], True),
            ((0, 1), [b"b1", b"b2"], True),
            ((0, 2), [b"c1"], True),
        ]


class TestConsumeResultsOutOfOrder:
    """Segments complete out of order; verdicts must hit their own audio."""

    def test_lookahead_eos_does_not_flush_playhead_held_tail(self):
        # seg1 (lookahead) reaches codec EOS while seg0 (playhead) is still
        # streaming; seg0 then loop-aborts. Its condemned held tail must be
        # discarded — a premature flush at seg1's END would have leaked it.
        session = _two_segment_session()
        c0a, c0b, c1a = b"\x01", b"\x02", b"\x03"
        results = [
            _chunk(0, c0a),  # released by the 100ms window
            _chunk(0, c0b),  # held
            _chunk(1, c1a),  # buffered in reorder
            _seg_end(1, "codec_eos", 12),
            _seg_end(0, "loop_abort", 24, tail=12),  # condemns c0b
            EngineResult(type=ResultType.SESSION_DONE, session_id="s1"),
        ]
        delivered, _ = _run_consume(session, results)
        assert delivered == [c0a * (FRAME_BYTES * 12), c1a * (FRAME_BYTES * 12)]

    def test_lookahead_abort_discards_its_own_tail_at_drain_time(self):
        # seg1 (lookahead) aborts first: its garbage sits in the reorder, not
        # the hold. The verdict must apply when seg1's audio drains — not to
        # whatever the hold contains at its SEGMENT_END (seg0's audio).
        session = _two_segment_session()
        c0a, c1a, c1b = b"\x01", b"\x02", b"\x03"
        results = [
            _chunk(0, c0a),  # released
            _chunk(1, c1a),  # buffered in reorder
            _chunk(1, c1b),  # buffered in reorder (condemned tail)
            _seg_end(1, "loop_abort", 24, tail=12),  # condemns c1b
            _seg_end(0, "codec_eos", 12),
            EngineResult(type=ResultType.SESSION_DONE, session_id="s1"),
        ]
        delivered, _ = _run_consume(session, results)
        assert delivered == [c0a * (FRAME_BYTES * 12), c1a * (FRAME_BYTES * 12)]


class TestPrefillDoneDedup:
    def test_rerun_prefill_done_emitted_once_per_segment(self):
        session = _guarded_session()
        session.config.output_policy.config = {"delivery": "firehose"}
        prefill = EngineResult(
            type=ResultType.PREFILL_DONE, session_id="s1", segment_idx=0
        )
        results = [
            prefill,
            prefill,  # rerun re-prefills the same segment
            _seg_end(0, "codec_eos", 12),
            EngineResult(type=ResultType.SESSION_DONE, session_id="s1"),
        ]
        events = []
        _run_consume(session, results, events=events)
        assert [e["type"] for e in events if e["type"] == "prefill_done"] == [
            "prefill_done"
        ]
