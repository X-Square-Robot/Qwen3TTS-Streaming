from qwen3tts import (
    AudioChunk,
    AudioFormat,
    PlaybackProgressTracker,
    StreamEvent,
    TextProgressAnchor,
)
from qwen3tts.exceptions import ProtocolError
import pytest


def _anchor(seq, start, end, raw_end, normalized_end):
    return TextProgressAnchor(
        anchor_seq=seq,
        output_sample_start=start,
        output_sample_end=end,
        output_sample_rate=24000,
        raw_codepoint_start=0,
        raw_codepoint_end=raw_end,
        normalized_codepoint_start=0,
        normalized_codepoint_end=normalized_end,
    )


def test_tracker_keeps_confirmed_and_interpolated_cursors_separate():
    tracker = PlaybackProgressTracker(
        [_anchor(1, 0, 100, 5, 5), _anchor(2, 100, 200, 10, 10)]
    )
    state = tracker.update_playback_progress(
        played_through_sample=150,
        buffered_through_sample=180,
    )
    assert state.confirmed.raw_codepoint == 5
    assert state.estimated.raw_codepoint == 8
    assert state.buffered_through_sample == 180


def test_tracker_interpolates_inside_first_anchor_without_future_extrapolation():
    tracker = PlaybackProgressTracker([_anchor(1, 0, 100, 10, 10)])
    state = tracker.update_playback_progress(played_through_sample=50)
    assert state.confirmed.available is False
    assert state.estimated.raw_codepoint == 5
    assert state.estimated.normalized_codepoint == 5


def test_tracker_does_not_extrapolate_past_latest_anchor_and_completes_at_terminal():
    tracker = PlaybackProgressTracker([_anchor(1, 0, 100, 5, 5)])
    state = tracker.update_playback_progress(played_through_sample=100)
    assert state.confirmed.raw_codepoint == 5
    assert state.estimated.raw_codepoint == 5
    tracker.observe(
        StreamEvent(type="done", meta={"final_output_sample": "100"})
    )
    state = tracker.update_playback_progress(played_through_sample=100)
    assert state.playback_complete is True


def test_tracker_reads_anchor_from_audio_meta():
    tracker = PlaybackProgressTracker()
    tracker.observe(
        AudioChunk(
            pcm_bytes=b"\0" * 8,
            audio=AudioFormat(encoding="pcm_f32", sample_rate=24000, channels=1),
            meta={
                "anchor_seq": "1",
                "output_sample_start": "0",
                "output_sample_end": "2",
                "output_sample_rate": "24000",
                "raw_codepoint_start": "0",
                "raw_codepoint_end": "2",
                "normalized_codepoint_start": "0",
                "normalized_codepoint_end": "2",
            },
        )
    )
    assert tracker.latest.available is True


def test_tracker_rejects_conflicting_anchor_replay():
    tracker = PlaybackProgressTracker([_anchor(1, 0, 10, 1, 1)])
    try:
        tracker.add_anchor(_anchor(1, 0, 11, 1, 1))
    except ProtocolError:
        pass
    else:  # pragma: no cover - assertion documents the wire contract
        raise AssertionError("conflicting replay must be rejected")


def test_tracker_accepts_exact_replay_after_newer_anchor():
    tracker = PlaybackProgressTracker([_anchor(1, 0, 10, 1, 1), _anchor(2, 10, 20, 2, 2)])
    tracker.add_anchor(_anchor(1, 0, 10, 1, 1))
    assert [anchor.anchor_seq for anchor in tracker.anchors] == [1, 2]


def test_tracker_rejects_partial_anchor_meta_instead_of_defaulting_spans():
    tracker = PlaybackProgressTracker()
    with pytest.raises(ProtocolError, match="malformed text progress anchor"):
        tracker.observe(
            AudioChunk(
                pcm_bytes=b"\0" * 8,
                audio=AudioFormat(encoding="pcm_f32", sample_rate=24000, channels=1),
                meta={
                    "anchor_seq": "1",
                    "output_sample_start": "0",
                    "output_sample_end": "2",
                    "output_sample_rate": "24000",
                },
            )
        )


def test_tracker_playback_update_is_transactional():
    tracker = PlaybackProgressTracker([_anchor(1, 0, 100, 5, 5)])
    with pytest.raises(ProtocolError, match="buffered sample"):
        tracker.update_playback_progress(
            played_through_sample=50,
            buffered_through_sample=40,
        )
    state = tracker.latest
    assert state.played_through_sample == 0
    assert state.buffered_through_sample == 0
