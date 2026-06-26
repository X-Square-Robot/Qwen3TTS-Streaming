"""Tests for RealtimeAudioStream and TimedAudio."""

from __future__ import annotations

import struct
import time
from unittest.mock import patch

import pytest

from qwen3tts_protocol import AudioChunk, AudioFormat, StreamEvent
from qwen3tts._session import BaseStreamSession
from qwen3tts.realtime import (
    RealtimeAudioStream,
    TimedAudio,
    _audio_duration_s,
    _make_silence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000


def _pcm_f32_silence(duration_s: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate pcm_f32 silence bytes of the given duration."""
    num_samples = int(sample_rate * duration_s)
    return b"\x00" * (num_samples * 4)


def _pcm_f32_tone(duration_s: float, freq: float = 440.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate a pcm_f32 sine tone for distinguishable test data."""
    import math

    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        value = math.sin(2 * math.pi * freq * i / sample_rate) * 0.5
        samples.append(struct.pack("<f", value))
    return b"".join(samples)


def _make_audio_chunk(duration_s: float, sample_rate: int = SAMPLE_RATE) -> AudioChunk:
    """Create an AudioChunk with pcm_f32 data of the given duration."""
    return AudioChunk(
        pcm_bytes=_pcm_f32_tone(duration_s, sample_rate=sample_rate),
        audio=AudioFormat(encoding="pcm_f32", sample_rate=sample_rate),
    )


def _session_with_chunks(*chunks: AudioChunk | None) -> BaseStreamSession:
    """Create a BaseStreamSession pre-loaded with chunks.

    ``None`` entries are not put into the session (they represent gaps
    controlled by the caller via timing).  A ``StreamEvent(type="done")``
    is appended automatically.
    """
    session = BaseStreamSession(session_id="test", transport="test")
    for chunk in chunks:
        if chunk is not None:
            session._put_message(chunk)
    session._put_message(StreamEvent(type="done", session_id="test"))
    return session


# ---------------------------------------------------------------------------
# TimedAudio & helpers
# ---------------------------------------------------------------------------


class TestTimedAudio:
    def test_fields(self):
        ta = TimedAudio(data=b"\x00" * 8, duration_s=0.02, is_silence=True)
        assert ta.data == b"\x00" * 8
        assert ta.duration_s == 0.02
        assert ta.is_silence is True

    def test_default_not_silence(self):
        ta = TimedAudio(data=b"\x00" * 8, duration_s=0.02)
        assert ta.is_silence is False


class TestMakeSilence:
    def test_duration(self):
        frame = _make_silence(0.02, SAMPLE_RATE)
        assert frame.is_silence is True
        # 24000 * 0.02 = 480 samples * 4 bytes = 1920 bytes
        assert len(frame.data) == 480 * 4
        assert frame.duration_s == pytest.approx(0.02, abs=1e-6)

    def test_zero_duration(self):
        frame = _make_silence(0.0, SAMPLE_RATE)
        assert frame.data == b""
        assert frame.duration_s == 0.0


class TestAudioDurationS:
    def test_known_duration(self):
        # 480 samples at 24000 Hz = 0.02 s
        pcm = b"\x00" * (480 * 4)
        assert _audio_duration_s(pcm, SAMPLE_RATE) == pytest.approx(0.02, abs=1e-6)

    def test_empty(self):
        assert _audio_duration_s(b"", SAMPLE_RATE) == 0.0


# ---------------------------------------------------------------------------
# Passthrough mode (fill_silence=False)
# ---------------------------------------------------------------------------


class TestPassthroughMode:
    def test_yields_audio_chunks_as_timed_audio(self):
        chunk = _make_audio_chunk(0.02)
        session = _session_with_chunks(chunk)
        stream = RealtimeAudioStream(session, fill_silence=False, sample_rate=SAMPLE_RATE)

        frames = list(stream)
        # One audio frame + no silence fills
        audio_frames = [f for f in frames if not f.is_silence]
        assert len(audio_frames) == 1
        assert audio_frames[0].duration_s == pytest.approx(0.02, abs=1e-4)

    def test_no_silence_inserted(self):
        session = _session_with_chunks(_make_audio_chunk(0.02))
        stream = RealtimeAudioStream(session, fill_silence=False, sample_rate=SAMPLE_RATE)
        frames = list(stream)
        silence_frames = [f for f in frames if f.is_silence]
        assert len(silence_frames) == 0

    def test_ignores_stream_events(self):
        """StreamEvent messages (start, done, etc.) should be skipped."""
        session = BaseStreamSession(session_id="test", transport="test")
        session._put_message(StreamEvent(type="start", session_id="test"))
        session._put_message(_make_audio_chunk(0.02))
        session._put_message(StreamEvent(type="done", session_id="test"))

        stream = RealtimeAudioStream(session, fill_silence=False, sample_rate=SAMPLE_RATE)
        frames = list(stream)
        assert len(frames) == 1
        assert not frames[0].is_silence

    def test_empty_stream(self):
        session = _session_with_chunks()
        stream = RealtimeAudioStream(session, fill_silence=False, sample_rate=SAMPLE_RATE)
        frames = list(stream)
        assert len(frames) == 0


# ---------------------------------------------------------------------------
# Paced mode (fill_silence=True)
# ---------------------------------------------------------------------------


class TestPacedMode:
    def test_first_frame_output_immediately(self):
        """The first audio frame should be yielded without silence prefix."""
        chunk = _make_audio_chunk(0.02)
        session = _session_with_chunks(chunk)
        stream = RealtimeAudioStream(
            session, fill_silence=True, chunk_s=0.02, sample_rate=SAMPLE_RATE
        )

        # Use a very short sleep mock to avoid real-time waits
        with patch("time.sleep"):
            frames = list(stream)

        audio_frames = [f for f in frames if not f.is_silence]
        assert len(audio_frames) == 1
        # First frame should be the actual audio, not silence
        assert not frames[0].is_silence

    def test_silence_fill_on_gap(self):
        """When the engine is late, silence should be inserted."""
        # We simulate a gap by having the feeder thread sleep before
        # putting the second chunk.  With a small chunk_s timeout, the
        # paced loop will emit silence while waiting.
        session = BaseStreamSession(session_id="test", transport="test")
        chunk1 = _make_audio_chunk(0.02)
        chunk2 = _make_audio_chunk(0.02)

        import threading

        def _delayed_feed():
            session._put_message(chunk1)
            time.sleep(0.06)  # Introduce a gap longer than chunk_s
            session._put_message(chunk2)
            session._put_message(StreamEvent(type="done", session_id="test"))

        feeder = threading.Thread(target=_delayed_feed, daemon=True)
        feeder.start()

        stream = RealtimeAudioStream(
            session, fill_silence=True, chunk_s=0.02, sample_rate=SAMPLE_RATE
        )

        with patch("time.sleep"):
            frames = list(stream)

        silence_frames = [f for f in frames if f.is_silence]
        # There should be at least one silence frame during the gap
        assert len(silence_frames) >= 1

    def test_stream_terminates_on_done(self):
        """Stream should end when the session sends a 'done' event."""
        session = _session_with_chunks(_make_audio_chunk(0.02))
        stream = RealtimeAudioStream(
            session, fill_silence=True, chunk_s=0.02, sample_rate=SAMPLE_RATE
        )

        with patch("time.sleep"):
            frames = list(stream)

        # Should have terminated without hanging
        audio_frames = [f for f in frames if not f.is_silence]
        assert len(audio_frames) == 1

    def test_empty_stream_no_hang(self):
        """An empty stream (no audio chunks) should not hang."""
        session = _session_with_chunks()
        stream = RealtimeAudioStream(
            session, fill_silence=True, chunk_s=0.01, sample_rate=SAMPLE_RATE
        )

        with patch("time.sleep"):
            frames = list(stream)

        # May have silence frames from timeout, but should terminate
        # The done event causes the feeder to put None, ending the loop
        assert isinstance(frames, list)

    def test_multiple_chunks(self):
        """Multiple audio chunks should all be yielded."""
        chunks = [_make_audio_chunk(0.02) for _ in range(3)]
        session = _session_with_chunks(*chunks)
        stream = RealtimeAudioStream(
            session, fill_silence=True, chunk_s=0.02, sample_rate=SAMPLE_RATE
        )

        with patch("time.sleep"):
            frames = list(stream)

        audio_frames = [f for f in frames if not f.is_silence]
        assert len(audio_frames) == 3


# ---------------------------------------------------------------------------
# Early break / cleanup
# ---------------------------------------------------------------------------


class TestEarlyBreak:
    def test_breaking_out_of_iterator_does_not_hang(self):
        """Breaking out of the iterator early should not leave threads hanging."""
        # Feed many chunks slowly
        session = BaseStreamSession(session_id="test", transport="test")

        import threading

        def _slow_feed():
            for i in range(100):
                session._put_message(_make_audio_chunk(0.02))
                time.sleep(0.01)
            session._put_message(StreamEvent(type="done", session_id="test"))

        feeder = threading.Thread(target=_slow_feed, daemon=True)
        feeder.start()

        stream = RealtimeAudioStream(
            session, fill_silence=True, chunk_s=0.02, sample_rate=SAMPLE_RATE
        )

        with patch("time.sleep"):
            count = 0
            for frame in stream:
                count += 1
                if count >= 3:
                    break

        # Should have exited the loop without hanging
        assert count == 3


# ---------------------------------------------------------------------------
# Import from top-level package
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_import_from_package(self):
        from qwen3tts import RealtimeAudioStream as RAS
        from qwen3tts import TimedAudio as TA

        assert RAS is RealtimeAudioStream
        assert TA is TimedAudio

    def test_in_all(self):
        import qwen3tts

        assert "RealtimeAudioStream" in qwen3tts.__all__
        assert "TimedAudio" in qwen3tts.__all__
