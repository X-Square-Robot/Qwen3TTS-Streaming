"""Realtime audio stream adapter — converts non-isochronous AudioChunk flow
into an isochronous (wall-clock aligned) audio stream with optional silence
filling.

Typical consumers: WebRTC media tracks, local audio players, benchmark
harnesses that need to simulate real-time playback rhythm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Iterator

from qwen3tts_protocol import AudioChunk, StreamEvent

from ._session import BaseStreamSession


@dataclass
class TimedAudio:
    """An audio frame with timing metadata."""

    data: bytes
    """Raw PCM audio bytes."""

    duration_s: float
    """Duration of this frame in seconds."""

    is_silence: bool = False
    """True if this frame is a silence fill inserted to cover a gap."""


def _make_silence(duration_s: float, sample_rate: int) -> TimedAudio:
    """Generate a silence frame of the given duration.

    Uses float32 PCM (4 bytes per sample) to match the engine's native
    output encoding.
    """
    num_samples = int(sample_rate * duration_s)
    # float32 zero bytes — matches pcm_f32 encoding
    silence_bytes = b"\x00" * (num_samples * 4)
    return TimedAudio(data=silence_bytes, duration_s=duration_s, is_silence=True)


def _audio_duration_s(pcm_bytes: bytes, sample_rate: int) -> float:
    """Compute the duration of a pcm_f32 byte buffer at *sample_rate*."""
    num_samples = len(pcm_bytes) // 4  # float32 = 4 bytes per sample
    return num_samples / sample_rate


class RealtimeAudioStream:
    """Convert a non-isochronous ``AudioChunk`` stream into an isochronous
    (wall-clock aligned) audio stream.

    The engine pushes audio frames as fast as they are generated — the
    inter-frame timing is irregular (e.g. ~800 ms for the first chunk, then
    ~2 s per subsequent chunk).  Consumers that need a steady rhythm (WebRTC
    media tracks, local playback, benchmarks) can wrap a
    :class:`BaseStreamSession` with this adapter to receive frames at
    real-time pace, with silence automatically inserted to cover gaps.

    Usage::

        session = client.open_stream(...)
        for frame in RealtimeAudioStream(session):
            webrtc_track.write(frame.data)

    Args:
        session: The streaming session to consume.
        fill_silence: When *True* (default), silence frames are inserted
            whenever the engine is late producing the next audio chunk.
            When *False*, frames are yielded as-is without timing alignment.
        chunk_s: Timeout / minimum granularity for silence fills, in
            seconds.  Default 0.02 (20 ms) aligns with the WebRTC Opus
            frame size.
        sample_rate: Audio sample rate in Hz.  Default 24000 matches the
            engine's native output.
    """

    def __init__(
        self,
        session: BaseStreamSession,
        *,
        fill_silence: bool = True,
        chunk_s: float = 0.02,
        sample_rate: int = 24000,
    ) -> None:
        self._session = session
        self._fill_silence = fill_silence
        self._chunk_s = chunk_s
        self._sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[TimedAudio]:
        if self._fill_silence:
            yield from self._iter_paced()
        else:
            yield from self._iter_passthrough()

    # ------------------------------------------------------------------
    # Passthrough mode (fill_silence=False)
    # ------------------------------------------------------------------

    def _iter_passthrough(self) -> Iterator[TimedAudio]:
        """Yield frames as they arrive, no timing alignment."""
        for message in self._session.iter_messages():
            if isinstance(message, AudioChunk):
                duration = _audio_duration_s(message.pcm_bytes, self._sample_rate)
                yield TimedAudio(data=message.pcm_bytes, duration_s=duration)

    # ------------------------------------------------------------------
    # Paced mode (fill_silence=True)
    # ------------------------------------------------------------------

    def _iter_paced(self) -> Iterator[TimedAudio]:
        """Yield frames at real-time pace, inserting silence for gaps.

        The algorithm maintains a virtual *play_clock* that tracks how much
        audio time has been "played".  When the wall-clock advances faster
        than the play clock (i.e. the engine is late), silence is inserted
        to keep the output isochronous.
        """
        # Bridge queue: the session iterator runs in the calling thread,
        # and we need a timeout-capable get().  A small bounded queue is
        # sufficient because the consumer (this loop) drains it promptly.
        bridge: Queue[AudioChunk | None] = Queue(maxsize=64)

        # Feed the session's AudioChunks into the bridge queue on a
        # helper thread so we can use Queue.get(timeout=) for silence
        # fill timing.
        import threading

        def _feeder():
            for message in self._session.iter_messages():
                if isinstance(message, AudioChunk):
                    bridge.put(message)
            bridge.put(None)  # sentinel

        feeder_thread = threading.Thread(target=_feeder, daemon=True)
        feeder_thread.start()

        play_clock: float | None = None

        try:
            while True:
                wall_now = time.monotonic()

                try:
                    audio = bridge.get(timeout=self._chunk_s)
                except Empty:
                    # No new frame arrived within chunk_s → fill silence
                    elapsed = time.monotonic() - wall_now
                    yield _make_silence(elapsed, self._sample_rate)
                    if play_clock is not None:
                        play_clock += elapsed
                    continue

                # Sentinel — stream ended
                if audio is None:
                    break

                duration = _audio_duration_s(audio.pcm_bytes, self._sample_rate)

                # First frame: initialise play clock, output immediately
                if play_clock is None:
                    yield TimedAudio(data=audio.pcm_bytes, duration_s=duration)
                    play_clock = duration
                    # Simulate first-frame playback time
                    time.sleep(duration)
                    continue

                # Compute wall-clock elapsed since last iteration start
                elapsed = time.monotonic() - wall_now
                gap = elapsed - play_clock

                # If the engine was late, fill the gap with silence
                if gap > 0:
                    yield _make_silence(gap, self._sample_rate)
                    play_clock += gap

                # Output the actual audio frame
                yield TimedAudio(data=audio.pcm_bytes, duration_s=duration)

                # If the frame is longer than remaining wall-clock time,
                # sleep to simulate playback; otherwise just advance the
                # play clock.
                remaining = duration - (time.monotonic() - wall_now - gap)
                if remaining > 0:
                    time.sleep(remaining)
                    play_clock = duration
                else:
                    play_clock += duration
        finally:
            # Ensure the feeder thread is not left hanging if the caller
            # breaks out of the iterator early.
            if feeder_thread.is_alive():
                # Drain remaining items so the feeder can exit cleanly
                while not bridge.empty():
                    try:
                        bridge.get_nowait()
                    except Empty:
                        break
