"""Realtime audio stream adapter — converts non-isochronous AudioChunk flow
into an isochronous (wall-clock aligned) audio stream with optional silence
filling.

Typical consumers: WebRTC media tracks, local audio players, benchmark
harnesses that need to simulate real-time playback rhythm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Iterator

from qwen3tts_protocol import AudioChunk

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

    output_sample_start: int | None = None
    output_sample_end: int | None = None
    playout_sample_start: int | None = None
    playout_sample_end: int | None = None


def _make_silence(
    duration_s: float,
    sample_rate: int,
    *,
    channels: int = 1,
    bytes_per_sample: int = 4,
    playout_sample_start: int | None = None,
    output_sample_start: int | None = None,
) -> TimedAudio:
    """Generate local playout silence using the active PCM layout.

    ``output_sample_start`` is accepted as a source-compatible deprecated
    alias for callers that used the old helper directly.  It is intentionally
    mapped onto the local playout axis; silence never advances server output
    sample coordinates.
    """
    if playout_sample_start is None:
        playout_sample_start = output_sample_start
    num_samples = int(sample_rate * duration_s)
    # float32 zero bytes — matches pcm_f32 encoding
    silence_bytes = b"\x00" * (num_samples * channels * bytes_per_sample)
    playout_end = (
        None
        if playout_sample_start is None
        else playout_sample_start + num_samples
    )
    return TimedAudio(
        data=silence_bytes,
        duration_s=duration_s,
        is_silence=True,
        output_sample_start=None,
        output_sample_end=None,
        playout_sample_start=playout_sample_start,
        playout_sample_end=playout_end,
    )


def _audio_duration_s(
    pcm_bytes: bytes,
    sample_rate: int,
    *,
    encoding: str = "pcm_f32",
    channels: int = 1,
) -> float:
    """Compute PCM duration using the declared encoding and channel count."""
    bytes_per_sample = _bytes_per_sample(encoding)
    num_samples = len(pcm_bytes) // (bytes_per_sample * max(1, channels))
    return num_samples / max(1, sample_rate)


def _bytes_per_sample(encoding: str) -> int:
    return {"pcm_f32": 4, "pcm_s16le": 2, "pcm_s16": 2, "pcm_u8": 1}.get(
        encoding.lower(), 4
    )


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
        playout_sample_cursor = 0
        for message in self._session.iter_messages():
            if isinstance(message, AudioChunk):
                duration = _audio_duration_s(
                    message.pcm_bytes,
                    message.audio.sample_rate or self._sample_rate,
                    encoding=message.audio.encoding,
                    channels=message.audio.channels,
                )
                yield TimedAudio(
                    data=message.pcm_bytes,
                    duration_s=duration,
                    output_sample_start=message.output_sample_start,
                    output_sample_end=message.output_sample_end,
                    playout_sample_start=playout_sample_cursor,
                    playout_sample_end=playout_sample_cursor
                    + int(round(duration * (message.audio.sample_rate or self._sample_rate))),
                )
                playout_sample_cursor += int(
                    round(duration * (message.audio.sample_rate or self._sample_rate))
                )

    # ------------------------------------------------------------------
    # Paced mode (fill_silence=True)
    # ------------------------------------------------------------------

    def _iter_paced(self) -> Iterator[TimedAudio]:
        """Yield frames at real-time pace, inserting silence for gaps.

        The clock is anchored at the first audio frame (``wall_start``).
        ``play_clock`` is the *cumulative* duration of everything emitted so
        far (audio + silence) — i.e. the virtual playhead. On every step we
        compare the playhead against the real elapsed time since the anchor:

        - playhead behind wall-clock (engine was late) → emit silence to catch
          up, keeping the output isochronous;
        - playhead ahead of wall-clock (engine produced faster than real time)
          → sleep until real time catches up, so we never dump audio faster
          than it plays.

        The previous implementation reset the wall anchor every iteration, so
        ``elapsed`` (per-iteration) and ``play_clock`` (cumulative) were
        dimensionally mismatched and no pacing/silence ever happened after the
        first frame.
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

        wall_start: float | None = None  # anchored at the first audio frame
        play_clock = 0.0  # cumulative emitted duration (playhead)
        output_sample_cursor = 0
        playout_sample_cursor = 0
        active_sample_rate = self._sample_rate
        active_channels = 1
        active_bytes_per_sample = 4

        try:
            while True:
                try:
                    audio = bridge.get(timeout=self._chunk_s)
                except Empty:
                    # No frame within chunk_s. Before the first frame there is
                    # nothing to pace against, so just keep waiting. After it,
                    # fill the gap between the playhead and the wall clock.
                    if wall_start is None:
                        continue
                    gap = (time.monotonic() - wall_start) - play_clock
                    if gap > 0:
                        silence = _make_silence(
                            gap,
                            active_sample_rate,
                            channels=active_channels,
                            bytes_per_sample=active_bytes_per_sample,
                            playout_sample_start=playout_sample_cursor,
                        )
                        playout_sample_cursor = (
                            silence.playout_sample_end or playout_sample_cursor
                        )
                        yield silence
                        play_clock += gap
                    continue

                # Sentinel — stream ended
                if audio is None:
                    break

                duration = _audio_duration_s(
                    audio.pcm_bytes,
                    audio.audio.sample_rate or self._sample_rate,
                    encoding=audio.audio.encoding,
                    channels=audio.audio.channels,
                )
                active_sample_rate = audio.audio.sample_rate or self._sample_rate
                active_channels = max(1, audio.audio.channels)
                active_bytes_per_sample = _bytes_per_sample(audio.audio.encoding)

                if wall_start is None:
                    # First frame: anchor the clock and emit immediately.
                    wall_start = time.monotonic()
                    start = (
                        audio.output_sample_start
                        if audio.output_sample_start is not None
                        else output_sample_cursor
                    )
                    end = (
                        audio.output_sample_end
                        if audio.output_sample_end is not None
                        else start + int(round(duration * active_sample_rate))
                    )
                    yield TimedAudio(
                        data=audio.pcm_bytes,
                        duration_s=duration,
                        output_sample_start=start,
                        output_sample_end=end,
                        playout_sample_start=playout_sample_cursor,
                        playout_sample_end=playout_sample_cursor
                        + int(round(duration * active_sample_rate)),
                    )
                    output_sample_cursor = max(output_sample_cursor, end)
                    playout_sample_cursor += int(round(duration * active_sample_rate))
                    play_clock = duration
                else:
                    # Fill any catch-up gap (engine was late), then emit.
                    gap = (time.monotonic() - wall_start) - play_clock
                    if gap > 0:
                        silence = _make_silence(
                            gap,
                            active_sample_rate,
                            channels=active_channels,
                            bytes_per_sample=active_bytes_per_sample,
                            playout_sample_start=playout_sample_cursor,
                        )
                        playout_sample_cursor = (
                            silence.playout_sample_end or playout_sample_cursor
                        )
                        yield silence
                        play_clock += gap
                    start = (
                        audio.output_sample_start
                        if audio.output_sample_start is not None
                        else output_sample_cursor
                    )
                    end = (
                        audio.output_sample_end
                        if audio.output_sample_end is not None
                        else start + int(round(duration * active_sample_rate))
                    )
                    yield TimedAudio(
                        data=audio.pcm_bytes,
                        duration_s=duration,
                        output_sample_start=start,
                        output_sample_end=end,
                        playout_sample_start=playout_sample_cursor,
                        playout_sample_end=playout_sample_cursor
                        + int(round(duration * active_sample_rate)),
                    )
                    output_sample_cursor = max(output_sample_cursor, end)
                    playout_sample_cursor += int(round(duration * active_sample_rate))
                    play_clock += duration

                # If the playhead is ahead of real time, sleep so the output
                # stays wall-clock aligned instead of draining at full speed.
                ahead = play_clock - (time.monotonic() - wall_start)
                if ahead > 0:
                    time.sleep(ahead)
        finally:
            # Ensure the feeder thread is not left hanging if the caller
            # breaks out of the iterator early.  A single drain pass is not
            # enough: with the consumer gone the engine keeps producing, the
            # bounded bridge fills up again and the feeder blocks forever on
            # ``put`` — it never reaches its terminal sentinel (thread leak).
            # Cancel the session so the stream terminates, then keep draining
            # until the feeder exits (bounded, in case cancel is a no-op
            # because ``end()`` was already sent and the engine keeps going).
            if feeder_thread.is_alive():
                cancel = getattr(self._session, "cancel", None)
                if cancel is not None:
                    try:
                        cancel(reason="realtime consumer stopped")
                    except Exception:
                        pass
                drain_deadline = time.monotonic() + 5.0
                while feeder_thread.is_alive() and time.monotonic() < drain_deadline:
                    try:
                        bridge.get(timeout=0.1)
                    except Empty:
                        pass
