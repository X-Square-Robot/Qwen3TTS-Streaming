"""In-process state for resumable WebSocket TTS streams.

The classes in this module deliberately know nothing about ``aiohttp``.  A
logical synthesis owns one :class:`ResumableSession`; websocket handlers only
attach to it for as long as their physical connection is alive.  This keeps an
engine session running during a short transport outage and fences a superseded
handler when a client reconnects.

Resume tokens are capability secrets.  They are registry keys, but are never
included in frames, errors, or log messages.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .session_identity import GatewaySessionIdentity


class ResumeProtocolError(ValueError):
    """A stable, machine-readable resumable-stream protocol error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResumeDelivery:
    """One exactly replayable logical server delivery.

    Audio uses one sequence number for the JSON ``audio_header`` and the raw
    binary frame that immediately follows it.
    """

    delivery_seq: int
    frame: dict[str, Any]
    start_sample: int
    end_sample: int
    retained_bytes: int


@dataclass(frozen=True, slots=True)
class ResumeFence:
    """Wake a handler whose attachment has been superseded."""


@dataclass(frozen=True, slots=True)
class ResumeFailure:
    """Wake an attached handler when replay can no longer be guaranteed."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ResumeAttachment:
    generation: int
    queue: asyncio.Queue[ResumeDelivery | ResumeFence | ResumeFailure]


def start_fingerprint(message: dict[str, Any]) -> str:
    """Return a deterministic fingerprint without retaining the resume token."""

    payload = {key: value for key, value in message.items() if key != "resume"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ResumableSession:
    """Mutable logical-stream state shared by successive websocket handlers."""

    def __init__(
        self,
        registry: ResumableSessionRegistry,
        *,
        token: str,
        identity: GatewaySessionIdentity,
        start_request: Any,
        config_fingerprint: str,
        max_buffer_bytes: int,
    ) -> None:
        self._registry = registry
        self.token = token
        self.identity = identity
        self.start_request = start_request
        self.config_fingerprint = config_fingerprint
        self.max_buffer_bytes = max_buffer_bytes

        self._lock = asyncio.Lock()
        self._input_lock = asyncio.Lock()
        self._records: deque[ResumeDelivery] = deque()
        self._retained_bytes = 0
        self._trimmed_through_seq = 0
        self._trimmed_audio_sample = 0
        self._next_delivery_seq = 1
        self._next_audio_sample = 0

        self._generation = 0
        self._attachment: ResumeAttachment | None = None
        self._grace_task: asyncio.Task | None = None
        self._terminal_task: asyncio.Task | None = None

        self._text_hashes: dict[int, bytes] = {}
        self.acked_text_seq = 0
        self.input_closed = False
        self.final_text_seq: int | None = None
        self.played_through_sample = 0
        self.buffered_through_sample = 0

        self.terminal = False
        self.engine_finished = False
        self.engine_cancelled = False
        self.failure_code: str | None = None
        self.failure_message: str = ""
        self.expiring = False
        self.closed = False

        self.ready = asyncio.Event()
        self.initialization_error: BaseException | None = None

    @property
    def client_session_id(self) -> str:
        return self.identity.client_session_id

    @property
    def internal_session_id(self) -> str:
        return self.identity.internal_session_id

    async def mark_initialized(self) -> None:
        self.ready.set()

    async def mark_initialization_failed(self, exc: BaseException) -> None:
        self.initialization_error = exc
        self.ready.set()

    async def wait_until_ready(self) -> None:
        await self.ready.wait()
        if self.initialization_error is not None:
            raise ResumeProtocolError(
                "resume_session_initialization_failed",
                "resumable stream could not be initialized",
            ) from self.initialization_error

    async def put(self, frame: dict[str, Any]) -> None:
        """Publish one output frame."""
        await self.put_batch([frame])

    async def put_batch(self, frames: list[dict[str, Any]]) -> None:
        """Register a batch atomically before waking the attachment.

        Audio and its progress event are still separate wire deliveries, but
        both records enter the replay ledger under one lock.  A disconnect
        between their physical sends therefore replays the missing sibling
        instead of losing the text cursor.
        """

        schedule_overflow = False
        schedule_terminal = False
        attachment: ResumeAttachment | None = None
        failure: ResumeFailure | None = None

        async with self._lock:
            if self.closed or self.failure_code is not None or self.terminal:
                return
            staged: list[ResumeDelivery] = []
            staged_bytes = 0
            staged_sample = self._next_audio_sample
            for frame in frames:
                seq = self._next_delivery_seq + len(staged)
                copied = _copy_frame(frame)
                start_sample = staged_sample
                end_sample = start_sample
                if copied.get("type") == "audio":
                    audio = copied["audio"]
                    end_sample += _audio_sample_count(audio)
                else:
                    copied["delivery_seq"] = seq

                retained_bytes = _retained_size(copied)
                staged.append(
                    ResumeDelivery(
                        delivery_seq=seq,
                        frame=copied,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        retained_bytes=retained_bytes,
                    )
                )
                staged_bytes += retained_bytes
                staged_sample = end_sample
                if _is_terminal(copied):
                    break

            if self._retained_bytes + staged_bytes > self.max_buffer_bytes:
                self.failure_code = "resume_buffer_exceeded"
                self.failure_message = (
                    "resumable stream output exceeded the server replay window"
                )
                failure = ResumeFailure(self.failure_code, self.failure_message)
                attachment = self._attachment
                schedule_overflow = True
            else:
                self._records.extend(staged)
                self._retained_bytes += staged_bytes
                self._next_delivery_seq += len(staged)
                self._next_audio_sample = staged_sample
                attachment = self._attachment
                for delivery in staged:
                    if _is_terminal(delivery.frame):
                        self.terminal = True
                        self.engine_finished = True
                        schedule_terminal = True
                    if attachment is not None:
                        attachment.queue.put_nowait(delivery)

        if failure is not None and attachment is not None:
            attachment.queue.put_nowait(failure)
        if schedule_overflow:
            asyncio.create_task(self._registry.fail_overflow(self))
        if schedule_terminal:
            self._registry.schedule_terminal_expiry(self)

    async def attach(
        self,
        *,
        last_delivery_seq: int,
        audio_through_sample: int,
    ) -> ResumeAttachment:
        async with self._lock:
            if self.closed or self.expiring:
                raise ResumeProtocolError(
                    "resume_session_not_found",
                    "resumable stream is no longer available",
                )
            if self.failure_code is not None:
                raise ResumeProtocolError(self.failure_code, self.failure_message)

            self._validate_cursor_locked(last_delivery_seq, audio_through_sample)

            previous = self._attachment
            if previous is not None:
                previous.queue.put_nowait(ResumeFence())

            self._generation += 1
            attachment = ResumeAttachment(self._generation, asyncio.Queue())
            self._attachment = attachment
            for delivery in self._records:
                if delivery.delivery_seq > last_delivery_seq:
                    attachment.queue.put_nowait(delivery)

            self._cancel_task_locked("_grace_task")
            return attachment

    async def detach(self, generation: int) -> bool:
        """Detach if ``generation`` still owns the stream.

        Returns false for a fenced handler; that is the key invariant that
        prevents an old handler's ``finally`` block from cancelling a newly
        attached stream.
        """

        async with self._lock:
            if self.closed or self._attachment is None:
                return False
            if self._attachment.generation != generation:
                return False
            self._attachment = None
            self._cancel_task_locked("_grace_task")
            # A terminal already has its own retention TTL. Starting a second
            # detached timer would neither extend safety nor improve cleanup.
            if not self.terminal:
                self._grace_task = asyncio.create_task(
                    self._registry.expire_detached(self, generation)
                )
            return True

    async def is_current_generation(self, generation: int) -> bool:
        async with self._lock:
            return (
                not self.closed
                and self._attachment is not None
                and self._attachment.generation == generation
            )

    async def accept_text(
        self,
        generation: int,
        *,
        seq_no: int,
        text: str,
        push: Callable[[], Awaitable[None]],
    ) -> tuple[int, bool]:
        """Push one text chunk exactly once and return (acked seq, duplicate)."""

        if seq_no <= 0:
            raise ResumeProtocolError(
                "invalid_text_sequence", "resumable text seq_no must start at 1"
            )
        digest = hashlib.sha256(text.encode("utf-8")).digest()

        async with self._input_lock:
            async with self._lock:
                self._require_generation_locked(generation)
                if seq_no <= self.acked_text_seq:
                    if self._text_hashes.get(seq_no) != digest:
                        raise ResumeProtocolError(
                            "text_sequence_conflict",
                            "text seq_no was already accepted with different content",
                        )
                    return self.acked_text_seq, True
                if seq_no != self.acked_text_seq + 1:
                    raise ResumeProtocolError(
                        "text_sequence_gap",
                        f"expected text seq_no {self.acked_text_seq + 1}",
                    )
                if self.input_closed:
                    raise ResumeProtocolError(
                        "input_already_closed", "cannot send text after stop"
                    )

            # Do not hold the state lock: engine callbacks publish output here.
            await push()

            async with self._lock:
                self._text_hashes[seq_no] = digest
                self.acked_text_seq = seq_no
                return self.acked_text_seq, False

    async def close_input(
        self,
        generation: int,
        *,
        final_seq_no: int,
        close: Callable[[], Awaitable[None]],
    ) -> tuple[int, bool]:
        """Mark input complete once and return (final seq, duplicate)."""

        async with self._input_lock:
            async with self._lock:
                self._require_generation_locked(generation)
                if final_seq_no != self.acked_text_seq:
                    raise ResumeProtocolError(
                        "final_text_sequence_mismatch",
                        f"stop final_seq_no must equal {self.acked_text_seq}",
                    )
                if self.input_closed:
                    if self.final_text_seq != final_seq_no:
                        raise ResumeProtocolError(
                            "final_text_sequence_conflict",
                            "stop conflicts with the already accepted final_seq_no",
                        )
                    return final_seq_no, True
                self.input_closed = True
                self.final_text_seq = final_seq_no

            try:
                await close()
            except BaseException:
                async with self._lock:
                    if not self.terminal:
                        self.input_closed = False
                        self.final_text_seq = None
                raise
            return final_seq_no, False

    async def acknowledge(
        self,
        generation: int,
        *,
        through_delivery_seq: int,
        audio_through_sample: int,
    ) -> None:
        async with self._lock:
            self._require_generation_locked(generation)
            if through_delivery_seq < self._trimmed_through_seq:
                # A delayed cumulative ACK is harmless.
                return
            self._validate_cursor_locked(through_delivery_seq, audio_through_sample)
            while (
                self._records and self._records[0].delivery_seq <= through_delivery_seq
            ):
                delivery = self._records.popleft()
                self._retained_bytes -= delivery.retained_bytes
            self._trimmed_through_seq = through_delivery_seq
            self._trimmed_audio_sample = audio_through_sample

    async def record_playback_progress(
        self,
        generation: int,
        *,
        played_through_sample: int,
        buffered_through_sample: int,
        observed_delivery_seq: int | None = None,
    ) -> None:
        """Record untrusted playback telemetry without affecting delivery."""
        async with self._lock:
            self._require_generation_locked(generation)
            if played_through_sample < 0 or buffered_through_sample < 0:
                raise ResumeProtocolError(
                    "invalid_playback_progress",
                    "playback samples must be non-negative",
                )
            if buffered_through_sample < played_through_sample:
                raise ResumeProtocolError(
                    "invalid_playback_progress",
                    "buffered_through_sample must be >= played_through_sample",
                )
            completely_old = (
                played_through_sample <= self.played_through_sample
                and buffered_through_sample <= self.buffered_through_sample
            )
            # Feedback can be replayed after reconnect.  Once both cursors are
            # at or behind the accepted state it is idempotent, even if the
            # old delivery sequence has already been trimmed from the ledger.
            if completely_old:
                return
            if observed_delivery_seq is None:
                # Keep older SDKs source-compatible; new SDKs always send the
                # cumulative delivery sequence so RB can be checked against
                # the exact replay ledger rather than a byte counter.
                observed_delivery_seq = self._next_delivery_seq - 1
            if observed_delivery_seq < 0:
                raise ResumeProtocolError(
                    "invalid_playback_progress",
                    "observed_delivery_seq must be non-negative",
                )
            last_delivery_seq = self._next_delivery_seq - 1
            if observed_delivery_seq > last_delivery_seq:
                raise ResumeProtocolError(
                    "unknown_playback_delivery",
                    "observed_delivery_seq is ahead of the replay ledger",
                )
            delivery_sample_limit = self._audio_end_through_delivery_locked(
                observed_delivery_seq
            )
            if buffered_through_sample > delivery_sample_limit:
                raise ResumeProtocolError(
                    "invalid_playback_progress",
                    "buffered_through_sample is ahead of server output/observed delivery",
                )
            if (
                played_through_sample < self.played_through_sample
                or buffered_through_sample < self.buffered_through_sample
            ):
                raise ResumeProtocolError(
                    "invalid_playback_progress",
                    "playback progress cannot partially move backwards",
                )
            self.played_through_sample = played_through_sample
            self.buffered_through_sample = buffered_through_sample

    def _audio_end_through_delivery_locked(self, delivery_seq: int) -> int:
        if delivery_seq == self._trimmed_through_seq:
            return self._trimmed_audio_sample
        if delivery_seq < self._trimmed_through_seq:
            raise ResumeProtocolError(
                "unknown_playback_delivery",
                "observed_delivery_seq is no longer in the replay ledger",
            )
        if delivery_seq >= self._next_delivery_seq - 1:
            return self._next_audio_sample
        for delivery in self._records:
            if delivery.delivery_seq == delivery_seq:
                return delivery.end_sample
        raise ResumeProtocolError(
            "unknown_playback_delivery",
            "observed_delivery_seq is no longer in the replay ledger",
        )

    async def resume_info(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "type": "resumed",
                "session_id": self.client_session_id,
                "acked_text_seq": self.acked_text_seq,
                "input_closed": self.input_closed,
                "final_seq_no": self.final_text_seq,
                "last_delivery_seq": self._next_delivery_seq - 1,
            }

    async def cancel_engine_once(self) -> bool:
        async with self._lock:
            if self.engine_finished or self.engine_cancelled:
                return False
            self.engine_cancelled = True
        await self._registry.engine.cancel(self.internal_session_id)
        return True

    async def publish_cancelled(self, reason: str) -> None:
        async with self._lock:
            if self.terminal or self.closed:
                return
            self.input_closed = True
            if self.final_text_seq is None:
                self.final_text_seq = self.acked_text_seq
        # Local import avoids coupling this transport-state module to the
        # gateway's frame constructors.
        frame = {
            "type": "event",
            "event": {
                "type": "done",
                "session_id": self.client_session_id,
                "segment_id": -1,
                "text": "",
                "message": reason,
                "meta": {
                    "terminal_reason": "cancelled",
                    "cancel_reason": reason,
                    "websocket_connection_reusable": "true",
                },
            },
        }
        await self.cancel_engine_once()
        await self.put(frame)

    async def mark_closed(self, *, fence: bool) -> None:
        async with self._lock:
            if self.closed:
                return
            self.closed = True
            self._cancel_task_locked("_grace_task")
            self._cancel_task_locked("_terminal_task")
            if fence and self._attachment is not None:
                self._attachment.queue.put_nowait(ResumeFence())
            self._attachment = None
            self._records.clear()
            self._retained_bytes = 0

    async def claim_detached_expiration(self, generation: int) -> bool:
        async with self._lock:
            if (
                self.closed
                or self.expiring
                or self._attachment is not None
                or self._generation != generation
            ):
                return False
            # Fence a reconnect atomically before engine cancellation starts.
            # Otherwise an attach at the grace boundary could succeed between
            # the expiry check and ``cancel_engine_once``.
            self.expiring = True
            return True

    async def claim_terminal_expiration(self) -> bool:
        async with self._lock:
            if self.closed or self.expiring or not self.terminal:
                return False
            self.expiring = True
            return True

    def _validate_cursor_locked(self, seq: int, audio_sample: int) -> None:
        if seq < 0 or audio_sample < 0:
            raise ResumeProtocolError(
                "invalid_resume_cursor", "resume cursor values must be non-negative"
            )
        highest = self._next_delivery_seq - 1
        if seq > highest:
            raise ResumeProtocolError(
                "invalid_resume_cursor", "resume cursor is ahead of server output"
            )
        if seq < self._trimmed_through_seq:
            raise ResumeProtocolError(
                "resume_window_exceeded",
                "requested output is older than the retained replay window",
            )
        if seq == self._trimmed_through_seq:
            expected_sample = self._trimmed_audio_sample
        else:
            expected_sample = None
            for delivery in self._records:
                if delivery.delivery_seq == seq:
                    expected_sample = delivery.end_sample
                    break
            if expected_sample is None:
                raise ResumeProtocolError(
                    "resume_window_exceeded",
                    "requested output is no longer retained",
                )
        if audio_sample != expected_sample:
            raise ResumeProtocolError(
                "resume_audio_cursor_mismatch",
                f"audio_through_sample must equal {expected_sample} at this delivery cursor",
            )

    def _require_generation_locked(self, generation: int) -> None:
        if (
            self.closed
            or self._attachment is None
            or self._attachment.generation != generation
        ):
            raise ResumeProtocolError(
                "resume_attachment_superseded",
                "this websocket no longer owns the resumable stream",
            )

    def _cancel_task_locked(self, name: str) -> None:
        task = getattr(self, name)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        setattr(self, name, None)


class ResumableSessionRegistry:
    """Process-local token registry and lifecycle owner."""

    def __init__(
        self,
        engine: Any,
        *,
        grace_seconds: float,
        max_buffer_bytes: int,
    ) -> None:
        self.engine = engine
        self.grace_seconds = max(0.0, float(grace_seconds))
        self.max_buffer_bytes = max(1, int(max_buffer_bytes))
        self._sessions: dict[str, ResumableSession] = {}
        self._lock = asyncio.Lock()

    async def claim_start(
        self,
        *,
        token: str,
        identity: GatewaySessionIdentity,
        start_request: Any,
        config_fingerprint: str,
    ) -> tuple[ResumableSession, bool]:
        async with self._lock:
            current = self._sessions.get(token)
            if current is not None:
                if current.config_fingerprint != config_fingerprint:
                    raise ResumeProtocolError(
                        "resume_token_conflict",
                        "resume token is already bound to a different start request",
                    )
                return current, False
            session = ResumableSession(
                self,
                token=token,
                identity=identity,
                start_request=start_request,
                config_fingerprint=config_fingerprint,
                max_buffer_bytes=self.max_buffer_bytes,
            )
            self._sessions[token] = session
            return session, True

    async def find(self, token: str) -> ResumableSession:
        async with self._lock:
            session = self._sessions.get(token)
        if session is None:
            raise ResumeProtocolError(
                "resume_session_not_found", "resumable stream is no longer available"
            )
        await session.wait_until_ready()
        return session

    async def remove(self, session: ResumableSession, *, fence: bool) -> bool:
        async with self._lock:
            if self._sessions.get(session.token) is not session:
                return False
            del self._sessions[session.token]
        await session.mark_closed(fence=fence)
        return True

    async def fail_initialization(
        self, session: ResumableSession, exc: BaseException
    ) -> None:
        await session.mark_initialization_failed(exc)
        # Registry cleanup is mandatory even when a partially initialized
        # engine rejects cancellation. Otherwise the capability token remains
        # pinned forever and every idempotent retry finds a poisoned record.
        try:
            await session.cancel_engine_once()
        finally:
            await self.remove(session, fence=True)

    async def fail_overflow(self, session: ResumableSession) -> None:
        await session.cancel_engine_once()

    async def expire_detached(self, session: ResumableSession, generation: int) -> None:
        try:
            await asyncio.sleep(self.grace_seconds)
            if not await session.claim_detached_expiration(generation):
                return
            try:
                await session.cancel_engine_once()
            finally:
                await self.remove(session, fence=False)
        except asyncio.CancelledError:
            return

    def schedule_terminal_expiry(self, session: ResumableSession) -> None:
        async def expire() -> None:
            try:
                await asyncio.sleep(self.grace_seconds)
                if not await session.claim_terminal_expiration():
                    return
                await self.remove(session, fence=True)
            except asyncio.CancelledError:
                return

        old = session._terminal_task
        if old is not None and not old.done():
            old.cancel()
        session._terminal_task = asyncio.create_task(expire())

    async def terminal_ack(
        self,
        session: ResumableSession,
        generation: int,
        *,
        through_delivery_seq: int,
        audio_through_sample: int,
    ) -> None:
        if not session.terminal:
            raise ResumeProtocolError(
                "terminal_not_emitted", "terminal_ack received before a terminal event"
            )
        await session.acknowledge(
            generation,
            through_delivery_seq=through_delivery_seq,
            audio_through_sample=audio_through_sample,
        )
        await self.remove(session, fence=False)

    async def close(self) -> None:
        """Cancel every active execution during gateway shutdown."""

        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        first_error: BaseException | None = None
        for session in sessions:
            try:
                await session.cancel_engine_once()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                await session.mark_closed(fence=True)
        if first_error is not None:
            raise first_error


def _copy_frame(frame: dict[str, Any]) -> dict[str, Any]:
    copied = dict(frame)
    if copied.get("type") == "audio":
        audio = dict(copied.get("audio") or {})
        audio["meta"] = dict(audio.get("meta") or {})
        audio["pcm_data"] = bytes(audio.get("pcm_data") or b"")
        copied["audio"] = audio
    elif copied.get("type") == "event":
        event = dict(copied.get("event") or {})
        event["meta"] = dict(event.get("meta") or {})
        if isinstance(event.get("audio"), dict):
            event["audio"] = dict(event["audio"])
        copied["event"] = event
    return copied


def _audio_sample_count(audio: dict[str, Any]) -> int:
    encoding = str(audio.get("encoding", "pcm_f32") or "pcm_f32").lower()
    bytes_per_sample = 2 if encoding in {"pcm_s16", "pcm_s16le", "s16le"} else 4
    channels = max(1, int(audio.get("channels", 1) or 1))
    return len(audio.get("pcm_data") or b"") // (bytes_per_sample * channels)


def _retained_size(frame: dict[str, Any]) -> int:
    if frame.get("type") == "audio":
        audio = frame["audio"]
        metadata = {key: value for key, value in audio.items() if key != "pcm_data"}
        return len(audio.get("pcm_data") or b"") + len(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    return len(
        json.dumps(frame, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _is_terminal(frame: dict[str, Any]) -> bool:
    return frame.get("type") == "event" and frame.get("event", {}).get("type") in {
        "done",
        "error",
    }
