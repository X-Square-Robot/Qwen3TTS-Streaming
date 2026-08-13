"""Protocol-neutral delivery ledger for resumable logical sessions.

The ledger stores typed canonical outputs.  Native WebSocket and Realtime
projectors may serialize the same record differently, but replay never stores
or replays protocol-specific JSON/binary frames.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

from .types import AudioOutput, SessionOutput, TerminalOutput


class LedgerError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ReliableDelivery:
    delivery_seq: int
    payload: SessionOutput
    start_sample: int
    end_sample: int
    retained_bytes: int


@dataclass(frozen=True, slots=True)
class AttachmentFence:
    generation: int


@dataclass(frozen=True, slots=True)
class AttachmentFailure:
    code: str
    message: str


LedgerMessage = ReliableDelivery | AttachmentFence | AttachmentFailure
T = TypeVar("T")


class DeliveryAttachment(Generic[T]):
    def __init__(self, generation: int, queue: asyncio.Queue[LedgerMessage]):
        self.generation = generation
        self.queue = queue


class DeliveryLedger:
    """Bounded, cumulative-ACK ledger for one logical execution."""

    def __init__(
        self,
        *,
        token: str,
        config_fingerprint: str,
        protocol: str,
        max_buffer_bytes: int = 16 * 1024 * 1024,
        attachment_queue_maxsize: int = 4096,
    ) -> None:
        if not token:
            raise ValueError("token must not be empty")
        if not config_fingerprint:
            raise ValueError("config_fingerprint must not be empty")
        if not protocol:
            raise ValueError("protocol must not be empty")
        if max_buffer_bytes <= 0 or attachment_queue_maxsize <= 0:
            raise ValueError("ledger limits must be positive")
        self.token = token
        self.config_fingerprint = config_fingerprint
        self.protocol = protocol
        self.max_buffer_bytes = max_buffer_bytes
        self.attachment_queue_maxsize = attachment_queue_maxsize
        self._lock = asyncio.Lock()
        self._records: deque[ReliableDelivery] = deque()
        self._retained_bytes = 0
        self._next_delivery_seq = 1
        self._next_audio_sample = 0
        self._trimmed_through_seq = 0
        self._trimmed_audio_sample = 0
        self._acked_delivery_seq = 0
        self._acked_audio_sample = 0
        self._generation = 0
        self._attachment: DeliveryAttachment[SessionOutput] | None = None
        self._terminal: TerminalOutput | None = None
        self._terminal_acked = False

    @property
    def terminal(self) -> TerminalOutput | None:
        return self._terminal

    @property
    def next_delivery_seq(self) -> int:
        return self._next_delivery_seq

    @property
    def next_audio_sample(self) -> int:
        return self._next_audio_sample

    @property
    def retained_bytes(self) -> int:
        return self._retained_bytes

    async def attach(
        self,
        *,
        last_delivery_seq: int = 0,
        audio_through_sample: int = 0,
    ) -> DeliveryAttachment[SessionOutput]:
        if last_delivery_seq < 0 or audio_through_sample < 0:
            raise LedgerError("invalid_cursor", "resume cursors must be non-negative")
        async with self._lock:
            self._validate_cursor(last_delivery_seq, audio_through_sample)
            old = self._attachment
            self._generation += 1
            attachment = DeliveryAttachment(
                self._generation,
                asyncio.Queue(maxsize=self.attachment_queue_maxsize),
            )
            self._attachment = attachment
            if old is not None:
                self._put_control(old.queue, AttachmentFence(self._generation))
            for record in self._records:
                if record.delivery_seq <= last_delivery_seq:
                    continue
                if record.end_sample and record.end_sample <= audio_through_sample:
                    continue
                self._put_control(attachment.queue, record)
            return attachment

    async def detach(self, generation: int) -> None:
        async with self._lock:
            if self._attachment is not None and self._attachment.generation == generation:
                self._attachment = None

    async def publish(self, payload: SessionOutput) -> ReliableDelivery:
        async with self._lock:
            if self._terminal is not None:
                raise LedgerError("session_terminal", "cannot publish after terminal")
            start = self._next_audio_sample
            end = start
            if isinstance(payload, AudioOutput):
                start = payload.output_sample_start
                end = payload.output_sample_end
                if start < 0 or end < start or start != self._next_audio_sample:
                    raise LedgerError(
                        "sample_gap",
                        "audio output samples must continue the canonical cursor",
                    )
                expected = _audio_sample_count(payload)
                if end - start != expected:
                    raise LedgerError(
                        "sample_range_mismatch",
                        "audio output sample range does not match PCM payload",
                    )
            retained = _payload_size(payload)
            if self._retained_bytes + retained > self.max_buffer_bytes:
                failure = AttachmentFailure(
                    "resume_buffer_exceeded",
                    "logical output exceeded the replay window",
                )
                attachment = self._attachment
                if attachment is not None:
                    self._put_control(attachment.queue, failure)
                raise LedgerError(failure.code, failure.message)
            record = ReliableDelivery(
                delivery_seq=self._next_delivery_seq,
                payload=payload,
                start_sample=start,
                end_sample=end,
                retained_bytes=retained,
            )
            self._records.append(record)
            self._retained_bytes += retained
            self._next_delivery_seq += 1
            self._next_audio_sample = max(self._next_audio_sample, end)
            if isinstance(payload, TerminalOutput):
                self._terminal = payload
            attachment = self._attachment
            if attachment is not None:
                self._put_control(attachment.queue, record)
            return record

    async def acknowledge(
        self,
        generation: int,
        *,
        through_delivery_seq: int,
        audio_through_sample: int,
    ) -> None:
        if through_delivery_seq < 0 or audio_through_sample < 0:
            raise LedgerError("invalid_ack", "ACK values must be non-negative")
        async with self._lock:
            self._require_generation(generation)
            self._validate_cursor(through_delivery_seq, audio_through_sample)
            if through_delivery_seq < self._acked_delivery_seq:
                raise LedgerError("ack_regression", "delivery ACK moved backwards")
            if audio_through_sample < self._acked_audio_sample:
                raise LedgerError("ack_regression", "audio ACK moved backwards")
            self._acked_delivery_seq = through_delivery_seq
            self._acked_audio_sample = audio_through_sample
            while self._records:
                record = self._records[0]
                if record.delivery_seq > through_delivery_seq:
                    break
                if record.end_sample > audio_through_sample:
                    break
                self._records.popleft()
                self._retained_bytes -= record.retained_bytes
                self._trimmed_through_seq = record.delivery_seq
                self._trimmed_audio_sample = max(
                    self._trimmed_audio_sample, record.end_sample
                )

    async def terminal_ack(self, generation: int) -> None:
        async with self._lock:
            self._require_generation(generation)
            if self._terminal is None:
                raise LedgerError("terminal_not_ready", "terminal output is not ready")
            self._terminal_acked = True

    def _validate_cursor(self, delivery_seq: int, audio_sample: int) -> None:
        if delivery_seq > self._next_delivery_seq - 1:
            raise LedgerError("cursor_ahead", "delivery cursor is ahead of output")
        if audio_sample > self._next_audio_sample:
            raise LedgerError("cursor_ahead", "audio cursor is ahead of output")
        if delivery_seq < self._trimmed_through_seq:
            raise LedgerError("cursor_expired", "delivery cursor is outside replay window")
        if audio_sample < self._trimmed_audio_sample:
            raise LedgerError("cursor_expired", "audio cursor is outside replay window")
        if delivery_seq == self._trimmed_through_seq:
            expected_sample = self._trimmed_audio_sample
        else:
            expected_sample = None
            for record in self._records:
                if record.delivery_seq == delivery_seq:
                    expected_sample = record.end_sample
                    break
            if expected_sample is None:
                raise LedgerError(
                    "cursor_expired",
                    "delivery cursor is no longer retained",
                )
        if audio_sample != expected_sample:
            raise LedgerError(
                "cursor_mismatch",
                "audio cursor does not match the delivery cursor",
            )

    def _require_generation(self, generation: int) -> None:
        if self._attachment is None or self._attachment.generation != generation:
            raise LedgerError("stale_attachment", "attachment has been fenced")

    @staticmethod
    def _put_control(queue: asyncio.Queue, message: LedgerMessage) -> None:
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull as exc:
            raise LedgerError("attachment_overflow", "attachment queue is full") from exc


def _payload_size(payload: SessionOutput) -> int:
    if isinstance(payload, AudioOutput):
        return len(payload.pcm_bytes)
    if isinstance(payload, TerminalOutput):
        body = {
            "status": payload.status.value,
            "message": payload.message,
            "metrics": payload.metrics,
            "usage": payload.usage,
        }
    else:
        body = repr(payload)
    return len(json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"))


def _audio_sample_count(payload: AudioOutput) -> int:
    widths = {
        "pcm_f32": 4,
        "pcm_s16le": 2,
        "pcm_s16": 2,
        "pcm_u8": 1,
    }
    width = widths.get(payload.audio.encoding.lower())
    if width is None:
        raise LedgerError(
            "unsupported_audio_encoding",
            f"unsupported audio encoding {payload.audio.encoding!r}",
        )
    channels = max(1, int(payload.audio.channels))
    divisor = width * channels
    if len(payload.pcm_bytes) % divisor:
        raise LedgerError(
            "audio_payload_alignment",
            "audio payload is not aligned to encoding and channel count",
        )
    return len(payload.pcm_bytes) // divisor


__all__ = [
    "AttachmentFailure",
    "AttachmentFence",
    "DeliveryAttachment",
    "DeliveryLedger",
    "LedgerError",
    "ReliableDelivery",
]
