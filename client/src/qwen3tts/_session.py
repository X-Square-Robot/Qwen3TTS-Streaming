from __future__ import annotations

import asyncio
import queue
import threading
from typing import Iterator

from qwen3tts_protocol import AudioChunk, StreamEvent

from .exceptions import StreamClosedError

_QUEUE_SENTINEL = object()


def _is_terminal_message(message: StreamEvent | AudioChunk) -> bool:
    return isinstance(message, StreamEvent) and message.type in {"done", "error"}


class BaseStreamSession:
    def __init__(self, *, session_id: str, transport: str) -> None:
        self.session_id = session_id
        self.transport = transport
        self.degraded_to_oneshot = False
        self._messages: queue.Queue[object] = queue.Queue()
        self._closed = False
        self._send_closed = False
        self._lock = threading.Lock()

    def _put_message(self, message: StreamEvent | AudioChunk) -> None:
        self._messages.put(message)
        if _is_terminal_message(message):
            self._close_message_queue()

    def _close_message_queue(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._messages.put(_QUEUE_SENTINEL)

    def _check_send_open(self) -> None:
        if self._send_closed:
            raise StreamClosedError(
                f"stream session {self.session_id} is already closed for sending"
            )

    def _mark_send_closed(self) -> None:
        self._send_closed = True

    def iter_messages(self) -> Iterator[StreamEvent | AudioChunk]:
        while True:
            item = self._messages.get()
            if item is _QUEUE_SENTINEL:
                return
            yield item  # type: ignore[misc]


class AsyncStreamSession:
    def __init__(self, sync_session: BaseStreamSession) -> None:
        self._sync = sync_session
        self.session_id = sync_session.session_id
        self.transport = sync_session.transport
        self.degraded_to_oneshot = sync_session.degraded_to_oneshot

    async def send_text(
        self,
        text: str,
        *,
        seq_no: int | None = None,
        client_timestamp_ms: int | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._sync.send_text,
            text,
            seq_no=seq_no,
            client_timestamp_ms=client_timestamp_ms,
        )

    async def end(self, *, client_timestamp_ms: int | None = None) -> None:
        await asyncio.to_thread(self._sync.end, client_timestamp_ms=client_timestamp_ms)

    async def cancel(self, reason: str = "") -> None:
        await asyncio.to_thread(self._sync.cancel, reason=reason)

    async def aiter_messages(self):
        iterator = iter(self._sync.iter_messages())
        while True:
            try:
                item = await asyncio.to_thread(self._next_or_sentinel, iterator)
            except RuntimeError as exc:
                if exc.args and exc.args[0] == "__stop__":
                    return
                raise
            yield item

    @staticmethod
    def _next_or_sentinel(iterator):
        try:
            return next(iterator)
        except StopIteration:
            raise RuntimeError("__stop__")
