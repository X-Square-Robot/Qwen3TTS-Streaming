from __future__ import annotations

import asyncio
import queue
import threading
import time
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

    def close(self, reason: str = "client closed") -> None:
        """Best-effort cancel and unblock message consumers immediately.

        ``cancel()`` is an in-band protocol message and cannot by itself
        guarantee that a broken transport will ever produce a terminal event.
        ``close()`` is the public hard-stop primitive for relay/worker shutdown:
        adapters may additionally tear down their transport, while this base
        implementation always closes the local message queue so
        ``iter_messages()`` cannot remain pinned.
        """
        cancel = getattr(self, "cancel", None)
        try:
            if callable(cancel):
                try:
                    cancel(reason=reason)
                except Exception:
                    # Closing a dead transport is still successful from the
                    # consumer's perspective: the local queue must be
                    # released even when the in-band cancel cannot be sent.
                    pass
            else:
                self._mark_send_closed()
        finally:
            # Closing is terminal for the caller's send side even when an
            # adapter's cancel implementation is a no-op or raises because
            # the transport has already disappeared.
            self._mark_send_closed()
            self._close_message_queue()

    def iter_messages(
        self,
        *,
        post_send_idle_timeout: float | None = None,
    ) -> Iterator[StreamEvent | AudioChunk]:
        """Iterate stream output until a terminal event or ``close()``.

        When ``post_send_idle_timeout`` is positive, silence is unlimited while
        the caller may still send text, but becomes bounded after ``end()`` or
        ``cancel()`` closes the send side.  The deadline starts at that observed
        edge and is re-armed on every message.  This lets relays detect a remote
        stream that accepted END but never produced a terminal event without
        reaching into the SDK's private queue/reader fields.
        """
        idle_timeout = (
            max(0.0, float(post_send_idle_timeout))
            if post_send_idle_timeout is not None
            else 0.0
        )
        last_message_at = time.monotonic()
        send_closed_seen = False
        while True:
            if self._send_closed and not send_closed_seen:
                send_closed_seen = True
                last_message_at = time.monotonic()
            if idle_timeout <= 0:
                item = self._messages.get()
            else:
                try:
                    item = self._messages.get(timeout=min(0.5, idle_timeout))
                except queue.Empty:
                    if not self._send_closed:
                        continue
                    now = time.monotonic()
                    if not send_closed_seen:
                        send_closed_seen = True
                        last_message_at = now
                        continue
                    if now - last_message_at >= idle_timeout:
                        raise TimeoutError(
                            f"stream session {self.session_id} received no message "
                            f"for {idle_timeout:.1f}s after send side closed"
                        )
                    continue
            if item is _QUEUE_SENTINEL:
                return
            last_message_at = time.monotonic()
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

    async def aclose(self, reason: str = "client closed") -> None:
        await asyncio.to_thread(self._sync.close, reason=reason)

    async def aiter_messages(self, *, post_send_idle_timeout: float | None = None):
        iterator = iter(
            self._sync.iter_messages(
                post_send_idle_timeout=post_send_idle_timeout,
            )
        )
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
