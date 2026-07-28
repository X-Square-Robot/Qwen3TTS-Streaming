from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import random
import socket
import threading
import time
import weakref

from qwen3tts_protocol import (
    AudioChunk,
    AudioFormat,
    BytesResult,
    Capabilities,
    SessionStartRequest,
    StreamCancelRequest,
    StreamEvent,
    StreamTextChunk,
)

from .._internal.raw_websocket import (
    RawWebSocketConnection,
    RawWebSocketError,
    ws_close,
    ws_connect,
    ws_recv_frame,
    ws_send_json,
)
from .._internal.utils import (
    build_bytes_result,
    capabilities_from_payload,
    decode_stream_event,
    stream_text_chunk_to_mapping,
    synthesis_config_to_mapping,
)
from .._session import BaseStreamSession, _is_terminal_message
from ..constants import TRANSPORT_ENGINE_WEBSOCKET
from ..exceptions import (
    EngineVersionMismatchError,
    PoolAcquireTimeoutError,
    PoolSaturatedError,
    ProtocolError,
    ProtocolVersionMismatchError,
    StreamClosedError,
)

_WEBSOCKET_REUSABLE_META_KEY = "websocket_connection_reusable"
_DEFAULT_ACQUIRE_TIMEOUT = object()


class _ConnectionNotReusableError(RuntimeError):
    """The peer is healthy but does not support persistent websocket reuse."""


def _optional_timeout(
    value: float | None,
    *,
    name: str,
    zero_disables: bool,
) -> float | None:
    """Validate an optional duration while keeping ``None`` explicit.

    ``idle_ttl`` and ``max_lifetime`` accept both zero and ``None`` as the
    compatibility spelling for disabled eviction. ``acquire_timeout=0`` is
    intentionally different: it requests immediate, non-blocking admission.
    """

    if value is None:
        return None
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"{name} must be a finite non-negative number or None")
    if zero_disables and duration == 0:
        return None
    return duration


def _finalize_connection_pool(
    pool_lock: threading.Lock,
    connections: set[RawWebSocketConnection],
    idle_connections: list[RawWebSocketConnection],
    keepalive_stop: threading.Event,
    close_connection,
) -> None:
    """Release sockets without retaining the adapter from a daemon thread."""

    keepalive_stop.set()
    with pool_lock:
        tracked = list(connections)
        connections.clear()
        idle_connections.clear()
    for conn in tracked:
        close_connection(conn)


class EngineWebSocketAdapter:
    transport_name = TRANSPORT_ENGINE_WEBSOCKET

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float,
        connect_timeout: float | None = None,
        headers: dict[str, str] | None = None,
        reconnect_attempts: int = 1,
        max_connections: int = 32,
        max_idle_connections: int = 8,
        max_pending_acquires: int = 256,
        acquire_timeout: float | None = 30.0,
        idle_ttl: float | None = None,
        max_lifetime: float | None = None,
        keepalive_interval: float = 15.0,
        keepalive_jitter: float = 0.2,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        # Keep the historical behavior when omitted, while allowing callers
        # to bound a stalled handshake independently from a long stream's idle
        # timeout.  A single timeout previously made a 120-second stream budget
        # also occupy a worker thread for up to 120 seconds during connect.
        self.connect_timeout = timeout if connect_timeout is None else connect_timeout
        self.headers = dict(headers or {})
        # A websocket carries one logical TTS session at a time because audio
        # frames are raw binary data without a session id.  Finished sockets
        # are therefore pooled and reused serially; concurrent sessions simply
        # check out different sockets from the pool.
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.max_connections = int(max_connections)
        self.max_idle_connections = int(max_idle_connections)
        self.max_pending_acquires = int(max_pending_acquires)
        if self.max_connections <= 0:
            raise ValueError("max_connections must be greater than zero")
        if self.max_idle_connections < 0:
            raise ValueError("max_idle_connections must be non-negative")
        if self.max_idle_connections > self.max_connections:
            raise ValueError("max_idle_connections must not exceed max_connections")
        if self.max_pending_acquires < 0:
            raise ValueError("max_pending_acquires must be non-negative")
        self.acquire_timeout = _optional_timeout(
            acquire_timeout,
            name="acquire_timeout",
            zero_disables=False,
        )
        self.idle_ttl = _optional_timeout(
            idle_ttl,
            name="idle_ttl",
            zero_disables=True,
        )
        self.max_lifetime = _optional_timeout(
            max_lifetime,
            name="max_lifetime",
            zero_disables=True,
        )
        self.keepalive_interval = max(0.0, float(keepalive_interval))
        self.keepalive_jitter = float(keepalive_jitter)
        if not math.isfinite(self.keepalive_jitter) or not (
            0.0 <= self.keepalive_jitter < 1.0
        ):
            raise ValueError("keepalive_jitter must be in the range [0, 1)")
        self._pool_lock = threading.Lock()
        self._pool_condition = threading.Condition(self._pool_lock)
        self._idle_connections: list[RawWebSocketConnection] = []
        self._connections: set[RawWebSocketConnection] = set()
        self._connection_created_at: dict[RawWebSocketConnection, float] = {}
        self._connection_idle_since: dict[RawWebSocketConnection, float] = {}
        self._connecting = 0
        # Each token represents exactly one blocked caller. Only the token at
        # the head may claim a released connection or reserve new capacity,
        # preventing fresh callers from starving an existing waiter.
        self._acquire_waiters: deque[object] = deque()
        self._waiter_handoffs: dict[object, RawWebSocketConnection] = {}
        self._new_connection_waiters: set[object] = set()
        self._closed = False
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        self._finalizer = weakref.finalize(
            self,
            _finalize_connection_pool,
            self._pool_lock,
            self._connections,
            self._idle_connections,
            self._keepalive_stop,
            ws_close,
        )

    def connect(self) -> None:
        """Eagerly establish and retain one authenticated websocket.

        The adapter otherwise connects lazily on the first request.  Long-lived
        workers can call this during startup so a PaaS/LB handshake never sits
        on the first user's latency path.
        """
        # Legacy gateways answer capabilities and then close the socket.  They
        # remain a valid connect() target (the adapter safely falls back to a
        # fresh connection per session), but an explicit public prewarm() must
        # report that its requested reusable capacity was not reached.
        self._prewarm(
            connections=1,
            timeout=self.connect_timeout,
            allow_legacy_shortfall=True,
        )

    def prewarm(
        self,
        connections: int = 1,
        *,
        timeout: float | None = None,
    ) -> int:
        """Fill the pool to a target number of reusable idle websockets.

        Missing connections are established concurrently, so startup latency
        is bounded by the slowest gateway handshake rather than their sum.
        The target is capped by ``max_idle_connections``.  Successfully warmed
        sockets stay in the pool if another attempt fails; in that case this
        method raises an error that reports the requested target and the
        actual reusable capacity.

        ``timeout`` bounds each capabilities round-trip.  When omitted, the
        shorter connection timeout is used rather than the stream idle timeout.
        """

        return self._prewarm(
            connections=connections,
            timeout=timeout,
            allow_legacy_shortfall=False,
        )

    def _prewarm(
        self,
        *,
        connections: int,
        timeout: float | None,
        allow_legacy_shortfall: bool,
    ) -> int:
        try:
            requested = int(connections)
        except (TypeError, ValueError) as exc:
            raise ValueError("connections must be a non-negative integer") from exc
        if requested < 0:
            raise ValueError("connections must be a non-negative integer")

        request_timeout = (
            self.connect_timeout if timeout is None else max(0.0, float(timeout))
        )
        target = min(
            requested,
            self.max_idle_connections,
            self.max_connections,
        )
        self._expire_idle_connections()
        with self._pool_lock:
            if self._closed:
                raise StreamClosedError("websocket adapter is closed")
            idle_count = len(self._idle_connections)
        missing = max(0, target - idle_count)
        if missing == 0:
            return idle_count

        failures: list[Exception] = []
        with ThreadPoolExecutor(
            max_workers=missing,
            thread_name_prefix="qwen3tts-ws-prewarm",
        ) as executor:
            futures = [
                executor.submit(self._prewarm_one, request_timeout, target)
                for _ in range(missing)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(exc)

        with self._pool_lock:
            actual = len(self._idle_connections)
        # Release/version skew is a startup-fatal compatibility problem, not a
        # recoverable pool-capacity warning.  Preserve its public exception
        # type even when several handshakes ran concurrently.
        for failure in failures:
            if isinstance(
                failure,
                (ProtocolVersionMismatchError, EngineVersionMismatchError),
            ):
                raise failure

        if actual >= target:
            return actual

        if (
            allow_legacy_shortfall
            and failures
            and all(
                isinstance(failure, _ConnectionNotReusableError) for failure in failures
            )
        ):
            return actual
        if allow_legacy_shortfall and len(failures) == 1:
            # connect() historically exposed the underlying handshake/probe
            # exception.  Keep that behavior while sharing the prewarm path.
            raise failures[0]

        details = "; ".join(
            f"{type(failure).__name__}: {failure}" for failure in failures[:3]
        )
        message = (
            "websocket prewarm reached "
            f"{actual}/{target} reusable idle connections "
            f"(requested={requested}, max_idle_connections="
            f"{self.max_idle_connections})"
        )
        if details:
            message = f"{message}; failures: {details}"
        error = RuntimeError(message)
        if failures:
            raise error from failures[0]
        raise error

    def _prewarm_one(self, timeout: float, target: int) -> None:
        if not self._reserve_new_connection(
            acquire_timeout=timeout,
            idle_target=target,
        ):
            return
        conn = self._new_connection()
        try:
            _capabilities, reusable = self._request_capabilities(
                conn,
                timeout=timeout,
            )
        except Exception:
            self._discard_connection(conn)
            raise
        if not reusable:
            self._discard_connection(conn)
            raise _ConnectionNotReusableError(
                "capabilities response did not advertise "
                f"{_WEBSOCKET_REUSABLE_META_KEY}=true"
            )
        self._release_connection(conn)

    def close(self) -> None:
        """Close all idle and active websocket connections held by the pool."""
        with self._pool_condition:
            if self._closed:
                return
            self._closed = True
            connections = list(self._connections)
            self._connections.clear()
            self._idle_connections.clear()
            self._connection_created_at.clear()
            self._connection_idle_since.clear()
            self._acquire_waiters.clear()
            self._waiter_handoffs.clear()
            self._new_connection_waiters.clear()
            # Connecting sockets are not visible yet. Their reserved slots are
            # released by _new_connection() when the handshake returns and
            # the newly created socket is closed there before it can escape.
            self._pool_condition.notify_all()
        self._keepalive_stop.set()
        self._finalizer.detach()
        for conn in connections:
            ws_close(conn)

    def _new_connection(self) -> RawWebSocketConnection:
        """Establish a socket for one capacity slot reserved by checkout."""

        last_error: BaseException | None = None
        reservation_released = False
        try:
            for attempt in range(self.reconnect_attempts + 1):
                with self._pool_lock:
                    if self._closed:
                        raise StreamClosedError("websocket adapter is closed")
                try:
                    conn = ws_connect(
                        self.endpoint,
                        timeout=self.connect_timeout,
                        headers=self.headers,
                    )
                except (OSError, RawWebSocketError) as exc:
                    last_error = exc
                    if attempt >= self.reconnect_attempts:
                        raise
                    time.sleep(min(0.2, 0.05 * (2**attempt)))
                    continue

                created_at = time.monotonic()
                with self._pool_condition:
                    self._connecting -= 1
                    reservation_released = True
                    closed = self._closed
                    if not closed:
                        self._connections.add(conn)
                        self._connection_created_at[conn] = created_at
                    self._pool_condition.notify_all()
                if closed:
                    # close() may have returned while the network handshake
                    # was in flight and therefore could not see this socket.
                    ws_close(conn)
                    raise StreamClosedError("websocket adapter closed during connect")
                return conn
        except BaseException:
            # Successful handshakes release the reservation in the block above.
            # Every other exit must do it here so a failed connect cannot leak
            # capacity and permanently stall the FIFO queue.
            with self._pool_condition:
                if not reservation_released:
                    self._connecting -= 1
                    reservation_released = True
                self._pool_condition.notify_all()
            raise
        assert last_error is not None  # pragma: no cover - loop always returns/raises
        raise last_error

    def _checkout_connection(
        self,
        *,
        acquire_timeout: float | None | object = _DEFAULT_ACQUIRE_TIMEOUT,
    ) -> tuple[RawWebSocketConnection, bool]:
        """Acquire one exclusive socket with bounded, FIFO admission.

        ``_connections + _connecting`` never exceeds ``max_connections``.
        A caller only enters the wait queue after both idle reuse and a new
        connection reservation are unavailable.
        """

        if acquire_timeout is _DEFAULT_ACQUIRE_TIMEOUT:
            wait_timeout = self.acquire_timeout
        else:
            wait_timeout = _optional_timeout(
                acquire_timeout,  # type: ignore[arg-type]
                name="acquire_timeout",
                zero_disables=False,
            )
        deadline = None if wait_timeout is None else time.monotonic() + wait_timeout
        waiter: object | None = None

        while True:
            self._expire_idle_connections()
            reserve_connection = False
            leased: RawWebSocketConnection | None = None
            with self._pool_condition:
                if self._closed:
                    if waiter is not None:
                        self._remove_waiter_locked(waiter)
                    raise StreamClosedError("websocket adapter is closed")

                if waiter is not None:
                    leased = self._waiter_handoffs.pop(waiter, None)
                    if leased is not None:
                        self._connection_idle_since.pop(leased, None)

                is_turn = waiter is None and not self._acquire_waiters
                if waiter is not None and self._acquire_waiters:
                    is_turn = self._acquire_waiters[0] is waiter

                if leased is None and is_turn:
                    while self._idle_connections:
                        candidate = self._idle_connections.pop()
                        self._connection_idle_since.pop(candidate, None)
                        if candidate in self._connections:
                            leased = candidate
                            break

                    if leased is None and (
                        len(self._connections) + self._connecting < self.max_connections
                    ):
                        self._connecting += 1
                        reserve_connection = True

                    if leased is not None or reserve_connection:
                        if waiter is not None:
                            self._remove_waiter_locked(waiter)
                        self._pool_condition.notify_all()

                if leased is None and not reserve_connection:
                    if waiter is None:
                        if len(self._acquire_waiters) >= self.max_pending_acquires:
                            raise PoolSaturatedError(
                                "websocket connection pool is saturated "
                                f"(max_connections={self.max_connections}, "
                                "max_pending_acquires="
                                f"{self.max_pending_acquires})"
                            )
                        waiter = object()
                        self._acquire_waiters.append(waiter)

                    remaining = (
                        None
                        if deadline is None
                        else max(0.0, deadline - time.monotonic())
                    )
                    if remaining == 0:
                        self._remove_waiter_locked(waiter)
                        self._pool_condition.notify_all()
                        raise PoolAcquireTimeoutError(
                            "timed out waiting for a websocket connection "
                            f"after {wait_timeout:.3f}s "
                            f"(max_connections={self.max_connections})"
                        )
                    self._pool_condition.wait(timeout=remaining)

            if leased is not None:
                return leased, True
            if reserve_connection:
                return self._new_connection(), False

    def _remove_waiter_locked(self, waiter: object) -> None:
        try:
            self._acquire_waiters.remove(waiter)
        except ValueError:
            pass
        self._waiter_handoffs.pop(waiter, None)
        self._new_connection_waiters.discard(waiter)

    def _reserve_new_connection(
        self,
        *,
        acquire_timeout: float | None,
        idle_target: int,
    ) -> bool:
        """Reserve physical capacity for prewarm without consuming idle WS.

        Parallel prewarm workers must not check out the idle sockets counted
        when ``missing`` was calculated: doing so can turn target=4,idle=1
        into only three physical sockets. A worker stops early if another
        release has already satisfied the shared idle target.
        """

        wait_timeout = _optional_timeout(
            acquire_timeout,
            name="acquire_timeout",
            zero_disables=False,
        )
        deadline = None if wait_timeout is None else time.monotonic() + wait_timeout
        waiter: object | None = None
        while True:
            self._expire_idle_connections()
            with self._pool_condition:
                if self._closed:
                    if waiter is not None:
                        self._remove_waiter_locked(waiter)
                    raise StreamClosedError("websocket adapter is closed")
                if len(self._idle_connections) >= idle_target:
                    if waiter is not None:
                        self._remove_waiter_locked(waiter)
                        self._pool_condition.notify_all()
                    return False

                is_turn = waiter is None and not self._acquire_waiters
                if waiter is not None and self._acquire_waiters:
                    is_turn = self._acquire_waiters[0] is waiter
                if is_turn and (
                    len(self._connections) + self._connecting < self.max_connections
                ):
                    self._connecting += 1
                    if waiter is not None:
                        self._remove_waiter_locked(waiter)
                    self._pool_condition.notify_all()
                    return True

                if waiter is None:
                    if len(self._acquire_waiters) >= self.max_pending_acquires:
                        raise PoolSaturatedError(
                            "websocket connection pool is saturated during prewarm "
                            f"(max_connections={self.max_connections}, "
                            "max_pending_acquires="
                            f"{self.max_pending_acquires})"
                        )
                    waiter = object()
                    self._acquire_waiters.append(waiter)
                    self._new_connection_waiters.add(waiter)

                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if remaining == 0:
                    self._remove_waiter_locked(waiter)
                    self._pool_condition.notify_all()
                    raise PoolAcquireTimeoutError(
                        "timed out waiting for websocket prewarm capacity "
                        f"after {wait_timeout:.3f}s "
                        f"(max_connections={self.max_connections})"
                    )
                self._pool_condition.wait(timeout=remaining)

    def _acquire_connection(self) -> RawWebSocketConnection:
        """Return a live exclusive connection, reconnecting stale idle ones.

        The background keepalive validates idle sockets when enabled.  If it is
        disabled, a capabilities round-trip before reuse detects connections
        reaped by aiohttp, a proxy, or the remote peer.  In either mode an
        initial ``start`` write failure is retried by ``open_stream``.
        """
        while True:
            conn, reused = self._checkout_connection()
            if not reused or self.keepalive_interval > 0:
                return conn
            try:
                _capabilities, reusable_protocol = self._request_capabilities(
                    conn,
                    timeout=self.connect_timeout,
                )
                conn.settimeout(self.connect_timeout)
            except Exception:
                self._discard_connection(conn)
                continue
            if not reusable_protocol:
                self._discard_connection(conn)
                continue
            return conn

    def _release_connection(
        self,
        conn: RawWebSocketConnection,
        *,
        idle_since: float | None = None,
    ) -> None:
        try:
            # Receive loops poll with 200/500 ms socket timeouts. Do not leak
            # that transport detail into the next session's initial write.
            conn.settimeout(self.connect_timeout)
        except Exception:
            self._discard_connection(conn)
            return
        close = False
        retained = False
        now = time.monotonic()
        with self._pool_condition:
            if self._closed or conn not in self._connections:
                close = True
            else:
                created_at = self._connection_created_at.setdefault(conn, now)
                lifetime_expired = bool(
                    self.max_lifetime is not None
                    and now - created_at >= self.max_lifetime
                )
                if lifetime_expired:
                    close = True

            if close:
                self._connections.discard(conn)
                self._connection_created_at.pop(conn, None)
                self._connection_idle_since.pop(conn, None)
                try:
                    self._idle_connections.remove(conn)
                except ValueError:
                    pass
            elif (
                self._acquire_waiters
                and self._acquire_waiters[0] not in self._new_connection_waiters
            ):
                # Direct handoff keeps max_idle_connections=0 useful: an
                # already-waiting caller receives the live socket without an
                # avoidable close/reconnect cycle, while FIFO order is strict.
                waiter = self._acquire_waiters.popleft()
                self._waiter_handoffs[waiter] = conn
            elif len(self._idle_connections) >= self.max_idle_connections:
                self._connections.discard(conn)
                self._connection_created_at.pop(conn, None)
                self._connection_idle_since.pop(conn, None)
                close = True
            elif conn not in self._idle_connections:
                self._idle_connections.append(conn)
                self._connection_idle_since[conn] = (
                    now if idle_since is None else idle_since
                )
                retained = True
            self._pool_condition.notify_all()
        if close:
            ws_close(conn)
        elif retained:
            self._ensure_keepalive_thread()

    def _ensure_keepalive_thread(self) -> None:
        interval = self._maintenance_interval()
        if interval is None:
            return
        with self._pool_lock:
            if self._closed:
                return
            if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
                return
            thread = threading.Thread(
                target=self._keepalive_worker,
                args=(
                    weakref.ref(self),
                    self._keepalive_stop,
                    interval,
                    self.keepalive_jitter if self.keepalive_interval > 0 else 0.0,
                    self.keepalive_interval > 0,
                ),
                name="qwen3tts-ws-keepalive",
                daemon=True,
            )
            self._keepalive_thread = thread
            thread.start()

    def _maintenance_interval(self) -> float | None:
        intervals = [
            value
            for value in (
                self.keepalive_interval if self.keepalive_interval > 0 else None,
                self.idle_ttl,
                self.max_lifetime,
            )
            if value is not None and value > 0
        ]
        if not intervals:
            return None
        return max(0.01, min(intervals))

    @staticmethod
    def _keepalive_worker(
        adapter_ref,
        stop: threading.Event,
        interval: float,
        jitter: float = 0.0,
        probe: bool = True,
    ) -> None:
        while True:
            spread = interval * jitter
            delay = random.uniform(interval - spread, interval + spread)
            if stop.wait(max(0.01, delay)):
                return
            adapter = adapter_ref()
            if adapter is None:
                return
            adapter._keepalive_once(stop, probe=probe)
            # Do not retain the adapter while sleeping; otherwise a forgotten
            # client.close() leaks both this daemon thread and every idle socket.
            del adapter

    def _keepalive_once(
        self,
        stop: threading.Event,
        *,
        probe: bool = True,
    ) -> None:
        self._expire_idle_connections()
        if not probe:
            return
        with self._pool_lock:
            idle = list(self._idle_connections)
        for conn in idle:
            if stop.is_set():
                return
            with self._pool_condition:
                # Never let maintenance jump ahead of business traffic. Once
                # a caller is queued, leave every remaining idle socket for
                # normal FIFO checkout/direct handoff.
                if self._acquire_waiters:
                    return
                if conn not in self._connections or conn not in self._idle_connections:
                    continue
                self._idle_connections.remove(conn)
                idle_since = self._connection_idle_since.pop(conn, None)
            try:
                # This both creates application traffic for idle-reaping
                # proxies and drains/responds to websocket ping frames.
                _capabilities, reusable_protocol = self._request_capabilities(
                    conn,
                    timeout=self.connect_timeout,
                )
            except Exception:
                self._discard_connection(conn)
            else:
                if reusable_protocol:
                    # A transport keepalive is not business use. Preserve the
                    # original idle timestamp so probes cannot defeat idle_ttl.
                    self._release_connection(conn, idle_since=idle_since)
                else:
                    self._discard_connection(conn)

    def _expire_idle_connections(self) -> None:
        if self.idle_ttl is None and self.max_lifetime is None:
            return
        now = time.monotonic()
        expired: list[RawWebSocketConnection] = []
        with self._pool_condition:
            retained: list[RawWebSocketConnection] = []
            for conn in self._idle_connections:
                if conn not in self._connections:
                    self._connection_created_at.pop(conn, None)
                    self._connection_idle_since.pop(conn, None)
                    continue
                created_at = self._connection_created_at.setdefault(conn, now)
                idle_since = self._connection_idle_since.setdefault(conn, now)
                lifetime_expired = bool(
                    self.max_lifetime is not None
                    and now - created_at >= self.max_lifetime
                )
                idle_expired = bool(
                    self.idle_ttl is not None and now - idle_since >= self.idle_ttl
                )
                if lifetime_expired or idle_expired:
                    self._connections.discard(conn)
                    self._connection_created_at.pop(conn, None)
                    self._connection_idle_since.pop(conn, None)
                    expired.append(conn)
                else:
                    retained.append(conn)
            if expired or len(retained) != len(self._idle_connections):
                self._idle_connections[:] = retained
                self._pool_condition.notify_all()
        for conn in expired:
            ws_close(conn)

    def _discard_connection(self, conn: RawWebSocketConnection) -> None:
        with self._pool_condition:
            self._connections.discard(conn)
            self._connection_created_at.pop(conn, None)
            self._connection_idle_since.pop(conn, None)
            try:
                self._idle_connections.remove(conn)
            except ValueError:
                pass
            self._pool_condition.notify_all()
        ws_close(conn)

    def _request_capabilities(
        self,
        conn: RawWebSocketConnection,
        *,
        timeout: float | None = None,
    ) -> tuple[Capabilities, bool]:
        request_timeout = self.timeout if timeout is None else max(0.0, float(timeout))
        conn.settimeout(request_timeout)
        ws_send_json(conn, {"type": "get_capabilities"})
        deadline = time.perf_counter() + request_timeout
        while time.perf_counter() < deadline:
            conn.settimeout(max(0.05, min(0.2, deadline - time.perf_counter())))
            try:
                opcode, payload = ws_recv_frame(conn)
            except socket.timeout:
                continue
            if opcode == 0x8:
                raise RawWebSocketError("websocket closed during capabilities request")
            if opcode != 0x1:
                # A well-behaved persistent server has no leftover session
                # frames after its terminal event.  Ignore any stale binary
                # frame defensively while waiting for the probe response.
                continue
            message = json.loads(payload.decode("utf-8"))
            if message.get("type") == "capabilities":
                reusable = _is_truthy(message.get(_WEBSOCKET_REUSABLE_META_KEY, False))
                return (
                    capabilities_from_payload(message.get("capabilities", {})),
                    reusable,
                )
        raise TimeoutError("websocket capabilities request timed out")

    def get_capabilities(self, *, timeout: float | None = None) -> Capabilities:
        last_error: BaseException | None = None
        for attempt in range(self.reconnect_attempts + 1):
            conn, _reused = self._checkout_connection()
            try:
                capabilities, reusable = self._request_capabilities(
                    conn, timeout=timeout
                )
            except Exception as exc:
                last_error = exc
                self._discard_connection(conn)
                if attempt >= self.reconnect_attempts:
                    raise
                continue
            if reusable:
                self._release_connection(conn)
            else:
                self._discard_connection(conn)
            return capabilities
        assert last_error is not None  # pragma: no cover
        raise last_error

    def synthesize_bytes(self, text: str, *, request) -> BytesResult:
        session_id = request.session_id or ""
        payload = {
            "type": "oneshot",
            "session_id": session_id,
            "text": text,
            "config": synthesis_config_to_mapping(request.config),
        }
        conn: RawWebSocketConnection | None = None
        last_error: BaseException | None = None
        for attempt in range(self.reconnect_attempts + 1):
            candidate = self._acquire_connection()
            try:
                ws_send_json(candidate, payload)
            except (OSError, RawWebSocketError) as exc:
                last_error = exc
                self._discard_connection(candidate)
                if attempt >= self.reconnect_attempts:
                    raise
                continue
            conn = candidate
            break
        if conn is None:  # pragma: no cover - loop always assigns or raises
            assert last_error is not None
            raise last_error
        events: list[StreamEvent] = []
        warnings: list[str] = []
        audio_format = request.config.audio
        audio_parts: list[bytes] = []
        reusable = False
        try:
            terminal_seen = False
            for message in _iter_conn_messages(conn, timeout=self.timeout):
                if isinstance(message, AudioChunk):
                    audio_parts.append(message.pcm_bytes)
                    audio_format = message.audio
                    continue
                events.append(message)
                if message.type == "warning" and message.message:
                    warnings.append(message.message)
                if message.type in {"done", "error"}:
                    terminal_seen = True
                    reusable = _terminal_allows_connection_reuse(message)
                    break
            if not terminal_seen:
                # Connection closed (opcode 0x8) before done/error: the
                # audio collected so far is silently truncated. Fail loudly
                # instead of returning a partial result with no signal.
                raise ProtocolError(
                    "websocket stream closed without terminal event "
                    f"({len(audio_parts)} audio chunks received)"
                )
        finally:
            if reusable:
                self._release_connection(conn)
            else:
                self._discard_connection(conn)
        return build_bytes_result(
            audio_bytes=b"".join(audio_parts),
            audio_format=audio_format,
            session_id=session_id,
            transport=self.transport_name,
            events=events,
            warnings=warnings,
            details={},
        )

    def open_stream(self, start_request: SessionStartRequest):
        last_error: BaseException | None = None
        for attempt in range(self.reconnect_attempts + 1):
            conn = self._acquire_connection()
            try:
                return EngineWebSocketStreamSession(
                    adapter=self,
                    start_request=start_request,
                    conn=conn,
                )
            except (OSError, RawWebSocketError) as exc:
                # No text has been submitted yet.  Retrying a failed initial
                # start write on a fresh connection is safe; once the session
                # object is returned, mid-stream replay is deliberately left
                # to the caller because it could duplicate audio.
                last_error = exc
                self._discard_connection(conn)
                if attempt >= self.reconnect_attempts:
                    raise
        assert last_error is not None  # pragma: no cover
        raise last_error


def _iter_conn_messages(
    conn: RawWebSocketConnection,
    *,
    timeout: float,
):
    # ``timeout`` is an *idle* limit: the clock re-arms on every received
    # frame, so a healthy long synthesis can stream for arbitrarily long
    # while a silent link still fails within ``timeout``.  It used to be an
    # absolute deadline for the whole stream, which truncated any synthesis
    # whose wall time exceeded it (~230 chars of text at the observed
    # generation speed with the default 120 s).
    idle_deadline = time.perf_counter() + timeout
    current_audio = AudioFormat()
    while time.perf_counter() < idle_deadline:
        conn.settimeout(max(0.02, min(0.5, idle_deadline - time.perf_counter())))
        try:
            opcode, payload = ws_recv_frame(conn)
        except socket.timeout:
            continue
        idle_deadline = time.perf_counter() + timeout
        if opcode == 0x2:
            yield AudioChunk(
                pcm_bytes=payload,
                audio=current_audio,
                meta={},
            )
            continue
        if opcode == 0x8:
            return
        if opcode != 0x1:
            continue
        message = json.loads(payload.decode("utf-8"))
        if message.get("type") != "event":
            continue
        event = decode_stream_event(message.get("event") or {})
        if event.audio is not None:
            current_audio = event.audio
        yield event
        if event.type in {"done", "error"}:
            return
    raise TimeoutError(
        f"websocket stream idle for {timeout:.0f}s waiting for terminal event"
    )


def _terminal_allows_connection_reuse(message: StreamEvent) -> bool:
    """Only pool sockets when the gateway advertises the persistent protocol.

    Legacy gateways close the physical websocket after ``done``. Treating a
    marker-less terminal event as reusable creates a race where the next
    ``start`` can be written just before the peer's close frame arrives.
    """

    return _is_truthy(message.meta.get(_WEBSOCKET_REUSABLE_META_KEY, ""))


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class EngineWebSocketStreamSession(BaseStreamSession):
    def __init__(
        self,
        *,
        adapter: EngineWebSocketAdapter,
        start_request: SessionStartRequest,
        conn: RawWebSocketConnection,
    ) -> None:
        super().__init__(
            session_id=start_request.session_id, transport=adapter.transport_name
        )
        self._adapter = adapter
        self._start_request = start_request
        self._conn = conn
        # Serializes every send with terminal release. Without this lock a
        # caller can pass the send-open check, then the reader can return the
        # socket to the pool and a late send can corrupt the next session.
        self._transport_lock = threading.RLock()
        self._transport_finished = False
        try:
            ws_send_json(
                self._conn,
                {
                    "type": "start",
                    "session_id": start_request.session_id,
                    "config": synthesis_config_to_mapping(start_request.config),
                },
            )
        except BaseException:
            raise
        self._reader = threading.Thread(
            target=self._reader_loop, name=f"ws-session-{self.session_id}", daemon=True
        )
        self._reader.start()

    def _reader_loop(self) -> None:
        terminal_seen = False
        try:
            for message in _iter_conn_messages(
                self._conn, timeout=self._adapter.timeout
            ):
                if _is_terminal_message(message):
                    terminal_seen = True
                    with self._transport_lock:
                        self._mark_send_closed()
                        # Return the physical websocket before publishing the
                        # terminal event. A consumer that immediately calls
                        # close() after seeing done must not race and tear down
                        # a healthy connection that is already reusable.
                        self._finish_transport(
                            reusable=_terminal_allows_connection_reuse(message)
                        )
                self._put_message(message)
        except Exception as exc:
            terminal_seen = True
            self._finish_transport(reusable=False)
            self._put_message(
                StreamEvent(
                    type="error",
                    session_id=self.session_id,
                    message=str(exc),
                )
            )
        finally:
            if not terminal_seen:
                self._finish_transport(reusable=False)
                # Clean reader exit without done/error — e.g. the gateway
                # closed the connection mid-redeploy (close frame, opcode
                # 0x8). Without a terminal event the queue sentinel is never
                # enqueued and iter_messages() blocks forever, permanently
                # pinning the caller's thread (this starved a relay worker
                # pool in production). Surface it as an error so callers can
                # log and run their error path.
                self._put_message(
                    StreamEvent(
                        type="error",
                        session_id=self.session_id,
                        message="connection closed without terminal event",
                    )
                )
            # Last-resort unblock: idempotent, and covers any exit path the
            # branches above might miss.
            self._close_message_queue()

    def _finish_transport(self, *, reusable: bool) -> None:
        with self._transport_lock:
            if self._transport_finished:
                return
            self._transport_finished = True
        if reusable:
            self._adapter._release_connection(self._conn)
        else:
            self._adapter._discard_connection(self._conn)

    def send_text(
        self,
        text: str,
        *,
        seq_no: int | None = None,
        client_timestamp_ms: int | None = None,
    ) -> None:
        chunk = StreamTextChunk(
            text=text,
            seq_no=int(seq_no or 0),
            client_timestamp_ms=int(client_timestamp_ms or 0),
        )
        payload = {"type": "text"}
        payload.update(stream_text_chunk_to_mapping(chunk))
        self._send_or_close(payload)

    def end(self, *, client_timestamp_ms: int | None = None) -> None:
        self._finish_input("end", client_timestamp_ms=client_timestamp_ms)

    def stop(self, *, client_timestamp_ms: int | None = None) -> None:
        """Gracefully stop text input and drain generated audio.

        ``end()`` remains the compatibility spelling; persistent gateways also
        accept the explicit ``stop`` control message.
        """
        self._finish_input("stop", client_timestamp_ms=client_timestamp_ms)

    def _finish_input(
        self, message_type: str, *, client_timestamp_ms: int | None = None
    ) -> None:
        payload = {"type": message_type}
        if client_timestamp_ms is not None:
            payload["client_timestamp_ms"] = int(client_timestamp_ms)
        self._send_or_close(payload, close_send=True)

    def cancel(self, reason: str = "") -> None:
        request = StreamCancelRequest(reason=reason)
        try:
            self._send_or_close(
                {"type": "cancel", "reason": request.reason},
                close_send=True,
            )
        except StreamClosedError:
            # Best-effort: a dead connection already achieves what cancel
            # wanted (the server tears the session down on disconnect).
            pass

    def close(self, reason: str = "client closed") -> None:
        """Cancel and force-close the websocket from any caller thread.

        This is intentionally stronger than ``cancel()``: a stalled or broken
        link may never deliver a terminal event, so relays need a public way to
        unblock both the SDK reader and ``iter_messages()`` without reaching
        into ``session._conn`` or importing private websocket helpers.
        """
        try:
            super().close(reason=reason)
        finally:
            # Normal terminal events release the connection before they are
            # visible to consumers.  A close while the session is still active
            # remains a hard stop and discards that one physical connection so
            # a blocked reader is interrupted immediately.
            self._finish_transport(reusable=False)

    def _send_or_close(self, payload: dict, *, close_send: bool = False) -> None:
        """Send a control/text payload, mapping a dead connection to
        ``StreamClosedError``.

        From the caller's perspective a connection that died mid-stream is
        the same condition as sending after ``end()`` — the stream is closed
        for sending — so both surface the same exception type and existing
        handlers cover both.  The terminal error event still arrives through
        the reader path.
        """
        with self._transport_lock:
            self._check_send_open()
            if close_send:
                self._mark_send_closed()
            try:
                self._conn.settimeout(self._adapter.connect_timeout)
                ws_send_json(self._conn, payload)
            except (OSError, RawWebSocketError) as exc:
                self._mark_send_closed()
                self._finish_transport(reusable=False)
                raise StreamClosedError(
                    f"stream session {self.session_id} connection closed while sending"
                ) from exc
