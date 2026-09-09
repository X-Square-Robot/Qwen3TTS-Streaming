"""Thread-safe requests for engine-owned, same-boundary state migration."""

from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock
from typing import Callable

from .speech_state import SpeechStateContractError


@dataclass(frozen=True, slots=True)
class MigrationRequest:
    session_id: str
    segment_idx: int
    expected_attempt_id: int
    expected_allocation_epoch: int
    max_tensor_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise SpeechStateContractError("migration session id is empty")
        for name in (
            "segment_idx", "expected_attempt_id", "expected_allocation_epoch",
            "max_tensor_bytes",
        ):
            value = getattr(self, name)
            minimum = 1 if name == "expected_allocation_epoch" else 0
            if type(value) is not int or value < minimum:
                raise SpeechStateContractError(f"invalid migration {name}")


class SpeechStateMigrationQueue:
    """Only metadata crosses this queue; tensor payloads never leave its owner."""

    def __init__(self, *, max_pending: int) -> None:
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._lock = Lock()
        self._max_pending = max_pending
        self._pending: list[tuple[MigrationRequest, Future]] = []
        self._closed = False

    def submit(self, request: MigrationRequest) -> Future:
        future: Future = Future()
        with self._lock:
            if self._closed:
                future.set_exception(SpeechStateContractError("migration queue is closed"))
            elif len(self._pending) >= self._max_pending:
                future.set_exception(SpeechStateContractError("migration queue is full"))
            else:
                self._pending.append((request, future))
        return future

    def drain(self, migrate: Callable[[MigrationRequest], None]) -> None:
        """Called only on the engine thread at a completed step boundary."""
        with self._lock:
            pending, self._pending = self._pending, []
        for request, future in pending:
            if not future.set_running_or_notify_cancel():
                continue
            try:
                migrate(request)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)

    def close(self, reason: str) -> None:
        with self._lock:
            self._closed = True
            pending, self._pending = self._pending, []
        for _, future in pending:
            if future.set_running_or_notify_cancel():
                future.set_exception(SpeechStateContractError(reason))
