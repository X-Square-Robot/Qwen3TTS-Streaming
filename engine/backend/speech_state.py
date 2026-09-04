"""Backend-side speech-state adapter boundary.

The adapter is intentionally inert in the first release.  It gives a future
model implementation one narrow place to capture/restore state while keeping
CUDA tensors, KV pools, RNG state, and lifecycle fences on the engine thread.
No caller should infer support from ``WAIT_TEXT`` or from the presence of a
method: capability advertisement is explicit and fail-closed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..core.speech_state import SpeechStateCapability, SpeechStateHandle


@runtime_checkable
class SpeechStateAdapter(Protocol):
    """Minimal model/backend hook for a future stateful implementation.

    Implementations must keep the payload behind the backend boundary.  The
    returned handle is an opaque process-local identity, not a tensor bundle or
    a portable checkpoint.
    """

    @property
    def capability(self) -> SpeechStateCapability:
        """Advertised operations for this model/runtime."""

    def capture(
        self,
        *,
        session_id: str,
        segment_idx: int,
        runtime_state: Any = None,
    ) -> SpeechStateHandle | None:
        """Capture at a backend-defined safe point, or return ``None``."""

    def restore(
        self,
        handle: SpeechStateHandle,
        *,
        session_id: str,
        segment_idx: int,
        runtime_state: Any = None,
    ) -> bool:
        """Restore a handle at a backend-defined launch boundary."""

    def release(self, handle: SpeechStateHandle) -> None:
        """Release backend storage associated with a handle."""


class NullSpeechStateAdapter:
    """Safe default used by official models and legacy test doubles."""

    capability = SpeechStateCapability.disabled()

    def capture(
        self,
        *,
        session_id: str,
        segment_idx: int,
        runtime_state: Any = None,
    ) -> None:
        return None

    def restore(
        self,
        handle: SpeechStateHandle,
        *,
        session_id: str,
        segment_idx: int,
        runtime_state: Any = None,
    ) -> bool:
        return False

    def release(self, handle: SpeechStateHandle) -> None:
        return None


def capability_from_adapter(adapter: Any) -> SpeechStateCapability:
    """Read an adapter capability without ever enabling malformed metadata."""

    if adapter is None:
        return SpeechStateCapability.disabled()
    try:
        value = getattr(adapter, "capability", None)
        # Accept a method as a small convenience for adapters that cannot
        # construct their descriptor until runtime; properties remain the
        # documented shape.
        if callable(value):
            value = value()
        if isinstance(value, SpeechStateCapability):
            return value
        return SpeechStateCapability.from_mapping(value)
    except Exception:
        return SpeechStateCapability.disabled()


def coerce_speech_state_adapter(adapter: Any = None) -> SpeechStateAdapter:
    """Return a valid adapter, otherwise a no-op fail-closed adapter.

    This helper is used at construction boundaries so old executor stubs and
    official models need no changes.  A malformed adapter is never allowed to
    advertise a capability or crash startup.
    """

    if adapter is None:
        return NullSpeechStateAdapter()
    methods = ("capture", "restore", "release")
    if not all(callable(getattr(adapter, name, None)) for name in methods):
        return NullSpeechStateAdapter()
    # A typed capability object is required on the retained adapter.  Wrapping
    # arbitrary mappings here would hide implementation errors and could make
    # a future capture call unsafe.
    try:
        capability = getattr(adapter, "capability", None)
        if callable(capability):
            capability = capability()
    except Exception:
        return NullSpeechStateAdapter()
    if not isinstance(capability, SpeechStateCapability):
        return NullSpeechStateAdapter()
    return adapter


__all__ = [
    "NullSpeechStateAdapter",
    "SpeechStateAdapter",
    "capability_from_adapter",
    "coerce_speech_state_adapter",
]
