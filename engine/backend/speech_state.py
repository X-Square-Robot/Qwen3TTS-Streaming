"""Backend-side speech-state adapter boundary.

The adapter is intentionally inert in the first release.  It gives a future
model implementation one narrow place to capture/restore state while keeping
CUDA tensors, KV pools, RNG state, and lifecycle fences on the engine thread.
No caller should infer support from ``WAIT_TEXT`` or from the presence of a
method: capability advertisement is explicit and fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from ..core.speech_state import (
    SpeechStateCapability,
    SpeechStateHandle,
    SpeechStateOperation,
    SpeechStateTransfer,
)


class SpeechStatePhase(str, Enum):
    """Engine-thread phase at which an adapter is invoked."""

    CAPTURE = "capture"
    RESTORE = "restore"


class SpeechStateContractError(ValueError):
    """A handle/context pair cannot be safely used for state transfer."""


@dataclass(frozen=True, slots=True)
class SegmentRuntimeMetadata:
    """Detached engine-thread metadata required beside tensor state.

    This deliberately excludes slot pointers, cursor plans, handles and
    timestamps. Plans and handles have their own ownership contracts; wall
    clock fields must be recomputed on admission rather than replayed.
    """

    session_id: str
    segment_idx: int
    state: str
    input_complete: bool
    trailing_idx: int
    text_tokens_consumed: int
    decode_start_frame: int
    pending_token_ids: tuple[int, ...]
    eos_trailing_added: bool
    first_raw_audio_sent: bool
    loop_token: int
    loop_run: int
    loop_max_run: int
    loop_suspect_count: int
    loop_recovery_count: int
    retry_idx: int
    audio_frames_seen: int
    audible_frame_seen: bool
    cache_hit: bool
    cache_tokens_reused: int
    max_decode_batch: int
    cursor_plan_revision: int


@dataclass(frozen=True, slots=True)
class SpeechStateSnapshotBundle:
    """Engine-thread assembly contract for one exact-transfer snapshot.

    The fields are backend-owned payloads produced by the slot/pool/arena
    primitives. This type performs identity validation only; it does not
    clone tensors, assert a CUDA fence, or advertise runtime capability.
    """

    source_session_id: str
    source_segment_idx: int
    source_slot_id: int
    source_allocation_epoch: int
    segment_metadata: SegmentRuntimeMetadata
    slot_payload: Any
    pooled_talker_payload: Any
    pooled_c2w_payload: Any
    c2w_arena_payload: Any
    # Physical pool keys may include the segment, unlike the logical session.
    source_slot_session_id: str = ""

    def __post_init__(self) -> None:
        source_session_id = str(self.source_session_id or "").strip()
        if not source_session_id:
            raise SpeechStateContractError("snapshot source session is empty")
        slot_session_id = str(self.source_slot_session_id or "").strip()
        if not slot_session_id:
            slot_session_id = source_session_id
        object.__setattr__(self, "source_session_id", source_session_id)
        object.__setattr__(self, "source_slot_session_id", slot_session_id)
        for name, value, minimum in (
            ("source_segment_idx", self.source_segment_idx, 0),
            ("source_slot_id", self.source_slot_id, 0),
            ("source_allocation_epoch", self.source_allocation_epoch, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise SpeechStateContractError(f"invalid snapshot {name}")
        if not isinstance(self.segment_metadata, SegmentRuntimeMetadata):
            raise SpeechStateContractError("snapshot segment metadata is invalid")
        if self.segment_metadata.session_id != self.source_session_id:
            raise SpeechStateContractError("snapshot session metadata mismatch")
        if self.segment_metadata.segment_idx != self.source_segment_idx:
            raise SpeechStateContractError("snapshot segment metadata mismatch")


def validate_segment_runtime_metadata(
    metadata: SegmentRuntimeMetadata,
    *,
    session_id: str | None = None,
    segment_idx: int | None = None,
) -> None:
    """Validate detached metadata without mutating an engine segment."""
    if not isinstance(metadata, SegmentRuntimeMetadata):
        raise SpeechStateContractError("invalid segment runtime metadata type")
    if (
        not isinstance(metadata.session_id, str)
        or not metadata.session_id.strip()
        or isinstance(metadata.segment_idx, bool)
        or not isinstance(metadata.segment_idx, Integral)
        or metadata.segment_idx < 0
        or not isinstance(metadata.state, str)
        or not metadata.state
    ):
        raise SpeechStateContractError("invalid segment runtime identity")
    if session_id is not None and metadata.session_id != session_id:
        raise SpeechStateContractError("segment runtime session mismatch")
    if segment_idx is not None and metadata.segment_idx != segment_idx:
        raise SpeechStateContractError("segment runtime index mismatch")
    if not isinstance(metadata.pending_token_ids, tuple) or any(
        isinstance(token, bool) or not isinstance(token, Integral)
        for token in metadata.pending_token_ids
    ):
        raise SpeechStateContractError("pending token ids must be integers")
    boolean_fields = {
        "input_complete", "eos_trailing_added", "first_raw_audio_sent",
        "audible_frame_seen", "cache_hit",
    }
    for name in boolean_fields:
        if type(getattr(metadata, name)) is not bool:
            raise SpeechStateContractError(f"invalid segment runtime {name}")
    for name in SegmentRuntimeMetadata.__dataclass_fields__:
        if name in boolean_fields or name in {
            "session_id", "segment_idx", "state", "pending_token_ids",
        }:
            continue
        value = getattr(metadata, name)
        minimum = -1 if name in {"loop_token", "cursor_plan_revision"} else 0
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise SpeechStateContractError(f"invalid segment runtime {name}")


def capture_segment_runtime_metadata(segment: Any) -> SegmentRuntimeMetadata:
    """Copy only stable EngineSegment fields into detached metadata."""
    try:
        values = {
            name: getattr(segment, name)
            for name in SegmentRuntimeMetadata.__dataclass_fields__
        }
        values["pending_token_ids"] = tuple(values["pending_token_ids"])
    except (AttributeError, TypeError) as exc:
        raise SpeechStateContractError("segment runtime metadata is incomplete") from exc
    try:
        result = SegmentRuntimeMetadata(**values)
    except (TypeError, ValueError) as exc:
        raise SpeechStateContractError("invalid segment runtime metadata") from exc
    validate_segment_runtime_metadata(result)
    return result


def restore_segment_runtime_metadata(segment: Any, metadata: SegmentRuntimeMetadata) -> None:
    """Apply validated detached metadata without touching slot/plan ownership."""
    validate_segment_runtime_metadata(
        metadata,
        session_id=getattr(segment, "session_id", None),
        segment_idx=getattr(segment, "segment_idx", None),
    )
    if getattr(segment, "session_id", None) != metadata.session_id:
        raise SpeechStateContractError("segment runtime session mismatch")
    if getattr(segment, "segment_idx", None) != metadata.segment_idx:
        raise SpeechStateContractError("segment runtime index mismatch")
    values = {
        name: getattr(metadata, name)
        for name in SegmentRuntimeMetadata.__dataclass_fields__
        if name not in {"session_id", "segment_idx"}
    }
    for name, value in values.items():
        setattr(segment, name, list(value) if name == "pending_token_ids" else value)


def project_successor_runtime_metadata(
    source: SegmentRuntimeMetadata,
    *,
    target_segment_idx: int,
    pending_token_ids: tuple[int, ...] = (),
    input_complete: bool = False,
    retry_idx: int = 0,
) -> SegmentRuntimeMetadata:
    """Build fresh runtime metadata for a successor segment.

    Acoustic state may cross a segment boundary, but segment-local scheduler
    and output bookkeeping must not. This projection carries only the session
    identity and explicit successor input facts; source audio, loop, cache,
    retry and cursor counters start at their target defaults. Tensor payload
    restore remains a separate engine-thread step.
    """

    validate_segment_runtime_metadata(source)
    if isinstance(target_segment_idx, bool) or not isinstance(target_segment_idx, Integral):
        raise SpeechStateContractError("successor segment index must be an integer")
    if target_segment_idx <= source.segment_idx:
        raise SpeechStateContractError("successor segment index must follow source")
    if type(input_complete) is not bool:
        raise SpeechStateContractError("successor input_complete must be a boolean")
    if isinstance(retry_idx, bool) or not isinstance(retry_idx, Integral) or retry_idx < 0:
        raise SpeechStateContractError("successor retry index must be non-negative")
    if not isinstance(pending_token_ids, tuple) or any(
        isinstance(token, bool) or not isinstance(token, Integral)
        for token in pending_token_ids
    ):
        raise SpeechStateContractError("successor pending token ids must be integers")
    result = SegmentRuntimeMetadata(
        session_id=source.session_id,
        segment_idx=int(target_segment_idx),
        state="pending_prefill",
        input_complete=input_complete,
        trailing_idx=0,
        text_tokens_consumed=len(pending_token_ids),
        decode_start_frame=0,
        pending_token_ids=tuple(int(token) for token in pending_token_ids),
        eos_trailing_added=False,
        first_raw_audio_sent=False,
        loop_token=-1,
        loop_run=0,
        loop_max_run=0,
        loop_suspect_count=0,
        loop_recovery_count=0,
        retry_idx=int(retry_idx),
        audio_frames_seen=0,
        audible_frame_seen=False,
        cache_hit=False,
        cache_tokens_reused=0,
        max_decode_batch=0,
        cursor_plan_revision=-1,
    )
    validate_segment_runtime_metadata(result)
    return result


@dataclass(slots=True)
class _StoredSpeechState:
    handle: SpeechStateHandle
    payload: Any


class SpeechStateHandleStore:
    """Engine-thread-owned opaque payload registry.

    The registry deliberately has no serialization or cross-thread API. A
    successful ``consume`` removes the entry before returning its payload,
    making duplicate restore fail closed. Callers own quiescence ordering
    before inserting or consuming payloads.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _StoredSpeechState] = {}

    def put(self, handle: SpeechStateHandle, payload: Any) -> None:
        if handle.handle_id in self._entries:
            raise SpeechStateContractError("duplicate speech state handle")
        self._entries[handle.handle_id] = _StoredSpeechState(handle, payload)

    def create(
        self,
        *,
        generation: int,
        attempt_id: int,
        slot_allocation_epoch: int,
        owner_session_id: str,
        source_segment_idx: int,
        transfer: SpeechStateTransfer,
        model_fingerprint: str,
        runtime_fingerprint: str,
        payload: Any,
    ) -> SpeechStateHandle:
        handle = SpeechStateHandle(
            handle_id=uuid4().hex,
            generation=generation,
            attempt_id=attempt_id,
            slot_allocation_epoch=slot_allocation_epoch,
            owner_session_id=owner_session_id,
            source_segment_idx=source_segment_idx,
            transfer=transfer,
            model_fingerprint=model_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
        )
        self.put(handle, payload)
        return handle

    def consume(
        self,
        handle: SpeechStateHandle,
        context: "SpeechStateContext",
    ) -> Any:
        entry = self._entries.get(handle.handle_id)
        if entry is None:
            raise SpeechStateContractError("unknown or already consumed speech state handle")
        if entry.handle != handle:
            raise SpeechStateContractError("speech state handle metadata changed")
        validate_restore_context(handle, context)
        del self._entries[handle.handle_id]
        return entry.payload

    def release(self, handle: SpeechStateHandle) -> None:
        entry = self._entries.get(handle.handle_id)
        if entry is None:
            return
        if entry.handle != handle:
            raise SpeechStateContractError("speech state handle metadata changed")
        del self._entries[handle.handle_id]

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class SpeechStateContext:
    """Non-payload context for one backend-owned state operation.

    This object deliberately contains no tensors, CUDA events, queues, or
    serialized state.  It is safe to construct on the engine thread and is
    only an admission fence for an opaque handle.
    """

    operation: SpeechStateOperation
    phase: SpeechStatePhase
    session_id: str
    segment_idx: int
    attempt_id: int
    slot_allocation_epoch: int
    source_segment_idx: int = -1
    source_attempt_id: int = 0
    source_slot_allocation_epoch: int = 0
    model_fingerprint: str = ""
    runtime_fingerprint: str = ""
    source_generation: int | None = None

    def __post_init__(self) -> None:
        try:
            operation = (
                self.operation
                if isinstance(self.operation, SpeechStateOperation)
                else SpeechStateOperation(str(self.operation).strip().lower())
            )
        except (TypeError, ValueError) as exc:
            raise SpeechStateContractError("invalid speech state operation") from exc
        try:
            phase = (
                self.phase
                if isinstance(self.phase, SpeechStatePhase)
                else SpeechStatePhase(str(self.phase).strip().lower())
            )
        except (TypeError, ValueError) as exc:
            raise SpeechStateContractError("invalid speech state phase") from exc
        values = (
            ("segment_idx", self.segment_idx, 0),
            ("attempt_id", self.attempt_id, 0),
            ("slot_allocation_epoch", self.slot_allocation_epoch, 0),
            ("source_segment_idx", self.source_segment_idx, -1),
            ("source_attempt_id", self.source_attempt_id, 0),
            ("source_slot_allocation_epoch", self.source_slot_allocation_epoch, 0),
        )
        normalized: dict[str, int] = {}
        if self.source_generation is not None:
            values += (("source_generation", self.source_generation, 0),)
        for name, value, minimum in values:
            if isinstance(value, bool):
                raise SpeechStateContractError(f"{name} must be an integer")
            try:
                integer = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SpeechStateContractError(f"{name} must be an integer") from exc
            if integer != value or integer < minimum:
                raise SpeechStateContractError(f"invalid {name}")
            normalized[name] = integer
        session_id = str(self.session_id or "").strip()
        if not session_id:
            raise SpeechStateContractError("session_id must not be empty")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "session_id", session_id)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "model_fingerprint", str(self.model_fingerprint or ""))
        object.__setattr__(
            self,
            "runtime_fingerprint",
            str(self.runtime_fingerprint or ""),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "phase": self.phase.value,
            "session_id": self.session_id,
            "segment_idx": self.segment_idx,
            "attempt_id": self.attempt_id,
            "slot_allocation_epoch": self.slot_allocation_epoch,
            "source_segment_idx": self.source_segment_idx,
            "source_attempt_id": self.source_attempt_id,
            "source_slot_allocation_epoch": self.source_slot_allocation_epoch,
            "model_fingerprint": self.model_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "source_generation": self.source_generation,
        }


def validate_restore_context(
    handle: SpeechStateHandle,
    context: SpeechStateContext,
) -> None:
    """Fail closed before an adapter can apply an incompatible handle."""

    if context.phase is not SpeechStatePhase.RESTORE:
        raise SpeechStateContractError("restore requires RESTORE context")
    if handle.owner_session_id != context.session_id:
        raise SpeechStateContractError("speech state belongs to another session")
    if context.source_segment_idx < 0:
        raise SpeechStateContractError("restore requires a source segment")
    if handle.source_segment_idx != context.source_segment_idx:
        raise SpeechStateContractError("speech state source segment mismatch")
    if handle.attempt_id != context.source_attempt_id:
        raise SpeechStateContractError("speech state source attempt mismatch")
    if handle.slot_allocation_epoch != context.source_slot_allocation_epoch:
        raise SpeechStateContractError("speech state source slot epoch mismatch")
    if handle.transfer is SpeechStateTransfer.NONE:
        raise SpeechStateContractError("speech state handle has no transfer class")
    if context.source_generation is None:
        raise SpeechStateContractError("restore requires a source generation")
    if handle.generation != context.source_generation:
        raise SpeechStateContractError("speech state source generation mismatch")
    if (
        not context.model_fingerprint.strip()
        or not handle.model_fingerprint.strip()
        or context.model_fingerprint != handle.model_fingerprint
    ):
        raise SpeechStateContractError("speech state model fingerprint mismatch")
    if (
        not context.runtime_fingerprint.strip()
        or not handle.runtime_fingerprint.strip()
        or context.runtime_fingerprint != handle.runtime_fingerprint
    ):
        raise SpeechStateContractError("speech state runtime fingerprint mismatch")


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
    "SegmentRuntimeMetadata",
    "SpeechStateContext",
    "SpeechStateContractError",
    "SpeechStatePhase",
    "SpeechStateSnapshotBundle",
    "SpeechStateAdapter",
    "capability_from_adapter",
    "capture_segment_runtime_metadata",
    "coerce_speech_state_adapter",
    "project_successor_runtime_metadata",
    "restore_segment_runtime_metadata",
    "validate_segment_runtime_metadata",
    "validate_restore_context",
]
