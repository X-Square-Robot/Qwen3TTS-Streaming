"""Typed contract for model-owned speech runtime state.

The streaming text/TN path and acoustic state handoff are deliberately
separate concerns.  This module only describes what a model/backend can do;
it does not contain tensors, CUDA objects, or scheduling policy.  The async
``Session`` may retain :class:`SpeechStateCapability`, while an actual handle
is owned by the engine thread/backend (see :mod:`engine.backend.speech_state`).

Keeping this contract small is important for backwards compatibility: models
which have not proved an exact Talker + Code2Wav handoff advertise the default
disabled capability and continue to use the existing inference path unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


SPEECH_STATE_PROTOCOL_VERSION = "qwen.speech-state.v1"


class SpeechStateHandleKind(str, Enum):
    """How a backend represents a captured state to its owner."""

    NONE = "none"
    OPAQUE = "opaque"


class SpeechStateOperation(str, Enum):
    """Operations that may be implemented by a model/backend adapter."""

    PAUSE_RESUME = "pause_resume"
    SEGMENT_HANDOFF = "segment_handoff"
    CONTEXT_ROLLOVER = "context_rollover"


class SpeechStateTransfer(str, Enum):
    """Accuracy class of a state transfer implementation."""

    NONE = "none"
    EXACT = "exact"
    TRAINED_APPROXIMATE = "trained_approximate"
    RECONSTRUCTION_APPROXIMATE = "reconstruction_approximate"




def _coerce_enum(value: Any, enum_type: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).strip().lower())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"invalid {field_name}: {value!r}")


def _coerce_nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid {field_name}: {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"invalid {field_name}: {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def _coerce_operations(value: Any) -> tuple[SpeechStateOperation, ...]:
    """Parse a list or a compact ``{operation: bool}`` capability shape."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        values = [
            key
            for key, enabled in value.items()
            if _coerce_bool(enabled, field_name="operation")
        ]
    elif isinstance(value, (str, SpeechStateOperation)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise ValueError(f"invalid operations: {value!r}") from exc

    parsed: set[SpeechStateOperation] = set()
    for item in values:
        parsed.add(
            _coerce_enum(item, SpeechStateOperation, field_name="operation")
        )
    # Enum declaration order is the canonical wire order.  This prevents
    # capability JSON from changing merely because a caller used a set.
    return tuple(
        operation for operation in SpeechStateOperation if operation in parsed
    )


@dataclass(frozen=True, slots=True)
class SpeechStateCapability:
    """Read-only model/backend capability descriptor.

    ``supported=False`` is the safe default.  A supported descriptor must use
    an opaque handle and identify the transfer accuracy class; otherwise it is
    rejected/falls back to :meth:`disabled` by :meth:`from_mapping`.
    """

    supported: bool = False
    protocol_version: str = SPEECH_STATE_PROTOCOL_VERSION
    handle_kind: SpeechStateHandleKind = SpeechStateHandleKind.NONE
    operations: tuple[SpeechStateOperation, ...] = field(default_factory=tuple)
    transfer: SpeechStateTransfer = SpeechStateTransfer.NONE
    max_handle_bytes: int = 0

    def __post_init__(self) -> None:
        supported = _coerce_bool(self.supported, field_name="supported")
        protocol_version = str(self.protocol_version or "").strip()
        if not supported:
            # Ignore all stale/unknown detail fields on a disabled descriptor.
            # This makes direct construction just as fail-closed as
            # ``from_mapping`` and avoids turning optional metadata into a
            # startup failure.
            object.__setattr__(
                self,
                "supported",
                False,
            )
            object.__setattr__(
                self,
                "protocol_version",
                SPEECH_STATE_PROTOCOL_VERSION,
            )
            object.__setattr__(self, "handle_kind", SpeechStateHandleKind.NONE)
            object.__setattr__(self, "operations", ())
            object.__setattr__(self, "transfer", SpeechStateTransfer.NONE)
            object.__setattr__(self, "max_handle_bytes", 0)
            return
        if not protocol_version:
            raise ValueError("protocol_version must not be empty")
        handle_kind = _coerce_enum(
            self.handle_kind, SpeechStateHandleKind, field_name="handle_kind"
        )
        transfer = _coerce_enum(
            self.transfer, SpeechStateTransfer, field_name="transfer"
        )
        operations = _coerce_operations(self.operations)
        max_handle_bytes = _coerce_nonnegative_int(
            self.max_handle_bytes, field_name="max_handle_bytes"
        )

        if protocol_version != SPEECH_STATE_PROTOCOL_VERSION:
            raise ValueError(f"unsupported speech-state protocol: {protocol_version!r}")
        if handle_kind is not SpeechStateHandleKind.OPAQUE:
            raise ValueError("supported speech state requires an opaque handle")
        if not operations:
            raise ValueError("supported speech state requires at least one operation")
        if transfer is SpeechStateTransfer.NONE:
            raise ValueError("supported speech state requires a transfer class")

        object.__setattr__(self, "supported", supported)
        object.__setattr__(self, "protocol_version", protocol_version)
        object.__setattr__(self, "handle_kind", handle_kind)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "transfer", transfer)
        object.__setattr__(self, "max_handle_bytes", max_handle_bytes)

    @classmethod
    def disabled(cls) -> "SpeechStateCapability":
        """Return the canonical fail-closed descriptor."""

        return cls()

    @classmethod
    def from_mapping(cls, value: Any) -> "SpeechStateCapability":
        """Parse untrusted manifest/config metadata fail-closed.

        Capability discovery must never make startup fail or accidentally
        enable state handoff because of malformed optional metadata.  Invalid
        mappings therefore return the disabled descriptor.
        """

        if not isinstance(value, Mapping):
            return cls.disabled()
        try:
            supported_value = value.get("supported", value.get("enabled", False))
            supported = _coerce_bool(supported_value, field_name="supported")
            if not supported:
                return cls.disabled()

            handle_value = value.get(
                "handle_kind", value.get("handle_type", "opaque")
            )
            operations = value.get("operations", ())
            transfer = value.get("transfer", value.get("accuracy", "none"))
            return cls(
                supported=True,
                protocol_version=str(
                    value.get("protocol_version", SPEECH_STATE_PROTOCOL_VERSION)
                    or SPEECH_STATE_PROTOCOL_VERSION
                ),
                handle_kind=handle_value,
                operations=operations,
                transfer=transfer,
                max_handle_bytes=value.get("max_handle_bytes", 0),
            )
        except Exception:
            # Optional capability metadata must never make model startup fail;
            # malformed/custom mapping objects are treated like unsupported
            # models as well.
            return cls.disabled()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stable public descriptor (no backend state)."""

        return {
            "supported": self.supported,
            "protocol_version": self.protocol_version,
            "handle_kind": self.handle_kind.value,
            "operations": [operation.value for operation in self.operations],
            "transfer": self.transfer.value,
            "max_handle_bytes": self.max_handle_bytes,
        }

    def supports(self, operation: SpeechStateOperation | str) -> bool:
        try:
            operation = _coerce_enum(
                operation, SpeechStateOperation, field_name="operation"
            )
        except ValueError:
            return False
        return self.supported and operation in self.operations

    @property
    def supports_pause_resume(self) -> bool:
        return self.supports(SpeechStateOperation.PAUSE_RESUME)

    @property
    def supports_segment_handoff(self) -> bool:
        return self.supports(SpeechStateOperation.SEGMENT_HANDOFF)

    @property
    def supports_context_rollover(self) -> bool:
        return self.supports(SpeechStateOperation.CONTEXT_ROLLOVER)


def coerce_speech_state_capability(value: Any = None) -> SpeechStateCapability:
    """Normalize an optional capability at a component boundary.

    Typed descriptors pass through unchanged; JSON-like metadata is parsed
    fail-closed.  This keeps Session/EngineLoop constructors tolerant of old
    callers without duplicating policy in each layer.
    """

    if isinstance(value, SpeechStateCapability):
        return value
    return SpeechStateCapability.from_mapping(value)


@dataclass(frozen=True, slots=True)
class SpeechStateHandle:
    """Backend-owned opaque state identity.

    The ID is intentionally excluded from normal metadata serialization.  It
    can be a process-local table key, but must never contain tensors or be
    treated as a portable checkpoint.  ``metadata(include_id=True)`` exists
    only for engine-thread diagnostics/tests.
    """

    handle_id: str = field(repr=False)
    protocol_version: str = SPEECH_STATE_PROTOCOL_VERSION
    generation: int = 0
    attempt_id: int = 0
    slot_allocation_epoch: int = 0
    owner_session_id: str = ""
    source_segment_idx: int = -1
    transfer: SpeechStateTransfer = SpeechStateTransfer.NONE
    model_fingerprint: str = ""
    runtime_fingerprint: str = ""

    def __post_init__(self) -> None:
        handle_id = str(self.handle_id or "").strip()
        if not handle_id:
            raise ValueError("handle_id must not be empty")
        protocol_version = str(self.protocol_version or "").strip()
        if not protocol_version:
            raise ValueError("protocol_version must not be empty")
        if protocol_version != SPEECH_STATE_PROTOCOL_VERSION:
            raise ValueError(f"unsupported speech-state protocol: {protocol_version!r}")
        generation = _coerce_nonnegative_int(self.generation, field_name="generation")
        attempt_id = _coerce_nonnegative_int(self.attempt_id, field_name="attempt_id")
        slot_allocation_epoch = _coerce_nonnegative_int(
            self.slot_allocation_epoch, field_name="slot_allocation_epoch"
        )
        if isinstance(self.source_segment_idx, bool):
            raise ValueError("source_segment_idx must be an integer")
        if isinstance(self.source_segment_idx, float) and not self.source_segment_idx.is_integer():
            raise ValueError("source_segment_idx must be an integer")
        try:
            source_segment_idx = int(self.source_segment_idx)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("source_segment_idx must be an integer") from exc
        if source_segment_idx < -1:
            raise ValueError("source_segment_idx must be >= -1")
        transfer = _coerce_enum(
            self.transfer, SpeechStateTransfer, field_name="transfer"
        )
        object.__setattr__(self, "handle_id", handle_id)
        object.__setattr__(self, "protocol_version", protocol_version)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "slot_allocation_epoch", slot_allocation_epoch)
        object.__setattr__(self, "owner_session_id", str(self.owner_session_id or ""))
        object.__setattr__(self, "source_segment_idx", source_segment_idx)
        object.__setattr__(self, "transfer", transfer)
        object.__setattr__(self, "model_fingerprint", str(self.model_fingerprint or ""))
        object.__setattr__(self, "runtime_fingerprint", str(self.runtime_fingerprint or ""))

    def metadata(self, *, include_id: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "generation": self.generation,
            "attempt_id": self.attempt_id,
            "slot_allocation_epoch": self.slot_allocation_epoch,
            "owner_session_id": self.owner_session_id,
            "source_segment_idx": self.source_segment_idx,
            "transfer": self.transfer.value,
            "model_fingerprint": self.model_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
        }
        if include_id:
            result["handle_id"] = self.handle_id
        return result

    def to_dict(self, *, include_id: bool = False) -> dict[str, Any]:
        return self.metadata(include_id=include_id)

    def __repr__(self) -> str:
        return (
            "SpeechStateHandle(<opaque>, "
            f"protocol_version={self.protocol_version!r}, "
            f"generation={self.generation}, "
            f"attempt_id={self.attempt_id}, "
            f"slot_allocation_epoch={self.slot_allocation_epoch}, "
            f"owner_session_id={self.owner_session_id!r}, "
            f"source_segment_idx={self.source_segment_idx}, "
            f"transfer={self.transfer.value!r})"
        )


# More explicit spelling for callers that want to emphasize that the value is
# not a serializable checkpoint.  Keep the short name as the primary API.
OpaqueSpeechStateHandle = SpeechStateHandle


__all__ = [
    "SPEECH_STATE_PROTOCOL_VERSION",
    "OpaqueSpeechStateHandle",
    "SpeechStateCapability",
    "coerce_speech_state_capability",
    "SpeechStateHandle",
    "SpeechStateHandleKind",
    "SpeechStateOperation",
    "SpeechStateTransfer",
]
