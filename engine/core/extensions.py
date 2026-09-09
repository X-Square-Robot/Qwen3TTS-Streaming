"""Optional lifecycle extensions for method-layer policies.

The engine owns lifecycle ordering and GPU tensors; an extension owns its
method state.  The default is an empty, dependency-free extension set so the
official Qwen3-TTS path remains unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PolicyFactory = Callable[[str, Any], Any]
FailureCallback = Callable[[BaseException], None]

_CURSOR_RECURRENT_FIELDS = (
    "cursor_mu",
    "cursor_frames_since_advance",
    "cursor_delta_history",
    "cursor_conv_history",
    "cursor_last_trunk_input",
    "cursor_seen_frames",
)


@dataclass(frozen=True, slots=True)
class EngineExtensions:
    """Factories for optional session-scoped frontend/backend policies."""

    name: str = ""
    version: str = ""
    commitment_factory: Optional[PolicyFactory] = None
    continuity_factory: Optional[PolicyFactory] = None


@dataclass(frozen=True, slots=True)
class CursorContinuationState:
    """Detached recurrent state for the cursor fused into the TRT graph."""

    cursor_mu: Any
    cursor_frames_since_advance: Any
    cursor_delta_history: Any
    cursor_conv_history: Any
    cursor_last_trunk_input: Any
    cursor_seen_frames: Any

    def restore_into(self, slot: Any) -> None:
        import torch

        prepared = []
        for name in _CURSOR_RECURRENT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"cursor continuation field {name} is not a tensor")
            current = getattr(slot, name, None)
            if isinstance(current, torch.Tensor) and (
                tuple(current.shape) != tuple(value.shape)
                or current.dtype != value.dtype
                or current.device != value.device
            ):
                raise ValueError(f"cursor continuation field {name} shape/device mismatch")
            # Prepare every detached value before changing the target. A later
            # ABI failure must not leave an earlier recurrent field installed.
            prepared.append((name, value.detach().clone()))
        for name, value in prepared:
            setattr(slot, name, value)


class CursorContextOverlay:
    """Expose engine-owned cursor state alongside an external C2W context.

    X2 method policies own their acoustic continuation type and may use a
    frozen/slot-based object that cannot be extended in place.  The engine
    only needs the narrow C2W attributes plus ``cursor_state`` during restore,
    so delegation keeps the external policy contract unchanged.
    """

    __slots__ = ("_base", "cursor_state")

    def __init__(self, base: Any, cursor_state: CursorContinuationState):
        self._base = base
        self.cursor_state = cursor_state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def overlay_cursor_state(
    context: Any,
    cursor_state: CursorContinuationState | None,
) -> Any:
    """Attach engine-owned cursor state without mutating a policy object."""

    if context is None or cursor_state is None:
        return context
    if (
        isinstance(context, CursorContextOverlay)
        and context.cursor_state is cursor_state
    ):
        return context
    return CursorContextOverlay(context, cursor_state)


@dataclass(frozen=True, slots=True)
class C2WContinuationState:
    """Detached Code2Wav state passed to a successor policy."""

    c2w_kv: Any
    c2w_conv_states: tuple[Any, ...]
    c2w_transconv_states: tuple[Any, ...]
    frame_idx: int
    cursor_state: CursorContinuationState | None = None

    @property
    def c2w_conv(self) -> tuple[Any, ...]:
        """Compatibility alias used by the X2 method-layer contract."""

        return self.c2w_conv_states

    @property
    def c2w_transconv(self) -> tuple[Any, ...]:
        """Compatibility alias used by the X2 method-layer contract."""

        return self.c2w_transconv_states


def capture_c2w_state(slot: Any, kv_pool: Any = None) -> Optional[C2WContinuationState]:
    """Clone a complete C2W state without exposing a mutable slot or pool view."""

    try:
        if getattr(slot, "c2w_kv", None) is not None:
            c2w_kv = slot.c2w_kv.detach().clone()
        elif getattr(slot, "c2w_pooled", False) and kv_pool is not None:
            reader = getattr(kv_pool, "read_c2w_right_aligned", None)
            if not callable(reader):
                return None
            c2w_kv = reader(slot.slot_id, int(getattr(slot, "c2w_len", 0)))
        else:
            return None
        conv = tuple(
            tensor.detach().clone()
            for tensor in (getattr(slot, "c2w_conv_states", None) or ())
        )
        transconv = tuple(
            tensor.detach().clone()
            for tensor in (getattr(slot, "c2w_transconv_states", None) or ())
        )
        if not conv or not transconv:
            return None
        return C2WContinuationState(
            c2w_kv=c2w_kv,
            c2w_conv_states=conv,
            c2w_transconv_states=transconv,
            frame_idx=int(getattr(slot, "frame_idx", 0)),
            cursor_state=capture_cursor_state(slot),
        )
    except Exception:
        logger.exception("Could not capture optional Code2Wav continuation state")
        return None


def capture_cursor_state(slot: Any) -> CursorContinuationState | None:
    """Clone cursor recurrent tensors when a slot has a complete cursor ABI."""

    try:
        values = {
            name: getattr(slot, name, None)
            for name in _CURSOR_RECURRENT_FIELDS
        }
        if any(not hasattr(value, "detach") for value in values.values()):
            return None
        return CursorContinuationState(
            **{name: value.detach().clone() for name, value in values.items()}
        )
    except Exception:
        logger.exception("Could not capture optional native cursor continuation state")
        return None


def invoke_policy(
    policy: Any,
    method: str,
    *args: Any,
    default: Any = None,
    on_failure: Optional[FailureCallback] = None,
    **kwargs: Any,
) -> Any:
    """Invoke an optional callback and fail closed on policy errors."""

    if policy is None:
        return default
    callback = getattr(policy, method, None)
    if not callable(callback):
        return default
    try:
        return callback(*args, **kwargs)
    except Exception as exc:
        logger.exception("Optional policy callback failed: %s", method)
        if on_failure is not None:
            try:
                on_failure(exc)
            except Exception:
                logger.exception("Optional policy failure callback failed: %s", method)
        invalidate = getattr(policy, "invalidate", None)
        if method != "invalidate" and callable(invalidate):
            try:
                invalidate(f"callback_failed:{method}")
            except Exception:
                logger.exception("Optional policy invalidation failed")
        return default


__all__ = [
    "CursorContextOverlay",
    "CursorContinuationState",
    "C2WContinuationState",
    "EngineExtensions",
    "capture_c2w_state",
    "capture_cursor_state",
    "invoke_policy",
    "overlay_cursor_state",
]
