"""Detached snapshots for standalone slots, not an engine handoff adapter.

The caller must quiesce decode and complete all state writes before capture.
Pool/arena extraction, engine-segment metadata, handle lifetime and admission
are intentionally outside this primitive; it does not advertise a capability.
"""

from __future__ import annotations

from dataclasses import fields
from numbers import Integral
import time

import torch

from .kv_cache_pool import SlotKVState
from .speech_state import SpeechStateContractError


_STORAGE_FIELDS = frozenset({
    "slot_id", "is_free", "last_active_time", "allocation_epoch",
    "talker_kv", "past_len", "c2w_kv", "c2w_len", "c2w_pooled",
    "c2w_conv_states", "c2w_transconv_states", "_c2w_conv_write",
    "_c2w_transconv_write", "c2w_arena_backed", "c2w_write_in_a",
})

_OWNERSHIP_FIELDS = frozenset({
    "slot_id", "is_free", "last_active_time", "allocation_epoch",
    "c2w_pooled", "c2w_arena_backed",
})

# Method-layer extension state is owned by the session policy, not by the
# generic backend snapshot. Copying an arbitrary policy object here would
# violate the tensor/handle contract and can retain stale bridge state.
_EXTENSION_FIELDS = frozenset({"extension_state"})


def _size(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, torch.Generator):
        return _size(value.get_state())
    if isinstance(value, list):
        return sum(_size(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return 0
    raise SpeechStateContractError(f"unsupported slot state type: {type(value).__name__}")


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, torch.Generator):
        generator = torch.Generator(device=value.device)
        generator.set_state(value.get_state().clone())
        return generator
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise SpeechStateContractError(f"{name} must be a nonnegative integer")
    return int(value)


class StandaloneSlotSnapshot:
    """Independent same-device copy of one non-pooled logical segment.

    Restore creates an unregistered slot object, never mutates a live slot or
    reduces its KV length. Adoption by a pool and same-segment resume still
    require the separate engine-thread lifecycle implementation.
    """

    def __init__(self, slot: SlotKVState, *, max_tensor_bytes: int) -> None:
        budget = _nonnegative_int(max_tensor_bytes, "max_tensor_bytes")
        if slot.is_free:
            raise SpeechStateContractError("cannot snapshot a free slot")
        if slot.c2w_pooled or slot.c2w_arena_backed:
            raise SpeechStateContractError("pooled/arena slot requires pool-aware capture")
        if slot.c2w_len < 0 or (slot.c2w_len > 0 and slot.c2w_kv is None):
            raise SpeechStateContractError("C2W KV does not contain the valid history")
        if slot.past_len < 0:
            raise SpeechStateContractError("invalid Talker past length")
        if slot.past_len and (
            slot.talker_kv is None
            or slot.talker_kv.ndim != 5
            or slot.talker_kv.shape[3] != slot.past_len
        ):
            raise SpeechStateContractError("Talker KV does not contain the valid history")
        values = {
            item.name: getattr(slot, item.name)
            for item in fields(SlotKVState)
            if item.name not in _EXTENSION_FIELDS
        }
        # Count before allocating: malformed or oversized input leaves both
        # source and snapshot storage untouched. Shared views count separately.
        self._tensor_bytes = sum(_size(value) for value in values.values())
        if self._tensor_bytes > budget:
            raise SpeechStateContractError("slot snapshot exceeds tensor byte budget")
        self._values = {name: _clone(value) for name, value in values.items()}

    @property
    def tensor_bytes(self) -> int:
        """Tensor/RNG bytes, excluding Python object/container overhead."""
        return self._tensor_bytes

    def restore(self, *, slot_id: int) -> SlotKVState:
        target_id = _nonnegative_int(slot_id, "slot_id")
        values = {name: _clone(value) for name, value in self._values.items()}
        values["slot_id"] = target_id
        values["last_active_time"] = time.monotonic()
        return SlotKVState(**values)

    def restore_into(self, slot: SlotKVState) -> None:
        """Apply a standalone payload to an already allocated local slot."""
        if slot.is_free:
            raise SpeechStateContractError("cannot restore into a free slot")
        if slot.c2w_pooled or slot.c2w_arena_backed:
            raise SpeechStateContractError(
                "standalone payload requires a non-pooled, non-arena target"
            )
        values = {name: _clone(value) for name, value in self._values.items()}
        values["slot_id"] = slot.slot_id
        values["allocation_epoch"] = slot.allocation_epoch
        values["is_free"] = False
        values["last_active_time"] = time.monotonic()
        for name, value in values.items():
            setattr(slot, name, value)


class SlotAuxiliarySnapshot:
    """Detached non-pool state for a pooled/arena-backed slot.

    KV tensors, C2W arenas, ownership fields and storage parity are excluded;
    those must be captured by their owner-specific pool primitives. This
    snapshot carries the remaining cursor, text, sampling and decode state.
    """

    def __init__(self, slot: SlotKVState, *, max_tensor_bytes: int) -> None:
        budget = _nonnegative_int(max_tensor_bytes, "max_tensor_bytes")
        if slot.is_free:
            raise SpeechStateContractError("cannot snapshot a free slot")
        values = {
            item.name: getattr(slot, item.name)
            for item in fields(SlotKVState)
            if item.name not in _STORAGE_FIELDS | _EXTENSION_FIELDS
        }
        self._tensor_bytes = sum(_size(value) for value in values.values())
        if self._tensor_bytes > budget:
            raise SpeechStateContractError("slot auxiliary snapshot exceeds tensor byte budget")
        self._values = {name: _clone(value) for name, value in values.items()}

    @property
    def tensor_bytes(self) -> int:
        return self._tensor_bytes

    def restore_into(self, slot: SlotKVState) -> None:
        """Apply detached auxiliary fields to an already allocated target."""
        if slot.is_free:
            raise SpeechStateContractError("cannot restore into a free slot")
        values = {name: _clone(value) for name, value in self._values.items()}
        for name, value in values.items():
            setattr(slot, name, value)


class SlotOwnedSnapshot:
    """Detached slot fields whose storage is owned by the slot itself.

    Pool rows and arena rows are supplied separately. Scalar valid lengths are
    retained here because they are the metadata that interprets those rows.
    The ownership flags and physical row identity are intentionally excluded.
    """

    def __init__(
        self,
        slot: SlotKVState,
        *,
        pooled_talker: bool,
        pooled_c2w: bool,
        arena_backed: bool,
        max_tensor_bytes: int,
    ) -> None:
        budget = _nonnegative_int(max_tensor_bytes, "max_tensor_bytes")
        if slot.is_free:
            raise SpeechStateContractError("cannot snapshot a free slot")
        excluded = set(_OWNERSHIP_FIELDS)
        if pooled_talker:
            excluded.add("talker_kv")
        if pooled_c2w:
            excluded.add("c2w_kv")
        if arena_backed:
            excluded.update({
                "c2w_conv_states", "c2w_transconv_states",
                "_c2w_conv_write", "_c2w_transconv_write",
                "c2w_write_in_a",
            })
        values = {
            item.name: getattr(slot, item.name)
            for item in fields(SlotKVState)
            if item.name not in excluded | _EXTENSION_FIELDS
        }
        if slot.past_len < 0 or slot.c2w_len < 0:
            raise SpeechStateContractError("invalid KV history length")
        if slot.past_len and not pooled_talker and (
            slot.talker_kv is None
            or slot.talker_kv.ndim != 5
            or slot.talker_kv.shape[3] != slot.past_len
        ):
            raise SpeechStateContractError("Talker KV does not contain the valid history")
        if slot.c2w_len and not pooled_c2w and (
            slot.c2w_kv is None
            or slot.c2w_kv.ndim != 5
            or slot.c2w_kv.shape[3] != slot.c2w_len
        ):
            raise SpeechStateContractError("C2W KV does not contain the valid history")
        self._tensor_bytes = sum(_size(value) for value in values.values())
        if self._tensor_bytes > budget:
            raise SpeechStateContractError(
                "slot-owned snapshot exceeds tensor byte budget"
            )
        self._values = {name: _clone(value) for name, value in values.items()}

    @property
    def tensor_bytes(self) -> int:
        return self._tensor_bytes

    def restore_into(self, slot: SlotKVState) -> None:
        if slot.is_free:
            raise SpeechStateContractError("cannot restore into a free slot")
        for name, value in ((name, _clone(value)) for name, value in self._values.items()):
            setattr(slot, name, value)
