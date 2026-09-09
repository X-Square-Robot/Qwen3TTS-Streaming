"""Typed model semantics for cross-segment acoustic state handoff."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .speech_state import SpeechStateTransfer


class SpeechStateCheckpointBoundary(str, Enum):
    """Model-defined points where a successor checkpoint may be published."""

    EXPLICIT_PAUSE = "explicit_pause"
    PRE_CODEC_EOS = "pre_codec_eos"
    POST_CODEC_EOS = "post_codec_eos"
    SEGMENT_END = "segment_end"


class SpeechStateSuccessorStart(str, Enum):
    """How the successor segment is allowed to enter acoustic decode."""

    READY_TO_DECODE = "ready_to_decode"
    MODEL_PREFILL = "model_prefill"
    HARD_BOUNDARY_PREFILL = "hard_boundary_prefill"


class SpeechStateCursorPolicy(str, Enum):
    """What happens to native cursor state when text ownership changes."""

    MIGRATE = "migrate"
    REANCHOR = "reanchor"
    DISABLE = "disable"


class SpeechStateBoundaryPhase(str, Enum):
    """Logical model phase associated with a published checkpoint."""

    TEXT_COMMIT = "text_commit"
    DECODE_FRAME = "decode_frame"
    CODEC_EOS = "codec_eos"
    SOFT_DRAIN = "soft_drain"


class SpeechStateBoundaryTokenPolicy(str, Enum):
    """Token rule that makes the checkpoint boundary reproducible."""

    EXACT_MODEL_TOKEN = "exact_model_token"
    TEXT_EOS = "text_eos"
    NO_EOS = "no_eos"


class SpeechStateSuccessorTextEntry(str, Enum):
    """How successor text is admitted after acoustic state transfer."""

    APPEND_TOKENS = "append_tokens"
    PREFILL_SUCCESSOR_TEXT = "prefill_successor_text"
    NO_TEXT_REPLAY = "no_text_replay"


class SpeechStateTalkerCarry(str, Enum):
    """How a successor receives Talker-side acoustic context."""

    NONE = "none"
    DIRECT_KV = "direct_kv"
    HIDDEN_TAIL_BRIDGE = "hidden_tail_bridge"


@dataclass(frozen=True, slots=True)
class SpeechStateModelContract:
    """Explicit X2 successor contract; declaration alone never enables it."""

    model_fingerprint: str
    checkpoint_id: str
    boundaries: tuple[SpeechStateCheckpointBoundary, ...]
    boundary_phase: SpeechStateBoundaryPhase
    boundary_token_policy: SpeechStateBoundaryTokenPolicy
    boundary_token_id: int | None
    successor_start: SpeechStateSuccessorStart
    successor_text_entry: SpeechStateSuccessorTextEntry
    cursor_policy: SpeechStateCursorPolicy
    transfer: SpeechStateTransfer
    retain_talker_kv: bool
    retain_code_predictor_state: bool
    retain_c2w_kv: bool
    retain_c2w_history: bool
    training_scope: str
    talker_carry: SpeechStateTalkerCarry = SpeechStateTalkerCarry.DIRECT_KV
    talker_hidden_tail: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpeechStateModelContract":
        """Parse a complete model-owned contract without enabling runtime state transfer."""

        if not isinstance(value, Mapping):
            raise ValueError("speech state model contract must be an object")
        required = {
            "model_fingerprint",
            "checkpoint_id",
            "boundaries",
            "boundary_phase",
            "boundary_token_policy",
            "boundary_token_id",
            "successor_start",
            "successor_text_entry",
            "cursor_policy",
            "transfer",
            "retain_talker_kv",
            "retain_code_predictor_state",
            "retain_c2w_kv",
            "retain_c2w_history",
            "training_scope",
            "talker_carry",
            "talker_hidden_tail",
        }
        missing = sorted(key for key in required if key not in value)
        if missing:
            raise ValueError(
                "speech state model contract is missing: " + ", ".join(missing)
            )
        boundaries = value["boundaries"]
        if isinstance(boundaries, (str, bytes)):
            raise ValueError("speech state model contract boundaries must be a list")
        try:
            boundaries = tuple(boundaries)
        except TypeError as exc:
            raise ValueError(
                "speech state model contract boundaries must be a list"
            ) from exc
        return cls(
            model_fingerprint=value["model_fingerprint"],
            checkpoint_id=value["checkpoint_id"],
            boundaries=boundaries,
            boundary_phase=value["boundary_phase"],
            boundary_token_policy=value["boundary_token_policy"],
            boundary_token_id=value["boundary_token_id"],
            successor_start=value["successor_start"],
            successor_text_entry=value["successor_text_entry"],
            cursor_policy=value["cursor_policy"],
            transfer=value["transfer"],
            retain_talker_kv=value["retain_talker_kv"],
            retain_code_predictor_state=value["retain_code_predictor_state"],
            retain_c2w_kv=value["retain_c2w_kv"],
            retain_c2w_history=value["retain_c2w_history"],
            training_scope=value["training_scope"],
            talker_carry=value["talker_carry"],
            talker_hidden_tail=value["talker_hidden_tail"],
        )

    def __post_init__(self) -> None:
        model_fingerprint = (
            self.model_fingerprint.strip()
            if isinstance(self.model_fingerprint, str)
            else ""
        )
        checkpoint_id = (
            self.checkpoint_id.strip()
            if isinstance(self.checkpoint_id, str)
            else ""
        )
        training_scope = (
            self.training_scope.strip()
            if isinstance(self.training_scope, str)
            else ""
        )
        if not model_fingerprint or not checkpoint_id or not training_scope:
            raise ValueError(
                "speech state model contract requires identity and training scope"
            )
        try:
            boundaries = tuple(
                item
                if isinstance(item, SpeechStateCheckpointBoundary)
                else SpeechStateCheckpointBoundary(str(item).strip().lower())
                for item in self.boundaries
            )
            successor_start = (
                self.successor_start
                if isinstance(self.successor_start, SpeechStateSuccessorStart)
                else SpeechStateSuccessorStart(str(self.successor_start).strip().lower())
            )
            cursor_policy = (
                self.cursor_policy
                if isinstance(self.cursor_policy, SpeechStateCursorPolicy)
                else SpeechStateCursorPolicy(str(self.cursor_policy).strip().lower())
            )
            transfer = (
                self.transfer
                if isinstance(self.transfer, SpeechStateTransfer)
                else SpeechStateTransfer(str(self.transfer).strip().lower())
            )
            boundary_phase = (
                self.boundary_phase
                if isinstance(self.boundary_phase, SpeechStateBoundaryPhase)
                else SpeechStateBoundaryPhase(str(self.boundary_phase).strip().lower())
            )
            boundary_token_policy = (
                self.boundary_token_policy
                if isinstance(self.boundary_token_policy, SpeechStateBoundaryTokenPolicy)
                else SpeechStateBoundaryTokenPolicy(
                    str(self.boundary_token_policy).strip().lower()
                )
            )
            successor_text_entry = (
                self.successor_text_entry
                if isinstance(self.successor_text_entry, SpeechStateSuccessorTextEntry)
                else SpeechStateSuccessorTextEntry(
                    str(self.successor_text_entry).strip().lower()
                )
            )
            talker_carry = (
                self.talker_carry
                if isinstance(self.talker_carry, SpeechStateTalkerCarry)
                else SpeechStateTalkerCarry(str(self.talker_carry).strip().lower())
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid speech state model contract enum") from exc
        if not boundaries or len(set(boundaries)) != len(boundaries):
            raise ValueError("speech state model contract requires unique boundaries")
        for name in (
            "retain_talker_kv", "retain_code_predictor_state",
            "retain_c2w_kv", "retain_c2w_history",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        if boundary_token_policy is SpeechStateBoundaryTokenPolicy.NO_EOS:
            if self.boundary_token_id is not None:
                raise ValueError("no_eos must not declare a boundary token")
        elif type(self.boundary_token_id) is not int or self.boundary_token_id < 0:
            raise ValueError("token boundary requires a nonnegative token id")
        if (
            isinstance(self.talker_hidden_tail, bool)
            or not isinstance(self.talker_hidden_tail, int)
            or self.talker_hidden_tail < 0
        ):
            raise ValueError("talker_hidden_tail must be a nonnegative integer")
        if talker_carry is SpeechStateTalkerCarry.DIRECT_KV:
            if not self.retain_talker_kv or self.talker_hidden_tail:
                raise ValueError("direct_kv requires Talker KV and no hidden tail")
        elif talker_carry is SpeechStateTalkerCarry.HIDDEN_TAIL_BRIDGE:
            if self.retain_talker_kv or self.talker_hidden_tail <= 0:
                raise ValueError(
                    "hidden_tail_bridge requires a positive hidden tail without Talker KV"
                )
        elif self.retain_talker_kv or self.talker_hidden_tail:
            raise ValueError("no Talker carry cannot retain Talker state")
        if (
            successor_start is SpeechStateSuccessorStart.READY_TO_DECODE
            and successor_text_entry is SpeechStateSuccessorTextEntry.PREFILL_SUCCESSOR_TEXT
        ):
            raise ValueError(
                "ready_to_decode cannot require successor prefill"
            )
        object.__setattr__(self, "model_fingerprint", model_fingerprint)
        object.__setattr__(self, "checkpoint_id", checkpoint_id)
        object.__setattr__(self, "training_scope", training_scope)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "successor_start", successor_start)
        object.__setattr__(self, "boundary_phase", boundary_phase)
        object.__setattr__(self, "boundary_token_policy", boundary_token_policy)
        object.__setattr__(self, "successor_text_entry", successor_text_entry)
        object.__setattr__(self, "cursor_policy", cursor_policy)
        object.__setattr__(self, "transfer", transfer)
        object.__setattr__(self, "talker_carry", talker_carry)

    @property
    def supports_segment_handoff(self) -> bool:
        return (
            any(boundary is not SpeechStateCheckpointBoundary.EXPLICIT_PAUSE
                for boundary in self.boundaries)
            and self.transfer is not SpeechStateTransfer.NONE
            and self.successor_start is not SpeechStateSuccessorStart.HARD_BOUNDARY_PREFILL
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "checkpoint_id": self.checkpoint_id,
            "boundaries": [item.value for item in self.boundaries],
            "boundary_phase": self.boundary_phase.value,
            "boundary_token_policy": self.boundary_token_policy.value,
            "boundary_token_id": self.boundary_token_id,
            "successor_start": self.successor_start.value,
            "successor_text_entry": self.successor_text_entry.value,
            "cursor_policy": self.cursor_policy.value,
            "transfer": self.transfer.value,
            "retain_talker_kv": self.retain_talker_kv,
            "retain_code_predictor_state": self.retain_code_predictor_state,
            "retain_c2w_kv": self.retain_c2w_kv,
            "retain_c2w_history": self.retain_c2w_history,
            "training_scope": self.training_scope,
            "talker_carry": self.talker_carry.value,
            "talker_hidden_tail": self.talker_hidden_tail,
        }
