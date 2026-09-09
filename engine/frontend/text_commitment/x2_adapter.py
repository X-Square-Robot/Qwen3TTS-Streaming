"""Adapter from the engine's committed TN stream to an optional X2 policy.

The main ``IncrementalTextCommitter`` remains the only owner of raw Unicode,
normalization, and spoken-form decisions.  This module accepts only the
already committed ``TextCommit`` records plus caller-owned token counts and
boundary levels.  In particular, it intentionally has no ``feed_text``
method: an X2 policy cannot install a second online text-normalization path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class X2BoundaryLevel(IntEnum):
    """Boundary level used by the X2 capacity controller."""

    NONE = 0
    L1 = 1
    L2 = 2
    L3 = 3


@dataclass(frozen=True, slots=True)
class CommitmentObservation:
    """Evidence produced after one main-TN commit is consumed."""

    accepted: bool
    reason: str
    commit_id: int
    fence: int
    raw_start: int
    raw_end: int
    spoken_text: str
    token_count: int
    boundary_level: X2BoundaryLevel
    decisions: tuple[Any, ...] = ()
    force_boundary: bool = False
    force_boundary_before: bool = False


class X2CommitmentAdapter:
    """Fail-closed bridge for a session-scoped X2 commitment policy.

    ``policy`` is deliberately duck-typed so the engine does not depend on
    the optional X2 package.  The adapter forwards only capacity/controller
    methods.  It never calls a policy's raw-text method, even if one exists.
    """

    def __init__(self, policy: Any):
        if policy is None:
            raise ValueError("X2 commitment policy must not be None")
        self._policy = policy
        self._disabled = False
        self._disabled_reason = ""
        self._last_commit_id = 0
        self._last_fence = 0
        self._last_raw_end = 0
        self._finished = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason

    def _disable(self, reason: str, exc: BaseException | None = None) -> None:
        if not self._disabled:
            self._disabled_reason = str(reason or "policy_error")
            logger.warning(
                "Disabling X2 commitment policy: %s",
                self._disabled_reason,
                exc_info=exc is not None,
            )
        self._disabled = True

    @staticmethod
    def _coerce_nonnegative_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool) or (
            isinstance(value, float) and not value.is_integer()
        ):
            raise ValueError(f"{field_name} must be a nonnegative integer")
        result = int(value)
        if result < 0:
            raise ValueError(f"{field_name} must be a nonnegative integer")
        return result

    @staticmethod
    def _coerce_boundary(value: Any) -> X2BoundaryLevel:
        try:
            return X2BoundaryLevel(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("boundary_level must be one of 0, 1, 2, 3") from exc

    def consume_commit(
        self,
        commit: Any,
        *,
        spoken_text: str,
        token_count: int,
        boundary_level: int | X2BoundaryLevel = X2BoundaryLevel.NONE,
    ) -> CommitmentObservation:
        """Consume one already-committed main-TN record.

        ``spoken_text`` and ``token_count`` are supplied by the main frontend;
        this adapter does not normalize, tokenize, or reinterpret them.
        Duplicate, skipped, or overlapping records disable the optional policy
        so its state can never silently diverge from the engine text stream.
        """

        commit_id = 0
        fence = 0
        raw_start = 0
        raw_end = 0
        try:
            if self._disabled:
                return CommitmentObservation(
                    False, self._disabled_reason or "disabled", 0, 0, 0, 0,
                    str(spoken_text), 0, X2BoundaryLevel.NONE,
                )
            if not isinstance(spoken_text, str):
                raise ValueError("spoken_text must be supplied as a string")
            commit_id = self._coerce_nonnegative_int(
                getattr(commit, "commit_id", 0), "commit_id"
            )
            fence = self._coerce_nonnegative_int(
                getattr(commit, "fence", 0), "fence"
            )
            raw_start = self._coerce_nonnegative_int(
                getattr(commit, "raw_start", 0), "raw_start"
            )
            raw_end = self._coerce_nonnegative_int(
                getattr(commit, "raw_end", raw_start), "raw_end"
            )
            token_count = self._coerce_nonnegative_int(token_count, "token_count")
            level = self._coerce_boundary(boundary_level)
            if raw_end < raw_start:
                raise ValueError("raw_end must not precede raw_start")
            if commit_id:
                expected = self._last_commit_id + 1
                if self._last_commit_id and commit_id != expected:
                    self._disable("commit_id_gap")
                    return CommitmentObservation(
                        False, "commit_id_gap", commit_id, fence, raw_start,
                        raw_end, spoken_text, token_count, level,
                    )
            elif self._last_commit_id:
                self._disable("commit_id_missing")
                return CommitmentObservation(
                    False, "commit_id_missing", commit_id, fence, raw_start,
                    raw_end, spoken_text, token_count, level,
                )
            if fence:
                if self._last_fence and fence != self._last_fence + 1:
                    self._disable("fence_gap")
                    return CommitmentObservation(
                        False, "fence_gap", commit_id, fence, raw_start, raw_end,
                        spoken_text, token_count, level,
                    )
            elif self._last_fence:
                self._disable("fence_missing")
                return CommitmentObservation(
                    False, "fence_missing", commit_id, fence, raw_start, raw_end,
                    spoken_text, token_count, level,
                )
            if raw_start < self._last_raw_end:
                self._disable("raw_span_overlap")
                return CommitmentObservation(
                    False, "raw_span_overlap", commit_id, fence, raw_start,
                    raw_end, spoken_text, token_count, level,
                )

            decisions: list[Any] = []
            feed_token = getattr(self._policy, "feed_token", None)
            if feed_token is not None and not callable(feed_token):
                raise TypeError("policy.feed_token must be callable")
            if callable(feed_token):
                for index in range(token_count):
                    decision = feed_token(
                        punct_level=int(level) if index == token_count - 1 else 0
                    )
                    decisions.append(decision)

            force_boundary = bool(level)
            force_boundary = force_boundary or any(
                bool(getattr(decision, "close_after", False))
                for decision in decisions
            )
            # A committed TN span is only a stable mapping/normalization
            # unit.  It is not a clause boundary: ``99%`` becoming
            # ``百分之九十九`` must remain adjacent to surrounding text so
            # the splitter can make the boundary decision from punctuation,
            # capacity, and EOS.  In particular, do not split before every
            # numeric/URL/identifier span.
            force_boundary_before = False

            self._last_commit_id = max(self._last_commit_id, commit_id)
            self._last_fence = max(self._last_fence, fence)
            self._last_raw_end = max(self._last_raw_end, raw_end)
            return CommitmentObservation(
                True, "accepted", commit_id, fence, raw_start, raw_end,
                spoken_text, token_count, level, tuple(decisions),
                force_boundary,
                force_boundary_before,
            )
        except Exception as exc:
            self._disable("policy_or_commit_error", exc)
            return CommitmentObservation(
                False, self._disabled_reason, commit_id, fence, raw_start, raw_end,
                str(spoken_text), 0, X2BoundaryLevel.NONE,
            )

    def splitter_config(self) -> dict[str, Any]:
        """Return optional X2 splitter parameters, or an empty safe config."""

        if self._disabled:
            return {}
        callback = getattr(self._policy, "splitter_config", None)
        if callback is None:
            return {}
        if not callable(callback):
            self._disable("invalid_splitter_config")
            return {}
        try:
            value = callback()
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise TypeError("splitter_config must return a mapping")
            return dict(value)
        except Exception as exc:
            self._disable("splitter_config_error", exc)
            return {}

    def bind_engine_budget(self, remaining_kv: int) -> None:
        """Bind the loaded engine's post-prefill KV budget when supported."""

        if self._disabled:
            return
        callback = getattr(self._policy, "bind_engine_budget", None)
        if callback is None:
            return
        if not callable(callback):
            self._disable("invalid_bind_engine_budget")
            return
        try:
            callback(int(remaining_kv))
        except Exception as exc:
            self._disable("bind_engine_budget_error", exc)

    def observe_segment(
        self,
        *,
        audio_steps: int,
        text_tokens: int,
        overflow: bool = False,
    ) -> Any:
        """Forward measured segment feedback without exposing engine objects."""

        if self._disabled:
            return None
        callback = getattr(self._policy, "observe_segment", None)
        if callback is None:
            return None
        try:
            return callback(
                audio_steps=int(audio_steps),
                text_tokens=int(text_tokens),
                overflow=bool(overflow),
            )
        except Exception as exc:
            self._disable("observe_segment_error", exc)
            return None

    def finish(self) -> Any:
        """Finish the optional controller once at input completion."""

        if self._disabled or self._finished:
            return None
        self._finished = True
        callback = getattr(self._policy, "finish", None)
        if callback is None:
            return None
        try:
            return callback()
        except Exception as exc:
            self._disable("finish_error", exc)
            return None

    def reset(self) -> None:
        """Reset policy state during session teardown."""

        callback = getattr(self._policy, "reset", None)
        if callable(callback):
            try:
                callback()
            except Exception as exc:
                self._disable("reset_error", exc)


__all__ = (
    "CommitmentObservation",
    "X2BoundaryLevel",
    "X2CommitmentAdapter",
)
