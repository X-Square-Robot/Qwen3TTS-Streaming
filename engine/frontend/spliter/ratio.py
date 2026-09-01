"""Duration estimation and conservative split-capacity control.

The two ratios deliberately have different update rules:

* ``duration_ratio`` is a two-sided EMA learned only from sufficiently large,
  naturally completed segments.  It is used for text-progress attribution and
  guarded-delivery duration estimates.
* ``safety_ratio`` is the conservative ratio used to derive a segment's text
  capacity.  It never decreases within a session.  Censored overflows and
  confirmed runaway failures may tighten it, but never contaminate the
  duration EMA.

Keeping these responsibilities separate prevents a few short, normal segments
from expanding the hard text budget of later segments and prevents truncated
failure observations from being treated as exact duration samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class RatioOutcome(str, Enum):
    """Typed classification of one segment's ratio feedback."""

    CODEC_EOS = "codec_eos"
    KV_OVERFLOW = "kv_overflow"
    LOOP_ABORT = "loop_abort"
    LENGTH_ABORT = "length_abort"
    IGNORED_FAILURE = "ignored_failure"

    @classmethod
    def from_eos_reason(
        cls,
        eos_reason: str,
        *,
        overflow: bool = False,
    ) -> "RatioOutcome":
        reason = str(eos_reason or "").strip().lower()
        if overflow or reason == cls.KV_OVERFLOW.value:
            return cls.KV_OVERFLOW
        if reason == cls.CODEC_EOS.value or not reason:
            return cls.CODEC_EOS
        if reason == cls.LOOP_ABORT.value:
            return cls.LOOP_ABORT
        if reason == cls.LENGTH_ABORT.value:
            return cls.LENGTH_ABORT
        return cls.IGNORED_FAILURE

    @classmethod
    def from_retry_reason(cls, retry_reason: str) -> "RatioOutcome":
        reason = str(retry_reason or "").strip().lower()
        if reason == "loop":
            return cls.LOOP_ABORT
        if reason == "length":
            return cls.LENGTH_ABORT
        return cls.IGNORED_FAILURE


@dataclass(frozen=True)
class RatioObservation:
    """Before/after state produced by :meth:`SplitRatioController.observe`."""

    outcome: RatioOutcome
    observed_ratio: float
    duration_before: float
    duration_after: float
    safety_before: float
    safety_after: float
    duration_sample_accepted: bool
    safety_tightened: bool
    text_tokens: int
    audio_steps: int


class SplitRatioController:
    """Own the independent duration EMA and monotonic safety ratio."""

    def __init__(
        self,
        *,
        duration_initial: float,
        safety_initial: float,
        duration_alpha: float,
        failure_alpha: float,
        min_ratio: float,
        max_ratio: float,
        min_duration_tokens: int,
        failure_multiplier: float,
    ) -> None:
        values = {
            "duration_initial": duration_initial,
            "safety_initial": safety_initial,
            "duration_alpha": duration_alpha,
            "failure_alpha": failure_alpha,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "failure_multiplier": failure_multiplier,
        }
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("ratio controller values must be finite")
        if min_ratio < 1.0 or max_ratio < min_ratio:
            raise ValueError("ratio bounds must satisfy 1 <= min_ratio <= max_ratio")
        if not 0.0 <= duration_alpha <= 1.0:
            raise ValueError("duration_alpha must be between 0 and 1")
        if not 0.0 <= failure_alpha <= 1.0:
            raise ValueError("failure_alpha must be between 0 and 1")
        if min_duration_tokens < 1:
            raise ValueError("min_duration_tokens must be positive")
        if failure_multiplier < 1.0:
            raise ValueError("failure_multiplier must be at least 1")

        if float(duration_initial) > float(max_ratio):
            raise ValueError("duration_initial must not exceed max_ratio")
        if float(safety_initial) > float(max_ratio):
            raise ValueError("safety_initial must not exceed max_ratio")

        self._duration_initial = self._clamp_static(
            float(duration_initial), float(min_ratio), float(max_ratio)
        )
        self._safety_initial = self._clamp_static(
            max(float(safety_initial), self._duration_initial),
            float(min_ratio),
            float(max_ratio),
        )
        self._duration_alpha = float(duration_alpha)
        self._failure_alpha = float(failure_alpha)
        self._min_ratio = float(min_ratio)
        self._max_ratio = float(max_ratio)
        self._min_duration_tokens = int(min_duration_tokens)
        self._failure_multiplier = float(failure_multiplier)

        self._duration_ratio = self._duration_initial
        self._safety_ratio = self._safety_initial

    @staticmethod
    def _clamp_static(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _clamp(self, value: float) -> float:
        return self._clamp_static(value, self._min_ratio, self._max_ratio)

    @property
    def duration_ratio(self) -> float:
        return self._duration_ratio

    @property
    def safety_ratio(self) -> float:
        return self._safety_ratio

    def set_duration_ratio(self, ratio: float) -> None:
        """Apply a legacy external duration update without weakening safety."""
        value = float(ratio)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("duration ratio must be finite and positive")
        self._duration_ratio = self._clamp(value)
        self._safety_ratio = max(self._safety_ratio, self._duration_ratio)

    def observe(
        self,
        *,
        audio_steps: int,
        text_tokens: int,
        outcome: RatioOutcome,
        segment_safety_ratio: float | None = None,
    ) -> RatioObservation:
        """Update the appropriate estimator for one terminal/retry outcome.

        ``KV_OVERFLOW`` is a censored lower-bound observation.  Loop/length
        failures may not have a meaningful observed ratio at all.  Both use
        the safety snapshot frozen when the segment opened, so two concurrent
        failures from the same planning epoch do not compound the backoff.
        """
        steps = max(0, int(audio_steps))
        tokens = max(0, int(text_tokens))
        observed = steps / tokens if steps > 0 and tokens > 0 else 0.0
        duration_before = self._duration_ratio
        safety_before = self._safety_ratio
        duration_accepted = False

        if outcome is RatioOutcome.CODEC_EOS:
            if steps > 0 and tokens >= self._min_duration_tokens:
                updated = (
                    (1.0 - self._duration_alpha) * self._duration_ratio
                    + self._duration_alpha * observed
                )
                self._duration_ratio = self._clamp(updated)
                # A genuinely slower successful segment proves that future
                # capacity must allow at least this observed source-token
                # ratio.  Keep the duration mean smoothed, but do not smooth
                # away that safety evidence; shorter observations can never
                # loosen the hard budget.
                self._safety_ratio = max(
                    self._safety_ratio,
                    self._clamp(observed),
                )
                duration_accepted = True
        elif outcome in (
            RatioOutcome.KV_OVERFLOW,
            RatioOutcome.LOOP_ABORT,
            RatioOutcome.LENGTH_ABORT,
        ):
            base = (
                float(segment_safety_ratio)
                if segment_safety_ratio is not None
                else self._safety_ratio
            )
            if not math.isfinite(base) or base <= 0.0:
                base = self._safety_ratio
            # Loop/length ratios are hallucination-inflated and therefore not
            # measurements.  KV overflow is different: generation was
            # truncated at the KV boundary, so steps/source-tokens is a
            # censored lower bound and safety must never remain below it.
            lower_bound = observed if outcome is RatioOutcome.KV_OVERFLOW else 0.0
            target = max(base * self._failure_multiplier, lower_bound)
            backed_off = (
                (1.0 - self._failure_alpha) * base
                + self._failure_alpha * target
            )
            self._safety_ratio = max(
                self._safety_ratio,
                self._clamp(lower_bound),
                self._clamp(backed_off),
            )

        return RatioObservation(
            outcome=outcome,
            observed_ratio=observed,
            duration_before=duration_before,
            duration_after=self._duration_ratio,
            safety_before=safety_before,
            safety_after=self._safety_ratio,
            duration_sample_accepted=duration_accepted,
            safety_tightened=self._safety_ratio > safety_before,
            text_tokens=tokens,
            audio_steps=steps,
        )

    def reset(self) -> None:
        self._duration_ratio = self._duration_initial
        self._safety_ratio = self._safety_initial


__all__ = (
    "RatioObservation",
    "RatioOutcome",
    "SplitRatioController",
)
