"""Conservative typed commit policy primitives."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .candidate_resolver import CandidateSet
from .types import SemanticFamily


class CommitAction(str, Enum):
    WAIT = "wait"
    COMMIT = "commit"
    FALLBACK = "fallback"
    ERROR = "error"


@dataclass(frozen=True)
class CommitPolicyDecision:
    action: CommitAction
    reason: str
    confidence: float | None = None


class CommitPolicy:
    def decide(self, *, family: SemanticFamily, closed: bool, final: bool,
               candidates: CandidateSet, margin_threshold: float | None = None):
        if not closed and not final:
            return CommitPolicyDecision(CommitAction.WAIT, "open_span")
        if not candidates.candidates:
            return CommitPolicyDecision(
                CommitAction.FALLBACK if final else CommitAction.WAIT,
                "no_candidate" if final else "candidate_pending",
            )
        if candidates.cost_margin is None:
            return CommitPolicyDecision(CommitAction.COMMIT, "ranked_candidate")
        if margin_threshold is not None and candidates.cost_margin < margin_threshold:
            return CommitPolicyDecision(
                CommitAction.FALLBACK if final else CommitAction.WAIT,
                "ambiguous_final" if final else "ambiguous_margin",
            )
        return CommitPolicyDecision(CommitAction.COMMIT, "margin")
