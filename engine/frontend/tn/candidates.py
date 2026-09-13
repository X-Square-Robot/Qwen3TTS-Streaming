"""Optional WFST/candidate hooks for semantic recognizers.

The production WeText backend already exposes an n-best candidate contract.
This protocol keeps recognizers independent from that concrete package and
allows a real WFST implementation to be installed later without changing the
FSM or raw-coordinate contract.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from ..text_commitment.normalizer_backend import BackendCandidate
from ..text_commitment.types import LanguageKind, SpanKind


class WfstCandidateProvider(Protocol):
    def candidates(
        self,
        text: str,
        *,
        language: LanguageKind,
        domain: SpanKind,
        nbest: int,
    ) -> Sequence[BackendCandidate]: ...


__all__ = ("WfstCandidateProvider",)
