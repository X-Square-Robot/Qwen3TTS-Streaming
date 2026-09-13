"""State contract for the frontend TN span lifecycle."""
from ..text_commitment.types import CommitmentState

# CommitmentState is the existing public state contract used by the committer.
# Alias it here so the TN component does not introduce a second incompatible
# state vocabulary.
SpanState = CommitmentState

__all__ = ("SpanState", "CommitmentState")
