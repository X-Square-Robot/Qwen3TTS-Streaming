from engine.frontend.text_commitment.commit_policy import CommitAction, CommitPolicy
from engine.frontend.text_commitment.candidate_resolver import CandidateSet
from engine.frontend.text_commitment.types import SemanticFamily


def test_open_span_is_waited_without_candidates():
    decision = CommitPolicy().decide(
        family=SemanticFamily.QUANTITY,
        closed=False,
        final=False,
        candidates=CandidateSet(source="open_span"),
    )
    assert decision.action is CommitAction.WAIT


def test_final_without_candidates_is_fallback():
    decision = CommitPolicy().decide(
        family=SemanticFamily.QUANTITY,
        closed=True,
        final=True,
        candidates=CandidateSet(source="empty"),
    )
    assert decision.action is CommitAction.FALLBACK


def test_closed_ranked_candidate_commits():
    decision = CommitPolicy().decide(
        family=SemanticFamily.QUANTITY,
        closed=True,
        final=False,
        candidates=CandidateSet(
            candidates=(object(),), source="wetext", unique=True
        ),
    )
    assert decision.action is CommitAction.COMMIT
