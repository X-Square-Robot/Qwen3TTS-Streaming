from engine.frontend.text_commitment.candidate_resolver import (
    CandidateResolver,
    family_for_kind,
)
from engine.frontend.text_commitment.normalizer_backend import BackendCandidate
from engine.frontend.text_commitment.types import (
    LanguageKind,
    SemanticFamily,
    SemioticSpan,
    SpanKind,
)
from engine.frontend.text_commitment.commit_policy import CommitAction, CommitPolicy
from engine.frontend.text_commitment.semantic_spans import SpanDetector


class _Backend:
    backend_name = "fake"

    def candidates(self, text, *, language, domain=None, nbest=1):
        return [
            BackendCandidate("百分之九十九", "percent", -1.0, cost=1.0, rank=0),
            BackendCandidate("九十九", "cardinal", -2.0, cost=2.0, rank=1),
        ][:nbest]


def test_family_mapping_groups_semantically_related_spans():
    assert family_for_kind(SpanKind.NUMBER) is SemanticFamily.QUANTITY
    assert family_for_kind(SpanKind.ID_CARD) is SemanticFamily.IDENTIFIER
    assert family_for_kind(SpanKind.PHONE) is SemanticFamily.CONTACT
    assert family_for_kind(SpanKind.MARKDOWN) is SemanticFamily.STRUCTURED


def test_candidate_resolver_exposes_relative_margin_not_probability():
    span = SemioticSpan(
        span_id=1,
        raw_start=0,
        raw_end=3,
        raw_text="99%",
        kind=SpanKind.NUMBER,
        family=SemanticFamily.QUANTITY,
        closed=True,
    )
    result = CandidateResolver(_Backend(), nbest=8).resolve(
        span, language=LanguageKind.ZH
    )
    assert result.candidate_count == 2
    assert result.best_cost == 1.0
    assert result.second_cost == 2.0
    assert result.cost_margin == 1.0
    assert result.margin_per_input_unit == 1 / 3
    assert result.source == "wetext"


def test_commits_carry_family_and_candidate_diagnostics():
    from engine.frontend.text_commitment.committer import IncrementalTextCommitter

    commits = IncrementalTextCommitter().feed("中文99%", final=True).commits
    number = next(item for item in commits if item.raw_text == "99%")
    assert number.semantic_family is SemanticFamily.QUANTITY
    assert number.candidate_count >= 1
    assert number.decision_source in {"wetext", "rule", "empty", "resolver_error"}


def test_policy_waits_for_open_or_ambiguous_spans():
    span = SemioticSpan(1, 0, 3, "99%", SpanKind.NUMBER,
                        family=SemanticFamily.QUANTITY, closed=False)
    candidates = CandidateResolver(_Backend()).resolve(span, language=LanguageKind.ZH)
    policy = CommitPolicy()
    assert policy.decide(family=span.family, closed=False, final=False,
                         candidates=candidates).action is CommitAction.WAIT
    assert policy.decide(family=span.family, closed=True, final=False,
                         candidates=candidates, margin_threshold=2.0).action is CommitAction.WAIT


def test_detector_keeps_legacy_classifier_behind_typed_boundary():
    detector = SpanDetector(lambda raw: SpanKind.NUMBER if raw.isdigit() else SpanKind.PLAIN)
    assert detector.classify("99") is SpanKind.NUMBER
    assert detector.family(SpanKind.NUMBER) is SemanticFamily.QUANTITY
