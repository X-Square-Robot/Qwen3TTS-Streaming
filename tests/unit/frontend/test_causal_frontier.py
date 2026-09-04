from __future__ import annotations

import pytest

from engine.frontend.text_commitment.causal import (
    CausalFrontier,
    FrontierViolation,
    longest_common_unit_prefix,
)


def test_frontier_commits_only_whole_english_units():
    frontier = CausalFrontier()
    update = frontier.advance(["twenty first", "twenty one"], pending_text="21")
    assert update.stable_delta == "twenty"
    assert update.stable_text == "twenty"
    assert not update.stable_text.endswith(" ")


def test_frontier_does_not_commit_divergent_cjk_prefix():
    assert longest_common_unit_prefix(["九十九", "百分之九十九"]) == ()


def test_frontier_never_retracts():
    frontier = CausalFrontier()
    frontier.advance(["twenty first", "twenty one"])
    with pytest.raises(FrontierViolation):
        frontier.advance(["ninety nine"])


def test_closed_frontier_uses_one_deterministic_candidate():
    frontier = CausalFrontier()
    update = frontier.advance(["百分之九十九"], closed=True)
    assert update.stable_delta == "百分之九十九"
    assert update.closed is True
