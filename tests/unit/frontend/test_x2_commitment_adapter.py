from types import SimpleNamespace

from engine.frontend.text_commitment.x2_adapter import (
    X2BoundaryLevel,
    X2CommitmentAdapter,
)


class _Policy:
    def __init__(self):
        self.tokens = []
        self.feed_text_calls = 0
        self.segments = []
        self.finished = 0
        self.reset_count = 0

    def feed_text(self, *_args, **_kwargs):
        self.feed_text_calls += 1
        raise AssertionError("the main TN owns feed_text")

    def feed_token(self, *, punct_level=0):
        self.tokens.append(punct_level)
        return len(self.tokens)

    def splitter_config(self):
        return {"ema_ratio": 4.0, "safety_margin": 6}

    def observe_segment(self, **kwargs):
        self.segments.append(kwargs)
        return 4.0

    def finish(self):
        self.finished += 1
        return "finished"

    def reset(self):
        self.reset_count += 1


def _commit(commit_id, fence, raw_start=0, raw_end=2):
    return SimpleNamespace(
        commit_id=commit_id,
        fence=fence,
        raw_start=raw_start,
        raw_end=raw_end,
    )


def test_adapter_consumes_main_tn_commit_without_calling_feed_text():
    policy = _Policy()
    adapter = X2CommitmentAdapter(policy)

    assert not hasattr(adapter, "feed_text")
    result = adapter.consume_commit(
        _commit(1, 1),
        spoken_text="你好",
        token_count=3,
        boundary_level=X2BoundaryLevel.L2,
    )

    assert result.accepted is True
    assert result.spoken_text == "你好"
    assert result.token_count == 3
    assert policy.tokens == [0, 0, 2]
    assert policy.feed_text_calls == 0
    assert adapter.splitter_config() == {"ema_ratio": 4.0, "safety_margin": 6}


def test_special_main_tn_span_requests_boundary_before_successor():
    adapter = X2CommitmentAdapter(_Policy())

    first = adapter.consume_commit(
        SimpleNamespace(
            commit_id=1,
            fence=1,
            raw_start=0,
            raw_end=2,
            span_kind="plain",
            commit_kind="literal",
        ),
        spoken_text="温度",
        token_count=2,
    )
    second = adapter.consume_commit(
        SimpleNamespace(
            commit_id=2,
            fence=2,
            raw_start=2,
            raw_end=5,
            span_kind=SimpleNamespace(value="number"),
            commit_kind=SimpleNamespace(value="normalized"),
        ),
        spoken_text="二十五",
        token_count=3,
    )

    assert first.force_boundary_before is False
    assert second.accepted is True
    assert second.force_boundary_before is True


def test_adapter_rejects_commit_gaps_and_overlapping_spans_fail_closed():
    policy = _Policy()
    adapter = X2CommitmentAdapter(policy)

    first = adapter.consume_commit(
        _commit(1, 1, 0, 2), spoken_text="a", token_count=1
    )
    gap = adapter.consume_commit(
        _commit(3, 3, 2, 4), spoken_text="c", token_count=1
    )

    assert first.accepted is True
    assert gap.accepted is False
    assert gap.reason == "commit_id_gap"
    assert adapter.disabled is True
    assert policy.tokens == [0]

    overlap_policy = _Policy()
    overlap = X2CommitmentAdapter(overlap_policy)
    assert overlap.consume_commit(
        _commit(1, 1, 0, 3), spoken_text="abc", token_count=1
    ).accepted
    result = overlap.consume_commit(
        _commit(2, 2, 2, 5), spoken_text="bc", token_count=1
    )
    assert result.reason == "raw_span_overlap"
    assert overlap.disabled is True


def test_adapter_handles_missing_optional_methods_and_lifecycle():
    class Minimal:
        def feed_token(self, *, punct_level=0):
            return punct_level

    adapter = X2CommitmentAdapter(Minimal())
    result = adapter.consume_commit(
        _commit(1, 1), spoken_text="x", token_count=0, boundary_level=3
    )

    assert result.accepted is True
    assert result.decisions == ()
    assert adapter.splitter_config() == {}
    assert adapter.observe_segment(audio_steps=4, text_tokens=1) is None
    assert adapter.finish() is None
    adapter.reset()


def test_policy_errors_disable_adapter_and_do_not_retry():
    class Broken:
        def __init__(self):
            self.calls = 0

        def feed_token(self, *, punct_level=0):
            self.calls += 1
            raise RuntimeError("boom")

    policy = Broken()
    adapter = X2CommitmentAdapter(policy)
    result = adapter.consume_commit(
        _commit(1, 1), spoken_text="x", token_count=1
    )

    assert result.accepted is False
    assert adapter.disabled is True
    assert policy.calls == 1
    ignored = adapter.consume_commit(
        _commit(2, 2, 2, 3), spoken_text="y", token_count=1
    )
    assert ignored.accepted is False
    assert policy.calls == 1
