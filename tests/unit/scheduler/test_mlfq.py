"""Tests for engine.core.mlfq.MLFQScheduler."""

import pytest

from engine.core.mlfq import MLFQConfig, MLFQMeta, MLFQScheduler


class FakeSegment:
    def __init__(self, name: str):
        self.name = name
        self.meta = MLFQMeta()


class TestMLFQScheduler:
    def test_initial_level_is_zero(self):
        meta = MLFQMeta()
        assert meta.level == 0

    def test_demotion_q0_to_q1(self):
        cfg = MLFQConfig(q1_threshold=3, q2_threshold=10)
        sched = MLFQScheduler(cfg)
        meta = MLFQMeta()
        for _ in range(3):
            sched.on_step_done(meta)
        assert meta.level == 1

    def test_demotion_q1_to_q2(self):
        cfg = MLFQConfig(q1_threshold=2, q2_threshold=5)
        sched = MLFQScheduler(cfg)
        meta = MLFQMeta()
        for _ in range(5):
            sched.on_step_done(meta)
        assert meta.level == 2

    def test_segment_boundary_resets(self):
        cfg = MLFQConfig(q1_threshold=2, q2_threshold=5)
        sched = MLFQScheduler(cfg)
        meta = MLFQMeta()
        for _ in range(5):
            sched.on_step_done(meta)
        assert meta.level == 2
        sched.on_segment_boundary(meta)
        assert meta.level == 0
        assert meta.decode_steps == 0

    def test_select_batch_ordering(self):
        cfg = MLFQConfig(q1_threshold=5, q2_threshold=100)
        sched = MLFQScheduler(cfg)

        s_q0 = FakeSegment("fresh")
        s_q1 = FakeSegment("mid")
        for _ in range(5):
            sched.on_step_done(s_q1.meta)
        s_q2 = FakeSegment("old")
        s_q2.meta.level = 2

        ordered = sched.select_batch(
            [s_q2, s_q1, s_q0], max_batch=3,
            get_meta=lambda s: s.meta,
        )
        assert [s.name for s in ordered] == ["fresh", "mid", "old"]

    def test_select_batch_truncates(self):
        sched = MLFQScheduler()
        segs = [FakeSegment(f"s{i}") for i in range(10)]
        ordered = sched.select_batch(
            segs, max_batch=3,
            get_meta=lambda s: s.meta,
        )
        assert len(ordered) == 3

    def test_anti_starvation_aging(self):
        cfg = MLFQConfig(
            q1_threshold=2, q2_threshold=5,
            aging_interval=10, starvation_limit=5,
        )
        sched = MLFQScheduler(cfg)
        meta = MLFQMeta()
        meta.level = 2
        meta.steps_since_schedule = 10

        for _ in range(10):
            boosted = sched.tick([meta])
        assert meta.level == 0

    def test_on_scheduled_resets_counter(self):
        sched = MLFQScheduler()
        meta = MLFQMeta()
        meta.steps_since_schedule = 42
        sched.on_scheduled(meta)
        assert meta.steps_since_schedule == 0

    def test_global_step_increments(self):
        sched = MLFQScheduler()
        assert sched.global_step == 0
        sched.tick()
        sched.tick()
        assert sched.global_step == 2
