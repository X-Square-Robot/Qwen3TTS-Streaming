"""Tests for engine.core.mlfq.MLFQScheduler."""

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
            [s_q2, s_q1, s_q0],
            max_batch=3,
            get_meta=lambda s: s.meta,
        )
        assert [s.name for s in ordered] == ["fresh", "mid", "old"]

    def test_select_batch_truncates(self):
        sched = MLFQScheduler()
        segs = [FakeSegment(f"s{i}") for i in range(10)]
        ordered = sched.select_batch(
            segs,
            max_batch=3,
            get_meta=lambda s: s.meta,
        )
        assert len(ordered) == 3

    def test_anti_starvation_aging(self):
        cfg = MLFQConfig(
            q1_threshold=2,
            q2_threshold=5,
            aging_interval=10,
            starvation_limit=5,
        )
        sched = MLFQScheduler(cfg)
        meta = MLFQMeta()
        meta.level = 2
        meta.steps_since_schedule = 10

        for _ in range(10):
            boosted = sched.tick([meta])
        assert meta.level == 0

    def test_unscheduled_segment_ages_while_scheduled_does_not(self):
        # Regression: steps_since_schedule must advance for active segments
        # that are NOT selected into a batch, otherwise the starvation boost
        # can never fire. A scheduled segment must stay at 0.
        cfg = MLFQConfig(
            q1_threshold=1000,
            q2_threshold=2000,
            aging_interval=5,
            starvation_limit=5,
        )
        sched = MLFQScheduler(cfg)

        starved = MLFQMeta()
        starved.level = 2
        scheduled = MLFQMeta()  # stays Q0, picked every step

        all_metas = [starved, scheduled]
        for _ in range(5):
            # Emulate the engine loop: only `scheduled` makes it into the batch.
            sched.on_scheduled(scheduled)
            sched.on_step_done(scheduled)
            sched.tick(all_metas)

        # The never-scheduled segment was aged...
        assert starved.steps_since_schedule == 0  # reset by the boost
        assert starved.level == 0  # boosted out of Q2 by anti-starvation
        # ...while the continuously-scheduled one never accrued starvation.
        assert scheduled.steps_since_schedule == 0

    def test_starved_segment_eventually_scheduled_when_candidates_exceed_batch(self):
        # Drive candidates > max_batch so select_batch truncates and the
        # lowest-priority segment is never picked until aging rescues it.
        cfg = MLFQConfig(
            q1_threshold=1000,
            q2_threshold=2000,
            aging_interval=5,
            starvation_limit=5,
        )
        sched = MLFQScheduler(cfg)
        max_batch = 2

        starved = FakeSegment("starved")
        starved.meta.level = 2
        hog_a = FakeSegment("hog_a")
        hog_b = FakeSegment("hog_b")
        candidates = [starved, hog_a, hog_b]  # 3 candidates, batch of 2

        scheduled_at = []
        for step in range(8):
            ordered = sched.select_batch(
                candidates,
                max_batch,
                get_meta=lambda s: s.meta,
            )
            for seg in ordered:
                sched.on_step_done(seg.meta)
            if starved in ordered:
                scheduled_at.append(step)
            sched.tick([s.meta for s in candidates])

        # Starved while the two Q0 hogs monopolised the batch...
        assert scheduled_at and scheduled_at[0] >= 4, (
            f"starved segment should be skipped until aging boosts it, "
            f"got first schedule at step {scheduled_at[0] if scheduled_at else None}"
        )
        # ...and the boost actually let it into a batch.
        assert len(scheduled_at) >= 1

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
