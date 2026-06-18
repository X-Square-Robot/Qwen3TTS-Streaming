from types import SimpleNamespace

from engine.backend.executor import Executor


class _FakeTRTEngine:
    def __init__(self, shape):
        self._shape = shape

    def get_input_profile_max_shape(self, name: str, profile_idx: int = 0):
        assert name == "talker_past_kv"
        assert profile_idx == 0
        return self._shape


class TestExecutorProfileLimits:
    def test_clamps_runtime_batch_and_seq_to_trt_profile(self):
        executor = Executor.__new__(Executor)
        executor._fused_engine = _FakeTRTEngine((32, 56, 8, 512, 128))
        executor._max_batch = 48
        executor._max_seq_len = 2048
        executor._config = SimpleNamespace(max_seq_len=2048)

        executor._apply_runtime_profile_limits()

        assert executor._max_batch == 32
        assert executor._max_seq_len == 512
        assert executor._config.max_seq_len == 512

    def test_keeps_lower_runtime_batch_and_seq(self):
        executor = Executor.__new__(Executor)
        executor._fused_engine = _FakeTRTEngine((32, 56, 8, 512, 128))
        executor._max_batch = 16
        executor._max_seq_len = 384
        executor._config = SimpleNamespace(max_seq_len=384)

        executor._apply_runtime_profile_limits()

        assert executor._max_batch == 16
        assert executor._max_seq_len == 384
        assert executor._config.max_seq_len == 384
