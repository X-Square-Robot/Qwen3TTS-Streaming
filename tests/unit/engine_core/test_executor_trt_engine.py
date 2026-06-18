from __future__ import annotations

import pytest
import torch

from engine.backend.executor import (
    Executor,
    GPUFuture,
    TRTEngine,
    _stable_sampling_seed,
)
from engine.backend.kv_cache_pool import ModelConfig, SlotKVState


class _FakeContext:
    def __init__(
        self,
        *,
        output_shapes_before: dict[str, tuple[int, ...]],
        output_shapes_after: dict[str, tuple[int, ...]] | None = None,
        infer_shapes_result: list[str] | None = None,
    ):
        self.output_shapes_before = dict(output_shapes_before)
        self.output_shapes_after = dict(output_shapes_after or output_shapes_before)
        self.infer_shapes_result = list(infer_shapes_result or [])
        self.input_shapes: dict[str, tuple[int, ...]] = {}
        self.tensor_addresses: dict[str, int] = {}
        self.set_input_shape_calls: list[str] = []
        self.infer_shapes_calls = 0
        self.execute_calls = 0
        self._after_infer_shapes = False

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
        self.set_input_shape_calls.append(name)
        self.input_shapes[name] = tuple(shape)

    def set_tensor_address(self, name: str, address: int) -> None:
        self.tensor_addresses[name] = address

    def infer_shapes(self) -> list[str]:
        self.infer_shapes_calls += 1
        self._after_infer_shapes = True
        return list(self.infer_shapes_result)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        if self._after_infer_shapes:
            return self.output_shapes_after[name]
        return self.output_shapes_before[name]

    def execute_async_v3(self, _stream_handle: int) -> None:
        self.execute_calls += 1


class _FakeStream:
    cuda_stream = 123


class _FakeComputeStream:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


def _make_engine(
    ctx: _FakeContext,
    *,
    input_names: set[str] | None = None,
    output_dtypes: dict[str, torch.dtype] | None = None,
) -> TRTEngine:
    engine = TRTEngine.__new__(TRTEngine)
    engine._plan_path = "fake.plan"
    engine._device = torch.device("cpu")
    engine._engine = object()
    engine._context = ctx
    engine._input_names = set(input_names or {"input_embeds"})
    engine._output_dtypes = dict(output_dtypes or {"wav": torch.float32})
    engine._prev_input_shapes = {}
    engine._output_buffers = {}
    return engine


def _make_sampling_executor(*, seed: int = 1234) -> Executor:
    executor = Executor.__new__(Executor)
    executor._device = torch.device("cpu")
    executor._config = ModelConfig(
        logits_topk=4,
        cp_num_stages=3,
    )
    executor._random_seed = seed
    return executor


def _make_sampling_slot(slot_id: int, session_id: str) -> SlotKVState:
    return SlotKVState(
        slot_id=slot_id,
        session_id=session_id,
        segment_idx=0,
        is_free=False,
    )


def test_infer_skips_unknown_inputs_and_resolves_dynamic_outputs():
    ctx = _FakeContext(
        output_shapes_before={"wav": (-1, 1920)},
        output_shapes_after={"wav": (1, 1920)},
    )
    engine = _make_engine(ctx, input_names={"input_embeds"})

    outputs = engine.infer(
        inputs={
            "input_embeds": torch.randn(1, 4),
            "cp_gumbel_noise": torch.randn(1, 15, 50),
        },
        output_names=["wav"],
        stream=_FakeStream(),
    )

    assert ctx.set_input_shape_calls == ["input_embeds"]
    assert "cp_gumbel_noise" not in ctx.tensor_addresses
    assert ctx.infer_shapes_calls == 1
    assert ctx.execute_calls == 1
    assert tuple(outputs["wav"].shape) == (1, 1920)


def test_stable_sampling_seed_is_repeatable_and_lane_specific():
    assert _stable_sampling_seed(7, "session-a", 0) == _stable_sampling_seed(
        7, "session-a", 0
    )
    assert _stable_sampling_seed(7, "session-a", 0) != _stable_sampling_seed(
        7, "session-b", 0
    )


def test_sampling_noise_is_independent_of_batch_membership():
    batched = _make_sampling_executor(seed=99)
    slot_a = _make_sampling_slot(0, "session-a:0")
    slot_b = _make_sampling_slot(1, "session-b:0")

    batch_gumbel, batch_cp_gumbel = batched._build_sampling_noise([slot_a, slot_b])

    separate = _make_sampling_executor(seed=99)
    single_a = _make_sampling_slot(0, "session-a:0")
    single_b = _make_sampling_slot(1, "session-b:0")
    a_gumbel, a_cp_gumbel = separate._build_sampling_noise([single_a])
    b_gumbel, b_cp_gumbel = separate._build_sampling_noise([single_b])

    assert torch.equal(batch_gumbel[0:1], a_gumbel)
    assert torch.equal(batch_cp_gumbel[0:1], a_cp_gumbel)
    assert torch.equal(batch_gumbel[1:2], b_gumbel)
    assert torch.equal(batch_cp_gumbel[1:2], b_cp_gumbel)
    assert slot_a.sampling_seed != slot_b.sampling_seed


def test_infer_retains_contiguous_bound_inputs():
    ctx = _FakeContext(output_shapes_before={"wav": (1, 1920)})
    engine = _make_engine(ctx, input_names={"input_embeds"})
    original = torch.randn(4, 2).t()
    assert not original.is_contiguous()
    inputs = {"input_embeds": original}

    engine.infer(
        inputs=inputs,
        output_names=["wav"],
        stream=_FakeStream(),
    )

    assert inputs["input_embeds"].is_contiguous()
    assert ctx.tensor_addresses["input_embeds"] == inputs["input_embeds"].data_ptr()
    assert inputs["input_embeds"].data_ptr() != original.data_ptr()


def test_gpu_future_keeps_input_refs_until_synchronize():
    stream = _FakeComputeStream()
    future = GPUFuture(
        _compute_stream=stream,
        _input_refs={"input_embeds": torch.ones(1)},
    )

    assert future._input_refs
    future.wait()
    assert stream.synchronize_calls == 1
    assert future._input_refs == {}


def test_infer_raises_for_missing_required_inputs():
    ctx = _FakeContext(output_shapes_before={"wav": (1, 1920)})
    engine = _make_engine(ctx, input_names={"input_embeds", "token_counts"})

    with pytest.raises(RuntimeError, match="missing required inputs"):
        engine.infer(
            inputs={"input_embeds": torch.randn(1, 4)},
            output_names=["wav"],
            stream=_FakeStream(),
        )


def test_infer_raises_when_output_shape_remains_dynamic():
    ctx = _FakeContext(
        output_shapes_before={"wav": (-1, 1920)},
        output_shapes_after={"wav": (-1, 1920)},
    )
    engine = _make_engine(ctx, input_names={"input_embeds"})

    with pytest.raises(RuntimeError, match="unresolved output shape"):
        engine.infer(
            inputs={"input_embeds": torch.randn(1, 4)},
            output_names=["wav"],
            stream=_FakeStream(),
        )
