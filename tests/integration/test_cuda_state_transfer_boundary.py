"""Real CUDA stream fence check, not a model transfer parity claim."""

from __future__ import annotations

import os

import pytest
import torch

from engine.backend.executor import Executor


def test_state_transfer_boundary_waits_for_both_real_cuda_streams():
    if os.environ.get("RUN_CUDA_STATE_TRANSFER_TESTS") != "1":
        pytest.skip("set RUN_CUDA_STATE_TRANSFER_TESTS=1 for the CUDA fence check")
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "_sleep"):
        pytest.skip("CUDA stream sleep support is unavailable")

    device = torch.device("cuda", 0)
    executor = Executor.__new__(Executor)
    executor._device = device
    executor._compute_stream = torch.cuda.Stream(device=device)
    copy_stream = torch.cuda.Stream(device=device)
    compute_done = torch.cuda.Event()
    copy_done = torch.cuda.Event()
    compute_value = torch.zeros(1, device=device)
    copied_value = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    try:
        with torch.cuda.stream(executor._compute_stream):
            # Keep compute pending while the independent copy stream finishes.
            torch.cuda._sleep(100_000_000)
            compute_value.fill_(7)
            compute_done.record()
        with torch.cuda.stream(copy_stream):
            copied_value.fill_(11)
            copy_done.record()
            executor.synchronize_state_transfer()

        assert compute_done.query()
        assert copy_done.query()
        assert compute_value.item() == 7
        assert copied_value.item() == 11
    finally:
        torch.cuda.synchronize(device)
