"""Executor: owns TRT engines and CUDA streams, executes GPU compute.

Replaces Triton BLS with direct TRT plan execution via torch, eliminating:
  - pb_utils.InferenceRequest overhead (~0.5ms/step)
  - Triton scheduling latency (~0.3ms/step)
  - dlpack round-trip for every KV tensor

Key optimisations:
  1. Packed KV tensors: 56 talker KV + 16 C2W KV bindings merged into
     2 packed tensors, reducing TRT I/O binding from ~201 to ~61.
  2. Pre-cached output metadata: dtype mapping and output buffer allocation
     happen once at init, not per-step.
  3. launch_decode_step() is asynchronous: enqueues CUDA kernels on a
     dedicated stream and returns a GPUFuture, letting the caller process
     previous step results on CPU while GPU is busy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .batch_helper import (
    pad_packed_kv,
    padded_attention_bias,
    uniform_past_seq_lens,
)
from .debug_dump import EngineDebugDumper
from ..core.lifecycle import LifecycleLogger
from .kv_cache_pool import KVCachePool, ModelConfig, SlotKVState

logger = logging.getLogger(__name__)

FUSED_CHUNK_T = 1
_FUSED_DUMMY_PAST_LEN = 1
_MAX_TORCH_SEED = (1 << 63) - 1


def _stable_sampling_seed(base_seed: int, *parts: object) -> int:
    """Derive a deterministic torch seed from stable logical identifiers."""
    h = hashlib.blake2b(digest_size=16)
    h.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little") & _MAX_TORCH_SEED


def _wait_stream_for_current(stream: Any, device: torch.device) -> None:
    """Make a custom CUDA stream wait for tensors built on the current stream."""
    if stream is None or not hasattr(stream, "wait_stream"):
        return
    try:
        current = torch.cuda.current_stream(device)
    except Exception:
        return
    if getattr(current, "cuda_stream", None) == getattr(stream, "cuda_stream", None):
        return
    stream.wait_stream(current)


def _append_c2w_delta(
    current_kv: torch.Tensor,
    delta_kv: torch.Tensor,
    max_past_len: int,
) -> torch.Tensor:
    if current_kv is None:
        out = delta_kv
    else:
        out = torch.cat([current_kv, delta_kv], dim=3)
    if out.shape[3] > max_past_len:
        out = out[:, :, :, -max_past_len:, :]
    return out.contiguous()


# ---------------------------------------------------------------------------
# TRT Engine wrapper
# ---------------------------------------------------------------------------


class TRTEngine:
    """Thin wrapper around a TensorRT plan loaded via torch.

    Optimisations over naive per-call binding:
      - Output dtypes are cached once at load time.
      - Output buffers are pre-allocated for common shapes and reused.
      - Shapes are only set when they actually change.
    """

    def __init__(self, plan_path: str, device: torch.device):
        self._plan_path = plan_path
        self._device = device
        self._engine = None
        self._context = None
        self._input_names: set[str] = set()
        self._input_dtypes: Dict[str, torch.dtype] = {}
        self._output_dtypes: Dict[str, torch.dtype] = {}
        self._prev_input_shapes: Dict[str, tuple] = {}
        self._output_buffers: Dict[str, torch.Tensor] = {}

    def load(self) -> None:
        """Load TRT engine from .plan file."""
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        plan_path = Path(self._plan_path)
        if not plan_path.exists():
            raise FileNotFoundError(f"TRT plan not found: {plan_path}")

        with open(plan_path, "rb") as f:
            engine_bytes = f.read()

        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(
                self._format_load_failure(
                    plan_path,
                    "TensorRT returned no engine while deserializing the plan",
                )
            )

        self._context = self._engine.create_execution_context()
        if self._context is None:
            self._engine = None
            raise RuntimeError(
                self._format_load_failure(
                    plan_path,
                    "TensorRT could not create an execution context for the plan",
                )
            )
        self._cache_io_names()
        self._cache_output_dtypes()
        logger.info(
            "Loaded TRT engine: %s (%d I/O tensors)",
            plan_path.name,
            self._engine.num_io_tensors,
        )

    def _format_load_failure(self, plan_path: Path, reason: str) -> str:
        memory_hint = ""
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(self._device)
                memory_hint = (
                    f" GPU memory on {self._device}: "
                    f"free={free_bytes / (1024**2):.0f} MiB, "
                    f"total={total_bytes / (1024**2):.0f} MiB."
                )
            except Exception:
                memory_hint = ""
        return (
            f"Failed to load TRT engine: {plan_path}. {reason}."
            f"{memory_hint} Check the TensorRT log lines immediately above: common "
            "causes are CUDA out-of-memory, TensorRT library version mismatch, "
            "or a plan built for a different GPU/SM. Rebuild engines after "
            "changing TensorRT, CUDA, or GPU target."
        )

    def _cache_io_names(self) -> None:
        """Cache input tensor names once at load time."""
        import tensorrt as trt

        self._input_names.clear()
        self._input_dtypes.clear()
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.add(name)
                self._input_dtypes[name] = self._trt_to_torch_dtype(
                    self._engine.get_tensor_dtype(name)
                )

    def _cache_output_dtypes(self) -> None:
        """Cache output tensor dtypes once at load time."""
        import tensorrt as trt

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                dtype_trt = self._engine.get_tensor_dtype(name)
                self._output_dtypes[name] = self._trt_to_torch_dtype(dtype_trt)

    def get_io_names(self) -> tuple[list[str], list[str]]:
        """Return (input_names, output_names)."""
        import tensorrt as trt

        inputs, outputs = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        return inputs, outputs

    def get_input_profile_max_shape(
        self,
        name: str,
        profile_idx: int = 0,
    ) -> Optional[tuple[int, ...]]:
        """Return the max profile shape for an input tensor if available."""
        if self._engine is None:
            return None
        try:
            shapes = self._engine.get_tensor_profile_shape(name, profile_idx)
        except Exception:
            return None
        if not shapes or len(shapes) != 3:
            return None
        try:
            return tuple(int(dim) for dim in shapes[2])
        except TypeError:
            return None

    def get_tensor_dtype(self, name: str) -> Optional[torch.dtype]:
        """Return the cached torch dtype for an input or output tensor."""
        return self._input_dtypes.get(name) or self._output_dtypes.get(name)

    def infer(
        self,
        inputs: Dict[str, torch.Tensor],
        output_names: List[str],
        stream: torch.cuda.Stream,
        output_overrides: Optional[Dict[str, torch.Tensor]] = None,
        context=None,
    ) -> Dict[str, torch.Tensor]:
        """Execute inference on the given CUDA stream.

        Uses torch tensors directly — zero copy via data_ptr().
        Skips set_input_shape for tensors whose shape hasn't changed.
        Re-uses output buffers when shapes match previous call.

        Args:
            output_overrides: pre-allocated tensors to use for specific outputs
                instead of the internal buffer cache.  TRT writes directly into
                these tensors — the caller owns them and must ensure they are not
                aliased with any input tensor.
            context: optional TRT execution context to use instead of the
                default one.  When provided, shape caching and output buffer
                reuse are disabled to avoid cross-context state conflicts
                (e.g. separate prefill context overlapping with decode).
        """
        ctx = context if context is not None else self._context
        use_cache = context is None

        valid_input_names = self._input_names
        if not valid_input_names and self._engine is not None:
            self._cache_io_names()
            valid_input_names = self._input_names

        if valid_input_names:
            missing_inputs = sorted(
                name for name in valid_input_names if name not in inputs
            )
            if missing_inputs:
                raise RuntimeError(
                    "TRT inference missing required inputs: "
                    f"{missing_inputs}. Provided inputs: {sorted(inputs.keys())}"
                )

        input_shapes: Dict[str, tuple] = {}
        shape_changed = False
        for name, tensor in inputs.items():
            if valid_input_names and name not in valid_input_names:
                continue
            tensor = tensor.contiguous()
            inputs[name] = tensor
            shape = tuple(tensor.shape)
            input_shapes[name] = shape
            if use_cache:
                if self._prev_input_shapes.get(name) != shape:
                    ctx.set_input_shape(name, shape)
                    self._prev_input_shapes[name] = shape
                    shape_changed = True
            else:
                ctx.set_input_shape(name, shape)
                shape_changed = True
            ctx.set_tensor_address(name, tensor.data_ptr())
            if tensor.is_cuda:
                try:
                    tensor.record_stream(stream)
                except Exception:
                    logger.debug(
                        "Could not record input tensor stream: %s",
                        name,
                        exc_info=True,
                    )

        if shape_changed and hasattr(ctx, "infer_shapes"):
            unresolved = ctx.infer_shapes()
            if unresolved:
                raise RuntimeError(
                    "TensorRT shape inference could not resolve tensors "
                    f"{list(unresolved)} for inputs {input_shapes}"
                )

        outputs = {}
        for name in output_names:
            shape = tuple(ctx.get_tensor_shape(name))
            if any(int(dim) < 0 for dim in shape):
                raise RuntimeError(
                    "TensorRT produced unresolved output shape "
                    f"{shape} for output '{name}' with inputs {input_shapes}"
                )
            dtype_torch = self._output_dtypes.get(name, torch.float32)

            override = output_overrides.get(name) if output_overrides else None
            if (
                override is not None
                and override.shape == shape
                and override.dtype == dtype_torch
            ):
                out_tensor = override
            elif use_cache:
                existing = self._output_buffers.get(name)
                if (
                    existing is not None
                    and existing.shape == shape
                    and existing.dtype == dtype_torch
                ):
                    out_tensor = existing
                else:
                    out_tensor = torch.empty(
                        shape,
                        dtype=dtype_torch,
                        device=self._device,
                    )
                    self._output_buffers[name] = out_tensor
            else:
                out_tensor = torch.empty(
                    shape,
                    dtype=dtype_torch,
                    device=self._device,
                )

            ctx.set_tensor_address(name, out_tensor.data_ptr())
            if out_tensor.is_cuda:
                try:
                    out_tensor.record_stream(stream)
                except Exception:
                    logger.debug(
                        "Could not record output tensor stream: %s",
                        name,
                        exc_info=True,
                    )
            outputs[name] = out_tensor

        ctx.execute_async_v3(stream.cuda_stream)
        return outputs

    @staticmethod
    def _trt_to_torch_dtype(trt_dtype) -> torch.dtype:
        import tensorrt as trt

        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)


# ---------------------------------------------------------------------------
# GPU Future — handle to in-flight computation
# ---------------------------------------------------------------------------


@dataclass
class GPUFuture:
    """Handle to async GPU work.  Call wait() to synchronize."""

    _compute_stream: Any = None
    _raw: Dict[str, torch.Tensor] = field(default_factory=dict)
    _slots: List[SlotKVState] = field(default_factory=list)
    # Keeps async TensorRT input buffers alive until the compute stream syncs.
    _input_refs: Dict[str, Any] = field(default_factory=dict)
    _original_past_lens: List[int] = field(default_factory=list)
    _padded_past_len: int = 0
    _seq: int = 1
    _c2w_conv_output_names: List[str] = field(default_factory=list)
    _c2w_transconv_output_names: List[str] = field(default_factory=list)
    _codec_eos_id: int = 2150
    _used_pingpong: bool = False
    _inputs: Dict[str, Any] = field(default_factory=dict)
    _dump_meta: Dict[str, Any] = field(default_factory=dict)
    _debug_dumper: Optional[EngineDebugDumper] = None

    def wait(self) -> StepOutput:
        """Synchronize GPU and extract results.

        Returns batch-level delta KV tensors for pool scatter.
        Conv/transconv states are still split per-slot (heterogeneous shapes).
        """
        if self._compute_stream is not None:
            self._compute_stream.synchronize()
        self._input_refs.clear()

        batch_size = len(self._slots)
        raw = self._raw

        if self._debug_dumper is not None and self._dump_meta:
            self._debug_dumper.dump_call(
                metadata=self._dump_meta,
                inputs=self._inputs,
                outputs=raw,
                inputs_snapshotted=True,
            )

        wav = raw.get("wav")
        codec_sum = raw.get("codec_sum")
        full_codec = raw.get("full_codec")
        updated_tc = raw.get("updated_token_counts")

        # Batch-level state tensors; the engine loop slices rows on demand
        # (legacy slots) or scatters them into the slot-indexed arenas in one
        # indexed copy per state.  Pre-splitting per slot created ~5k tensor
        # views per step at width 128.
        conv_tensors = [raw.get(n) for n in self._c2w_conv_output_names]
        transconv_tensors = [raw.get(n) for n in self._c2w_transconv_output_names]

        codec_eos_id = self._codec_eos_id
        eos_flags = []
        audio_chunks = []
        if full_codec is not None:
            # Codebook-0 token ids come back to the CPU (B int64s on the
            # already-synced stream) so the engine loop can run the token
            # loop guard; EOS detection reuses the same copy.
            codec0_list = full_codec[:, 0].cpu().tolist()
            eos_list = [t == codec_eos_id for t in codec0_list]
        else:
            codec0_list = [None] * batch_size
            eos_list = [False] * batch_size
        if wav is not None:
            wav_cpu = wav.cpu().float()
        else:
            wav_cpu = None

        for row_idx in range(batch_size):
            eos_flags.append(eos_list[row_idx])
            if wav_cpu is not None:
                chunk = wav_cpu[row_idx].reshape(-1).numpy()
                audio_chunks.append(chunk.tobytes())
            else:
                audio_chunks.append(None)

        return StepOutput(
            slots=self._slots,
            eos_flags=eos_flags,
            tokens=codec0_list,
            audio_chunks=audio_chunks,
            batch_talker_kv=raw.get("talker_new_kv"),
            batch_c2w_kv=raw.get("c2w_new_kv"),
            original_past_lens=self._original_past_lens,
            padded_past_len=self._padded_past_len,
            batch_c2w_conv=conv_tensors,
            batch_c2w_transconv=transconv_tensors,
            codec_sum=codec_sum,
            updated_tc=updated_tc,
            used_pingpong=self._used_pingpong,
        )


@dataclass
class StepOutput:
    """Results from one decode step.

    Talker/C2W KV are kept as batch-level delta tensors for direct append
    into per-slot cache state. Conv/transconv states are split per-slot
    because they have heterogeneous shapes.

    When ``used_pingpong`` is True, TRT wrote conv/transconv outputs
    directly into each slot's write buffers.  The engine loop only needs
    to call ``slot.flip_c2w_buffers()`` — no copy or clone required.
    """

    slots: List[SlotKVState]
    eos_flags: List[bool]
    audio_chunks: List[Optional[bytes]]
    # Per-row codebook-0 token ids (None when full_codec is unavailable);
    # may be empty for test-constructed outputs.
    tokens: List[Optional[int]] = field(default_factory=list)
    batch_talker_kv: Optional[torch.Tensor] = None
    batch_c2w_kv: Optional[torch.Tensor] = None
    original_past_lens: List[int] = field(default_factory=list)
    padded_past_len: int = 0
    split_c2w_conv: List[List[Optional[torch.Tensor]]] = field(default_factory=list)
    split_c2w_transconv: List[List[Optional[torch.Tensor]]] = field(
        default_factory=list
    )
    # Batch-level per-state output tensors ([B, ...] each); preferred over
    # the per-slot split lists, which remain for test-constructed outputs.
    batch_c2w_conv: Optional[List[Optional[torch.Tensor]]] = None
    batch_c2w_transconv: Optional[List[Optional[torch.Tensor]]] = None
    codec_sum: Optional[torch.Tensor] = None
    updated_tc: Optional[torch.Tensor] = None
    used_pingpong: bool = False


# ---------------------------------------------------------------------------
# CUDA-graph decode
# ---------------------------------------------------------------------------


class GraphedFusedDecode:
    """CUDA-graph replay path for fused decode steps.

    The fused decode enqueues ~2700 kernels per step; at batch 128 the CPU
    enqueue time (~33ms) matches the GPU compute time, and because decode is
    autoregressive the enqueue of step N+1 cannot overlap step N.  Replaying
    a captured graph reduces the per-step CPU cost to a single launch.

    One graph is captured per (batch_bucket, past_bucket) shape signature:
    - batch is bucketed to a small fixed ladder, padding rows are computed
      but discarded (their staging content is stale-but-finite, and every
      op in the fused graph is independent across the batch dim);
    - talker past_len is bucketed to PAST_BUCKET_STEP multiples; the KV pool
      gather runs at the bucket length and ``attention_bias`` masks each
      slot's real tail, so stale KV columns never enter the softmax;
    - the c2w window is small, so its past length is fixed at the sliding
      window max and masked the same way (no extra signature dimension).

    Graph replay requires stable device addresses, so all engine I/O is
    bound to persistent staging buffers.  Each staging buffer is one flat
    max-size allocation; every bucket views its first ``numel`` elements at
    the bucket shape, which keeps all bucket views contiguous while sharing
    storage.  Inputs are built by the normal eager path and copied in
    (~1-2ms at batch 128, dominated by the KV gather that the eager path
    performs anyway).

    Two hard-won constraints from the 2026-07-02 attempt:
    - The graph MUST own a dedicated TRT execution context.  Prefill shares
      the executor's context; running it between replays overwrites the
      context's scratch memory and silently corrupts replay output (audio
      drift max_abs≈0.26).  A private context isolates the workspace at the
      cost of ``engine.device_memory_size`` (~2GB).
    - TRT initializes lazy per-shape resources on the first enqueues, so
      each bucket is enqueued twice as warmup before capture.
    """

    PAST_BUCKET_STEP = 64
    _BATCH_LADDER = (1, 2, 4, 8, 16, 32, 48, 64, 96)

    def __init__(
        self,
        trt_engine: TRTEngine,
        device: torch.device,
        compute_stream: torch.cuda.Stream,
        max_shapes: Dict[str, tuple],
        max_batch: int,
        max_past: int,
        max_entries: int = 12,
        profile_idx: int = 0,
    ) -> None:
        self._device = device
        self._stream = compute_stream
        self._max_entries = max_entries
        self._max_batch = max_batch
        self._max_past = max_past
        self._batch_buckets = sorted(
            {b for b in self._BATCH_LADDER if b < max_batch} | {max_batch}
        )
        self._max_shapes = {name: tuple(shape) for name, shape in max_shapes.items()}

        engine = trt_engine._engine
        self._scratch: Optional[torch.Tensor] = None
        if profile_idx > 0:
            # Bind the dedicated context to the decode-only optimization
            # profile: its scratch is a fraction of the full profile's (the
            # full profile sizes scratch for prefill seq lengths — 7.1 GiB on
            # the 128×512 1.7b engine, which cannot be paid twice on a 32 GiB
            # card).  USER-managed memory keeps the allocation observable.
            scratch_bytes = int(
                engine.get_device_memory_size_for_profile_v2(profile_idx)
            )
            self._scratch = torch.empty(
                scratch_bytes, dtype=torch.uint8, device=device
            )
            self._context = engine.create_execution_context_without_device_memory()
            if self._context is None:
                raise RuntimeError(
                    "Could not create TRT execution context for CUDA-graph decode"
                )
            if not self._context.set_optimization_profile_async(
                profile_idx, compute_stream.cuda_stream
            ):
                raise RuntimeError(
                    f"Could not select optimization profile {profile_idx} for "
                    "CUDA-graph decode"
                )
            compute_stream.synchronize()
            try:
                self._context.set_device_memory(
                    self._scratch.data_ptr(), scratch_bytes
                )
            except TypeError:  # older binding: property setter, size implicit
                self._context.device_memory = self._scratch.data_ptr()
            logger.info(
                "CUDA-graph decode context on profile %d (scratch %.2f GiB)",
                profile_idx,
                scratch_bytes / (1024**3),
            )
        else:
            self._context = engine.create_execution_context()
            if self._context is None:
                raise RuntimeError(
                    "Could not create dedicated TRT execution context for "
                    "CUDA-graph decode (likely GPU OOM)"
                )

        in_names, out_names = trt_engine.get_io_names()
        missing = sorted(set(in_names) - set(self._max_shapes))
        if missing:
            raise RuntimeError(f"CUDA-graph staging missing input shapes: {missing}")
        self._out_names = out_names

        self._in_flat: Dict[str, torch.Tensor] = {}
        for name in in_names:
            shape = self._max_shapes[name]
            dtype = trt_engine.get_tensor_dtype(name) or torch.float32
            self._in_flat[name] = torch.zeros(
                math.prod(shape), dtype=dtype, device=device
            )
            self._context.set_input_shape(name, shape)
        unresolved = self._context.infer_shapes()
        if unresolved:
            raise RuntimeError(
                f"CUDA-graph staging: unresolved shapes at decode max: {unresolved}"
            )
        self._out_flat: Dict[str, torch.Tensor] = {}
        for name in out_names:
            shape = tuple(int(d) for d in self._context.get_tensor_shape(name))
            dtype = trt_engine.get_tensor_dtype(name) or torch.float32
            self._out_flat[name] = torch.zeros(
                math.prod(shape), dtype=dtype, device=device
            )

        # key -> {"graph", "in": views, "out": views}; LRU-capped because each
        # instantiated graph holds parameters for every captured kernel.
        self._graphs: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()

        staging_mb = sum(
            t.numel() * t.element_size() for t in self._in_flat.values()
        ) + sum(t.numel() * t.element_size() for t in self._out_flat.values())
        logger.info(
            "CUDA-graph decode ready: staging=%.0f MiB, batch buckets=%s, "
            "past step=%d (max %d), max graphs=%d",
            staging_mb / (1024**2),
            self._batch_buckets,
            self.PAST_BUCKET_STEP,
            max_past,
            max_entries,
        )

    def bucket(self, batch: int, max_past_len: int) -> Optional[tuple[int, int]]:
        """Return the (batch, past) bucket key, or None if out of range."""
        if batch > self._max_batch:
            return None
        b = next(x for x in self._batch_buckets if x >= batch)
        p = ((max(max_past_len, 1) + self.PAST_BUCKET_STEP - 1)
             // self.PAST_BUCKET_STEP) * self.PAST_BUCKET_STEP
        if p > self._max_past:
            return None
        return (b, p)

    def _bucket_shape(self, name: str, b: int, p: int) -> tuple[int, ...]:
        shape = list(self._max_shapes[name])
        shape[0] = b
        if name == "talker_past_kv":
            shape[3] = p
        elif name == "attention_bias":
            shape[3] = p + FUSED_CHUNK_T
        return tuple(shape)

    def _capture(self, key: tuple[int, int]) -> Dict[str, Any]:
        b, p = key
        ctx = self._context
        in_views: Dict[str, torch.Tensor] = {}
        for name, flat in self._in_flat.items():
            shape = self._bucket_shape(name, b, p)
            view = flat[: math.prod(shape)].view(shape)
            ctx.set_input_shape(name, shape)
            ctx.set_tensor_address(name, view.data_ptr())
            in_views[name] = view
        unresolved = ctx.infer_shapes()
        if unresolved:
            raise RuntimeError(
                f"CUDA-graph capture: unresolved shapes {list(unresolved)} for {key}"
            )
        out_views: Dict[str, torch.Tensor] = {}
        for name in self._out_names:
            shape = tuple(int(d) for d in ctx.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(
                    f"CUDA-graph capture: unresolved output '{name}' for {key}"
                )
            flat = self._out_flat[name]
            numel = math.prod(shape)
            if numel > flat.numel():
                raise RuntimeError(
                    f"CUDA-graph staging undersized for output '{name}' at {key}"
                )
            view = flat[:numel].view(shape)
            ctx.set_tensor_address(name, view.data_ptr())
            out_views[name] = view

        with torch.cuda.stream(self._stream):
            for _ in range(2):
                if not ctx.execute_async_v3(self._stream.cuda_stream):
                    raise RuntimeError("TRT enqueue failed during CUDA-graph warmup")
        self._stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self._stream):
            if not ctx.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TRT enqueue failed during CUDA-graph capture")

        entry = {"graph": graph, "in": in_views, "out": out_views}
        self._graphs[key] = entry
        if len(self._graphs) > self._max_entries:
            evicted_key, _ = self._graphs.popitem(last=False)
            logger.info("CUDA-graph LRU evicted bucket %s", evicted_key)
        logger.info("CUDA-graph captured for bucket %s", key)
        return entry

    def entry(self, key: tuple[int, int]) -> Dict[str, Any]:
        """Return the bucket's graph entry, capturing it on first use."""
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._capture(key)
        else:
            self._graphs.move_to_end(key)
        return entry

    def run(
        self,
        key: tuple[int, int],
        inputs: Dict[str, torch.Tensor],
        batch: int,
    ) -> Dict[str, torch.Tensor]:
        """Copy ``inputs`` into staging, replay the bucket's graph, and return
        views of the first ``batch`` output rows.

        Callers must fully consume (copy/scatter/cpu) the returned views
        before the next ``run`` call overwrites the staging buffers; the
        engine loop's process-then-launch ordering guarantees this.
        """
        entry = self.entry(key)

        for name, tensor in inputs.items():
            view = entry["in"].get(name)
            if view is None:
                continue
            if tensor.data_ptr() == view.data_ptr():
                continue  # caller already filled the staging view in place
            if tuple(tensor.shape) == tuple(view.shape):
                view.copy_(tensor, non_blocking=True)
            else:
                if tuple(tensor.shape[1:]) != tuple(view.shape[1:]):
                    raise RuntimeError(
                        f"CUDA-graph input '{name}' shape {tuple(tensor.shape)} "
                        f"does not fit bucket view {tuple(view.shape)}"
                    )
                view[: tensor.shape[0]].copy_(tensor, non_blocking=True)

        _wait_stream_for_current(self._stream, self._device)
        with torch.cuda.stream(self._stream):
            entry["graph"].replay()

        return {name: view[:batch] for name, view in entry["out"].items()}


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Executor:
    """Manages TRT engines and CUDA streams for pipelined decode.

    CUDA stream layout:
        compute_stream: fused Talker decode + Code2Wav in single engine call
    """

    def __init__(
        self,
        *,
        engine_dir: str = "",
        weights_dir: str = "",
        device_id: int = 0,
        max_batch_size: int = 48,
        max_seq_len: int = 512,
        model_config: Optional[ModelConfig] = None,
        do_sample: bool = False,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        random_seed: int = 0,
    ):
        self._engine_dir = Path(engine_dir) if engine_dir else None
        self._weights_dir = Path(weights_dir) if weights_dir else None
        self._device = torch.device("cuda", device_id)
        self._max_batch = max_batch_size
        self._max_seq_len = max_seq_len
        self._max_input_len = 0
        self._config = model_config or ModelConfig()
        self._do_sample = do_sample
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._random_seed = int(random_seed)

        self._compute_stream = torch.cuda.Stream(device=self._device)

        self._fused_engine: Optional[TRTEngine] = None
        self._embedding_weights = None
        self._kv_pool: Optional[KVCachePool] = None
        self._codec_eos_id: int = 2150

        # CUDA-graph decode (see GraphedFusedDecode).  ENGINE_CUDA_GRAPH_DECODE=0
        # disables it; init failures (e.g. staging OOM) and repeated per-step
        # failures fall back to the eager path automatically.
        self._graph_decode_enabled = os.environ.get(
            "ENGINE_CUDA_GRAPH_DECODE", "1"
        ).strip().lower() not in ("0", "false", "off")
        self._graph_decode: Optional[GraphedFusedDecode] = None
        self._graph_decode_failures = 0
        # Shared persistent gather arena for batched talker KV (flat, viewed
        # per step).  Replaces the per-step transient gather tensor that peaks
        # at B×L*2×H×past×D (7.5 GiB at 128×512) and OOMs under low headroom.
        # Points at the CUDA-graph staging when graphs are on (the two paths
        # run serially on one thread, so sharing is safe); lazily allocated
        # when graphs are off; None -> legacy transient gather.
        self._talker_gather_flat: Optional[torch.Tensor] = None
        self._talker_gather_flat_failed = False

        self._c2w_conv_input_names: list[str] = []
        self._c2w_conv_output_names: list[str] = []
        self._c2w_transconv_input_names: list[str] = []
        self._c2w_transconv_output_names: list[str] = []
        self._manifest: dict = {}
        self._debug_dumper = EngineDebugDumper(
            engine_dir=self._engine_dir,
            weights_dir=self._weights_dir,
            device=self._device,
        )
        if self._debug_dumper.enabled:
            # L3: mark where the tensor dump lands so a log grep can locate the
            # dump products. Per-step tensors + slot_rows (frame_idx/text_idx/
            # full_codec/eos/sampling_seed) align with the L2 decision records
            # via session_id, forming the decision→evidence closure.
            LifecycleLogger.emit(
                session_id="-",
                phase="dump_enabled",
                dump_dir=str(getattr(self._debug_dumper, "_dir", "")),
                do_sample=self._do_sample,
                temperature=self._temperature,
                repetition_penalty=self._repetition_penalty,
                random_seed=self._random_seed,
            )

        logger.info(
            "Executor created (device=%s, max_batch=%d, max_seq=%d)",
            self._device,
            max_batch_size,
            max_seq_len,
        )

    @property
    def max_batch_size(self) -> int:
        return self._max_batch

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def max_input_len(self) -> int:
        """Max prefill length the loaded TRT plan supports (0 if unknown)."""
        return self._max_input_len

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load TRT engines, embedding weights, and initialize KV pool."""
        fused_plan: Optional[Path] = None
        if self._engine_dir:
            mp = self._engine_dir / "model.plan"
            te = self._engine_dir / "talker_code2wav_fused.engine"
            if mp.exists():
                fused_plan = mp
            elif te.exists():
                fused_plan = te
            manifest_path = self._engine_dir / "triton_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    self._manifest = json.load(f)
        if self._engine_dir and fused_plan is not None:
            self._fused_engine = TRTEngine(
                str(fused_plan),
                self._device,
            )
            self._fused_engine.load()
            self._apply_runtime_profile_limits()
            self._validate_io_dtype_consistency()
            self._discover_c2w_io_names()

        else:
            logger.warning("No TRT plan found, running in stub mode")

        self._kv_pool = KVCachePool(
            max_slots=self._max_batch,
            config=self._config,
            device=self._device,
        )
        self._init_c2w_state_arenas()
        self._init_graph_decode()
        logger.info(
            "Executor loaded (engine=%s, effective_max_seq=%d, cuda_graph=%s)",
            "TRT" if self._fused_engine else "stub",
            self._max_seq_len,
            "on" if self._graph_decode is not None else "off",
        )

    def _graph_decode_max_shapes(self, max_past: int) -> Dict[str, tuple]:
        """Decode-time maximum shape per fused-engine input (seq is always 1).

        Sizes the CUDA-graph staging buffers; must mirror what
        ``_build_fused_inputs`` produces for a decode step.
        """
        cfg = self._config
        B = self._max_batch
        c2w_max_past = cfg.c2w_sliding_window - FUSED_CHUNK_T
        shapes: Dict[str, tuple] = {
            "input_embeds": (B, FUSED_CHUNK_T, cfg.hidden_size),
            "position_ids": (B, 3, FUSED_CHUNK_T, 1),
            "attention_bias": (B, 1, FUSED_CHUNK_T, max_past + FUSED_CHUNK_T),
            "token_counts": (B, cfg.codec_vocab_size),
            "gumbel_noise": (B, cfg.logits_topk),
            "cp_gumbel_noise": (B, cfg.cp_num_stages, cfg.logits_topk),
            "temperature": (B, 1),
            "penalty": (B, 1),
            "cache_position": (B, FUSED_CHUNK_T),
            "talker_past_kv": (
                B, cfg.num_layers * 2, cfg.kv_heads, max_past, cfg.head_dim
            ),
            "c2w_past_kv": (
                B, cfg.n_c2w_layers * 2, cfg.c2w_kv_heads, c2w_max_past,
                cfg.c2w_head_dim,
            ),
            "c2w_attention_bias": (
                B, 1, FUSED_CHUNK_T, c2w_max_past + FUSED_CHUNK_T
            ),
        }
        for idx, name in enumerate(self._c2w_conv_input_names):
            shape = list(self._c2w_conv_shapes[idx])
            shape[0] = B
            shapes[name] = tuple(shape)
        for idx, name in enumerate(self._c2w_transconv_input_names):
            shape = list(self._c2w_transconv_shapes[idx])
            shape[0] = B
            shapes[name] = tuple(shape)
        return shapes

    def _init_graph_decode(self) -> None:
        if not self._graph_decode_enabled or self._fused_engine is None:
            return
        if self._kv_pool is None or not self._kv_pool._preallocate:
            logger.warning("CUDA-graph decode requires the pre-allocated KV pool; off")
            return
        max_past_cap = min(
            self._config.max_seq_len,
            int(os.environ.get("ENGINE_CUDA_GRAPH_MAX_PAST", self._config.max_seq_len)),
        )
        # The KV staging buffer is max_batch × num_layers*2 × kv_heads ×
        # max_past × head_dim (7.5 GiB at 128×512 for the 1.7b) — try the
        # requested cap first and step down on OOM.  Steps beyond the cap
        # fall back to the eager path per step.
        # Prefer a decode-only optimization profile when the engine has one
        # (build_engines.sh emits it as profile 1): the dedicated context then
        # only pays decode-sized scratch instead of the full profile's.
        profile_idx = 0
        num_profiles = int(
            getattr(self._fused_engine._engine, "num_optimization_profiles", 1)
        )
        if num_profiles > 1:
            profile_idx = 1
        ladder = [p for p in (max_past_cap, 384, 256, 128) if p <= max_past_cap]
        for max_past in dict.fromkeys(ladder):
            try:
                self._graph_decode = GraphedFusedDecode(
                    self._fused_engine,
                    self._device,
                    self._compute_stream,
                    self._graph_decode_max_shapes(max_past),
                    max_batch=self._max_batch,
                    max_past=max_past,
                    max_entries=int(
                        os.environ.get("ENGINE_CUDA_GRAPH_MAX_ENTRIES", "16")
                    ),
                    profile_idx=profile_idx,
                )
                # The eager path shares the graph's KV staging as its gather
                # arena (serial use on the engine thread; the graph re-fills
                # it from the pool every step, so no state survives in it).
                self._talker_gather_flat = self._graph_decode._in_flat.get(
                    "talker_past_kv"
                )
                return
            except torch.cuda.OutOfMemoryError:
                self._graph_decode = None
                torch.cuda.empty_cache()
                logger.warning(
                    "CUDA-graph staging OOM at max_past=%d; trying smaller", max_past
                )
            except Exception:
                self._graph_decode = None
                logger.exception(
                    "CUDA-graph decode init failed; falling back to eager decode"
                )
                return
        logger.warning("CUDA-graph decode disabled: staging OOM at every ladder step")

    def set_embedding_weights(self, weights) -> None:
        self._embedding_weights = weights
        if hasattr(weights, "codec_eos_id"):
            self._codec_eos_id = int(weights.codec_eos_id)
            logger.info("codec_eos_id set to %d from weights", self._codec_eos_id)

    def _apply_runtime_profile_limits(self) -> None:
        """Clamp runtime batch/seq so they never exceed TRT profile bounds."""
        if self._fused_engine is None:
            return
        shape = self._fused_engine.get_input_profile_max_shape("talker_past_kv")
        input_shape = self._fused_engine.get_input_profile_max_shape("input_embeds")
        if input_shape is not None and len(input_shape) >= 2:
            self._max_input_len = int(input_shape[1])
        if shape is None or len(shape) < 4:
            return
        profile_max_batch = int(shape[0])
        profile_max_seq = int(shape[3])
        if profile_max_batch > 0 and self._max_batch > profile_max_batch:
            logger.warning(
                "Requested max_batch_size=%d exceeds TRT profile max=%d for talker_past_kv; "
                "clamping runtime max_batch_size to %d",
                self._max_batch,
                profile_max_batch,
                profile_max_batch,
            )
            self._max_batch = profile_max_batch
        if profile_max_seq <= 0:
            self._config.max_seq_len = min(self._config.max_seq_len, self._max_seq_len)
            return
        if self._max_seq_len > profile_max_seq:
            logger.warning(
                "Requested max_seq_len=%d exceeds TRT profile max=%d for talker_past_kv; "
                "clamping runtime max_seq_len to %d",
                self._max_seq_len,
                profile_max_seq,
                profile_max_seq,
            )
            self._max_seq_len = profile_max_seq
        self._config.max_seq_len = min(self._config.max_seq_len, self._max_seq_len)

    def _validate_io_dtype_consistency(self) -> None:
        """Validate that manifest-declared I/O dtype matches engine actual I/O dtype.

        Checks that the ``triton_io_float_dtype`` recorded in the manifest matches
        the actual float tensor dtypes exposed by the loaded TRT engine.  Mismatches
        cause incorrect tensor allocation at runtime (e.g. fp32 tensors fed to an
        engine expecting bf16), so this must fail early.
        """
        if not self._manifest or self._fused_engine is None:
            return

        manifest_dtype = self._manifest.get("triton_io_float_dtype", "")
        if not manifest_dtype:
            return

        # Normalize manifest dtype to torch dtype
        _DTYPE_MAP = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        expected_torch_dtype = _DTYPE_MAP.get(manifest_dtype)
        if expected_torch_dtype is None:
            logger.warning(
                "Unknown triton_io_float_dtype '%s' in manifest; "
                "skipping I/O dtype consistency check",
                manifest_dtype,
            )
            return

        # Check a representative float I/O tensor — input_embeds is the main
        # float input and should reflect the engine's I/O precision.
        representative_name = "input_embeds"
        actual_dtype = self._fused_engine.get_tensor_dtype(representative_name)
        if actual_dtype is None:
            # input_embeds may not exist in all engine variants; try another
            for name in ("hidden_states", "new_hidden"):
                actual_dtype = self._fused_engine.get_tensor_dtype(name)
                if actual_dtype is not None:
                    representative_name = name
                    break

        if actual_dtype is None:
            logger.info(
                "Could not find representative float I/O tensor for dtype check; skipping"
            )
            return

        if actual_dtype != expected_torch_dtype:
            raise RuntimeError(
                f"Manifest triton_io_float_dtype={manifest_dtype} "
                f"(torch: {expected_torch_dtype}) does not match engine actual "
                f"I/O dtype for '{representative_name}': {actual_dtype}. "
                f"This usually means the engine was rebuilt with different "
                f"precision settings but the manifest was not updated. "
                f"Re-run 'bash scripts/bash/build_engines.sh' to rebuild the engine."
            )

        logger.info(
            "I/O dtype consistency check passed: manifest=%s, engine=%s (via %s)",
            manifest_dtype,
            actual_dtype,
            representative_name,
        )

    def _validate_prefill_len(self, seq: int, stage: str) -> None:
        if self._max_input_len > 0 and seq > self._max_input_len:
            raise ValueError(
                f"{stage} input length {seq} exceeds TRT profile max_input_len="
                f"{self._max_input_len}. Rebuild Phase B with a larger "
                "--max-input-len or split the request into shorter segments."
            )

    def _discover_c2w_io_names(self) -> None:
        """Detect c2w conv/transconv I/O names from the loaded TRT engine.

        Also caches the static shape (batch=1) for each state tensor so
        we can create zero-initialized dummies for the first prefill call.
        """
        if self._fused_engine is None:
            return
        input_names, output_names = self._fused_engine.get_io_names()
        layout = self._manifest.get("code2wav_fused", {}) if self._manifest else {}
        layout_inputs = layout.get("c2w_state_input_names") or []
        layout_outputs = layout.get("c2w_state_output_names") or []

        def _natural_key(name: str) -> tuple[str, int]:
            m = re.search(r"^(.*?)(\d+)$", name)
            if m:
                return (m.group(1), int(m.group(2)))
            return (name, -1)

        def _ordered(
            names: list[str], prefix: str, layout_names: list[str]
        ) -> list[str]:
            from_layout = [
                n for n in layout_names if n.startswith(prefix) and n in names
            ]
            if from_layout:
                return from_layout
            return sorted((n for n in names if n.startswith(prefix)), key=_natural_key)

        self._c2w_conv_input_names = _ordered(
            input_names,
            "c2w_conv_state_",
            layout_inputs,
        )
        self._c2w_transconv_input_names = _ordered(
            input_names,
            "c2w_transconv_overlap_",
            layout_inputs,
        )
        self._c2w_conv_output_names = _ordered(
            output_names,
            "c2w_new_conv_state_",
            layout_outputs,
        )
        self._c2w_transconv_output_names = _ordered(
            output_names,
            "c2w_new_transconv_overlap_",
            layout_outputs,
        )

        eng = self._fused_engine._engine
        self._c2w_conv_shapes: list[tuple[int, ...]] = []
        for name in self._c2w_conv_input_names:
            shape = eng.get_tensor_shape(name)
            self._c2w_conv_shapes.append(tuple(1 if d == -1 else d for d in shape))
        self._c2w_transconv_shapes: list[tuple[int, ...]] = []
        for name in self._c2w_transconv_input_names:
            shape = eng.get_tensor_shape(name)
            self._c2w_transconv_shapes.append(tuple(1 if d == -1 else d for d in shape))

        logger.info(
            "C2W states: %d conv inputs, %d transconv inputs, "
            "%d conv outputs, %d transconv outputs",
            len(self._c2w_conv_input_names),
            len(self._c2w_transconv_input_names),
            len(self._c2w_conv_output_names),
            len(self._c2w_transconv_output_names),
        )

    @property
    def kv_pool(self) -> KVCachePool:
        return self._kv_pool

    def make_zero_conv_states(self) -> list[torch.Tensor]:
        """Create zero-initialized C2W conv states for one slot."""
        return [
            torch.zeros(shape, device=self._device, dtype=self._config.dtype)
            for shape in self._c2w_conv_shapes
        ]

    def make_zero_transconv_states(self) -> list[torch.Tensor]:
        """Create zero-initialized C2W transconv states for one slot."""
        return [
            torch.zeros(shape, device=self._device, dtype=self._config.dtype)
            for shape in self._c2w_transconv_shapes
        ]

    def make_zero_states_batch(
        self,
        count: int,
    ) -> list[
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ]
    ]:
        """Zero C2W states + ping-pong write buffers for ``count`` slots at once.

        One zero-fill kernel per state shape (instead of per slot); each slot
        receives contiguous [1, ...] row views of the batch allocation, which
        are valid TRT output bindings (distinct data_ptr, contiguous).

        Returns one (conv, transconv, conv_write, transconv_write) tuple per
        slot, matching make_zero_conv/transconv_states + init_pingpong_buffers.
        """

        def _rows(shapes: list[tuple[int, ...]]) -> list[list[torch.Tensor]]:
            per_slot: list[list[torch.Tensor]] = [[] for _ in range(count)]
            for shape in shapes:
                batch = torch.zeros(
                    (count,) + tuple(shape[1:]),
                    device=self._device,
                    dtype=self._config.dtype,
                )
                for i, row in enumerate(batch.split(1, dim=0)):
                    per_slot[i].append(row)
            return per_slot

        conv = _rows(self._c2w_conv_shapes)
        transconv = _rows(self._c2w_transconv_shapes)
        conv_write = _rows(self._c2w_conv_shapes)
        transconv_write = _rows(self._c2w_transconv_shapes)
        return [
            (conv[i], transconv[i], conv_write[i], transconv_write[i])
            for i in range(count)
        ]

    # ------------------------------------------------------------------
    # C2W state arenas (slot-indexed, persistent)
    # ------------------------------------------------------------------

    def _init_c2w_state_arenas(self) -> None:
        """Allocate persistent slot-indexed C2W state arenas (A/B ping-pong).

        Each conv/transconv state gets two [max_slots, ...] tensors; a slot's
        buffers are contiguous row views (valid TRT bindings).  This replaces
        per-admission allocations (variable-size cudaMalloc churn) and lets
        the post-step scatter run as one indexed copy per state instead of
        one small copy per slot per state (~2700 kernel enqueues/step at
        width 128).
        """
        self._c2w_arena_a: Optional[list[torch.Tensor]] = None
        self._c2w_arena_b: Optional[list[torch.Tensor]] = None
        shapes = list(getattr(self, "_c2w_conv_shapes", [])) + list(
            getattr(self, "_c2w_transconv_shapes", [])
        )
        if not shapes:
            return

        def _alloc() -> list[torch.Tensor]:
            return [
                torch.zeros(
                    (self._max_batch,) + tuple(shape[1:]),
                    device=self._device,
                    dtype=self._config.dtype,
                )
                for shape in shapes
            ]

        self._c2w_arena_a = _alloc()
        self._c2w_arena_b = _alloc()
        total_bytes = sum(t.numel() * t.element_size() for t in self._c2w_arena_a) * 2
        logger.info(
            "C2W state arenas: %d states x %d slots x 2 buffers (%.1f MB)",
            len(shapes),
            self._max_batch,
            total_bytes / 1e6,
        )

    @property
    def has_c2w_arenas(self) -> bool:
        return getattr(self, "_c2w_arena_a", None) is not None

    def _arena_row_views(
        self,
        arena: list[torch.Tensor],
        slot_id: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        n_conv = len(self._c2w_conv_shapes)
        rows = [t[slot_id : slot_id + 1] for t in arena]
        return rows[:n_conv], rows[n_conv:]

    def take_zeroed_state_rows(
        self,
        slot_ids: list[int],
    ) -> Optional[
        list[
            tuple[
                list[torch.Tensor],
                list[torch.Tensor],
                list[torch.Tensor],
                list[torch.Tensor],
            ]
        ]
    ]:
        """Zero the arena rows for ``slot_ids`` and hand out per-slot views.

        Read side starts in arena A, write side in arena B (callers must set
        slot.c2w_arena_backed=True and c2w_write_in_a=False after wiring).
        Returns None when arenas are unavailable (stub mode).
        """
        if not self.has_c2w_arenas:
            return None
        ids = torch.tensor(slot_ids, device=self._device, dtype=torch.long)
        for t in self._c2w_arena_a:
            t[ids] = 0
        results = []
        for slot_id in slot_ids:
            conv_a, transconv_a = self._arena_row_views(self._c2w_arena_a, slot_id)
            conv_b, transconv_b = self._arena_row_views(self._c2w_arena_b, slot_id)
            results.append((conv_a, transconv_a, conv_b, transconv_b))
        return results

    def adopt_c2w_states(self, slot: SlotKVState) -> None:
        """Move a slot's freestanding C2W states into its arena rows.

        Called once after a TRT prefill (cold miss / ICL) so every active
        slot is arena-backed and the post-step scatter can stay batched.
        """
        if not self.has_c2w_arenas or slot.c2w_conv_states is None:
            return
        if slot.c2w_arena_backed:
            return
        conv_a, transconv_a = self._arena_row_views(self._c2w_arena_a, slot.slot_id)
        conv_b, transconv_b = self._arena_row_views(self._c2w_arena_b, slot.slot_id)
        for dst, src in zip(conv_a, slot.c2w_conv_states):
            dst.copy_(src)
        for dst, src in zip(transconv_a, slot.c2w_transconv_states or []):
            dst.copy_(src)
        slot.c2w_conv_states = conv_a
        slot.c2w_transconv_states = transconv_a
        slot._c2w_conv_write = conv_b
        slot._c2w_transconv_write = transconv_b
        slot.c2w_arena_backed = True
        slot.c2w_write_in_a = False

    def scatter_c2w_states_batch(
        self,
        slots: List[SlotKVState],
        positions: List[int],
        batch_conv: list[torch.Tensor],
        batch_transconv: list[torch.Tensor],
    ) -> None:
        """Scatter a step's C2W state outputs into arena write rows, batched.

        ``slots`` must all be arena-backed; ``positions`` are their row
        indices in the step output batch.  Slots are grouped by write parity
        (arena A vs B) so each state needs at most two indexed copies.
        """
        groups: dict[bool, tuple[list[int], list[int]]] = {}
        for slot, pos in zip(slots, positions):
            ids, rows = groups.setdefault(slot.c2w_write_in_a, ([], []))
            ids.append(slot.slot_id)
            rows.append(pos)
        n_conv = len(self._c2w_conv_shapes)
        for write_in_a, (ids, rows) in groups.items():
            arena = self._c2w_arena_a if write_in_a else self._c2w_arena_b
            ids_t = torch.tensor(ids, device=self._device, dtype=torch.long)
            rows_t = torch.tensor(rows, device=self._device, dtype=torch.long)
            for j, out in enumerate(batch_conv):
                arena[j][ids_t] = out[rows_t]
            for j, out in enumerate(batch_transconv):
                arena[n_conv + j][ids_t] = out[rows_t]

    def apply_c2w_warm_state(
        self,
        slot: SlotKVState,
        c2w_kv: Any,
        conv_states: Optional[list[Any]],
        transconv_states: Optional[list[Any]],
        frame_idx: int,
    ) -> bool:
        """Install reference-warmed Code2Wav state before the first target frame."""
        if c2w_kv is None or conv_states is None or transconv_states is None:
            return False
        if len(conv_states) != len(self._c2w_conv_input_names):
            logger.warning(
                "Skipping Code2Wav warm state: conv state count %d != expected %d",
                len(conv_states),
                len(self._c2w_conv_input_names),
            )
            return False
        if len(transconv_states) != len(self._c2w_transconv_input_names):
            logger.warning(
                "Skipping Code2Wav warm state: transconv state count %d != expected %d",
                len(transconv_states),
                len(self._c2w_transconv_input_names),
            )
            return False

        max_past = max(1, self._config.c2w_sliding_window - FUSED_CHUNK_T)
        kv = c2w_kv.to(device=self._device, dtype=self._config.dtype).contiguous()
        if kv.shape[3] > max_past:
            kv = kv[:, :, :, -max_past:, :].contiguous()
        slot.c2w_kv = kv
        slot.c2w_conv_states = [
            t.to(device=self._device, dtype=self._config.dtype).contiguous()
            for t in conv_states
        ]
        slot.c2w_transconv_states = [
            t.to(device=self._device, dtype=self._config.dtype).contiguous()
            for t in transconv_states
        ]
        slot.frame_idx = max(0, int(frame_idx))
        return True

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(
        self,
        slot: SlotKVState,
        prefill_embeds: torch.Tensor,
    ) -> tuple[Optional[bytes], bool]:
        """Execute prefill for a single session on the shared CUDA stream.

        Prefill and decode share the same TensorRT execution context and run
        serially. This keeps Triton/standalone deployment memory lower by
        avoiding a second execution context allocation.

        Results are written directly to the pre-allocated KV pool via
        scatter (no per-slot tensor references).

        Args:
            slot: the GPU slot to store resulting KV cache
            prefill_embeds: [1, S, H] bfloat16

        Returns:
            (prefill_audio_bytes, prefill_eos): wav bytes from prefill step
            and whether the first codec token is EOS.
        """
        self._kv_pool.init_kv_tensors(slot)

        if self._fused_engine is None:
            slot.past_len = int(prefill_embeds.shape[1])
            slot.c2w_conv_states = []
            slot.c2w_transconv_states = []
            return None, False

        seq = int(prefill_embeds.shape[1])
        self._validate_prefill_len(seq, "prefill")
        c2w_past_before = slot.c2w_kv

        inputs = self._build_fused_inputs(
            input_embeds=prefill_embeds.to(self._config.dtype),
            slots=[slot],
            batched_talker_kv=None,
            past_seq_lens=None,
            use_dummy_kv=True,
        )

        out_names = self._build_output_names()
        input_snapshot = None
        if self._debug_dumper.enabled:
            input_snapshot = self._debug_dumper.capture(inputs, root_name="inputs")

        stream = self._compute_stream
        _wait_stream_for_current(stream, self._device)
        with torch.cuda.stream(stream):
            raw = self._fused_engine.infer(
                inputs,
                out_names,
                stream,
            )

        stream.synchronize()

        dump_meta = self._build_dump_metadata(
            stage="prefill",
            slots=[slot],
            seq=seq,
            use_dummy_kv=True,
            original_past_lens=[slot.past_len],
            padded_talker_past_len=int(inputs["talker_past_kv"].shape[3]),
        )
        if self._debug_dumper.enabled:
            self._debug_dumper.dump_call(
                metadata=dump_meta,
                inputs=input_snapshot if input_snapshot is not None else inputs,
                outputs=raw,
                inputs_snapshotted=input_snapshot is not None,
            )

        talker_kv = raw.get("talker_new_kv")
        if talker_kv is not None:
            stripped = talker_kv.contiguous()
            if self._kv_pool._preallocate:
                self._kv_pool.scatter_prefill_kv(slot.slot_id, stripped, seq)
            else:
                slot.talker_kv = stripped
        slot.past_len = seq

        c2w_kv = raw.get("c2w_new_kv")
        if c2w_kv is not None:
            c2w_kv = c2w_kv.clone().contiguous()
            if c2w_past_before is not None:
                c2w_kv = _append_c2w_delta(
                    c2w_past_before,
                    c2w_kv,
                    self._config.c2w_sliding_window - FUSED_CHUNK_T,
                )
            if (
                self._kv_pool._preallocate
                and self._kv_pool._c2w_kv_pool is not None
            ):
                # Pooled mode: KV lives right-aligned in the pool row; the
                # slot only tracks its valid length.
                slot.c2w_len = self._kv_pool.write_c2w_right_aligned(
                    slot.slot_id, c2w_kv
                )
                slot.c2w_pooled = True
                slot.c2w_kv = None
            else:
                slot.c2w_kv = c2w_kv
        slot.c2w_conv_states = [raw[n].clone() for n in self._c2w_conv_output_names]
        slot.c2w_transconv_states = [
            raw[n].clone() for n in self._c2w_transconv_output_names
        ]
        slot.init_pingpong_buffers()

        slot.frame_idx = int(slot.frame_idx) + FUSED_CHUNK_T
        # TRT output staging is reused across infer calls on the shared
        # context (same name+shape rebinds the same buffer), so slot state
        # must own copies — otherwise the NEXT serial prefill in the same
        # admission pass clobbers this session's repetition-penalty counts
        # and first decode input.
        updated_tc = raw.get("updated_token_counts")
        slot.token_counts = (
            updated_tc.clone()
            if updated_tc is not None
            else torch.zeros(
                1, self._config.codec_vocab_size, device=self._device, dtype=torch.int64
            )
        )
        codec_sum = raw.get("codec_sum")
        if codec_sum is not None:
            slot.next_embed = codec_sum.clone()
            slot.last_codec_sum = None

        wav = raw.get("wav")
        full_codec = raw.get("full_codec")
        prefill_audio: Optional[bytes] = None
        prefill_eos = False
        if wav is not None:
            prefill_audio = wav.cpu().float().reshape(-1).numpy().tobytes()
        if (
            full_codec is not None
            and int(full_codec[0, 0].item()) == self._codec_eos_id
        ):
            prefill_eos = True
        return prefill_audio, prefill_eos

    def prefill_prefix_only(
        self,
        slot: SlotKVState,
        prefill_embeds: torch.Tensor,
    ) -> None:
        """Build talker KV for a cacheable prefix without committing codec state.

        This supports the semantic split:
          prefill: fixed prefix only
          decode0: consume the first text token + codec BOS

        The fused engine still executes once, but sampling/C2W outputs are
        treated as scratch and must not affect the live slot state.
        """
        self._kv_pool.init_kv_tensors(slot)

        if self._fused_engine is None:
            slot.past_len = int(prefill_embeds.shape[1])
            return

        seq = int(prefill_embeds.shape[1])
        self._validate_prefill_len(seq, "prefill_prefix_only")

        inputs = self._build_fused_inputs(
            input_embeds=prefill_embeds.to(self._config.dtype),
            slots=[slot],
            batched_talker_kv=None,
            past_seq_lens=None,
            use_dummy_kv=True,
            sampling_mode="disabled",
        )

        out_names = self._build_output_names()
        input_snapshot = None
        if self._debug_dumper.enabled:
            input_snapshot = self._debug_dumper.capture(inputs, root_name="inputs")

        stream = self._compute_stream
        _wait_stream_for_current(stream, self._device)
        with torch.cuda.stream(stream):
            raw = self._fused_engine.infer(
                inputs,
                out_names,
                stream,
            )

        stream.synchronize()

        dump_meta = self._build_dump_metadata(
            stage="prefill_prefix_only",
            slots=[slot],
            seq=seq,
            use_dummy_kv=True,
            original_past_lens=[slot.past_len],
            padded_talker_past_len=int(inputs["talker_past_kv"].shape[3]),
        )
        if self._debug_dumper.enabled:
            self._debug_dumper.dump_call(
                metadata=dump_meta,
                inputs=input_snapshot if input_snapshot is not None else inputs,
                outputs=raw,
                inputs_snapshotted=input_snapshot is not None,
            )

        talker_kv = raw.get("talker_new_kv")
        if talker_kv is not None:
            stripped = talker_kv.contiguous()
            if self._kv_pool._preallocate:
                self._kv_pool.scatter_prefill_kv(slot.slot_id, stripped, seq)
            else:
                slot.talker_kv = stripped
        slot.past_len = seq

    def prefill_from_prefix(
        self,
        slot: SlotKVState,
        request_prefill_embeds: torch.Tensor,
    ) -> tuple[Optional[bytes], bool]:
        """Consume cached-prefix suffix embeds and emit the first C2W frame."""
        self._kv_pool.init_kv_tensors(slot)
        seq = int(request_prefill_embeds.shape[1])

        if self._fused_engine is None:
            slot.past_len += seq
            slot.c2w_conv_states = []
            slot.c2w_transconv_states = []
            slot.frame_idx += FUSED_CHUNK_T
            return None, False

        self._validate_prefill_len(seq, "prefill_from_prefix")

        original_past_len = int(slot.past_len)
        c2w_past_before = slot.c2w_kv
        if self._kv_pool._preallocate:
            batched_talker_kv = self._kv_pool.gather_talker_kv(
                [slot.slot_id],
                max(original_past_len, 1),
            )
        else:
            if slot.talker_kv is None:
                raise RuntimeError("Cached-prefix slot is missing talker_kv")
            batched_talker_kv = slot.talker_kv.contiguous()
        past_seq_lens = torch.tensor(
            [original_past_len],
            device=self._device,
            dtype=torch.long,
        )

        inputs = self._build_fused_inputs(
            input_embeds=request_prefill_embeds.to(self._config.dtype),
            slots=[slot],
            batched_talker_kv=batched_talker_kv,
            past_seq_lens=past_seq_lens,
            use_dummy_kv=False,
        )

        out_names = self._build_output_names()
        input_snapshot = None
        if self._debug_dumper.enabled:
            input_snapshot = self._debug_dumper.capture(inputs, root_name="inputs")

        stream = self._compute_stream
        _wait_stream_for_current(stream, self._device)
        with torch.cuda.stream(stream):
            raw = self._fused_engine.infer(
                inputs,
                out_names,
                stream,
            )

        stream.synchronize()

        dump_meta = self._build_dump_metadata(
            stage="prefill_from_prefix",
            slots=[slot],
            seq=seq,
            use_dummy_kv=False,
            original_past_lens=[original_past_len],
            padded_talker_past_len=int(inputs["talker_past_kv"].shape[3]),
        )
        if self._debug_dumper.enabled:
            self._debug_dumper.dump_call(
                metadata=dump_meta,
                inputs=input_snapshot if input_snapshot is not None else inputs,
                outputs=raw,
                inputs_snapshotted=input_snapshot is not None,
            )

        talker_kv = raw.get("talker_new_kv")
        if talker_kv is not None:
            talker_kv = talker_kv.contiguous()
            if self._kv_pool._preallocate:
                self._kv_pool.scatter_talker_kv_delta(
                    [slot.slot_id],
                    talker_kv,
                    [original_past_len],
                )
            elif slot.talker_kv is None:
                slot.talker_kv = talker_kv
            else:
                slot.talker_kv = torch.cat(
                    [slot.talker_kv, talker_kv], dim=3
                ).contiguous()
        slot.past_len = original_past_len + seq

        c2w_kv = raw.get("c2w_new_kv")
        if c2w_kv is not None:
            c2w_kv = c2w_kv.clone().contiguous()
            if c2w_past_before is not None:
                c2w_kv = _append_c2w_delta(
                    c2w_past_before,
                    c2w_kv,
                    self._config.c2w_sliding_window - FUSED_CHUNK_T,
                )
            if (
                self._kv_pool._preallocate
                and self._kv_pool._c2w_kv_pool is not None
            ):
                # Pooled mode: KV lives right-aligned in the pool row; the
                # slot only tracks its valid length.
                slot.c2w_len = self._kv_pool.write_c2w_right_aligned(
                    slot.slot_id, c2w_kv
                )
                slot.c2w_pooled = True
                slot.c2w_kv = None
            else:
                slot.c2w_kv = c2w_kv
        slot.c2w_conv_states = [raw[n].clone() for n in self._c2w_conv_output_names]
        slot.c2w_transconv_states = [
            raw[n].clone() for n in self._c2w_transconv_output_names
        ]
        slot.init_pingpong_buffers()

        slot.frame_idx = int(slot.frame_idx) + FUSED_CHUNK_T
        # Same staging-reuse hazard as prefill(): own copies, not views.
        updated_tc = raw.get("updated_token_counts")
        slot.token_counts = (
            updated_tc.clone()
            if updated_tc is not None
            else torch.zeros(
                1,
                self._config.codec_vocab_size,
                device=self._device,
                dtype=torch.int64,
            )
        )
        codec_sum = raw.get("codec_sum")
        if codec_sum is not None:
            slot.next_embed = codec_sum.clone()
            slot.last_codec_sum = None

        wav = raw.get("wav")
        full_codec = raw.get("full_codec")
        prefill_audio: Optional[bytes] = None
        prefill_eos = False
        if wav is not None:
            prefill_audio = wav.cpu().float().reshape(-1).numpy().tobytes()
        if (
            full_codec is not None
            and int(full_codec[0, 0].item()) == self._codec_eos_id
        ):
            prefill_eos = True
        return prefill_audio, prefill_eos

    # ------------------------------------------------------------------
    # Decode step (async / pipelined)
    # ------------------------------------------------------------------

    def launch_decode_step(self, slots: List[SlotKVState]) -> GPUFuture:
        """Launch one fused decode step for a batch.  Returns immediately.

        Uses the pre-allocated KV pool for zero-copy gather (no pad+cat).
        The CUDA kernels run on self._compute_stream.  Call future.wait()
        to synchronize and get results.
        """

        # Cast once after the cat when dtypes are homogeneous (the decode
        # path keeps next_embed in float32): one elementwise kernel instead
        # of one per slot, bitwise-identical values.
        first_dtype = slots[0].next_embed.dtype
        if all(s.next_embed.dtype == first_dtype for s in slots):
            input_embeds = torch.cat([s.next_embed for s in slots], dim=0).to(
                self._config.dtype
            )
        else:
            input_embeds = torch.cat(
                [s.next_embed.to(self._config.dtype) for s in slots],
                dim=0,
            )

        original_past_lens = [s.past_len for s in slots]

        if self._fused_engine is None:
            for s in slots:
                s.past_len += 1
                s.frame_idx += 1
            return GPUFuture(_slots=slots)

        slot_ids = [s.slot_id for s in slots]
        max_past_len = max(original_past_lens) if original_past_lens else 0
        if max_past_len == 0:
            max_past_len = 1

        if self._graph_decode is not None:
            key = self._graph_decode.bucket(len(slots), max_past_len)
            if key is not None:
                try:
                    return self._launch_decode_step_graphed(
                        slots,
                        slot_ids,
                        input_embeds,
                        original_past_lens,
                        key,
                    )
                except Exception:
                    self._graph_decode_failures += 1
                    logger.exception(
                        "CUDA-graph decode failed (%d/3); falling back to eager",
                        self._graph_decode_failures,
                    )
                    if self._graph_decode_failures >= 3:
                        logger.error("Disabling CUDA-graph decode after 3 failures")
                        # Drop the graphs, dedicated context and most staging
                        # (~2GiB); the talker-KV staging survives as the eager
                        # path's gather arena via self._talker_gather_flat, so
                        # the degraded mode keeps its OOM protection.
                        self._graph_decode = None
                        torch.cuda.empty_cache()

        if self._kv_pool is not None and self._kv_pool._preallocate:
            batched_talker_kv = self._gather_batched_talker_kv(
                slot_ids,
                max_past_len,
            )
            past_seq_lens = torch.tensor(
                original_past_lens,
                device=self._device,
                dtype=torch.long,
            )
        else:
            session_kv = [s.talker_kv for s in slots]
            batched_talker_kv, past_seq_lens = pad_packed_kv(
                session_kv,
                device=self._device,
                dtype=self._config.dtype,
            )

        padded_past_len = (
            int(batched_talker_kv.shape[3]) if batched_talker_kv is not None else 0
        )

        inputs = self._build_fused_inputs(
            input_embeds=input_embeds,
            slots=slots,
            batched_talker_kv=batched_talker_kv,
            past_seq_lens=past_seq_lens,
            use_dummy_kv=False,
        )

        out_names = self._build_output_names()

        # Build ping-pong output overrides: TRT writes directly into
        # each slot's write buffers, avoiding post-step clone/copy.
        output_overrides = self._build_pingpong_overrides(slots)
        input_snapshot = {}
        if self._debug_dumper.enabled and self._debug_dumper.should_dump(
            s.session_id or "" for s in slots
        ):
            input_snapshot = self._debug_dumper.capture(inputs, root_name="inputs")

        _wait_stream_for_current(self._compute_stream, self._device)
        with torch.cuda.stream(self._compute_stream):
            raw = self._fused_engine.infer(
                inputs,
                out_names,
                self._compute_stream,
                output_overrides=output_overrides,
            )

        dump_meta = self._build_dump_metadata(
            stage="decode",
            slots=slots,
            seq=1,
            use_dummy_kv=False,
            original_past_lens=original_past_lens,
            padded_talker_past_len=padded_past_len,
            output_overrides=output_overrides,
        )

        return GPUFuture(
            _compute_stream=self._compute_stream,
            _raw=raw,
            _slots=slots,
            _input_refs=inputs,
            _original_past_lens=original_past_lens,
            _padded_past_len=padded_past_len,
            _seq=1,
            _c2w_conv_output_names=self._c2w_conv_output_names,
            _c2w_transconv_output_names=self._c2w_transconv_output_names,
            _codec_eos_id=self._codec_eos_id,
            _used_pingpong=output_overrides is not None,
            _inputs=input_snapshot,
            _dump_meta=dump_meta,
            _debug_dumper=self._debug_dumper if self._debug_dumper.enabled else None,
        )

    def _ensure_talker_gather_flat(self) -> Optional[torch.Tensor]:
        """Return the persistent KV gather arena, allocating it if needed.

        With graphs on this is the graph staging (set at init).  With graphs
        off it is allocated once at decode max size; on OOM the failure is
        latched and the caller falls back to the legacy transient gather.
        """
        if self._talker_gather_flat is not None or self._talker_gather_flat_failed:
            return self._talker_gather_flat
        cfg = self._config
        numel = (
            self._max_batch
            * cfg.num_layers
            * 2
            * cfg.kv_heads
            * cfg.max_seq_len
            * cfg.head_dim
        )
        try:
            self._talker_gather_flat = torch.zeros(
                numel, dtype=cfg.dtype, device=self._device
            )
            logger.info(
                "Allocated persistent KV gather arena (%.2f GiB)",
                numel * self._talker_gather_flat.element_size() / (1024**3),
            )
        except torch.cuda.OutOfMemoryError:
            self._talker_gather_flat_failed = True
            torch.cuda.empty_cache()
            logger.warning(
                "KV gather arena allocation OOM; falling back to per-step "
                "transient gather (may OOM at large batch x past)"
            )
        return self._talker_gather_flat

    def _gather_batched_talker_kv(
        self,
        slot_ids: List[int],
        max_past_len: int,
    ) -> torch.Tensor:
        """Gather batched talker KV, preferring the persistent arena.

        The arena removes the transient B×L*2×H×past×D allocation (7.5 GiB at
        128×512).  Content and downstream numerics are identical either way —
        only the destination buffer differs.
        """
        cfg = self._config
        shape = (
            len(slot_ids),
            cfg.num_layers * 2,
            cfg.kv_heads,
            max_past_len,
            cfg.head_dim,
        )
        flat = self._ensure_talker_gather_flat()
        numel = math.prod(shape)
        if flat is not None and numel <= flat.numel():
            view = flat[:numel].view(shape)
            self._kv_pool.gather_talker_kv_into(slot_ids, view)
            return view
        try:
            return self._kv_pool.gather_talker_kv(slot_ids, max_past_len)
        except torch.cuda.OutOfMemoryError:
            # Fragmentation is the common cause; reclaim and retry once
            # before letting the step fail.
            torch.cuda.empty_cache()
            logger.warning(
                "Transient KV gather OOM at batch=%d past=%d; retrying after "
                "empty_cache",
                len(slot_ids),
                max_past_len,
            )
            return self._kv_pool.gather_talker_kv(slot_ids, max_past_len)

    def _launch_decode_step_graphed(
        self,
        slots: List[SlotKVState],
        slot_ids: List[int],
        input_embeds: torch.Tensor,
        original_past_lens: List[int],
        key: tuple[int, int],
    ) -> GPUFuture:
        """Decode step via CUDA-graph replay (see GraphedFusedDecode).

        Differences from the eager path:
        - the KV pool gather runs at the bucketed past length; the extra
          columns hold stale-but-finite pool bytes that ``attention_bias``
          masks per slot;
        - the c2w KV is padded to the sliding-window max so it does not
          enter the shape signature;
        - ping-pong output overrides are impossible (graph output addresses
          are fixed), so the engine loop always takes its copy path;
        - outputs that outlive one step (``codec_sum`` feeds next_embed and
          pad tracking) are cloned out of the staging buffers.
        """
        _, past_bucket = key

        # Gather the pool KV straight into the bucket's staging view: no
        # transient batch-KV tensor (up to several GiB at large batch), and
        # run() skips the copy because input and staging share the pointer.
        entry = self._graph_decode.entry(key)
        kv_view = entry["in"]["talker_past_kv"]
        self._kv_pool.gather_talker_kv_into(slot_ids, kv_view)
        past_seq_lens = torch.tensor(
            original_past_lens,
            device=self._device,
            dtype=torch.long,
        )

        inputs = self._build_fused_inputs(
            input_embeds=input_embeds,
            slots=slots,
            batched_talker_kv=kv_view,
            past_seq_lens=past_seq_lens,
            use_dummy_kv=False,
            c2w_past_len_override=self._config.c2w_sliding_window - FUSED_CHUNK_T,
        )

        input_snapshot = {}
        if self._debug_dumper.enabled and self._debug_dumper.should_dump(
            s.session_id or "" for s in slots
        ):
            input_snapshot = self._debug_dumper.capture(inputs, root_name="inputs")

        raw = dict(self._graph_decode.run(key, inputs, len(slots)))
        # codec_sum rows are stored on slots (last_codec_sum / next_embed) and
        # read after the next replay has overwritten the staging buffers, so
        # clone them out.  The clones MUST be enqueued on the compute stream:
        # replay is async on it, and a default-stream clone would race it and
        # read pre-replay staging bytes.
        with torch.cuda.stream(self._compute_stream):
            if raw.get("codec_sum") is not None:
                raw["codec_sum"] = raw["codec_sum"].clone()
            if raw.get("full_codec") is not None:
                raw["full_codec"] = raw["full_codec"].clone()

        dump_meta = self._build_dump_metadata(
            stage="decode",
            slots=slots,
            seq=1,
            use_dummy_kv=False,
            original_past_lens=original_past_lens,
            padded_talker_past_len=past_bucket,
            output_overrides=None,
        )

        return GPUFuture(
            _compute_stream=self._compute_stream,
            _raw=raw,
            _slots=slots,
            _input_refs=inputs,
            _original_past_lens=original_past_lens,
            _padded_past_len=past_bucket,
            _seq=1,
            _c2w_conv_output_names=self._c2w_conv_output_names,
            _c2w_transconv_output_names=self._c2w_transconv_output_names,
            _codec_eos_id=self._codec_eos_id,
            _used_pingpong=False,
            _inputs=input_snapshot,
            _dump_meta=dump_meta,
            _debug_dumper=self._debug_dumper if self._debug_dumper.enabled else None,
        )

    def _build_pingpong_overrides(
        self,
        slots: List[SlotKVState],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build output_overrides dict for ping-pong zero-copy.

        For batch=1: TRT writes directly into the slot's write buffer.
            After the step, flip_c2w_buffers() swaps read↔write — zero copy.
        For batch>1: returns None (no overrides).  TRT writes to its own
            internal buffer; the engine loop then uses copy_c2w_and_flip()
            to scatter results into each slot's pre-allocated write buffer
            and flip — avoids per-step allocation while supporting
            heterogeneous slot ordering.
        """
        if len(slots) != 1:
            return None

        slot = slots[0]
        if not slot.pingpong_ready:
            return None

        overrides: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self._c2w_conv_output_names):
            overrides[name] = slot._c2w_conv_write[idx]
        for idx, name in enumerate(self._c2w_transconv_output_names):
            overrides[name] = slot._c2w_transconv_write[idx]
        return overrides

    def _slot_sampling_generator(self, slot: SlotKVState) -> torch.Generator:
        """Return the deterministic sampling generator owned by one slot."""
        if slot.sampling_generator is not None:
            return slot.sampling_generator

        identity = (
            slot.session_id if slot.session_id is not None else f"slot:{slot.slot_id}"
        )
        seed = _stable_sampling_seed(
            self._random_seed,
            identity,
            slot.segment_idx,
        )
        gen = torch.Generator(device=self._device)
        gen.manual_seed(seed)
        slot.sampling_seed = seed
        slot.sampling_generator = gen
        return gen

    def _build_sampling_noise(
        self,
        slots: List[SlotKVState],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-lane Gumbel noise without cross-lane RNG coupling."""
        cfg = self._config
        gumbel_rows: list[torch.Tensor] = []
        cp_gumbel_rows: list[torch.Tensor] = []

        # rand MUST stay one call per slot per generator — each lane owns a
        # deterministic RNG stream keyed by session_id, and merging draws
        # would change every lane's random sequence (hallucination-seed
        # re-roll).  clamp/log are elementwise, so they run once on the
        # concatenated batch: identical per-element results, ~6 kernels
        # instead of ~6 per slot.
        for slot in slots:
            gen = self._slot_sampling_generator(slot)
            gumbel_rows.append(
                torch.rand(
                    1,
                    cfg.logits_topk,
                    device=self._device,
                    dtype=torch.float32,
                    generator=gen,
                )
            )
            cp_gumbel_rows.append(
                torch.rand(
                    1,
                    cfg.cp_num_stages,
                    cfg.logits_topk,
                    device=self._device,
                    dtype=torch.float32,
                    generator=gen,
                )
            )

        gumbel_u = torch.cat(gumbel_rows, dim=0).clamp(1e-8, 1.0)
        cp_gumbel_u = torch.cat(cp_gumbel_rows, dim=0).clamp(1e-8, 1.0)
        return -torch.log(-torch.log(gumbel_u)), -torch.log(-torch.log(cp_gumbel_u))

    # ------------------------------------------------------------------
    # Input / output name builders
    # ------------------------------------------------------------------

    def _build_fused_inputs(
        self,
        input_embeds: torch.Tensor,
        slots: List[SlotKVState],
        batched_talker_kv: Optional[torch.Tensor],
        past_seq_lens: Optional[torch.Tensor],
        use_dummy_kv: bool,
        sampling_mode: str = "default",
        c2w_past_len_override: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build the full input dict for the fused TRT engine.

        Uses packed KV format: single tensor per cache type.

        ``c2w_past_len_override`` forces the padded c2w KV length (the
        CUDA-graph path fixes it at the sliding-window max so the c2w length
        does not become a graph shape signature; padding is masked via
        ``c2w_attention_bias``).
        """
        cfg = self._config
        batch = int(input_embeds.shape[0])
        seq = int(input_embeds.shape[1])

        if use_dummy_kv:
            if past_seq_lens is None:
                past_seq_lens = uniform_past_seq_lens(batch, 0, self._device)
            attn_bias = padded_attention_bias(
                past_seq_lens,
                seq,
                _FUSED_DUMMY_PAST_LEN,
                self._device,
                cfg.dtype,
            )
        else:
            padded_past_len = (
                int(batched_talker_kv.shape[3]) if batched_talker_kv is not None else 0
            )
            attn_bias = padded_attention_bias(
                past_seq_lens,
                seq,
                padded_past_len,
                self._device,
                cfg.dtype,
            )

        if seq == 1:
            # Decode fast path: one H2D of the past_len vector instead of one
            # arange kernel per slot (values are exact integers — bitwise
            # identical to the general path below).
            position_ids = (
                torch.tensor(
                    [s.past_len for s in slots],
                    device=self._device,
                    dtype=torch.int64,
                )
                .view(batch, 1, 1)
                .expand(batch, 3, seq)
                .unsqueeze(-1)
            )
        else:
            position_ids = torch.stack(
                [
                    torch.arange(
                        s.past_len,
                        s.past_len + seq,
                        device=self._device,
                        dtype=torch.int64,
                    )
                    .unsqueeze(0)
                    .expand(3, seq)
                    for s in slots
                ],
                dim=0,
            ).unsqueeze(-1)

        # One H2D of the frame indices instead of one full+stack per slot
        # (frame_idx < 2^24, exact in float32 — bitwise identical).
        cache_position = (
            torch.tensor(
                [float(s.frame_idx) for s in slots],
                device=self._device,
                dtype=torch.float32,
            )
            .view(batch, 1)
            .expand(batch, FUSED_CHUNK_T)
        )

        tc = torch.cat(
            [
                s.token_counts
                if s.token_counts is not None
                else torch.zeros(
                    1, cfg.codec_vocab_size, device=self._device, dtype=torch.int64
                )
                for s in slots
            ],
            dim=0,
        )

        if sampling_mode == "disabled":
            gumbel = torch.zeros(
                batch,
                cfg.logits_topk,
                device=self._device,
                dtype=torch.float32,
            )
            cp_gumbel = torch.zeros(
                batch,
                cfg.cp_num_stages,
                cfg.logits_topk,
                device=self._device,
                dtype=torch.float32,
            )
            temperature = torch.ones(
                batch,
                1,
                device=self._device,
                dtype=torch.float32,
            )
            penalty = torch.ones(
                batch,
                1,
                device=self._device,
                dtype=torch.float32,
            )
        elif self._do_sample:
            gumbel, cp_gumbel = self._build_sampling_noise(slots)
            temperature = torch.full(
                (batch, 1),
                self._temperature,
                device=self._device,
                dtype=torch.float32,
            )
            penalty = torch.full(
                (batch, 1),
                self._repetition_penalty,
                device=self._device,
                dtype=torch.float32,
            )
        else:
            gumbel = torch.zeros(
                batch,
                cfg.logits_topk,
                device=self._device,
                dtype=torch.float32,
            )
            cp_gumbel = torch.zeros(
                batch,
                cfg.cp_num_stages,
                cfg.logits_topk,
                device=self._device,
                dtype=torch.float32,
            )
            temperature = torch.ones(
                batch,
                1,
                device=self._device,
                dtype=torch.float32,
            )
            penalty = torch.full(
                (batch, 1),
                self._repetition_penalty,
                device=self._device,
                dtype=torch.float32,
            )

        d: Dict[str, torch.Tensor] = {
            "input_embeds": input_embeds.contiguous(),
            "position_ids": position_ids.contiguous(),
            "attention_bias": attn_bias.contiguous(),
            "token_counts": tc.contiguous(),
            "gumbel_noise": gumbel.contiguous(),
            "cp_gumbel_noise": cp_gumbel.contiguous(),
            "temperature": temperature.contiguous(),
            "penalty": penalty.contiguous(),
            "cache_position": cache_position.contiguous(),
        }

        # --- Talker KV (packed) ---
        if use_dummy_kv:
            d["talker_past_kv"] = torch.zeros(
                batch,
                cfg.num_layers * 2,
                cfg.kv_heads,
                _FUSED_DUMMY_PAST_LEN,
                cfg.head_dim,
                device=self._device,
                dtype=cfg.dtype,
            )
        else:
            d["talker_past_kv"] = batched_talker_kv.contiguous()

        # --- C2W KV (packed) with sliding window safety clamp ---
        # Relationship: c2w_attention_bias key_dim = c2w_past_len + FUSED_CHUNK_T
        # TRT profile max for c2w_past_kv dim3 = c2w_sliding_window - FUSED_CHUNK_T
        # because attention key_total = past + chunk_T <= c2w_sliding_window.
        # Different slots may be at different decode steps, so c2w_kv lengths
        # can differ.  We pad to the max length and mask padding in attn bias.
        c2w_window = cfg.c2w_sliding_window
        c2w_max_past = c2w_window - FUSED_CHUNK_T

        per_slot_c2w_lens = []
        for s in slots:
            if s.c2w_pooled:
                per_slot_c2w_lens.append(min(int(s.c2w_len), c2w_max_past))
            elif s.c2w_kv is not None:
                per_slot_c2w_lens.append(min(int(s.c2w_kv.shape[3]), c2w_max_past))
            else:
                per_slot_c2w_lens.append(0)
        c2w_past_len = max(per_slot_c2w_lens) if per_slot_c2w_lens else 0
        if c2w_past_len_override is not None:
            c2w_past_len = max(c2w_past_len, int(c2w_past_len_override))
        if c2w_past_len < 1:
            c2w_past_len = 1

        c2w_key_total = c2w_past_len + FUSED_CHUNK_T

        c2w_attn = torch.zeros(
            batch,
            1,
            FUSED_CHUNK_T,
            c2w_key_total,
            device=self._device,
            dtype=cfg.dtype,
        )
        for bi, sl in enumerate(per_slot_c2w_lens):
            pad_cols = c2w_past_len - sl
            if pad_cols > 0:
                c2w_attn[bi, :, :, :pad_cols] = float("-inf")
        d["c2w_attention_bias"] = c2w_attn.contiguous()

        c2w_d1 = cfg.n_c2w_layers * 2
        c2w_d2 = cfg.c2w_kv_heads
        c2w_head = cfg.c2w_head_dim
        pooled_c2w = (
            self._kv_pool is not None
            and getattr(self._kv_pool, "_c2w_kv_pool", None) is not None
            and all(s.c2w_pooled for s in slots)
        )
        if pooled_c2w:
            # Rows are right-aligned, so "last c2w_past_len columns" is
            # bitwise identical to the legacy left-pad + cat layout — one
            # gather replaces the per-slot slice/pad/cat below.
            pool = self._kv_pool._c2w_kv_pool
            width = int(pool.shape[3])
            if batch == 1:
                sid = slots[0].slot_id
                d["c2w_past_kv"] = pool[
                    sid : sid + 1, :, :, width - c2w_past_len :, :
                ].contiguous()
            else:
                ids = torch.tensor(
                    [s.slot_id for s in slots],
                    device=self._device,
                    dtype=torch.long,
                )
                d["c2w_past_kv"] = pool[
                    ids, :, :, width - c2w_past_len :, :
                ].contiguous()
        elif batch == 1 and c2w_past_len_override is None:
            s = slots[0]
            if s.c2w_kv is not None:
                kv = s.c2w_kv
                if kv.shape[3] > c2w_max_past:
                    kv = kv[:, :, :, -c2w_max_past:, :]
            else:
                kv = torch.zeros(
                    1,
                    c2w_d1,
                    c2w_d2,
                    c2w_past_len,
                    c2w_head,
                    device=self._device,
                    dtype=cfg.dtype,
                )
            d["c2w_past_kv"] = kv.contiguous()
        else:
            padded = []
            for s, sl in zip(slots, per_slot_c2w_lens):
                if s.c2w_kv is not None:
                    kv = s.c2w_kv
                    if kv.shape[3] > c2w_max_past:
                        kv = kv[:, :, :, -c2w_max_past:, :]
                else:
                    kv = torch.zeros(
                        1,
                        c2w_d1,
                        c2w_d2,
                        0,
                        c2w_head,
                        device=self._device,
                        dtype=cfg.dtype,
                    )
                pad_cols = c2w_past_len - sl
                if pad_cols > 0:
                    kv = torch.nn.functional.pad(kv, (0, 0, pad_cols, 0))
                padded.append(kv)
            d["c2w_past_kv"] = torch.cat(padded, dim=0).contiguous()

        # --- C2W conv/transconv states (individual, heterogeneous shapes) ---
        has_conv = bool(slots[0].c2w_conv_states)
        for idx, name in enumerate(self._c2w_conv_input_names):
            if has_conv:
                if batch == 1:
                    d[name] = slots[0].c2w_conv_states[idx].contiguous()
                else:
                    d[name] = torch.cat(
                        [s.c2w_conv_states[idx] for s in slots],
                        dim=0,
                    ).contiguous()
            else:
                shape = list(self._c2w_conv_shapes[idx])
                shape[0] = batch
                d[name] = torch.zeros(shape, device=self._device, dtype=cfg.dtype)

        has_transconv = bool(slots[0].c2w_transconv_states)
        for idx, name in enumerate(self._c2w_transconv_input_names):
            if has_transconv:
                if batch == 1:
                    d[name] = slots[0].c2w_transconv_states[idx].contiguous()
                else:
                    d[name] = torch.cat(
                        [s.c2w_transconv_states[idx] for s in slots],
                        dim=0,
                    ).contiguous()
            else:
                shape = list(self._c2w_transconv_shapes[idx])
                shape[0] = batch
                d[name] = torch.zeros(shape, device=self._device, dtype=cfg.dtype)

        return d

    def _build_output_names(self) -> List[str]:
        names = [
            "wav",
            "codec_sum",
            "full_codec",
            "hidden",
            "logits",
            "updated_token_counts",
            "talker_new_kv",
            "c2w_new_kv",
        ]
        names.extend(self._c2w_conv_output_names)
        names.extend(self._c2w_transconv_output_names)
        return names

    def _build_dump_metadata(
        self,
        *,
        stage: str,
        slots: List[SlotKVState],
        seq: int,
        use_dummy_kv: bool,
        original_past_lens: List[int],
        padded_talker_past_len: int,
        output_overrides: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        return {
            "stage": stage,
            "batch_size": len(slots),
            "seq_len": seq,
            "use_dummy_kv": use_dummy_kv,
            "slot_ids": [int(s.slot_id) for s in slots],
            "slot_session_ids": [s.session_id or "" for s in slots],
            "slot_segment_indices": [int(s.segment_idx) for s in slots],
            "slot_prefill_sources": [str(s.prefill_source or "") for s in slots],
            "slot_past_len_before": [int(s.past_len) for s in slots],
            "slot_frame_idx_before": [int(s.frame_idx) for s in slots],
            "slot_text_idx_before": [int(s.text_idx) for s in slots],
            # L3 step_decision↔tensor alignment: the per-slot sampling seed that
            # produced this step's Gumbel noise (None until first decode step).
            "slot_sampling_seeds": [getattr(s, "sampling_seed", None) for s in slots],
            "slot_trailing_len": [len(s.trailing) for s in slots],
            "slot_has_next_embed": [s.next_embed is not None for s in slots],
            "slot_has_last_codec_sum": [s.last_codec_sum is not None for s in slots],
            "slot_c2w_len_before": [
                int(s.c2w_kv.shape[3]) if s.c2w_kv is not None else 0 for s in slots
            ],
            "original_talker_past_lens": [int(v) for v in original_past_lens],
            "padded_talker_past_len": int(padded_talker_past_len),
            "c2w_conv_input_names": list(self._c2w_conv_input_names),
            "c2w_conv_output_names": list(self._c2w_conv_output_names),
            "c2w_transconv_input_names": list(self._c2w_transconv_input_names),
            "c2w_transconv_output_names": list(self._c2w_transconv_output_names),
            "output_override_names": sorted(output_overrides.keys())
            if output_overrides
            else [],
            "config": {
                "num_layers": int(self._config.num_layers),
                "kv_heads": int(self._config.kv_heads),
                "head_dim": int(self._config.head_dim),
                "hidden_size": int(self._config.hidden_size),
                "codec_vocab_size": int(self._config.codec_vocab_size),
                "codec_eos_id": int(self._codec_eos_id),
                "logits_topk": int(self._config.logits_topk),
                "cp_num_stages": int(self._config.cp_num_stages),
                "n_c2w_layers": int(self._config.n_c2w_layers),
                "c2w_kv_heads": int(self._config.c2w_kv_heads),
                "c2w_head_dim": int(self._config.c2w_head_dim),
                "c2w_sliding_window": int(self._config.c2w_sliding_window),
            },
        }

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self, n_rounds: int = 3) -> None:
        """Run dummy inferences to warm up TRT engines and CUDA caches.

        Triggers JIT compilation of TRT tactics and populates GPU L2 cache.
        Runs on the compute stream with a temporary slot.
        """
        if self._fused_engine is None or self._kv_pool is None:
            logger.info("Warmup skipped (no TRT engine)")
            return

        logger.info("Warmup: running %d rounds ...", n_rounds)
        cfg = self._config
        dummy_slot = self._kv_pool.allocate("__warmup__")
        if dummy_slot is None:
            logger.warning("Warmup skipped (no free slot)")
            return

        try:
            dummy_embeds = torch.randn(
                1,
                4,
                cfg.hidden_size,
                device=self._device,
                dtype=cfg.dtype,
            )
            _ = self.prefill(dummy_slot, dummy_embeds)

            dummy_slot.next_embed = torch.randn(
                1,
                1,
                cfg.hidden_size,
                device=self._device,
                dtype=torch.float32,
            )

            for i in range(n_rounds):
                future = self.launch_decode_step([dummy_slot])
                output = future.wait()
                if self._kv_pool._preallocate and output.batch_talker_kv is not None:
                    self._kv_pool.scatter_talker_kv_delta(
                        [dummy_slot.slot_id],
                        output.batch_talker_kv,
                        [dummy_slot.past_len],
                    )
                elif output.batch_talker_kv is not None:
                    dummy_slot.talker_kv = torch.cat(
                        [dummy_slot.talker_kv, output.batch_talker_kv[:1]],
                        dim=3,
                    )
                if output.batch_c2w_kv is not None:
                    if dummy_slot.c2w_kv is None:
                        dummy_slot.c2w_kv = output.batch_c2w_kv[:1].clone()
                    else:
                        dummy_slot.c2w_kv = _append_c2w_delta(
                            dummy_slot.c2w_kv,
                            output.batch_c2w_kv[:1],
                            self._config.c2w_sliding_window - 1,
                        )
                if output.used_pingpong and dummy_slot.pingpong_ready:
                    dummy_slot.flip_c2w_buffers()
                dummy_slot.past_len += 1
                dummy_slot.frame_idx += 1
                if output.codec_sum is not None:
                    dummy_slot.next_embed = output.codec_sum[:1]

            torch.cuda.synchronize(self._device)
            logger.info("Warmup complete (%d rounds)", n_rounds)
        finally:
            self._kv_pool.release(dummy_slot.slot_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release GPU resources."""
        self._fused_engine = None
        self._kv_pool = None
        torch.cuda.empty_cache()
        logger.info("Executor shutdown")
