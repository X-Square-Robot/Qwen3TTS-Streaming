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
from numbers import Integral, Real
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
from ..runtime.release_gate import (
    ReleaseCapability,
    ReleaseGate,
    evaluate_release_gate,
)
from ..core.speech_state import SpeechStateCapability
from ..core.speech_state_bundle import (
    SpeechStateBundleValidation,
    validate_speech_state_bundle,
)
from ..core.speech_state_model import (
    SpeechStateCursorPolicy,
    SpeechStateModelContract,
)
from ..core.native_cursor import (
    CURSOR_RECURRENT_INPUT_BINDINGS,
    CURSOR_RECURRENT_OUTPUT_BINDINGS,
)
from ..config import ModelPackagePaths
from .slot_snapshot import SlotOwnedSnapshot, StandaloneSlotSnapshot
from .speech_state import (
    SegmentRuntimeMetadata,
    SpeechStateAdapter,
    SpeechStateContractError,
    SpeechStateSnapshotBundle,
    capability_from_adapter,
    coerce_speech_state_adapter,
)
from .kv_cache_pool import KVCachePool, ModelConfig, SlotKVState

logger = logging.getLogger(__name__)

FUSED_CHUNK_T = 1
_FUSED_DUMMY_PAST_LEN = 1
_MAX_TORCH_SEED = (1 << 63) - 1
_NATIVE_CURSOR_HEAD_FILENAME = "qwen3_tts_12hz_la1_seed0.pt"


def _stable_sampling_seed(base_seed: int, *parts: object) -> int:
    """Derive a deterministic torch seed from stable logical identifiers."""
    h = hashlib.blake2b(digest_size=16)
    h.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little") & _MAX_TORCH_SEED


def _select_cuda_graph_profile(
    num_profiles: int,
    *,
    cursor_enabled: bool,
    requested: str = "",
) -> int:
    """Select the decode-only profile for graph replay when available.

    Prefill remains on the fused engine's shared context/profile 0.  Cursor is
    a decoder-only observer, so its graph replay must use the same compact
    decode profile as the standard route; graph/eager parity is meaningful only
    when both executions use the same optimization profile.
    """
    if num_profiles <= 1:
        return 0
    default = 1
    value = str(requested or "").strip()
    if not value:
        return default
    try:
        profile_idx = int(value)
    except ValueError:
        return default
    if not 0 <= profile_idx < num_profiles:
        return default
    return profile_idx


def _sha256_file(path: Path) -> str:
    """Hash a bundle-owned artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    def select_optimization_profile(
        self, profile_idx: int, stream: torch.cuda.Stream
    ) -> None:
        """Select an optimization profile before direct diagnostic execution.

        Executor-owned CUDA Graph contexts select their profile during graph
        setup.  Standalone validation callers use this narrow hook to replay
        the same profile explicitly instead of relying on TensorRT's context
        default, which is especially important for recurrent cursor plans.
        """
        if self._engine is None or self._context is None:
            raise RuntimeError("TRT engine must be loaded before selecting a profile")
        count = int(getattr(self._engine, "num_optimization_profiles", 1))
        if not 0 <= int(profile_idx) < count:
            raise ValueError(
                f"optimization profile {profile_idx} is outside [0, {count})"
            )
        if count > 1 and not self._context.set_optimization_profile_async(
            int(profile_idx), stream.cuda_stream
        ):
            raise RuntimeError(f"could not select optimization profile {profile_idx}")
        stream.synchronize()
        self._prev_input_shapes.clear()
        self._output_buffers.clear()

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
    _cursor_output_names: List[str] = field(default_factory=list)
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
        hidden = raw.get("hidden")
        full_codec = raw.get("full_codec")
        codec0 = raw.get("codec0")
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
        if codec0 is not None:
            codec0_list = codec0.cpu().tolist()
            eos_list = [t == codec_eos_id for t in codec0_list]
        elif full_codec is not None:
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
            # EOS decode emits a waveform tensor for graph shape stability,
            # but that PCM is not a spoken frame and must never be published.
            if wav_cpu is not None and not eos_list[row_idx]:
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
            cursor_outputs={
                name: raw.get(name)
                for name in self._cursor_output_names
                if raw.get(name) is not None
            },
            codec_sum=codec_sum,
            hidden=hidden,
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
    batch_c2w_conv: Optional[List[Optional[torch.Tensor]]] = None
    batch_c2w_transconv: Optional[List[Optional[torch.Tensor]]] = None
    cursor_outputs: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    codec_sum: Optional[torch.Tensor] = None
    # Optional final Talker hidden state for method-layer continuity policies.
    # It is an observation only; the core engine never uses it as a second
    # inference path.
    hidden: Optional[torch.Tensor] = None
    updated_tc: Optional[torch.Tensor] = None
    used_pingpong: bool = False


@dataclass
class C2WArenaSnapshot:
    """Detached A/B C2W state rows for one pool allocation."""

    read_conv: list[torch.Tensor]
    read_transconv: list[torch.Tensor]
    write_conv: list[torch.Tensor]
    write_transconv: list[torch.Tensor]
    write_in_a: bool
    source_slot_id: int
    source_allocation_epoch: int

    @property
    def tensor_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for group in (
                self.read_conv,
                self.read_transconv,
                self.write_conv,
                self.write_transconv,
            )
            for tensor in group
        )

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
        self._profile_idx = int(profile_idx)
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
        speech_state_adapter: SpeechStateAdapter | None = None,
        package_paths: ModelPackagePaths | None = None,
    ):
        self._package_paths = package_paths
        if package_paths is not None:
            engine_dir = package_paths.engine_dir
            weights_dir = package_paths.weights_dir
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
        # The official model path has no acoustic state handoff yet.  Keeping
        # a no-op adapter here makes the capability boundary explicit without
        # changing any launch/prefill/decode behavior.
        self._speech_state_adapter: SpeechStateAdapter = coerce_speech_state_adapter(
            speech_state_adapter
        )

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
        self._cursor_enabled = False
        # Set only after a cursor-enabled fused graph exposes the complete
        # recurrent input/output ABI. Manifest and release gates still decide
        # whether a model may advertise successor handoff.
        self._cursor_state_handoff_enabled = False
        self._cursor_input_names: list[str] = []
        self._cursor_output_names: list[str] = []
        self._cursor_max_labels = 0
        self._cursor_d = 0
        self._cursor_history = 0
        self._cursor_vocab_size = 0
        self._cursor_head_path: Optional[Path] = None
        self._manifest: dict = {}
        self._speech_state_bundle_validation: SpeechStateBundleValidation = SpeechStateBundleValidation(
            False, "bundle_not_loaded"
        )
        self._release_gate: ReleaseGate = ReleaseGate.disabled()
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
    def speech_state_fingerprints(self) -> tuple[str, str]:
        """Verified model/runtime identity, empty until the bundle declares it."""
        manifest = getattr(self, "_manifest", {})
        declared = manifest.get("speech_state") if isinstance(manifest, dict) else {}
        declared = declared or {}
        if not isinstance(declared, dict):
            declared = {}

        def identity(value: Any) -> str:
            return value.strip() if isinstance(value, str) else ""

        return (
            identity(
                getattr(self, "_speech_state_model_fingerprint", "")
                or declared.get("model_fingerprint", "")
            ),
            identity(
                getattr(self, "_speech_state_runtime_fingerprint", "")
                or declared.get("runtime_fingerprint", "")
            ),
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

    @property
    def speech_state_adapter(self) -> SpeechStateAdapter:
        """Backend-owned adapter; callers must not retain its state payload."""

        return self._speech_state_adapter

    @property
    def speech_state_capability(self) -> SpeechStateCapability:
        """Stable, fail-closed capability advertised by the adapter."""
        capability = capability_from_adapter(self._speech_state_adapter)
        if not capability.supported:
            return capability

        # An adapter describes implementation support, but the loaded bundle
        # must also declare the exact model contract it was validated against.
        # Keeping this gate at the executor boundary prevents a generic adapter
        # from accidentally enabling handoff for an incompatible TRT plan.
        contract = self.speech_state_model_contract
        if contract is None:
            return SpeechStateCapability.disabled()
        if (
            capability.supports_segment_handoff
            and not contract.supports_segment_handoff
        ):
            return SpeechStateCapability.disabled()
        if (
            capability.supports_context_rollover
            and not contract.supports_segment_handoff
        ):
            return SpeechStateCapability.disabled()
        if capability.transfer is not contract.transfer:
            return SpeechStateCapability.disabled()
        # The current fused runtime has no cursor recurrent-state restore ABI.
        # A model contract asking for MIGRATE must therefore stay closed; a
        # fresh successor cursor would otherwise be mixed with inherited audio
        # state. REANCHOR/DISABLE remain explicit model-level policies.
        if (
            contract.cursor_policy is SpeechStateCursorPolicy.MIGRATE
            and not bool(getattr(self, "_cursor_state_handoff_enabled", False))
        ):
            return SpeechStateCapability.disabled()
        bundle_validation = getattr(self, "_speech_state_bundle_validation", None)
        if bundle_validation is not None and not bundle_validation.verified:
            return SpeechStateCapability.disabled()
        release_gate = getattr(self, "_release_gate", None)
        if release_gate is not None and not release_gate.verified(
            ReleaseCapability.SPEECH_STATE
        ):
            return SpeechStateCapability.disabled()
        return capability

    @property
    def speech_state_capability_reason(self) -> str:
        """Stable public explanation for the current fail-closed decision."""
        # Once a package has been loaded, its manifest-owned bundle result is
        # more actionable than the default NullSpeechStateAdapter.  Keeping
        # this reason ahead of adapter inspection makes standalone and Triton
        # capability discovery explain the same artifact failure.  The
        # sentinel is retained for pre-load and legacy injected executors.
        bundle_validation = getattr(self, "_speech_state_bundle_validation", None)
        if (
            bundle_validation is not None
            and not bundle_validation.verified
            and str(bundle_validation.reason or "") != "bundle_not_loaded"
        ):
            return str(bundle_validation.reason)
        # A verified package is authoritative even when the runtime still has
        # the default NullSpeechStateAdapter.  This keeps standalone and the
        # Triton compatibility layer aligned on the release-gate reason.
        if bundle_validation is not None and bundle_validation.verified:
            release_gate = getattr(self, "_release_gate", None)
            if release_gate is not None and not release_gate.verified(
                ReleaseCapability.SPEECH_STATE
            ):
                return release_gate.reason(ReleaseCapability.SPEECH_STATE)
        adapter_capability = capability_from_adapter(self._speech_state_adapter)
        if not adapter_capability.supported:
            return "adapter_disabled"
        model_fingerprint, runtime_fingerprint = self.speech_state_fingerprints
        if not model_fingerprint.strip() or not runtime_fingerprint.strip():
            return "missing_runtime_fingerprint"
        contract = self.speech_state_model_contract
        if contract is None:
            return "missing_or_invalid_model_contract"
        if (
            adapter_capability.supports_segment_handoff
            and not contract.supports_segment_handoff
        ):
            return "model_contract_disallows_segment_handoff"
        if (
            adapter_capability.supports_context_rollover
            and not contract.supports_segment_handoff
        ):
            return "model_contract_disallows_context_rollover"
        if adapter_capability.transfer is not contract.transfer:
            return "transfer_class_mismatch"
        if (
            contract.cursor_policy is SpeechStateCursorPolicy.MIGRATE
            and not bool(getattr(self, "_cursor_state_handoff_enabled", False))
        ):
            return "cursor_state_handoff_unavailable"
        bundle_validation = getattr(self, "_speech_state_bundle_validation", None)
        if bundle_validation is not None and not bundle_validation.verified:
            return bundle_validation.reason
        release_gate = getattr(self, "_release_gate", None)
        if release_gate is not None and not release_gate.verified(
            ReleaseCapability.SPEECH_STATE
        ):
            return release_gate.reason(ReleaseCapability.SPEECH_STATE)
        if not self.speech_state_capability.supported:
            return "runtime_gate_disabled"
        return "enabled"

    @property
    def speech_state_model_contract(self) -> Optional[SpeechStateModelContract]:
        """Return the validated bundle-owned model contract, if present."""
        manifest = getattr(self, "_manifest", {})
        section = manifest.get("speech_state") if isinstance(manifest, dict) else None
        if not isinstance(section, dict):
            return None
        raw_contract = section.get("model_contract")
        if raw_contract is None:
            return None
        try:
            contract = SpeechStateModelContract.from_mapping(raw_contract)
            model_fingerprint, runtime_fingerprint = self.speech_state_fingerprints
            if (
                not model_fingerprint.strip()
                or not runtime_fingerprint.strip()
                or contract.model_fingerprint != model_fingerprint.strip()
            ):
                return None
            return contract
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring malformed speech_state.model_contract; "
                "state capability remains disabled",
                exc_info=True,
            )
            return None

    @property
    def native_cursor_enabled(self) -> bool:
        """Whether the loaded fused plan exposes a validated cursor branch."""
        return bool(getattr(self, "_cursor_enabled", False))

    @property
    def native_cursor_capability(self) -> dict:
        """Manifest capability used by request-level progress routing."""
        value = getattr(self, "_manifest", {}).get("native_cursor") or {}
        result = dict(value) if self.native_cursor_enabled else {"enabled": False}
        if not self.native_cursor_enabled:
            return result
        result["enabled"] = True
        progress_available = result.get("progress_available", False)
        if progress_available is not True:
            if "progress_available" in result and not isinstance(
                progress_available, bool
            ):
                result["reason"] = "malformed_native_cursor_capability"
            result["progress_available"] = False
            return result
        # Release evidence qualifies a published artifact; it must not hide a
        # cursor route that has passed runtime graph/head/labelizer admission.
        # Offline ASR and performance evidence remain release-pipeline gates.
        result["progress_available"] = True
        result.pop("reason", None)
        result["supported_progress_modes"] = ["native", "ema", "disabled"]
        return result

    @property
    def native_cursor_head_path(self) -> Optional[Path]:
        """Model-owned head asset that opted this package into native cursor."""
        return self._cursor_head_path if self.native_cursor_enabled else None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load TRT engines, embedding weights, and initialize KV pool."""
        fused_plan: Optional[Path] = None
        if self._engine_dir:
            package_paths = getattr(self, "_package_paths", None)
            if package_paths is not None:
                candidate = Path(package_paths.runtime_artifact_path)
                if candidate.suffix.lower() in {".plan", ".engine"} and candidate.is_file():
                    fused_plan = candidate
                manifest_path = Path(package_paths.manifest_path)
                bundle_root = Path(package_paths.package_dir)
                evidence_paths = (
                    bundle_root / "capability_evidence.json",
                    self._engine_dir / "capability_evidence.json",
                )
            else:
                mp = self._engine_dir / "model.plan"
                te = self._engine_dir / "talker_code2wav_fused.engine"
                if mp.exists():
                    fused_plan = mp
                elif te.exists():
                    fused_plan = te
                manifest_path = self._engine_dir / "triton_manifest.json"
                bundle_root = (
                    self._engine_dir.parent
                    if self._engine_dir.name == "runtime"
                    else self._engine_dir
                )
                evidence_paths = (
                    bundle_root / "capability_evidence.json",
                    self._engine_dir / "capability_evidence.json",
                )
            if manifest_path.exists():
                with open(manifest_path) as f:
                    self._manifest = json.load(f)
            self._speech_state_bundle_validation = validate_speech_state_bundle(
                self._manifest,
                bundle_root=bundle_root,
                runtime_artifact_path=fused_plan,
            )
            evidence = None
            for evidence_path in (
                self._engine_dir / "capability_evidence.json",
                bundle_root / "capability_evidence.json",
            ):
                if not evidence_path.is_file():
                    continue
                try:
                    loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    loaded = None
                if isinstance(loaded, dict):
                    evidence = loaded
                    break
            self._release_gate = evaluate_release_gate(self._manifest, evidence)
        if self._engine_dir and fused_plan is not None:
            self._fused_engine = TRTEngine(
                str(fused_plan),
                self._device,
            )
            self._fused_engine.load()
            self._apply_runtime_profile_limits()
            self._validate_io_dtype_consistency()
            self._discover_c2w_io_names()
            self._discover_cursor_io_names()

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
        if self._cursor_enabled:
            shapes.update(
                {
                    "cursor_label_ids": (B, self._cursor_max_labels),
                    "cursor_label_count": (B,),
                    "cursor_active": (B,),
                    "cursor_mu_in": (B,),
                    "cursor_frames_since_advance_in": (B,),
                    "cursor_delta_history_in": (B, 8),
                    "cursor_conv_history_in": (B, self._cursor_history, self._cursor_d),
                    "cursor_last_trunk_input_in": (B, self._cursor_d),
                    "cursor_seen_frames_in": (B,),
                    "cursor_text_start_frame": (B,),
                    "cursor_override_valid": (B,),
                    "cursor_override_mu": (B,),
                }
            )
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
        # Profile 0 is shared by prefill on the base context. The dedicated
        # graph context uses the compact decode-only profile 1 for both the
        # standard and native-cursor routes.
        num_profiles = int(
            getattr(self._fused_engine._engine, "num_optimization_profiles", 1)
        )
        if num_profiles > 1:
            requested_profile = os.environ.get("ENGINE_CUDA_GRAPH_PROFILE", "").strip()
            profile_idx = _select_cuda_graph_profile(
                num_profiles,
                cursor_enabled=self._cursor_enabled,
                requested=requested_profile,
            )
            if requested_profile and str(profile_idx) != requested_profile:
                logger.warning(
                    "Using CUDA Graph profile %d for decode/parity safety; "
                    "requested profile was %r",
                    profile_idx,
                    requested_profile,
                )
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

    def _discover_cursor_io_names(self) -> None:
        """Discover optional fused native-cursor bindings from the manifest.

        A standard plan has no cursor capability and follows the legacy path.
        For a cursor-enabled plan, names come from the export manifest so an
        ABI change is explicit rather than inferred from tensor ordering.
        """
        capability = self._manifest.get("native_cursor") or {}
        if capability.get("enabled") is not True or self._fused_engine is None:
            return
        head_path = (
            self._weights_dir / _NATIVE_CURSOR_HEAD_FILENAME
            if self._weights_dir
            else None
        )
        if head_path is None or not head_path.is_file():
            logger.warning(
                "Native cursor plan is declared but "
                f"weights/{_NATIVE_CURSOR_HEAD_FILENAME} is missing; "
                "disabling native cursor for this package"
            )
            return
        expected_head_sha = str(capability.get("cursor_head_sha256", "") or "").strip().lower()
        if expected_head_sha:
            if not re.fullmatch(r"[0-9a-f]{64}", expected_head_sha):
                logger.warning(
                    "Native cursor manifest has an invalid cursor_head_sha256; "
                    "disabling native cursor for this package"
                )
                return
            actual_head_sha = _sha256_file(head_path)
            if actual_head_sha != expected_head_sha:
                logger.warning(
                    "Native cursor head hash mismatch; disabling native cursor: "
                    "manifest=%s actual=%s path=%s",
                    expected_head_sha,
                    actual_head_sha,
                    head_path,
                )
                return
        self._cursor_head_path = head_path
        if str(self._manifest.get("variant", "")) != "custom-1.7b":
            raise RuntimeError(
                "native cursor is only validated for the custom-1.7b artifact"
            )
        expected_codec_vocab = int(getattr(self._config, "codec_vocab_size", 3072))
        if int(capability.get("codec_vocab_size", expected_codec_vocab)) != expected_codec_vocab:
            raise RuntimeError(
                "native cursor/Talker codec vocabulary mismatch: "
                f"cursor={capability.get('codec_vocab_size')} talker={expected_codec_vocab}"
            )
        expected_codebooks = int(getattr(self._config, "cp_num_stages", 15)) + 1
        if int(capability.get("num_codebooks", expected_codebooks)) != expected_codebooks:
            raise RuntimeError(
                "native cursor/Talker codebook count mismatch: "
                f"cursor={capability.get('num_codebooks')} talker={expected_codebooks}"
            )
        input_names, output_names = self._fused_engine.get_io_names()
        declared_inputs = list(capability.get("input_names") or [])
        declared_outputs = list(capability.get("output_names") or [])
        if not declared_inputs or not declared_outputs:
            raise RuntimeError(
                "native cursor capability is enabled but its binding names are empty"
            )
        self._cursor_input_names = [n for n in declared_inputs if n in input_names]
        self._cursor_output_names = [n for n in declared_outputs if n in output_names]
        missing = sorted(set(declared_inputs) - set(self._cursor_input_names))
        missing_out = sorted(set(declared_outputs) - set(self._cursor_output_names))
        if missing or missing_out:
            raise RuntimeError(
                "native cursor manifest/engine ABI mismatch: "
                f"missing inputs={missing} outputs={missing_out}"
            )
        self._cursor_enabled = bool(self._cursor_input_names and self._cursor_output_names)
        self._cursor_state_handoff_enabled = (
            self._cursor_enabled
            and CURSOR_RECURRENT_INPUT_BINDINGS.issubset(self._cursor_input_names)
            and CURSOR_RECURRENT_OUTPUT_BINDINGS.issubset(self._cursor_output_names)
        )
        if self._cursor_enabled and not self._cursor_state_handoff_enabled:
            logger.warning(
                "Native cursor graph lacks the complete recurrent handoff ABI; "
                "successor cursor migration remains disabled"
            )
        self._cursor_max_labels = int(capability.get("max_labels", 512))
        self._cursor_d = int(capability.get("embedding_dim", 256))
        self._cursor_history = int(capability.get("history_width", 30))
        self._cursor_vocab_size = int(capability.get("vocab_size", 0))
        if self._cursor_vocab_size < 0:
            raise RuntimeError("native cursor label vocabulary size must be non-negative")
        logger.info(
            "Native cursor bindings: enabled=%s labels=%d d=%d history=%d",
            self._cursor_enabled,
            self._cursor_max_labels,
            self._cursor_d,
            self._cursor_history,
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

    def snapshot_c2w_arena(
        self,
        slot: SlotKVState,
        *,
        expected_allocation_epoch: int,
        max_tensor_bytes: Optional[int] = None,
    ) -> C2WArenaSnapshot:
        """Detach both rows after the caller has fenced all state writes."""
        self._validate_c2w_snapshot_target(slot, expected_allocation_epoch)
        if not slot.c2w_arena_backed:
            raise ValueError("C2W arena snapshot requires an arena-backed slot")
        if type(slot.c2w_write_in_a) is not bool:
            raise ValueError("C2W arena parity must be a bool")
        expected_read = self._arena_row_views(
            self._c2w_arena_a if not slot.c2w_write_in_a else self._c2w_arena_b,
            slot.slot_id,
        )
        expected_write = self._arena_row_views(
            self._c2w_arena_a if slot.c2w_write_in_a else self._c2w_arena_b,
            slot.slot_id,
        )
        actual = (
            slot.c2w_conv_states, slot.c2w_transconv_states,
            slot._c2w_conv_write, slot._c2w_transconv_write,
        )
        expected = (*expected_read, *expected_write)
        self._validate_c2w_snapshot_tensors(actual, expected, require_views=True)
        if max_tensor_bytes is not None:
            if (
                isinstance(max_tensor_bytes, bool)
                or not isinstance(max_tensor_bytes, Integral)
                or max_tensor_bytes < 0
            ):
                raise SpeechStateContractError("max_tensor_bytes must be nonnegative")
            tensor_bytes = sum(
                tensor.numel() * tensor.element_size()
                for group in actual
                for tensor in group
            )
            if tensor_bytes > int(max_tensor_bytes):
                raise SpeechStateContractError(
                    "C2W arena snapshot exceeds tensor byte budget"
                )
        return C2WArenaSnapshot(
            read_conv=[tensor.detach().clone() for tensor in actual[0]],
            read_transconv=[tensor.detach().clone() for tensor in actual[1]],
            write_conv=[tensor.detach().clone() for tensor in actual[2]],
            write_transconv=[tensor.detach().clone() for tensor in actual[3]],
            write_in_a=bool(slot.c2w_write_in_a),
            source_slot_id=slot.slot_id,
            source_allocation_epoch=slot.allocation_epoch,
        )

    @staticmethod
    def _speech_state_payload_bytes(payload: Any) -> int:
        """Count detached tensor storage in a capture payload."""
        if payload is None:
            return 0
        if isinstance(payload, torch.Tensor):
            return payload.numel() * payload.element_size()
        if isinstance(payload, (list, tuple)):
            return sum(Executor._speech_state_payload_bytes(item) for item in payload)
        tensor_bytes = getattr(payload, "tensor_bytes", None)
        if tensor_bytes is not None:
            return int(tensor_bytes)
        raise SpeechStateContractError(
            f"unsupported speech-state payload type: {type(payload).__name__}"
        )

    def capture_speech_state_bundle(
        self,
        slot: SlotKVState,
        segment_metadata: SegmentRuntimeMetadata,
        *,
        max_tensor_bytes: int,
        expected_slot_session_id: str | None = None,
    ) -> SpeechStateSnapshotBundle:
        """Assemble detached state after the caller has established quiescence.

        The caller must invoke this only after the decode future has completed,
        deferred KV/arena scatters have landed, and the slot's ping-pong parity
        is final. This method does not synchronize CUDA or infer that those
        conditions hold; it only validates ownership and assembles the four
        storage-owned payloads. It also does not advertise or attach a runtime
        capability.
        """
        if (
            isinstance(max_tensor_bytes, bool)
            or not isinstance(max_tensor_bytes, Integral)
            or max_tensor_bytes < 0
        ):
            raise SpeechStateContractError("max_tensor_bytes must be nonnegative")
        if not isinstance(segment_metadata, SegmentRuntimeMetadata):
            raise SpeechStateContractError("segment metadata type mismatch")
        if self._kv_pool is None:
            raise SpeechStateContractError("speech-state capture requires an initialized KV pool")
        if slot.is_free:
            raise SpeechStateContractError("cannot capture a free slot")
        if (
            self._kv_pool.get(slot.slot_id) is not slot
            or isinstance(slot.allocation_epoch, bool)
            or slot.allocation_epoch <= 0
        ):
            raise SpeechStateContractError("speech-state slot ownership mismatch")
        if (
            (expected_slot_session_id or segment_metadata.session_id) != slot.session_id
            or segment_metadata.segment_idx != slot.segment_idx
            or segment_metadata.retry_idx != slot.retry_idx
        ):
            raise SpeechStateContractError("speech-state segment ownership mismatch")

        epoch = int(slot.allocation_epoch)
        talker_pool = getattr(self._kv_pool, "_talker_kv_pool", None)
        uses_pooled_talker = talker_pool is not None
        if uses_pooled_talker and slot.talker_kv is not None and slot.past_len:
            raise SpeechStateContractError(
                "pooled Talker slot unexpectedly owns detached talker_kv"
            )

        uses_external_storage = uses_pooled_talker or slot.c2w_pooled or slot.c2w_arena_backed
        if not uses_external_storage:
            slot_payload = StandaloneSlotSnapshot(
                slot, max_tensor_bytes=int(max_tensor_bytes)
            )
            pooled_talker_payload = None
            pooled_c2w_payload = None
            c2w_arena_payload = None
        else:
            slot_payload = SlotOwnedSnapshot(
                slot,
                pooled_talker=uses_pooled_talker,
                pooled_c2w=slot.c2w_pooled,
                arena_backed=slot.c2w_arena_backed,
                max_tensor_bytes=int(max_tensor_bytes),
            )
            used_bytes = self._speech_state_payload_bytes(slot_payload)
            remaining_bytes = int(max_tensor_bytes) - used_bytes
            try:
                pooled_talker_payload = (
                    self._kv_pool.snapshot_pooled_talker_kv(
                        slot.slot_id,
                        expected_allocation_epoch=epoch,
                        max_tensor_bytes=remaining_bytes,
                    )
                    if uses_pooled_talker
                    else None
                )
            except ValueError as exc:
                raise SpeechStateContractError(str(exc)) from exc
            used_bytes += self._speech_state_payload_bytes(pooled_talker_payload)
            remaining_bytes = int(max_tensor_bytes) - used_bytes
            if uses_pooled_talker and slot.past_len and pooled_talker_payload is None:
                raise SpeechStateContractError("Talker state is missing for capture")

            if slot.c2w_pooled:
                try:
                    pooled_c2w_payload = self._kv_pool.snapshot_pooled_c2w_kv(
                        slot.slot_id,
                        expected_allocation_epoch=epoch,
                        max_tensor_bytes=remaining_bytes,
                    )
                except ValueError as exc:
                    raise SpeechStateContractError(str(exc)) from exc
                if slot.c2w_len and pooled_c2w_payload is None:
                    raise SpeechStateContractError("pooled C2W state is missing for capture")
            else:
                pooled_c2w_payload = None
                if slot.c2w_len and slot.c2w_kv is None:
                    raise SpeechStateContractError("C2W state is missing for capture")

            c2w_arena_payload = (
                self.snapshot_c2w_arena(
                    slot,
                    expected_allocation_epoch=epoch,
                    max_tensor_bytes=(
                        int(max_tensor_bytes)
                        - used_bytes
                        - self._speech_state_payload_bytes(pooled_c2w_payload)
                    ),
                )
                if slot.c2w_arena_backed
                else None
            )

        payloads = (
            slot_payload,
            pooled_talker_payload,
            pooled_c2w_payload,
            c2w_arena_payload,
        )
        total_bytes = sum(self._speech_state_payload_bytes(item) for item in payloads)
        if total_bytes > int(max_tensor_bytes):
            raise SpeechStateContractError(
                f"speech-state capture exceeds tensor byte budget: "
                f"{total_bytes} > {int(max_tensor_bytes)}"
            )
        return SpeechStateSnapshotBundle(
            source_session_id=str(segment_metadata.session_id),
            source_segment_idx=int(segment_metadata.segment_idx),
            source_slot_id=int(slot.slot_id),
            source_allocation_epoch=epoch,
            segment_metadata=segment_metadata,
            slot_payload=slot_payload,
            pooled_talker_payload=pooled_talker_payload,
            pooled_c2w_payload=pooled_c2w_payload,
            c2w_arena_payload=c2w_arena_payload,
            source_slot_session_id=slot.session_id,
        )

    def restore_speech_state_bundle(
        self,
        slot: SlotKVState,
        bundle: SpeechStateSnapshotBundle,
        *,
        expected_allocation_epoch: int,
    ) -> None:
        """Restore a same-segment bundle into an already allocated target.

        This baseline deliberately handles PAUSE_RESUME identity only. A
        successor segment needs a model-specific text/phase contract and is
        not silently treated as an ordinary slot restore.
        """
        if not isinstance(bundle, SpeechStateSnapshotBundle):
            raise SpeechStateContractError("speech-state bundle type mismatch")
        self._validate_restore_target(slot, expected_allocation_epoch)
        if (
            slot.session_id != bundle.source_slot_session_id
            or slot.segment_idx != bundle.source_segment_idx
        ):
            raise SpeechStateContractError(
                "same-segment restore requires matching slot identity"
            )

        slot_payload = bundle.slot_payload
        if isinstance(slot_payload, StandaloneSlotSnapshot):
            if any(
                payload is not None
                for payload in (
                    bundle.pooled_talker_payload,
                    bundle.pooled_c2w_payload,
                    bundle.c2w_arena_payload,
                )
            ):
                raise SpeechStateContractError(
                    "standalone bundle contains external storage payloads"
                )
            slot_payload.restore_into(slot)
            return
        if not isinstance(slot_payload, SlotOwnedSnapshot):
            raise SpeechStateContractError("unsupported speech-state slot payload")

        if bundle.pooled_c2w_payload is None and slot.c2w_pooled:
            raise SpeechStateContractError(
                "non-pooled C2W bundle cannot overwrite a pooled target"
            )
        if bundle.c2w_arena_payload is None and slot.c2w_arena_backed:
            raise SpeechStateContractError(
                "bundle without arena state cannot overwrite an arena target"
            )

        pool = self._kv_pool
        talker_pool = getattr(pool, "_talker_kv_pool", None)
        c2w_pool = getattr(pool, "_c2w_kv_pool", None)
        if talker_pool is not None:
            if not isinstance(bundle.pooled_talker_payload, torch.Tensor):
                raise SpeechStateContractError("pooled Talker payload is missing")
            pool._validate_pooled_payload(
                bundle.pooled_talker_payload, talker_pool, "Talker"
            )
        elif bundle.pooled_talker_payload is not None:
            raise SpeechStateContractError("unexpected pooled Talker payload")
        if bundle.pooled_c2w_payload is not None:
            if c2w_pool is None or not isinstance(bundle.pooled_c2w_payload, torch.Tensor):
                raise SpeechStateContractError("pooled C2W payload is incompatible")
            pool._validate_pooled_payload(
                bundle.pooled_c2w_payload, c2w_pool, "C2W"
            )

        arena = bundle.c2w_arena_payload
        if arena is not None:
            if not isinstance(arena, C2WArenaSnapshot):
                raise SpeechStateContractError("C2W arena payload type mismatch")
            self._validate_c2w_snapshot_target(slot, expected_allocation_epoch)
            if (
                arena.source_slot_id != bundle.source_slot_id
                or arena.source_allocation_epoch != bundle.source_allocation_epoch
            ):
                raise SpeechStateContractError("C2W arena source identity mismatch")
            if type(arena.write_in_a) is not bool:
                raise SpeechStateContractError("C2W arena parity is invalid")
            read_conv, read_trans = self._arena_row_views(
                self._c2w_arena_a if not arena.write_in_a else self._c2w_arena_b,
                slot.slot_id,
            )
            write_conv, write_trans = self._arena_row_views(
                self._c2w_arena_a if arena.write_in_a else self._c2w_arena_b,
                slot.slot_id,
            )
            self._validate_c2w_snapshot_tensors(
                (arena.read_conv, arena.read_transconv,
                 arena.write_conv, arena.write_transconv),
                (read_conv, read_trans, write_conv, write_trans),
            )

        # All representation checks above happen before the first target write.
        slot_payload.restore_into(slot)
        if talker_pool is not None:
            pool.restore_pooled_talker_kv(
                slot.slot_id,
                bundle.pooled_talker_payload,
                expected_allocation_epoch=expected_allocation_epoch,
            )
        if bundle.pooled_c2w_payload is not None:
            pool.restore_pooled_c2w_kv(
                slot.slot_id,
                bundle.pooled_c2w_payload,
                expected_allocation_epoch=expected_allocation_epoch,
            )
        if arena is not None:
            self.restore_c2w_arena(
                slot,
                arena,
                expected_allocation_epoch=expected_allocation_epoch,
                expected_source_slot_id=bundle.source_slot_id,
                expected_source_allocation_epoch=bundle.source_allocation_epoch,
            )

    def _validate_restore_target(
        self, slot: SlotKVState, expected_allocation_epoch: int
    ) -> None:
        if self._kv_pool is None:
            raise SpeechStateContractError("speech-state restore requires an initialized KV pool")
        if (
            slot.is_free
            or self._kv_pool.get(slot.slot_id) is not slot
            or type(expected_allocation_epoch) is not int
            or expected_allocation_epoch <= 0
            or slot.allocation_epoch != expected_allocation_epoch
        ):
            raise SpeechStateContractError("speech-state restore target owner mismatch")

    def synchronize_state_transfer(self) -> None:
        """Fence state copies/scatters queued by the engine thread."""
        if self._device.type == "cuda":
            # TRT runs on the executor stream; batch-end pool/arena copies
            # run on the engine thread's current stream. A state boundary
            # must settle both before capturing or releasing an allocation.
            current_stream = torch.cuda.current_stream(self._device)
            compute_stream = getattr(self, "_compute_stream", None)
            if compute_stream is not None:
                compute_stream.synchronize()
            if compute_stream is None or getattr(
                current_stream, "cuda_stream", id(current_stream)
            ) != getattr(compute_stream, "cuda_stream", id(compute_stream)):
                current_stream.synchronize()

    def restore_c2w_arena(
        self,
        slot: SlotKVState,
        snapshot: C2WArenaSnapshot,
        *,
        expected_allocation_epoch: int,
        expected_source_slot_id: int,
        expected_source_allocation_epoch: int,
    ) -> None:
        """Restore into a live target; source identity comes from its handle.

        Source and target epochs are independent: migration may follow source
        release and target allocation. The caller owns handle authentication,
        write completion and admission of the fully restored runtime state.
        """
        self._validate_c2w_snapshot_target(slot, expected_allocation_epoch)
        if not isinstance(snapshot, C2WArenaSnapshot):
            raise ValueError("C2W arena snapshot type mismatch")
        if (
            snapshot.source_slot_id != expected_source_slot_id
            or snapshot.source_allocation_epoch != expected_source_allocation_epoch
        ):
            raise ValueError("C2W arena snapshot source ownership mismatch")
        if type(snapshot.write_in_a) is not bool:
            raise ValueError("C2W arena parity must be a bool")
        read_conv, read_trans = self._arena_row_views(
            self._c2w_arena_a if not snapshot.write_in_a else self._c2w_arena_b,
            slot.slot_id,
        )
        write_conv, write_trans = self._arena_row_views(
            self._c2w_arena_a if snapshot.write_in_a else self._c2w_arena_b,
            slot.slot_id,
        )
        self._validate_c2w_snapshot_tensors(
            (snapshot.read_conv, snapshot.read_transconv,
             snapshot.write_conv, snapshot.write_transconv),
            (read_conv, read_trans, write_conv, write_trans),
        )
        for dst, src in zip(read_conv, snapshot.read_conv):
            dst.copy_(src)
        for dst, src in zip(read_trans, snapshot.read_transconv):
            dst.copy_(src)
        for dst, src in zip(write_conv, snapshot.write_conv):
            dst.copy_(src)
        for dst, src in zip(write_trans, snapshot.write_transconv):
            dst.copy_(src)
        slot.c2w_conv_states, slot.c2w_transconv_states = read_conv, read_trans
        slot._c2w_conv_write, slot._c2w_transconv_write = write_conv, write_trans
        slot.c2w_write_in_a = snapshot.write_in_a
        slot.c2w_arena_backed = True

    def _validate_c2w_snapshot_target(self, slot, epoch) -> None:
        if (
            isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch <= 0
            or isinstance(slot.slot_id, bool) or not isinstance(slot.slot_id, Integral)
            or not 0 <= slot.slot_id < self._kv_pool._max_slots
            or self._kv_pool.get(slot.slot_id) is not slot
            or slot.is_free or slot.allocation_epoch != epoch
        ):
            raise ValueError("C2W arena owner mismatch")
        shapes = [*self._c2w_conv_shapes, *self._c2w_transconv_shapes]
        for arena in (self._c2w_arena_a, self._c2w_arena_b):
            if arena is None or not shapes or len(arena) != len(shapes):
                raise ValueError("C2W arena storage unavailable or incomplete")
            for tensor, shape in zip(arena, shapes):
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tuple(tensor.shape) != (self._kv_pool._max_slots, *shape[1:])
                    or tensor.dtype != self._config.dtype
                    or tensor.device != self._device
                ):
                    raise ValueError("C2W arena storage ABI mismatch")

    @staticmethod
    def _validate_c2w_snapshot_tensors(actual, expected, *, require_views=False):
        for group, rows in zip(actual, expected):
            if not isinstance(group, (list, tuple)) or len(group) != len(rows):
                raise ValueError("C2W arena snapshot state count mismatch")
            for tensor, row in zip(group, rows):
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != row.shape or tensor.dtype != row.dtype
                    or tensor.device != row.device or tensor.layout != torch.strided
                ):
                    raise ValueError("C2W arena snapshot tensor ABI mismatch")
                if require_views and (
                    tensor.data_ptr() != row.data_ptr() or tensor.stride() != row.stride()
                ):
                    raise ValueError("C2W arena slot views are not bound to their owner row")

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

        # Prepare every tensor before mutating the target slot.  A malformed
        # state in the middle of the list must fall back as one transaction;
        # leaving half of the C2W history installed would mix a successor's
        # hard-boundary prefill with predecessor state.
        try:
            max_past = max(1, self._config.c2w_sliding_window - FUSED_CHUNK_T)
            prepared_kv = c2w_kv.to(
                device=self._device, dtype=self._config.dtype
            ).contiguous()
            if prepared_kv.ndim < 5:
                raise ValueError("Code2Wav KV state must have five dimensions")
            if prepared_kv.shape[3] > max_past:
                prepared_kv = prepared_kv[:, :, :, -max_past:, :].contiguous()
            prepared_conv = [
                t.to(device=self._device, dtype=self._config.dtype).contiguous()
                for t in conv_states
            ]
            prepared_transconv = [
                t.to(device=self._device, dtype=self._config.dtype).contiguous()
                for t in transconv_states
            ]
            prepared_frame_idx = max(0, int(frame_idx))
        except (
            AttributeError,
            IndexError,
            RuntimeError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            logger.warning(
                "Skipping Code2Wav warm state: tensor preparation failed",
                exc_info=True,
            )
            return False

        # A successor is commonly prefetched before its predecessor context is
        # available.  That prefill can already have promoted the slot to the
        # pooled C2W representation.  In that case assigning ``slot.c2w_kv``
        # would be invisible to decode, which gathers the pool row for pooled
        # slots.  Restore through the pool owner so the handoff replaces the
        # prefetched history atomically.
        if bool(getattr(slot, "c2w_pooled", False)):
            pool = getattr(self, "_kv_pool", None)
            restore_pooled = getattr(pool, "restore_pooled_c2w_kv", None)
            if not callable(restore_pooled):
                logger.warning(
                    "Skipping Code2Wav warm state: pooled C2W restore is unavailable"
                )
                return False
            try:
                restore_pooled(
                    slot.slot_id,
                    prepared_kv,
                    expected_allocation_epoch=slot.allocation_epoch,
                )
            except (
                AttributeError,
                IndexError,
                RuntimeError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                logger.warning(
                    "Skipping Code2Wav warm state: pooled C2W restore failed",
                    exc_info=True,
                )
                return False
            slot.c2w_kv = None
        else:
            slot.c2w_kv = prepared_kv
        slot.c2w_conv_states = prepared_conv
        slot.c2w_transconv_states = prepared_transconv
        slot.frame_idx = prepared_frame_idx
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
        # Cursor is a decoder-only recurrent observer.  The fused prefill
        # call still carries the fixed cursor ABI, but its cursor outputs are
        # intentionally scratch and must not advance or initialize the live
        # per-slot cursor state.  The first active decode owns frame zero.
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
        codec0 = raw.get("codec0")
        prefill_audio: Optional[bytes] = None
        prefill_eos = False
        if wav is not None:
            prefill_audio = wav.cpu().float().reshape(-1).numpy().tobytes()
        eos_token = (
            int(codec0[0].item())
            if codec0 is not None
            else int(full_codec[0, 0].item()) if full_codec is not None else None
        )
        if eos_token == self._codec_eos_id:
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
        codec0 = raw.get("codec0")
        prefill_audio: Optional[bytes] = None
        prefill_eos = False
        if wav is not None:
            prefill_audio = wav.cpu().float().reshape(-1).numpy().tobytes()
        eos_token = (
            int(codec0[0].item())
            if codec0 is not None
            else int(full_codec[0, 0].item()) if full_codec is not None else None
        )
        if eos_token == self._codec_eos_id:
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
            _cursor_output_names=self._cursor_output_names,
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
            _cursor_output_names=self._cursor_output_names,
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

        if slot.sampling_identity is not None:
            # Gateways use a private UUID as the engine registry key.  Keep
            # sampling tied to the public logical request ID so that identity
            # isolation does not silently change existing deterministic output.
            identity = f"{slot.sampling_identity}:{slot.segment_idx}"
        else:
            identity = (
                slot.session_id
                if slot.session_id is not None
                else f"slot:{slot.slot_id}"
            )
        if slot.retry_idx > 0:
            # Rerun re-roll: salt the derivation so attempt N+1 samples a fresh
            # trajectory (same seed would replay the hallucination bit-exactly).
            # Appended conditionally: retry-0 must keep the historical
            # (base_seed, session, segment) derivation bit-identical, or every
            # frozen halluprobe session id stops reproducing.
            seed = _stable_sampling_seed(
                self._random_seed,
                identity,
                slot.segment_idx,
                f"retry:{slot.retry_idx}",
            )
        else:
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

        # --- Optional native cursor state ---
        # The adapter may replace these per-slot tensors when a TN label plan
        # is available. Until then, inactive zero state keeps a cursor-enabled
        # plan ABI-safe without pretending that BPE ids are cursor labels.
        if self._cursor_enabled:
            def _cursor_batch(
                attr: str,
                tail_shape: tuple[int, ...],
                dtype: torch.dtype,
            ) -> torch.Tensor:
                rows = []
                for slot in slots:
                    value = getattr(slot, attr, None)
                    expected = (1,) + tail_shape
                    if value is None or tuple(value.shape) != expected:
                        value = torch.zeros(expected, device=self._device, dtype=dtype)
                    else:
                        value = value.to(device=self._device, dtype=dtype).contiguous()
                    rows.append(value)
                return torch.cat(rows, dim=0).contiguous()

            cursor_specs = {
                "cursor_label_ids": ("cursor_label_ids", (self._cursor_max_labels,), torch.int64),
                "cursor_label_count": ("cursor_label_count", (), torch.int64),
                "cursor_active": ("cursor_active", (), torch.int64),
                "cursor_mu_in": ("cursor_mu", (), cfg.dtype),
                "cursor_frames_since_advance_in": ("cursor_frames_since_advance", (), cfg.dtype),
                "cursor_delta_history_in": ("cursor_delta_history", (8,), cfg.dtype),
                "cursor_conv_history_in": ("cursor_conv_history", (self._cursor_history, self._cursor_d), cfg.dtype),
                "cursor_last_trunk_input_in": ("cursor_last_trunk_input", (self._cursor_d,), cfg.dtype),
                "cursor_seen_frames_in": ("cursor_seen_frames", (), torch.int64),
                "cursor_text_start_frame": ("cursor_text_start_frame", (), torch.int64),
                "cursor_override_valid": ("cursor_override_valid", (), torch.int64),
                "cursor_override_mu": ("cursor_override_mu", (), cfg.dtype),
            }
            for name in self._cursor_input_names:
                spec = cursor_specs.get(name)
                if spec is None:
                    raise RuntimeError(f"unknown native cursor input binding: {name}")
                d[name] = _cursor_batch(*spec)

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
        names.extend(self._cursor_output_names)
        return names

    def update_cursor_state(
        self,
        slot: SlotKVState,
        outputs: Dict[str, Optional[torch.Tensor]],
        row: int = 0,
    ) -> None:
        """Commit one fused cursor step's recurrent outputs to a slot.

        Text labels and reanchor overrides are owned by the CPU adapter; this
        method only forwards neural state.  Every value is cloned because
        eager output buffers and CUDA-graph staging are reused on the next
        execution.
        """
        if not self._cursor_enabled:
            return
        fields = {
            "cursor_mu": "cursor_mu",
            "cursor_frames_since_advance": "cursor_frames_since_advance",
            "cursor_delta_history": "cursor_delta_history",
            "cursor_conv_history": "cursor_conv_history",
            "cursor_last_trunk_input": "cursor_last_trunk_input",
            "cursor_seen_frames": "cursor_seen_frames",
        }
        for output_name, attr in fields.items():
            value = outputs.get(output_name)
            if value is None:
                continue
            setattr(slot, attr, value[row : row + 1].clone())
        if slot.cursor_override_valid is not None:
            slot.cursor_override_valid.zero_()

    def restore_cursor_state(self, slot: SlotKVState, state: Any) -> None:
        """Restore fused cursor recurrent state into an allocated slot.

        The successor's labels are installed separately from the primary TN
        plan; this method only copies neural state and never consumes text.
        """
        if not self._cursor_enabled or not bool(
            getattr(self, "_cursor_state_handoff_enabled", False)
        ):
            raise RuntimeError("native cursor state handoff is unavailable")
        if slot is None or slot.is_free:
            raise ValueError("cannot restore cursor state into a free slot")
        self._ensure_cursor_state(slot)
        restore = getattr(state, "restore_into", None)
        if not callable(restore):
            raise ValueError("invalid cursor continuation state")
        restore(slot)

    def validate_cursor_state(self, slot: SlotKVState, state: Any) -> None:
        """Validate a detached cursor payload without mutating the slot.

        EngineLoop uses this preflight before installing C2W warm state so a
        malformed cursor payload cannot leave a partially restored successor.
        """
        if not self._cursor_enabled or not bool(
            getattr(self, "_cursor_state_handoff_enabled", False)
        ):
            raise RuntimeError("native cursor state handoff is unavailable")
        if slot is None or slot.is_free:
            raise ValueError("cannot validate cursor state for a free slot")
        specs = {
            "cursor_mu": ((), self._config.dtype),
            "cursor_frames_since_advance": ((), self._config.dtype),
            "cursor_delta_history": ((8,), self._config.dtype),
            "cursor_conv_history": (
                (self._cursor_history, self._cursor_d),
                self._config.dtype,
            ),
            "cursor_last_trunk_input": ((self._cursor_d,), self._config.dtype),
            "cursor_seen_frames": ((), torch.int64),
        }
        for name, (tail_shape, dtype) in specs.items():
            value = getattr(state, name, None)
            expected_shape = (1, *tail_shape)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected_shape
                or value.dtype != dtype
                or value.device != self._device
                or not value.is_contiguous()
            ):
                raise ValueError(f"cursor continuation field {name} ABI mismatch")

    def set_cursor_text_plan(
        self,
        slot: SlotKVState,
        label_ids: torch.Tensor,
        *,
        label_count: Optional[int] = None,
        active: bool = True,
        text_start_frame: int = 0,
    ) -> None:
        """Install a CPU TN-produced padded label plan on one slot.

        ``label_ids`` must already be produced by the primary streaming TN
        layer. This API deliberately accepts labels, not raw text or Talker
        BPE ids, and pads to the manifest's fixed ``M_max`` capacity.
        """
        if not self._cursor_enabled:
            return

        # Validate the CPU-side plan before moving it to the execution device;
        # plan updates are infrequent, so this also keeps malformed labels from
        # reaching a device-side embedding/gather operation.
        ids = torch.as_tensor(label_ids)
        if ids.dim() == 2:
            if ids.shape[0] != 1:
                raise ValueError("cursor label plan must have one row per slot")
            ids = ids[0]
        if ids.dim() != 1:
            raise ValueError(f"cursor label plan must be [M], got {tuple(ids.shape)}")
        if ids.numel() and (
            ids.dtype == torch.bool
            or ids.is_floating_point()
            or ids.is_complex()
        ):
            raise ValueError("cursor label plan must contain integer labels")
        if ids.numel() and bool((ids < 0).any().item()):
            raise ValueError("cursor label plan must contain non-negative labels")
        vocab_size = int(getattr(self, "_cursor_vocab_size", 0) or 0)
        if vocab_size and ids.numel() and bool((ids >= vocab_size).any().item()):
            raise ValueError(
                f"cursor label plan contains an id outside vocabulary size {vocab_size}"
            )
        if ids.numel() > self._cursor_max_labels:
            raise ValueError(
                f"cursor label plan length={ids.numel()} exceeds capacity "
                f"{self._cursor_max_labels}"
            )
        ids = ids.to(device=self._device, dtype=torch.int64)
        count = ids.numel() if label_count is None else self._cursor_int(
            label_count, name="cursor label_count"
        )
        if count < 0 or count > int(ids.numel()):
            raise ValueError(
                f"cursor label_count={count} exceeds supplied plan length "
                f"{ids.numel()}"
            )
        if count > self._cursor_max_labels:
            raise ValueError(
                f"cursor label_count={count} exceeds plan capacity {self._cursor_max_labels}"
            )

        active_value = self._cursor_active_value(active)
        text_start_value = self._cursor_int(
            text_start_frame, name="cursor text_start_frame"
        )

        self._ensure_cursor_state(slot)
        slot.cursor_label_ids.zero_()
        if count:
            slot.cursor_label_ids[:, :count].copy_(ids[:count].reshape(1, count))
        slot.cursor_label_count.fill_(count)
        slot.cursor_active.fill_(active_value)
        slot.cursor_text_start_frame.fill_(text_start_value)

    def set_cursor_reanchor(self, slot: SlotKVState, mu: float) -> None:
        """Inject one CPU-computed TN tail-rewrite reanchor for the next step."""
        if not self._cursor_enabled:
            return
        mu_value = self._cursor_mu_value(mu)
        self._ensure_cursor_state(slot)
        slot.cursor_override_mu.fill_(mu_value)
        slot.cursor_override_valid.fill_(1)

    @staticmethod
    def _cursor_int(value: Any, *, name: str) -> int:
        """Validate an integer cursor control without coercive truncation."""
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer scalar, not bool")
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be an integer scalar")
            if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
                raise ValueError(f"{name} must be an integer scalar")
            value = value.detach().item()
        if not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer scalar")
        return int(value)

    @staticmethod
    def _cursor_active_value(value: Any) -> int:
        """Normalize the active control while rejecting ambiguous values."""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("cursor active must be a boolean scalar")
            if value.dtype == torch.bool:
                return int(value.detach().item())
            if value.is_floating_point() or value.is_complex():
                raise ValueError("cursor active must be a boolean scalar")
            value = value.detach().item()
        if isinstance(value, Integral) and int(value) in (0, 1):
            return int(value)
        raise ValueError("cursor active must be a boolean scalar")

    @staticmethod
    def _cursor_mu_value(value: Any) -> float:
        """Validate a finite scalar reanchor coordinate before slot mutation."""
        if isinstance(value, bool):
            raise ValueError("cursor reanchor mu must be a finite numeric scalar")
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("cursor reanchor mu must be a finite numeric scalar")
            if value.dtype == torch.bool or value.is_complex():
                raise ValueError("cursor reanchor mu must be a finite numeric scalar")
            value = value.detach().item()
        if not isinstance(value, Real):
            raise ValueError("cursor reanchor mu must be a finite numeric scalar")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("cursor reanchor mu must be a finite numeric scalar")
        return value

    def _ensure_cursor_state(self, slot: SlotKVState) -> None:
        """Lazily allocate correctly shaped state without resetting live values."""
        dtype = self._config.dtype

        def _ensure(attr: str, tail_shape: tuple[int, ...], tensor_dtype: torch.dtype) -> None:
            expected = (1,) + tail_shape
            value = getattr(slot, attr, None)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected
                or value.device != self._device
                or value.dtype != tensor_dtype
                or not value.is_contiguous()
            ):
                setattr(
                    slot,
                    attr,
                    torch.zeros(expected, device=self._device, dtype=tensor_dtype),
                )

        _ensure("cursor_mu", (), dtype)
        _ensure("cursor_frames_since_advance", (), dtype)
        _ensure("cursor_delta_history", (8,), dtype)
        _ensure("cursor_conv_history", (self._cursor_history, self._cursor_d), dtype)
        _ensure("cursor_last_trunk_input", (self._cursor_d,), dtype)
        _ensure("cursor_seen_frames", (), torch.int64)
        _ensure("cursor_label_ids", (self._cursor_max_labels,), torch.int64)
        _ensure("cursor_label_count", (), torch.int64)
        _ensure("cursor_active", (), torch.int64)
        _ensure("cursor_text_start_frame", (), torch.int64)
        _ensure("cursor_override_valid", (), torch.int64)
        _ensure("cursor_override_mu", (), dtype)

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
            self._kv_pool.release(
                dummy_slot.slot_id,
                expected_allocation_epoch=dummy_slot.allocation_epoch,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release GPU resources."""
        self._fused_engine = None
        self._kv_pool = None
        torch.cuda.empty_cache()
        logger.info("Executor shutdown")
