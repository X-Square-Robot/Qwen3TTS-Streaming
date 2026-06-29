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
import re
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
        logger.info("Loaded TRT engine: %s (%d I/O tensors)",
                     plan_path.name, self._engine.num_io_tensors)

    def _format_load_failure(self, plan_path: Path, reason: str) -> str:
        memory_hint = ""
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(self._device)
                memory_hint = (
                    f" GPU memory on {self._device}: "
                    f"free={free_bytes / (1024 ** 2):.0f} MiB, "
                    f"total={total_bytes / (1024 ** 2):.0f} MiB."
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
        self, name: str, profile_idx: int = 0,
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
            missing_inputs = sorted(name for name in valid_input_names if name not in inputs)
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
            if override is not None and override.shape == shape and override.dtype == dtype_torch:
                out_tensor = override
            elif use_cache:
                existing = self._output_buffers.get(name)
                if existing is not None and existing.shape == shape and existing.dtype == dtype_torch:
                    out_tensor = existing
                else:
                    out_tensor = torch.empty(
                        shape, dtype=dtype_torch, device=self._device,
                    )
                    self._output_buffers[name] = out_tensor
            else:
                out_tensor = torch.empty(
                    shape, dtype=dtype_torch, device=self._device,
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

        split_c2w_conv = []
        split_c2w_transconv = []
        conv_tensors = [raw.get(n) for n in self._c2w_conv_output_names]
        transconv_tensors = [raw.get(n) for n in self._c2w_transconv_output_names]
        for row_idx in range(batch_size):
            split_c2w_conv.append([
                t[row_idx:row_idx + 1] if t is not None else None
                for t in conv_tensors
            ])
            split_c2w_transconv.append([
                t[row_idx:row_idx + 1] if t is not None else None
                for t in transconv_tensors
            ])

        codec_eos_id = self._codec_eos_id
        eos_flags = []
        audio_chunks = []
        if full_codec is not None:
            eos_list = (full_codec[:, 0] == codec_eos_id).cpu().tolist()
        else:
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
            audio_chunks=audio_chunks,
            batch_talker_kv=raw.get("talker_new_kv"),
            batch_c2w_kv=raw.get("c2w_new_kv"),
            original_past_lens=self._original_past_lens,
            padded_past_len=self._padded_past_len,
            split_c2w_conv=split_c2w_conv,
            split_c2w_transconv=split_c2w_transconv,
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
    batch_talker_kv: Optional[torch.Tensor] = None
    batch_c2w_kv: Optional[torch.Tensor] = None
    original_past_lens: List[int] = field(default_factory=list)
    padded_past_len: int = 0
    split_c2w_conv: List[List[Optional[torch.Tensor]]] = field(default_factory=list)
    split_c2w_transconv: List[List[Optional[torch.Tensor]]] = field(default_factory=list)
    codec_sum: Optional[torch.Tensor] = None
    updated_tc: Optional[torch.Tensor] = None
    used_pingpong: bool = False


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

        logger.info(
            "Executor created (device=%s, max_batch=%d, max_seq=%d)",
            self._device, max_batch_size, max_seq_len,
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
                str(fused_plan), self._device,
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
        logger.info(
            "Executor loaded (engine=%s, effective_max_seq=%d)",
            "TRT" if self._fused_engine else "stub",
            self._max_seq_len,
        )

    def set_embedding_weights(self, weights) -> None:
        self._embedding_weights = weights
        if hasattr(weights, 'codec_eos_id'):
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
            manifest_dtype, actual_dtype, representative_name,
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

        def _ordered(names: list[str], prefix: str, layout_names: list[str]) -> list[str]:
            from_layout = [n for n in layout_names if n.startswith(prefix) and n in names]
            if from_layout:
                return from_layout
            return sorted((n for n in names if n.startswith(prefix)), key=_natural_key)

        self._c2w_conv_input_names = _ordered(
            input_names, "c2w_conv_state_", layout_inputs,
        )
        self._c2w_transconv_input_names = _ordered(
            input_names, "c2w_transconv_overlap_", layout_inputs,
        )
        self._c2w_conv_output_names = _ordered(
            output_names, "c2w_new_conv_state_", layout_outputs,
        )
        self._c2w_transconv_output_names = _ordered(
            output_names, "c2w_new_transconv_overlap_", layout_outputs,
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
                len(conv_states), len(self._c2w_conv_input_names),
            )
            return False
        if len(transconv_states) != len(self._c2w_transconv_input_names):
            logger.warning(
                "Skipping Code2Wav warm state: transconv state count %d != expected %d",
                len(transconv_states), len(self._c2w_transconv_input_names),
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
                inputs, out_names, stream,
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
            if self._kv_pool._preallocate:
                self._kv_pool.scatter_c2w_kv([slot.slot_id], c2w_kv)
            slot.c2w_kv = c2w_kv
        slot.c2w_conv_states = [raw[n].clone() for n in self._c2w_conv_output_names]
        slot.c2w_transconv_states = [raw[n].clone() for n in self._c2w_transconv_output_names]
        slot.init_pingpong_buffers()

        slot.frame_idx = int(slot.frame_idx) + FUSED_CHUNK_T
        slot.token_counts = raw.get(
            "updated_token_counts",
            torch.zeros(1, self._config.codec_vocab_size,
                        device=self._device, dtype=torch.int64),
        )
        codec_sum = raw.get("codec_sum")
        if codec_sum is not None:
            slot.next_embed = codec_sum
            slot.last_codec_sum = None

        wav = raw.get("wav")
        full_codec = raw.get("full_codec")
        prefill_audio: Optional[bytes] = None
        prefill_eos = False
        if wav is not None:
            prefill_audio = wav.cpu().float().reshape(-1).numpy().tobytes()
        if full_codec is not None and int(full_codec[0, 0].item()) == self._codec_eos_id:
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
                inputs, out_names, stream,
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
                inputs, out_names, stream,
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
            if self._kv_pool._preallocate:
                self._kv_pool.scatter_c2w_kv([slot.slot_id], c2w_kv)
            slot.c2w_kv = c2w_kv
        slot.c2w_conv_states = [raw[n].clone() for n in self._c2w_conv_output_names]
        slot.c2w_transconv_states = [raw[n].clone() for n in self._c2w_transconv_output_names]
        slot.init_pingpong_buffers()

        slot.frame_idx = int(slot.frame_idx) + FUSED_CHUNK_T
        slot.token_counts = raw.get(
            "updated_token_counts",
            torch.zeros(
                1,
                self._config.codec_vocab_size,
                device=self._device,
                dtype=torch.int64,
            ),
        )
        codec_sum = raw.get("codec_sum")
        if codec_sum is not None:
            slot.next_embed = codec_sum
            slot.last_codec_sum = None

        wav = raw.get("wav")
        full_codec = raw.get("full_codec")
        prefill_audio: Optional[bytes] = None
        prefill_eos = False
        if wav is not None:
            prefill_audio = wav.cpu().float().reshape(-1).numpy().tobytes()
        if full_codec is not None and int(full_codec[0, 0].item()) == self._codec_eos_id:
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
        batch_size = len(slots)

        input_embeds = torch.cat(
            [s.next_embed.to(self._config.dtype) for s in slots], dim=0,
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

        if self._kv_pool is not None and self._kv_pool._preallocate:
            batched_talker_kv = self._kv_pool.gather_talker_kv(
                slot_ids, max_past_len,
            )
            past_seq_lens = torch.tensor(
                original_past_lens, device=self._device, dtype=torch.long,
            )
        else:
            session_kv = [s.talker_kv for s in slots]
            batched_talker_kv, past_seq_lens = pad_packed_kv(
                session_kv, device=self._device, dtype=self._config.dtype,
            )

        padded_past_len = int(batched_talker_kv.shape[3]) if batched_talker_kv is not None else 0

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
                inputs, out_names, self._compute_stream,
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

    def _build_pingpong_overrides(
        self, slots: List[SlotKVState],
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

        identity = slot.session_id if slot.session_id is not None else f"slot:{slot.slot_id}"
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

        for slot in slots:
            gen = self._slot_sampling_generator(slot)
            gumbel_u = torch.rand(
                1, cfg.logits_topk,
                device=self._device, dtype=torch.float32,
                generator=gen,
            ).clamp(1e-8, 1.0)
            cp_gumbel_u = torch.rand(
                1, cfg.cp_num_stages, cfg.logits_topk,
                device=self._device, dtype=torch.float32,
                generator=gen,
            ).clamp(1e-8, 1.0)
            gumbel_rows.append(-torch.log(-torch.log(gumbel_u)))
            cp_gumbel_rows.append(-torch.log(-torch.log(cp_gumbel_u)))

        return torch.cat(gumbel_rows, dim=0), torch.cat(cp_gumbel_rows, dim=0)

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
    ) -> Dict[str, torch.Tensor]:
        """Build the full input dict for the fused TRT engine.

        Uses packed KV format: single tensor per cache type.
        """
        cfg = self._config
        batch = int(input_embeds.shape[0])
        seq = int(input_embeds.shape[1])

        if use_dummy_kv:
            if past_seq_lens is None:
                past_seq_lens = uniform_past_seq_lens(batch, 0, self._device)
            attn_bias = padded_attention_bias(
                past_seq_lens, seq, _FUSED_DUMMY_PAST_LEN,
                self._device, cfg.dtype,
            )
        else:
            padded_past_len = int(batched_talker_kv.shape[3]) if batched_talker_kv is not None else 0
            attn_bias = padded_attention_bias(
                past_seq_lens, seq, padded_past_len,
                self._device, cfg.dtype,
            )

        position_ids = torch.stack([
            torch.arange(s.past_len, s.past_len + seq,
                         device=self._device, dtype=torch.int64)
            .unsqueeze(0).expand(3, seq)
            for s in slots
        ], dim=0).unsqueeze(-1)

        cache_position = torch.stack([
            torch.full((FUSED_CHUNK_T,), s.frame_idx, device=self._device, dtype=torch.float32)
            for s in slots
        ], dim=0)

        tc = torch.cat([
            s.token_counts if s.token_counts is not None
            else torch.zeros(1, cfg.codec_vocab_size,
                             device=self._device, dtype=torch.int64)
            for s in slots
        ], dim=0)

        if sampling_mode == "disabled":
            gumbel = torch.zeros(
                batch, cfg.logits_topk,
                device=self._device, dtype=torch.float32,
            )
            cp_gumbel = torch.zeros(
                batch, cfg.cp_num_stages, cfg.logits_topk,
                device=self._device, dtype=torch.float32,
            )
            temperature = torch.ones(
                batch, 1, device=self._device, dtype=torch.float32,
            )
            penalty = torch.ones(
                batch, 1, device=self._device, dtype=torch.float32,
            )
        elif self._do_sample:
            gumbel, cp_gumbel = self._build_sampling_noise(slots)
            temperature = torch.full(
                (batch, 1), self._temperature,
                device=self._device, dtype=torch.float32,
            )
            penalty = torch.full(
                (batch, 1), self._repetition_penalty,
                device=self._device, dtype=torch.float32,
            )
        else:
            gumbel = torch.zeros(
                batch, cfg.logits_topk,
                device=self._device, dtype=torch.float32,
            )
            cp_gumbel = torch.zeros(
                batch, cfg.cp_num_stages, cfg.logits_topk,
                device=self._device, dtype=torch.float32,
            )
            temperature = torch.ones(
                batch, 1, device=self._device, dtype=torch.float32,
            )
            penalty = torch.full(
                (batch, 1), self._repetition_penalty,
                device=self._device, dtype=torch.float32,
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
                batch, cfg.num_layers * 2, cfg.kv_heads,
                _FUSED_DUMMY_PAST_LEN, cfg.head_dim,
                device=self._device, dtype=cfg.dtype,
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
            if s.c2w_kv is not None:
                per_slot_c2w_lens.append(
                    min(int(s.c2w_kv.shape[3]), c2w_max_past)
                )
            else:
                per_slot_c2w_lens.append(0)
        c2w_past_len = max(per_slot_c2w_lens) if per_slot_c2w_lens else 0
        if c2w_past_len < 1:
            c2w_past_len = 1

        c2w_key_total = c2w_past_len + FUSED_CHUNK_T

        c2w_attn = torch.zeros(
            batch, 1, FUSED_CHUNK_T, c2w_key_total,
            device=self._device, dtype=cfg.dtype,
        )
        for bi, sl in enumerate(per_slot_c2w_lens):
            pad_cols = c2w_past_len - sl
            if pad_cols > 0:
                c2w_attn[bi, :, :, :pad_cols] = float("-inf")
        d["c2w_attention_bias"] = c2w_attn.contiguous()

        c2w_d1 = cfg.n_c2w_layers * 2
        c2w_d2 = cfg.c2w_kv_heads
        c2w_head = cfg.c2w_head_dim
        if batch == 1:
            s = slots[0]
            if s.c2w_kv is not None:
                kv = s.c2w_kv
                if kv.shape[3] > c2w_max_past:
                    kv = kv[:, :, :, -c2w_max_past:, :]
            else:
                kv = torch.zeros(
                    1, c2w_d1, c2w_d2, c2w_past_len, c2w_head,
                    device=self._device, dtype=cfg.dtype,
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
                        1, c2w_d1, c2w_d2, 0, c2w_head,
                        device=self._device, dtype=cfg.dtype,
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
                        [s.c2w_conv_states[idx] for s in slots], dim=0,
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
                        [s.c2w_transconv_states[idx] for s in slots], dim=0,
                    ).contiguous()
            else:
                shape = list(self._c2w_transconv_shapes[idx])
                shape[0] = batch
                d[name] = torch.zeros(shape, device=self._device, dtype=cfg.dtype)

        return d

    def _build_output_names(self) -> List[str]:
        names = [
            "wav", "codec_sum", "full_codec", "hidden", "logits",
            "updated_token_counts",
            "talker_new_kv", "c2w_new_kv",
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
            "slot_trailing_len": [len(s.trailing) for s in slots],
            "slot_has_next_embed": [s.next_embed is not None for s in slots],
            "slot_has_last_codec_sum": [s.last_codec_sum is not None for s in slots],
            "slot_c2w_len_before": [
                int(s.c2w_kv.shape[3]) if s.c2w_kv is not None else 0
                for s in slots
            ],
            "original_talker_past_lens": [int(v) for v in original_past_lens],
            "padded_talker_past_len": int(padded_talker_past_len),
            "c2w_conv_input_names": list(self._c2w_conv_input_names),
            "c2w_conv_output_names": list(self._c2w_conv_output_names),
            "c2w_transconv_input_names": list(self._c2w_transconv_input_names),
            "c2w_transconv_output_names": list(self._c2w_transconv_output_names),
            "output_override_names": sorted(output_overrides.keys()) if output_overrides else [],
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
                1, 4, cfg.hidden_size,
                device=self._device, dtype=cfg.dtype,
            )
            _ = self.prefill(dummy_slot, dummy_embeds)

            dummy_slot.next_embed = torch.randn(
                1, 1, cfg.hidden_size, device=self._device, dtype=torch.float32,
            )

            for i in range(n_rounds):
                future = self.launch_decode_step([dummy_slot])
                output = future.wait()
                if self._kv_pool._preallocate and output.batch_talker_kv is not None:
                    self._kv_pool.scatter_talker_kv_delta(
                        [dummy_slot.slot_id], output.batch_talker_kv,
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
