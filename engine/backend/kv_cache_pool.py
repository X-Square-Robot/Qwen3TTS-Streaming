"""KV cache pool: pre-allocated GPU memory for concurrent sessions.

Instead of dynamically allocating KV tensors per session (which causes
fragmentation and OOM under 64-session load), we pre-allocate a fixed
pool and assign slots.

Memory layout per slot (packed tensor format):
    talker_kv:  [1, num_layers*2, kv_heads, cur_len, head_dim]  (single packed tensor)
    c2w_kv:     [1, n_c2w_layers*2, c2w_kv_heads, cur_len, c2w_head_dim]
    c2w_conv:   17 tensors with heterogeneous shapes (static except batch)
    c2w_transconv: 4 tensors with heterogeneous shapes (static except batch)

Total Talker KV per slot (example, 1.7B model, max_seq=512):
    28 layers × 2 × 8 heads × 128 dim × 512 max_seq × 2 bytes (bf16)
    = 28 × 2 × 8 × 128 × 512 × 2 = ~56 MB/slot
    32 slots = ~1.8 GB (fits on 24GB GPU with model weights)

Slot eviction:
    When the pool is full and a new session needs a slot, the pool can
    evict the least-recently-active slot.  The evicted session receives an
    error result and is cleaned up by the engine loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture parameters for KV cache sizing."""

    # Talker (1.7b defaults; 0.6b: hidden=1536, head_dim=64, vocab=2176)
    num_layers: int = 28
    kv_heads: int = 8
    head_dim: int = 128
    max_seq_len: int = 512
    hidden_size: int = 2048
    dtype: torch.dtype = torch.bfloat16
    codec_vocab_size: int = 3072
    logits_topk: int = 50
    cp_num_stages: int = 15

    # Code2Wav
    n_c2w_layers: int = 8
    c2w_kv_heads: int = 16
    c2w_head_dim: int = 64
    c2w_sliding_window: int = 72
    n_c2w_conv_states: int = 17
    n_c2w_transconv_states: int = 4

    @property
    def num_c2w_states(self) -> int:
        return (
            2 * self.n_c2w_layers + self.n_c2w_conv_states + self.n_c2w_transconv_states
        )


@dataclass
class SlotKVState:
    """GPU-side state for one session slot.

    Uses packed KV tensors: single [1, L*2, H, S, D] per cache type
    instead of lists of per-layer tensors.  This reduces TRT I/O binding
    count from ~201 to ~61.

    C2W conv/transconv states use **ping-pong double buffering** to avoid
    per-step clone() overhead.  Two buffer sets are pre-allocated once at
    prefill time.  After each decode step the read/write sets are swapped
    via pointer swap (zero-copy for batch=1, copy_-only for batch>1).
    """

    slot_id: int
    session_id: Optional[str] = None
    segment_idx: int = -1
    is_free: bool = True
    prefill_source: str = ""

    # Talker KV: [1, num_layers*2, kv_heads, cur_len, head_dim]
    talker_kv: Optional[torch.Tensor] = None
    past_len: int = 0

    # Code2Wav KV: [1, n_c2w*2, c2w_kv_heads, cur_len, c2w_head_dim].
    # In pooled mode (c2w_pooled=True) this stays None: the KV lives
    # right-aligned in the pool's _c2w_kv_pool row and c2w_len tracks the
    # valid column count.
    c2w_kv: Optional[torch.Tensor] = None
    c2w_len: int = 0
    c2w_pooled: bool = False
    # Conv/transconv states: heterogeneous shapes, kept as lists
    c2w_conv_states: Optional[list[torch.Tensor]] = None
    c2w_transconv_states: Optional[list[torch.Tensor]] = None
    frame_idx: int = 0

    # Ping-pong write buffers (swapped with read buffers each step)
    _c2w_conv_write: Optional[list[torch.Tensor]] = None
    _c2w_transconv_write: Optional[list[torch.Tensor]] = None
    # True when conv/transconv buffers are row views of the executor's
    # slot-indexed state arenas (enables batched post-step scatter).
    c2w_arena_backed: bool = False
    # Arena parity: True when the write side currently points at arena A.
    # Toggled by flip_c2w_buffers; meaningless when not arena-backed.
    c2w_write_in_a: bool = False

    # Decode tracking
    next_embed: Optional[torch.Tensor] = None
    last_codec_sum: Optional[torch.Tensor] = None
    token_counts: Optional[torch.Tensor] = None

    # Trailing text embeddings for streaming decode
    trailing: list = field(default_factory=list)
    text_idx: int = 0

    # Appended text token IDs for streaming APPEND_TOKENS
    token_queue: list = field(default_factory=list)

    # Pad phase tracking (aligned with old engine's Phase B controls)
    pad_start_frame: int = -1
    pad_consecutive_silence: int = 0

    # Per-slot sampling RNG.  Kept with the logical segment so batched
    # scheduling cannot change a lane's random sequence.
    sampling_seed: Optional[int] = None
    sampling_generator: Optional[torch.Generator] = None

    # Activity tracking for eviction
    last_active_time: float = field(default_factory=time.monotonic)

    @property
    def has_c2w_states(self) -> bool:
        return self.c2w_kv is not None

    @property
    def pingpong_ready(self) -> bool:
        """True when double buffers are allocated and ping-pong is usable."""
        return self._c2w_conv_write is not None

    def init_pingpong_buffers(
        self,
        conv_write: Optional[list[torch.Tensor]] = None,
        transconv_write: Optional[list[torch.Tensor]] = None,
    ) -> None:
        """Allocate the write-side buffers matching current read-side shapes.

        Called once after prefill populates the initial c2w states.
        Batch admission passes pre-allocated buffers (contiguous row views
        of one batch-level allocation) to avoid per-slot clone kernels.
        """
        if conv_write is not None:
            self._c2w_conv_write = conv_write
        elif self.c2w_conv_states:
            self._c2w_conv_write = [t.clone() for t in self.c2w_conv_states]
        if transconv_write is not None:
            self._c2w_transconv_write = transconv_write
        elif self.c2w_transconv_states:
            self._c2w_transconv_write = [t.clone() for t in self.c2w_transconv_states]

    def flip_c2w_buffers(self) -> None:
        """Swap read/write buffer pointers (zero-copy pointer swap)."""
        self.c2w_conv_states, self._c2w_conv_write = (
            self._c2w_conv_write,
            self.c2w_conv_states,
        )
        self.c2w_transconv_states, self._c2w_transconv_write = (
            self._c2w_transconv_write,
            self.c2w_transconv_states,
        )
        self.c2w_write_in_a = not self.c2w_write_in_a

    def copy_c2w_and_flip(
        self,
        conv_sources: list[Optional[torch.Tensor]],
        transconv_sources: list[Optional[torch.Tensor]],
    ) -> None:
        """Copy batched output slices into write buffers, then flip.

        Used for batch>1 where TRT wrote to its own output buffer.
        """
        if self._c2w_conv_write is not None:
            for j, src in enumerate(conv_sources):
                if src is not None:
                    self._c2w_conv_write[j].copy_(src)
        if self._c2w_transconv_write is not None:
            for j, src in enumerate(transconv_sources):
                if src is not None:
                    self._c2w_transconv_write[j].copy_(src)
        self.flip_c2w_buffers()

    def touch(self) -> None:
        """Update activity timestamp (call on each decode step)."""
        self.last_active_time = time.monotonic()

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_active_time


class KVCachePool:
    """Manages a fixed pool of KV cache slots on GPU.

    Slots are lazily initialized: KV tensors are only allocated when
    a session first does prefill.  On release, tensors are zeroed
    (not freed) to avoid reallocation.

    Pre-allocated pool mode (when enabled):
        A single contiguous tensor per cache type is pre-allocated for all
        slots.  Slots index into this tensor by slot_id, eliminating
        per-step pad+cat overhead during batch assembly.
    """

    def __init__(
        self,
        max_slots: int,
        config: ModelConfig,
        device: torch.device,
        *,
        preallocate: bool = True,
    ):
        self._max_slots = max_slots
        self._config = config
        self._device = device
        self._preallocate = preallocate

        self._slots: list[SlotKVState] = [
            SlotKVState(slot_id=i) for i in range(max_slots)
        ]
        self._free_slots: list[int] = list(range(max_slots))

        self._talker_kv_pool: Optional[torch.Tensor] = None
        self._c2w_kv_pool: Optional[torch.Tensor] = None

        if preallocate:
            self._init_pool_tensors()

        logger.info(
            "KV pool initialized: %d slots, ~%.1f MB/slot (talker KV), "
            "preallocated=%s, device=%s",
            max_slots,
            self._estimate_slot_mb(),
            preallocate,
            device,
        )

    def _init_pool_tensors(self) -> None:
        c = self._config
        self._talker_kv_pool = torch.zeros(
            self._max_slots,
            c.num_layers * 2,
            c.kv_heads,
            c.max_seq_len,
            c.head_dim,
            device=self._device,
            dtype=c.dtype,
        )
        self._c2w_kv_pool = torch.zeros(
            self._max_slots,
            c.n_c2w_layers * 2,
            c.c2w_kv_heads,
            c.c2w_sliding_window,
            c.c2w_head_dim,
            device=self._device,
            dtype=c.dtype,
        )

    def _estimate_slot_mb(self) -> float:
        c = self._config
        bytes_per_elem = 2 if c.dtype == torch.bfloat16 else 4
        kv_bytes = (
            c.num_layers * 2 * c.kv_heads * c.head_dim * c.max_seq_len * bytes_per_elem
        )
        return kv_bytes / (1024 * 1024)

    @property
    def max_seq_len(self) -> int:
        return self._config.max_seq_len

    @property
    def free_count(self) -> int:
        return len(self._free_slots)

    @property
    def used_count(self) -> int:
        return self._max_slots - len(self._free_slots)

    def allocate(self, session_id: str) -> Optional[SlotKVState]:
        """Allocate a slot for a new session.  Returns None if pool is full."""
        if not self._free_slots:
            return None
        slot_id = self._free_slots.pop()
        slot = self._slots[slot_id]
        slot.session_id = session_id
        slot.segment_idx = -1
        slot.is_free = False
        slot.prefill_source = ""
        slot.past_len = 0
        slot.frame_idx = 0
        slot.text_idx = 0
        slot.trailing = []
        slot.token_queue = []
        slot.pad_start_frame = -1
        slot.pad_consecutive_silence = 0
        slot.sampling_seed = None
        slot.sampling_generator = None
        slot.next_embed = None
        slot.last_codec_sum = None
        slot.token_counts = None
        slot.talker_kv = None
        slot.c2w_kv = None
        slot.c2w_conv_states = None
        slot.c2w_transconv_states = None
        slot._c2w_conv_write = None
        slot._c2w_transconv_write = None
        slot.c2w_arena_backed = False
        slot.c2w_write_in_a = False
        slot.c2w_len = 0
        slot.c2w_pooled = False
        slot.last_active_time = time.monotonic()
        if self._preallocate and self._talker_kv_pool is not None:
            self._talker_kv_pool[slot_id].zero_()
            self._c2w_kv_pool[slot_id].zero_()
        logger.debug("Allocated slot %d for session %s", slot_id, session_id)
        return slot

    def release(self, slot_id: int) -> None:
        """Return a slot to the pool."""
        slot = self._slots[slot_id]
        if slot.is_free:
            logger.debug("Ignoring duplicate release for free slot %d", slot_id)
            return
        slot.session_id = None
        slot.segment_idx = -1
        slot.is_free = True
        slot.prefill_source = ""
        slot.past_len = 0
        slot.frame_idx = 0
        slot.text_idx = 0
        slot.trailing = []
        slot.token_queue = []
        slot.pad_start_frame = -1
        slot.pad_consecutive_silence = 0
        slot.sampling_seed = None
        slot.sampling_generator = None
        slot.next_embed = None
        slot.last_codec_sum = None
        slot.token_counts = None
        slot.talker_kv = None
        slot.c2w_kv = None
        slot.c2w_conv_states = None
        slot.c2w_transconv_states = None
        slot._c2w_conv_write = None
        slot._c2w_transconv_write = None
        slot.c2w_arena_backed = False
        slot.c2w_write_in_a = False
        slot.c2w_len = 0
        slot.c2w_pooled = False
        self._free_slots.append(slot_id)
        logger.debug("Released slot %d (free: %d)", slot_id, len(self._free_slots))

    def get(self, slot_id: int) -> SlotKVState:
        return self._slots[slot_id]

    def get_by_session(self, session_id: str) -> Optional[SlotKVState]:
        for slot in self._slots:
            if not slot.is_free and slot.session_id == session_id:
                return slot
        return None

    def active_slots(self) -> list[SlotKVState]:
        """Return all slots with active sessions."""
        return [s for s in self._slots if not s.is_free]

    def init_kv_tensors(self, slot: SlotKVState) -> None:
        """Lazily initialize KV cache tensors for a slot.

        Called once at first prefill. Creates empty tensors that will be
        populated by the TRT engine's present_kv outputs.
        """
        c = self._config
        slot.token_counts = torch.zeros(
            1,
            c.codec_vocab_size,
            device=self._device,
            dtype=torch.int64,
        )
        logger.debug("Initialized KV tensors for slot %d", slot.slot_id)

    def reset_for_new_segment(self, slot: SlotKVState) -> None:
        """Reset decode state but preserve KV cache (for segment continuation)."""
        slot.text_idx = 0
        slot.trailing = []
        slot.next_embed = None
        slot.last_codec_sum = None
        slot.token_queue = []
        slot.pad_start_frame = -1
        slot.pad_consecutive_silence = 0
        slot.sampling_seed = None
        slot.sampling_generator = None

    def scatter_prefill_kv(
        self,
        slot_id: int,
        kv: torch.Tensor,
        seq_len: int,
    ) -> None:
        """Write prefill KV output directly to the pool.

        Args:
            slot_id: target slot
            kv: [1, L*2, H, seq_len, D] from TRT prefill output
            seq_len: number of tokens produced by prefill
        """
        if self._talker_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        self._talker_kv_pool[slot_id, :, :, :seq_len, :] = kv[0, :, :, :seq_len, :]

    def restore_prefix_batch(
        self,
        slot_ids: list[int],
        talker_kv: torch.Tensor,
        prefix_len: int,
    ) -> None:
        """Broadcast one cached prefix KV into many slots with a single write.

        Args:
            slot_ids: target slots (all sharing the same prefix cache entry)
            talker_kv: [1, L*2, H, >=prefix_len, D] cached prefix KV
            prefix_len: number of prefix tokens to copy
        """
        if self._talker_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        ids = torch.tensor(slot_ids, device=self._device, dtype=torch.long)
        self._talker_kv_pool[ids, :, :, :prefix_len, :] = talker_kv[
            0, :, :, :prefix_len, :
        ]

    def write_c2w_right_aligned(self, slot_id: int, kv: torch.Tensor) -> int:
        """Store a prefill's C2W KV right-aligned in the slot's pool row.

        Right alignment means "last N columns hold the newest N frames", so
        the decode-time gather (last max_len columns across the batch) is
        bitwise identical to the historical per-slot left-pad + cat layout.
        Returns the stored length (clamped to the attention window).
        """
        if self._c2w_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        pool = self._c2w_kv_pool
        width = int(pool.shape[3])
        # Same clamp as the legacy input-build trim (window - chunk).
        cap = width - 1
        s_len = min(int(kv.shape[3]), cap)
        row = pool[slot_id]
        row.zero_()
        if s_len > 0:
            row[:, :, width - s_len :, :] = kv[0, :, :, -s_len:, :]
        return s_len

    def append_c2w_frames(
        self,
        slot_ids: list[int],
        new_frames: torch.Tensor,
    ) -> None:
        """Sliding-window append for many slots in 3 batched kernels.

        Shifts each row one column left and writes the step's new frame at
        the last column.  Rows stay right-aligned; columns left of the valid
        window only ever receive zeros from the shift.

        Args:
            slot_ids: target slot rows.
            new_frames: [len(slot_ids), L*2, H, 1, D] new KV frames, ordered
                like slot_ids.
        """
        if self._c2w_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        pool = self._c2w_kv_pool
        ids = torch.tensor(slot_ids, device=self._device, dtype=torch.long)
        # pool[ids] gathers a copy, so the shifted write does not alias.
        shifted = pool[ids][:, :, :, 1:, :]
        pool[ids, :, :, :-1, :] = shifted
        pool[ids, :, :, -1:, :] = new_frames

    def scatter_prefill_c2w_kv(
        self,
        slot_id: int,
        kv: torch.Tensor,
    ) -> None:
        """Write prefill C2W KV output directly to the pool."""
        if self._c2w_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        s_len = kv.shape[3]
        self._c2w_kv_pool[slot_id, :, :, :s_len, :] = kv[0]

    # ------------------------------------------------------------------
    # Pre-allocated pool batch helpers
    # ------------------------------------------------------------------

    def gather_talker_kv(
        self,
        slot_ids: list[int],
        max_past_len: int,
    ) -> torch.Tensor:
        """Gather talker KV for a batch from the pre-allocated pool.

        Returns [B, L*2, H, max_past_len, D] without any pad+cat.
        """
        if self._talker_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        ids = torch.tensor(slot_ids, device=self._device, dtype=torch.long)
        return self._talker_kv_pool[ids, :, :, :max_past_len, :].contiguous()

    def gather_talker_kv_into(
        self,
        slot_ids: list[int],
        out: torch.Tensor,
    ) -> None:
        """Gather talker KV for a batch directly into ``out``'s leading rows.

        ``out`` is [B_out, L*2, H, past_out, D] with B_out >= len(slot_ids)
        and past_out <= pool capacity.  Used by the CUDA-graph decode path to
        fill its persistent staging buffer without the transient tensor that
        ``gather_talker_kv`` allocates (up to several GiB at large batch).
        """
        if self._talker_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        past = int(out.shape[3])
        for i, slot_id in enumerate(slot_ids):
            out[i].copy_(
                self._talker_kv_pool[slot_id, :, :, :past, :], non_blocking=True
            )

    def scatter_talker_kv(
        self,
        slot_ids: list[int],
        present_kv: torch.Tensor,
        original_past_lens: list[int],
        padded_past_len: int,
        seq: int,
    ) -> None:
        """Write decode output KV back to the pre-allocated pool.

        Handles de-padding for heterogeneous past lengths.
        """
        if self._talker_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        cap = self._config.max_seq_len
        uniform = len(set(original_past_lens)) <= 1
        for i, (slot_id, orig_pl) in enumerate(zip(slot_ids, original_past_lens)):
            new_total = min(orig_pl + seq, cap)
            if new_total <= orig_pl:
                continue
            if uniform or orig_pl >= padded_past_len:
                self._talker_kv_pool[slot_id, :, :, :new_total, :] = present_kv[
                    i, :, :, :new_total, :
                ]
            else:
                self._talker_kv_pool[slot_id, :, :, :orig_pl, :] = present_kv[
                    i, :, :, :orig_pl, :
                ]
                self._talker_kv_pool[slot_id, :, :, orig_pl:new_total, :] = present_kv[
                    i, :, :, padded_past_len : padded_past_len + seq, :
                ]

    def gather_c2w_kv(
        self,
        slot_ids: list[int],
        max_c2w_len: int,
    ) -> torch.Tensor:
        """Gather C2W KV from the pre-allocated pool."""
        if self._c2w_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        ids = torch.tensor(slot_ids, device=self._device, dtype=torch.long)
        return self._c2w_kv_pool[ids, :, :, :max_c2w_len, :].contiguous()

    def scatter_c2w_kv(
        self,
        slot_ids: list[int],
        present_kv: torch.Tensor,
    ) -> None:
        """Write C2W KV back to the pool (sliding window, per-slot cropping).

        Each slot may have different past_len, so we must crop per-slot
        to the sliding window size before writing to the pool.
        """
        if self._c2w_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        c2w_max_past = self._config.c2w_sliding_window - 1
        for i, slot_id in enumerate(slot_ids):
            kv = present_kv[i : i + 1]
            s_len = kv.shape[3]
            if s_len > c2w_max_past:
                # Crop to sliding window for this slot
                kv = kv[:, :, :, -c2w_max_past:, :]
                s_len = c2w_max_past
            self._c2w_kv_pool[slot_id, :, :, :s_len, :] = kv[0, :, :, :s_len, :]

    def scatter_talker_kv_delta(
        self,
        slot_ids: list[int],
        delta_kv: torch.Tensor,
        original_past_lens: list[int],
    ) -> None:
        """Append talker delta KV to the pre-allocated pool."""
        if self._talker_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        cap = self._config.max_seq_len
        delta_len = int(delta_kv.shape[3])
        for i, (slot_id, orig_pl) in enumerate(zip(slot_ids, original_past_lens)):
            if orig_pl >= cap or delta_len <= 0:
                continue
            write_len = min(delta_len, cap - orig_pl)
            end = orig_pl + write_len
            self._talker_kv_pool[slot_id, :, :, orig_pl:end, :] = delta_kv[
                i, :, :, :write_len, :
            ]

    def scatter_c2w_kv_delta(
        self,
        slot_ids: list[int],
        delta_kv: torch.Tensor,
        original_past_lens: list[int],
    ) -> None:
        """Append code2wav delta KV to the pre-allocated pool with sliding-window crop."""
        if self._c2w_kv_pool is None:
            raise RuntimeError("Pool not pre-allocated")
        window = self._config.c2w_sliding_window - 1
        delta_len = int(delta_kv.shape[3])
        for i, (slot_id, orig_pl) in enumerate(zip(slot_ids, original_past_lens)):
            if delta_len <= 0:
                continue
            keep_past = min(orig_pl, max(0, window - delta_len))
            write_len = min(delta_len, window)
            if keep_past > 0:
                self._c2w_kv_pool[slot_id, :, :, :keep_past, :] = self._c2w_kv_pool[
                    slot_id, :, :, orig_pl - keep_past : orig_pl, :
                ]
            self._c2w_kv_pool[slot_id, :, :, keep_past : keep_past + write_len, :] = (
                delta_kv[i, :, :, delta_len - write_len :, :]
            )

    # ------------------------------------------------------------------
    # Slot eviction
    # ------------------------------------------------------------------

    def find_eviction_candidate(
        self,
        max_idle_sec: float = 10.0,
    ) -> Optional[SlotKVState]:
        """Find the least-recently-active occupied slot that exceeds idle limit.

        Returns None if all slots are either free or within the idle window.
        """
        best: Optional[SlotKVState] = None
        for slot in self._slots:
            if slot.is_free:
                continue
            if slot.idle_seconds < max_idle_sec:
                continue
            if best is None or slot.idle_seconds > best.idle_seconds:
                best = slot
        return best

    def force_evict(self, slot_id: int) -> Optional[str]:
        """Force-release a slot, returning the evicted session_id.

        The caller is responsible for notifying the evicted session.
        """
        slot = self._slots[slot_id]
        if slot.is_free:
            return None
        evicted_session = slot.session_id
        logger.warning(
            "Force-evicting slot %d (session=%s, idle=%.1fs, past_len=%d)",
            slot_id,
            evicted_session,
            slot.idle_seconds,
            slot.past_len,
        )
        self.release(slot_id)
        return evicted_session

    @property
    def utilization(self) -> float:
        """Pool utilization ratio [0, 1]."""
        if self._max_slots == 0:
            return 0.0
        return self.used_count / self._max_slots
