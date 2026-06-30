"""Runtime dump helpers for fused prefill/decode debugging.

Enabled only when ``ENGINE_DUMP_DIR`` is set. Dumps are written as one
``.pt`` payload per executor call so they can be replayed offline against
the official PyTorch modules.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

logger = logging.getLogger(__name__)
_SKIP = object()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return safe.strip("._-") or "unknown"


def _split_patterns(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


class EngineDebugDumper:
    """Env-gated dumper for executor inputs/outputs."""

    def __init__(
        self,
        *,
        engine_dir: Optional[Path] = None,
        weights_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        dump_dir = os.environ.get("ENGINE_DUMP_DIR", "").strip()
        self._enabled = bool(dump_dir)
        self._dir = Path(dump_dir).expanduser() if dump_dir else None
        self._limit = int(os.environ.get("ENGINE_DUMP_LIMIT", "0") or "0")
        self._include_wav = _env_flag("ENGINE_DUMP_INCLUDE_WAV", True)
        self._write_text = _env_flag("ENGINE_DUMP_TEXT", True)
        self._write_summary = _env_flag("ENGINE_DUMP_SUMMARY", True)
        self._text_max_elements = int(
            os.environ.get("ENGINE_DUMP_TEXT_MAX_ELEMENTS", "0") or "0"
        )
        self._include_input_patterns = _split_patterns(
            os.environ.get("ENGINE_DUMP_INPUT_KEYS", "").strip()
        )
        self._include_output_patterns = _split_patterns(
            os.environ.get("ENGINE_DUMP_OUTPUT_KEYS", "").strip()
        )
        self._exclude_patterns = _split_patterns(
            os.environ.get("ENGINE_DUMP_EXCLUDE_KEYS", "").strip()
        )
        raw_sessions = os.environ.get("ENGINE_DUMP_SESSIONS", "").strip()
        self._session_filters = {
            item.strip() for item in raw_sessions.split(",") if item.strip()
        }
        self._engine_dir = str(engine_dir) if engine_dir is not None else ""
        self._weights_dir = str(weights_dir) if weights_dir is not None else ""
        self._device = str(device) if device is not None else ""
        self._lock = threading.Lock()
        self._counter = 0

        if self._enabled and self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Engine dump enabled: dir=%s limit=%s include_wav=%s sessions=%s",
                self._dir,
                self._limit if self._limit > 0 else "unlimited",
                self._include_wav,
                sorted(self._session_filters) if self._session_filters else "all",
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def should_dump(self, session_keys: Iterable[str]) -> bool:
        if not self._enabled:
            return False
        if self._limit > 0 and self._counter >= self._limit:
            return False
        keys = [k for k in session_keys if k]
        if not self._session_filters:
            return True
        for key in keys:
            if key in self._session_filters:
                return True
            base = key.split(":", 1)[0]
            if base in self._session_filters:
                return True
        return False

    def dump_call(
        self,
        *,
        metadata: Dict[str, Any],
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        inputs_snapshotted: bool = False,
        outputs_snapshotted: bool = False,
    ) -> Optional[Path]:
        if not self.should_dump(metadata.get("slot_session_ids", [])):
            return None

        with self._lock:
            if self._limit > 0 and self._counter >= self._limit:
                return None
            self._counter += 1
            dump_id = self._counter

        stage = str(metadata.get("stage", "unknown"))
        sessions = metadata.get("slot_session_ids") or ["unknown"]
        session_stub = "__".join(_safe_name(s) for s in sessions[:3])
        if len(sessions) > 3:
            session_stub += f"__plus{len(sessions) - 3}"
        filename = f"{dump_id:06d}_{_safe_name(stage)}_{session_stub}.pt"
        path = self._dir / filename

        payload = {
            "metadata": {
                **metadata,
                "dump_id": dump_id,
                "dump_time_unix": time.time(),
                "engine_dir": self._engine_dir,
                "weights_dir": self._weights_dir,
                "device": self._device,
            },
            "inputs": (
                inputs
                if inputs_snapshotted
                else self._snapshot(inputs, path=("inputs",))
            ),
            "outputs": (
                outputs
                if outputs_snapshotted
                else self._snapshot(
                    outputs,
                    include_wav=self._include_wav,
                    path=("outputs",),
                )
            ),
        }
        torch.save(payload, path)
        if self._write_text:
            self._write_text_dump(path.with_suffix(""), payload)
        if self._write_summary:
            self._write_summary_index(path, payload)
        logger.info(
            "Wrote engine dump %s (%s, batch=%s, sessions=%s)",
            path.name,
            stage,
            metadata.get("batch_size"),
            metadata.get("slot_session_ids"),
        )
        return path

    def capture(
        self,
        obj: Any,
        *,
        include_wav: bool = True,
        root_name: str = "",
    ) -> Any:
        path = (root_name,) if root_name else ()
        return self._snapshot(obj, include_wav=include_wav, path=path)

    def _snapshot(
        self,
        obj: Any,
        *,
        include_wav: bool = True,
        path: tuple[str, ...] = (),
    ) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().clone()
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                child_path = path + (str(key),)
                if not self._should_capture(child_path, include_wav=include_wav):
                    continue
                snap = self._snapshot(
                    value,
                    include_wav=include_wav,
                    path=child_path,
                )
                if snap is not _SKIP:
                    out[key] = snap
            return out
        if isinstance(obj, (list, tuple)):
            return [
                self._snapshot(
                    v,
                    include_wav=include_wav,
                    path=path + (str(idx),),
                )
                for idx, v in enumerate(obj)
            ]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return repr(obj)

    def _should_capture(
        self,
        path: tuple[str, ...],
        *,
        include_wav: bool,
    ) -> bool:
        if not path:
            return True
        leaf = path[-1]
        if not include_wav and leaf == "wav":
            return False

        full = ".".join(path)
        root = path[0]
        include_patterns = []
        if root == "inputs":
            include_patterns = self._include_input_patterns
        elif root == "outputs":
            include_patterns = self._include_output_patterns

        if include_patterns and not self._matches_any(include_patterns, full, leaf):
            return False
        if self._exclude_patterns and self._matches_any(
            self._exclude_patterns, full, leaf
        ):
            return False
        return True

    @staticmethod
    def _matches_any(patterns: list[str], full: str, leaf: str) -> bool:
        for pattern in patterns:
            if fnmatchcase(full, pattern) or fnmatchcase(leaf, pattern):
                return True
        return False

    def _write_summary_index(self, path: Path, payload: Dict[str, Any]) -> None:
        call_summary = self._build_call_summary(path, payload)
        index_path = self._dir / "timeline.jsonl"
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(call_summary, ensure_ascii=False) + "\n")

        rows = call_summary.get("slot_rows", [])
        self._append_tsv(self._dir / "timeline.tsv", rows)
        sessions_dir = self._dir / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        for row in rows:
            session_key = row.get("session_id") or "unknown"
            sess_dir = sessions_dir / _safe_name(session_key)
            sess_dir.mkdir(exist_ok=True)
            with (sess_dir / "timeline.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._append_tsv(sess_dir / "timeline.tsv", [row])

    def _build_call_summary(
        self, path: Path, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        meta = payload.get("metadata", {})
        inputs = payload.get("inputs", {})
        outputs = payload.get("outputs", {})
        batch = int(meta.get("batch_size", 0) or 0)
        rows = [
            self._build_slot_row(
                row_idx=i,
                dump_file=path.name,
                meta=meta,
                inputs=inputs,
                outputs=outputs,
            )
            for i in range(batch)
        ]
        return {
            "dump_id": int(meta.get("dump_id", 0) or 0),
            "dump_file": path.name,
            "stage": meta.get("stage", ""),
            "batch_size": batch,
            "seq_len": int(meta.get("seq_len", 0) or 0),
            "sessions": list(meta.get("slot_session_ids", [])),
            "segments": list(meta.get("slot_segment_indices", [])),
            "slot_rows": rows,
        }

    def _build_slot_row(
        self,
        *,
        row_idx: int,
        dump_file: str,
        meta: Dict[str, Any],
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = meta.get("config", {}) or {}
        c2w_window = int(config.get("c2w_sliding_window", 72) or 72) - 1
        row = {
            "dump_id": int(meta.get("dump_id", 0) or 0),
            "dump_file": dump_file,
            "stage": str(meta.get("stage", "")),
            "batch_size": int(meta.get("batch_size", 0) or 0),
            "seq_len": int(meta.get("seq_len", 0) or 0),
            "use_dummy_kv": bool(meta.get("use_dummy_kv", False)),
            "batch_row": row_idx,
            "slot_id": self._meta_at(meta, "slot_ids", row_idx, default=-1),
            "session_id": self._meta_at(meta, "slot_session_ids", row_idx, default=""),
            "segment_idx": self._meta_at(
                meta, "slot_segment_indices", row_idx, default=-1
            ),
            "prefill_source": self._meta_at(
                meta, "slot_prefill_sources", row_idx, default=""
            ),
            "past_len_before": self._meta_at(
                meta, "slot_past_len_before", row_idx, default=0
            ),
            "original_talker_past_len": self._meta_at(
                meta, "original_talker_past_lens", row_idx, default=0
            ),
            "frame_idx_before": self._meta_at(
                meta, "slot_frame_idx_before", row_idx, default=0
            ),
            "text_idx_before": self._meta_at(
                meta, "slot_text_idx_before", row_idx, default=0
            ),
            "trailing_len": self._meta_at(
                meta, "slot_trailing_len", row_idx, default=0
            ),
            "c2w_len_before": self._meta_at(
                meta, "slot_c2w_len_before", row_idx, default=0
            ),
            "slot_has_next_embed_before": self._meta_at(
                meta, "slot_has_next_embed", row_idx, default=False
            ),
            "slot_has_last_codec_sum_before": self._meta_at(
                meta, "slot_has_last_codec_sum", row_idx, default=False
            ),
        }
        row["c2w_window_remaining_before"] = max(
            0, c2w_window - int(row["c2w_len_before"])
        )
        row["c2w_window_full_before"] = int(row["c2w_len_before"]) >= c2w_window

        row["position_start"] = self._tensor_scalar(
            inputs.get("position_ids"), row_idx, (0, 0, 0)
        )
        row["cache_position"] = self._tensor_scalar(
            inputs.get("cache_position"), row_idx, (0,)
        )
        row["full_codec_0"] = self._tensor_scalar(
            outputs.get("full_codec"), row_idx, (0,)
        )
        row["full_codec_head"] = self._tensor_head(
            outputs.get("full_codec"), row_idx, take=4
        )
        row["wav_numel"] = self._tensor_numel(outputs.get("wav"), row_idx)
        row["talker_delta_len"] = self._tensor_dim(
            outputs.get("talker_new_kv"), row_idx, dim=3
        )
        row["c2w_delta_len"] = self._tensor_dim(
            outputs.get("c2w_new_kv"), row_idx, dim=3
        )
        codec_eos_id = int(config.get("codec_eos_id", -1) or -1)
        row["eos"] = bool(
            codec_eos_id >= 0
            and row["full_codec_0"] is not None
            and int(row["full_codec_0"]) == codec_eos_id
        )
        return row

    @staticmethod
    def _meta_at(meta: Dict[str, Any], key: str, idx: int, default: Any) -> Any:
        values = meta.get(key)
        if isinstance(values, list) and idx < len(values):
            return values[idx]
        return default

    @staticmethod
    def _slice_batch_tensor(tensor: Any, row_idx: int) -> Optional[torch.Tensor]:
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.ndim > 0 and row_idx < tensor.shape[0]:
            return tensor[row_idx]
        return tensor

    def _tensor_scalar(
        self,
        tensor: Any,
        row_idx: int,
        offset: tuple[int, ...],
    ) -> Optional[float | int]:
        sliced = self._slice_batch_tensor(tensor, row_idx)
        if sliced is None or sliced.numel() == 0:
            return None
        ref = sliced
        for idx in offset:
            if ref.ndim == 0 or ref.shape[0] <= idx:
                return None
            ref = ref[idx]
        if ref.ndim != 0:
            return None
        value = ref.item()
        return int(value) if isinstance(value, (int, bool)) else float(value)

    def _tensor_head(
        self,
        tensor: Any,
        row_idx: int,
        *,
        take: int,
    ) -> list[int]:
        sliced = self._slice_batch_tensor(tensor, row_idx)
        if sliced is None or sliced.numel() == 0:
            return []
        flat = sliced.reshape(-1)[:take].tolist()
        return [int(v) for v in flat]

    def _tensor_numel(self, tensor: Any, row_idx: int) -> int:
        sliced = self._slice_batch_tensor(tensor, row_idx)
        if sliced is None:
            return 0
        return int(sliced.numel())

    def _tensor_dim(self, tensor: Any, row_idx: int, *, dim: int) -> int:
        sliced = self._slice_batch_tensor(tensor, row_idx)
        if sliced is None or sliced.ndim <= dim:
            return 0
        return int(sliced.shape[dim])

    def _append_tsv(self, path: Path, rows: list[Dict[str, Any]]) -> None:
        if not rows:
            return
        header = [
            "dump_id",
            "dump_file",
            "stage",
            "batch_size",
            "seq_len",
            "use_dummy_kv",
            "batch_row",
            "slot_id",
            "session_id",
            "segment_idx",
            "prefill_source",
            "past_len_before",
            "original_talker_past_len",
            "frame_idx_before",
            "text_idx_before",
            "trailing_len",
            "c2w_len_before",
            "c2w_window_remaining_before",
            "c2w_window_full_before",
            "slot_has_next_embed_before",
            "slot_has_last_codec_sum_before",
            "position_start",
            "cache_position",
            "full_codec_0",
            "full_codec_head",
            "talker_delta_len",
            "c2w_delta_len",
            "wav_numel",
            "eos",
        ]
        write_header = not path.exists()
        with path.open("a", encoding="utf-8") as f:
            if write_header:
                f.write("\t".join(header) + "\n")
            for row in rows:
                f.write("\t".join(str(row.get(col, "")) for col in header) + "\n")

    def _write_text_dump(self, base_path: Path, payload: Dict[str, Any]) -> None:
        base_path.mkdir(parents=True, exist_ok=True)
        meta_path = base_path / "metadata.json"
        meta_path.write_text(
            self._json_dumps(payload["metadata"]),
            encoding="utf-8",
        )

        for group_name in ("inputs", "outputs"):
            group = payload.get(group_name, {})
            group_dir = base_path / group_name
            group_dir.mkdir(exist_ok=True)
            for key, value in group.items():
                self._write_value(group_dir, key, value)

    def _write_value(self, root: Path, key: str, value: Any) -> None:
        safe_key = _safe_name(key)
        if isinstance(value, torch.Tensor):
            shape_str = "x".join(str(int(dim)) for dim in value.shape) or "scalar"
            name = f"{safe_key}__{shape_str}__{value.dtype}.txt"
            self._write_tensor_txt(root / name, value)
            return
        if isinstance(value, list):
            list_dir = root / safe_key
            list_dir.mkdir(exist_ok=True)
            for idx, item in enumerate(value):
                self._write_value(list_dir, f"{idx:03d}", item)
            return
        if isinstance(value, dict):
            dict_dir = root / safe_key
            dict_dir.mkdir(exist_ok=True)
            for child_key, child_value in value.items():
                self._write_value(dict_dir, child_key, child_value)
            return
        out_path = root / f"{safe_key}.txt"
        out_path.write_text(f"{value}\n", encoding="utf-8")

    def _write_tensor_txt(self, path: Path, tensor: torch.Tensor) -> None:
        total = int(tensor.numel())
        limit = self._text_max_elements
        truncated = limit > 0 and total > limit
        if tensor.ndim == 0:
            scalar = tensor.item()
            path.write_text(
                f"{float(scalar) if tensor.is_floating_point() else scalar: .6f}\n",
                encoding="utf-8",
            )
            return

        cols = int(tensor.shape[-1])
        leading = [int(dim) for dim in tensor.shape[:-1]]
        rows = math.prod(leading) if leading else 1
        reshaped = tensor.reshape(rows, cols)
        shown_rows = rows
        if truncated and cols > 0:
            shown_rows = min(rows, max(1, limit // cols))
        shown = reshaped[:shown_rows]
        if tensor.is_floating_point():
            shown = shown.float()

        if shown_rows == 0:
            body = ""
        elif cols == 0:
            body = "\n" * shown_rows
        else:
            body = (
                "\n".join(
                    " ".join(str(value) for value in row) for row in shown.tolist()
                )
                + "\n"
            )
        path.write_text(body, encoding="utf-8")

        info_path = path.with_name(f"{path.stem}__layout.txt")
        info_lines = [
            f"shape={tuple(int(dim) for dim in tensor.shape)}",
            f"dtype={tensor.dtype}",
            f"numel={total}",
            f"txt_shape=({shown_rows}, {cols})",
            f"truncated={truncated}",
        ]
        if truncated:
            info_lines.append(
                "set ENGINE_DUMP_TEXT_MAX_ELEMENTS=0 for full reshape-preserving dump"
            )
        info_path.write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    @staticmethod
    def _json_dumps(obj: Dict[str, Any]) -> str:
        import json

        return json.dumps(obj, indent=2, ensure_ascii=False)
