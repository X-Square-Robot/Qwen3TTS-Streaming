"""Opt-in native cursor diagnostics for reproducing coordinate regressions."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def dump_cursor_event(phase: str, **fields: Any) -> None:
    """Append one JSONL cursor event when ``QWEN3_CURSOR_DUMP`` is enabled."""
    if os.environ.get("QWEN3_CURSOR_DUMP", "").lower() not in {"1", "true", "yes"}:
        return
    payload = {"ts": time.time(), "phase": str(phase), **fields}
    path = Path(os.environ.get("QWEN3_CURSOR_DUMP_PATH", "/tmp/qwen3-cursor-dump.jsonl"))
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)


__all__ = ("dump_cursor_event",)
