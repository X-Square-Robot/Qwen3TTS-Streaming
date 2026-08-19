from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
WORKSPACE_TRACE_DIR = Path(
    os.environ.get("QWEN_DEMO_TRACE_DIR", str(REPO_ROOT / "workspace" / "demo_traces"))
)


class TraceStore:
    """Persistent storage for demo defaults (currently just `default_request`).

    The earlier multi-backend race fixture has been retired; this class now
    only surfaces the default text/speaker/language/ms_per_token combo so the
    optional engineering lab has sensible initial form values.
    """

    def __init__(
        self,
        *,
        workspace_dir: Path = WORKSPACE_TRACE_DIR,
        fixture_dir: Path = FIXTURE_DIR,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.fixture_dir = fixture_dir

    def default_request_path(self) -> Path:
        workspace_path = self.workspace_dir / "default_request.json"
        if workspace_path.exists():
            return workspace_path
        return self.fixture_dir / "default_request.json"

    def load_default_request(self) -> dict[str, Any]:
        path = self.default_request_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
        return {}

    def save_default_request(self, payload: dict[str, Any]) -> Path:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        path = self.workspace_dir / "default_request.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return path
