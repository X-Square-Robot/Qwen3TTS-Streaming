"""Import bootstrap for directly executed tests/tools scripts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PYTHON_DIR = REPO_ROOT / "scripts" / "python"


def bootstrap_tool_imports() -> Path:
    for import_path in (REPO_ROOT, SCRIPTS_PYTHON_DIR):
        raw = str(import_path)
        if raw not in sys.path:
            sys.path.insert(0, raw)
    return REPO_ROOT
