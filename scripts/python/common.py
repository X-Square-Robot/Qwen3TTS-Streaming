from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPTS_PYTHON_DIR = REPO_ROOT / "scripts" / "python"
SCRIPTS_EXPORT_DIR = REPO_ROOT / "scripts" / "export"
TESTS_DIR = REPO_ROOT / "tests"
TESTS_INTEGRATION_DIR = TESTS_DIR / "integration"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
THIRD_PARTY_QWEN_DIR = THIRD_PARTY_DIR / "Qwen3-TTS"
WORKSPACE_DIR = REPO_ROOT / "workspace"

DEFAULT_ENGINE_GRPC = "localhost:50051"
DEFAULT_ENGINE_WS = "ws://localhost:50052/v1/ws"
DEFAULT_TRITON_HTTP = "http://localhost:8000"
DEFAULT_TRITON_GRPC = "localhost:8001"
DEFAULT_TRITON_MODEL = "tts_orchestrator"
DEFAULT_TRITON_HTTP_MODEL = "tts_orchestrator_http"
DEFAULT_TRITON_MODEL_VERSION = os.environ.get("TRITON_MODEL_VERSION", "1")
DEFAULT_SAMPLE_RATE = 24000

DEFAULT_PROBE_TARGETS = (
    "engine-grpc",
    "engine-websocket",
    "triton-http",
    "triton-grpc",
)
DEFAULT_SERVING_TARGETS = (
    "engine-grpc",
    "engine-websocket",
    "triton-grpc",
    "triton-http",
)

IMPORT_PATH_GROUPS = {
    "repo": REPO_ROOT,
    "scripts": SCRIPTS_DIR,
    "scripts_python": SCRIPTS_PYTHON_DIR,
    "scripts_export": SCRIPTS_EXPORT_DIR,
    "tests_integration": TESTS_INTEGRATION_DIR,
    "third_party_qwen": THIRD_PARTY_QWEN_DIR,
}


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def split_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def prepend_sys_paths(*paths: str | Path) -> list[str]:
    ordered = [str(Path(path)) for path in paths if path]
    for raw in reversed(ordered):
        while raw in sys.path:
            sys.path.remove(raw)
        sys.path.insert(0, raw)
    return ordered


def bootstrap_project_imports(*groups: str) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for name in groups:
        try:
            resolved.append(IMPORT_PATH_GROUPS[name])
        except KeyError as exc:
            known = ", ".join(sorted(IMPORT_PATH_GROUPS))
            raise ValueError(
                f"Unknown import path group: {name!r}. Known groups: {known}"
            ) from exc
    prepend_sys_paths(*resolved)
    return tuple(resolved)


def parse_host_port(endpoint: str, *, default_port: int) -> tuple[str, int]:
    value = endpoint.strip()
    host, sep, port_text = value.rpartition(":")
    if not sep:
        return value, default_port
    if not host:
        raise ValueError(f"invalid endpoint: {endpoint!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"invalid port in endpoint: {endpoint!r}") from exc
    return host, port


def normalize_http_base(url: str) -> str:
    return url.rstrip("/")


def story_path() -> Path:
    return TESTS_DIR / "data" / "story.txt"
