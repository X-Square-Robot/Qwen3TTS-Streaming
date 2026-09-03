"""Reproducibility preflight helpers for long-form hallucination studies.

The experiment runner deliberately keeps environment discovery separate from
synthesis.  A failed or incomplete preflight must never silently change an arm
or its model package.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    """Hash *path* without loading a multi-gigabyte engine into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_tree_identity(root: Path, paths: list[Path]) -> dict[str, Any]:
    """Hash a selected code/config tree, including uncommitted evaluator files."""

    resolved_root = root.resolve(strict=True)
    selected: set[Path] = set()
    for requested in paths:
        candidate = requested if requested.is_absolute() else resolved_root / requested
        if candidate.is_file():
            selected.add(candidate.resolve(strict=True))
        elif candidate.is_dir():
            selected.update(
                item.resolve(strict=True)
                for item in candidate.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts
            )
        else:
            raise FileNotFoundError(f"source identity path does not exist: {candidate}")
    files: list[dict[str, Any]] = []
    for path in sorted(selected):
        try:
            relative = path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"source identity path escapes root: {path}") from exc
        files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for entry in files:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(resolved_root),
        "file_count": len(files),
        "tree_sha256": digest.hexdigest(),
        "files": files,
    }


def exact_text_record(path: Path) -> dict[str, Any]:
    """Read a UTF-8 stimulus exactly and describe its immutable identity."""

    raw = path.read_bytes()
    text = raw.decode("utf-8")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "codepoints": len(text),
        "has_trailing_newline": text.endswith(("\n", "\r")),
        "text": text,
    }


def git_identity(repo_root: Path) -> dict[str, Any]:
    """Return the exact source identity without mutating the worktree."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--short"),
    }


def model_package_identity(package_dir: Path) -> dict[str, Any]:
    """Describe a packaged runtime and its large immutable artifacts."""

    resolved = package_dir.resolve()
    runtime = resolved / "runtime"
    plan = runtime / "model.plan"
    manifest_path = resolved / "triton_manifest.json"
    artifact_manifest_path = resolved / "artifact_manifest.json"
    payload: dict[str, Any] = {
        "path": str(resolved),
        "model_version": (resolved / "MODEL_VERSION").read_text().strip(),
        "engine_build_version": (resolved / "ENGINE_BUILD_VERSION").read_text().strip(),
        "plan": {
            "path": str(plan),
            "bytes": plan.stat().st_size,
            "sha256": sha256_file(plan),
        },
    }
    if manifest_path.is_file():
        payload["triton_manifest"] = json.loads(manifest_path.read_text())
        payload["triton_manifest_sha256"] = sha256_file(manifest_path)
    if artifact_manifest_path.is_file():
        payload["artifact_manifest"] = json.loads(artifact_manifest_path.read_text())
        payload["artifact_manifest_sha256"] = sha256_file(artifact_manifest_path)
    payload["packaged_code"] = source_tree_identity(
        resolved,
        [
            Path("model.py"),
            Path("engine"),
            Path("qwen3tts_protocol"),
        ],
    )
    payload["packaged_assets"] = source_tree_identity(
        resolved,
        [
            Path("tokenizer"),
            Path("weights"),
        ],
    )
    return payload


def tokenizer_text_identity(package_dir: Path, text_file: Path) -> dict[str, Any]:
    """Fingerprint the exact frozen tokenizer input IDs for the raw stimulus."""

    resolved = package_dir.resolve(strict=True)
    module_path = resolved / "engine" / "frontend" / "spliter" / "tokenizer.py"
    spec = importlib.util.spec_from_file_location(
        "_qwen3tts_frozen_longform_tokenizer", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen tokenizer module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.LightQwen3TTSTokenizer(str(resolved / "tokenizer"))
    text = text_file.read_bytes().decode("utf-8")
    ids = [
        int(token_id)
        for token_id in tokenizer.encode_ids(text, add_special_tokens=False)
    ]
    canonical_ids = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return {
        "tokenizer_class": "LightQwen3TTSTokenizer",
        "tokenizers_version": installed_distribution_version("tokenizers"),
        "add_special_tokens": False,
        "token_count": len(ids),
        "ids_sha256": hashlib.sha256(canonical_ids).hexdigest(),
        "ids": ids,
    }


def checkpoint_identity(checkpoint_dir: Path) -> dict[str, Any]:
    """Fingerprint the official checkpoint without relying on symlink names.

    Large model shards are hashed as well as configuration files.  This is a
    preflight operation, so the extra sequential I/O is preferable to an
    ambiguous "same checkpoint" claim in the final report.
    """

    requested = checkpoint_dir.absolute()
    resolved = checkpoint_dir.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"checkpoint is not a directory: {resolved}")
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = path.relative_to(resolved)
        # Cache files and editor debris are not model inputs.
        if any(part.startswith(".") for part in relative.parts):
            continue
        files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for entry in files:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "file_count": len(files),
        "tree_sha256": digest.hexdigest(),
        "files": files,
    }


def wheel_identity(wheel_path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Hash the exact SDK wheel used to create the dedicated ASR venv."""

    resolved = wheel_path.resolve(strict=True)
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise RuntimeError(
            "FunASR wheel hash mismatch: "
            f"expected {expected_sha256.lower()}, found {digest.lower()}"
        )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest,
    }


def installed_distribution_version(name: str) -> str:
    """Return installed package metadata, failing rather than guessing."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is not installed: {name}") from exc


def runtime_identity() -> dict[str, Any]:
    """Capture process, CUDA/TRT and GPU identities used by a phase."""

    distributions: dict[str, str] = {}
    for name in (
        "torch",
        "tokenizers",
        "tensorrt",
        "tritonclient",
        "numpy",
        "grpcio",
        "PyYAML",
        "soxr",
        "wetext",
        "aiohttp",
        "qwen3-tts-client",
        "funasrnano",
        "cn2an",
        "rapidfuzz",
        "librosa",
        "soundfile",
    ):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    gpu: list[dict[str, str]] = []
    try:
        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        for line in raw.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) == 5:
                gpu.append(
                    dict(
                        zip(
                            ("index", "name", "uuid", "driver_version", "memory_mib"),
                            values,
                            strict=True,
                        )
                    )
                )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "distributions": distributions,
        "gpus": gpu,
    }


def capabilities_url(asr_ws_url: str) -> str:
    """Derive the typed-v1 capabilities endpoint from a FunASR WebSocket URL."""

    parsed = urllib.parse.urlparse(asr_ws_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError(f"invalid ASR WebSocket URL: {asr_ws_url!r}")
    parts = parsed.path.rstrip("/").split("/")
    if len(parts) < 2 or parts[-1] != "ws":
        raise ValueError("ASR WebSocket URL must end in /<generation>/ws")
    scheme = "https" if parsed.scheme == "wss" else "http"
    path = "/".join(parts[:-1]) + "/capabilities"
    return urllib.parse.urlunparse((scheme, parsed.netloc, path, "", "", ""))


def fetch_asr_capabilities(
    asr_ws_url: str,
    *,
    timeout_s: float = 10.0,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch service capabilities fail-closed for a reproducible experiment."""

    request_headers = {"Accept": "application/json", **dict(headers or {})}
    request = urllib.request.Request(
        capabilities_url(asr_ws_url), headers=request_headers
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ASR capabilities response must be a JSON object")
    return payload


def assert_read_only_package(package_dir: Path) -> None:
    """Validate frozen artifacts; launch helpers enforce a read-only mount."""

    important = (
        package_dir / "MODEL_VERSION",
        package_dir / "triton_manifest.json",
        package_dir / "runtime" / "model.plan",
    )
    missing = [str(path) for path in important if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete model package: {missing}")
