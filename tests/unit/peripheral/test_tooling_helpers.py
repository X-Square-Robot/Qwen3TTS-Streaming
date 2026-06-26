from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PYTHON = REPO_ROOT / "scripts" / "python"
if str(SCRIPTS_PYTHON) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PYTHON))

from audio import save_wav
from common import (
    REPO_ROOT as HELPER_REPO_ROOT,
    SCRIPTS_EXPORT_DIR,
    THIRD_PARTY_QWEN_DIR,
    bootstrap_project_imports,
    dedupe_keep_order,
    parse_host_port,
    prepend_sys_paths,
    split_csv_arg,
)


def test_parse_host_port_uses_default_port_when_missing():
    assert parse_host_port("localhost", default_port=50051) == ("localhost", 50051)


def test_parse_host_port_rejects_invalid_port():
    with pytest.raises(ValueError, match="invalid port"):
        parse_host_port("localhost:not-a-port", default_port=50051)


def test_split_csv_arg_and_dedupe_keep_order():
    values = split_csv_arg(" engine-grpc, triton-http ,, engine-grpc ")
    assert values == ["engine-grpc", "triton-http", "engine-grpc"]
    assert dedupe_keep_order(values) == ["engine-grpc", "triton-http"]


def test_prepend_sys_paths_preserves_requested_order(monkeypatch):
    monkeypatch.setattr(sys, "path", ["keep-me", str(SCRIPTS_EXPORT_DIR)])
    prepend_sys_paths(REPO_ROOT, SCRIPTS_EXPORT_DIR, THIRD_PARTY_QWEN_DIR)
    assert sys.path[:4] == [
        str(REPO_ROOT),
        str(SCRIPTS_EXPORT_DIR),
        str(THIRD_PARTY_QWEN_DIR),
        "keep-me",
    ]


def test_bootstrap_project_imports_uses_named_groups(monkeypatch):
    monkeypatch.setattr(sys, "path", ["keep-me"])
    resolved = bootstrap_project_imports("repo", "scripts_export", "third_party_qwen")
    assert resolved == (HELPER_REPO_ROOT, SCRIPTS_EXPORT_DIR, THIRD_PARTY_QWEN_DIR)
    assert sys.path[:4] == [
        str(HELPER_REPO_ROOT),
        str(SCRIPTS_EXPORT_DIR),
        str(THIRD_PARTY_QWEN_DIR),
        "keep-me",
    ]


def test_bootstrap_project_imports_rejects_unknown_group():
    with pytest.raises(ValueError, match="Unknown import path group"):
        bootstrap_project_imports("not-a-group")


def test_save_wav_creates_parent_and_clips_audio(tmp_path: Path):
    output = tmp_path / "nested" / "audio.wav"
    save_wav(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32), output, sample_rate=24000)

    assert output.is_file()
    with wave.open(str(output), "rb") as wf:
        assert wf.getframerate() == 24000
        assert wf.getnchannels() == 1
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    assert samples.tolist() == [-32767, -32767, 0, 32767, 32767]
