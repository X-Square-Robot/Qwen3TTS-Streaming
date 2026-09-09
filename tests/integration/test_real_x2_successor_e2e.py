"""Environment-gated real X2 successor-continuity E2E evidence.

This test invokes the external X2 checkpoint runner against a real fused TRT
artifact.  It proves that the main streaming TN, splitter, C2W state bridge,
and native cursor continuation can complete multiple text packets.  It does
not by itself prove speech-state model equivalence or release readiness.

Run explicitly with the real model/artifact environment available::

    RUN_REAL_X2_E2E_TESTS=1 pytest -q \
        tests/integration/test_real_x2_successor_e2e.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _required_paths() -> tuple[Path, Path, Path, Path]:
    if os.environ.get("RUN_REAL_X2_E2E_TESTS", "").strip() != "1":
        pytest.skip("set RUN_REAL_X2_E2E_TESTS=1 for real X2 E2E evidence")

    x2_root = Path(
        os.environ.get("X2STREAMING_ROOT", "/home/rime/workspace/x2streaming")
    ).resolve()
    engine_dir = Path(
        os.environ.get(
            "QWEN_REAL_X2_ENGINE_DIR",
            "/home/rime/workspace/models/x2-exported/custom-1.7b",
        )
    ).resolve()
    weights_dir = Path(
        os.environ.get("QWEN_REAL_X2_WEIGHTS_DIR", str(engine_dir / "weights"))
    ).resolve()
    tokenizer_dir = Path(
        os.environ.get(
            "QWEN_REAL_X2_TOKENIZER_DIR",
            "/home/rime/workspace/models/X2Streaming-TTS-1.7B",
        )
    ).resolve()

    runner = x2_root / "scripts" / "run_checkpoint_e2e.py"
    required = (runner, engine_dir / "talker_code2wav_fused.engine", weights_dir, tokenizer_dir)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip("real X2 E2E inputs are missing: " + ", ".join(missing))
    return x2_root, engine_dir, weights_dir, tokenizer_dir


def _json_result(stdout: str) -> dict:
    stdout = stdout.strip()
    decoder = json.JSONDecoder()
    for index in range(len(stdout) - 1, -1, -1):
        if stdout[index] != "{":
            continue
        try:
            value, end = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if end == len(stdout) - index and isinstance(value, dict):
            return value
    raise AssertionError("the X2 E2E runner did not emit a JSON result")


def _run_case(
    *,
    x2_root: Path,
    engine_dir: Path,
    weights_dir: Path,
    tokenizer_dir: Path,
    disable_x2: bool = False,
    output_wav: Path | None = None,
) -> tuple[dict, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(x2_root / "src"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["ENGINE_CUDA_GRAPH_DECODE"] = "0"
    packets = ("今天温度25", "℃。明天降至18", "℃，请注意保暖。")
    command = [
        sys.executable,
        str(x2_root / "scripts" / "run_checkpoint_e2e.py"),
        "--upstream-root",
        str(Path(__file__).resolve().parents[2]),
        "--engine-dir",
        str(engine_dir),
        "--weights-dir",
        str(weights_dir),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--speaker",
        "robot_service_v1",
        "--timeout",
        "240",
        "--warmup-rounds",
        "0",
    ]
    if disable_x2:
        command.append("--disable-x2")
    if output_wav is not None:
        command.extend(("--output-wav", str(output_wav)))
    for packet in packets:
        command.extend(("--packet", packet))

    completed = subprocess.run(
        command,
        cwd=x2_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    result = _json_result(completed.stdout)
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert result["audio_bytes"] > 0
    return result, completed.stderr


def test_real_x2_successor_continuity_with_native_cursor() -> None:
    x2_root, engine_dir, weights_dir, tokenizer_dir = _required_paths()
    result, stderr = _run_case(
        x2_root=x2_root,
        engine_dir=engine_dir,
        weights_dir=weights_dir,
        tokenizer_dir=tokenizer_dir,
    )
    assert result["segment_count"] >= 3
    assert result["continuity_carried"] == ["True"] * len(result["continuity_carried"])
    assert result["continuity_carried"]
    assert result["done_metrics"]["server_final_synthesized_text"] == (
        "今天温度二十五摄氏度。明天降至十八摄氏度，请注意保暖。"
    )

    successor_prefills = [
        event
        for event in result["events"]
        if event.get("type") == "prefill_done"
        and event.get("meta", {}).get("cursor_progress") == "native_continuation"
    ]
    assert len(successor_prefills) >= 2
    assert all(
        event["meta"]["extension_continuity"] == "restored"
        and event["meta"]["extension_bridge"] == "prepared"
        and event["meta"]["cursor_progress"] == "native_continuation"
        for event in successor_prefills
    )
    assert "cursor labelization failed" not in stderr.lower()


def test_real_x2_continuity_ab_changes_the_audio_route(tmp_path: Path) -> None:
    x2_root, engine_dir, weights_dir, tokenizer_dir = _required_paths()
    continuity_wav = tmp_path / "continuity.wav"
    hard_boundary_wav = tmp_path / "hard-boundary.wav"
    continuity, continuity_stderr = _run_case(
        x2_root=x2_root,
        engine_dir=engine_dir,
        weights_dir=weights_dir,
        tokenizer_dir=tokenizer_dir,
        output_wav=continuity_wav,
    )
    hard_boundary, hard_boundary_stderr = _run_case(
        x2_root=x2_root,
        engine_dir=engine_dir,
        weights_dir=weights_dir,
        tokenizer_dir=tokenizer_dir,
        disable_x2=True,
        output_wav=hard_boundary_wav,
    )
    assert continuity_wav.stat().st_size > 44
    assert hard_boundary_wav.stat().st_size > 44
    assert continuity["segment_count"] >= 3
    assert continuity["continuity_carried"]
    assert hard_boundary["segment_count"] == 1
    assert hard_boundary["continuity_carried"] == []
    assert continuity["audio_bytes"] != hard_boundary["audio_bytes"]
    assert "cursor labelization failed" not in (
        continuity_stderr + hard_boundary_stderr
    ).lower()
