#!/usr/bin/env python3
"""Run a single long-text standalone-engine case.

Examples:
  mamba run -n qwen3-tts python tests/tools/run_engine_long_case.py --case 4a
  mamba run -n qwen3-tts python tests/tools/run_engine_long_case.py --case story
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tests.tools._bootstrap import bootstrap_tool_imports
except ImportError:
    from _bootstrap import bootstrap_tool_imports

REPO_ROOT = bootstrap_tool_imports()

from tests.support.engine_standalone import (
    CUSTOM_VOICE_INSTRUCT_ZH,
    LONG_TEXT,
    OUTPUT_DIR,
    SAMPLE_RATE,
    _check_server,
    _print_result,
    _save_wav,
    _synthesize_oneshot,
)


def _load_story_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Story text file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Story text file is empty: {path}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["4a", "story"], required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--story-path",
        default=str(REPO_ROOT / "tests" / "data" / "story.txt"),
        help="Input text file for --case story",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Optional instruct override for --case story; default uses CUSTOM_VOICE_INSTRUCT_ZH",
    )
    args = parser.parse_args()

    if not _check_server(args.host, args.port):
        raise SystemExit(f"Server not reachable at {args.host}:{args.port}")

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / "single_cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.case == "4a":
        text = LONG_TEXT
        instruct = ""
        timeout = args.timeout if args.timeout is not None else 180.0
        session_id = args.session_id or "longtext-medium-greedy"
        wav_name = "test4a_long_text_medium_single.wav"
        meta_name = "test4a_long_text_medium_single.json"
    else:
        story_path = Path(args.story_path).expanduser().resolve()
        text = _load_story_text(story_path)
        instruct = args.instruct if args.instruct is not None else CUSTOM_VOICE_INSTRUCT_ZH
        timeout = args.timeout if args.timeout is not None else 600.0
        session_id = args.session_id or "longtext-story-greedy"
        wav_name = "test4d_story_single.wav"
        meta_name = "test4d_story_single.json"

    result = _synthesize_oneshot(
        args.host,
        args.port,
        text=text,
        speaker=args.speaker,
        instruct=instruct,
        session_id=session_id,
        timeout=timeout,
    )
    _print_result(result, args.case)

    report = {
        "case": args.case,
        "session_id": result.session_id,
        "speaker": args.speaker,
        "sample_rate": SAMPLE_RATE,
        "timeout_sec": timeout,
        "text_chars": len(text),
        "instruct": instruct,
        "error": result.error,
        "warnings": result.warnings,
        "first_chunk_ms": result.first_chunk_ms,
        "ttft_ms": result.ttft_ms,
        "start_to_first_audio_ms": result.start_to_first_audio_ms,
        "total_ms": result.total_ms,
        "num_chunks": result.num_chunks,
        "total_samples": result.total_samples,
        "duration_sec": result.duration_sec,
        "rtf": result.rtf,
        "decode_step_mean_ms": result.decode_step_mean_ms,
        "decode_step_p50_ms": result.decode_step_p50_ms,
        "decode_step_p95_ms": result.decode_step_p95_ms,
    }

    if result.audio is not None and result.audio.size > 0:
        wav_path = output_dir / wav_name
        _save_wav(result.audio, str(wav_path))
        report["wav_path"] = str(wav_path)
        print(f"Saved wav: {wav_path}")

    report_path = output_dir / meta_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report: {report_path}")

    if result.error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
