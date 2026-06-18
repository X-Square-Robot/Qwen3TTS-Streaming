"""Run the TTS engine server with full-dump configuration.

A Python replacement for the former ``scripts/bash/run_engine_story_full_dump.sh``.
Sets up environment variables for engine dump mode and launches the engine server.

Usage::

    python scripts/python/run_engine_dump.py --help
    python scripts/python/run_engine_dump.py --dump-tag 4a_greedy --session longtext-4a-greedy
    python scripts/python/run_engine_dump.py --session longtext-story-greedy --do-sample true
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run TTS engine server with dump configuration.",
    )
    p.add_argument("--config", default="engine.yaml", help="Engine config path (default: engine.yaml)")
    p.add_argument("--dump-tag", default="full_dump", help="Prefix for auto dump dir name (default: full_dump)")
    p.add_argument("--dump-dir", default="", help="Explicit dump dir; overrides --dump-tag")
    p.add_argument("--session", default="", help="Comma-separated session ids to dump")
    p.add_argument("--do-sample", default="false", help="Sampling switch (default: false)")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    p.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty (default: 1.05)")
    p.add_argument("--top-k", type=int, default=50, help="Sampling top-k (default: 50)")
    p.add_argument("--include-wav", default="0", help="Include wav tensor in dumps (0|1, default: 0)")
    p.add_argument("--dump-limit", type=int, default=0, help="Max dump calls; 0 means unlimited (default: 0)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Determine dump directory
    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_dir = _REPO_ROOT / "workspace" / "engine_dumps" / f"{args.dump_tag}_{timestamp}"

    # Set environment variables for engine dump mode
    env_overrides = {
        "ENGINE_SAMPLING_DO_SAMPLE": args.do_sample,
        "ENGINE_SAMPLING_TEMPERATURE": str(args.temperature),
        "ENGINE_SAMPLING_REPETITION_PENALTY": str(args.repetition_penalty),
        "ENGINE_SAMPLING_TOP_K": str(args.top_k),
        "ENGINE_SPLITER_MAX_CONCURRENT_SEGMENTS": os.environ.get("ENGINE_SPLITER_MAX_CONCURRENT_SEGMENTS", "8"),
        "ENGINE_SERVER_REQUEST_TIMEOUT_SEC": os.environ.get("ENGINE_SERVER_REQUEST_TIMEOUT_SEC", "1800"),
        "ENGINE_SCHEDULER_SESSION_TIMEOUT_SEC": os.environ.get("ENGINE_SCHEDULER_SESSION_TIMEOUT_SEC", "1800"),
        "ENGINE_SESSION_RESULT_QUEUE_MAXSIZE": os.environ.get("ENGINE_SESSION_RESULT_QUEUE_MAXSIZE", "4096"),
        "ENGINE_GRPC_AUDIO_QUEUE_MAXSIZE": os.environ.get("ENGINE_GRPC_AUDIO_QUEUE_MAXSIZE", "4096"),
        "ENGINE_DUMP_DIR": str(dump_dir),
        "ENGINE_DUMP_SESSIONS": args.session,
        "ENGINE_DUMP_LIMIT": str(args.dump_limit),
        "ENGINE_DUMP_TEXT": os.environ.get("ENGINE_DUMP_TEXT", "1"),
        "ENGINE_DUMP_SUMMARY": os.environ.get("ENGINE_DUMP_SUMMARY", "1"),
        "ENGINE_DUMP_INCLUDE_WAV": args.include_wav,
        "ENGINE_DUMP_TEXT_MAX_ELEMENTS": os.environ.get("ENGINE_DUMP_TEXT_MAX_ELEMENTS", "0"),
        "ENGINE_DUMP_INPUT_KEYS": os.environ.get(
            "ENGINE_DUMP_INPUT_KEYS",
            "input_embeds,position_ids,token_counts,gumbel_noise,cp_gumbel_noise,temperature,penalty,"
            "attention_bias,talker_past_kv,cache_position,c2w_attention_bias,c2w_past_kv,"
            "c2w_conv_state_*,c2w_transconv_overlap_*",
        ),
        "ENGINE_DUMP_OUTPUT_KEYS": os.environ.get(
            "ENGINE_DUMP_OUTPUT_KEYS",
            "full_codec,codec_sum,updated_token_counts,talker_new_kv,c2w_new_kv,"
            "c2w_new_conv_state_*,c2w_new_transconv_overlap_*,wav",
        ),
        "ENGINE_DUMP_EXCLUDE_KEYS": os.environ.get("ENGINE_DUMP_EXCLUDE_KEYS", "hidden,logits"),
    }

    for key, value in env_overrides.items():
        os.environ[key] = value

    # Create dump directory
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print(f"Config: {args.config}")
    for key in [
        "ENGINE_SAMPLING_DO_SAMPLE",
        "ENGINE_SAMPLING_TEMPERATURE",
        "ENGINE_SAMPLING_REPETITION_PENALTY",
        "ENGINE_SAMPLING_TOP_K",
        "ENGINE_SPLITER_MAX_CONCURRENT_SEGMENTS",
        "ENGINE_SERVER_REQUEST_TIMEOUT_SEC",
        "ENGINE_SCHEDULER_SESSION_TIMEOUT_SEC",
        "ENGINE_SESSION_RESULT_QUEUE_MAXSIZE",
        "ENGINE_GRPC_AUDIO_QUEUE_MAXSIZE",
        "ENGINE_DUMP_DIR",
        "ENGINE_DUMP_SESSIONS",
        "ENGINE_DUMP_LIMIT",
        "ENGINE_DUMP_TEXT",
        "ENGINE_DUMP_SUMMARY",
        "ENGINE_DUMP_INCLUDE_WAV",
        "ENGINE_DUMP_TEXT_MAX_ELEMENTS",
        "ENGINE_DUMP_INPUT_KEYS",
        "ENGINE_DUMP_OUTPUT_KEYS",
        "ENGINE_DUMP_EXCLUDE_KEYS",
    ]:
        print(f"{key}={os.environ[key]}")

    # Exec into the engine server (same semantics as the Bash `exec`)
    # Try mamba first, then conda, then plain python
    python_cmd = ["python", "-m", "engine.server", "--config", args.config]
    try:
        os.execvp("mamba", ["mamba", "run", "-n", "qwen3-tts"] + python_cmd)
    except FileNotFoundError:
        try:
            os.execvp("conda", ["conda", "run", "-n", "qwen3-tts"] + python_cmd)
        except FileNotFoundError:
            os.execvp("python", ["python"] + python_cmd)


if __name__ == "__main__":
    main()
