#!/usr/bin/env python3
"""Replay difficult text through the incremental TN committer.

The token mode is a deterministic transport approximation (ASCII runs and
single CJK/codepoint units), not a claim about a particular LLM tokenizer.  It
is useful for checking arbitrary packet boundaries and append-only invariants.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from engine.frontend.text_commitment import IncrementalTextCommitter


def _token_chunks(text: str, *, max_chars: int = 8) -> list[str]:
    chunks: list[str] = []
    current = ""
    for ch in text:
        ascii_run = ch.isascii() and (ch.isalnum() or ch in "_@.$:/+-")
        previous_ascii = bool(current) and current[-1].isascii() and (current[-1].isalnum() or current[-1] in "_@.$:/+-")
        if current and (not ascii_run or not previous_ascii or len(current) >= max_chars):
            chunks.append(current)
            current = ""
        current += ch
    if current:
        chunks.append(current)
    return chunks


def _chunks(text: str, mode: str, rng: random.Random) -> list[str]:
    if mode == "codepoint":
        return list(text)
    if mode == "token":
        return _token_chunks(text)
    out: list[str] = []
    i = 0
    while i < len(text):
        size = rng.randint(1, 4)
        out.append(text[i : i + size])
        i += size
    return out


def _run(text: str, chunks: list[str]) -> tuple[str, list[dict], str]:
    committer = IncrementalTextCommitter()
    output: list[str] = []
    commits: list[dict] = []
    last_raw_end = 0
    last_fence = -1
    errors: list[str] = []
    for chunk in chunks:
        try:
            decision = committer.feed(chunk)
        except Exception as exc:  # pragma: no cover - defensive replay guard
            errors.append(f"feed:{type(exc).__name__}:{exc}")
            continue
        for commit in decision.commits:
            if commit.raw_end < last_raw_end or commit.fence < last_fence:
                errors.append("non_monotonic_commit")
            last_raw_end = commit.raw_end
            last_fence = commit.fence
            output.append(commit.tts_text)
            commits.append(
                {
                    "raw_start": commit.raw_start,
                    "raw_end": commit.raw_end,
                    "text": commit.tts_text,
                    "kind": commit.span_kind.value,
                    "commit": commit.commit_kind.value,
                    "fence": commit.fence,
                }
            )
    try:
        final = committer.feed("", final=True)
    except Exception as exc:  # pragma: no cover
        errors.append(f"final:{type(exc).__name__}:{exc}")
        final = None
    if final is not None:
        for commit in final.commits:
            if commit.raw_end < last_raw_end or commit.fence < last_fence:
                errors.append("non_monotonic_commit")
            last_raw_end = commit.raw_end
            last_fence = commit.fence
            output.append(commit.tts_text)
            commits.append(
                {
                    "raw_start": commit.raw_start,
                    "raw_end": commit.raw_end,
                    "text": commit.tts_text,
                    "kind": commit.span_kind.value,
                    "commit": commit.commit_kind.value,
                    "fence": commit.fence,
                }
            )
    return "".join(output), commits, ";".join(errors)


def _load_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or value.startswith("共"):
            continue
        lines.append(value.replace("\\n", "\n"))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("resources/dataset/badcase/difficult.txt"))
    parser.add_argument("--extra", type=Path, default=Path("resources/dataset/badcase/tn_streaming_cases.txt"))
    parser.add_argument("--mode", choices=("token", "codepoint", "random"), default="token")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    cases = _load_lines(args.input)
    if args.extra and args.extra.exists():
        cases.extend(_load_lines(args.extra))
    rng = random.Random(args.seed)
    rows: list[dict] = []
    for text in cases:
        full, _, full_error = _run(text, [text])
        streamed, commits, stream_error = _run(text, _chunks(text, args.mode, rng))
        rows.append(
            {
                "input": text,
                "full": full,
                "streamed": streamed,
                "consistent": full == streamed,
                "pending_safe": not stream_error,
                "commit_count": len(commits),
                "error": ";".join(item for item in (full_error, stream_error) if item),
            }
        )
    summary = {
        "input": str(args.input),
        "extra": str(args.extra),
        "mode": args.mode,
        "seed": args.seed,
        "cases": len(rows),
        "consistent": sum(row["consistent"] for row in rows),
        "inconsistent": sum(not row["consistent"] for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "rows": rows,
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"cases={summary['cases']} consistent={summary['consistent']} inconsistent={summary['inconsistent']} errors={summary['errors']}")
        for row in rows:
            if not row["consistent"] or row["error"]:
                print(f"DIFF input={row['input']!r} full={row['full']!r} streamed={row['streamed']!r} error={row['error']!r}")
    return 1 if args.fail_on_error and (summary["inconsistent"] or summary["errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
