"""Validate release evidence before advertising native runtime capabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow both ``python -m ...`` from the repository and direct release-script
# invocation via ``python scripts/python/validate_capability_evidence.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.runtime.release_gate import ReleaseCapability, evaluate_release_gate


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--require-native-cursor",
        action="store_true",
        help="fail unless native cursor release evidence is complete",
    )
    parser.add_argument(
        "--require-speech-state",
        action="store_true",
        help="fail unless speech-state release evidence is complete",
    )
    args = parser.parse_args(argv)
    try:
        manifest = _read_object(args.manifest)
        evidence = _read_object(args.evidence)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    gate = evaluate_release_gate(manifest, evidence)
    result = {
        "native_cursor": {
            "verified": gate.verified(ReleaseCapability.NATIVE_CURSOR),
            "reason": gate.reason(ReleaseCapability.NATIVE_CURSOR),
        },
        "speech_state": {
            "verified": gate.verified(ReleaseCapability.SPEECH_STATE),
            "reason": gate.reason(ReleaseCapability.SPEECH_STATE),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    required = (
        args.require_native_cursor
        and not gate.verified(ReleaseCapability.NATIVE_CURSOR)
    ) or (
        args.require_speech_state
        and not gate.verified(ReleaseCapability.SPEECH_STATE)
    )
    return 1 if required else 0


if __name__ == "__main__":
    raise SystemExit(main())
