"""Status subcommand — show project pipeline status."""

from __future__ import annotations

import argparse
import sys


def run_status(args: argparse.Namespace) -> int:
    """Show project status using the Python implementation."""
    try:
        from qwen3tts_tools.status import check_all, format_status

        status = check_all()
        if getattr(args, "json", False):
            import json
            from dataclasses import asdict
            print(json.dumps(asdict(status), indent=2))
        else:
            print(format_status(status))
        return 0
    except ImportError:
        print("Error: qwen3tts_tools not available. Install with: pip install -e .", file=sys.stderr)
        return 1
