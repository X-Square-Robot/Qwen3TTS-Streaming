"""Conservative readable-content projection for structured payloads."""
from __future__ import annotations

import json
import re
from typing import Any


def project_readable(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            value = json.loads(stripped)
            vals: list[str] = []

            def walk(v: Any) -> None:
                if isinstance(v, dict):
                    for item in v.values():
                        walk(item)
                elif isinstance(v, list):
                    for item in v:
                        walk(item)
                elif isinstance(v, (str, int, float, bool)):
                    vals.append(str(v))

            walk(value)
            return " ".join(vals)
        except Exception:
            pass
    if re.fullmatch(r"[0-9+\-*/=×÷().\s]+", stripped):
        return text
    if "```" in text:
        blocks = re.findall(r"```[^\n]*\n?(.*?)```", text, flags=re.S)
        readable: list[str] = []
        for block in blocks:
            readable.extend(re.findall(r"//[^\n]*|#[^\n]*|/\*.*?\*/|'[^']*'|\"[^\"]*\"", block, flags=re.S))
        return " ".join(readable)
    # Keep visible markdown prose and link labels, dropping formatting/URLs.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(`{1,3}|\*{1,3}|_{1,3}|^\s{0,3}#{1,6}\s*|^\s*[-*>+]\s+)", "", text, flags=re.M)
    return text
