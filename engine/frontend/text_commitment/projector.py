"""Conservative readable-content projection for structured payloads."""
from __future__ import annotations

import json
import re
from typing import Any


_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")
_AUTOLINK_RE = re.compile(r"<(?:https?://|mailto:|www\.)[^>]+>", re.I)


def _has_delimited_pair(text: str, marker: str) -> bool:
    """Return whether *text* contains a safe, matching format pair.

    A single ``_`` or ``~`` is common in identifiers and prose.  We only
    classify a marker as Markdown when the same delimiter occurs at least
    twice.  Single underscores additionally need a word boundary on one side;
    this keeps ``a_b_c`` and similar identifiers literal while still handling
    ``_emphasis_`` and ``before_important_``.
    """

    if len(marker) == 1:
        # Treat a consecutive run as one delimiter.  Otherwise an unclosed
        # ``**bold`` would be mistaken for a pair of single-star markers.
        positions = [
            match.start()
            for match in re.finditer(
                rf"(?<!{re.escape(marker)}){re.escape(marker)}(?!{re.escape(marker)})",
                text,
            )
        ]
    else:
        positions = [match.start() for match in re.finditer(re.escape(marker), text)]
    if len(positions) < 2:
        return False
    if marker == "_":
        for left, right in zip(positions, positions[1:]):
            before = text[left - 1] if left else ""
            after_index = right + len(marker)
            after = text[after_index] if after_index < len(text) else ""
            if not (before.isalnum() and after.isalnum()):
                return True
        return False
    return True


def is_markdown_structured(text: str) -> bool:
    """Return whether *text* has enough evidence for Markdown projection.

    This predicate is deliberately conservative.  It is used by the
    incremental lexer when a formatting character is adjacent to an ordinary
    word; uncertain punctuation must remain literal rather than being
    silently deleted.
    """

    if not text:
        return False
    if "```" in text and text.count("```") >= 2:
        return True
    if _IMAGE_RE.search(text) or _LINK_RE.search(text) or _AUTOLINK_RE.search(text):
        return True
    if re.search(r"(?m)^\s{0,3}#{1,6}\s+", text):
        return True
    if re.search(r"(?m)^\s*(?:[-+*>]|\d{1,4}[.)])\s+", text):
        return True
    if re.search(r"(?m)^\s*(?:-{3,}|_{3,}|\*{3,})\s*$", text):
        return True
    # Longest delimiters first prevents a triple marker from being counted as
    # three independent single-star/underscore pairs.
    for marker in ("***", "___", "~~", "**", "__", "`", "*", "_"):
        if _has_delimited_pair(text, marker):
            return True
    return False


def _strip_inline_markers(text: str) -> str:
    """Remove only delimiters that have a confirmed matching pair."""

    for marker in ("***", "___", "~~", "**", "__", "`", "*", "_"):
        if _has_delimited_pair(text, marker):
            text = text.replace(marker, "")
    return text


def project_readable(text: str) -> str:
    stripped = text.strip()
    # A provisional Markdown line marker may be closed before its body arrives
    # (for example ``>`` | ``quote``).  It is formatting-only and must not be
    # spoken as a comparison/operator symbol.
    if re.fullmatch(r"[-+>]\s*", text):
        return ""
    if re.fullmatch(r"\d{1,4}[.)]\s*", text):
        return ""
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
    if re.search(r"\d", stripped) and re.fullmatch(
        r"[0-9+\-*/=×÷xX().\s<>≤≥^]+", stripped
    ):
        return text
    if "```" in text and text.count("```") >= 2:
        blocks = re.findall(r"```[^\n]*\n?(.*?)```", text, flags=re.S)
        readable: list[str] = []
        for block in blocks:
            readable.extend(re.findall(r"//[^\n]*|#[^\n]*|/\*.*?\*/|'[^']*'|\"[^\"]*\"", block, flags=re.S))
        return " ".join(readable)
    # Keep visible markdown prose and link labels, dropping formatting/URLs.
    text = _IMAGE_RE.sub("", text)
    # Link labels are the readable payload.  Accept an empty/whitespace-only
    # label and trim its formatting padding so ``[ ](url)`` projects to an
    # empty string rather than leaking a space.
    text = _AUTOLINK_RE.sub("", text)
    text = re.sub(
        r"\[([^\]]*)\]\([^)]*\)",
        lambda match: match.group(1).strip(),
        text,
    )
    # Markdown markers can appear inline in generated prose after a Chinese
    # clause (for example ``比如# 标题，1. 项目``).  The lexer keeps the
    # complete structured span together, so the line-start-only rule below is
    # insufficient and would leak marker digits into the spoken text.
    text = re.sub(
        r"(^|[\s，。！？!?；;：:])([ \t]*#{1,6}[ \t]+)",
        r"\1",
        text,
        flags=re.M,
    )
    text = re.sub(
        r"(^|[\s，。！？!?；;：:])([ \t]*(?:[-+*>]|\d{1,4}[.)])[ \t]+)",
        r"\1",
        text,
        flags=re.M,
    )
    # A compact Chinese list marker is also commonly written without a space
    # before the dash: ``比如- 项目一``.  Require whitespace after the dash
    # so ordinary hyphenated words remain untouched.
    text = re.sub(r"([\u3400-\u9fff])[-+*>][ \t]+", r"\1", text)
    text = re.sub(
        # Put line-prefix markers before the generic ``*`` delimiter.  In
        # ``* item`` the latter would otherwise win the alternation and leave
        # the marker's separating space behind.
        r"(^\s*[-*>+]\s+|^\s*\d{1,4}[.)]\s+|^\s{0,3}#{1,6}\s*)",
        "",
        text,
        flags=re.M,
    )
    # A marker-only line has no readable payload.  Removing it here also
    # covers a provisional line-prefix span that closes before its body.
    text = re.sub(r"(?m)^\s*[-+>]\s*$", "", text)
    text = re.sub(r"(?m)^\s*(?:-{3,}|_{3,}|\*{3,})\s*$", "", text)
    return _strip_inline_markers(text)


__all__ = ["is_markdown_structured", "project_readable"]
