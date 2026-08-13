"""Session-global raw/canonical text coordinates for progress anchors."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable


@dataclass
class CanonicalTextJournal:
    """Append-only raw text and a monotonic canonical representation.

    ``normalize`` is the exact frontend normalizer.  Boundary maps use Python
    Unicode code-point offsets and are rebuilt after each append; rebuilding
    keeps packetization from changing public coordinates while the normalizer
    remains deliberately small.
    """

    normalize: Callable[[str], str]
    raw_text: str = ""
    normalized_text: str = ""
    normalized_to_raw: list[int] = field(default_factory=lambda: [0])
    input_final: bool = False

    def append(self, raw_delta: str) -> tuple[str, int]:
        old_normalized = self.normalized_text
        self.raw_text += raw_delta or ""
        normalized = str(self.normalize(self.raw_text))
        if not normalized.startswith(old_normalized):
            raise ValueError("canonical text normalization rewrote committed text")
        self.normalized_text = normalized
        self.normalized_to_raw = _boundary_map(self.raw_text, normalized)
        return normalized[len(old_normalized) :], len(old_normalized)

    def finish(self) -> None:
        self.input_final = True

    def trim_normalized(self) -> str:
        """Apply the frontend's full-text outer trim while keeping offsets."""
        start = len(self.normalized_text) - len(self.normalized_text.lstrip())
        end = len(self.normalized_text.rstrip())
        if start == 0 and end == len(self.normalized_text):
            return self.normalized_text
        raw_start = self.normalized_to_raw[start]
        raw_end = self.normalized_to_raw[end]
        boundaries = self.normalized_to_raw[start : end + 1]
        self.normalized_text = self.normalized_text[start:end]
        self.normalized_to_raw = [
            max(raw_start, min(raw_end, value)) for value in boundaries
        ]
        return self.normalized_text

    def raw_span(self, normalized_start: int, normalized_end: int) -> tuple[int, int]:
        start = max(0, min(int(normalized_start), len(self.normalized_text)))
        end = max(start, min(int(normalized_end), len(self.normalized_text)))
        return self.normalized_to_raw[start], self.normalized_to_raw[end]


def _boundary_map(raw: str, normalized: str) -> list[int]:
    """Return monotonic normalized-boundary -> raw-boundary coordinates."""

    out: list[int | None] = [None] * (len(normalized) + 1)
    matcher = SequenceMatcher(a=raw, b=normalized, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1 + 1):
                out[j1 + offset] = i1 + offset
        elif j2 > j1:
            raw_width = i2 - i1
            norm_width = j2 - j1
            for offset in range(norm_width + 1):
                out[j1 + offset] = i1 + round(raw_width * offset / norm_width)
        elif j1 < len(out) and out[j1] is None:
            out[j1] = i2
    out[0] = 0
    out[-1] = len(raw)
    last = 0
    for idx, value in enumerate(out):
        if value is None:
            out[idx] = last
        else:
            last = max(last, int(value))
            out[idx] = last
    return [int(value) for value in out]


__all__ = ("CanonicalTextJournal",)
