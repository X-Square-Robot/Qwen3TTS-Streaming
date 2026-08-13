"""Session-global raw/canonical text coordinates for progress anchors."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    raw_to_normalized: list[int] = field(default_factory=lambda: [0])
    input_final: bool = False

    def append(self, raw_delta: str) -> tuple[str, int]:
        old_normalized = self.normalized_text
        self.raw_text += raw_delta or ""
        normalized = str(self.normalize(self.raw_text))
        if not normalized.startswith(old_normalized):
            raise ValueError("canonical text normalization rewrote committed text")
        self.normalized_text = normalized
        self.normalized_to_raw, self.raw_to_normalized = _provenance_maps(
            self.raw_text,
            normalized,
            include_trailing_deleted=self.input_final,
        )
        return normalized[len(old_normalized) :], len(old_normalized)

    def finish(self) -> None:
        self.input_final = True
        self.normalized_to_raw, self.raw_to_normalized = _provenance_maps(
            self.raw_text,
            self.normalized_text,
            include_trailing_deleted=True,
        )

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
        self.raw_to_normalized = _inverse_boundaries(
            self.raw_text,
            self.normalized_to_raw,
        )
        return self.normalized_text

    def raw_span(self, normalized_start: int, normalized_end: int) -> tuple[int, int]:
        start = max(0, min(int(normalized_start), len(self.normalized_text)))
        end = max(start, min(int(normalized_end), len(self.normalized_text)))
        return self.normalized_to_raw[start], self.normalized_to_raw[end]


def _provenance_maps(
    raw: str,
    normalized: str,
    *,
    include_trailing_deleted: bool,
) -> tuple[list[int], list[int]]:
    """Build both boundary maps using the frontend's normalization contract.

    This is deliberately a forward provenance walk rather than a best-effort
    sequence diff.  A normalized character consumes the next matching raw
    character, or one raw character when normalization changed its spelling.
    Deleted characters are consumed immediately before the next speakable
    character, so they fold to that character's left boundary.  A non-final
    journal keeps a trailing deletion pending; this is what makes ``"hello"``
    + a later emoji packet equivalent to one packet containing both.

    Whitespace is special because the normalizer folds a run into one space.
    The normalized space covers the whole raw run, which also makes trimming
    return the raw boundary after leading whitespace.
    """

    normalized_to_raw = [0]
    raw_cursor = 0
    for char in normalized:
        if char.isspace():
            # Only consume a raw whitespace run when it starts at the current
            # cursor.  A space inserted around a removed emoji is zero-width
            # in raw coordinates; searching farther ahead would skip real
            # speakable text before the next raw space.
            match = (
                raw_cursor
                if raw_cursor < len(raw) and raw[raw_cursor].isspace()
                else None
            )
            if match is None:
                normalized_to_raw.append(raw_cursor)
                continue
            run_end = match
            while run_end < len(raw) and raw[run_end].isspace():
                run_end += 1
            normalized_to_raw[-1] = max(normalized_to_raw[-1], match)
            normalized_to_raw.append(run_end)
            raw_cursor = run_end
            continue

        match = raw.find(char, raw_cursor)
        if match >= 0:
            # Any removed/rewritten raw characters before this character are
            # folded to the next speakable boundary.
            normalized_to_raw[-1] = max(normalized_to_raw[-1], match)
            raw_cursor = match + 1
            normalized_to_raw.append(raw_cursor)
        else:
            # Formatting removal or character translation (for example a
            # full-width space) has no literal match. Consume one raw code
            # point and attribute the normalized character to that span.
            start = raw_cursor
            raw_cursor = min(len(raw), raw_cursor + 1)
            normalized_to_raw[-1] = max(normalized_to_raw[-1], start)
            normalized_to_raw.append(raw_cursor)

    if include_trailing_deleted:
        normalized_to_raw[-1] = len(raw)
    else:
        normalized_to_raw[-1] = max(normalized_to_raw[-1], raw_cursor)

    normalized_to_raw = _monotonic(normalized_to_raw)
    return normalized_to_raw, _inverse_boundaries(raw, normalized_to_raw)


def _monotonic(values: list[int]) -> list[int]:
    out: list[int] = []
    last = 0
    for value in values:
        last = max(last, int(value))
        out.append(last)
    return out


def _inverse_boundaries(raw: str, normalized_to_raw: list[int]) -> list[int]:
    """Invert normalized→raw boundaries with deleted input folding forward."""

    raw_to_normalized: list[int] = []
    norm_boundary = 0
    for raw_boundary in range(len(raw) + 1):
        while (
            norm_boundary < len(normalized_to_raw) - 1
            and normalized_to_raw[norm_boundary] < raw_boundary
        ):
            norm_boundary += 1
        raw_to_normalized.append(norm_boundary)
    return raw_to_normalized


__all__ = ("CanonicalTextJournal",)
