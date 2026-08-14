"""Session-global raw/canonical text coordinates for progress anchors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..text_normalization import is_emoji_char, split_pending_emoji


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
    strip_leading_whitespace: bool = False

    def append(self, raw_delta: str) -> tuple[str, int]:
        old_normalized = self.normalized_text
        self.raw_text += raw_delta or ""
        # Keep a possible keycap base out of the committed canonical stream.
        # Without this small journal-level hold, appending ``"1"`` followed by
        # ``"\ufe0f\u20e3"`` would first publish ``"1"`` and then rewrite it
        # away.  The frontend has the same carry for streaming tokenization;
        # keeping it here also makes the public coordinate journal invariant
        # when it is fed directly by a transport or an oracle.
        committed_raw = self.raw_text
        if not self.input_final:
            committed_raw, _ = split_pending_emoji(committed_raw)
        normalized, normalized_to_raw, raw_to_normalized = self._project(
            committed_raw,
            include_trailing_deleted=self.input_final,
        )
        if not normalized.startswith(old_normalized):
            raise ValueError("canonical text normalization rewrote committed text")
        self.normalized_text = normalized
        self.normalized_to_raw = normalized_to_raw
        self.raw_to_normalized = raw_to_normalized
        return normalized[len(old_normalized) :], len(old_normalized)

    def finish(self) -> None:
        self.input_final = True
        # A non-final append may have held a keycap base.  Recompute the
        # canonical text now that no future packet can complete that sequence.
        (
            self.normalized_text,
            self.normalized_to_raw,
            self.raw_to_normalized,
        ) = self._project(
            self.raw_text,
            include_trailing_deleted=True,
        )

    def _project(
        self,
        committed_raw: str,
        *,
        include_trailing_deleted: bool,
    ) -> tuple[str, list[int], list[int]]:
        """Project accumulated raw input into canonical text and coordinates.

        Leading-whitespace filtering is session-global: it is evaluated over
        the accumulated input rather than independently for each transport
        packet.  Raw whitespace is removed before normalization so deleted
        formatting characters retain the correct raw origin.  A second trim
        after normalization catches whitespace exposed by removing a leading
        emoji.  Interior whitespace is never affected.
        """

        raw_prefix = 0
        if self.strip_leading_whitespace:
            raw_prefix = len(self.raw_text) - len(self.raw_text.lstrip())

        normalization_input = committed_raw[raw_prefix:]
        provenance_input = self.raw_text[raw_prefix:]
        untrimmed = str(self.normalize(normalization_input))
        normalized_prefix = 0
        if self.strip_leading_whitespace:
            normalized_prefix = len(untrimmed) - len(untrimmed.lstrip())
        normalized = untrimmed[normalized_prefix:]

        untrimmed_to_raw, _ = _provenance_maps(
            provenance_input,
            untrimmed,
            include_trailing_deleted=include_trailing_deleted,
        )
        normalized_to_raw = [
            raw_prefix + boundary for boundary in untrimmed_to_raw[normalized_prefix:]
        ]
        raw_to_normalized = _inverse_boundaries(
            self.raw_text,
            normalized_to_raw,
        )
        return normalized, normalized_to_raw, raw_to_normalized

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

    # Walk the normalization result and the raw stream together.  The
    # normalizer is context-sensitive (emoji can insert a separating space and
    # whitespace can collapse), so a plain ``str.find`` cannot distinguish a
    # repeated character before and after a deleted sequence.  Each deleted
    # run is carried to the next speakable boundary; a trailing run is kept
    # pending until ``finish()``.
    char_spans: list[tuple[int, int]] = []
    raw_cursor = 0
    pending_deleted_start: int | None = None
    for char in normalized:
        removed_end = _removed_sequence_end(raw, raw_cursor)
        if removed_end > raw_cursor:
            next_raw = raw[removed_end] if removed_end < len(raw) else ""
            previous_raw = raw[raw_cursor - 1] if raw_cursor > 0 else ""
            inserts_separator = (
                char == " "
                and _is_ascii_word(previous_raw)
                and _is_ascii_word(next_raw)
            )
            if not inserts_separator:
                if pending_deleted_start is None:
                    pending_deleted_start = raw_cursor
                raw_cursor = removed_end
                removed_end = _removed_sequence_end(raw, raw_cursor)
            else:
                start = (
                    pending_deleted_start
                    if pending_deleted_start is not None
                    else raw_cursor
                )
                raw_cursor = removed_end
                char_spans.append((start, raw_cursor))
                pending_deleted_start = None
                continue

        if raw_cursor < len(raw) and raw[raw_cursor] == char:
            start = (
                pending_deleted_start
                if pending_deleted_start is not None
                else raw_cursor
            )
            raw_cursor += 1
            char_spans.append((start, raw_cursor))
            pending_deleted_start = None
            continue

        match = raw.find(char, raw_cursor)
        if match >= 0:
            if pending_deleted_start is None and match > raw_cursor:
                pending_deleted_start = raw_cursor
            start = (
                pending_deleted_start
                if pending_deleted_start is not None
                else raw_cursor
            )
            raw_cursor = match + 1
            char_spans.append((start, raw_cursor))
            pending_deleted_start = None
            continue

        # A normalized character with no literal raw match is either an
        # inserted separator or a spelling/formatting rewrite.  Consume one
        # raw code point only for the latter; insertion stays zero-width.
        if raw_cursor < len(raw) and (
            char != " " or raw[raw_cursor].isspace() or is_emoji_char(raw[raw_cursor])
        ):
            start = (
                pending_deleted_start
                if pending_deleted_start is not None
                else raw_cursor
            )
            raw_cursor += 1
            char_spans.append((start, raw_cursor))
            pending_deleted_start = None
        else:
            point = (
                pending_deleted_start
                if pending_deleted_start is not None
                else raw_cursor
            )
            char_spans.append((point, point))
            pending_deleted_start = None

    if raw_cursor < len(raw):
        if pending_deleted_start is None:
            pending_deleted_start = raw_cursor
        raw_cursor = len(raw)

    # A folded whitespace run belongs to its normalized space, not to the
    # following word.  This also handles the deletion half of a collapsed run.
    for index, char in enumerate(normalized):
        if not char.isspace() or index >= len(char_spans):
            continue
        start, end = char_spans[index]
        if start < len(raw) and raw[start].isspace():
            while end < len(raw) and raw[end].isspace():
                end += 1
            char_spans[index] = (start, end)

    normalized_to_raw = [0]
    for start, end in char_spans:
        start = max(normalized_to_raw[-1], int(start))
        end = max(start, int(end))
        normalized_to_raw.append(end)

    if normalized:
        if include_trailing_deleted:
            normalized_to_raw[-1] = len(raw)
        elif pending_deleted_start is not None:
            normalized_to_raw[-1] = max(
                normalized_to_raw[-2], pending_deleted_start
            )
    elif include_trailing_deleted:
        normalized_to_raw[0] = len(raw)

    normalized_to_raw = _monotonic(normalized_to_raw)
    return normalized_to_raw, _inverse_boundaries(raw, normalized_to_raw)


def _is_ascii_word(value: str) -> bool:
    return bool(value) and value.isascii() and value.isalnum()


def _removed_sequence_end(raw: str, start: int) -> int:
    """Return the end of an emoji/keycap run beginning at ``start``."""
    if start >= len(raw):
        return start
    if raw[start] in "0123456789#*":
        if start + 1 < len(raw) and ord(raw[start + 1]) == 0x20E3:
            return start + 2
        if (
            start + 2 < len(raw)
            and ord(raw[start + 1]) == 0xFE0F
            and ord(raw[start + 2]) == 0x20E3
        ):
            return start + 3
    if not is_emoji_char(raw[start]):
        return start
    end = start + 1
    while end < len(raw):
        if raw[end] in "0123456789#*":
            keycap_end = _removed_sequence_end(raw, end)
            if keycap_end > end:
                end = keycap_end
                continue
        if not is_emoji_char(raw[end]):
            break
        end += 1
    return end


def _monotonic(values: list[int]) -> list[int]:
    out: list[int] = []
    last = 0
    for value in values:
        last = max(last, int(value))
        out.append(last)
    return out


def _inverse_boundaries(raw: str, normalized_to_raw: list[int]) -> list[int]:
    """Invert boundaries, folding deleted interiors to the next text edge.

    ``normalized_to_raw[i:i+2]`` describes the raw interval attributed to
    normalized character ``i``.  Raw boundaries inside that interval belong to
    its left (next speakable) normalized boundary; only the interval's right
    edge advances the normalized cursor.  This is different from a
    ``bisect_right`` inversion, which incorrectly moves an emoji/keycap
    deletion to the end of the following character.
    """

    from bisect import bisect_left

    raw_to_normalized: list[int] = []
    for raw_boundary in range(len(raw) + 1):
        index = bisect_left(normalized_to_raw, raw_boundary)
        if index < len(normalized_to_raw) and normalized_to_raw[index] == raw_boundary:
            raw_to_normalized.append(index)
        else:
            raw_to_normalized.append(max(0, index - 1))
    return raw_to_normalized


__all__ = ("CanonicalTextJournal",)
