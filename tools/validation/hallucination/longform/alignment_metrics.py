"""Character alignment and repetition diagnostics for long-form transcripts."""

from __future__ import annotations

from typing import Any

from .text_normalization import canonicalize_numbers, normalize_transcript

try:  # RapidFuzz is installed in the dedicated evaluation environment.
    from rapidfuzz.distance import Levenshtein as _RapidFuzzLevenshtein
except ImportError:  # Keep artifact inspection importable in the base environment.
    _RapidFuzzLevenshtein = None


class AlignmentOpcode(tuple):
    """RapidFuzz-order tuple with mapping-style field access for JSON tooling."""

    __slots__ = ()
    _fields = ("tag", "src_start", "src_end", "dest_start", "dest_end")
    _field_indexes = {name: index for index, name in enumerate(_fields)}

    def __new__(
        cls,
        tag: str,
        src_start: int,
        src_end: int,
        dest_start: int,
        dest_end: int,
    ) -> AlignmentOpcode:
        return tuple.__new__(
            cls, (tag, src_start, src_end, dest_start, dest_end)
        )

    def __getitem__(self, index: int | slice | str) -> Any:
        if isinstance(index, str):
            try:
                index = self._field_indexes[index]
            except KeyError as exc:
                raise KeyError(index) from exc
        return tuple.__getitem__(self, index)

    @property
    def tag(self) -> str:
        return tuple.__getitem__(self, 0)

    @property
    def src_start(self) -> int:
        return tuple.__getitem__(self, 1)

    @property
    def src_end(self) -> int:
        return tuple.__getitem__(self, 2)

    @property
    def dest_start(self) -> int:
        return tuple.__getitem__(self, 3)

    @property
    def dest_end(self) -> int:
        return tuple.__getitem__(self, 4)

    def _asdict(self) -> dict[str, int | str]:
        return dict(zip(self._fields, self, strict=True))

    def __repr__(self) -> str:
        values = ", ".join(
            f"{name}={value!r}"
            for name, value in zip(self._fields, self, strict=True)
        )
        return f"AlignmentOpcode({values})"


def _fallback_opcodes(source: str, destination: str) -> list[AlignmentOpcode]:
    """Deterministic Levenshtein opcodes for environments without RapidFuzz."""

    rows, columns = len(source), len(destination)
    distance = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        distance[row][0] = row
    for column in range(columns + 1):
        distance[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            substitution = distance[row - 1][column - 1] + int(
                source[row - 1] != destination[column - 1]
            )
            distance[row][column] = min(
                substitution,
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
            )

    atomic: list[tuple[str, int, int, int, int]] = []
    row, column = rows, columns
    while row or column:
        if (
            row
            and column
            and source[row - 1] == destination[column - 1]
            and distance[row][column] == distance[row - 1][column - 1]
        ):
            atomic.append(("equal", row - 1, row, column - 1, column))
            row -= 1
            column -= 1
        elif (
            row
            and column
            and distance[row][column] == distance[row - 1][column - 1] + 1
        ):
            atomic.append(("replace", row - 1, row, column - 1, column))
            row -= 1
            column -= 1
        elif row and distance[row][column] == distance[row - 1][column] + 1:
            atomic.append(("delete", row - 1, row, column, column))
            row -= 1
        else:
            atomic.append(("insert", row, row, column - 1, column))
            column -= 1

    atomic.reverse()
    merged: list[list[int | str]] = []
    for tag, src_start, src_end, dst_start, dst_end in atomic:
        if (
            merged
            and merged[-1][0] == tag
            and merged[-1][2] == src_start
            and merged[-1][4] == dst_start
        ):
            merged[-1][2] = src_end
            merged[-1][4] = dst_end
        else:
            merged.append([tag, src_start, src_end, dst_start, dst_end])
    return [
        AlignmentOpcode(
            str(tag),
            int(src_start),
            int(src_end),
            int(dst_start),
            int(dst_end),
        )
        for tag, src_start, src_end, dst_start, dst_end in merged
    ]


def levenshtein_opcodes(reference: str, hypothesis: str) -> list[AlignmentOpcode]:
    """Return RapidFuzz-compatible half-open character alignment opcodes."""

    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    if _RapidFuzzLevenshtein is None:
        return _fallback_opcodes(reference, hypothesis)

    result: list[AlignmentOpcode] = []
    for opcode in _RapidFuzzLevenshtein.opcodes(reference, hypothesis):
        if hasattr(opcode, "tag"):
            values = (
                opcode.tag,
                opcode.src_start,
                opcode.src_end,
                opcode.dest_start,
                opcode.dest_end,
            )
        else:
            values = tuple(opcode)
        tag, src_start, src_end, dst_start, dst_end = values
        result.append(
            AlignmentOpcode(
                str(tag),
                int(src_start),
                int(src_end),
                int(dst_start),
                int(dst_end),
            )
        )
    return result


def repetition_spans(
    text: str,
    *,
    min_repetitions: int = 2,
    max_unit_length: int | None = None,
) -> list[dict[str, Any]]:
    """Find maximal contiguous tandem repeats, including multi-character units."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if min_repetitions < 2:
        raise ValueError("min_repetitions must be at least two")
    if max_unit_length is not None and max_unit_length < 1:
        raise ValueError("max_unit_length must be positive")

    candidates: list[dict[str, Any]] = []
    length = len(text)
    # A hallucinated speech unit is short in practice. Bounding the implicit
    # default prevents quadratic candidate explosion on a multi-minute one-unit
    # loop while still detecting repeated words and phrases.
    effective_max_unit_length = max_unit_length if max_unit_length is not None else 32
    for start in range(length):
        best_at_start: dict[str, Any] | None = None
        maximum = (length - start) // min_repetitions
        maximum = min(maximum, effective_max_unit_length)
        for unit_length in range(1, maximum + 1):
            unit = text[start : start + unit_length]
            repetitions = 1
            cursor = start + unit_length
            while text[cursor : cursor + unit_length] == unit:
                repetitions += 1
                cursor += unit_length
            if repetitions >= min_repetitions:
                candidate = {
                    "start": start,
                    "end": cursor,
                    "length": cursor - start,
                    "unit": unit,
                    "unit_length": unit_length,
                    "repetitions": repetitions,
                    "text": text[start:cursor],
                }
                if best_at_start is None or (
                    int(candidate["length"]),
                    int(candidate["repetitions"]),
                    -int(candidate["unit_length"]),
                ) > (
                    int(best_at_start["length"]),
                    int(best_at_start["repetitions"]),
                    -int(best_at_start["unit_length"]),
                ):
                    best_at_start = candidate
        if best_at_start is not None:
            candidates.append(best_at_start)

    # Prefer the longest/highest-repeat representation and remove contained spans.
    candidates.sort(
        key=lambda item: (
            -int(item["length"]),
            -int(item["repetitions"]),
            int(item["unit_length"]),
            int(item["start"]),
        )
    )
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            int(existing["start"]) <= int(candidate["start"])
            and int(existing["end"]) >= int(candidate["end"])
            for existing in selected
        ):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return selected


def character_error_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    """Return canonicalized CER, S/D/I, opcodes, insertion and repeat spans."""

    normalized_reference = normalize_transcript(reference)
    normalized_hypothesis = normalize_transcript(hypothesis)
    if not normalized_reference:
        raise ValueError("reference must contain at least one alphanumeric character")

    opcodes = levenshtein_opcodes(normalized_reference, normalized_hypothesis)
    substitutions = deletions = insertions = 0
    insertion_spans: list[dict[str, Any]] = []
    for opcode in opcodes:
        tag = opcode.tag
        src_start = opcode.src_start
        src_end = opcode.src_end
        dst_start = opcode.dest_start
        dst_end = opcode.dest_end
        source_length = src_end - src_start
        destination_length = dst_end - dst_start
        if tag == "replace":
            shared = min(source_length, destination_length)
            substitutions += shared
            deletions += source_length - shared
            insertions += destination_length - shared
        elif tag == "delete":
            deletions += source_length
        elif tag == "insert":
            insertions += destination_length
            inserted = normalized_hypothesis[dst_start:dst_end]
            insertion_spans.append(
                {
                    "hypothesis_start": dst_start,
                    "hypothesis_end": dst_end,
                    "length": destination_length,
                    "text": inserted,
                    "repetition_spans": repetition_spans(inserted),
                }
            )
        elif tag != "equal":
            raise ValueError(f"unsupported opcode tag: {tag}")

    distance = substitutions + deletions + insertions
    repeats = repetition_spans(normalized_hypothesis)
    return {
        "reference_raw": reference,
        "hypothesis_raw": hypothesis,
        "reference_canonical": canonicalize_numbers(reference),
        "hypothesis_canonical": canonicalize_numbers(hypothesis),
        "reference_normalized": normalized_reference,
        "hypothesis_normalized": normalized_hypothesis,
        "reference_characters": len(normalized_reference),
        "hypothesis_characters": len(normalized_hypothesis),
        "distance": distance,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "cer": distance / len(normalized_reference),
        "opcodes": [opcode._asdict() for opcode in opcodes],
        "opcode_tuples": [tuple(opcode) for opcode in opcodes],
        "alignment_backend": (
            "rapidfuzz" if _RapidFuzzLevenshtein else "python_fallback"
        ),
        "insertion_spans": insertion_spans,
        "longest_insertion_span": max(
            insertion_spans, key=lambda span: int(span["length"]), default=None
        ),
        "repetition_spans": repeats,
        "longest_repetition_span": max(
            repeats, key=lambda span: int(span["length"]), default=None
        ),
    }


__all__ = [
    "AlignmentOpcode",
    "character_error_metrics",
    "levenshtein_opcodes",
    "repetition_spans",
]
