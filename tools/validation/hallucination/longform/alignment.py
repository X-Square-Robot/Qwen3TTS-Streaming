"""Map occurrence-aware reference sentences onto ASR time coordinates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .metrics import levenshtein_opcodes, normalize_transcript
from .models import ReferenceSentence, parse_reference_sentences


@dataclass(frozen=True)
class TimedSentence:
    index: int
    reference_text: str
    raw_start: int
    raw_end: int
    start_ms: int
    end_ms: int
    clip_start_ms: int
    clip_end_ms: int
    alignment_quality: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_reference_sentences(text: str) -> list[ReferenceSentence]:
    """Compatibility wrapper around the single owner in ``models``."""

    if not isinstance(text, str) or not text:
        raise ValueError("reference text must be a non-empty string")
    return list(parse_reference_sentences(text))


def _segment_time_values(segment: Mapping[str, Any]) -> tuple[int, int]:
    start = int(segment.get("start_ms", 0) or 0)
    end = int(segment.get("end_ms", start) or start)
    return max(0, start), max(start, end)


def _hypothesis_timeline(
    segments: Sequence[Mapping[str, Any]],
) -> tuple[str, list[tuple[float, float]]]:
    text_parts: list[str] = []
    times: list[tuple[float, float]] = []
    for segment in segments:
        normalized = normalize_transcript(str(segment.get("text", "")))
        if not normalized:
            continue
        start, end = _segment_time_values(segment)
        width = max(1.0, float(end - start))
        for index in range(len(normalized)):
            char_start = start + width * index / len(normalized)
            char_end = start + width * (index + 1) / len(normalized)
            times.append((char_start, char_end))
        text_parts.append(normalized)
    return "".join(text_parts), times


def align_reference_sentences(
    reference_text: str,
    asr_segments: Sequence[Mapping[str, Any]],
    *,
    audio_duration_ms: int,
    margin_ms: int = 750,
) -> list[TimedSentence]:
    """Map one-based reference sentences to conservative audio intervals.

    ASR remains a localization aid.  If a sentence has no aligned characters,
    the interval falls back to its normalized character share of total audio;
    the downstream reviewer, not this function, supplies truth.
    """

    if audio_duration_ms <= 0:
        raise ValueError("audio_duration_ms must be positive")
    if margin_ms < 0:
        raise ValueError("margin_ms must be non-negative")
    sentences = split_reference_sentences(reference_text)
    canonical_parts = [normalize_transcript(sentence.text) for sentence in sentences]
    reference = "".join(canonical_parts)
    hypothesis, hyp_times = _hypothesis_timeline(asr_segments)
    opcodes = levenshtein_opcodes(reference, hypothesis)

    mapped: list[list[tuple[float, float]]] = [[] for _ in reference]
    for opcode in opcodes:
        tag = str(opcode["tag"])
        i1, i2 = int(opcode["src_start"]), int(opcode["src_end"])
        j1, j2 = int(opcode["dest_start"]), int(opcode["dest_end"])
        if tag not in {"equal", "replace"} or i2 <= i1 or j2 <= j1:
            continue
        ref_width, hyp_width = i2 - i1, j2 - j1
        for ref_offset in range(ref_width):
            fraction = (ref_offset + 0.5) / ref_width
            hyp_offset = min(hyp_width - 1, int(fraction * hyp_width))
            hyp_index = j1 + hyp_offset
            if 0 <= hyp_index < len(hyp_times):
                mapped[i1 + ref_offset].append(hyp_times[hyp_index])

    results: list[TimedSentence] = []
    canonical_cursor = 0
    for sentence, canonical in zip(sentences, canonical_parts, strict=True):
        begin = canonical_cursor
        end = begin + len(canonical)
        canonical_cursor = end
        evidence = [time for bucket in mapped[begin:end] for time in bucket]
        if evidence:
            start_ms = max(0, int(min(item[0] for item in evidence)))
            end_ms = min(audio_duration_ms, int(max(item[1] for item in evidence)))
            quality = "aligned"
        else:
            denominator = max(1, len(reference))
            start_ms = round(audio_duration_ms * begin / denominator)
            end_ms = round(audio_duration_ms * end / denominator)
            quality = "proportional_fallback"
        if end_ms <= start_ms:
            end_ms = min(audio_duration_ms, start_ms + 1)
        results.append(
            TimedSentence(
                index=sentence.ordinal,
                reference_text=sentence.text,
                raw_start=sentence.start,
                raw_end=sentence.end,
                start_ms=start_ms,
                end_ms=end_ms,
                clip_start_ms=max(0, start_ms - margin_ms),
                clip_end_ms=min(audio_duration_ms, end_ms + margin_ms),
                alignment_quality=quality,
            )
        )
    return results


def transcript_from_segments(segments: Iterable[Mapping[str, Any]]) -> str:
    return "".join(str(segment.get("text", "")) for segment in segments)


__all__ = [
    "TimedSentence",
    "align_reference_sentences",
    "split_reference_sentences",
    "transcript_from_segments",
]
