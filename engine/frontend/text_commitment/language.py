"""Language hypotheses for the causal text commitment frontend.

The resolver is intentionally a small evidence collector, not a language
detector pretending to be a semantic authority.  Numbers and symbols carry no
language by themselves; they remain unresolved until an explicit hint or
nearby script evidence makes a route safe.  This keeps the decision separate
from the WeText verbalizer and avoids case-specific connector dictionaries.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import unicodedata

from .types import (
    LanguageEvidence,
    LanguageHypothesis,
    LanguageKind,
    SpanKind,
)


_SENTENCE_BREAKS = set("。！？!?;；\n\r")


def _is_han(ch: str) -> bool:
    name = unicodedata.name(ch, "")
    # Include the common CJK extension names while excluding arbitrary
    # non-ASCII punctuation, which was a source of false Chinese routes.
    return "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name


def _is_latin(ch: str) -> bool:
    return "LATIN" in unicodedata.name(ch, "") and ch.isalpha()


def script_evidence(text: str) -> tuple[LanguageEvidence, ...]:
    """Return distinct script evidence present in *text*."""

    has_han = any(_is_han(ch) for ch in text)
    has_latin = any(_is_latin(ch) for ch in text)
    evidence: list[LanguageEvidence] = []
    if has_han:
        evidence.append(LanguageEvidence.HAN_SCRIPT)
    if has_latin:
        evidence.append(LanguageEvidence.LATIN_SCRIPT)
    return tuple(evidence)


def _clause_context(context_before: str, context_after: str) -> tuple[str, str]:
    """Limit contextual evidence to the current sentence/clause.

    A numeric span must not inherit the dominant script of an earlier
    sentence.  Keeping only the suffix after the last sentence break and the
    prefix before the next one also bounds the cost for long sessions.
    """

    before = context_before
    last_break = max((before.rfind(marker) for marker in _SENTENCE_BREAKS), default=-1)
    if last_break >= 0:
        before = before[last_break + 1 :]
    after = context_after
    cut = len(after)
    for index, char in enumerate(after):
        if char in _SENTENCE_BREAKS:
            cut = index
            break
    return before[-128:], after[:cut][:128]


def _nearest_script(
    text: str,
    *,
    reverse: bool = False,
) -> tuple[LanguageKind, int] | None:
    """Find the nearest Han/Latin evidence and its code-point distance."""

    sequence = reversed(text) if reverse else iter(text)
    distance = 0
    for char in sequence:
        if _is_han(char):
            return LanguageKind.ZH, distance
        if _is_latin(char):
            return LanguageKind.EN, distance
        # Digits, whitespace, operators and delimiters are neutral.  Other
        # scripts are intentionally skipped rather than guessed as Chinese or
        # English.
        distance += 1
    return None


def _local_script_resolution(
    context_before: str,
    context_after: str,
) -> tuple[LanguageKind, float] | None:
    """Resolve the closest local script, or return ``None`` on a tie."""

    before, after = _clause_context(context_before, context_after)
    candidates: list[tuple[LanguageKind, int]] = []
    previous = _nearest_script(before, reverse=True)
    following = _nearest_script(after)
    if previous is not None:
        candidates.append(previous)
    if following is not None:
        candidates.append(following)
    if not candidates:
        return None
    nearest_distance = min(distance for _, distance in candidates)
    nearest = [language for language, distance in candidates if distance == nearest_distance]
    if len(set(nearest)) != 1:
        # Equally close code-switch evidence is genuinely ambiguous; keep the
        # span unresolved so the caller can wait or apply its deterministic
        # final fallback instead of selecting a graph by global vote.
        return None
    confidence = 0.96 if nearest_distance <= 1 else 0.82
    return nearest[0], confidence


@dataclass(frozen=True)
class LanguageResolution:
    """A ranked language result for one open span."""

    hypotheses: tuple[LanguageHypothesis, ...]
    selected: LanguageKind = LanguageKind.UNKNOWN


class LanguageResolver:
    """Collect script/context evidence without making TN decisions.

    ``mode`` accepts ``zh``, ``en`` and ``mixed_zh_en``.  In mixed mode a
    numeric or symbol-only span is ``UNKNOWN`` unless the caller supplies
    explicit evidence.  A previous language is used only as weak context and
    is reset at sentence boundaries; it never overrides contradictory script
    evidence.
    """

    def __init__(self, mode: str = "mixed_zh_en") -> None:
        self.mode = str(mode or "mixed_zh_en").strip().lower()
        self._last_confirmed = LanguageKind.UNKNOWN
        self._scores: Counter[LanguageKind] = Counter()

    @property
    def last_confirmed(self) -> LanguageKind:
        return self._last_confirmed

    def reset(self) -> None:
        self._last_confirmed = LanguageKind.UNKNOWN
        self._scores.clear()

    def observe(self, text: str, language: LanguageKind | None = None) -> None:
        """Record language evidence from an already committed text unit."""

        if not text:
            return
        if any(ch in _SENTENCE_BREAKS for ch in text):
            # Keep evidence for the unit before the boundary, then prevent it
            # from leaking into a new sentence/code-switch island.
            self._scores.clear()
            self._last_confirmed = LanguageKind.UNKNOWN
        if language in (LanguageKind.ZH, LanguageKind.EN):
            self._last_confirmed = language
            self._scores[language] += 3
            return
        evidence = script_evidence(text)
        if LanguageEvidence.HAN_SCRIPT in evidence and LanguageEvidence.LATIN_SCRIPT not in evidence:
            self._last_confirmed = LanguageKind.ZH
            self._scores[LanguageKind.ZH] += 2
        elif LanguageEvidence.LATIN_SCRIPT in evidence and LanguageEvidence.HAN_SCRIPT not in evidence:
            self._last_confirmed = LanguageKind.EN
            self._scores[LanguageKind.EN] += 2

    def _configured(self) -> LanguageKind | None:
        if self.mode.startswith("zh"):
            return LanguageKind.ZH
        if self.mode.startswith("en"):
            return LanguageKind.EN
        return None

    @staticmethod
    def _context_scores(context: str) -> Counter[LanguageKind]:
        scores: Counter[LanguageKind] = Counter()
        for ch in context:
            if _is_han(ch):
                scores[LanguageKind.ZH] += 1
            elif _is_latin(ch):
                scores[LanguageKind.EN] += 1
        return scores

    def resolve(
        self,
        raw: str,
        kind: SpanKind,
        *,
        hint: LanguageKind | None = None,
        context_before: str = "",
        context_after: str = "",
    ) -> LanguageResolution:
        configured = self._configured()
        if configured is not None:
            return LanguageResolution(
                (LanguageHypothesis(configured, 1.0, (LanguageEvidence.SESSION,)),),
                configured,
            )
        if hint in (LanguageKind.ZH, LanguageKind.EN):
            return LanguageResolution(
                (LanguageHypothesis(hint, 1.0, (LanguageEvidence.EXPLICIT,)),),
                hint,
            )

        evidence = script_evidence(raw)
        has_han = LanguageEvidence.HAN_SCRIPT in evidence
        has_latin = LanguageEvidence.LATIN_SCRIPT in evidence
        # A complete alphabetic/URL/ordinal island is strong English evidence.
        # Identifiers are kept slightly less certain because they can be
        # product names in a Chinese sentence.
        if has_han and not has_latin:
            return LanguageResolution(
                (LanguageHypothesis(LanguageKind.ZH, 0.98, (LanguageEvidence.HAN_SCRIPT,)),),
                LanguageKind.ZH,
            )
        # Product identifiers and versions often contain Latin characters but
        # are embedded in Chinese prose (``模型版本号: researcher@...``).  Their
        # internal alphabet is not enough to select the English graph; use the
        # surrounding script first.
        local = _local_script_resolution(context_before, context_after)

        if kind in (SpanKind.IDENTIFIER, SpanKind.VERSION) or (
            kind is SpanKind.ENGLISH_WORD and any(char.isdigit() for char in raw)
        ):
            if local is not None:
                language, confidence = local
                return LanguageResolution(
                    (
                        LanguageHypothesis(
                            language,
                            confidence,
                            (LanguageEvidence.CONTEXT,),
                        ),
                    ),
                    language,
                )

        if has_latin and not has_han and kind not in (SpanKind.NUMBER, SpanKind.MATH):
            confidence = 0.90 if kind in (SpanKind.ENGLISH_WORD, SpanKind.ORDINAL, SpanKind.URL, SpanKind.EMAIL) else 0.75
            return LanguageResolution(
                (LanguageHypothesis(LanguageKind.EN, confidence, (LanguageEvidence.LATIN_SCRIPT,)),),
                LanguageKind.EN,
            )

        # Numeric/formula spans inherit only local script evidence.  This is
        # deliberately bounded to the current raw island; arbitrary old
        # session text must not silently choose a graph.
        if local is not None:
            language, confidence = local
            return LanguageResolution(
                (
                    LanguageHypothesis(
                        language,
                        confidence,
                        (LanguageEvidence.CONTEXT,),
                    ),
                ),
                language,
            )

        # Previous evidence is useful only when it is unambiguous and the
        # current span has no competing script.  Keep it below the threshold
        # used for a hard selection by callers that require certainty.
        if self._last_confirmed in (LanguageKind.ZH, LanguageKind.EN):
            other = LanguageKind.EN if self._last_confirmed is LanguageKind.ZH else LanguageKind.ZH
            return LanguageResolution(
                (
                    LanguageHypothesis(self._last_confirmed, 0.60, (LanguageEvidence.CONTEXT,)),
                    LanguageHypothesis(other, 0.40, (LanguageEvidence.CONTEXT,)),
                ),
                LanguageKind.UNKNOWN,
            )

        return LanguageResolution(
            (
                LanguageHypothesis(LanguageKind.ZH, 0.5),
                LanguageHypothesis(LanguageKind.EN, 0.5),
            ),
            LanguageKind.UNKNOWN,
        )


__all__ = ["LanguageResolution", "LanguageResolver", "script_evidence"]
