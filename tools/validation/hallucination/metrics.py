"""Transcript error metrics and automatic suspect classification."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import NormalDist
from typing import Any

from .models import (
    SCREENING_LABEL,
    AsrStatus,
    SuspectReason,
    SuspectThresholds,
    TrialStatus,
)


def normalize_transcript(text: str) -> str:
    """Keep Unicode alphanumerics and lowercase Latin characters."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return "".join(character.lower() for character in text if character.isalnum())


def character_error_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    """Return deterministic character-level CER and S/D/I accounting."""

    normalized_reference = normalize_transcript(reference)
    normalized_hypothesis = normalize_transcript(hypothesis)
    if not normalized_reference:
        raise ValueError("reference must contain at least one alphanumeric character")

    ref = list(normalized_reference)
    hyp = list(normalized_hypothesis)
    distance = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for row in range(len(ref) + 1):
        distance[row][0] = row
    for column in range(len(hyp) + 1):
        distance[0][column] = column
    for row in range(1, len(ref) + 1):
        for column in range(1, len(hyp) + 1):
            substitution_cost = int(ref[row - 1] != hyp[column - 1])
            distance[row][column] = min(
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
                distance[row - 1][column - 1] + substitution_cost,
            )

    substitutions = deletions = insertions = 0
    row, column = len(ref), len(hyp)
    while row > 0 or column > 0:
        if (
            row > 0
            and column > 0
            and ref[row - 1] == hyp[column - 1]
            and distance[row][column] == distance[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
        elif (
            row > 0
            and column > 0
            and distance[row][column] == distance[row - 1][column - 1] + 1
        ):
            substitutions += 1
            row -= 1
            column -= 1
        elif row > 0 and distance[row][column] == distance[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1

    edits = distance[-1][-1]
    if edits != substitutions + deletions + insertions:
        raise AssertionError("edit accounting does not equal Levenshtein distance")
    return {
        "reference_normalized": normalized_reference,
        "hypothesis_normalized": normalized_hypothesis,
        "reference_characters": len(ref),
        "hypothesis_characters": len(hyp),
        "distance": edits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "cer": edits / len(ref),
    }


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Return a two-sided Wilson score interval for a binomial rate."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if total == 0:
        return {
            "successes": successes,
            "total": total,
            "rate": None,
            "confidence": confidence,
            "low": None,
            "high": None,
        }
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "rate": proportion,
        "confidence": confidence,
        "low": max(0.0, centre - margin),
        "high": min(1.0, centre + margin),
    }


def classify_trial(
    record: Mapping[str, Any],
    *,
    thresholds: SuspectThresholds,
    asr_requested: bool,
) -> dict[str, Any]:
    """Classify automatic evidence without asserting human hallucination truth."""

    # Retain the keyword for API compatibility and to make the caller's intent
    # explicit.  Classification itself depends on evidence presence: merely
    # disabling ASR must never turn an unscored sample into a negative result.
    del asr_requested

    if record.get("status") != TrialStatus.OK.value:
        return {
            "label": SCREENING_LABEL,
            "is_suspect": None,
            "reasons": [],
            "excluded_reason": "tts_not_ok",
        }

    reasons: list[str] = []
    if float(record.get("duration_s", 0.0)) >= thresholds.duration_s:
        reasons.append(SuspectReason.DURATION.value)
    asr = record.get("asr")
    if isinstance(asr, Mapping) and asr.get("status") == AsrStatus.OK.value:
        metrics = asr.get("metrics")
        if isinstance(metrics, Mapping):
            if float(metrics.get("cer", 0.0)) >= thresholds.cer:
                reasons.append(SuspectReason.CER.value)
            if int(metrics.get("insertions", 0)) >= thresholds.insertions:
                reasons.append(SuspectReason.INSERTIONS.value)
    elif not reasons:
        return {
            "label": SCREENING_LABEL,
            "is_suspect": None,
            "reasons": [],
            "excluded_reason": "asr_unavailable",
        }

    return {
        "label": SCREENING_LABEL,
        "is_suspect": bool(reasons),
        "reasons": reasons,
        "requires_blind_human_review": True,
    }
