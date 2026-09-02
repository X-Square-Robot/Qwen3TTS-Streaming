"""Blind-review manifest construction and adjudicated truth aggregation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .models import ArmKind, ReviewLabel, RunStatus


SEVERE_LABELS = frozenset(
    {
        ReviewLabel.SINGLE_UNIT_LOOP,
        ReviewLabel.ABNORMAL_NOISE,
        ReviewLabel.UNSUPPORTED_SPEECH,
    }
)


def derive_severe_hallucination(
    label: ReviewLabel | str,
    *,
    repetition_count: int | None = None,
    duration_s: float | None = None,
    confirmed_unsupported_speech: bool | None = None,
) -> bool | None:
    """Derive the locked severe numerator from a final blind-review label.

    Review labels are normally assigned only after applying the documented
    thresholds.  When acoustic evidence is supplied here, the same thresholds
    are enforced: loop repetition >=3 or duration >=0.5 s, noise >=0.5 s, and
    explicitly confirmed unsupported speech.  ``UNSCORABLE`` remains missing.
    """

    selected = ReviewLabel(label)
    if repetition_count is not None and (
        not isinstance(repetition_count, int)
        or isinstance(repetition_count, bool)
        or repetition_count < 0
    ):
        raise ValueError("repetition_count must be a non-negative integer")
    if duration_s is not None and (
        not isinstance(duration_s, (int, float))
        or isinstance(duration_s, bool)
        or not math.isfinite(float(duration_s))
        or duration_s < 0
    ):
        raise ValueError("duration_s must be a finite non-negative number")
    if confirmed_unsupported_speech not in (True, False, None):
        raise TypeError("confirmed_unsupported_speech must be bool or None")

    if selected is ReviewLabel.UNSCORABLE:
        return None
    if selected is ReviewLabel.SINGLE_UNIT_LOOP:
        if repetition_count is None and duration_s is None:
            return True
        return bool(
            (repetition_count is not None and repetition_count >= 3)
            or (duration_s is not None and duration_s >= 0.5)
        )
    if selected is ReviewLabel.ABNORMAL_NOISE:
        return True if duration_s is None else duration_s >= 0.5
    if selected is ReviewLabel.UNSUPPORTED_SPEECH:
        return (
            True
            if confirmed_unsupported_speech is None
            else confirmed_unsupported_speech
        )
    return False


def _field(record: Mapping[str, Any] | Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _safe_source_path(record: Mapping[str, Any] | Any, *names: str) -> str | None:
    value = _field(record, *names)
    if value is None:
        return None
    return str(value)


def assert_blinded_manifest(
    manifest: Sequence[Mapping[str, Any]],
    *,
    sensitive_values: Sequence[str] = (),
) -> None:
    """Reject accidental arm/session/seed/source disclosure in public items."""

    forbidden_key_fragments = ("arm", "session", "seed", "source", "run_id")
    normalized_sensitive = [value for value in sensitive_values if value]
    for item in manifest:
        for key, value in item.items():
            normalized_key = str(key).casefold()
            if any(fragment in normalized_key for fragment in forbidden_key_fragments):
                raise ValueError(f"blind manifest leaks private key {key!r}")
            if isinstance(value, str) and any(
                sensitive in value for sensitive in normalized_sensitive
            ):
                raise ValueError(f"blind manifest value for {key!r} leaks source identity")


def build_blind_manifest(
    records: Sequence[Mapping[str, Any] | Any],
    *,
    random_seed: int = 818,
    id_prefix: str = "LF",
    audio_directory: str = "audio",
    context_directory: str = "context",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Randomize observations into a public manifest and a separate private key.

    Public paths are newly generated blind package paths.  Original paths,
    arm/session identifiers, and seeds exist only in the returned private key.
    Callers must persist these two values to separate files/directories.
    """

    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise TypeError("random_seed must be an integer")
    if not id_prefix or any(character in id_prefix for character in "/\\"):
        raise ValueError("id_prefix must be a non-empty path-safe token")
    if not records:
        return [], {}

    shuffled = list(records)
    random.Random(random_seed).shuffle(shuffled)
    width = max(4, len(str(len(shuffled))))
    public_manifest: list[dict[str, Any]] = []
    private_key: dict[str, dict[str, Any]] = {}
    sensitive_values: list[str] = []

    for index, record in enumerate(shuffled, start=1):
        blind_id = f"{id_prefix}-{index:0{width}d}"
        reference_text = _field(
            record, "reference_text", "reference_sentence", "text"
        )
        if not isinstance(reference_text, str) or not reference_text:
            raise ValueError("every blind record requires non-empty reference text")
        source_audio = _safe_source_path(
            record, "sentence_audio_path", "audio_path", "clip", "wav_path"
        )
        if not source_audio:
            raise ValueError("every blind record requires a sentence audio path")
        source_context = _safe_source_path(
            record, "context_audio_path", "context_clip", "fallback_audio_path"
        )
        ordinal = _field(record, "sentence_ordinal", "ordinal", "sentence_index")
        if ordinal is None:
            raise ValueError("every blind record requires a sentence ordinal")
        occurrence = _field(record, "occurrence", default=1)

        public_item: dict[str, Any] = {
            "blind_id": blind_id,
            "sentence_ordinal": ordinal,
            "occurrence": occurrence,
            "reference_text": reference_text,
            "audio_path": str(PurePosixPath(audio_directory) / f"{blind_id}.wav"),
            "context_audio_path": (
                str(PurePosixPath(context_directory) / f"{blind_id}.wav")
                if source_context
                else None
            ),
            "allowed_labels": [label.value for label in ReviewLabel],
        }
        context_text = _field(record, "context_text")
        if context_text is not None:
            public_item["context_text"] = str(context_text)
        public_manifest.append(public_item)

        raw_arm = _field(record, "arm")
        if raw_arm is None:
            raise ValueError("every blind record requires an arm for the private key")
        try:
            arm = ArmKind(raw_arm).value
        except ValueError:
            arm = str(raw_arm)
        session_id = _field(record, "session_id")
        seed = _field(record, "seed")
        sentence_id = _field(record, "sentence_id")
        private_key[blind_id] = {
            "arm": arm,
            "seed": seed,
            "session_id": session_id,
            "sentence_id": sentence_id,
            "sentence_ordinal": ordinal,
            "occurrence": occurrence,
            "source_audio_path": source_audio,
            "source_context_audio_path": source_context,
        }
        sensitive_values.extend(
            value
            for value in (
                arm,
                str(session_id) if session_id is not None else "",
                PurePosixPath(source_audio).name,
                PurePosixPath(source_context).name if source_context else "",
            )
            if value
        )

    assert_blinded_manifest(public_manifest, sensitive_values=sensitive_values)
    return public_manifest, private_key


def aggregate_reviews(
    reviews: Sequence[Mapping[str, Any] | Any],
    *,
    private_key: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Resolve reviewer agreement/adjudication into sentence-level truth.

    An adjudicator label takes precedence.  Without one, unanimous labels are
    resolved and disagreement stays missing with ``REVIEW_PENDING``.  Missing and
    ``UNSCORABLE`` outcomes are never converted into clean negatives.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_reviewer_item: set[tuple[str, str, bool]] = set()
    for review in reviews:
        blind_id = _field(review, "blind_id")
        reviewer_id = _field(review, "reviewer_id", "reviewer")
        raw_label = _field(review, "label", "review_label")
        if not isinstance(blind_id, str) or not blind_id:
            raise ValueError("each review requires a blind_id")
        if not isinstance(reviewer_id, str) or not reviewer_id:
            raise ValueError("each review requires a reviewer_id")
        if raw_label is None:
            raise ValueError("each review requires a label")
        label = ReviewLabel(raw_label)
        role = str(_field(review, "role", default="")).casefold()
        adjudication = bool(
            _field(review, "is_adjudication", "adjudicated", default=False)
            or role == "adjudicator"
        )
        unique_key = (blind_id, reviewer_id, adjudication)
        if unique_key in seen_reviewer_item:
            raise ValueError(
                f"duplicate review for blind_id={blind_id!r}, reviewer={reviewer_id!r}"
            )
        seen_reviewer_item.add(unique_key)
        grouped[blind_id].append(
            {
                "reviewer_id": reviewer_id,
                "label": label,
                "is_adjudication": adjudication,
            }
        )

    if private_key is not None:
        unknown = set(grouped) - set(private_key)
        if unknown:
            raise ValueError(f"reviews contain blind IDs absent from private key: {sorted(unknown)}")
        blind_ids = list(private_key)
    else:
        blind_ids = sorted(grouped)

    aggregated: list[dict[str, Any]] = []
    for blind_id in blind_ids:
        item_reviews = grouped.get(blind_id, [])
        adjudications = [
            review["label"] for review in item_reviews if review["is_adjudication"]
        ]
        ordinary_labels = [
            review["label"] for review in item_reviews if not review["is_adjudication"]
        ]
        selected_label: ReviewLabel | None
        preliminary_label: ReviewLabel | None = None
        resolution: str
        if adjudications:
            if len(set(adjudications)) != 1:
                selected_label = None
                resolution = "conflicting_adjudications"
            else:
                selected_label = adjudications[0]
                resolution = "adjudicated"
        elif ordinary_labels and len(set(ordinary_labels)) == 1:
            preliminary_label = ordinary_labels[0]
            if len(ordinary_labels) == 1 and preliminary_label is not ReviewLabel.OK:
                selected_label = None
                resolution = "needs_second_review"
            else:
                selected_label = preliminary_label
                resolution = (
                    "unanimous" if len(ordinary_labels) > 1 else "single_reviewer"
                )
        elif ordinary_labels:
            selected_label = None
            resolution = "needs_adjudication"
        else:
            selected_label = None
            resolution = "missing_review"

        severe = (
            derive_severe_hallucination(selected_label)
            if selected_label is not None
            else None
        )
        if selected_label is ReviewLabel.UNSCORABLE:
            status = RunStatus.INVALID
        elif selected_label is None:
            status = RunStatus.REVIEW_PENDING
        else:
            status = RunStatus.REVIEWED
        result: dict[str, Any] = {
            "blind_id": blind_id,
            "label": selected_label.value if selected_label is not None else None,
            "preliminary_label": (
                preliminary_label.value if preliminary_label is not None else None
            ),
            "severe_hallucination": severe,
            "valid": isinstance(severe, bool),
            "status": status.value,
            "resolution": resolution,
            "requires_second_review": resolution == "needs_second_review",
            "requires_adjudication": resolution in {
                "needs_adjudication",
                "conflicting_adjudications",
            },
            "review_count": len(ordinary_labels),
            "adjudication_count": len(adjudications),
            "reviewer_labels": [
                {
                    "reviewer_id": review["reviewer_id"],
                    "label": review["label"].value,
                    "is_adjudication": review["is_adjudication"],
                }
                for review in item_reviews
            ],
        }
        if private_key is not None:
            result.update(dict(private_key[blind_id]))
        aggregated.append(result)
    return aggregated


def severe_label(label: ReviewLabel | str) -> bool | None:
    """Compatibility shorthand for label-only severe truth derivation."""

    return derive_severe_hallucination(label)


__all__ = [
    "SEVERE_LABELS",
    "aggregate_reviews",
    "assert_blinded_manifest",
    "build_blind_manifest",
    "derive_severe_hallucination",
    "severe_label",
]
