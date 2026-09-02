"""Sentence-rate intervals, paired bootstrap, and locked decision gates."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import NormalDist
from typing import Any

from .formal_statistics import (
    FORMAL_BOOTSTRAP_CONFIDENCE,
    FORMAL_BOOTSTRAP_ITERATIONS,
    FORMAL_BOOTSTRAP_SEED,
    FORMAL_RESAMPLING_SCHEME,
)
from .models import ArmKind, ReviewLabel, RunStatus, SentenceOutcome


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = FORMAL_BOOTSTRAP_CONFIDENCE,
) -> dict[str, float | int | None]:
    """Return a two-sided Wilson score interval for a binomial sentence rate."""

    if (
        not isinstance(successes, int)
        or isinstance(successes, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total < 0
        or successes < 0
        or successes > total
    ):
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
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "confidence": confidence,
        "low": max(0.0, centre - margin),
        "high": min(1.0, centre + margin),
    }


def _field(record: Mapping[str, Any] | Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _outcome_parts(
    record: Mapping[str, Any] | SentenceOutcome,
) -> tuple[str, int, str, bool | None]:
    arm_value = _field(record, "arm")
    if arm_value is None:
        raise ValueError("sentence observation is missing arm")
    try:
        arm = ArmKind(arm_value).value
    except ValueError:
        arm = str(arm_value)
    seed = _field(record, "seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("sentence observation seed must be an integer")
    sentence_key = _field(record, "sentence_id")
    if sentence_key is None:
        ordinal = _field(record, "sentence_ordinal", "ordinal", "sentence_index")
        if ordinal is None:
            raise ValueError("sentence observation is missing sentence identity")
        occurrence = _field(record, "occurrence", default=1)
        sentence_key = f"{ordinal}:{occurrence}"
    severe = _field(
        record,
        "severe_hallucination",
        "is_severe_hallucination",
        "is_severe",
        "severe",
    )
    label = _field(record, "review_label", "label")
    if severe is None and label is not None:
        selected_label = ReviewLabel(label)
        severe = selected_label in {
            ReviewLabel.SINGLE_UNIT_LOOP,
            ReviewLabel.ABNORMAL_NOISE,
            ReviewLabel.UNSUPPORTED_SPEECH,
        }
        if selected_label is ReviewLabel.UNSCORABLE:
            severe = None
    elif severe not in (True, False, None):
        raise TypeError("severe truth must be bool or None")
    valid_for_rate = _field(record, "valid_for_rate")
    if valid_for_rate is not None and not isinstance(valid_for_rate, bool):
        raise TypeError("valid_for_rate must be bool or None")
    status = _field(record, "status")
    if status is not None:
        try:
            terminal_status = RunStatus(status)
        except ValueError:
            terminal_status = None
        if terminal_status in {
            RunStatus.TTS_FAILED,
            RunStatus.INVALID,
            RunStatus.ERROR,
            RunStatus.PENDING,
            RunStatus.RUNNING,
            RunStatus.REVIEW_PENDING,
        }:
            severe = None
    # ASR is diagnostic-only.  A failed ASR pass must not erase an already
    # adjudicated boolean truth, even if an upstream diagnostic row carried a
    # stale ``valid_for_rate=false`` flag.  Rows without truth remain invalid.
    if valid_for_rate is False and not (
        status is not None
        and terminal_status is RunStatus.ASR_FAILED
        and isinstance(severe, bool)
    ):
        severe = None
    return arm, seed, str(sentence_key), severe


def _risk_ratio(numerator_rate: float, denominator_rate: float) -> float:
    """Return an internal extended-real ratio used only for percentile math."""

    if denominator_rate == 0:
        return 1.0 if numerator_rate == 0 else math.inf
    return numerator_rate / denominator_rate


def _serialized_ratio(value: float) -> tuple[float | None, bool]:
    """Encode an extended-real ratio without non-standard JSON numbers."""

    if math.isinf(value) and value > 0:
        return None, True
    if not math.isfinite(value) or value < 0:
        raise ValueError("risk ratios must be finite non-negative or +infinity")
    return value, False


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    if lower_index == upper_index or lower == upper:
        return lower
    fraction = position - lower_index
    if not math.isfinite(lower) or not math.isfinite(upper):
        return upper if fraction > 0 else lower
    return lower + (upper - lower) * fraction


def paired_bootstrap_comparison(
    observations: Sequence[Mapping[str, Any] | SentenceOutcome],
    baseline_arm: ArmKind | str,
    comparison_arm: ArmKind | str,
    *,
    iterations: int = FORMAL_BOOTSTRAP_ITERATIONS,
    confidence: float = FORMAL_BOOTSTRAP_CONFIDENCE,
    random_seed: int = FORMAL_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired two-way crossed bootstrap over seed and sentence position.

    Invalid/unreviewed outcomes are excluded pairwise, never imputed as clean.
    Risk difference and risk ratio are oriented as comparison minus/over baseline.
    """

    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < 1
    ):
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise TypeError("random_seed must be an integer")
    baseline = ArmKind(baseline_arm).value
    comparison = ArmKind(comparison_arm).value
    if baseline == comparison:
        raise ValueError("baseline and comparison arms must differ")

    indexed: dict[tuple[int, str, str], bool | None] = {}
    for observation in observations:
        arm, seed, sentence_key, severe = _outcome_parts(observation)
        if arm not in {baseline, comparison}:
            continue
        key = (seed, sentence_key, arm)
        if key in indexed:
            raise ValueError(f"duplicate sentence observation for {key!r}")
        indexed[key] = severe

    all_pair_keys = sorted(
        {(seed, sentence) for seed, sentence, _arm in indexed},
        key=lambda key: (key[0], key[1]),
    )
    paired_cells: dict[tuple[int, str], tuple[bool, bool]] = {}
    paired_by_seed: dict[int, list[tuple[bool, bool]]] = defaultdict(list)
    excluded_pairs = 0
    missing_pairs = 0
    for seed, sentence in all_pair_keys:
        baseline_key = (seed, sentence, baseline)
        comparison_key = (seed, sentence, comparison)
        if baseline_key not in indexed or comparison_key not in indexed:
            missing_pairs += 1
            continue
        baseline_value = indexed[baseline_key]
        comparison_value = indexed[comparison_key]
        if not isinstance(baseline_value, bool) or not isinstance(
            comparison_value, bool
        ):
            excluded_pairs += 1
            continue
        pair = (baseline_value, comparison_value)
        paired_cells[(seed, sentence)] = pair
        paired_by_seed[seed].append(pair)

    paired_by_seed = {seed: pairs for seed, pairs in paired_by_seed.items() if pairs}
    if not paired_by_seed:
        raise ValueError("no pairwise-valid seed/sentence observations")
    seeds = sorted(paired_by_seed)
    sentence_positions = sorted({sentence for _seed, sentence in paired_cells})
    point_pairs = [pair for seed in seeds for pair in paired_by_seed[seed]]
    baseline_rate = sum(int(pair[0]) for pair in point_pairs) / len(point_pairs)
    comparison_rate = sum(int(pair[1]) for pair in point_pairs) / len(point_pairs)

    rng = random.Random(random_seed)
    bootstrap_differences: list[float] = []
    bootstrap_ratios: list[float] = []
    while len(bootstrap_differences) < iterations:
        selected_seeds = [rng.choice(seeds) for _ in seeds]
        selected_sentences = [
            rng.choice(sentence_positions) for _ in sentence_positions
        ]
        baseline_positive = comparison_positive = sample_total = 0
        for selected_seed in selected_seeds:
            for selected_sentence in selected_sentences:
                pair = paired_cells.get((selected_seed, selected_sentence))
                if pair is None:
                    continue
                baseline_value, comparison_value = pair
                baseline_positive += int(baseline_value)
                comparison_positive += int(comparison_value)
                sample_total += 1
        # Pairwise invalid cells are never imputed.  With sparse exploratory
        # inputs a crossed draw can contain no valid cells, so redraw it rather
        # than manufacturing a clean observation or a zero denominator.
        if sample_total == 0:
            continue
        sampled_baseline_rate = baseline_positive / sample_total
        sampled_comparison_rate = comparison_positive / sample_total
        bootstrap_differences.append(sampled_comparison_rate - sampled_baseline_rate)
        bootstrap_ratios.append(
            _risk_ratio(sampled_comparison_rate, sampled_baseline_rate)
        )

    alpha = (1 - confidence) / 2
    seed_differences: dict[str, float] = {}
    for seed in seeds:
        seed_pairs = paired_by_seed[seed]
        seed_baseline = sum(int(pair[0]) for pair in seed_pairs) / len(seed_pairs)
        seed_comparison = sum(int(pair[1]) for pair in seed_pairs) / len(seed_pairs)
        seed_differences[str(seed)] = seed_comparison - seed_baseline

    point_ratio, point_ratio_infinite = _serialized_ratio(
        _risk_ratio(comparison_rate, baseline_rate)
    )
    ratio_ci_extended = [
        _percentile(bootstrap_ratios, alpha),
        _percentile(bootstrap_ratios, 1 - alpha),
    ]
    ratio_ci_parts = [_serialized_ratio(value) for value in ratio_ci_extended]

    return {
        "baseline_arm": baseline,
        "comparison_arm": comparison,
        "matched_pairs": len(point_pairs),
        "excluded_invalid_pairs": excluded_pairs,
        "missing_unpaired_observations": missing_pairs,
        "seeds": seeds,
        "seed_count": len(seeds),
        "sentence_position_count": len(sentence_positions),
        "baseline_rate": baseline_rate,
        "comparison_rate": comparison_rate,
        "baseline_wilson": wilson_interval(
            sum(int(pair[0]) for pair in point_pairs),
            len(point_pairs),
            confidence=confidence,
        ),
        "comparison_wilson": wilson_interval(
            sum(int(pair[1]) for pair in point_pairs),
            len(point_pairs),
            confidence=confidence,
        ),
        "risk_difference": comparison_rate - baseline_rate,
        "risk_difference_ci": [
            _percentile(bootstrap_differences, alpha),
            _percentile(bootstrap_differences, 1 - alpha),
        ],
        "risk_ratio": point_ratio,
        "risk_ratio_infinite": point_ratio_infinite,
        "risk_ratio_ci": [value for value, _infinite in ratio_ci_parts],
        "risk_ratio_ci_infinite": [infinite for _value, infinite in ratio_ci_parts],
        "seed_differences": seed_differences,
        "iterations": iterations,
        "confidence": confidence,
        "random_seed": random_seed,
        "resampling_scheme": FORMAL_RESAMPLING_SCHEME,
    }


def evaluate_invalidity(
    arm_counts: Mapping[ArmKind | str, Mapping[str, Any] | Sequence[Any]],
    *,
    max_invalid_rate: float = 0.05,
    max_between_arm_gap: float = 0.02,
) -> dict[str, Any]:
    """Apply the predeclared >5% invalid and >2pp between-arm rules."""

    if not 0 <= max_invalid_rate <= 1 or not 0 <= max_between_arm_gap <= 1:
        raise ValueError("invalid-rate thresholds must be between zero and one")
    arm_rates: dict[str, dict[str, int | float | None]] = {}
    reasons: list[str] = []
    if not arm_counts:
        reasons.append("no_arms")
    for raw_arm, raw_counts in arm_counts.items():
        try:
            arm = ArmKind(raw_arm).value
        except ValueError:
            arm = str(raw_arm)
        if isinstance(raw_counts, Mapping):
            total = raw_counts.get("total", raw_counts.get("expected"))
            invalid = raw_counts.get("invalid")
            if invalid is None:
                tts_failed = raw_counts.get(
                    "tts_failed", raw_counts.get("tts_failed_or_no_audio", 0)
                )
                invalid = (
                    int(tts_failed or 0)
                    + int(raw_counts.get("unscorable", 0) or 0)
                    + int(raw_counts.get("uncertain", 0) or 0)
                )
        elif isinstance(raw_counts, Sequence) and not isinstance(
            raw_counts, (str, bytes)
        ):
            if len(raw_counts) == 2 and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in raw_counts
            ):
                invalid, total = raw_counts
            else:
                total = len(raw_counts)
                invalid = sum(
                    _field(item, "severe_hallucination", "is_severe", "severe")
                    not in (True, False)
                    for item in raw_counts
                )
        else:
            raise TypeError("each arm count must be a mapping or observation sequence")
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise ValueError(f"arm {arm!r} has an invalid total")
        if (
            not isinstance(invalid, int)
            or isinstance(invalid, bool)
            or not 0 <= invalid <= total
        ):
            raise ValueError(f"arm {arm!r} has an invalid invalid-count")
        rate = invalid / total if total else None
        arm_rates[arm] = {"invalid": invalid, "total": total, "rate": rate}
        if rate is None:
            reasons.append(f"{arm}:no_observations")
        elif rate - max_invalid_rate > 1e-12:
            reasons.append(f"{arm}:invalid_rate_above_threshold")

    valid_rates = [
        float(counts["rate"])
        for counts in arm_rates.values()
        if counts["rate"] is not None
    ]
    between_arm_gap = max(valid_rates) - min(valid_rates) if valid_rates else None
    if between_arm_gap is not None and between_arm_gap - max_between_arm_gap > 1e-12:
        reasons.append("between_arm_invalid_rate_gap_above_threshold")
    insufficient = bool(reasons)
    return {
        "arms": arm_rates,
        "max_invalid_rate": max_invalid_rate,
        "max_between_arm_gap": max_between_arm_gap,
        "between_arm_gap": between_arm_gap,
        "insufficient": insufficient,
        "conclusion_sufficient": not insufficient,
        "reasons": reasons,
    }


def evaluate_comparison_gate(
    baseline_rate_or_result: float | Mapping[str, Any],
    comparison_rate: float | None = None,
    *,
    risk_difference_ci: Sequence[float] | None = None,
    seed_differences: Mapping[Any, float] | Iterable[float] | None = None,
    invalidity: Mapping[str, Any] | None = None,
    absolute_gap_threshold: float = 0.10,
    risk_ratio_threshold: float = 1.5,
    min_consistent_seeds: int = 2,
    required_seed_count: int = 3,
) -> dict[str, Any]:
    """Evaluate the locked gate in the declared comparison-over-baseline direction.

    For the primary experiment the comparator/prototype is the baseline and
    ``current_head`` is the comparison.  A cleaner current arm therefore cannot
    trigger an engine-problem conclusion merely because the absolute rates differ.
    """

    if not 0 <= absolute_gap_threshold <= 1:
        raise ValueError("absolute_gap_threshold must be between zero and one")
    if risk_ratio_threshold < 1 or not math.isfinite(risk_ratio_threshold):
        raise ValueError("risk_ratio_threshold must be finite and at least one")
    if min_consistent_seeds < 1 or required_seed_count < min_consistent_seeds:
        raise ValueError("seed requirements are inconsistent")
    if isinstance(baseline_rate_or_result, Mapping):
        result = baseline_rate_or_result
        baseline_rate = float(result["baseline_rate"])
        selected_comparison_rate = float(result["comparison_rate"])
        selected_ci = result.get("risk_difference_ci")
        selected_seed_differences = result.get("seed_differences")
        reported_ratio = result.get("risk_ratio")
        reported_ratio_infinite = result.get("risk_ratio_infinite")
    else:
        baseline_rate = float(baseline_rate_or_result)
        if comparison_rate is None:
            raise TypeError(
                "comparison_rate is required when no bootstrap result is supplied"
            )
        selected_comparison_rate = float(comparison_rate)
        selected_ci = risk_difference_ci
        selected_seed_differences = seed_differences
        reported_ratio = None
        reported_ratio_infinite = None
    if comparison_rate is not None and isinstance(baseline_rate_or_result, Mapping):
        raise TypeError("comparison_rate must not accompany a bootstrap result")
    if not 0 <= baseline_rate <= 1 or not 0 <= selected_comparison_rate <= 1:
        raise ValueError("rates must be between zero and one")
    if selected_ci is None or len(selected_ci) != 2:
        raise ValueError("risk_difference_ci must contain low and high bounds")
    ci_low, ci_high = float(selected_ci[0]), float(selected_ci[1])
    if not math.isfinite(ci_low) or not math.isfinite(ci_high) or ci_low > ci_high:
        raise ValueError("risk_difference_ci low bound exceeds high bound")

    if selected_seed_differences is None:
        differences: list[float] = []
    elif isinstance(selected_seed_differences, Mapping):
        differences = [float(value) for value in selected_seed_differences.values()]
    else:
        differences = [float(value) for value in selected_seed_differences]
    if any(not math.isfinite(difference) for difference in differences):
        raise ValueError("seed differences must be finite")
    point_difference = selected_comparison_rate - baseline_rate
    consistent_seed_count = sum(
        difference > 0
        for difference in differences
    )
    directional_ratio, directional_ratio_infinite = _serialized_ratio(
        _risk_ratio(selected_comparison_rate, baseline_rate)
    )
    if isinstance(baseline_rate_or_result, Mapping):
        if not isinstance(reported_ratio_infinite, bool):
            raise ValueError("risk_ratio_infinite must be an explicit boolean")
        if reported_ratio_infinite:
            if reported_ratio is not None:
                raise ValueError("an infinite risk ratio must be represented as null")
        elif (
            not isinstance(reported_ratio, (int, float))
            or isinstance(reported_ratio, bool)
            or not math.isfinite(float(reported_ratio))
        ):
            raise ValueError("a finite risk ratio must be a finite JSON number")
        if (
            reported_ratio_infinite != directional_ratio_infinite
            or (
                not directional_ratio_infinite
                and float(reported_ratio) != directional_ratio
            )
        ):
            raise ValueError("risk ratio representation disagrees with arm rates")
    absolute_gap = abs(point_difference)
    ci_excludes_zero_in_declared_direction = ci_low > 0
    invalidity_insufficient = bool(invalidity and invalidity.get("insufficient"))
    checks = {
        "absolute_gap": absolute_gap + 1e-12 >= absolute_gap_threshold,
        "risk_ratio": (
            directional_ratio_infinite
            or (
                directional_ratio is not None
                and directional_ratio + 1e-12 >= risk_ratio_threshold
            )
        ),
        "risk_difference_ci_excludes_zero": (
            ci_excludes_zero_in_declared_direction
        ),
        "seed_direction": (
            len(differences) == required_seed_count
            and consistent_seed_count >= min_consistent_seeds
        ),
        "validity": not invalidity_insufficient,
    }
    passes = all(checks.values())
    return {
        "passes": passes,
        "enter_root_cause_analysis": passes,
        "checks": checks,
        "baseline_rate": baseline_rate,
        "comparison_rate": selected_comparison_rate,
        "risk_difference": point_difference,
        "absolute_gap": absolute_gap,
        "comparison_minus_baseline_gap": point_difference,
        "absolute_gap_threshold": absolute_gap_threshold,
        "risk_ratio": directional_ratio,
        "risk_ratio_infinite": directional_ratio_infinite,
        "risk_ratio_threshold": risk_ratio_threshold,
        "risk_difference_ci": [ci_low, ci_high],
        "consistent_seed_count": consistent_seed_count,
        "seed_count": len(differences),
        "required_seed_count": required_seed_count,
        "min_consistent_seeds": min_consistent_seeds,
        "invalidity_insufficient": invalidity_insufficient,
    }


# Semantic aliases retained for report and caller readability.
paired_hierarchical_bootstrap = paired_bootstrap_comparison
evaluate_invalid_sample_rules = evaluate_invalidity


__all__ = [
    "evaluate_comparison_gate",
    "evaluate_invalid_sample_rules",
    "evaluate_invalidity",
    "paired_bootstrap_comparison",
    "paired_hierarchical_bootstrap",
    "wilson_interval",
]
