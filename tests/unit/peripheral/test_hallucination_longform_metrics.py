from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.validation.hallucination.longform.acoustic import acoustic_diagnostics
from tools.validation.hallucination.longform.artifacts import write_json
from tools.validation.hallucination.longform import metrics as metrics_facade
from tools.validation.hallucination.longform import rate_statistics
from tools.validation.hallucination.longform import text_normalization
from tools.validation.hallucination.longform import alignment_metrics
from tools.validation.hallucination.longform import (
    ArmKind,
    ReviewLabel,
    RunStatus,
    aggregate_reviews,
    build_blind_manifest,
    canonicalize_numbers,
    character_error_metrics,
    derive_severe_hallucination,
    evaluate_comparison_gate,
    evaluate_invalidity,
    paired_bootstrap_comparison,
    parse_reference_sentences,
    wilson_interval,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_acoustic_diagnostics_uses_the_configured_stft_frame_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba-cache"))
    sample_rate = 24_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = 0.1 * np.sin(2.0 * np.pi * 440.0 * time)

    result = acoustic_diagnostics(samples, sample_rate)

    assert result["sample_count"] == sample_rate
    assert result["duration_s"] == pytest.approx(1.0)
    assert result["spectral_flatness_median"] is not None
    assert result["high_frequency_ratio_median"] is not None
    assert result["diagnostic_only"] is True


def test_metrics_module_is_a_compatibility_facade_over_focused_owners():
    assert (
        metrics_facade.canonicalize_numbers is text_normalization.canonicalize_numbers
    )
    assert (
        metrics_facade.normalize_transcript is text_normalization.normalize_transcript
    )
    assert metrics_facade.AlignmentOpcode is alignment_metrics.AlignmentOpcode
    assert (
        metrics_facade.character_error_metrics
        is alignment_metrics.character_error_metrics
    )
    assert metrics_facade.levenshtein_opcodes is alignment_metrics.levenshtein_opcodes
    assert metrics_facade.repetition_spans is alignment_metrics.repetition_spans
    assert metrics_facade.wilson_interval is rate_statistics.wilson_interval
    assert (
        metrics_facade.paired_bootstrap_comparison
        is rate_statistics.paired_bootstrap_comparison
    )
    assert metrics_facade.evaluate_invalidity is rate_statistics.evaluate_invalidity
    assert (
        metrics_facade.evaluate_comparison_gate
        is rate_statistics.evaluate_comparison_gate
    )


def test_verylong_reference_has_38_occurrence_aware_sentences():
    text = (REPO_ROOT / "resources/dataset/badcase/verylong.txt").read_text(
        encoding="utf-8"
    )

    sentences = parse_reference_sentences(text)

    assert len(sentences) == 38
    assert "第一万七千九百五十项" in sentences[0].text
    assert sentences[-1].text.endswith("。")
    assert "".join(sentence.text for sentence in sentences) == text


def test_repeated_reference_sentences_keep_distinct_occurrences_and_offsets():
    text = "甲。乙。甲。"

    sentences = parse_reference_sentences(text)

    assert [sentence.text for sentence in sentences] == ["甲。", "乙。", "甲。"]
    assert [sentence.occurrence for sentence in sentences] == [1, 1, 2]
    assert len({sentence.sentence_id for sentence in sentences}) == 3
    assert [text[sentence.start : sentence.end] for sentence in sentences] == [
        sentence.text for sentence in sentences
    ]


def test_number_canonicalization_unifies_chinese_and_fullwidth_arabic_numbers():
    normalized = canonicalize_numbers("第两万四千零七十项，编号２０２４")

    assert "24070" in normalized
    assert "2024" in normalized
    assert character_error_metrics("第二万四千零七十项", "第24070项")["cer"] == 0


def test_character_alignment_reports_sdi_opcodes_insertions_and_repeat_span():
    metrics = character_error_metrics("甲乙丙", "甲丁丙哈哈哈")

    assert metrics["substitutions"] == 1
    assert metrics["deletions"] == 0
    assert metrics["insertions"] == 3
    assert metrics["distance"] == 4
    assert metrics["longest_insertion_span"]["text"] == "哈哈哈"
    assert metrics["longest_repetition_span"]["unit"] == "哈"
    assert metrics["longest_repetition_span"]["repetitions"] == 3
    assert {opcode["tag"] for opcode in metrics["opcodes"]} >= {
        "equal",
        "replace",
        "insert",
    }


def test_character_alignment_handles_a_long_reference_without_truncation():
    reference = "关于设备联网系统正在核验" * 40
    midpoint = len(reference) // 2
    hypothesis = reference[:midpoint] + "哈哈哈" + reference[midpoint:]

    metrics = character_error_metrics(reference, hypothesis)

    assert metrics["reference_characters"] > 400
    assert metrics["distance"] == 3
    assert metrics["insertions"] == 3
    assert metrics["longest_insertion_span"]["text"] == "哈哈哈"


def test_severe_truth_uses_locked_loop_noise_and_unsupported_speech_rules():
    assert derive_severe_hallucination(ReviewLabel.SINGLE_UNIT_LOOP) is True
    assert (
        derive_severe_hallucination(
            ReviewLabel.SINGLE_UNIT_LOOP, repetition_count=2, duration_s=0.49
        )
        is False
    )
    assert (
        derive_severe_hallucination(
            ReviewLabel.SINGLE_UNIT_LOOP, repetition_count=3, duration_s=0.1
        )
        is True
    )
    assert (
        derive_severe_hallucination(ReviewLabel.ABNORMAL_NOISE, duration_s=0.49)
        is False
    )
    assert (
        derive_severe_hallucination(ReviewLabel.ABNORMAL_NOISE, duration_s=0.5) is True
    )
    assert (
        derive_severe_hallucination(
            ReviewLabel.UNSUPPORTED_SPEECH, confirmed_unsupported_speech=False
        )
        is False
    )
    assert derive_severe_hallucination(ReviewLabel.OMISSION) is False
    assert derive_severe_hallucination(ReviewLabel.UNSCORABLE) is None


def _blind_records():
    return [
        {
            "arm": arm,
            "seed": seed,
            "session_id": f"private-session-{seed}",
            "sentence_id": f"sentence-{seed}",
            "sentence_ordinal": seed,
            "occurrence": 1,
            "reference_text": f"参考句{seed}。",
            "audio_path": f"/private/{arm.value}/session-{seed}.wav",
            "context_audio_path": f"/private/context/session-{seed}.wav",
        }
        for seed, arm in enumerate(
            (ArmKind.CURRENT_HEAD, ArmKind.TRITON_0818, ArmKind.PYTORCH_0818),
            start=1,
        )
    ]


def test_blind_manifest_randomizes_into_generated_paths_without_identity_leakage():
    manifest, private_key = build_blind_manifest(_blind_records(), random_seed=7)
    repeated_manifest, repeated_key = build_blind_manifest(
        _blind_records(), random_seed=7
    )

    assert manifest == repeated_manifest
    assert private_key == repeated_key
    public_json = json.dumps(manifest, ensure_ascii=False)
    for private_value in (
        "current_head",
        "triton_0818",
        "pytorch_0818",
        "private-session",
        "/private/",
    ):
        assert private_value not in public_json
    assert all(item["audio_path"].startswith("audio/LF-") for item in manifest)
    assert {entry["arm"] for entry in private_key.values()} == {
        arm.value
        for arm in (
            ArmKind.CURRENT_HEAD,
            ArmKind.TRITON_0818,
            ArmKind.PYTORCH_0818,
        )
    }


def test_review_aggregation_requires_hidden_adjudication_for_disagreement():
    reviews = [
        {"blind_id": "LF-0001", "reviewer_id": "r1", "label": "OK"},
        {
            "blind_id": "LF-0001",
            "reviewer_id": "r2",
            "label": "SINGLE_UNIT_LOOP",
        },
    ]

    unresolved = aggregate_reviews(reviews)[0]
    resolved = aggregate_reviews(
        reviews
        + [
            {
                "blind_id": "LF-0001",
                "reviewer_id": "judge",
                "label": "SINGLE_UNIT_LOOP",
                "is_adjudication": True,
            }
        ]
    )[0]

    assert unresolved["status"] == RunStatus.REVIEW_PENDING.value
    assert unresolved["severe_hallucination"] is None
    assert unresolved["requires_adjudication"] is True
    assert resolved["resolution"] == "adjudicated"
    assert resolved["severe_hallucination"] is True
    assert resolved["status"] == RunStatus.REVIEWED.value


def test_review_aggregation_does_not_finalize_a_single_positive_review():
    record = aggregate_reviews(
        [
            {
                "blind_id": "LF-0001",
                "reviewer_id": "r1",
                "label": "ABNORMAL_NOISE",
            }
        ]
    )[0]

    assert record["preliminary_label"] == ReviewLabel.ABNORMAL_NOISE.value
    assert record["label"] is None
    assert record["severe_hallucination"] is None
    assert record["requires_second_review"] is True
    assert record["status"] == RunStatus.REVIEW_PENDING.value


def test_review_aggregation_keeps_unscorable_and_missing_reviews_invalid():
    key = {
        "LF-0001": {"arm": ArmKind.CURRENT_HEAD.value, "seed": 1},
        "LF-0002": {"arm": ArmKind.CURRENT_HEAD.value, "seed": 2},
    }

    records = aggregate_reviews(
        [
            {
                "blind_id": "LF-0001",
                "reviewer_id": "r1",
                "label": "UNSCORABLE",
            },
            {
                "blind_id": "LF-0001",
                "reviewer_id": "r2",
                "label": "UNSCORABLE",
            },
        ],
        private_key=key,
    )

    assert records[0]["status"] == RunStatus.INVALID.value
    assert records[0]["severe_hallucination"] is None
    assert records[1]["status"] == RunStatus.REVIEW_PENDING.value
    assert records[1]["severe_hallucination"] is None


def _paired_observations() -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for seed in (11, 22, 33):
        for sentence in range(1, 21):
            observations.extend(
                [
                    {
                        "arm": ArmKind.CURRENT_HEAD,
                        "seed": seed,
                        "sentence_ordinal": sentence,
                        "severe_hallucination": sentence <= 2,
                    },
                    {
                        "arm": ArmKind.TRITON_0818,
                        "seed": seed,
                        "sentence_ordinal": sentence,
                        "severe_hallucination": sentence <= 8,
                    },
                ]
            )
    return observations


def test_paired_hierarchical_bootstrap_is_deterministic_and_preserves_pairing():
    first = paired_bootstrap_comparison(
        _paired_observations(),
        ArmKind.CURRENT_HEAD,
        ArmKind.TRITON_0818,
        iterations=1_000,
        random_seed=123,
    )
    second = paired_bootstrap_comparison(
        _paired_observations(),
        ArmKind.CURRENT_HEAD,
        ArmKind.TRITON_0818,
        iterations=1_000,
        random_seed=123,
    )

    assert first == second
    assert first["matched_pairs"] == 60
    assert first["baseline_rate"] == pytest.approx(0.1)
    assert first["comparison_rate"] == pytest.approx(0.4)
    assert first["risk_difference"] == pytest.approx(0.3)
    assert first["risk_ratio"] == pytest.approx(4.0)
    assert first["risk_difference_ci"][0] > 0
    assert list(first["seed_differences"].values()) == pytest.approx([0.3] * 3)


def _observations_with_extra_current_truth(
    extra_fields: dict[str, object],
) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for seed in (11, 22, 33):
        observations.extend(
            [
                {
                    "arm": ArmKind.CURRENT_HEAD.value,
                    "seed": seed,
                    "sentence_ordinal": 1,
                    "severe_hallucination": False,
                    "valid_for_rate": True,
                    "status": RunStatus.REVIEWED.value,
                },
                {
                    "arm": ArmKind.TRITON_0818.value,
                    "seed": seed,
                    "sentence_ordinal": 1,
                    "severe_hallucination": False,
                    "valid_for_rate": True,
                    "status": RunStatus.REVIEWED.value,
                },
            ]
        )
    observations.extend(
        [
            {
                "arm": ArmKind.CURRENT_HEAD.value,
                "seed": 11,
                "sentence_ordinal": 2,
                "severe_hallucination": True,
                **extra_fields,
            },
            {
                "arm": ArmKind.TRITON_0818.value,
                "seed": 11,
                "sentence_ordinal": 2,
                "severe_hallucination": False,
                "valid_for_rate": True,
                "status": RunStatus.REVIEWED.value,
            },
        ]
    )

    return observations


def test_paired_bootstrap_excludes_explicitly_invalid_review_truth() -> None:
    result = paired_bootstrap_comparison(
        _observations_with_extra_current_truth(
            {"valid_for_rate": False, "status": RunStatus.REVIEWED.value}
        ),
        ArmKind.CURRENT_HEAD,
        ArmKind.TRITON_0818,
        iterations=20,
        random_seed=7,
    )

    assert result["matched_pairs"] == 3
    assert result["excluded_invalid_pairs"] == 1
    assert result["baseline_rate"] == result["comparison_rate"] == 0.0


@pytest.mark.parametrize("valid_for_rate", (True, False))
def test_asr_failure_does_not_erase_existing_human_truth(
    valid_for_rate: bool,
) -> None:
    result = paired_bootstrap_comparison(
        _observations_with_extra_current_truth(
            {
                "valid_for_rate": valid_for_rate,
                "status": RunStatus.ASR_FAILED.value,
            }
        ),
        ArmKind.CURRENT_HEAD,
        ArmKind.TRITON_0818,
        iterations=20,
        random_seed=7,
    )

    assert result["matched_pairs"] == 4
    assert result["excluded_invalid_pairs"] == 0
    assert result["baseline_rate"] == pytest.approx(0.25)
    assert result["comparison_rate"] == 0.0


def test_comparison_gate_requires_every_locked_condition():
    comparison = paired_bootstrap_comparison(
        _paired_observations(),
        ArmKind.CURRENT_HEAD,
        ArmKind.TRITON_0818,
        iterations=1_000,
        random_seed=123,
    )

    passing = evaluate_comparison_gate(comparison)
    inconsistent = evaluate_comparison_gate(
        0.10,
        0.25,
        risk_difference_ci=(0.01, 0.30),
        seed_differences=(0.15, -0.10, -0.05),
    )

    assert passing["passes"] is True
    assert passing["consistent_seed_count"] == 3
    assert inconsistent["passes"] is False
    assert inconsistent["checks"]["seed_direction"] is False


def test_comparison_gate_accepts_inclusive_rate_boundaries_and_two_of_three_seeds():
    result = evaluate_comparison_gate(
        0.20,
        0.30,
        risk_difference_ci=(0.0001, 0.20),
        seed_differences=(0.10, 0.10, -0.01),
    )

    assert result["passes"] is True
    assert result["checks"]["absolute_gap"] is True
    assert result["checks"]["risk_ratio"] is True
    assert result["consistent_seed_count"] == 2
    assert result["risk_ratio"] == pytest.approx(1.5)
    assert result["risk_ratio_infinite"] is False


def test_comparison_gate_rejects_ci_endpoint_zero_and_only_one_consistent_seed():
    ci_touches_zero = evaluate_comparison_gate(
        0.20,
        0.31,
        risk_difference_ci=(0.0, 0.20),
        seed_differences=(0.11, 0.11, -0.01),
    )
    one_seed = evaluate_comparison_gate(
        0.20,
        0.31,
        risk_difference_ci=(0.001, 0.20),
        seed_differences=(0.11, -0.01, 0.0),
    )

    assert ci_touches_zero["passes"] is False
    assert ci_touches_zero["checks"]["risk_difference_ci_excludes_zero"] is False
    assert one_seed["passes"] is False
    assert one_seed["checks"]["seed_direction"] is False


def test_comparison_gate_never_treats_a_cleaner_comparison_as_engine_regression():
    reverse = evaluate_comparison_gate(
        0.30,
        0.10,
        risk_difference_ci=(-0.30, -0.10),
        seed_differences=(-0.20, -0.20, -0.20),
    )

    assert reverse["absolute_gap"] == pytest.approx(0.20)
    assert reverse["passes"] is False
    assert reverse["checks"] == {
        "absolute_gap": True,
        "risk_ratio": False,
        "risk_difference_ci_excludes_zero": False,
        "seed_direction": False,
        "validity": True,
    }


def _crossed_cluster_observations(*, clustered: bool) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        for ordinal in range(1, 11):
            comparison_severe = ordinal == (1 if clustered else seed)
            observations.extend(
                [
                    {
                        "arm": ArmKind.TRITON_0818.value,
                        "seed": seed,
                        "sentence_ordinal": ordinal,
                        "severe_hallucination": False,
                    },
                    {
                        "arm": ArmKind.CURRENT_HEAD.value,
                        "seed": seed,
                        "sentence_ordinal": ordinal,
                        "severe_hallucination": comparison_severe,
                    },
                ]
            )
    return observations


def test_two_way_crossed_bootstrap_preserves_clustered_bad_sentence_uncertainty():
    clustered = paired_bootstrap_comparison(
        _crossed_cluster_observations(clustered=True),
        ArmKind.TRITON_0818,
        ArmKind.CURRENT_HEAD,
        iterations=10_000,
        random_seed=8182028,
    )
    dispersed = paired_bootstrap_comparison(
        _crossed_cluster_observations(clustered=False),
        ArmKind.TRITON_0818,
        ArmKind.CURRENT_HEAD,
        iterations=10_000,
        random_seed=8182028,
    )

    assert clustered["resampling_scheme"] == (
        "two_way_crossed_seed_sentence_cartesian"
    )
    assert clustered["risk_difference"] == dispersed["risk_difference"] == 0.1
    assert clustered["risk_difference_ci"][1] > dispersed["risk_difference_ci"][1]


def test_zero_baseline_ratio_uses_strict_json_extended_real_representation():
    result = paired_bootstrap_comparison(
        _crossed_cluster_observations(clustered=True),
        ArmKind.TRITON_0818,
        ArmKind.CURRENT_HEAD,
        iterations=1_000,
        random_seed=19,
    )
    gate = evaluate_comparison_gate(result)

    assert result["risk_ratio"] is None
    assert result["risk_ratio_infinite"] is True
    assert result["risk_ratio_ci_infinite"] == [False, True]
    assert result["risk_ratio_ci"][1] is None
    assert gate["risk_ratio"] is None
    assert gate["risk_ratio_infinite"] is True
    assert gate["checks"]["risk_ratio"] is True
    assert "Infinity" not in json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("non_finite", (math.inf, -math.inf, math.nan))
def test_write_json_rejects_non_finite_numbers(
    tmp_path: Path,
    non_finite: float,
) -> None:
    path = tmp_path / "strict.json"

    with pytest.raises(ValueError, match="Out of range float"):
        write_json(path, {"value": non_finite})

    assert not path.exists()


def test_invalid_sample_rules_do_not_count_missing_as_clean():
    insufficient = evaluate_invalidity(
        {
            ArmKind.CURRENT_HEAD: {"total": 100, "tts_failed": 2, "unscorable": 4},
            ArmKind.TRITON_0818: {"total": 100, "invalid": 1},
            ArmKind.PYTORCH_0818: {"total": 100, "invalid": 1},
        }
    )
    boundary = evaluate_invalidity(
        {
            ArmKind.CURRENT_HEAD: {"total": 100, "invalid": 5},
            ArmKind.TRITON_0818: {"total": 100, "invalid": 3},
        }
    )

    assert insufficient["insufficient"] is True
    assert insufficient["arms"][ArmKind.CURRENT_HEAD.value]["rate"] == 0.06
    assert "between_arm_invalid_rate_gap_above_threshold" in insufficient["reasons"]
    assert boundary["insufficient"] is False


def test_wilson_interval_is_reportable_for_each_arm_and_unknown_for_no_truth():
    interval = wilson_interval(10, 114)
    empty = wilson_interval(0, 0)

    assert interval["rate"] == pytest.approx(10 / 114)
    assert interval["low"] < interval["rate"] < interval["high"]
    assert empty["rate"] is empty["low"] is empty["high"] is None
