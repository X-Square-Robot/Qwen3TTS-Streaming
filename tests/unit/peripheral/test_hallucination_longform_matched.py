from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qwen3tts_protocol import AudioChunk, AudioFormat, StreamEvent

from tools.validation.hallucination.longform.arm_types import (
    AudioChunkRecord,
    CollectedRun,
    stable_sampling_seed,
)
from tools.validation.hallucination.longform.artifacts import read_json, write_json
from tools.validation.hallucination.longform.collection import initialize_experiment
from tools.validation.hallucination.longform.endpoint_arm import EngineGrpcArmAdapter
from tools.validation.hallucination.longform.formal_statistics import (
    FORMAL_BOOTSTRAP_CONFIDENCE,
    FORMAL_BOOTSTRAP_ITERATIONS,
    FORMAL_BOOTSTRAP_SEED,
)
from tools.validation.hallucination.longform.matched import (
    MATCHED_GENERATION,
    MatchedEndpointArm,
    MatchedOfficialArm,
    greedy_hash_comparison,
    prepare_matched_experiment,
)
from tools.validation.hallucination.longform.models import (
    ArmKind,
    ReviewLabel,
    RunStatus,
    parse_reference_sentences,
)
from tools.validation.hallucination.longform.official_arm import (
    OfficialPyTorchArmAdapter,
)
from tools.validation.hallucination.longform.reference_groups import (
    freeze_reference_groups,
)
from tools.validation.hallucination.longform.reporting import generate_report


_SENTENCE_TEXTS = tuple(f"第{ordinal}句。" for ordinal in range(1, 39))
_PART_SIZES = (5, 5, 4, 4, 4, 4, 4, 4, 4)


def _parts() -> tuple[str, ...]:
    result: list[str] = []
    cursor = 0
    for index, size in enumerate(_PART_SIZES):
        text = "".join(_SENTENCE_TEXTS[cursor : cursor + size])
        if index == 0:
            text = "  " + text
        if index in {1, 6, 8}:
            text += "\n"
        result.append(text)
        cursor += size
    assert cursor == 38
    return tuple(result)


_PARTS = _parts()
_SEEDS = (17041, 28411, 39671)


def _content_identity(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _groups(parts: tuple[str, ...] = _PARTS) -> list[dict[str, Any]]:
    source_text = "".join(parts)
    sentences = parse_reference_sentences(source_text)
    cursor = 0
    result: list[dict[str, Any]] = []
    for segment_id, text in enumerate(parts):
        end = cursor + len(text)
        result.append(
            {
                "segment_id": segment_id,
                "group_index": segment_id + 1,
                "raw_start": cursor,
                "raw_end": end,
                "text": text,
                "event_text": text,
                "sentence_ordinals": [
                    sentence.ordinal
                    for sentence in sentences
                    if sentence.end > cursor and sentence.start < end
                ],
            }
        )
        cursor = end
    return result


def _primary_experiment(
    tmp_path: Path,
    *,
    gate_passes: bool,
    insufficient: bool = False,
    groups: list[dict[str, Any]] | None = None,
    bootstrap_iterations: int = FORMAL_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = FORMAL_BOOTSTRAP_SEED,
) -> tuple[Path, str, list[dict[str, Any]]]:
    frozen_groups = groups if groups is not None else _groups()
    source_text = "".join(_PARTS)
    source = tmp_path / "verylong.txt"
    source.write_text(source_text, encoding="utf-8")
    root = tmp_path / "experiment"
    initialize_experiment(
        root,
        text_file=source,
        seeds=(17041, 28411, 39671),
        identities={"asr": {"release": "0.2.0a6"}},
    )
    write_json(
        root / "reference_groups.json",
        {
            "schema_version": 1,
            "source_arm": ArmKind.TRITON_0818.value,
            "group_count": len(frozen_groups),
            "groups": frozen_groups,
        },
    )
    write_json(
        root / "review" / "private" / "final_labels.json",
        {
            "rows": [
                {
                    "arm": arm.value,
                    "seed": seed,
                    "sentence_ordinal": sentence.ordinal,
                    "sentence_id": sentence.sentence_id,
                    "reference_text": sentence.text,
                    "blind_id": (f"blind-{arm.value}-{seed}-{sentence.ordinal:03d}"),
                    "review_label": (
                        ReviewLabel.UNSCORABLE.value
                        if invalid
                        else ReviewLabel.SINGLE_UNIT_LOOP.value
                        if severe
                        else ReviewLabel.OK.value
                    ),
                    "review_resolution": "fixture_two_reviewer_agreement",
                    "severe_hallucination": None if invalid else severe,
                    "valid_for_rate": not invalid,
                    "status": (
                        RunStatus.INVALID.value if invalid else RunStatus.REVIEWED.value
                    ),
                }
                for arm in (
                    ArmKind.CURRENT_HEAD,
                    ArmKind.TRITON_0818,
                    ArmKind.PYTORCH_0818,
                )
                for seed in _SEEDS
                for sentence in parse_reference_sentences(source_text)
                for invalid in (
                    insufficient
                    and arm is ArmKind.CURRENT_HEAD
                    and seed == _SEEDS[0]
                    and sentence.ordinal <= 6,
                )
                for severe in (
                    gate_passes
                    and (
                        sentence.ordinal <= 15
                        if arm is ArmKind.CURRENT_HEAD
                        else sentence.ordinal <= 1
                    ),
                )
            ]
        },
    )
    generate_report(
        root,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    return root, source_text, frozen_groups


def _triton_commit_experiment(
    tmp_path: Path,
    *,
    parts_by_seed: dict[int, tuple[str, ...]] | None = None,
    one_valid_offset: bool = False,
    all_valid_offsets: bool = False,
) -> tuple[Path, str]:
    source_text = "".join(_PARTS)
    source = tmp_path / "verylong.txt"
    source.write_text(source_text, encoding="utf-8")
    root = tmp_path / "commit-experiment"
    initialize_experiment(
        root,
        text_file=source,
        seeds=_SEEDS,
        identities={"asr": {"release": "0.2.0a6"}},
    )
    selected = parts_by_seed or {seed: _PARTS for seed in _SEEDS}
    for seed in _SEEDS:
        cursor = 0
        events = []
        for segment_id, text in enumerate(selected[seed]):
            end = cursor + len(text)
            use_real_offset = all_valid_offsets or (
                one_valid_offset and seed == _SEEDS[0] and segment_id == 0
            )
            events.append(
                {
                    "type": "text_boundary_commit",
                    "session_id": f"triton-{seed}",
                    "segment_id": segment_id,
                    "text": text,
                    "message": "",
                    "audio": None,
                    "meta": {
                        "raw_codepoint_start": str(cursor if use_real_offset else 0),
                        "raw_codepoint_end": str(end if use_real_offset else 0),
                    },
                }
            )
            cursor = end
        write_json(
            root / "arms" / ArmKind.TRITON_0818.value / f"seed_{seed:04d}" / "run.json",
            {
                "arm": ArmKind.TRITON_0818.value,
                "seed": seed,
                "status": RunStatus.OK.value,
                "events": events,
            },
        )
    return root, source_text


def test_zero_commit_offsets_rebuild_nine_groups_from_exact_text(
    tmp_path: Path,
) -> None:
    root, source_text = _triton_commit_experiment(tmp_path)

    groups = freeze_reference_groups(root)
    payload = read_json(root / "reference_groups.json")

    assert len(groups) == payload["group_count"] == 9
    assert payload["seed_count"] == 3
    assert payload["provenance"] == "derived_from_exact_commit_text"
    assert {group["provenance"] for group in groups} == {
        "derived_from_exact_commit_text"
    }
    assert [group["segment_id"] for group in groups] == list(range(9))
    assert [group["raw_start"] for group in groups] == [
        0,
        *[sum(len(part) for part in _PARTS[:index]) for index in range(1, 9)],
    ]
    assert groups[-1]["raw_end"] == len(source_text)
    assert "".join(group["text"] for group in groups) == source_text


def test_complete_real_offsets_are_verified_before_text_derivation(
    tmp_path: Path,
) -> None:
    root, source_text = _triton_commit_experiment(tmp_path, all_valid_offsets=True)

    groups = freeze_reference_groups(root)
    payload = read_json(root / "reference_groups.json")

    assert len(groups) == 9
    assert payload["provenance"] == "verified_event_raw_codepoint_offsets"
    assert {group["provenance"] for group in groups} == {
        "verified_event_raw_codepoint_offsets"
    }
    assert "".join(group["text"] for group in groups) == source_text


def test_zero_offset_rebuild_rejects_seed_commit_text_disagreement(
    tmp_path: Path,
) -> None:
    shifted = list(_PARTS)
    shifted[0] += shifted[1][0]
    shifted[1] = shifted[1][1:]
    root, _ = _triton_commit_experiment(
        tmp_path,
        parts_by_seed={
            _SEEDS[0]: _PARTS,
            _SEEDS[1]: tuple(shifted),
            _SEEDS[2]: _PARTS,
        },
    )

    with pytest.raises(RuntimeError, match="commit texts differ"):
        freeze_reference_groups(root)

    assert not (root / "reference_groups.json").exists()


def test_reference_group_freeze_rejects_partially_valid_offsets(tmp_path: Path) -> None:
    root, _ = _triton_commit_experiment(tmp_path, one_valid_offset=True)

    with pytest.raises(RuntimeError, match="only partially valid"):
        freeze_reference_groups(root)

    assert not (root / "reference_groups.json").exists()


@pytest.mark.parametrize(
    ("gate_passes", "insufficient", "message"),
    [
        (False, False, "did not pass the root-cause gate"),
        (True, True, "insufficient under the invalid-sample rule"),
    ],
)
def test_matched_replay_hard_rejects_before_primary_rate_gate(
    tmp_path: Path,
    gate_passes: bool,
    insufficient: bool,
    message: str,
) -> None:
    root, _, _ = _primary_experiment(
        tmp_path,
        gate_passes=gate_passes,
        insufficient=insufficient,
    )

    with pytest.raises(RuntimeError, match=message):
        prepare_matched_experiment(root)

    assert not (root / "matched").exists()


@pytest.mark.parametrize("iterations", (1, 20, 200))
def test_matched_replay_rejects_nonformal_fast_bootstrap_reports(
    tmp_path: Path,
    iterations: int,
) -> None:
    root, _, _ = _primary_experiment(
        tmp_path,
        gate_passes=True,
        bootstrap_iterations=iterations,
    )

    with pytest.raises(RuntimeError, match="invalid primary bootstrap configuration"):
        prepare_matched_experiment(root)

    assert not (root / "matched").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (("confidence", 0.90), ("random_seed", 7)),
)
def test_matched_replay_rejects_tampered_formal_bootstrap_protocol(
    tmp_path: Path,
    field: str,
    value: float | int,
) -> None:
    root, _, _ = _primary_experiment(tmp_path, gate_passes=True)
    report_path = root / "report" / "report.json"
    report = read_json(report_path)
    report["comparisons"][0]["bootstrap"][field] = value
    write_json(report_path, report)
    provenance_path = root / "report" / "report_provenance.json"
    provenance = read_json(provenance_path)
    provenance["artifacts"]["report"] = _content_identity(root, report_path)
    write_json(provenance_path, provenance)

    with pytest.raises(RuntimeError, match="invalid primary bootstrap configuration"):
        prepare_matched_experiment(root)

    assert not (root / "matched").exists()


def test_matched_replay_recomputes_gate_instead_of_trusting_passes(
    tmp_path: Path,
) -> None:
    root, _, _ = _primary_experiment(tmp_path, gate_passes=False)
    report_path = root / "report" / "report.json"
    report = read_json(report_path)
    report["comparisons"][0]["gate"]["passes"] = True
    report["comparisons"][0]["gate"]["enter_root_cause_analysis"] = True
    write_json(report_path, report)
    provenance_path = root / "report" / "report_provenance.json"
    provenance = read_json(provenance_path)
    provenance["artifacts"]["report"] = _content_identity(root, report_path)
    write_json(provenance_path, provenance)

    with pytest.raises(RuntimeError, match="gate is not reproducible"):
        prepare_matched_experiment(root)

    assert not (root / "matched").exists()


def test_matched_replay_rejects_final_labels_changed_after_report(
    tmp_path: Path,
) -> None:
    root, _, _ = _primary_experiment(tmp_path, gate_passes=True)
    final_labels_path = root / "review" / "private" / "final_labels.json"
    final_labels = read_json(final_labels_path)
    clean_row = next(
        row
        for row in final_labels["rows"]
        if row["review_label"] == ReviewLabel.OK.value
    )
    clean_row["review_label"] = ReviewLabel.MISPRONUNCIATION.value
    write_json(final_labels_path, final_labels)

    with pytest.raises(RuntimeError, match="report input hashes"):
        prepare_matched_experiment(root)

    assert not (root / "matched").exists()


def test_matched_replay_requires_exactly_342_final_labels(tmp_path: Path) -> None:
    root, _, _ = _primary_experiment(tmp_path, gate_passes=True)
    final_labels_path = root / "review" / "private" / "final_labels.json"
    final_labels = read_json(final_labels_path)
    final_labels["rows"].pop()
    write_json(final_labels_path, final_labels)

    with pytest.raises(RuntimeError, match="exactly 342 rows"):
        prepare_matched_experiment(root)

    assert not (root / "matched").exists()


def test_matched_replay_accepts_dynamic_frozen_groups_that_reconstruct_source(
    tmp_path: Path,
) -> None:
    root, source_text, groups = _primary_experiment(tmp_path, gate_passes=True)

    matched_root = prepare_matched_experiment(root, mode="sample")

    frozen = read_json(matched_root / "reference_groups.json")["groups"]
    manifest = read_json(matched_root / "manifest.json")
    assert len(frozen) == len(_PARTS) == 9
    assert frozen == groups
    assert "".join(group["text"] for group in frozen) == source_text
    assert manifest["text"]["text"] == source_text
    assert manifest["identities"]["frozen_reference_groups"]["groups"] == groups
    assert manifest["identities"]["endpoint_input_mode"] == "long_segment"
    assert manifest["identities"]["endpoint_group_policy"] == "auto"
    parent_paths = {
        "parent_manifest_sha256": root / "manifest.json",
        "parent_reference_groups_sha256": root / "reference_groups.json",
        "parent_final_labels_sha256": (
            root / "review" / "private" / "final_labels.json"
        ),
        "parent_report_sha256": root / "report" / "report.json",
        "parent_report_provenance_sha256": (root / "report" / "report_provenance.json"),
    }
    for identity, path in parent_paths.items():
        assert (
            manifest["identities"][identity]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_matched_replay_does_not_hardcode_the_formal_nine_group_count(
    tmp_path: Path,
) -> None:
    dynamic_parts = (
        "".join(_PARTS[:2]),
        "".join(_PARTS[2:6]),
        "".join(_PARTS[6:]),
    )
    groups = _groups(dynamic_parts)
    root, source_text, _ = _primary_experiment(
        tmp_path,
        gate_passes=True,
        groups=groups,
    )

    matched_root = prepare_matched_experiment(root)
    frozen = read_json(matched_root / "reference_groups.json")["groups"]

    assert len(frozen) == 3
    assert "".join(group["text"] for group in frozen) == source_text


def test_matched_replay_rejects_groups_that_do_not_reconstruct_source(
    tmp_path: Path,
) -> None:
    bad_groups = _groups()
    bad_groups[3] = {**bad_groups[3], "text": "篡改。"}
    root, _, _ = _primary_experiment(
        tmp_path,
        gate_passes=True,
        groups=bad_groups,
    )

    with pytest.raises(RuntimeError, match="disagrees with source"):
        prepare_matched_experiment(root)

    assert not (root / "matched").exists()


@pytest.mark.parametrize("bad_segment_id", (0, 9))
def test_matched_replay_rejects_duplicate_or_gapped_segment_ids(
    tmp_path: Path,
    bad_segment_id: int,
) -> None:
    bad_groups = _groups()
    bad_groups[1] = {**bad_groups[1], "segment_id": bad_segment_id}
    root, _, _ = _primary_experiment(
        tmp_path,
        gate_passes=True,
        groups=bad_groups,
    )

    with pytest.raises(RuntimeError, match="continuous, unique"):
        prepare_matched_experiment(root)

    assert not (root / "matched").exists()


class _FakeSession:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.ended = False

    def send_text(self, text: str) -> None:
        self.sent.append(text)

    def end(self) -> None:
        self.ended = True

    def iter_messages(self, *, post_send_idle_timeout: float):
        del post_send_idle_timeout
        for segment_id, text in enumerate(self.sent):
            yield StreamEvent(
                type="text_boundary_commit",
                session_id="matched-sid",
                segment_id=segment_id,
                text=text,
            )
        yield AudioChunk(
            pcm_bytes=np.asarray([0.125], dtype="<f4").tobytes(),
            audio=AudioFormat(encoding="pcm_f32", sample_rate=24_000, channels=1),
            chunk_index=0,
            first_chunk=True,
            final_chunk=True,
        )
        yield StreamEvent(type="done", session_id="matched-sid")

    def close(self, reason: str = "") -> None:
        del reason


class _FakeClient:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session
        self.requests: list[Any] = []

    def open_stream(self, request: Any) -> _FakeSession:
        self.requests.append(request)
        return self.session

    def close(self) -> None:
        pass


def test_matched_endpoint_uses_dynamic_packets_with_firehose_and_no_vad() -> None:
    groups = _groups()
    source_text = "".join(group["text"] for group in groups)
    session = _FakeSession()
    client = _FakeClient(session)
    endpoint = EngineGrpcArmAdapter(
        "fake-engine:50051",
        client_factory=lambda _endpoint, **_kwargs: client,
    )
    arm = MatchedEndpointArm(endpoint, groups)

    result = arm.collect(source_text, session_id="matched-sid", seed=17041)

    assert result.status is RunStatus.OK
    assert len(client.requests) == 1
    assert session.sent == [group["text"] for group in groups]
    assert session.ended is True
    request = client.requests[0]
    assert request.session_id == "matched-sid"
    assert request.config.input_mode == "long_segment"
    assert request.config.group_policy == "auto"
    assert request.config.output_policy is request.output_policy
    assert request.output_policy.vad.enabled is False
    assert request.output_policy.vad.strategy == "disabled"
    assert request.output_policy.config == {"delivery": "firehose"}


def test_matched_endpoint_fails_closed_when_endpoint_resegments_packets() -> None:
    groups = _groups()
    source_text = "".join(group["text"] for group in groups)
    session = _FakeSession()
    client = _FakeClient(session)
    endpoint = EngineGrpcArmAdapter(
        "fake-engine:50051",
        client_factory=lambda _endpoint, **_kwargs: client,
    )
    arm = MatchedEndpointArm(endpoint, groups)

    # Simulate a frontend that coalesced all frozen packets before committing.
    session.sent = [source_text]
    result = arm.collect(source_text, session_id="matched-sid", seed=17041)

    assert result.status is RunStatus.ERROR
    assert result.eos_reason == "matched_boundary_mismatch"
    assert "not preserved" in str(result.error)


class _FakeOfficialModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_custom_voice(self, **kwargs: Any):
        self.calls.append(dict(kwargs))
        value = len(self.calls) / 10.0
        return [np.asarray([value], dtype=np.float32)], 24_000


def test_matched_official_uses_real_segment_indices_and_explicit_generation_kwargs(
    tmp_path: Path,
) -> None:
    groups = _groups()
    source_text = "".join(group["text"] for group in groups)
    model = _FakeOfficialModel()
    seeded: list[int] = []
    endpoint = OfficialPyTorchArmAdapter(
        tmp_path / "fake-checkpoint",
        sampling_base_seed=0,
        model_factory=lambda _checkpoint: model,
        seed_setter=seeded.append,
    )
    arm = MatchedOfficialArm(
        endpoint,
        groups,
        generation_kwargs=MATCHED_GENERATION,
    )

    result = arm.collect(source_text, session_id="official-sid", seed=28411)

    expected_seeds = [
        stable_sampling_seed(0, "official-sid", segment_index)
        for segment_index in range(len(groups))
    ]
    assert seeded == expected_seeds
    assert len(set(seeded)) == len(groups)
    assert result.status is RunStatus.OK
    assert result.seed == 28411
    assert [event["meta"]["sampling_seed"] for event in result.events[::3]] == [
        str(value) for value in expected_seeds
    ]
    assert model.calls == [
        {
            "text": group["text"],
            "speaker": "001",
            "language": "Auto",
            **MATCHED_GENERATION,
        }
        for group in groups
    ]


def _chunk(values: np.ndarray) -> AudioChunkRecord:
    pcm = np.asarray(values, dtype="<f4").tobytes()
    return AudioChunkRecord(
        sequence_index=0,
        chunk_index=0,
        sample_start=0,
        sample_end=int(values.size),
        sample_count=int(values.size),
        sample_rate=24_000,
        channels=1,
        encoding="pcm_f32",
        first_chunk=True,
        final_chunk=True,
        output_sample_start=0,
        output_sample_end=int(values.size),
        meta={"source": "fake_official"},
        pcm_sha256=hashlib.sha256(pcm).hexdigest(),
    )


class _BoundaryOfficialAdapter:
    arm = ArmKind.PYTORCH_0818

    def __init__(self, lengths: tuple[int, ...]) -> None:
        self.lengths = lengths

    def collect_segment(
        self,
        text: str,
        session_id: str,
        trial_seed: int,
        segment_index: int,
        generation_kwargs: dict[str, Any],
    ) -> CollectedRun:
        del text, generation_kwargs
        samples = np.full(
            self.lengths[segment_index], segment_index + 1, dtype=np.float32
        )
        return CollectedRun(
            arm=ArmKind.PYTORCH_0818,
            seed=trial_seed,
            session_id=session_id,
            status=RunStatus.OK,
            samples=samples,
            sample_rate=24_000,
            duration_s=samples.size / 24_000,
            audio_chunks=[_chunk(samples)],
            total_ms=segment_index + 1,
            sampling_seed=1000 + segment_index,
        )


def test_matched_official_merge_preserves_pcm_and_cumulative_segment_boundaries() -> (
    None
):
    groups = _groups()
    source_text = "".join(group["text"] for group in groups)
    lengths = (2, 1, 4, 3, 2, 5, 1, 3, 2)
    arm = MatchedOfficialArm(
        _BoundaryOfficialAdapter(lengths),
        groups,
        generation_kwargs=MATCHED_GENERATION,
    )

    result = arm.collect(source_text, session_id="boundary-sid", seed=39671)

    expected = np.concatenate(
        [
            np.full(length, index + 1, dtype=np.float32)
            for index, length in enumerate(lengths)
        ]
    )
    assert result.samples.tolist() == expected.tolist()
    ends = np.cumsum(lengths).tolist()
    starts = [0, *ends[:-1]]
    assert [chunk.sample_start for chunk in result.audio_chunks] == starts
    assert [chunk.sample_end for chunk in result.audio_chunks] == ends
    assert [chunk.output_sample_start for chunk in result.audio_chunks] == starts
    assert [chunk.output_sample_end for chunk in result.audio_chunks] == ends
    assert [chunk.sequence_index for chunk in result.audio_chunks] == list(
        range(len(groups))
    )
    assert [chunk.meta["segment_index"] for chunk in result.audio_chunks] == [
        str(index) for index in range(len(groups))
    ]

    progress = [event for event in result.events if event["type"] == "text_progress"]
    boundaries = [event for event in result.events if event["type"] == "segment_end"]
    assert [event["segment_id"] for event in progress] == list(range(len(groups)))
    assert [int(event["meta"]["output_sample_end"]) for event in progress] == ends
    assert [event["text"] for event in boundaries] == [
        group["text"] for group in groups
    ]
    assert result.duration_s == pytest.approx(sum(lengths) / 24_000)
    assert result.total_ms == sum(range(1, len(groups) + 1))
    assert result.terminal_event == "done"
    assert result.eos_reason == "all_segments_complete"


def _greedy_record(root: Path, arm: ArmKind, seed: int, digest: str) -> None:
    run_path = (
        root
        / "matched"
        / "greedy"
        / "arms"
        / arm.value
        / f"seed_{seed:04d}"
        / "run.json"
    )
    write_json(
        run_path,
        {
            "arm": arm.value,
            "seed": seed,
            "status": RunStatus.OK.value,
            "pcm_s16le_sha256": digest,
        },
    )


def test_greedy_hash_comparison_pairs_current_and_triton_by_seed(
    tmp_path: Path,
) -> None:
    equal_hash = "a" * 64
    current_only_hash = "b" * 64
    triton_other_hash = "c" * 64
    for arm in (ArmKind.CURRENT_HEAD, ArmKind.TRITON_0818):
        _greedy_record(tmp_path, arm, 17041, equal_hash)
    _greedy_record(tmp_path, ArmKind.CURRENT_HEAD, 28411, current_only_hash)
    _greedy_record(tmp_path, ArmKind.TRITON_0818, 28411, triton_other_hash)

    result = greedy_hash_comparison(tmp_path)

    assert result == {
        "comparisons": [
            {
                "seed": 17041,
                "current_sha256": equal_hash,
                "triton_sha256": equal_hash,
                "current_equals_triton": True,
            },
            {
                "seed": 28411,
                "current_sha256": current_only_hash,
                "triton_sha256": triton_other_hash,
                "current_equals_triton": False,
            },
        ],
        "all_current_triton_equal": False,
        "diagnostic_only": True,
    }
    assert (
        read_json(tmp_path / "matched" / "greedy" / "greedy_hash_comparison.json")
        == result
    )
