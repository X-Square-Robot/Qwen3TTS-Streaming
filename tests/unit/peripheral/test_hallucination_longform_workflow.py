from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tools.validation.hallucination.longform import (
    collection,
    commands,
    preflight,
    scoring,
)
from tools.validation.hallucination.longform.arm_types import CollectedRun
from tools.validation.hallucination.longform.artifacts import read_json, write_json
from tools.validation.hallucination.longform.audio_io import write_mono_wav
from tools.validation.hallucination.longform.collection import (
    DEFAULT_SEEDS,
    collect_arm_runs,
    initialize_experiment,
    logical_session_id,
)
from tools.validation.hallucination.longform.evidence_inventory import (
    runtime_evidence_inventory,
)
from tools.validation.hallucination.longform.models import (
    ArmKind,
    ReviewLabel,
    RunStatus,
)
from tools.validation.hallucination.longform.preflight import (
    exact_text_record,
    wheel_identity,
)
from tools.validation.hallucination.longform.reporting import generate_report
from tools.validation.hallucination.longform.review_package import (
    build_review_package,
    build_second_review,
    finalize_reviews,
)
from tools.validation.hallucination.longform.review_provenance import (
    verify_review_provenance,
)
from tools.validation.hallucination.longform.scoring import score_run
from tools.validation.hallucination.longform.segments import (
    extract_delivered_segments,
)
from tools.validation.hallucination.longform.telemetry import (
    summarize_engine_events,
)


def test_preflight_preserves_exact_text_and_verifies_wheel_hash(tmp_path: Path) -> None:
    raw_text = b"  first line\nsecond line \n"
    text_path = tmp_path / "verylong.txt"
    text_path.write_bytes(raw_text)

    record = exact_text_record(text_path)

    assert record["text"] == raw_text.decode("utf-8")
    assert record["bytes"] == len(raw_text)
    assert record["codepoints"] == len(raw_text.decode("utf-8"))
    assert record["has_trailing_newline"] is True
    assert record["sha256"] == hashlib.sha256(raw_text).hexdigest()

    wheel_path = tmp_path / "funasrnano-0.2.0a6-py3-none-any.whl"
    wheel_payload = b"an immutable test wheel payload"
    wheel_path.write_bytes(wheel_payload)
    expected_hash = hashlib.sha256(wheel_payload).hexdigest()

    identity = wheel_identity(wheel_path, expected_sha256=expected_hash.upper())

    assert identity == {
        "path": str(wheel_path.resolve()),
        "bytes": len(wheel_payload),
        "sha256": expected_hash,
    }
    with pytest.raises(RuntimeError, match="wheel hash mismatch"):
        wheel_identity(wheel_path, expected_sha256="0" * 64)


def test_runtime_evidence_inventory_hashes_files_without_following_links(
    tmp_path: Path,
) -> None:
    missing = runtime_evidence_inventory(tmp_path / "missing-experiment")
    assert missing["exists"] is False
    assert missing["audit_ready"] is False
    assert missing["entries"] == []

    output_root = tmp_path / "experiment"
    runtime_root = output_root / "runtime"
    (runtime_root / "nested").mkdir(parents=True)
    version_payload = b'{"triton":"2.62.0"}\n'
    log_payload = b"ready\n"
    (runtime_root / "versions.json").write_bytes(version_payload)
    (runtime_root / "nested" / "smoke.log").write_bytes(log_payload)
    (runtime_root / "model-link").symlink_to("/frozen/tts_orchestrator/2")

    inventory = runtime_evidence_inventory(output_root)

    assert inventory["audit_ready"] is True
    assert inventory["file_count"] == 2
    assert inventory["symlink_count"] == 1
    assert inventory["total_file_bytes"] == len(version_payload) + len(log_payload)
    by_path = {entry["path"]: entry for entry in inventory["entries"]}
    assert (
        by_path["runtime/versions.json"]["sha256"]
        == hashlib.sha256(version_payload).hexdigest()
    )
    assert by_path["runtime/model-link"] == {
        "path": "runtime/model-link",
        "kind": "symlink",
        "link_target": "/frozen/tts_orchestrator/2",
        "bytes": len(b"/frozen/tts_orchestrator/2"),
        "sha256": hashlib.sha256(b"/frozen/tts_orchestrator/2").hexdigest(),
        "hash_scope": "literal_symlink_target",
    }
    assert len(inventory["tree_sha256"]) == 64


def test_tokenizer_identity_uses_exact_bytes_without_special_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    module_path = package / "engine" / "frontend" / "spliter" / "tokenizer.py"
    module_path.parent.mkdir(parents=True)
    (package / "tokenizer").mkdir()
    module_path.write_text(
        """
class LightQwen3TTSTokenizer:
    def __init__(self, tokenizer_dir):
        self.tokenizer_dir = tokenizer_dir

    def encode_ids(self, text, add_special_tokens=True):
        assert text == "  甲。\\n"
        assert add_special_tokens is False
        return [7, 11, 7]
""".lstrip(),
        encoding="utf-8",
    )
    text_path = tmp_path / "verylong.txt"
    text_path.write_bytes("  甲。\n".encode("utf-8"))
    monkeypatch.setattr(
        preflight,
        "installed_distribution_version",
        lambda name: "0.test" if name == "tokenizers" else "unexpected",
    )

    identity = preflight.tokenizer_text_identity(package, text_path)

    canonical_ids = json.dumps([7, 11, 7], separators=(",", ":")).encode()
    assert identity == {
        "tokenizer_class": "LightQwen3TTSTokenizer",
        "tokenizers_version": "0.test",
        "add_special_tokens": False,
        "token_count": 3,
        "ids_sha256": hashlib.sha256(canonical_ids).hexdigest(),
        "ids": [7, 11, 7],
    }


class _RecordingArm:
    arm = ArmKind.CURRENT_HEAD

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def collect(
        self,
        text: str,
        *,
        session_id: str,
        seed: int = 0,
    ) -> CollectedRun:
        self.calls.append((text, session_id, seed))
        return CollectedRun(
            arm=self.arm,
            seed=seed,
            session_id=session_id,
            status=RunStatus.OK,
            samples=np.empty(0, dtype=np.float32),
            sample_rate=24_000,
            duration_s=0.0,
            terminal_event="done",
            eos_reason="natural_eos",
        )


def test_collection_runs_exactly_three_seeds_without_stripping_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_text = "  甲。\n乙。 \n"
    text_path = tmp_path / "verylong.txt"
    text_path.write_text(source_text, encoding="utf-8")
    output_root = tmp_path / "experiment"
    monkeypatch.setattr(collection, "runtime_identity", lambda: {"fake": True})

    manifest = initialize_experiment(
        output_root,
        text_file=text_path,
        identities={"test": "frozen"},
    )
    adapter = _RecordingArm()
    records = collect_arm_runs(output_root, adapter)

    assert manifest["seeds"] == list(DEFAULT_SEEDS)
    assert manifest["text"]["text"] == source_text
    assert len(adapter.calls) == 3
    assert [call[2] for call in adapter.calls] == list(DEFAULT_SEEDS)
    assert [call[1] for call in adapter.calls] == [
        logical_session_id(seed) for seed in DEFAULT_SEEDS
    ]
    assert all(call[0] == source_text for call in adapter.calls)
    assert len(records) == 3
    assert all(record["source_text"] == source_text for record in records)
    assert all(
        record["source_text_sha256"] == manifest["text"]["sha256"] for record in records
    )


def test_segment_end_and_final_text_progress_define_monotonic_audio_slices() -> None:
    events = [
        {
            "type": "text_progress",
            "segment_id": 0,
            "text": "暂态",
            "meta": {"output_sample_end": "10"},
        },
        {
            "type": "segment_end",
            "segment_id": 1,
            "text": "第二段。",
            "meta": {"eos_reason": "natural_eos"},
        },
        {
            "type": "text_progress",
            "segment_id": 1,
            "text": "第二段。",
            "meta": {
                "progress_final": "true",
                "output_sample_end": "80",
            },
        },
        {
            "type": "segment_end",
            "segment_id": 0,
            "text": "第一段。",
            "meta": {"eos_reason": "natural_eos"},
        },
        {
            "type": "text_progress",
            "segment_id": 0,
            "text": "第一段。",
            "meta": {
                "alignment_final": "1",
                "output_sample_end": "30",
            },
        },
    ]

    delivered = extract_delivered_segments(events, total_samples=100)
    samples = np.arange(100, dtype=np.float32)
    clips = [
        samples[item.output_sample_start : item.output_sample_end] for item in delivered
    ]

    assert [item.segment_id for item in delivered] == [0, 1]
    assert [item.text for item in delivered] == ["第一段。", "第二段。"]
    assert [
        (item.output_sample_start, item.output_sample_end) for item in delivered
    ] == [(0, 30), (30, 100)]
    assert clips[0].tolist() == list(range(30))
    assert clips[1].tolist() == list(range(30, 100))


def test_loop_abort_uses_last_actually_delivered_nonfinal_progress_boundary() -> None:
    events = [
        {"type": "segment_end", "segment_id": 0, "text": "循环段。"},
        {
            "type": "text_progress",
            "segment_id": 0,
            "meta": {
                "progress_final": "false",
                "output_sample_end": "10",
            },
        },
        {
            "type": "text_progress",
            "segment_id": 0,
            "meta": {
                "progress_final": "false",
                "output_sample_end": "20",
            },
        },
        {"type": "segment_end", "segment_id": 1, "text": "后续段。"},
        {
            "type": "text_progress",
            "segment_id": 1,
            "meta": {
                "alignment_final": "true",
                "output_sample_end": "80",
            },
        },
    ]

    delivered = extract_delivered_segments(events, total_samples=100)

    assert [item.segment_id for item in delivered] == [0, 1]
    assert [
        (item.output_sample_start, item.output_sample_end) for item in delivered
    ] == [(0, 20), (20, 100)]
    assert delivered[0].progress_meta["delivery_boundary_source"] == (
        "max_nonfinal_progress_output_end"
    )
    assert delivered[1].progress_meta["delivery_boundary_source"] == (
        "final_progress_output_end"
    )


def test_retry_telemetry_does_not_treat_retry_idx_zero_as_a_retry() -> None:
    summary = summarize_engine_events(
        [
            {
                "type": "segment_end",
                "segment_id": 0,
                "meta": {"retry_idx": "0", "loop_recovery_count": "3"},
            },
            {
                "type": "segment_end",
                "segment_id": 1,
                "meta": {"retry_idx": "1"},
            },
        ]
    )

    assert len(summary["retry_events"]) == 1
    assert summary["retry_events"][0]["segment_id"] == 1


def test_sentence_transcript_slices_overlapping_asr_segment_by_character_time() -> None:
    segments = [
        {"text": "甲，乙丙丁。", "start_ms": 0, "end_ms": 4_000},
        {"text": "戊己", "start_ms": 4_000, "end_ms": 6_000},
    ]

    transcript = scoring._overlapping_transcript(segments, 1_000, 5_000)

    assert transcript == "乙丙丁戊"


class _FreshFakeFunASRClient:
    instances: list[_FreshFakeFunASRClient] = []

    def __init__(self, uri: str, **options: Any) -> None:
        self.uri = uri
        self.options = options
        self.paths: list[str] = []
        self.entered = False
        self.exited = False
        self.__class__.instances.append(self)

    async def __aenter__(self) -> _FreshFakeFunASRClient:
        self.entered = True
        return self

    async def __aexit__(self, *_args: Any) -> None:
        self.exited = True

    async def transcribe_file(
        self,
        path: str,
        **options: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        self.paths.append(path)
        assert options == {
            "chunk_ms": 960,
            "delivery_mode": "offline",
            "pacing": "none",
        }
        yield {
            "type": "segment_final",
            "segment": {"text": "甲", "start_ms": 0, "end_ms": 31_000},
        }
        yield {"type": "stream_done", "reason": "complete"}


def test_scoring_does_not_skip_over_30_seconds_and_opens_one_connection_per_wav(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FreshFakeFunASRClient.instances = []
    output_root = tmp_path / "experiment"
    run_dir = output_root / "arms" / ArmKind.CURRENT_HEAD.value / "seed_0001"
    wav_path = run_dir / "full.wav"
    samples = np.zeros(31_000, dtype=np.float32)
    write_mono_wav(wav_path, samples, 1_000)
    monkeypatch.setattr(
        scoring,
        "acoustic_diagnostics",
        lambda audio, sample_rate: {
            "sample_count": int(np.asarray(audio).size),
            "sample_rate": sample_rate,
            "diagnostic_only": True,
        },
    )
    record = {
        "_record_path": str((run_dir / "run.json").relative_to(output_root)),
        "arm": ArmKind.CURRENT_HEAD.value,
        "seed": 1,
        "session_id": "private-long-session",
        "status": RunStatus.OK.value,
        "source_text": "甲。",
        "events": [
            {"type": "segment_end", "segment_id": 0, "text": "甲。"},
            {
                "type": "text_progress",
                "segment_id": 0,
                "text": "甲。",
                "meta": {
                    "alignment_final": "true",
                    "output_sample_end": str(samples.size),
                },
            },
        ],
        "artifacts": {"wav": str(wav_path.relative_to(output_root))},
    }
    write_json(run_dir / "run.json", record)

    observations = score_run(
        output_root,
        record,
        client_class=_FreshFakeFunASRClient,
        asr_url="wss://asr.example.test/infer/test/v1/ws",
    )

    # One complete 31-second WAV plus its one delivered-segment WAV.
    assert len(_FreshFakeFunASRClient.instances) == 2
    assert all(
        instance.entered and instance.exited
        for instance in _FreshFakeFunASRClient.instances
    )
    assert all(
        len(instance.paths) == 1 for instance in _FreshFakeFunASRClient.instances
    )
    assert (
        len({instance.paths[0] for instance in _FreshFakeFunASRClient.instances}) == 2
    )
    assert all(
        instance.options["partial_mode"] == "off"
        for instance in _FreshFakeFunASRClient.instances
    )
    assert all(
        instance.options["hotwords"] == []
        for instance in _FreshFakeFunASRClient.instances
    )
    scoring_record = read_json(run_dir / "scoring.json")
    assert scoring_record["full_asr"]["status"] == "ok"
    assert scoring_record["full_asr"]["stream_done"]["reason"] == "complete"
    assert observations[0]["status"] == RunStatus.REVIEW_PENDING.value


class _ScriptedFunASRClient:
    scripts: list[list[dict[str, Any]] | BaseException] = []
    instances = 0

    def __init__(self, _uri: str, **_options: Any) -> None:
        type(self).instances += 1
        self.script = type(self).scripts.pop(0)

    async def __aenter__(self) -> _ScriptedFunASRClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def transcribe_file(
        self,
        _path: str,
        **_options: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        if isinstance(self.script, BaseException):
            raise self.script
        for event in self.script:
            yield event


def _ok_asr_events(text: str = "甲") -> list[dict[str, Any]]:
    return [
        {
            "type": "segment_final",
            "segment": {"text": text, "start_ms": 0, "end_ms": 100},
        },
        {"type": "stream_done", "reason": "complete"},
    ]


def _write_scoring_run(
    output_root: Path,
    *,
    arm: ArmKind = ArmKind.CURRENT_HEAD,
    seed: int = 1,
    status: RunStatus = RunStatus.OK,
    text: str = "甲。",
    events: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path]:
    run_dir = output_root / "arms" / arm.value / f"seed_{seed:04d}"
    wav_path = run_dir / "full.wav"
    samples = np.zeros(200, dtype=np.float32)
    write_mono_wav(wav_path, samples, 1_000)
    record = {
        "_record_path": str((run_dir / "run.json").relative_to(output_root)),
        "arm": arm.value,
        "seed": seed,
        "session_id": f"formal-seed-{seed}",
        "status": status.value,
        "source_text": text,
        "events": list(events or []),
        "error": "partial TTS" if status is not RunStatus.OK else None,
        "artifacts": {"wav": str(wav_path.relative_to(output_root))},
    }
    write_json(run_dir / "run.json", record)
    return record, run_dir


def _quiet_acoustic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scoring,
        "acoustic_diagnostics",
        lambda audio, sample_rate: {
            "sample_count": int(np.asarray(audio).size),
            "sample_rate": sample_rate,
            "diagnostic_only": True,
        },
    )


def test_official_fallback_uses_a_fresh_connection_for_each_wav(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _quiet_acoustic(monkeypatch)
    _ScriptedFunASRClient.instances = 0
    _ScriptedFunASRClient.scripts = [
        _ok_asr_events(),
        _ok_asr_events(),
        _ok_asr_events(),
    ]
    output_root = tmp_path / "experiment"
    record, run_dir = _write_scoring_run(
        output_root,
        arm=ArmKind.PYTORCH_0818,
    )

    score_run(
        output_root,
        record,
        client_class=_ScriptedFunASRClient,
        asr_url="wss://asr.example.test/v1/ws",
    )

    assert _ScriptedFunASRClient.instances == 2
    persisted = read_json(run_dir / "scoring.json")
    assert persisted["delivered_segments"][0]["asr_provenance"]["source"] == (
        "fresh_connection"
    )
    assert (
        persisted["delivered_segments"][0]["asr_provenance"]["origin_source"]
        == "fresh_connection"
    )
    assert (run_dir / "full.wav.asr.json").is_file()
    assert (run_dir / "segments" / "segment_000.wav.asr.json").is_file()

    # Derived files are rebuilt, while successful exact-input ASR is recovered.
    (run_dir / "scoring.json").write_text("{interrupted", encoding="utf-8")
    score_run(
        output_root,
        record,
        client_class=_ScriptedFunASRClient,
        asr_url="wss://asr.example.test/v1/ws",
    )
    assert _ScriptedFunASRClient.instances == 2
    assert (
        read_json(run_dir / "scoring.json")["full_asr_provenance"]["cache_hit"] is True
    )

    # A legacy sidecar copied from the full WAV is not valid evidence for the
    # segment WAV, even though both files contain the same bytes.
    segment_sidecar = run_dir / "segments" / "segment_000.wav.asr.json"
    legacy = read_json(segment_sidecar)
    legacy["provenance"] = {"source": "reused_full_wav"}
    write_json(segment_sidecar, legacy)
    score_run(
        output_root,
        record,
        client_class=_ScriptedFunASRClient,
        asr_url="wss://asr.example.test/v1/ws",
    )
    assert _ScriptedFunASRClient.instances == 3
    assert read_json(segment_sidecar)["provenance"] == {"source": "fresh_connection"}


def test_punctuation_only_delivered_segment_has_no_cer_denominator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _quiet_acoustic(monkeypatch)
    _ScriptedFunASRClient.instances = 0
    _ScriptedFunASRClient.scripts = [
        _ok_asr_events("甲"),
        _ok_asr_events("。"),
    ]
    output_root = tmp_path / "experiment"
    events = [
        {"type": "segment_end", "segment_id": 0, "text": "。"},
        {
            "type": "text_progress",
            "segment_id": 0,
            "meta": {"alignment_final": "true", "output_sample_end": "200"},
        },
    ]
    record, run_dir = _write_scoring_run(output_root, events=events)

    score_run(
        output_root,
        record,
        client_class=_ScriptedFunASRClient,
        asr_url="wss://asr.example.test/v1/ws",
    )

    segment = read_json(run_dir / "scoring.json")["delivered_segments"][0]
    assert segment["text"] == "。"
    assert segment["character_errors"] is None
    assert segment["character_errors_unavailable_reason"] == (
        "empty_normalized_reference"
    )


def test_error_sidecar_retries_without_blocking_audible_human_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _quiet_acoustic(monkeypatch)
    _ScriptedFunASRClient.instances = 0
    _ScriptedFunASRClient.scripts = [
        [],
        _ok_asr_events(),
        _ok_asr_events(),
    ]
    output_root = tmp_path / "experiment"
    record, run_dir = _write_scoring_run(output_root)

    failed = score_run(
        output_root,
        record,
        client_class=_ScriptedFunASRClient,
        asr_url="wss://asr.example.test/v1/ws",
    )
    failed_scoring = read_json(run_dir / "scoring.json")

    assert _ScriptedFunASRClient.instances == 2
    assert failed_scoring["status"] == RunStatus.ASR_FAILED.value
    assert failed_scoring["full_character_errors"] is None
    assert failed[0]["status"] == RunStatus.REVIEW_PENDING.value
    assert failed[0]["valid_for_review"] is True
    assert failed[0]["asr_status"] == "error"
    assert failed[0]["character_errors"] is None
    assert failed[0]["diagnostic_warnings"] == ["full_wav_asr_failed"]

    recovered = score_run(
        output_root,
        record,
        client_class=_ScriptedFunASRClient,
        asr_url="wss://asr.example.test/v1/ws",
    )

    assert _ScriptedFunASRClient.instances == 3
    assert recovered[0]["status"] == RunStatus.REVIEW_PENDING.value
    assert recovered[0]["valid_for_review"] is True
    assert read_json(run_dir / "scoring.json")["status"] == RunStatus.OK.value


def test_partial_tts_wav_is_not_marked_review_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _quiet_acoustic(monkeypatch)
    _ScriptedFunASRClient.instances = 0
    _ScriptedFunASRClient.scripts = [
        _ok_asr_events(),
        _ok_asr_events(),
    ]
    output_root = tmp_path / "experiment"
    record, run_dir = _write_scoring_run(
        output_root,
        status=RunStatus.TTS_FAILED,
    )

    observations = score_run(
        output_root,
        record,
        client_class=_ScriptedFunASRClient,
        asr_url="wss://asr.example.test/v1/ws",
    )

    assert observations[0]["status"] == RunStatus.TTS_FAILED.value
    assert observations[0]["tts_complete"] is False
    assert observations[0]["valid_for_review"] is False
    assert read_json(run_dir / "scoring.json")["status"] == (RunStatus.TTS_FAILED.value)


def test_interrupted_segment_scoring_resumes_only_missing_wav(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _quiet_acoustic(monkeypatch)

    class SimulatedInterruption(BaseException):
        pass

    calls: list[str] = []
    outcomes: list[dict[str, Any] | BaseException] = [
        {
            "status": "ok",
            "transcript": "甲乙",
            "segments": [
                {"text": "甲", "start_ms": 0, "end_ms": 100},
                {"text": "乙", "start_ms": 100, "end_ms": 200},
            ],
            "stream_done": {"reason": "complete"},
        },
        {
            "status": "ok",
            "transcript": "甲",
            "segments": [{"text": "甲", "start_ms": 0, "end_ms": 100}],
            "stream_done": {"reason": "complete"},
        },
        SimulatedInterruption(),
        {
            "status": "ok",
            "transcript": "乙",
            "segments": [{"text": "乙", "start_ms": 0, "end_ms": 100}],
            "stream_done": {"reason": "complete"},
        },
    ]

    async def scripted_transcribe(
        _client_class: Any,
        wav_path: Path,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(wav_path.name)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(scoring, "_transcribe", scripted_transcribe)
    events = [
        {"type": "segment_end", "segment_id": 0, "text": "甲。"},
        {
            "type": "text_progress",
            "segment_id": 0,
            "meta": {"alignment_final": "true", "output_sample_end": "100"},
        },
        {"type": "segment_end", "segment_id": 1, "text": "乙。"},
        {
            "type": "text_progress",
            "segment_id": 1,
            "meta": {"alignment_final": "true", "output_sample_end": "200"},
        },
    ]
    output_root = tmp_path / "experiment"
    record, run_dir = _write_scoring_run(
        output_root,
        text="甲。乙。",
        events=events,
    )

    with pytest.raises(SimulatedInterruption):
        score_run(
            output_root,
            record,
            client_class=object,
            asr_url="wss://asr.example.test/v1/ws",
        )

    assert calls == ["full.wav", "segment_000.wav", "segment_001.wav"]
    assert (run_dir / "full.wav.asr.json").is_file()
    assert (run_dir / "segments" / "segment_000.wav.asr.json").is_file()
    assert not (run_dir / "segments" / "segment_001.wav.asr.json").exists()

    score_run(
        output_root,
        record,
        client_class=object,
        asr_url="wss://asr.example.test/v1/ws",
    )

    assert calls == [
        "full.wav",
        "segment_000.wav",
        "segment_001.wav",
        "segment_001.wav",
    ]
    assert not outcomes


def test_corrupt_or_request_mismatched_sidecar_is_not_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _quiet_acoustic(monkeypatch)
    _ScriptedFunASRClient.instances = 0
    _ScriptedFunASRClient.scripts = [
        _ok_asr_events(),
        _ok_asr_events(),
        _ok_asr_events(),
        _ok_asr_events(),
        _ok_asr_events(),
    ]
    output_root = tmp_path / "experiment"
    record, run_dir = _write_scoring_run(output_root)

    score_run(
        output_root,
        record,
        client_class=_ScriptedFunASRClient,
        asr_url="wss://asr.example.test/v1/ws",
    )
    (run_dir / "full.wav.asr.json").write_text("{broken", encoding="utf-8")
    score_run(
        output_root,
        record,
        client_class=_ScriptedFunASRClient,
        asr_url="wss://asr.example.test/v1/ws",
    )
    score_run(
        output_root,
        record,
        client_class=_ScriptedFunASRClient,
        asr_url="wss://asr.example.test/v1/ws",
        language="English",
    )

    assert _ScriptedFunASRClient.instances == 5


def test_scoring_summary_separates_asr_degradation_from_review_blockers(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    seeds = [11, 22, 33]
    write_json(
        output_root / "manifest.json",
        {"seeds": seeds, "text": {"text": "甲。"}},
    )
    run_dirs: list[Path] = []
    for arm in ArmKind:
        for seed in seeds:
            record, run_dir = _write_scoring_run(
                output_root,
                arm=arm,
                seed=seed,
            )
            run_dirs.append(run_dir)
            write_json(
                run_dir / "scoring.json",
                {
                    "status": RunStatus.OK.value,
                    "full_wav": str((run_dir / "full.wav").relative_to(output_root)),
                    "full_asr": {
                        "status": "ok",
                        "transcript": "甲",
                        "segments": [],
                        "stream_done": {},
                    },
                    "full_asr_provenance": {"origin_source": "fresh_connection"},
                    "delivered_segments": [
                        {
                            "wav": "segment_000.wav",
                            "asr": {"status": "ok", "stream_done": {}},
                            "asr_provenance": {"origin_source": "fresh_connection"},
                        }
                    ],
                },
            )
            write_json(
                run_dir / "sentence_observations.json",
                {
                    "observations": [
                        {
                            "arm": arm.value,
                            "seed": seed,
                            "sentence_ordinal": 1,
                            "valid_for_review": True,
                            "record": record["_record_path"],
                        }
                    ]
                },
            )

    ready = scoring.scoring_completion_summary(output_root)

    assert ready["run_count"] == ready["expected_run_count"] == 9
    assert ready["observation_count"] == ready["expected_observation_count"] == 9
    assert ready["asr_wav_count"] == 18
    assert ready["fresh_connection_origin_wav_count"] == 18
    assert ready["ready_for_review"] is True
    assert ready["blocking_reasons"] == []
    assert ready["diagnostic_warnings"] == []
    assert ready["diagnostic_degraded"] is False

    broken = read_json(run_dirs[0] / "scoring.json")
    broken["delivered_segments"][0]["asr_provenance"] = {
        "origin_source": "reused_full_wav"
    }
    write_json(run_dirs[0] / "scoring.json", broken)

    invalid_provenance = scoring.scoring_completion_summary(output_root)
    assert invalid_provenance["invalid_asr_provenance_wav_count"] == 1
    assert invalid_provenance["ready_for_review"] is True
    assert invalid_provenance["blocking_reasons"] == []
    assert "asr_wav_without_fresh_connection_origin" in invalid_provenance[
        "diagnostic_warnings"
    ]

    broken["delivered_segments"][0]["asr_provenance"] = {
        "origin_source": "fresh_connection"
    }
    broken["delivered_segments"][0]["asr"] = {
        "status": "error",
        "error": "timeout",
    }
    write_json(run_dirs[0] / "scoring.json", broken)

    incomplete = scoring.scoring_completion_summary(output_root)
    assert incomplete["segment_asr_failure_count"] == 1
    assert incomplete["ready_for_review"] is True
    assert incomplete["reasons"] == []
    assert "segment_asr_failed_wavs" in incomplete["diagnostic_warnings"]

    broken["full_asr"] = {"status": "error", "error": "timeout"}
    write_json(run_dirs[0] / "scoring.json", broken)

    full_failure = scoring.scoring_completion_summary(output_root)
    assert full_failure["full_asr_failure_count"] == 1
    assert full_failure["ready_for_review"] is True
    assert "full_asr_failed_runs" in full_failure["diagnostic_warnings"]

    failed_run = read_json(run_dirs[0] / "run.json")
    failed_run["status"] = RunStatus.TTS_FAILED.value
    write_json(run_dirs[0] / "run.json", failed_run)

    tts_failure = scoring.scoring_completion_summary(output_root)
    assert tts_failure["ready_for_review"] is False
    assert "tts_failed_runs" in tts_failure["blocking_reasons"]


def test_command_score_returns_nonzero_without_recording_complete_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    asr_url = "wss://asr.example.test/v1/ws"
    write_json(
        output_root / "manifest.json",
        {
            "identities": {
                "asr": {
                    "websocket_url": asr_url,
                    "wheel": {"sha256": "frozen-wheel"},
                }
            }
        },
    )
    monkeypatch.setattr(
        commands,
        "wheel_identity",
        lambda _path: {"sha256": "frozen-wheel"},
    )
    monkeypatch.setattr(
        commands,
        "installed_distribution_version",
        lambda _name: "0.2.0a6",
    )
    monkeypatch.setattr(
        commands,
        "fetch_asr_capabilities",
        lambda _url: {"version": "0.2.0a6"},
    )
    monkeypatch.setattr(commands, "import_funasr_client", lambda _src: object)
    monkeypatch.setattr(commands, "score_all_runs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        commands,
        "scoring_completion_summary",
        lambda _root: {
            "ready_for_review": False,
            "blocking_reasons": ["unscored_runs"],
            "diagnostic_warnings": ["full_asr_failed_runs"],
            "reasons": ["unscored_runs"],
        },
    )
    phases: list[str] = []
    monkeypatch.setattr(
        commands,
        "record_phase",
        lambda _root, phase, **_kwargs: phases.append(phase),
    )
    monkeypatch.setattr(commands, "_print", lambda _payload: None)
    args = SimpleNamespace(
        output_dir=output_root,
        asr_url=asr_url,
        funasr_wheel=tmp_path / "sdk.whl",
        funasr_client_src=None,
        language="中文",
        chunk_ms=960,
    )

    assert commands.command_score(args) == 1
    assert phases == []


def _write_review_observations(
    output_root: Path,
    observations: list[dict[str, Any]],
) -> None:
    for index, observation in enumerate(observations, start=1):
        clip = output_root / "source_audio" / f"clip-{index:03d}.wav"
        context = output_root / "source_audio" / f"context-{index:03d}.wav"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(f"clip-{index}".encode("ascii"))
        context.write_bytes(f"context-{index}".encode("ascii"))
        observation["clip"] = str(clip.relative_to(output_root))
        observation["context_clip"] = str(context.relative_to(output_root))
    write_json(
        output_root / "sentence_observations.json", {"observations": observations}
    )


def _observation(
    ordinal: int,
    *,
    arm: ArmKind = ArmKind.CURRENT_HEAD,
    seed: int = 987_654_321,
    tts_status: RunStatus = RunStatus.OK,
) -> dict[str, Any]:
    return {
        "arm": arm.value,
        "seed": seed,
        "session_id": f"SID-TOKEN-ALPHA-{ordinal}",
        "sentence_ordinal": ordinal,
        "sentence_id": f"sentence-{ordinal:03d}",
        "reference_text": f"参考句{ordinal}。",
        "previous_reference": "上文。",
        "next_reference": "下文。",
        "status": RunStatus.REVIEW_PENDING.value,
        "tts_run_status": tts_status.value,
        "valid_for_review": tts_status is RunStatus.OK,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0]) if rows else ["blind_id", "label", "notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fill_labels(path: Path, labels: Mapping[str, ReviewLabel | str]) -> None:
    rows = _read_csv(path)
    for row in rows:
        selected = labels[row["blind_id"]]
        row["label"] = selected.value if isinstance(selected, ReviewLabel) else selected
    _write_csv_rows(path, rows)


def _round1_submission(output_root: Path) -> Path:
    """Copy the sealed public template before simulating a reviewer edit."""

    source = output_root / "review" / "public" / "review_round1.csv"
    target = output_root / "review" / "submissions" / "review_round1_filled.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _prepare_second_review(
    output_root: Path,
    round1_csv: Path,
    *,
    selection_seed: int,
) -> dict[str, Any]:
    return build_second_review(
        output_root,
        round1_csv,
        reviewer1_id="reviewer-alpha",
        reviewer2_id="reviewer-beta",
        selection_seed=selection_seed,
    )


def test_public_blind_package_does_not_leak_arm_session_seed_or_source_paths(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    observations = [
        _observation(1, arm=ArmKind.CURRENT_HEAD),
        _observation(2, arm=ArmKind.TRITON_0818),
        _observation(3, arm=ArmKind.PYTORCH_0818),
    ]
    _write_review_observations(output_root, observations)

    result = build_review_package(output_root, review_seed=73)

    assert result["sample_count"] == 3
    public_root = output_root / "review" / "public"
    public_payload = b"\n".join(
        path.read_bytes() for path in sorted(public_root.rglob("*")) if path.is_file()
    )
    for secret in (
        ArmKind.CURRENT_HEAD.value,
        ArmKind.TRITON_0818.value,
        ArmKind.PYTORCH_0818.value,
        "SID-TOKEN-ALPHA",
        "987654321",
        "source_audio",
    ):
        assert secret.encode("utf-8") not in public_payload
    public_manifest = read_json(public_root / "manifest.json")
    assert public_manifest["arm_blind"] is True
    assert all(
        set(row)
        == {
            "audio",
            "blind_id",
            "context_audio",
            "label",
            "next_reference",
            "notes",
            "previous_reference",
            "reference_text",
        }
        for row in public_manifest["rows"]
    )


def test_review_index_is_offline_keyboard_ready_and_javascript_is_valid(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    _write_review_observations(
        output_root,
        [_observation(ordinal) for ordinal in range(1, 4)],
    )
    build_review_package(output_root, review_seed=79)

    index = (output_root / "review" / "public" / "index.html").read_text(
        encoding="utf-8"
    )
    for marker in (
        'id="progress-track"',
        'id="import-file"',
        'id="export-progress"',
        'id="export-final" type="button" disabled',
        'id="previous"',
        'id="next"',
        'id="next-incomplete"',
        "localStorage.getItem(storageKey)",
        "localStorage.setItem(storageKey",
        "review_round1_progress.csv",
        "review_round1_filled.csv",
        "event.code === 'Space'",
        "event.key === 'ArrowLeft'",
        "@media(prefers-reduced-motion:reduce)",
        ":focus-visible",
    ):
        assert marker in index
    assert all(label.value in index for label in ReviewLabel)
    assert "https://" not in index
    assert "http://" not in index

    scripts = re.findall(
        r'<script(?![^>]*type="application/json")[^>]*>(.*?)</script>',
        index,
        flags=re.DOTALL,
    )
    assert len(scripts) == 1
    executable_script = scripts[0]
    # Regression: Python used to turn ``'\n'`` into a physical newline inside
    # a JavaScript single-quoted string, making the generated page unparseable.
    assert "join('\n')" not in executable_script
    assert r"join('\r\n')" in executable_script
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for generated JavaScript syntax checking")
    checked = subprocess.run(
        [node, "--check"],
        input=executable_script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr


def test_formal_review_package_rejects_an_incomplete_three_arm_grid(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    write_json(
        output_root / "manifest.json",
        {
            "seeds": [11, 22, 33],
            "text": {"text": "甲。乙。"},
        },
    )
    _write_review_observations(output_root, [_observation(1, seed=11)])

    with pytest.raises(RuntimeError, match="incomplete blind package"):
        build_review_package(output_root, review_seed=73)


def test_formal_review_package_requires_successful_scoring_gate(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    seeds = [11, 22, 33]
    write_json(
        output_root / "manifest.json",
        {"seeds": seeds, "text": {"text": "甲。"}},
    )
    observations = [
        _observation(1, arm=arm, seed=seed) for arm in ArmKind for seed in seeds
    ]
    _write_review_observations(output_root, observations)
    write_json(
        output_root / "scoring_summary.json",
        {
            "ready_for_review": False,
            "reasons": ["full_asr_failed_runs"],
        },
    )

    with pytest.raises(RuntimeError, match="scoring is incomplete"):
        build_review_package(output_root, review_seed=73)

    write_json(
        output_root / "scoring_summary.json",
        {"ready_for_review": True, "reasons": []},
    )
    result = build_review_package(output_root, review_seed=73)
    assert result["sample_count"] == 9


def test_second_review_selects_all_severe_and_unscorable_plus_ten_percent_negatives(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    _write_review_observations(
        output_root,
        [_observation(ordinal) for ordinal in range(1, 21)],
    )
    build_review_package(output_root, review_seed=17)
    round1 = _round1_submission(output_root)
    rows = _read_csv(round1)
    severe_id = rows[0]["blind_id"]
    unscorable_id = rows[1]["blind_id"]
    labels = {row["blind_id"]: ReviewLabel.OK for row in rows}
    labels[severe_id] = ReviewLabel.SINGLE_UNIT_LOOP
    labels[unscorable_id] = ReviewLabel.UNSCORABLE
    _fill_labels(round1, labels)

    result = _prepare_second_review(output_root, round1, selection_seed=29)

    selected_ids = {row["blind_id"] for row in _read_csv(Path(result["path"]))}
    assert result == {
        "selected": 4,
        "mandatory": 2,
        "random_negative": 2,
        "path": str(output_root / "review" / "public" / "review_round2.csv"),
    }
    assert {severe_id, unscorable_id} <= selected_ids
    assert len(selected_ids - {severe_id, unscorable_id}) == 2


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_finalize_requires_round2_ids_to_exactly_match_frozen_selection(
    tmp_path: Path,
    mutation: str,
) -> None:
    output_root = tmp_path / "experiment"
    _write_review_observations(
        output_root,
        [_observation(ordinal) for ordinal in range(1, 21)],
    )
    build_review_package(output_root, review_seed=31)
    round1 = _round1_submission(output_root)
    round1_rows = _read_csv(round1)
    labels = {row["blind_id"]: ReviewLabel.OK for row in round1_rows}
    labels[round1_rows[0]["blind_id"]] = ReviewLabel.UNSCORABLE
    _fill_labels(round1, labels)
    second = _prepare_second_review(output_root, round1, selection_seed=37)
    round2 = Path(second["path"])
    _fill_labels(round2, labels)

    round2_rows = _read_csv(round2)
    if mutation == "missing":
        round2_rows.pop()
    else:
        selected = {row["blind_id"] for row in round2_rows}
        extra = next(row for row in round1_rows if row["blind_id"] not in selected)
        extra["label"] = ReviewLabel.OK.value
        round2_rows.append(extra)
    _write_csv_rows(round2, round2_rows)

    with pytest.raises(ValueError, match="do not match frozen selection"):
        finalize_reviews(output_root, round1, round2)


def test_finalize_rejects_frozen_selection_that_omits_mandatory_review(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    _write_review_observations(
        output_root,
        [_observation(ordinal) for ordinal in range(1, 21)],
    )
    build_review_package(output_root, review_seed=41)
    round1 = _round1_submission(output_root)
    round1_rows = _read_csv(round1)
    mandatory_id = round1_rows[0]["blind_id"]
    labels = {row["blind_id"]: ReviewLabel.OK for row in round1_rows}
    labels[mandatory_id] = ReviewLabel.UNSCORABLE
    _fill_labels(round1, labels)
    second = _prepare_second_review(output_root, round1, selection_seed=43)
    round2 = Path(second["path"])
    _fill_labels(round2, labels)

    selection_path = output_root / "review" / "private" / "round2_selection.json"
    selection = read_json(selection_path)
    selection["blind_ids"].remove(mandatory_id)
    write_json(selection_path, selection)
    _write_csv_rows(
        round2,
        [row for row in _read_csv(round2) if row["blind_id"] != mandatory_id],
    )

    with pytest.raises(ValueError, match="omits mandatory non-OK/uncertain"):
        finalize_reviews(output_root, round1, round2)


def test_private_key_seals_blind_audio_and_public_controls(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    _write_review_observations(
        output_root,
        [_observation(ordinal) for ordinal in range(1, 5)],
    )
    build_review_package(output_root, review_seed=47)
    key = read_json(output_root / "review" / "private" / "key.json")
    public_root = output_root / "review" / "public"
    public = {
        row["blind_id"]: row for row in read_json(public_root / "manifest.json")["rows"]
    }
    assert key["schema_version"] == 3
    assert key["public_control_sha256"] == {
        filename: hashlib.sha256((public_root / filename).read_bytes()).hexdigest()
        for filename in (
            "manifest.json",
            "review_round1.csv",
            "README.md",
            "index.html",
        )
    }
    for item in key["rows"]:
        public_row = public[item["blind_id"]]
        assert (
            item["audio_sha256"]
            == hashlib.sha256(
                (public_root / public_row["audio"]).read_bytes()
            ).hexdigest()
        )
        assert (
            item["context_audio_sha256"]
            == hashlib.sha256(
                (public_root / public_row["context_audio"]).read_bytes()
            ).hexdigest()
        )

    round1 = _round1_submission(output_root)
    labels = {row["blind_id"]: ReviewLabel.OK for row in _read_csv(round1)}
    _fill_labels(round1, labels)
    second = _prepare_second_review(output_root, round1, selection_seed=53)
    round2 = Path(second["path"])
    _fill_labels(round2, labels)
    first = key["rows"][0]
    (public_root / public[first["blind_id"]]["audio"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="blind review asset hash mismatch"):
        finalize_reviews(output_root, round1, round2)


def test_private_key_rejects_a_tampered_review_index(tmp_path: Path) -> None:
    output_root = tmp_path / "experiment"
    _write_review_observations(
        output_root,
        [_observation(ordinal) for ordinal in range(1, 5)],
    )
    build_review_package(output_root, review_seed=53)
    round1 = _round1_submission(output_root)
    labels = {row["blind_id"]: ReviewLabel.OK for row in _read_csv(round1)}
    _fill_labels(round1, labels)
    index = output_root / "review" / "public" / "index.html"
    index.write_bytes(index.read_bytes() + b"<!-- tampered -->")

    with pytest.raises(
        ValueError, match=r"control file hash mismatch: file=index\.html"
    ):
        _prepare_second_review(output_root, round1, selection_seed=61)


@pytest.mark.parametrize("legacy_schema", [1, 2])
def test_legacy_private_keys_remain_readable(
    tmp_path: Path,
    legacy_schema: int,
) -> None:
    output_root = tmp_path / "experiment"
    _write_review_observations(
        output_root,
        [_observation(ordinal) for ordinal in range(1, 5)],
    )
    build_review_package(output_root, review_seed=59)
    key_path = output_root / "review" / "private" / "key.json"
    key = read_json(key_path)
    key["schema_version"] = legacy_schema
    key.pop("public_control_sha256")
    if legacy_schema == 1:
        for item in key["rows"]:
            item.pop("audio_sha256")
            item.pop("context_audio_sha256")
    write_json(key_path, key)

    round1 = _round1_submission(output_root)
    labels = {row["blind_id"]: ReviewLabel.OK for row in _read_csv(round1)}
    _fill_labels(round1, labels)
    second = _prepare_second_review(output_root, round1, selection_seed=61)
    round2 = Path(second["path"])
    _fill_labels(round2, labels)

    assert len(finalize_reviews(output_root, round1, round2)) == 4


def test_unscorable_and_tts_failed_reviews_are_invalid_not_clean(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    observations = [
        _observation(1, tts_status=RunStatus.TTS_FAILED),
        _observation(2),
        _observation(3),
        _observation(4),
    ]
    _write_review_observations(output_root, observations)
    build_review_package(output_root, review_seed=41)
    private_rows = read_json(output_root / "review" / "private" / "key.json")["rows"]
    ordinal_by_id = {
        row["blind_id"]: row["observation"]["sentence_ordinal"] for row in private_rows
    }
    label_by_ordinal = {
        1: ReviewLabel.OK,
        2: ReviewLabel.UNSCORABLE,
        3: ReviewLabel.ABNORMAL_NOISE,
        4: ReviewLabel.OK,
    }
    labels = {
        blind_id: label_by_ordinal[ordinal]
        for blind_id, ordinal in ordinal_by_id.items()
    }
    round1 = _round1_submission(output_root)
    _fill_labels(round1, labels)
    second = _prepare_second_review(output_root, round1, selection_seed=43)
    round2 = Path(second["path"])
    _fill_labels(round2, labels)

    final = finalize_reviews(output_root, round1, round2)
    by_ordinal = {int(row["sentence_ordinal"]): row for row in final}

    assert by_ordinal[1]["tts_run_status"] == RunStatus.TTS_FAILED.value
    assert by_ordinal[1]["review_label"] == ReviewLabel.OK.value
    assert by_ordinal[1]["valid_for_rate"] is False
    assert by_ordinal[1]["status"] == RunStatus.INVALID.value
    assert by_ordinal[2]["review_label"] == ReviewLabel.UNSCORABLE.value
    assert by_ordinal[2]["valid_for_rate"] is False
    assert by_ordinal[2]["status"] == RunStatus.INVALID.value
    assert by_ordinal[3]["valid_for_rate"] is True
    assert by_ordinal[3]["severe_hallucination"] is True


def _reference_groups() -> list[dict[str, Any]]:
    boundaries = ((1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36), (37, 38))
    return [
        {
            "group_index": index,
            "sentence_ordinals": list(range(start, end + 1)),
        }
        for index, (start, end) in enumerate(boundaries, start=1)
    ]


def _report_reference_text() -> str:
    return "".join(f"第{ordinal}句。" for ordinal in range(1, 39))


def test_report_accounts_for_342_observations_114_per_arm_and_emits_large_gap_gate(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    seeds = [11, 22, 33]
    write_json(
        output_root / "manifest.json",
        {"seeds": seeds, "text": {"text": _report_reference_text()}},
    )
    write_json(
        output_root / "reference_groups.json",
        {"groups": _reference_groups()},
    )
    rows: list[dict[str, Any]] = []
    positive_cutoff = {
        ArmKind.CURRENT_HEAD: 15,
        ArmKind.TRITON_0818: 1,
        ArmKind.PYTORCH_0818: 1,
    }
    for arm, cutoff in positive_cutoff.items():
        for seed in seeds:
            for ordinal in range(1, 39):
                severe = ordinal <= cutoff
                rows.append(
                    {
                        "arm": arm.value,
                        "seed": seed,
                        "sentence_ordinal": ordinal,
                        "sentence_id": f"sentence-{ordinal:03d}",
                        "review_label": (
                            ReviewLabel.SINGLE_UNIT_LOOP.value
                            if severe
                            else ReviewLabel.OK.value
                        ),
                        "severe_hallucination": severe,
                        "valid_for_rate": True,
                        "status": RunStatus.REVIEWED.value,
                    }
                )
    assert len(rows) == 342
    write_json(
        output_root / "review" / "private" / "final_labels.json",
        {"rows": rows},
    )
    runtime_payload = b'{"image_id":"sha256:fixed"}\n'
    runtime_path = output_root / "runtime" / "triton_image_inspect.json"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_bytes(runtime_payload)

    report = generate_report(
        output_root,
        bootstrap_iterations=500,
        bootstrap_seed=101,
    )

    assert report["expected_sentences_per_arm"] == 114
    assert report["expected_sentence_observations_total"] == 342
    assert report["observation_completeness"]["integrity_complete"] is True
    assert {
        arm: values["sentence_rate"]["expected"]
        for arm, values in report["arms"].items()
    } == {
        ArmKind.CURRENT_HEAD.value: 114,
        ArmKind.TRITON_0818.value: 114,
        ArmKind.PYTORCH_0818.value: 114,
    }
    assert (
        sum(values["sentence_rate"]["expected"] for values in report["arms"].values())
        == 342
    )
    current_vs_triton = next(
        item
        for item in report["comparisons"]
        if item.get("bootstrap", {}).get("baseline_arm") == ArmKind.TRITON_0818.value
        and item.get("bootstrap", {}).get("comparison_arm")
        == ArmKind.CURRENT_HEAD.value
    )
    assert current_vs_triton["bootstrap"]["risk_difference"] == pytest.approx(14 / 38)
    assert current_vs_triton["bootstrap"]["risk_ratio"] == pytest.approx(15.0)
    assert current_vs_triton["gate"]["checks"] == {
        "absolute_gap": True,
        "risk_ratio": True,
        "risk_difference_ci_excludes_zero": True,
        "seed_direction": True,
        "validity": True,
    }
    assert current_vs_triton["gate"]["enter_root_cause_analysis"] is True
    assert report["conclusion"] == "DEFAULT_GAP_REQUIRES_MATCHED_REPLAY"
    with (output_root / "report" / "sentences.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        assert sum(1 for _row in csv.DictReader(stream)) == 342
    persisted = json.loads(
        (output_root / "report" / "report.json").read_text(encoding="utf-8")
    )
    assert persisted["expected_sentences_per_arm"] == 114
    assert (
        persisted["report_inputs"]["experiment_manifest"]["sha256"]
        == hashlib.sha256((output_root / "manifest.json").read_bytes()).hexdigest()
    )
    assert (
        persisted["report_inputs"]["final_labels"]["sha256"]
        == hashlib.sha256(
            (output_root / "review" / "private" / "final_labels.json").read_bytes()
        ).hexdigest()
    )
    assert persisted["runtime_evidence"]["audit_ready"] is True
    assert (
        persisted["runtime_evidence"]["tree_sha256"]
        == report["runtime_evidence"]["tree_sha256"]
    )
    assert persisted["runtime_evidence"]["entries"] == [
        {
            "bytes": len(runtime_payload),
            "hash_scope": "file_contents",
            "kind": "file",
            "path": "runtime/triton_image_inspect.json",
            "sha256": hashlib.sha256(runtime_payload).hexdigest(),
        }
    ]
    markdown = (output_root / "report" / "report.md").read_text(encoding="utf-8")
    assert "## 可追溯性" in markdown
    assert "runtime/triton_image_inspect.json" in markdown
    assert hashlib.sha256(runtime_payload).hexdigest() in markdown


def test_report_counts_manifest_grid_rows_missing_from_final_truth_as_invalid(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    seeds = [11, 22, 33]
    write_json(
        output_root / "manifest.json",
        {"seeds": seeds, "text": {"text": _report_reference_text()}},
    )
    write_json(
        output_root / "reference_groups.json",
        {"groups": _reference_groups()},
    )
    rows = [
        {
            "arm": arm.value,
            "seed": seed,
            "sentence_ordinal": ordinal,
            "sentence_id": f"sentence-{ordinal:03d}",
            "review_label": ReviewLabel.OK.value,
            "severe_hallucination": False,
            "valid_for_rate": True,
            "status": RunStatus.REVIEWED.value,
        }
        for arm in (
            ArmKind.CURRENT_HEAD,
            ArmKind.TRITON_0818,
            ArmKind.PYTORCH_0818,
        )
        for seed in seeds
        for ordinal in range(1, 39)
        if not (arm is ArmKind.CURRENT_HEAD and seed == seeds[0] and ordinal <= 6)
    ]
    write_json(
        output_root / "review" / "private" / "final_labels.json",
        {"rows": rows},
    )

    report = generate_report(
        output_root,
        bootstrap_iterations=20,
        bootstrap_seed=101,
    )

    current = report["observation_completeness"]["arms"][ArmKind.CURRENT_HEAD.value]
    assert current == {
        "expected": 114,
        "observed_rows": 108,
        "observed_expected_positions": 108,
        "missing_observations": 6,
        "duplicate_positions": 0,
        "duplicate_extra_rows": 0,
        "unexpected_rows": 0,
        "all_expected_positions_present": False,
        "integrity_complete": False,
    }
    current_rate = report["arms"][ArmKind.CURRENT_HEAD.value]["sentence_rate"]
    assert current_rate["expected"] == 114
    assert current_rate["observed"] == 108
    assert current_rate["missing"] == 6
    assert current_rate["invalid"] == 6
    assert report["invalidity"]["arms"][ArmKind.CURRENT_HEAD.value] == {
        "invalid": 6,
        "total": 114,
        "rate": pytest.approx(6 / 114),
    }
    assert report["invalidity"]["insufficient"] is True
    assert report["conclusion"] == "INSUFFICIENT_INVALID_SAMPLES"
    with (output_root / "report" / "sentences.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        sentence_rows = list(csv.DictReader(stream))
    assert len(sentence_rows) == 342
    missing_rows = [
        row
        for row in sentence_rows
        if row["report_invalid_reason"] == "missing_observation"
    ]
    assert len(missing_rows) == 6
    markdown = (output_root / "report" / "report.md").read_text(encoding="utf-8")
    assert "## 观测完整性" in markdown
    assert "| current_head | 114 | 108 | 6 | 0 | 0 |" in markdown


def test_report_rejects_duplicate_and_out_of_grid_truth_rows(tmp_path: Path) -> None:
    output_root = tmp_path / "experiment"
    seeds = [11, 22, 33]
    write_json(
        output_root / "manifest.json",
        {"seeds": seeds, "text": {"text": _report_reference_text()}},
    )
    write_json(
        output_root / "reference_groups.json",
        {"groups": _reference_groups()},
    )
    rows = [
        {
            "arm": arm.value,
            "seed": seed,
            "sentence_ordinal": ordinal,
            "sentence_id": f"sentence-{ordinal:03d}",
            "review_label": ReviewLabel.OK.value,
            "severe_hallucination": False,
            "valid_for_rate": True,
            "status": RunStatus.REVIEWED.value,
        }
        for arm in (
            ArmKind.CURRENT_HEAD,
            ArmKind.TRITON_0818,
            ArmKind.PYTORCH_0818,
        )
        for seed in seeds
        for ordinal in range(1, 39)
    ]
    rows.append(dict(rows[0]))
    rows.append({**dict(rows[0]), "sentence_ordinal": 999})
    rows.append({**dict(rows[0]), "arm": "unexpected_arm"})
    write_json(
        output_root / "review" / "private" / "final_labels.json",
        {"rows": rows},
    )

    report = generate_report(
        output_root,
        bootstrap_iterations=20,
        bootstrap_seed=101,
    )

    completeness = report["observation_completeness"]
    current = completeness["arms"][ArmKind.CURRENT_HEAD.value]
    assert current["missing_observations"] == 0
    assert current["duplicate_positions"] == 1
    assert current["duplicate_extra_rows"] == 1
    assert current["unexpected_rows"] == 1
    assert completeness["unexpected_arm_rows"] == 1
    assert completeness["integrity_complete"] is False
    assert report["invalidity"]["observation_grid_integrity"] == {
        "complete": False,
        "reasons": [
            "observation_grid:unexpected_arm_rows",
            "current_head:duplicate_positions",
            "current_head:unexpected_rows",
        ],
    }
    assert report["invalidity"]["insufficient"] is True
    assert report["conclusion"] == "INSUFFICIENT_INVALID_SAMPLES"
    for comparison in report["comparisons"]:
        assert comparison["gate"]["checks"]["validity"] is False
