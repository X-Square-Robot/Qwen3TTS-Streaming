from __future__ import annotations

import json

import numpy as np
import pytest
from qwen3tts_protocol import AudioChunk, AudioFormat, StreamEvent

from tools.validation import hallucination_sweep as legacy_sweep
from tools.validation.hallucination.metrics import (
    character_error_metrics,
    classify_trial,
    wilson_interval,
)
from tools.validation.hallucination.models import (
    DEFAULT_BODY_TEXT,
    AsrStatus,
    ChunkPattern,
    SuspectReason,
    SuspectThresholds,
    TextPacket,
    TrialStatus,
    build_text_packets,
)
from tools.validation.hallucination.report import persist_trial, summarize_records
from tools.validation.hallucination.synthesis import synthesize_once


def test_legacy_module_reexports_the_same_public_contracts():
    assert legacy_sweep.ChunkPattern is ChunkPattern
    assert legacy_sweep.TextPacket is TextPacket
    assert legacy_sweep.build_text_packets is build_text_packets
    assert legacy_sweep.character_error_metrics is character_error_metrics
    assert legacy_sweep.persist_trial is persist_trial
    assert legacy_sweep.synthesize_once is synthesize_once
    assert legacy_sweep.wilson_interval is wilson_interval


@pytest.mark.parametrize(
    ("pattern", "expected", "expected_delays"),
    [
        (ChunkPattern.WHOLE_LEADING, [f" {DEFAULT_BODY_TEXT}"], [0.0]),
        (ChunkPattern.SPLIT_LEADING, [" ", DEFAULT_BODY_TEXT], [0.05, 0.0]),
        (ChunkPattern.CLEAN, [DEFAULT_BODY_TEXT], [0.0]),
    ],
)
def test_packet_patterns_preserve_the_exact_leading_space(
    pattern: ChunkPattern,
    expected: list[str],
    expected_delays: list[float],
):
    packets = build_text_packets(
        DEFAULT_BODY_TEXT,
        leading_prefix=" ",
        pattern=pattern,
        split_delay_ms=50,
    )

    assert [packet.text for packet in packets] == expected
    assert [packet.delay_after_s for packet in packets] == expected_delays


def test_character_error_metrics_reports_substitution_deletion_and_insertion():
    substitution = character_error_metrics("abc", "adc")
    deletion = character_error_metrics("abc", "ac")
    insertion = character_error_metrics("abc", "axbc")

    assert substitution["substitutions"] == 1
    assert substitution["deletions"] == substitution["insertions"] == 0
    assert deletion["deletions"] == 1
    assert deletion["substitutions"] == deletion["insertions"] == 0
    assert insertion["insertions"] == 1
    assert insertion["substitutions"] == insertion["deletions"] == 0
    assert insertion["cer"] == pytest.approx(1 / 3)


def test_character_error_metrics_normalizes_punctuation_and_case():
    metrics = character_error_metrics("好啦, Cola!", "好啦cola")

    assert metrics["distance"] == 0
    assert metrics["cer"] == 0.0


def test_wilson_interval_zero_of_one_hundred_has_expected_upper_bound():
    interval = wilson_interval(0, 100)

    assert interval["rate"] == 0.0
    assert interval["low"] == pytest.approx(0.0)
    assert interval["high"] == pytest.approx(0.0369935, rel=1e-5)


class _FakeSession:
    def __init__(self):
        self.sent: list[str] = []
        self.ended = False

    def send_text(self, text: str):
        self.sent.append(text)

    def end(self):
        self.ended = True

    def iter_messages(self, *, post_send_idle_timeout: float):
        assert post_send_idle_timeout == 12.0
        yield AudioChunk(
            pcm_bytes=np.array([0.25, -0.25], dtype=np.float32).tobytes(),
            audio=AudioFormat(encoding="pcm_f32", sample_rate=24000, channels=1),
        )
        yield StreamEvent(type="done", meta={"reason": "natural_eos"})


class _FakeClient:
    def __init__(self):
        self.session = _FakeSession()
        self.session_id = ""

    def open_stream(self, request):
        self.session_id = request.session_id
        return self.session


def test_synthesize_once_sends_split_packets_in_order_with_real_seam():
    client = _FakeClient()
    sleeps: list[float] = []
    packets = [TextPacket(" ", 0.05), TextPacket(DEFAULT_BODY_TEXT)]

    result = synthesize_once(
        client,
        packets,
        speaker="serena",
        session_id="paired-0001",
        timeout=12.0,
        sleep_fn=sleeps.append,
    )

    assert client.session_id == "paired-0001"
    assert client.session.sent == [" ", DEFAULT_BODY_TEXT]
    assert client.session.ended is True
    assert sleeps == [0.05]
    assert result.status is TrialStatus.OK
    assert result.samples.tolist() == pytest.approx([0.25, -0.25])
    assert result.eos_reason == "natural_eos"


def test_persist_trial_writes_wav_and_exact_packet_json(tmp_path):
    client = _FakeClient()
    packets = [TextPacket(f" {DEFAULT_BODY_TEXT}")]
    result = synthesize_once(
        client,
        packets,
        speaker="serena",
        session_id="paired-0007",
        timeout=12.0,
    )

    record = persist_trial(
        tmp_path,
        trial_index=7,
        session_id="paired-0007",
        pattern=ChunkPattern.WHOLE_LEADING,
        packets=packets,
        result=result,
    )

    wav_path = tmp_path / record["artifacts"]["wav"]
    json_path = tmp_path / record["artifacts"]["json"]
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert wav_path.is_file()
    assert sidecar["packets"][0]["text"] == f" {DEFAULT_BODY_TEXT}"
    assert sidecar["session_id"] == "paired-0007"
    assert len(sidecar["pcm_s16le_sha256"]) == 64
    assert set(sidecar) == {
        "artifacts",
        "chunks",
        "duration_s",
        "eos_reason",
        "error",
        "events",
        "pattern",
        "pcm_f32le_sha256",
        "pcm_s16le_sha256",
        "sample_rate",
        "session_id",
        "status",
        "terminal_event",
        "total_ms",
        "trial_index",
        "ttft_ms",
        "wav_sha256",
        "packets",
    }


def test_classification_uses_configurable_asr_thresholds_without_claiming_truth():
    record = {
        "status": TrialStatus.OK.value,
        "duration_s": 4.0,
        "asr": {
            "status": AsrStatus.OK.value,
            "metrics": {"cer": 0.31, "insertions": 2},
        },
    }

    screening = classify_trial(
        record,
        thresholds=SuspectThresholds(duration_s=30.0, cer=0.30, insertions=3),
        asr_requested=True,
    )

    assert screening["is_suspect"] is True
    assert screening["reasons"] == [SuspectReason.CER.value]
    assert "requires blind human review" in screening["label"]


def test_asr_failure_is_unknown_instead_of_being_counted_clean():
    screening = classify_trial(
        {
            "status": TrialStatus.OK.value,
            "duration_s": 4.0,
            "asr": {"status": AsrStatus.ERROR.value, "error": "offline"},
        },
        thresholds=SuspectThresholds(),
        asr_requested=True,
    )

    assert screening["is_suspect"] is None
    assert screening["excluded_reason"] == "asr_unavailable"


def test_disabled_asr_is_unknown_instead_of_being_counted_clean():
    screening = classify_trial(
        {
            "status": TrialStatus.OK.value,
            "duration_s": 4.0,
            "asr": {"status": AsrStatus.DISABLED.value},
        },
        thresholds=SuspectThresholds(),
        asr_requested=False,
    )

    assert screening["is_suspect"] is None
    assert screening["excluded_reason"] == "asr_unavailable"


def test_unknown_trials_do_not_produce_a_false_zero_rate():
    summary = summarize_records(
        [
            {
                "status": TrialStatus.OK.value,
                "duration_s": 4.0,
                "screening": {"is_suspect": None},
            }
        ],
        confidence=0.95,
    )

    assert summary["screened"] == 0
    assert summary["screening_unknown"] == 1
    assert summary["asr_supported_suspect_rate_wilson"]["rate"] is None
