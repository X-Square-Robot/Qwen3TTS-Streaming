from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np
import pytest
from qwen3tts_protocol import (
    AudioChunk,
    AudioFormat,
    OutputPolicy,
    StreamEvent,
    VADPolicy,
)

from tools.validation.hallucination.longform import arms as arms_facade
from tools.validation.hallucination.longform.arm_types import (
    AudioChunkRecord as SplitAudioChunkRecord,
    CollectedRun as SplitCollectedRun,
)
from tools.validation.hallucination.longform.arms import (
    DEFAULT_SAMPLE_RATE,
    EngineGrpcArmAdapter,
    OfficialPyTorchArmAdapter,
    TritonGrpcArmAdapter,
    stable_sampling_seed,
)
from tools.validation.hallucination.longform.endpoint_arm import (
    EngineGrpcArmAdapter as SplitEngineGrpcArmAdapter,
    TritonGrpcArmAdapter as SplitTritonGrpcArmAdapter,
)
from tools.validation.hallucination.longform.models import ArmKind, RunStatus
from tools.validation.hallucination.longform.official_arm import (
    OfficialPyTorchArmAdapter as SplitOfficialPyTorchArmAdapter,
)


def test_arms_facade_preserves_the_original_public_objects():
    assert arms_facade.AudioChunkRecord is SplitAudioChunkRecord
    assert arms_facade.CollectedRun is SplitCollectedRun
    assert arms_facade.EngineGrpcArmAdapter is SplitEngineGrpcArmAdapter
    assert arms_facade.TritonGrpcArmAdapter is SplitTritonGrpcArmAdapter
    assert arms_facade.OfficialPyTorchArmAdapter is SplitOfficialPyTorchArmAdapter
    assert arms_facade.EngineGrpcArm is SplitEngineGrpcArmAdapter
    assert arms_facade.TritonGrpcArm is SplitTritonGrpcArmAdapter
    assert arms_facade.OfficialPyTorchArm is SplitOfficialPyTorchArmAdapter


class _FakeSession:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent: list[str] = []
        self.ended = False
        self.timeout: float | None = None
        self.closed_reason: str | None = None

    def send_text(self, text: str) -> None:
        self.sent.append(text)

    def end(self) -> None:
        self.ended = True

    def iter_messages(self, *, post_send_idle_timeout: float):
        self.timeout = post_send_idle_timeout
        yield from self.messages

    def close(self, reason: str = "") -> None:
        self.closed_reason = reason


class _FakeClient:
    def __init__(self, session: _FakeSession):
        self.session = session
        self.requests = []
        self.closed = False

    def open_stream(self, request):
        self.requests.append(request)
        return self.session

    def close(self) -> None:
        self.closed = True


def _pcm_chunk(
    values,
    *,
    chunk_index: int,
    output_start: int | None = None,
    output_end: int | None = None,
    meta=None,
):
    chunk_meta = dict(meta or {})
    if output_start is not None:
        chunk_meta["output_sample_start"] = str(output_start)
    if output_end is not None:
        chunk_meta["output_sample_end"] = str(output_end)
    return AudioChunk(
        pcm_bytes=np.asarray(values, dtype="<f4").tobytes(),
        audio=AudioFormat(
            encoding="pcm_f32", sample_rate=DEFAULT_SAMPLE_RATE, channels=1
        ),
        chunk_index=chunk_index,
        first_chunk=chunk_index == 4,
        final_chunk=chunk_index == 5,
        meta=chunk_meta,
    )


def test_engine_arm_sends_exact_full_text_and_collects_all_stream_evidence():
    text = "  第一段。\n第二段。\n"
    session = _FakeSession(
        [
            StreamEvent(
                type="start",
                session_id="logical-01",
                audio=AudioFormat(
                    encoding="pcm_f32", sample_rate=24000, channels=1
                ),
                meta={"engine": "head"},
            ),
            _pcm_chunk(
                [0.25, -0.25],
                chunk_index=4,
                output_start=10,
                output_end=12,
                meta={"segment_id": "0"},
            ),
            StreamEvent(
                type="text_progress",
                session_id="logical-01",
                segment_id=0,
                text="第一段。",
                meta={"text_end": "4"},
            ),
            _pcm_chunk(
                [0.5],
                chunk_index=5,
                meta={"output_sample_start": "12", "output_sample_end": "13"},
            ),
            StreamEvent(
                type="done",
                session_id="logical-01",
                message="complete",
                meta={"reason": "natural_eos"},
            ),
        ]
    )
    client = _FakeClient(session)
    connect_calls = []

    def factory(endpoint, **kwargs):
        connect_calls.append((endpoint, kwargs))
        return client

    arm = EngineGrpcArmAdapter(
        "127.0.0.1:50051", timeout=17.0, client_factory=factory
    )
    result = arm.collect(text, session_id="logical-01", seed=7)

    assert connect_calls == [
        (
            "127.0.0.1:50051",
            {
                "transport": "engine-grpc",
                "timeout": 17.0,
            },
        )
    ]
    assert result.arm is ArmKind.CURRENT_HEAD
    assert result.status is RunStatus.OK
    assert result.seed == 7
    assert result.session_id == "logical-01"
    assert session.sent == [text]
    assert session.ended is True
    assert session.timeout == 17.0

    request = client.requests[0]
    assert request.session_id == "logical-01"
    assert request.config.task_type == "custom_voice"
    assert request.config.speaker == "001"
    assert request.config.input_mode == "full_text"
    assert request.config.group_policy == "auto"
    assert request.config.output_policy.config == {}
    assert request.config.audio == AudioFormat(
        encoding="pcm_f32", sample_rate=24000, channels=1
    )

    assert result.samples.tolist() == pytest.approx([0.25, -0.25, 0.5])
    assert result.samples.size == sum(
        chunk.sample_count for chunk in result.audio_chunks
    )
    assert result.pcm_bytes == np.asarray(
        [0.25, -0.25, 0.5], dtype="<f4"
    ).tobytes()
    assert result.duration_s == pytest.approx(3 / 24000)
    assert [event["type"] for event in result.events] == [
        "start",
        "text_progress",
        "done",
    ]
    assert result.events[0]["audio"] == {
        "encoding": "pcm_f32",
        "sample_rate": 24000,
        "channels": 1,
    }
    assert result.events[1]["text"] == "第一段。"
    assert result.terminal_event == "done"
    assert result.eos_reason == "natural_eos"

    first, second = result.audio_chunks
    assert (first.sample_start, first.sample_end) == (0, 2)
    assert (first.output_sample_start, first.output_sample_end) == (10, 12)
    assert (second.sample_start, second.sample_end) == (2, 3)
    assert (second.output_sample_start, second.output_sample_end) == (12, 13)
    assert first.meta == {
        "segment_id": "0",
        "output_sample_start": "10",
        "output_sample_end": "12",
    }
    assert len(first.pcm_sha256) == 64
    # Chunk records are safe to persist as JSON metadata: PCM lives on the run.
    assert isinstance(asdict(first)["pcm_sha256"], str)

    arm.close()
    assert client.closed is True


def test_endpoint_matched_replay_sends_dynamic_packets_in_one_unchanged_session():
    packets = [
        " 第一段。",
        "第二段。\n",
        "第三段，含 001。",
        "第四段。",
        "第五段。",
        "第六段。",
        "第七段。\n",
        "第八段。",
        "第九段。\n",
    ]
    segment_events = [
        StreamEvent(
            type="segment_done",
            session_id="matched-sid",
            segment_id=index,
            meta={"sampling_seed": f"server-seed-{index}"},
        )
        for index in range(len(packets))
    ]
    session = _FakeSession(
        [
            _pcm_chunk([0.1, -0.1], chunk_index=0),
            *segment_events,
            StreamEvent(type="done", session_id="matched-sid"),
        ]
    )
    client = _FakeClient(session)
    output_policy = OutputPolicy(
        vad=VADPolicy(enabled=False, strategy="disabled"),
        config={"delivery": "firehose"},
    )
    arm = EngineGrpcArmAdapter(
        "engine:50051", client_factory=lambda _endpoint, **_: client
    )

    result = arm.collect_packets(
        packets,
        "matched-sid",
        seed=31,
        output_policy=output_policy,
    )

    assert result.status is RunStatus.OK
    assert result.seed == 31
    assert result.session_id == "matched-sid"
    assert session.sent == packets
    assert session.ended is True
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request.session_id == "matched-sid"
    assert request.config.input_mode == "long_segment"
    assert request.config.group_policy == "none"
    assert request.config.output_policy is output_policy
    assert request.output_policy is output_policy
    assert request.config.output_policy.vad.enabled is False
    assert request.config.output_policy.config["delivery"] == "firehose"
    assert [event["segment_id"] for event in result.events[:-1]] == list(
        range(len(packets))
    )
    assert [
        event["meta"]["sampling_seed"] for event in result.events[:-1]
    ] == [f"server-seed-{index}" for index in range(len(packets))]


def test_triton_arm_uses_sdk_triton_transport_and_frozen_model_coordinates():
    session = _FakeSession(
        [
            _pcm_chunk([0.1], chunk_index=0, output_start=0, output_end=1),
            StreamEvent(type="done"),
        ]
    )
    client = _FakeClient(session)
    calls = []
    arm = TritonGrpcArmAdapter(
        "localhost:8001",
        model_name="tts_orchestrator",
        model_version="1",
        client_factory=lambda endpoint, **kwargs: calls.append((endpoint, kwargs))
        or client,
    )

    result = arm.collect("原文", session_id="same-sid")

    assert result.arm is ArmKind.TRITON_0818
    assert result.status is RunStatus.OK
    assert calls[0][0] == "localhost:8001"
    assert calls[0][1]["transport"] == "triton-grpc"
    assert calls[0][1]["model_name"] == "tts_orchestrator"
    assert calls[0][1]["model_version"] == "1"


def test_endpoint_error_event_is_tts_failed_and_keeps_partial_pcm_and_events():
    session = _FakeSession(
        [
            _pcm_chunk([0.2], chunk_index=0),
            StreamEvent(type="warning", message="retrying"),
            StreamEvent(type="error", message="guard abort"),
        ]
    )
    arm = EngineGrpcArmAdapter(
        "engine:50051", client_factory=lambda _endpoint, **_: _FakeClient(session)
    )

    result = arm.collect("全文", session_id="bad-run")

    assert result.status is RunStatus.TTS_FAILED
    assert result.error == "guard abort"
    assert result.samples.tolist() == pytest.approx([0.2])
    assert [event["type"] for event in result.events] == ["warning", "error"]
    assert result.terminal_event == "error"


def test_endpoint_transport_exception_is_error_and_closes_the_session():
    class BrokenSession(_FakeSession):
        def iter_messages(self, *, post_send_idle_timeout: float):
            raise TimeoutError("stalled after END")
            yield  # pragma: no cover

    session = BrokenSession([])
    arm = EngineGrpcArmAdapter(
        "engine:50051", client_factory=lambda _endpoint, **_: _FakeClient(session)
    )

    result = arm.collect("全文", session_id="timeout")

    assert result.status is RunStatus.ERROR
    assert result.error == "TimeoutError: stalled after END"
    assert session.closed_reason == "long-form collection failed"


class _FakeOfficialModel:
    def __init__(self):
        self.calls = []

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        return [np.asarray([0.125, -0.5], dtype=np.float32)], 24000


def test_official_arm_loads_lazily_seeds_stably_and_calls_high_level_api(tmp_path):
    model = _FakeOfficialModel()
    factory_calls = []
    seeds = []

    def factory(checkpoint):
        factory_calls.append(checkpoint)
        return model

    arm = OfficialPyTorchArmAdapter(
        tmp_path / "0818-checkpoint",
        model_factory=factory,
        seed_setter=seeds.append,
        generation_kwargs={"non_streaming_mode": False, "max_new_tokens": 512},
    )
    assert factory_calls == []

    text = "不去掉末尾换行。\n"
    result = arm.collect(text, session_id="shared-session-0001", seed=23)
    # ``seed`` is a trial label.  Like both endpoint engines, the official arm
    # derives its actual RNG track from server base seed 0 and the common SID.
    expected_seed = stable_sampling_seed(0, "shared-session-0001", 0)

    assert factory_calls == [tmp_path / "0818-checkpoint"]
    assert seeds == [expected_seed]
    assert model.calls == [
        {
            "text": text,
            "speaker": "001",
            "language": "Auto",
            "non_streaming_mode": False,
            "max_new_tokens": 512,
        }
    ]
    assert result.arm is ArmKind.PYTORCH_0818
    assert result.status is RunStatus.OK
    assert result.sampling_seed == expected_seed
    assert result.samples.tolist() == pytest.approx([0.125, -0.5])
    assert result.events == []
    assert result.audio_chunks[0].output_sample_start == 0
    assert result.audio_chunks[0].output_sample_end == 2
    assert result.audio_chunks[0].meta["sampling_seed"] == str(expected_seed)

    # Reusing one arm does not reload a multi-gigabyte checkpoint.
    second = arm.collect("第二次", session_id="shared-session-0001", seed=99)
    assert factory_calls == [tmp_path / "0818-checkpoint"]
    assert second.seed == 99
    assert second.sampling_seed == expected_seed
    assert seeds == [expected_seed, expected_seed]


def test_official_arm_rejects_non_24khz_output_without_discarding_audio(tmp_path):
    class WrongRateModel:
        def generate_custom_voice(self, **kwargs):
            return [np.asarray([0.1], dtype=np.float32)], 16000

    arm = OfficialPyTorchArmAdapter(
        tmp_path,
        model_factory=lambda _: WrongRateModel(),
        seed_setter=lambda _: None,
    )

    result = arm.collect("测试", session_id="wrong-rate", seed=1)

    assert result.status is RunStatus.TTS_FAILED
    assert result.sample_rate == 16000
    assert result.samples.tolist() == pytest.approx([0.1])
    assert "expected 24000 Hz" in result.error


def test_official_matched_segment_uses_real_segment_index_and_local_overrides(
    tmp_path,
):
    model = _FakeOfficialModel()
    seeded = []
    arm = OfficialPyTorchArmAdapter(
        tmp_path,
        sampling_base_seed=5,
        generation_kwargs={"temperature": 0.9, "top_k": 50},
        model_factory=lambda _: model,
        seed_setter=seeded.append,
    )

    first = arm.collect_segment(
        "冻结第一段。",
        session_id="matched-official-sid",
        trial_seed=41,
        segment_index=0,
    )
    fourth = arm.collect_segment(
        "冻结第四段。",
        session_id="matched-official-sid",
        trial_seed=41,
        segment_index=3,
        generation_kwargs_override={"temperature": 0.0, "do_sample": False},
    )

    expected_first = stable_sampling_seed(5, "matched-official-sid", 0)
    expected_fourth = stable_sampling_seed(5, "matched-official-sid", 3)
    assert seeded == [expected_first, expected_fourth]
    assert expected_first != expected_fourth
    assert first.seed == fourth.seed == 41
    assert first.sampling_seed == expected_first
    assert fourth.sampling_seed == expected_fourth
    assert model.calls == [
        {
            "text": "冻结第一段。",
            "speaker": "001",
            "language": "Auto",
            "temperature": 0.9,
            "top_k": 50,
        },
        {
            "text": "冻结第四段。",
            "speaker": "001",
            "language": "Auto",
            "temperature": 0.0,
            "top_k": 50,
            "do_sample": False,
        },
    ]


def test_stable_sampling_seed_is_process_stable_and_segment_sensitive():
    assert stable_sampling_seed(0, "sid", 0) == stable_sampling_seed(0, "sid", 0)
    assert stable_sampling_seed(0, "sid", 0) != stable_sampling_seed(0, "sid", 1)
    assert stable_sampling_seed(0, "sid", 0) != stable_sampling_seed(1, "sid", 0)
    assert 0 <= stable_sampling_seed(0, "sid", 0) <= (1 << 63) - 1


def test_stable_sampling_seed_matches_gateway_bound_executor_identity():
    digest = hashlib.blake2b(digest_size=16)
    digest.update(b"5")
    for part in ("shared-session:3", 3):
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    expected = int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)

    assert stable_sampling_seed(5, "shared-session", 3) == expected
