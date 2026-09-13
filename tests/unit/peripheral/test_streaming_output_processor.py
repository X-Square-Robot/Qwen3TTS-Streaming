from __future__ import annotations

import numpy as np

from engine.core.types import AttributedAudioChunk, AudioConfig, SessionConfig
from engine.interface import (
    SessionStartRequest,
    StreamingOutputProcessor,
    parse_output_policy,
    parse_timing_context,
)
from engine.interface.vad import (
    EnergyVADProcessor,
    SampleProvenanceSpan,
    TTSVADConfig,
    VADMode,
    create_vad_processor,
)


def _processor(sample_rate: int, *, vad=None) -> StreamingOutputProcessor:
    start = SessionStartRequest(
        "output-test",
        SessionConfig(audio=AudioConfig(sample_rate=sample_rate)),
        output_policy=parse_output_policy({}),
        timing=parse_timing_context({}),
    )
    return StreamingOutputProcessor(
        start,
        vad_processor=vad
        or create_vad_processor(
            TTSVADConfig(mode=VADMode.DISABLED), sample_rate=24000
        ),
    )


def _collect(processor: StreamingOutputProcessor, chunks: list[np.ndarray]) -> tuple[bytes, list[dict]]:
    audio: list[bytes] = []
    anchors: list[dict] = []
    for index, chunk in enumerate(chunks):
        event = {
            "type": "text_progress",
            "segment_idx": 0,
                "meta": {
                    "anchor_seq": str(index + 1),
                    "raw_codepoint_start": str(index),
                    "raw_codepoint_end": str(index + 1),
                    "normalized_codepoint_start": str(index),
                    "normalized_codepoint_end": str(index + 1),
                },
        }
        batch = processor.process(
            AttributedAudioChunk(chunk.tobytes(), event, segment_idx=0)
        )
        if batch.audio is not None:
            audio.append(batch.audio.pcm_bytes)
        anchors.extend(batch.anchors)
    for batch in processor.finish():
        if batch.audio is not None:
            audio.append(batch.audio.pcm_bytes)
        anchors.extend(batch.anchors)
    return b"".join(audio), anchors


def test_disabled_processor_keeps_anchor_out_of_audio_meta():
    processor = _processor(24000)
    audio, anchors = _collect(processor, [np.ones(240, dtype=np.float32)])
    assert len(audio) == 240 * 4
    assert len(anchors) == 1
    assert anchors[0]["meta"]["output_sample_start"] == "0"
    assert anchors[0]["meta"]["output_sample_end"] == "240"


def test_vad_provenance_drops_marker_attached_to_prefix_silence():
    vad = EnergyVADProcessor(
        TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.3,
            begin_count=2,
            end_threshold=0.2,
            end_count=100,
            start_margin_ms=0,
        ),
        sample_rate=24000,
    )
    processor = _processor(24000, vad=vad)
    silence = np.zeros(384 * 2, dtype=np.float32)
    tone = np.full(384 * 3, 0.5, dtype=np.float32)
    audio, anchors = _collect(processor, [silence, tone])
    assert audio
    # The discarded prefix candidate does not consume a public sequence.
    assert [a["meta"]["anchor_seq"] for a in anchors] == ["1"]


def test_soxr_stream_flush_produces_final_output_samples():
    processor = _processor(16000)
    audio, _anchors = _collect(
        processor, [np.linspace(-0.5, 0.5, 2400, dtype=np.float32)]
    )
    assert len(audio) // 4 == 1600


def test_resampled_anchor_boundaries_use_integer_native_ratio():
    processor = _processor(16000)
    _audio, anchors = _collect(
        processor,
        [
            np.linspace(-0.5, 0.0, 1200, dtype=np.float32),
            np.linspace(0.0, 0.5, 1200, dtype=np.float32),
        ],
    )
    assert [
        (item["meta"]["output_sample_start"], item["meta"]["output_sample_end"])
        for item in anchors
    ] == [("0", "800"), ("800", "1600")]


def test_segment_final_marker_stays_before_next_segment_anchor():
    processor = _processor(24000)
    first = processor.process(
        AttributedAudioChunk(
            np.ones(240, dtype=np.float32).tobytes(),
            {
                "type": "text_progress",
                "segment_idx": 0,
                "meta": {
                    "raw_codepoint_start": "0",
                    "raw_codepoint_end": "1",
                    "normalized_codepoint_start": "0",
                    "normalized_codepoint_end": "1",
                },
            },
            segment_idx=0,
        )
    )
    final = processor.process_event(
        {
            "type": "text_progress",
            "segment_idx": 0,
            "meta": {
                "alignment_final": "true",
                "raw_codepoint_start": "0",
                "raw_codepoint_end": "1",
                "normalized_codepoint_start": "0",
                "normalized_codepoint_end": "1",
            },
        }
    )
    second = processor.process(
        AttributedAudioChunk(
            np.ones(240, dtype=np.float32).tobytes(),
            {
                "type": "text_progress",
                "segment_idx": 1,
                "meta": {
                    "raw_codepoint_start": "1",
                    "raw_codepoint_end": "2",
                    "normalized_codepoint_start": "1",
                    "normalized_codepoint_end": "2",
                },
            },
            segment_idx=1,
        )
    )
    anchors = [*first.anchors, *final.anchors, *second.anchors]
    assert [item["meta"]["anchor_seq"] for item in anchors] == ["1", "2", "3"]
    assert anchors[1]["meta"]["alignment_final"] == "true"
    assert anchors[1]["meta"]["output_sample_end"] == "240"


def test_segment_final_waits_for_late_audio_frame_anchor():
    processor = _processor(24000)
    first = processor.process(
        AttributedAudioChunk(
            np.ones(240, dtype=np.float32).tobytes(),
            {
                "type": "text_progress",
                "segment_idx": 0,
                "meta": {
                    "raw_codepoint_start": "0",
                    "raw_codepoint_end": "1",
                    "normalized_codepoint_start": "0",
                    "normalized_codepoint_end": "1",
                },
            },
            segment_idx=0,
            source_frame_end=1,
        )
    )
    # SEGMENT_END may be observed before the last audio chunk reaches the
    # output processor. The final anchor must remain pending at that point.
    processor.process_event(
        {
            "type": "segment_end",
            "segment_idx": 0,
            "meta": {"segment_decode_steps": "2"},
        }
    )
    pending = processor.process_event(
        {
            "type": "text_progress",
            "segment_idx": 0,
            "meta": {
                "alignment_final": "true",
                "raw_codepoint_start": "0",
                "raw_codepoint_end": "1",
                "normalized_codepoint_start": "0",
                "normalized_codepoint_end": "1",
            },
        }
    )
    second = processor.process(
        AttributedAudioChunk(
            np.ones(240, dtype=np.float32).tobytes(),
            {
                "type": "text_progress",
                "segment_idx": 0,
                "meta": {
                    "raw_codepoint_start": "1",
                    "raw_codepoint_end": "2",
                    "normalized_codepoint_start": "1",
                    "normalized_codepoint_end": "2",
                },
            },
            segment_idx=0,
            source_frame_end=2,
        )
    )
    anchors = [*first.anchors, *pending.anchors, *second.anchors]
    finals = [item for item in anchors if item["meta"].get("alignment_final") == "true"]
    assert len(finals) == 1
    assert finals[0]["meta"]["output_sample_end"] == "480"


def test_released_cross_segment_anchor_coordinates_keep_global_high_water():
    processor = _processor(24000)

    first = processor.process(
        AttributedAudioChunk(
            np.ones(240, dtype=np.float32).tobytes(),
            {
                "type": "text_progress",
                "segment_idx": 0,
                "meta": {
                    "raw_codepoint_start": "0",
                    "raw_codepoint_end": "10",
                    "normalized_codepoint_start": "0",
                    "normalized_codepoint_end": "10",
                    "display_raw_position": "10.0",
                    "display_normalized_position": "10.0",
                },
            },
            segment_idx=0,
        )
    )
    second = processor.process(
        AttributedAudioChunk(
            np.ones(240, dtype=np.float32).tobytes(),
            {
                "type": "text_progress",
                "segment_idx": 1,
                "meta": {
                    # A segment-local projector may restart at a lower
                    # boundary. Once released, the public anchor must not.
                    "raw_codepoint_start": "5",
                    "raw_codepoint_end": "6",
                    "normalized_codepoint_start": "5",
                    "normalized_codepoint_end": "6",
                    "display_raw_position": "6.0",
                    "display_normalized_position": "6.0",
                },
            },
            segment_idx=1,
        )
    )
    final = processor.process_event(
        {
            "type": "text_progress",
            "segment_idx": 1,
            "meta": {
                "alignment_final": "true",
                "raw_codepoint_start": "5",
                "raw_codepoint_end": "6",
                "normalized_codepoint_start": "5",
                "normalized_codepoint_end": "6",
                "display_raw_position": "6.0",
                "display_normalized_position": "6.0",
            },
        }
    )

    anchors = [*first.anchors, *second.anchors, *final.anchors]
    assert [item["meta"]["raw_codepoint_end"] for item in anchors] == [
        "10",
        "10",
        "10",
    ]
    assert [item["meta"]["normalized_codepoint_end"] for item in anchors] == [
        "10",
        "10",
        "10",
    ]
    assert [item["meta"]["display_raw_position"] for item in anchors] == [
        "10.000000",
        "10.000000",
        "10.000000",
    ]


def test_display_coordinate_cannot_fall_below_confirmed_integer_boundary():
    processor = _processor(24000)
    first = processor.process(
        AttributedAudioChunk(
            np.ones(240, dtype=np.float32).tobytes(),
            {
                "type": "text_progress",
                "segment_idx": 0,
                "meta": {
                    "raw_codepoint_start": "0",
                    "raw_codepoint_end": "20",
                    "normalized_codepoint_start": "0",
                    "normalized_codepoint_end": "20",
                },
            },
            segment_idx=0,
        )
    )
    second = processor.process(
        AttributedAudioChunk(
            np.ones(240, dtype=np.float32).tobytes(),
            {
                "type": "text_progress",
                "segment_idx": 1,
                "meta": {
                    "raw_codepoint_start": "20",
                    "raw_codepoint_end": "20",
                    "normalized_codepoint_start": "20",
                    "normalized_codepoint_end": "20",
                    "display_raw_position": "6.0",
                    "display_normalized_position": "6.0",
                },
            },
            segment_idx=1,
        )
    )

    anchors = [*first.anchors, *second.anchors]
    assert anchors[1]["meta"]["display_raw_position"] == "20.000000"
    assert anchors[1]["meta"]["display_normalized_position"] == "20.000000"


def test_vad_attributed_frame_spans_are_rebased_and_monotonic():
    vad = EnergyVADProcessor(
        TTSVADConfig(mode=VADMode.ENERGY, begin_threshold=0.0, begin_count=1),
        sample_rate=24000,
    )
    result = vad.process_attributed_chunk(
        np.ones(384, dtype=np.int16),
        [SampleProvenanceSpan(0, 384, segment_idx=7)],
    )
    assert result.samples.size == 384
    assert [(span.sample_start, span.sample_end) for span in result.provenance] == [
        (0, 384)
    ]


def test_vad_keeps_marker_pending_until_partial_source_tail_is_decided():
    vad = EnergyVADProcessor(
        TTSVADConfig(
            mode=VADMode.ENERGY,
            begin_threshold=0.3,
            begin_count=1,
            end_threshold=0.2,
            end_count=100,
            start_margin_ms=0,
        ),
        sample_rate=24000,
    )
    processor = _processor(24000, vad=vad)
    batch = processor.process(
        AttributedAudioChunk(
            np.full(500, 0.5, dtype=np.float32).tobytes(),
            {
                "type": "text_progress",
                "segment_idx": 0,
                "meta": {
                    "raw_codepoint_start": "0",
                    "raw_codepoint_end": "1",
                    "normalized_codepoint_start": "0",
                    "normalized_codepoint_end": "1",
                },
            },
            segment_idx=0,
        )
    )
    assert batch.audio is not None
    assert batch.audio.meta["output_sample_end"] == "384"
    assert batch.anchors == []

    flushed = processor.finish()
    assert flushed[0].audio is not None
    assert flushed[0].audio.meta["output_sample_start"] == "384"
    assert flushed[0].audio.meta["output_sample_end"] == "500"
    assert flushed[0].anchors[0]["meta"]["output_sample_end"] == "500"


def test_abort_discards_vad_and_resampler_pending_audio_without_anchor():
    vad = EnergyVADProcessor(
        TTSVADConfig(mode=VADMode.ENERGY, begin_threshold=0.3, begin_count=1),
        sample_rate=24000,
    )
    processor = _processor(16000, vad=vad)
    processor.process(
        AttributedAudioChunk(
            np.full(500, 0.5, dtype=np.float32).tobytes(),
            {
                "type": "text_progress",
                "segment_idx": 0,
                "meta": {
                    "raw_codepoint_start": "0",
                    "raw_codepoint_end": "1",
                    "normalized_codepoint_start": "0",
                    "normalized_codepoint_end": "1",
                },
            },
            segment_idx=0,
        )
    )
    assert processor.finish(emit_final=False) == []


def test_legacy_progress_without_text_span_is_forwarded_without_v1_anchor():
    processor = _processor(24000)
    event = {
        "type": "text_progress",
        "segment_idx": 0,
        "meta": {
            "anchor_seq": "9",
            "output_sample_start": "0",
            "output_sample_end": "240",
            "text_progress": "0.5",
        },
    }
    batch = processor.process_event(event)
    assert batch.anchors == []
    assert batch.events[0]["meta"].get("anchor_seq") is None
