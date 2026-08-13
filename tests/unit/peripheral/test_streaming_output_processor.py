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
                "raw_codepoint_end": str(index + 1),
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
    assert [a["meta"]["anchor_seq"] for a in anchors] == ["2"]


def test_soxr_stream_flush_produces_final_output_samples():
    processor = _processor(16000)
    audio, _anchors = _collect(
        processor, [np.linspace(-0.5, 0.5, 2400, dtype=np.float32)]
    )
    assert len(audio) // 4 == 1600


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
