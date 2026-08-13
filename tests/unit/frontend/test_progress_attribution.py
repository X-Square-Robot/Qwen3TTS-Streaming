from engine.core.types import AttributedAudioChunk
from engine.frontend.spliter.reorder import AudioReorder


def test_attributed_audio_stays_bound_to_audio_through_reorder():
    reorder = AudioReorder()
    lookahead = AttributedAudioChunk(
        b"lookahead",
        progress_event={"meta": {"anchor_seq": "2"}},
        segment_idx=1,
    )
    assert reorder.push(0, 1, lookahead) == []

    first = AttributedAudioChunk(
        b"first",
        progress_event={"meta": {"anchor_seq": "1"}},
        segment_idx=0,
    )
    assert reorder.push(0, 0, first) == [first]
    assert reorder.mark_done(0, 0) == [lookahead]
    assert lookahead.progress_event["meta"]["anchor_seq"] == "2"
