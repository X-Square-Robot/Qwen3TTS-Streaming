from types import SimpleNamespace

from engine.core.native_cursor import CursorLabelPlan, CursorOwnerSpan
from engine.core.text_journal import CanonicalTextJournal
from engine.core.types import AttributedAudioChunk
from engine.frontend.interface import FrontendInterface
from engine.frontend.spliter.reorder import AudioReorder


def session(native):
    journal = CanonicalTextJournal(lambda text: text)
    journal.install_projection("ab99%", "ab百分之九十九", [0, 1, 2, 2, 2, 2, 2, 2, 5])
    plan = CursorLabelPlan(
        label_ids=tuple(range(6)), revision=1,
        owner_spans=(CursorOwnerSpan(7, 0, 6, 2, 8, 2, 5),),
        label_normalized_spans=tuple((i, i + 1) for i in range(2, 8)),
    ) if native else None
    return SimpleNamespace(
        spliter=SimpleNamespace(ema_ratio_for_segment=lambda _: 1.0),
        segment_progress_frames={}, segment_token_emitted_count={},
        text_progress_estimators={}, native_cursor_projectors={},
        segment_token_spans={1: [
            dict(normalized_start=i, normalized_end=i + 1, raw_start=2, raw_end=5)
            for i in range(2, 8)
        ]}, text_journal=journal, cursor_segment_plans={1: plan},
        cursor_label_plan=plan, input_complete=True, session_id="test",
    )


def event(state, frame, mu=None, final=False, valid=1):
    metrics = dict(source_frame_start=frame - 1, source_frame_end=frame, text_tokens=6)
    if mu is not None:
        metrics.update(cursor_mu=mu, cursor_valid=valid, cursor_plan_revision=1)
    return FrontendInterface._make_text_progress_event(None, state, 1, metrics, final=final)


def test_ema_native_share_codec_token_raw_contract_at_every_frontier():
    ema, native = session(False), session(True)
    keys = ["source_frame_start", "source_frame_end", "text_token_start",
            "text_token_end", "normalized_codepoint_start", "normalized_codepoint_end",
            "raw_codepoint_start", "raw_codepoint_end"]
    for frame in range(1, 7):
        left, right = event(ema, frame), event(native, frame, float(frame))
        assert left["meta"]["progress_basis"] == "ema_frame_ratio_v1"
        assert right["meta"]["progress_basis"] == "native_cursor_v1"
        assert {key: left["meta"][key] for key in keys} == {
            key: right["meta"][key] for key in keys
        }
        assert right["meta"]["raw_codepoint_end"] == ("2" if frame < 6 else "5")


def test_precise_native_events_publish_owner_display_interpolation():
    result = event(session(True), 1, 1.0)

    assert result["meta"]["raw_codepoint_end"] == "2"
    assert float(result["meta"]["display_raw_position"]) == 2.5


def test_native_lookahead_holds_and_bad_estimate_fallback_does_not_retract():
    state = session(True)
    first = event(state, 1, 4.0)
    held = event(state, 2, 6.0, valid=0)
    assert held["meta"]["progress_basis"] == "native_cursor_v1"
    assert held["meta"]["text_token_end"] == first["meta"]["text_token_end"] == "4"
    fallback = event(state, 3, float("nan"))
    assert fallback["meta"]["progress_basis"] == "ema_frame_ratio_v1"
    assert fallback["meta"]["text_token_end"] == "4"
    assert fallback["meta"]["text_progress"] == first["meta"]["text_progress"]
    assert event(state, 4, final=True)["meta"]["progress_basis"] == "ema_frame_ratio_v1"


def test_final_without_neural_observation_never_claims_native():
    final = event(session(True), 1, final=True)
    assert final["meta"]["progress_basis"] == "ema_frame_ratio_v1"


def test_valid_neural_observation_requires_a_position():
    result = FrontendInterface._make_text_progress_event(
        None, session(True), 1,
        {"source_frame_start": 0, "source_frame_end": 1, "text_tokens": 6,
         "cursor_plan_revision": 1, "cursor_valid": 1},
    )
    assert result["meta"]["progress_basis"] == "ema_frame_ratio_v1"


def test_stalled_native_cursor_downgrades_to_ema_without_retracting_progress():
    state = session(True)
    events = [event(state, frame, 0.0) for frame in range(1, 18)]

    assert all(item["meta"]["progress_basis"] == "native_cursor_v1"
               for item in events[:16])
    assert events[-1]["meta"]["progress_basis"] == "ema_frame_ratio_v1"
    assert state.native_cursor_disabled is True
    assert state.native_cursor_fallback_reason == "stalled"
    assert int(events[-1]["meta"]["text_token_end"]) >= int(
        events[-2]["meta"]["text_token_end"]
    )


def test_codec_and_text_coordinates_travel_with_pcm_through_reorder():
    progress = event(session(True), 1, 1.0)
    successor = AttributedAudioChunk(
        b"successor", progress_event=progress, segment_idx=1,
        source_frame_start=0, source_frame_end=1,
    )
    reorder = AudioReorder()
    assert reorder.push(0, 1, successor) == []
    predecessor = AttributedAudioChunk(b"prefix", segment_idx=0)
    assert reorder.push(0, 0, predecessor) == [predecessor]
    delivered = reorder.mark_done(0, 0)[0]
    assert delivered is successor
    assert delivered.progress_event["meta"]["source_frame_end"] == "1"
    assert delivered.progress_event["meta"]["normalized_codepoint_end"] == "3"
    assert delivered.progress_event["meta"]["raw_codepoint_end"] == "2"
