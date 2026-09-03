from engine.frontend.text_commitment.throughput import AudioCreditEstimator


def test_audio_credit_tracks_codec_frames_separately_from_text_tokens():
    e = AudioCreditEstimator(codec_frame_rate=12.5, reserve_ms=20)
    e.observe_raw_tokens(10)
    e.observe_normalized_tokens(20, service_ms=20)
    e.observe_codec_frames(25)
    e.observe_played_audio(400)
    e.set_reserves(recovery_ms=30, jitter_ms=10)
    s = e.snapshot()
    assert s.raw_tokens == 10
    assert s.normalized_tokens == 20
    assert s.codec_frames == 25
    assert s.generated_audio_ms == 2000
    assert s.audio_credit_ms == 1580
    assert s.safe_wait_ms == 1540
