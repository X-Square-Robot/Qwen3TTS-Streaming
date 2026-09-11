from engine.core.session import Session
from engine.core.timing import ServerTimingAccumulator


def test_session_first_raw_audio_timestamp_is_kept_across_segments():
    session = Session("session")

    session.record_first_audio(10.0)
    session.record_first_audio(20.0)

    assert session.first_raw_audio_at == 10.0


def test_server_prefill_completion_timestamp_is_session_first_event():
    accumulator = ServerTimingAccumulator()

    accumulator.record_prefill_completed(10.0)
    accumulator.record_prefill_completed(20.0)

    assert accumulator.prefill_completed_monotonic == 10.0
