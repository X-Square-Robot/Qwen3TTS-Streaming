"""Regression tests for client-side latency diagnostics parsing/math."""

from __future__ import annotations

from qwen3tts.analyzers import LatencyAnalyzer
from qwen3tts.timing import ServerTimingReport


class TestServerTimingParse:
    def test_session_created_prefers_monotonic_key(self):
        # The real session-created timestamp is emitted under *_monotonic; the
        # *_epoch_ms key carries the coarse request-received approximation.
        report = ServerTimingReport.from_done_meta(
            {
                "server_session_created_monotonic": "1700000000000",
                "server_session_created_epoch_ms": "1699999990000",
            }
        )
        assert report.server_session_created_epoch_ms == 1700000000000

    def test_session_created_falls_back_for_older_server(self):
        report = ServerTimingReport.from_done_meta(
            {"server_session_created_epoch_ms": "1699999990000"}
        )
        assert report.server_session_created_epoch_ms == 1699999990000

    def test_zero_client_timestamp_is_honored(self):
        # client_request_ts_ms == 0 is a legitimate value, not "missing".
        report = ServerTimingReport(
            client_request_ts_ms=0,
            server_first_effective_audio_epoch_ms=1000,
            server_first_raw_audio_epoch_ms=900,
        )
        assert report.client_request_to_server_first_audio_ms == 1000
        assert report.client_request_to_server_first_raw_audio_ms == 900


class TestRegressionCheck:
    def test_zero_baseline_does_not_divide_by_zero(self):
        analyzer = LatencyAnalyzer()
        analyzer.add_session(ServerTimingReport(server_engine_prefill_ms=5.0))

        # baseline 0 previously raised ZeroDivisionError and aborted the scan.
        result = analyzer.regression_check({"server_engine_prefill_ms": 0.0})
        assert any("server_engine_prefill_ms" in line for line in result)

    def test_zero_baseline_zero_current_not_flagged(self):
        analyzer = LatencyAnalyzer()
        analyzer.add_session(ServerTimingReport(server_engine_prefill_ms=0.0))
        assert analyzer.regression_check({"server_engine_prefill_ms": 0.0}) == []


class TestSessionDiagnostics:
    """L0 client self-analysis: five-question summary + escalation hints."""

    _DONE_META = {
        "request_id": "r1",
        "server_session_create_to_first_raw_audio_ms": "230.0",
        "server_first_text_dequeue_to_first_raw_audio_ms": "210.0",
        "server_engine_prefill_ms": "28.0",
        "server_total_latency_ms": "1520.0",
        "server_cache_hit": "true",
        "server_cache_tokens_reused": "512",
        "server_batch_summary": '{"segments":3,"batched":2,"solo":1,"max_batch_size_seen":4}',
        "server_final_synthesized_text": "今天天气很好",
    }

    def test_summary_answers_five_questions(self):
        from qwen3tts.diagnostics import SessionDiagnostics

        d = SessionDiagnostics.from_done_meta(
            self._DONE_META,
            [{"eos_reason": "codec_eos"}, {"eos_reason": "kv_overflow"}],
        )
        s = d.summary()
        assert s["ttft"]["create_to_first_raw_ms"] == 230.0
        assert s["batch"]["batched"] == 2 and s["batch"]["max_batch_size_seen"] == 4
        assert s["synthesized_text"] == "今天天气很好"
        assert s["cache"]["hit"] is True
        assert s["eos_reasons"] == ["codec_eos", "kv_overflow"]

    def test_explain_emits_escalation_hints(self):
        from qwen3tts.diagnostics import SessionDiagnostics

        d = SessionDiagnostics.from_done_meta(
            self._DONE_META, [{"eos_reason": "kv_overflow"}]
        )
        ex = d.explain()
        # abnormal ending -> segment_synthesis; inference-dominant -> split_decision
        assert "segment_synthesis" in ex
        assert "split_decision" in ex
        assert "obs_level=debug" in ex

    def test_from_messages_picks_done_and_segment_events(self):
        from qwen3tts.diagnostics import SessionDiagnostics

        class Msg:
            def __init__(self, type, meta):
                self.type = type
                self.meta = meta

        msgs = [
            Msg("segment_end", {"eos_reason": "codec_eos"}),
            Msg("audio", {"chunk_index": "0"}),
            Msg("done", self._DONE_META),
        ]
        d = SessionDiagnostics.from_messages(msgs)
        assert d.summary()["synthesized_text"] == "今天天气很好"
        assert d.summary()["eos_reasons"] == ["codec_eos"]
