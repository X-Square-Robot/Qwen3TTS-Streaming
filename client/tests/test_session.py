from __future__ import annotations

import asyncio
import threading
import time

import pytest

from qwen3tts_protocol import AudioChunk, AudioFormat, StreamEvent
from qwen3tts._session import BaseStreamSession, AsyncStreamSession
from qwen3tts.exceptions import StreamClosedError


class TestBaseStreamSession:
    def test_put_and_iter_messages(self):
        session = BaseStreamSession(session_id="s1", transport="test")
        session._put_message(StreamEvent(type="start", session_id="s1"))
        session._put_message(AudioChunk(pcm_bytes=b"\x00\x01", audio=AudioFormat()))
        session._put_message(StreamEvent(type="done", session_id="s1"))

        messages = list(session.iter_messages())
        assert len(messages) == 3
        assert messages[0].type == "start"
        assert isinstance(messages[1], AudioChunk)
        assert messages[2].type == "done"

    def test_terminal_event_closes_queue(self):
        session = BaseStreamSession(session_id="s2", transport="test")
        session._put_message(StreamEvent(type="done", session_id="s2"))

        messages = list(session.iter_messages())
        assert len(messages) == 1
        assert session._closed is True

    def test_error_is_terminal(self):
        session = BaseStreamSession(session_id="s3", transport="test")
        session._put_message(StreamEvent(type="error", session_id="s3", message="fail"))

        messages = list(session.iter_messages())
        assert len(messages) == 1
        assert session._closed is True

    def test_invalid_text_progress_does_not_terminate_audio_stream(self):
        session = BaseStreamSession(session_id="s-progress", transport="test")
        session._put_message(
            AudioChunk(
                pcm_bytes=b"\x00" * 8,
                audio=AudioFormat(sample_rate=24000),
                output_sample_start=0,
                output_sample_end=20,
            )
        )
        session._put_message(
            StreamEvent(
                type="text_progress",
                session_id="s-progress",
                meta={
                    "anchor_seq": "1",
                    "output_sample_start": "0",
                    "output_sample_end": "10",
                    "output_sample_rate": "24000",
                    "raw_codepoint_start": "0",
                    "raw_codepoint_end": "2",
                    "normalized_codepoint_start": "0",
                    "normalized_codepoint_end": "2",
                },
            )
        )

        # The second anchor regresses its raw end, matching the production
        # mixed-segment failure. It is retained as a diagnostic message, but
        # must not turn the stream into a terminal error.
        session._put_message(
            StreamEvent(
                type="text_progress",
                session_id="s-progress",
                meta={
                    "anchor_seq": "2",
                    "output_sample_start": "10",
                    "output_sample_end": "20",
                    "output_sample_rate": "24000",
                    "raw_codepoint_start": "1",
                    "raw_codepoint_end": "1",
                    "normalized_codepoint_start": "1",
                    "normalized_codepoint_end": "1",
                },
            )
        )
        session._put_message(StreamEvent(type="done", session_id="s-progress"))

        messages = list(session.iter_messages())
        assert [type(message).__name__ for message in messages] == [
            "AudioChunk",
            "StreamEvent",
            "StreamEvent",
            "StreamEvent",
        ]
        assert messages[-1].type == "done"
        assert session.progress_tracking_degraded is True

    def test_check_send_open_raises_after_close(self):
        session = BaseStreamSession(session_id="s4", transport="test")
        session._mark_send_closed()
        with pytest.raises(StreamClosedError, match="already closed"):
            session._check_send_open()

    def test_close_cancels_and_unblocks_iter_messages(self):
        session = BaseStreamSession(session_id="s-close", transport="test")
        cancel_reasons = []
        session.cancel = lambda reason="": cancel_reasons.append(reason)

        session.close(reason="worker shutdown")

        assert cancel_reasons == ["worker shutdown"]
        assert list(session.iter_messages()) == []
        assert session._closed is True

    def test_stop_delegates_to_end(self):
        session = BaseStreamSession(session_id="s-stop", transport="test")
        timestamps = []
        session.end = lambda **kwargs: timestamps.append(
            kwargs.get("client_timestamp_ms")
        )

        session.stop(client_timestamp_ms=123)

        assert timestamps == [123]

    def test_post_send_idle_timeout_does_not_count_pre_end_silence(self):
        session = BaseStreamSession(session_id="s-wait", transport="test")

        def finish_later():
            time.sleep(0.08)
            session._put_message(StreamEvent(type="done", session_id="s-wait"))

        producer = threading.Thread(target=finish_later, daemon=True)
        producer.start()
        messages = list(session.iter_messages(post_send_idle_timeout=0.02))
        producer.join(timeout=1.0)

        assert [message.type for message in messages] == ["done"]

    def test_post_send_idle_timeout_fires_after_end_without_terminal(self):
        session = BaseStreamSession(session_id="s-idle", transport="test")
        session._mark_send_closed()

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="after send side closed"):
            list(session.iter_messages(post_send_idle_timeout=0.03))

        assert time.monotonic() - started < 0.5


class TestAsyncStreamSession:
    def test_async_iter_messages(self):
        sync_session = BaseStreamSession(session_id="s1", transport="test")
        sync_session._put_message(StreamEvent(type="start", session_id="s1"))
        sync_session._put_message(StreamEvent(type="done", session_id="s1"))

        async def _run():
            session = AsyncStreamSession(sync_session)
            messages = []
            async for msg in session.aiter_messages():
                messages.append(msg)
            return messages

        messages = asyncio.get_event_loop().run_until_complete(_run())
        assert len(messages) == 2
        assert messages[0].type == "start"
        assert messages[1].type == "done"

    def test_async_iter_messages_passes_post_send_idle_timeout(self):
        sync_session = BaseStreamSession(session_id="s-idle-async", transport="test")
        sync_session._mark_send_closed()

        async def _run():
            session = AsyncStreamSession(sync_session)
            with pytest.raises(TimeoutError, match="after send side closed"):
                async for _message in session.aiter_messages(
                    post_send_idle_timeout=0.03
                ):
                    pass

        asyncio.get_event_loop().run_until_complete(_run())

    def test_close_still_unblocks_when_cancel_fails(self):
        session = BaseStreamSession(session_id="s-close-fail", transport="test")

        def fail_cancel(reason=""):
            raise RuntimeError("transport already gone")

        session.cancel = fail_cancel
        session.close(reason="worker shutdown")

        assert list(session.iter_messages()) == []
        with pytest.raises(StreamClosedError, match="already closed"):
            session._check_send_open()

    def test_async_delegates_to_sync(self):
        sync_session = BaseStreamSession(session_id="s1", transport="test")
        sync_session.send_text = lambda text, **kw: None
        sync_session.end = lambda **kw: None
        sync_session.cancel = lambda reason="": None

        async def _run():
            session = AsyncStreamSession(sync_session)
            assert session.session_id == "s1"
            assert session.transport == "test"
            assert session.degraded_to_oneshot is False

        asyncio.get_event_loop().run_until_complete(_run())

    def test_async_close_delegates_to_sync(self):
        sync_session = BaseStreamSession(session_id="s1", transport="test")
        close_reasons = []
        sync_session.close = lambda reason="": close_reasons.append(reason)

        async def _run():
            session = AsyncStreamSession(sync_session)
            await session.aclose(reason="async shutdown")

        asyncio.get_event_loop().run_until_complete(_run())
        assert close_reasons == ["async shutdown"]

    def test_async_stop_delegates_to_sync(self):
        sync_session = BaseStreamSession(session_id="s-stop", transport="test")
        timestamps = []
        sync_session.stop = lambda **kwargs: timestamps.append(
            kwargs.get("client_timestamp_ms")
        )

        async def _run():
            session = AsyncStreamSession(sync_session)
            await session.stop(client_timestamp_ms=456)

        asyncio.get_event_loop().run_until_complete(_run())
        assert timestamps == [456]
