from __future__ import annotations

import asyncio

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

    def test_check_send_open_raises_after_close(self):
        session = BaseStreamSession(session_id="s4", transport="test")
        session._mark_send_closed()
        with pytest.raises(StreamClosedError, match="already closed"):
            session._check_send_open()


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
