from __future__ import annotations

import asyncio

from engine.core.types import (
    EngineResult,
    GroupPolicy,
    InputMode,
    OutputPolicyConfig,
    RequestType,
    ResultType,
    SessionConfig,
)
from engine.frontend.interface import FrontendInterface
from engine.frontend.text_commitment.types import (
    TextNormalizationConfig as FrozenTextNormalizationConfig,
)


class _CharTokenizer:
    def encode_ids(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def encode_with_text(self, text, add_special_tokens=False):
        return self.encode_ids(text, add_special_tokens=add_special_tokens), list(text)


async def _drain(queue):
    values = []
    while not queue.empty():
        values.append(queue.get_nowait())
    return values


async def _stop_consumer(session) -> None:
    await session.result_queue.put(
        EngineResult(type=ResultType.SESSION_DONE, session_id=session.session_id)
    )
    # Let the frontend's result consumer observe SESSION_DONE and clean up its
    # callback task before the event loop closes.
    await asyncio.sleep(0)


def test_full_text_packetized_tn_reaches_tokenizer_and_keeps_raw_map():
    async def run():
        inbox = asyncio.Queue(maxsize=256)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=128,
        )
        session = await interface.create_session(
            "full-tn-map",
            config=SessionConfig(
                language="zh",
                input_mode=InputMode.FULL_TEXT,
                group_policy=GroupPolicy.NONE,
            ),
        )

        # FULL_TEXT packets are retained as raw input until the one final TN
        # flush.  In particular, the expanded spoken form must not be replaced
        # by the journal's pre-TN snapshot at completion.
        await interface.push_text_input(session.session_id, "是99")
        await interface.push_text_input(session.session_id, "%的概率")
        await interface.mark_input_complete(session.session_id)

        requests = await _drain(inbox)
        token_text = "".join(
            chr(token_id)
            for request in requests
            for token_id in (request.token_ids or [])
        )
        assert token_text == "是百分之九十九的概率"
        assert session.text_journal.raw_text == "是99%的概率"
        assert session.text_journal.normalized_text == token_text
        # The backend currently exposes a span-level (rather than
        # character-level) alignment for verbalization expansions; the whole
        # spoken phrase still points back to the original ``99%`` interval.
        assert session.text_journal.raw_span(1, 7) == (1, 4)
        assert any(request.type is RequestType.SESSION_TOKENS_DONE for request in requests)

        await _stop_consumer(session)

    asyncio.run(run())


def test_feed_full_text_direct_call_uses_raw_argument_for_tn():
    async def run():
        inbox = asyncio.Queue(maxsize=256)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=128,
        )
        session = await interface.create_session(
            "direct-full-tn",
            config=SessionConfig(
                language="en",
                input_mode=InputMode.FULL_TEXT,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.feed_full_text(session.session_id, "20% of 100 is 20")
        requests = await _drain(inbox)
        token_text = "".join(
            chr(token_id)
            for request in requests
            for token_id in (request.token_ids or [])
        )
        assert token_text == "twenty percent of one hundred is twenty"
        assert session.text_journal.raw_text == "20% of 100 is 20"
        assert session.text_journal.normalized_text == token_text

        await _stop_consumer(session)

    asyncio.run(run())


def test_streaming_tn_keeps_raw_source_while_committing_spoken_expansion():
    async def run():
        inbox = asyncio.Queue(maxsize=256)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=128,
        )
        session = await interface.create_session(
            "stream-tn-map",
            config=SessionConfig(
                language="zh",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.push_text_input(session.session_id, "是99")
        # The trailing digit may be an incomplete keycap prefix, but it is
        # still part of the raw source coordinate space immediately.
        assert session.text_journal.raw_text == "是99"
        assert session.text_journal.normalized_text == "是"

        await interface.push_text_input(session.session_id, "%的概率")
        assert session.text_journal.raw_text == "是99%的概率"
        assert session.text_journal.normalized_text == "是百分之九十九的概率"
        assert session.text_journal.raw_span(1, 7) == (1, 4)

        await interface.mark_input_complete(session.session_id)
        assert session.text_journal.input_final is True
        assert session.text_journal.normalized_text == "是百分之九十九的概率"
        assert any(
            request.type is RequestType.SESSION_TOKENS_DONE
            for request in await _drain(inbox)
        )
        await _stop_consumer(session)

    asyncio.run(run())


def test_frozen_committer_config_accepts_session_override_without_mutation():
    async def run():
        inbox = asyncio.Queue(maxsize=256)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=128,
        )
        frozen = FrozenTextNormalizationConfig(enabled=True, language="en")
        session = await interface.create_session(
            "frozen-tn-config",
            config=SessionConfig(
                language="en",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
                text_normalization=frozen,
                output_policy=OutputPolicyConfig(config={"tn_enabled": False}),
            ),
        )
        assert session.text_committer.config.enabled is False
        assert frozen.enabled is True
        await _stop_consumer(session)

    asyncio.run(run())
