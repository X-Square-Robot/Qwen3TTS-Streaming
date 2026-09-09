from __future__ import annotations

import asyncio

import pytest

from engine.core.cursor_plan_adapter import CursorLabelPlanAdapter
from engine.core.types import GroupPolicy, InputMode, RequestType, SessionConfig
from engine.frontend.interface import FrontendInterface

pytestmark = pytest.mark.integration


class _CharTokenizer:
    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def encode_with_text(
        self, text: str, add_special_tokens: bool = False
    ) -> tuple[list[int], list[str]]:
        return self.encode_ids(text, add_special_tokens), list(text)


async def _drain(queue: asyncio.Queue) -> list:
    values = []
    while not queue.empty():
        values.append(await queue.get())
    return values


def test_streaming_tn_plan_precedes_tokens_and_final_flush_does_not_duplicate():
    async def run() -> None:
        inbox: asyncio.Queue = asyncio.Queue(maxsize=128)
        frontend = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=256,
            max_concurrent_segments=2,
            cursor_plan_adapter_factory=lambda: CursorLabelPlanAdapter(
                lambda spoken: list(range(1, len(spoken) + 1))
            ),
        )
        session = await frontend.create_session(
            "tn-cursor-order",
            config=SessionConfig(
                input_mode=InputMode.LONG_SEGMENT,
                group_policy=GroupPolicy.NONE,
            ),
        )
        await _drain(inbox)

        # Establish an already-admitted segment so the later resolved tail
        # exercises the active-segment UPDATE_CURSOR_PLAN path as well as the
        # new successor START_TOKENS path.
        await frontend.push_text_input(session.session_id, "你好。")
        prefix_requests = await _drain(inbox)
        prefix_plan_requests = [
            request
            for request in prefix_requests
            if request.type is RequestType.START_TOKENS
        ]
        assert prefix_plan_requests
        assert prefix_plan_requests[0].cursor_label_plan.revision == 1

        await frontend.push_text_input(session.session_id, "99%")
        pending_requests = await _drain(inbox)
        assert session.text_committer.pending_raw == "99%"
        assert session.cursor_plan_revision == 1
        assert not any(
            request.type
            in {
                RequestType.UPDATE_CURSOR_PLAN,
                RequestType.START_TOKENS,
                RequestType.APPEND_TOKENS,
            }
            for request in pending_requests
        )

        await frontend.push_text_input(session.session_id, "，我目前")
        committed_requests = await _drain(inbox)
        assert session.text_committer.pending_raw == ""
        assert session.cursor_plan_revision == 2
        assert session.text_journal.normalized_text == "你好。百分之九十九，我目前"

        plan_requests = [
            request
            for request in committed_requests
            if request.type is RequestType.UPDATE_CURSOR_PLAN
        ]
        token_requests = [
            request
            for request in committed_requests
            if request.type
            in {RequestType.START_TOKENS, RequestType.APPEND_TOKENS}
        ]
        assert plan_requests
        assert {request.cursor_label_plan.revision for request in plan_requests} == {2}
        assert committed_requests.index(plan_requests[0]) < committed_requests.index(
            token_requests[0]
        )
        spoken = "".join(
            chr(token)
            for request in token_requests
            for token in (request.token_ids or [])
        )
        assert spoken == "百分之九十九，我目前"

        await frontend.mark_input_complete(session.session_id)
        final_requests = await _drain(inbox)
        assert session.cursor_plan_revision == 3
        final_plan_requests = [
            request
            for request in final_requests
            if request.type is RequestType.UPDATE_CURSOR_PLAN
        ]
        assert final_plan_requests
        assert {request.cursor_label_plan.revision for request in final_plan_requests} == {3}
        assert not any(
            request.type
            in {RequestType.START_TOKENS, RequestType.APPEND_TOKENS}
            for request in final_requests
        )

        await frontend.cancel_session(session.session_id)

    asyncio.run(run())
