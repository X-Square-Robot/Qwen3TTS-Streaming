from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

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


class _IdentityNormalizer:
    def normalize(self, text: str) -> str:
        return text


def _external_x2():
    root_value = os.environ.get("X2STREAMING_ROOT", "").strip()
    if not root_value:
        pytest.skip("X2STREAMING_ROOT is not set")
    root = Path(root_value).resolve()
    src = root / "src"
    if not (src / "x2streaming_tts").is_dir():
        pytest.fail(f"X2Streaming source package is missing at {src}")
    sys.path.insert(0, str(src))
    from x2streaming_tts import X2StreamingPolicy
    from x2streaming_tts.adapters.qwen3tts_streaming import build_policy_factories

    return X2StreamingPolicy(text_normalizer=_IdentityNormalizer()), build_policy_factories


async def _drain(queue: asyncio.Queue) -> list:
    values = []
    while not queue.empty():
        values.append(await queue.get())
    return values


@pytest.mark.parametrize(
    "input_mode",
    [InputMode.FULL_TEXT, InputMode.LONG_SEGMENT, InputMode.TOKEN],
)
def test_real_x2_policy_consumes_main_tn_projection_across_input_modes(input_mode):
    policy, build_policy_factories = _external_x2()
    from engine.core.extensions import EngineExtensions

    async def run() -> None:
        inbox: asyncio.Queue = asyncio.Queue(maxsize=128)
        extensions = build_policy_factories(policy).to_upstream(EngineExtensions)
        frontend = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=256,
            max_concurrent_segments=2,
            extensions=extensions,
        )
        session = await frontend.create_session(
            f"x2-real-{input_mode.value}",
            config=SessionConfig(
                input_mode=input_mode,
                group_policy=GroupPolicy.NONE,
            ),
        )
        text = "温度是25度。"
        if input_mode is InputMode.FULL_TEXT:
            await frontend.feed_full_text(session.session_id, text)
        else:
            await frontend.push_text_input(session.session_id, text[:3])
            await frontend.push_text_input(session.session_id, text[3:])
            await frontend.mark_input_complete(session.session_id)

        assert session.text_journal.normalized_text == "温度是二十五度。"
        requests = await _drain(inbox)
        spoken_by_segment: dict[int, str] = {}
        for request in requests:
            if request.type not in {
                RequestType.START_TOKENS,
                RequestType.APPEND_TOKENS,
            }:
                continue
            spoken_by_segment[request.segment_idx] = spoken_by_segment.get(
                request.segment_idx, ""
            ) + "".join(chr(token) for token in (request.token_ids or []))
        assert "".join(spoken_by_segment.values()) == "温度是二十五度。"
        await frontend.cancel_session(session.session_id)

    asyncio.run(run())


def test_real_x2_policy_keeps_an_incomplete_tn_tail_out_of_the_splitter():
    """An open semantic span is held until later text resolves it.

    This is the long-text streaming case: the X2 adapter may observe committed
    main-TN facts, but it must not turn ``99%`` into a label/audio request before
    the following Chinese context makes the pronunciation deterministic.
    """

    policy, build_policy_factories = _external_x2()
    from engine.core.extensions import EngineExtensions

    async def run() -> None:
        inbox: asyncio.Queue = asyncio.Queue(maxsize=128)
        extensions = build_policy_factories(policy).to_upstream(EngineExtensions)
        frontend = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=256,
            max_concurrent_segments=2,
            extensions=extensions,
        )
        session = await frontend.create_session(
            "x2-real-incomplete-long",
            config=SessionConfig(
                input_mode=InputMode.LONG_SEGMENT,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await frontend.push_text_input(session.session_id, "99%")
        assert session.text_committer.pending_raw == "99%"
        first_requests = await _drain(inbox)
        assert not any(
            request.type in {RequestType.START_TOKENS, RequestType.APPEND_TOKENS}
            for request in first_requests
        )

        await frontend.push_text_input(session.session_id, "，我目前")
        second_requests = await _drain(inbox)
        second_spoken = "".join(
            chr(token)
            for request in second_requests
            if request.type in {RequestType.START_TOKENS, RequestType.APPEND_TOKENS}
            for token in (request.token_ids or [])
        )
        assert session.text_committer.pending_raw == ""
        assert second_spoken == "百分之九十九，我目前"

        await frontend.mark_input_complete(session.session_id)
        requests = await _drain(inbox)
        assert not any(
            request.type in {RequestType.START_TOKENS, RequestType.APPEND_TOKENS}
            for request in requests
        )
        await frontend.cancel_session(session.session_id)

    asyncio.run(run())
