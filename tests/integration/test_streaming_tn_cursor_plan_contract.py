from __future__ import annotations

import asyncio

import pytest

from engine.core.cursor_plan_adapter import CursorLabelPlanAdapter
from engine.core.native_cursor_labelizer import NativeCursorLabelizer
from engine.core.text_progress import NativeCursorProgressProjector
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
        committed_revision = session.cursor_plan_revision
        assert committed_revision >= 2
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
        assert max(request.cursor_label_plan.revision for request in plan_requests) == committed_revision
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
        assert session.cursor_plan_revision == committed_revision + 1
        final_plan_requests = [
            request
            for request in final_requests
            if request.type is RequestType.UPDATE_CURSOR_PLAN
        ]
        assert final_plan_requests
        assert {request.cursor_label_plan.revision for request in final_plan_requests} == {
            committed_revision + 1
        }
        assert not any(
            request.type
            in {RequestType.START_TOKENS, RequestType.APPEND_TOKENS}
            for request in final_requests
        )

        await frontend.cancel_session(session.session_id)

    asyncio.run(run())


def test_tn_expansion_projects_native_cursor_back_to_raw_coordinates():
    async def run() -> None:
        inbox: asyncio.Queue = asyncio.Queue(maxsize=128)
        frontend = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=1,
            engine_max_decode_len=256,
            max_concurrent_segments=1,
            cursor_plan_adapter_factory=lambda: CursorLabelPlanAdapter(
                lambda spoken: list(range(1, len(spoken) + 1))
            ),
        )
        session = await frontend.create_session(
            "tn-cursor-projection",
            config=SessionConfig(
                input_mode=InputMode.LONG_SEGMENT,
                group_policy=GroupPolicy.NONE,
            ),
        )
        await _drain(inbox)

        # 99% is held as an incomplete semantic span until its boundary is
        # known.  The next packet closes it and the main TN expands it to a
        # spoken form with a different length from the raw source.
        await frontend.push_text_input(session.session_id, "99%")
        await _drain(inbox)
        await frontend.push_text_input(session.session_id, "，我目前")
        await _drain(inbox)

        plan = session.cursor_label_plan
        assert plan is not None
        assert session.text_journal.raw_text == "99%，我目前"
        assert session.text_journal.normalized_text == "百分之九十九，我目前"
        assert plan.owner_spans

        first_owner = plan.owner_spans[0]
        assert first_owner.raw_start == 0
        assert first_owner.raw_end == 3
        assert first_owner.normalized_start == 0
        assert first_owner.normalized_end == len("百分之九十九")

        projector = NativeCursorProgressProjector(segment_idx=0, plan=plan)
        estimate = projector.update(mu=float(first_owner.label_end), confidence=1.0)
        assert estimate is not None
        assert estimate.raw_codepoint_end == 3
        assert estimate.normalized_codepoint_end == len("百分之九十九")
        assert estimate.raw_codepoint_end <= len(session.text_journal.raw_text)

        await frontend.cancel_session(session.session_id)

    asyncio.run(run())


def test_active_segment_label_window_grows_before_appended_tokens():
    async def run():
        inbox = asyncio.Queue(maxsize=256)
        labelizer = NativeCursorLabelizer({f"en:{char}": i + 1 for i, char in enumerate(
            "abcdefghijklmnopqrstuvwxyz"
        )})
        frontend = FrontendInterface(
            engine_inbox=inbox, tokenizer=_CharTokenizer(),
            cursor_plan_adapter_factory=lambda: CursorLabelPlanAdapter(labelizer),
        )
        session = await frontend.create_session(
            "growing-window", config=SessionConfig(
                input_mode=InputMode.TOKEN, group_policy=GroupPolicy.NONE,
            ),
        )
        await _drain(inbox)
        await frontend.push_text_input(session.session_id, "hello ")
        await _drain(inbox)
        old = session.cursor_segment_plans[0]
        await frontend.push_text_input(session.session_id, "world ")
        requests = await _drain(inbox)
        current = session.cursor_segment_plans[0]
        assert current.label_ids == labelizer("hello world")
        assert current.revision > old.revision
        update = next(i for i, request in enumerate(requests)
                      if request.type is RequestType.UPDATE_CURSOR_PLAN
                      and request.cursor_label_plan == current)
        append = next(i for i, request in enumerate(requests)
                      if request.type is RequestType.APPEND_TOKENS)
        assert update < append
        await frontend.cancel_session(session.session_id)

    asyncio.run(run())


def test_full_long_and_token_inputs_share_tn_cursor_result_set():
    async def run() -> None:
        result_sets = []
        for input_mode in (
            InputMode.FULL_TEXT,
            InputMode.LONG_SEGMENT,
            InputMode.TOKEN,
        ):
            inbox: asyncio.Queue = asyncio.Queue(maxsize=128)
            frontend = FrontendInterface(
                engine_inbox=inbox,
                tokenizer=_CharTokenizer(),
                max_sessions=1,
                engine_max_decode_len=256,
                max_concurrent_segments=2,
                cursor_plan_adapter_factory=lambda: CursorLabelPlanAdapter(
                    NativeCursorLabelizer({label: index + 1 for index, label in enumerate(
                        "wen du shi er wu".split()
                    )})
                ),
            )
            session = await frontend.create_session(
                f"tn-cursor-{input_mode.value}",
                config=SessionConfig(
                    input_mode=input_mode,
                    group_policy=GroupPolicy.NONE,
                ),
            )
            await _drain(inbox)
            raw = "温度是25度。"
            if input_mode is InputMode.FULL_TEXT:
                await frontend.feed_full_text(session.session_id, raw)
            else:
                packets = list(raw) if input_mode is InputMode.TOKEN else [raw[:3], raw[3:]]
                for packet in packets:
                    await frontend.push_text_input(session.session_id, packet)
                await frontend.mark_input_complete(session.session_id)
            requests = await _drain(inbox)
            plan = session.cursor_label_plan
            assert plan is not None
            spoken = "".join(
                chr(token)
                for request in requests
                if request.type in {RequestType.START_TOKENS, RequestType.APPEND_TOKENS}
                for token in (request.token_ids or [])
            )
            result_sets.append(
                (
                    session.text_journal.raw_text,
                    session.text_journal.normalized_text,
                    spoken,
                    plan.label_ids,
                    plan.label_normalized_spans,
                    tuple(session.text_journal.normalized_to_raw),
                )
            )
            await frontend.cancel_session(session.session_id)

        assert result_sets[0] == result_sets[1] == result_sets[2]
        assert result_sets[0][0] == "温度是25度。"
        assert result_sets[0][1] == "温度是二十五度。"
        assert result_sets[0][2] == result_sets[0][1]

    asyncio.run(run())


def test_splitter_boundary_inside_tn_owner_preserves_native_global_coordinates():
    """Ordinary segmentation must not disable a precisely mapped label plan."""

    async def run() -> None:
        inbox: asyncio.Queue = asyncio.Queue(maxsize=1024)
        frontend = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=1,
            engine_max_decode_len=80,
            prefill_len=12,
            safety_margin=2,
            max_concurrent_segments=4,
            cursor_plan_adapter_factory=lambda: CursorLabelPlanAdapter(
                NativeCursorLabelizer({label: index + 1 for index, label in enumerate(
                    "bai fen zhi jiu shi he jia yi bing ding wu ji geng xin ren gui "
                    "zi chou yin mao chen si wei shen you".split()
                )})
            ),
        )
        session = await frontend.create_session(
            "tn-cursor-splitter-owner-boundary",
            config=SessionConfig(
                input_mode=InputMode.LONG_SEGMENT,
                group_policy=GroupPolicy.NONE,
            ),
        )
        await _drain(inbox)

        # The first semantic owner expands 99% to six spoken characters.  The
        # small real splitter budget forces later punctuation segments while
        # the remaining committed text is still one owner.  Several segment
        # bounds therefore cut through that owner.
        await frontend.push_text_input(
            session.session_id,
            "99%和甲乙。丙丁。戊己。庚辛。壬癸。子丑。寅卯。辰巳。午未。申酉。",
        )
        await frontend.mark_input_complete(session.session_id)
        await _drain(inbox)

        assert len(session.segment_texts) >= 3
        assert session.cursor_label_plan is not None
        assert session.cursor_label_plan.owner_spans[0].normalized_end == len(
            "百分之九十九"
        )
        assert any(
            start < session.cursor_label_plan.owner_spans[-1].normalized_end
            and end > session.cursor_label_plan.owner_spans[-1].normalized_start
            and not (
                start <= session.cursor_label_plan.owner_spans[-1].normalized_start
                and end >= session.cursor_label_plan.owner_spans[-1].normalized_end
            )
            for start, end in session.cursor_segment_bounds.values()
        )
        # Exact label offsets preserve native computation across partial
        # owners; only the journal decides the conservative raw frontier.
        assert all(
            plan.active and plan.label_normalized_spans
            for plan in session.cursor_segment_plans.values()
        )

        # Exercise the real result-loop event builder, including pre-final
        # neural observations. No event may change to EMA merely due to cuts.
        events = []
        for segment_idx in sorted(session.segment_token_spans):
            plan = session.cursor_segment_plans[segment_idx]
            partial = frontend._make_text_progress_event(
                session, segment_idx,
                {"source_frame_start": 0, "source_frame_end": 1,
                 "cursor_valid": 1, "cursor_mu": 1.0,
                 "cursor_plan_revision": plan.revision},
            )
            assert partial["meta"]["progress_basis"] == "native_cursor_v1"
            event = frontend._make_text_progress_event(
                session,
                segment_idx,
                {
                    "source_frame_start": 1,
                    "source_frame_end": 2,
                    "text_tokens": len(session.segment_token_spans[segment_idx]),
                },
                final=True,
            )
            if event is not None:
                assert event["meta"]["progress_basis"] == "native_cursor_v1"
                events.append(event)
        assert len(events) >= 3
        raw_ends = [int(event["meta"]["raw_codepoint_end"]) for event in events]
        normalized_ends = [
            int(event["meta"]["normalized_codepoint_end"]) for event in events
        ]
        assert raw_ends == sorted(raw_ends)
        assert normalized_ends == sorted(normalized_ends)
        assert raw_ends[-1] == len(session.text_journal.raw_text)
        assert normalized_ends[-1] == len(session.text_journal.normalized_text)

        await frontend.cancel_session(session.session_id)

    asyncio.run(run())
