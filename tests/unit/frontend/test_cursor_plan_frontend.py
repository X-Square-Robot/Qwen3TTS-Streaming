import asyncio
from types import SimpleNamespace

from engine.core.cursor_plan_adapter import CursorLabelPlanAdapter
from engine.core.session import SegmentOrderMeta
from engine.core.types import RequestType
from engine.frontend.interface import FrontendInterface


class _Tokenizer:
    def encode_ids(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def encode_with_text(self, text, add_special_tokens=False):
        ids = self.encode_ids(text, add_special_tokens=add_special_tokens)
        return ids, list(text)


def _commit(
    text: str,
    *,
    raw_start: int = 0,
    raw_end: int | None = None,
    commit_id: int = 1,
    span_id: int = 1,
):
    return SimpleNamespace(
        raw_start=raw_start,
        raw_end=len(text) if raw_end is None else raw_end,
        tts_text=text,
        mapping=((0, len(text)),),
        commit_id=commit_id,
        span_id=span_id,
    )


def test_injected_plan_adapter_publishes_typed_update_request():
    async def run():
        inbox = asyncio.Queue(maxsize=64)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_Tokenizer(),
            cursor_plan_adapter_factory=lambda: CursorLabelPlanAdapter(
                lambda text: [len(text)]
            ),
        )
        session = await interface.create_session("s")
        # Discard NEW_SESSION; the segment order represents a segment already
        # admitted by the frontend and therefore needs a live plan update.
        await inbox.get()
        session.segment_order[3] = SegmentOrderMeta(3, 0)
        session.cursor_segment_bounds[3] = (0, 2)

        await interface._ingest_commits(session, [_commit("你好")])
        requests = [inbox.get_nowait() for _ in range(inbox.qsize())]
        request = next(
            item
            for item in requests
            if item.type is RequestType.UPDATE_CURSOR_PLAN and item.segment_idx == 3
        )
        assert request.type is RequestType.UPDATE_CURSOR_PLAN
        assert request.segment_idx == 3
        assert request.cursor_label_plan is session.cursor_segment_plans[3]
        assert request.cursor_label_plan.label_ids == (2,)

        await interface.cancel_session("s")

    asyncio.run(run())


def test_rebuilt_plan_cannot_rewrite_published_prefix():
    async def run():
        inbox = asyncio.Queue(maxsize=64)
        labelizer = lambda text: [ord(char) for char in text]
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_Tokenizer(),
            cursor_plan_adapter_factory=lambda: CursorLabelPlanAdapter(labelizer),
        )
        session = await interface.create_session("s")
        await inbox.get()
        session.segment_order[0] = SegmentOrderMeta(0, 0)
        session.cursor_segment_bounds[0] = (0, 2)

        await interface._ingest_commits(session, [_commit("a")])
        first_update = next(
            item
            for item in (inbox.get_nowait() for _ in range(inbox.qsize()))
            if item.type is RequestType.UPDATE_CURSOR_PLAN
        )

        # Simulate a bad model labelizer or owner reconstruction on a later
        # revision. The already-published prefix must fail closed.
        labelizer = lambda text: [ord(char) + 1 for char in text]
        await interface._ingest_commits(
            session,
            [_commit("b", raw_start=1, raw_end=2, commit_id=2, span_id=2)],
        )

        assert first_update.cursor_label_plan.label_ids == (ord("a"),)
        assert session.cursor_label_plan.label_ids == (ord("a"),)
        assert not any(
            item.type is RequestType.UPDATE_CURSOR_PLAN
            for item in (inbox.get_nowait() for _ in range(inbox.qsize()))
        )
        assert session.session_id not in interface._cursor_plan_adapters

        await interface.cancel_session("s")

    asyncio.run(run())
