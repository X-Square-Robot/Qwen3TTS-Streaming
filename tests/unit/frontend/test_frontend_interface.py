from __future__ import annotations

import asyncio
import json
import logging

from engine.core.types import (
    EngineResult,
    GroupPolicy,
    InputMode,
    RequestType,
    ResultType,
    SessionConfig,
)
from engine.frontend.interface import FrontendInterface, _normalize_tts_text


def test_tts_text_normalization_strips_emoji_noise():
    assert _normalize_tts_text("你好😊，世界🌍！") == "你好，世界！"
    assert _normalize_tts_text("good😊morning") == "good morning"
    assert _normalize_tts_text("第1️⃣步完成✅。") == "第步完成。"
    assert _normalize_tts_text("😊🚀") == ""
    assert _normalize_tts_text("湿度19%，气温为23℃。") == "湿度19%，气温为23℃。"
    assert _normalize_tts_text("湿度19%.") == "湿度19%."


class _CharTokenizer:
    def encode_ids(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def encode_with_text(self, text, add_special_tokens=False):
        ids = self.encode_ids(text, add_special_tokens=add_special_tokens)
        return ids, list(text)

    def debug_snapshot(self, text, add_special_tokens=False):
        ids = self.encode_ids(text, add_special_tokens=add_special_tokens)
        return {
            "text": text,
            "ids": ids,
            "pieces": [
                {
                    "index": idx,
                    "id": token_id,
                    "token": ch,
                    "span": ch,
                    "offset": [idx, idx + 1],
                }
                for idx, (token_id, ch) in enumerate(zip(ids, text))
            ],
        }


async def _drain_requests(inbox: asyncio.Queue) -> list:
    requests = []
    while not inbox.empty():
        requests.append(await inbox.get())
    return requests


def test_token_mode_preserves_whitespace_only_chunks():
    async def run():
        inbox = asyncio.Queue(maxsize=16)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=64,
        )

        session = await interface.create_session(
            "space-token",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.push_text_input("space-token", " ")

        requests = []
        while not inbox.empty():
            requests.append(await inbox.get())

        assert [request.type for request in requests] == [
            RequestType.NEW_SESSION,
            RequestType.START_TOKENS,
        ]
        assert requests[-1].token_ids == [ord(" ")]

        await session.result_queue.put(
            EngineResult(type=ResultType.SESSION_DONE, session_id="space-token")
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_token_mode_serial_segments_defers_session_done_until_buffer_drains():
    async def run():
        inbox = asyncio.Queue(maxsize=128)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=40,
            ema_ratio=2.0,
            max_concurrent_segments=1,
        )

        session = await interface.create_session(
            "serial-token",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.push_text_input(
            "serial-token",
            "第一句很长很长很长。第二句也很长很长很长。",
        )
        await interface.mark_input_complete("serial-token")

        initial = await _drain_requests(inbox)
        assert any(
            request.type == RequestType.SEGMENT_TOKENS_DONE
            and request.segment_idx == 0
            for request in initial
        )
        assert not any(request.type == RequestType.SESSION_TOKENS_DONE for request in initial)

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="serial-token",
                segment_idx=0,
                metrics={"audio_steps": 10, "text_tokens": 10},
            )
        )
        await asyncio.sleep(0)

        after_first_segment = await _drain_requests(inbox)
        request_types = [request.type for request in after_first_segment]
        second_start_idx = request_types.index(RequestType.START_TOKENS)
        session_done_idx = request_types.index(RequestType.SESSION_TOKENS_DONE)
        assert second_start_idx < session_done_idx
        assert after_first_segment[second_start_idx].segment_idx == 1

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="serial-token",
                segment_idx=1,
                metrics={"audio_steps": 10, "text_tokens": 10},
            )
        )
        await asyncio.sleep(0)

        assert await _drain_requests(inbox) == []

        await session.result_queue.put(
            EngineResult(type=ResultType.SESSION_DONE, session_id="serial-token")
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_token_mode_final_punct_flush_sends_session_done_without_empty_segment():
    async def run():
        inbox = asyncio.Queue(maxsize=128)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=100,
            ema_ratio=10.0,
            max_concurrent_segments=2,
        )

        session = await interface.create_session(
            "boundary-token",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.push_text_input("boundary-token", "甲乙丙丁戊己。")
        await interface.mark_input_complete("boundary-token")

        initial = await _drain_requests(inbox)
        assert any(
            request.type == RequestType.SEGMENT_TOKENS_DONE
            and request.segment_idx == 0
            for request in initial
        )
        assert any(request.type == RequestType.SESSION_TOKENS_DONE for request in initial)
        assert not any(
            request.type == RequestType.START_TOKENS
            and request.segment_idx == 1
            for request in initial
        )

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="boundary-token",
                segment_idx=0,
                metrics={"audio_steps": 10, "text_tokens": 7},
            )
        )
        await asyncio.sleep(0)

        assert await _drain_requests(inbox) == []

        await session.result_queue.put(
            EngineResult(type=ResultType.SESSION_DONE, session_id="boundary-token")
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_prefill_done_event_exposes_reference_metadata():
    async def run():
        inbox = asyncio.Queue(maxsize=16)
        events = []

        async def on_event(sid: str, event: dict):
            events.append((sid, event))

        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=64,
        )

        session = await interface.create_session(
            "icl-meta",
            config=SessionConfig(
                task_type="voice_clone",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
            on_event=on_event,
        )

        await session.result_queue.put(
            EngineResult(
                type=ResultType.PREFILL_DONE,
                session_id="icl-meta",
                segment_idx=0,
                metrics={
                    "ref_source": "registry",
                    "ref_id": "Vivian",
                    "ref_audio_sha256": "abcdef123456",
                    "icl_cache_hit": "true",
                    "ref_preprocess_runtime": "trt",
                },
            )
        )
        await asyncio.sleep(0)

        assert events == [
            (
                "icl-meta",
                {
                    "type": "prefill_done",
                    "segment_idx": 0,
                    "text": "",
                    "meta": {
                        "ref_source": "registry",
                        "ref_id": "Vivian",
                        "ref_audio_sha256": "abcdef123456",
                        "icl_cache_hit": "true",
                        "ref_preprocess_runtime": "trt",
                    },
                },
            )
        ]

        await session.result_queue.put(
            EngineResult(type=ResultType.SESSION_DONE, session_id="icl-meta")
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_tokenizer_observability_logs_raw_and_normalized_text(caplog):
    interface = FrontendInterface(
        engine_inbox=asyncio.Queue(maxsize=16),
        tokenizer=_CharTokenizer(),
        max_sessions=2,
        engine_max_decode_len=64,
    )

    with caplog.at_level(logging.DEBUG, logger="engine.frontend.interface"):
        interface._tokenize_segment_text("湿度19%，气温为23℃。")

    observability_logs = [
        record.message for record in caplog.records
        if "Tokenizer observability:" in record.message
    ]
    assert observability_logs, "expected tokenizer observability logs"
    payload = json.loads(observability_logs[-1].split("Tokenizer observability: ", 1)[1])
    assert payload["field"] == "segment_text"
    assert payload["normalized_text"] == "湿度19%，气温为23℃。"
    assert payload["tokenizer"]["ids"]
    assert payload["tokenizer"]["pieces"]
