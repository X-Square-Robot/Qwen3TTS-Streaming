import asyncio

from engine.core.types import EngineResult, GroupPolicy, InputMode, OutputPolicyConfig, ResultType, SessionConfig
from engine.frontend.interface import FrontendInterface
from engine.frontend.text_commitment.start_gate import SemanticStartGate


class _CharTokenizer:
    def encode_ids(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def test_semantic_gate_hides_pending_audio_until_resolution():
    now = [0.0]
    gate = SemanticStartGate(bytes_per_sec=10.0, min_audio_ms=100.0, max_hold_ms=300.0, time_fn=lambda: now[0])
    gate.push([b"12345"])
    assert gate.release_due(semantic_pending=True) == []
    now[0] = 0.1
    gate.push([b"67890"])
    assert gate.release_due(semantic_pending=False) == [b"12345", b"67890"]
    assert gate.released


def test_semantic_gate_max_hold_releases_even_if_pending():
    now = [0.0]
    gate = SemanticStartGate(bytes_per_sec=100.0, min_audio_ms=1000.0, max_hold_ms=50.0, time_fn=lambda: now[0])
    gate.push([b"x"])
    now[0] = 0.051
    assert gate.release_due(semantic_pending=True) == [b"x"]


def test_frontend_phase1_gate_releases_after_max_hold():
    async def run():
        inbox = asyncio.Queue()
        received = []

        async def on_audio(_sid, chunk):
            received.append(bytes(chunk))

        interface = FrontendInterface(engine_inbox=inbox, tokenizer=_CharTokenizer(), max_sessions=1)
        session = await interface.create_session(
            "gate",
            config=SessionConfig(
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
                output_policy=OutputPolicyConfig(
                    config={"semantic_start_gate": True, "semantic_start_min_audio_ms": 1000, "semantic_start_max_hold_ms": 40}
                ),
            ),
            on_audio=on_audio,
        )
        await session.result_queue.put(
            EngineResult(
                type=ResultType.AUDIO_CHUNK, session_id="gate", audio_bytes=b"frame"
            )
        )
        await asyncio.sleep(0.08)
        assert received == [b"frame"]
        await session.result_queue.put(
            EngineResult(
                type=ResultType.SESSION_DONE, session_id="gate"
            )
        )
        await asyncio.sleep(0)

    asyncio.run(run())
