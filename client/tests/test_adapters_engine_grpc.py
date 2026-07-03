"""Regression test: EngineGrpcAdapter must reuse one gRPC channel across
sessions instead of opening (and closing) a fresh channel per session.

A channel already multiplexes many streams over one HTTP/2 connection, so
tearing it down after every ``start -> text -> end`` turn paid a needless
TCP+HTTP/2 handshake per turn.
"""

from __future__ import annotations

import pytest

pytest.importorskip("grpc")
import grpc
from concurrent import futures

from qwen3tts._adapters.engine_grpc import EngineGrpcAdapter
from qwen3tts._proto import tts_pb2, tts_pb2_grpc
from qwen3tts_protocol import SessionStartRequest, SynthesisConfig


class _FakeTTSService(tts_pb2_grpc.TTSServiceServicer):
    """No-op backend: acks start/text/end without touching a model."""

    def SynthesizeStream(self, request_iterator, context):
        session_id = ""
        for request in request_iterator:
            which = request.WhichOneof("request")
            if which in ("start", "init"):
                session_id = (
                    request.start.session_id
                    if which == "start"
                    else request.init.session_id
                )
            elif which in ("end", "done"):
                yield tts_pb2.SynthesizeResponse(
                    event=tts_pb2.StreamEvent(type="done", session_id=session_id)
                )
                return
            elif which == "cancel":
                return


@pytest.fixture
def fake_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tts_pb2_grpc.add_TTSServiceServicer_to_server(_FakeTTSService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.stop(None)


def _run_session(adapter: EngineGrpcAdapter, session_id: str):
    session = adapter.open_stream(
        SessionStartRequest(
            session_id=session_id, config=SynthesisConfig(task_type="custom_voice")
        )
    )
    session.end()
    for _ in session.iter_messages():
        pass


def test_channel_is_reused_across_sessions(fake_server):
    adapter = EngineGrpcAdapter(fake_server, timeout=5.0, metadata=None, headers=None)

    _run_session(adapter, "s1")
    channel_after_first = adapter._grpc_channel
    assert channel_after_first is not None

    _run_session(adapter, "s2")
    channel_after_second = adapter._grpc_channel

    assert channel_after_second is channel_after_first, (
        "adapter opened a new channel for the second session instead of reusing it"
    )


def test_close_releases_the_channel(fake_server):
    adapter = EngineGrpcAdapter(fake_server, timeout=5.0, metadata=None, headers=None)
    _run_session(adapter, "s1")
    assert adapter._grpc_channel is not None

    adapter.close()
    assert adapter._grpc_channel is None
