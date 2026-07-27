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


class _AuthCapturingTTSService(tts_pb2_grpc.TTSServiceServicer):
    def __init__(self):
        self.calls: list[tuple[str, dict[str, str]]] = []

    def _capture(self, name, context):
        self.calls.append(
            (
                name,
                {item.key: item.value for item in context.invocation_metadata()},
            )
        )

    def GetCapabilities(self, request, context):
        self._capture("capabilities", context)
        return tts_pb2.GetCapabilitiesResponse()

    def SynthesizeOnce(self, request, context):
        self._capture("oneshot", context)
        yield tts_pb2.SynthesizeResponse(
            event=tts_pb2.StreamEvent(type="done", session_id=request.session_id)
        )

    def SynthesizeStream(self, request_iterator, context):
        self._capture("stream", context)
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


@pytest.fixture
def auth_server():
    service = _AuthCapturingTTSService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tts_pb2_grpc.add_TTSServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"127.0.0.1:{port}", service
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


def test_all_engine_grpc_rpcs_forward_lowercase_metadata(auth_server):
    endpoint, service = auth_server
    adapter = EngineGrpcAdapter(
        endpoint,
        timeout=5.0,
        metadata=(("X-Meta", "value"),),
        headers={"Authorization": "Bearer secret"},
    )
    config = SynthesisConfig(task_type="custom_voice")
    start = SessionStartRequest(
        session_id="s-auth",
        config=config,
        output_policy=config.output_policy,
        timing=config.timing_context,
    )

    adapter.get_capabilities()
    adapter.synthesize_bytes("hello", request=start)
    session = adapter.open_stream(start)
    session.end()
    list(session.iter_messages())

    assert [name for name, _metadata in service.calls] == [
        "capabilities",
        "oneshot",
        "stream",
    ]
    for _name, metadata in service.calls:
        assert metadata["authorization"] == "Bearer secret"
        assert metadata["x-meta"] == "value"


class _NoTerminalTTSService(tts_pb2_grpc.TTSServiceServicer):
    """Ends the response stream WITHOUT a done/error event (e.g. a redeploy
    tearing the server down mid-session)."""

    def SynthesizeStream(self, request_iterator, context):
        for request in request_iterator:
            which = request.WhichOneof("request")
            if which in ("end", "done", "cancel"):
                return  # stream ends cleanly, no terminal event


@pytest.fixture
def no_terminal_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tts_pb2_grpc.add_TTSServiceServicer_to_server(_NoTerminalTTSService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.stop(None)


def test_stream_end_without_terminal_does_not_hang(no_terminal_server):
    """Regression: a clean stream end without done/error used to leave the
    message queue without a sentinel — iter_messages() blocked forever and
    permanently pinned the consumer thread."""
    import threading

    from qwen3tts_protocol import StreamEvent

    adapter = EngineGrpcAdapter(
        no_terminal_server, timeout=5.0, metadata=None, headers=None
    )
    session = adapter.open_stream(
        SessionStartRequest(
            session_id="s-no-terminal",
            config=SynthesisConfig(task_type="custom_voice"),
        )
    )
    session.end()

    messages: list = []

    def consume():
        for msg in session.iter_messages():
            messages.append(msg)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(timeout=5.0)
    assert not consumer.is_alive(), "iter_messages() hung after clean stream end"
    assert messages, "expected a synthesized terminal event"
    last = messages[-1]
    # The terminal may be the synthesized "without terminal event" error (clean
    # stream end) or a transport-level RpcError (depending on how the server
    # tears the stream down) — either way iteration MUST end with an error.
    assert isinstance(last, StreamEvent) and last.type == "error"
