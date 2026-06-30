"""gRPC streaming gateway for the TTS engine.

Handles bidirectional streaming: clients send text chunks, while the gateway
normalizes that transport protocol into the engine's internal token contract.

Protocol:
  Client sends:  StartRequest → TextChunk* → EndRequest
  Server sends:  AudioChunk* → StatusUpdate(done)

Uses grpcio.aio for async compatibility with the engine's asyncio event loop.

To regenerate proto stubs:
    python -m grpc_tools.protoc -I engine/gateway \
        --python_out=engine/gateway \
        --grpc_python_out=engine/gateway \
        engine/gateway/tts.proto
Afterwards, replace ``import tts_pb2`` with ``from . import tts_pb2`` in ``tts_pb2_grpc.py``
(protoc emits a top-level import that breaks the ``engine.gateway`` package).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np

from ..core.types import (
    AudioConfig,
    AudioEncoding,
    GroupPolicy,
    InputMode,
    SessionConfig,
)
from ..core.timing import ServerTimingAccumulator
from ..core.lifecycle import LifecycleLogger
from ..core import observability as obs
from ..interface import (
    OutputPipeline,
    SessionStartRequest,
    build_done_event,
    build_forward_event,
    build_start_event,
    normalize_capabilities,
    parse_output_policy,
    parse_timing_context,
    serialize_stream_event,
    to_core_output_policy,
    to_core_timing_context,
)
from ..interface.vad import (
    TTSVADConfig,
    VADMode,
    create_vad_processor,
    vad_config_from_dict,
)
from . import tts_pb2, tts_pb2_grpc

if TYPE_CHECKING:
    from ..server import TTSEngine


_GRPC_AUDIO_QUEUE_MAXSIZE = int(
    os.environ.get("ENGINE_GRPC_AUDIO_QUEUE_MAXSIZE", "4096") or "4096"
)

logger = logging.getLogger(__name__)

ENGINE_SAMPLE_RATE = 24000


class TTSServicer(tts_pb2_grpc.TTSServiceServicer):
    """gRPC servicer that bridges streaming requests to TTSEngine."""

    def __init__(self, engine: TTSEngine):
        self._engine = engine

    async def GetCapabilities(self, request, context):
        return _make_capabilities_response(
            normalize_capabilities(self._engine.describe_capabilities())
        )

    async def _create_session(
        self,
        session_id: str | None,
        *,
        start_request: SessionStartRequest,
        audio_queue: asyncio.Queue,
    ) -> str:
        session_id = session_id or str(uuid.uuid4())
        config = start_request.config

        # Create server timing accumulator for cross-thread observability
        timing_acc = ServerTimingAccumulator()
        timing_acc.request_received_epoch_ms = int(round(time.time() * 1000.0))
        timing_acc.session_created_epoch_ms = timing_acc.request_received_epoch_ms
        timing_acc.vad_policy = config.output_policy.vad.strategy or "disabled"
        timing_acc.text_input_mode = config.input_mode.value

        LifecycleLogger.emit(
            session_id=session_id,
            phase="request.accepted",
            request_id=config.timing.request_id or None,
            turn_id=config.timing.turn_id or None,
            transport="grpc",
            client_request_ts_ms=config.timing.client_request_ts_ms or None,
        )

        # Store accumulator reference in timing extra for engine thread access
        config.timing.extra["_server_timing_accumulator"] = timing_acc

        pipeline = OutputPipeline(
            start_request,
            request_received_monotonic=time.monotonic(),
            timing_accumulator=timing_acc,
        )

        # Create per-session VAD processor from config
        vad_config = _build_vad_config(config)
        vad_processor = create_vad_processor(vad_config, sample_rate=ENGINE_SAMPLE_RATE)

        def _drain_vad_transitions():
            # L2 vad_transition: emit per begin/end transition with session
            # context (answers "为什么裁了这段"). No-op unless recording enabled.
            for tr in vad_processor.drain_transitions():
                LifecycleLogger.emit(
                    session_id=session_id,
                    phase="vad_transition",
                    request_id=config.timing.request_id or None,
                    session_level=config.observability_level,
                    min_level=obs.ObsLevel.DEBUG,
                    **tr,
                )

        first_effective_logged = False

        async def on_audio(sid, data):
            nonlocal first_effective_logged
            # Apply VAD filtering before output pipeline
            raw = np.frombuffer(data, dtype=np.float32)
            if raw.size == 0:
                return
            # Convert to int16 for VAD processing
            audio_int16 = np.clip(raw, -1.0, 1.0)
            audio_int16 = (audio_int16 * 32767.0).astype(np.int16)

            filtered_int16 = vad_processor.process_chunk(audio_int16)
            _drain_vad_transitions()
            if filtered_int16.size == 0:
                return

            if not first_effective_logged:
                first_effective_logged = True
                LifecycleLogger.emit(
                    session_id=session_id,
                    phase="output.audio.first_effective",
                    request_id=config.timing.request_id or None,
                    session_level=config.observability_level,
                )

            # Convert back to float32 bytes for OutputPipeline
            filtered_f32 = (filtered_int16.astype(np.float32) / 32767.0)
            filtered_bytes = filtered_f32.tobytes()

            frame = pipeline.convert_audio_chunk(filtered_bytes)
            await audio_queue.put(("audio", _make_audio_response(frame.pcm_bytes, frame.audio, meta=frame.meta)))

        async def on_event(sid, event: dict):
            await audio_queue.put((
                "event",
                _make_event_response_from_contract(
                    build_forward_event(sid, event, start_request)
                ),
            ))

        async def on_done(sid, metrics):
            # Flush any remaining audio from VAD
            final_int16 = vad_processor.flush()
            if final_int16.size > 0:
                final_f32 = (final_int16.astype(np.float32) / 32767.0)
                final_bytes = final_f32.tobytes()
                frame = pipeline.convert_audio_chunk(final_bytes)
                await audio_queue.put(("audio", _make_audio_response(frame.pcm_bytes, frame.audio, meta=frame.meta)))

            _drain_vad_transitions()
            # Inject VAD observability into metrics
            _inject_vad_metrics(vad_processor, pipeline, metrics)
            await audio_queue.put((
                "event",
                _make_event_response_from_contract(build_done_event(sid, metrics, pipeline)),
            ))

        await self._engine.start_session(
            session_id,
            config=config,
            on_audio=on_audio,
            on_done=on_done,
            on_event=on_event,
        )
        # The frontend resolved the per-session observability level during
        # start_session; enable VAD transition recording before audio flows.
        vad_processor.enable_transition_recording(
            obs.is_enabled(obs.ObsLevel.DEBUG, config.observability_level)
        )
        await audio_queue.put((
            "event",
            _make_event_response_from_contract(build_start_event(session_id, start_request)),
        ))
        logger.info("gRPC session started: %s", session_id)
        return session_id

    async def _drain_available_audio(self, audio_queue: asyncio.Queue):
        while not audio_queue.empty():
            msg_type_q, payload = audio_queue.get_nowait()
            response = _queue_message_to_response(msg_type_q, payload)
            if response is not None:
                yield response
            if msg_type_q == "event" and _is_done_response(response):
                return

    async def _drain_until_done(
        self,
        session_id: str,
        audio_queue: asyncio.Queue,
        *,
        timeout: float = 300.0,
    ):
        while True:
            try:
                msg_type_q, payload = await asyncio.wait_for(
                    audio_queue.get(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("gRPC session %s: audio wait timeout", session_id)
                break
            response = _queue_message_to_response(msg_type_q, payload)
            if response is not None:
                yield response
            if msg_type_q == "event" and _is_done_response(response):
                return

    async def _pump_requests(
        self,
        request_iterator,
        request_queue: asyncio.Queue,
    ) -> None:
        try:
            async for request in request_iterator:
                await request_queue.put(("request", request))
        except Exception as exc:
            await request_queue.put(("error", exc))
        finally:
            await request_queue.put(("eof", None))

    async def SynthesizeOnce(self, request, context):
        """Handle unary full-text requests with streamed audio output."""
        session_id = None
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=_GRPC_AUDIO_QUEUE_MAXSIZE)

        try:
            start_request = _start_request_from_oneshot_request(request)
            session_id = await self._create_session(
                request.session_id,
                start_request=start_request,
                audio_queue=audio_queue,
            )
            await self._engine.push_text_input(session_id, request.text)
            await self._engine.mark_input_complete(session_id)
            async for response in self._drain_until_done(session_id, audio_queue):
                yield response
            return
        except asyncio.CancelledError:
            logger.info("gRPC oneshot cancelled: %s", session_id)
        except Exception as e:
            logger.error("gRPC oneshot error: %s: %s", session_id, e)
            yield _make_event_response(event_type="error", session_id=session_id or "", message=str(e))
        finally:
            if session_id:
                await self._engine.cancel(session_id)

        yield _make_event_response(event_type="done", session_id=session_id or "", message="Stream ended")

    async def SynthesizeStream(self, request_iterator, context):
        """Handle one bidirectional stream.

        Protocol semantics:
        - ``StartRequest`` declares task config, input mode, and audio format.
        - ``TextChunk`` carries transport text only; the frontend owns
          normalization/tokenization before the backend sees it.
        - ``EndRequest`` / legacy ``TextComplete`` signals no more transport input.
        """
        session_id = None
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=_GRPC_AUDIO_QUEUE_MAXSIZE)
        request_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        got_done = False
        got_cancel = False
        input_eof = False
        first_text_received = False
        start_request: SessionStartRequest | None = None
        request_task: asyncio.Task | None = None
        audio_task: asyncio.Task | None = None
        pump_task = asyncio.create_task(self._pump_requests(request_iterator, request_queue))

        try:
            while True:
                if request_task is None and not input_eof:
                    request_task = asyncio.create_task(request_queue.get())
                if audio_task is None and session_id and not got_cancel:
                    audio_task = asyncio.create_task(audio_queue.get())

                wait_set = {t for t in (request_task, audio_task) if t is not None}
                if not wait_set:
                    break

                done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                # Flush any ready audio BEFORE handling a control frame, so a
                # parked chunk is not reordered behind chunks the request branch
                # drains via get_nowait (full-duplex: the client keeps sending
                # text while receiving audio).
                if audio_task in done:
                    msg_type_q, payload = audio_task.result()
                    audio_task = None
                    response = _queue_message_to_response(msg_type_q, payload)
                    if response is not None:
                        yield response
                        if _is_done_response(response):
                            return

                if request_task in done:
                    kind, payload = request_task.result()
                    request_task = None

                    if kind == "error":
                        raise payload

                    if kind == "eof":
                        input_eof = True
                        if session_id and not got_done and not got_cancel:
                            await self._engine.mark_input_complete(session_id)
                            got_done = True
                    else:
                        request = payload
                        msg_type = request.WhichOneof("request")

                        if msg_type in {"start", "init"}:
                            start_request = _start_request_from_stream_request(request)
                            start_req = request.start if msg_type == "start" else request.init
                            session_id = await self._create_session(
                                start_req.session_id,
                                start_request=start_request,
                                audio_queue=audio_queue,
                            )

                        elif msg_type == "text":
                            if session_id:
                                if start_request is not None and request.text.client_timestamp_ms > 0:
                                    start_request.timing.client_text_ts_ms = int(request.text.client_timestamp_ms)
                                if not first_text_received:
                                    first_text_received = True
                                    LifecycleLogger.emit(
                                        session_id=session_id,
                                        phase="text.first_received",
                                        transport="grpc",
                                        request_id=(start_request.timing.request_id
                                                    if start_request else None) or None,
                                        client_text_ts_ms=request.text.client_timestamp_ms or None,
                                    )
                                await self._engine.push_text_input(session_id, request.text.text)

                        elif msg_type in {"end", "done"}:
                            if session_id:
                                if start_request is not None and msg_type == "end" and request.end.client_timestamp_ms > 0:
                                    start_request.timing.client_end_ts_ms = int(request.end.client_timestamp_ms)
                                await self._engine.mark_input_complete(session_id)
                            got_done = True

                        elif msg_type == "cancel":
                            if session_id:
                                await self._engine.cancel(session_id)
                            got_cancel = True
                            # Stop reading further requests so the loop can end
                            # even if the client never half-closes the stream
                            # (otherwise the handler parks on request_queue.get()
                            # forever). Mirrors the WebSocket handler's behavior.
                            input_eof = True

                        async for response in self._drain_available_audio(audio_queue):
                            yield response
                            if _is_done_response(response):
                                return

                if input_eof and got_cancel:
                    break

        except asyncio.CancelledError:
            logger.info("gRPC stream cancelled: %s", session_id)
        except Exception as e:
            logger.error("gRPC stream error: %s: %s", session_id, e)
            yield _make_event_response(event_type="error", session_id=session_id or "", message=str(e))
        finally:
            for task in (request_task, audio_task, pump_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (request_task, audio_task, pump_task):
                if task is not None:
                    try:
                        await task
                    except (asyncio.CancelledError, StopAsyncIteration):
                        pass
            if session_id:
                await self._engine.cancel(session_id)

        yield _make_event_response(event_type="done", session_id=session_id or "", message="Stream ended")


def _make_audio_response(
    pcm_bytes: bytes,
    audio_config: AudioConfig,
    *,
    meta: dict[str, str] | None = None,
) -> tts_pb2.SynthesizeResponse:
    return tts_pb2.SynthesizeResponse(
        audio=tts_pb2.AudioChunk(
            pcm_data=pcm_bytes,
            sample_rate=audio_config.sample_rate,
            encoding=_audio_encoding_to_proto(audio_config.encoding),
            channels=audio_config.channels,
            meta=meta or {},
        )
    )


def _make_event_response(
    *,
    event_type: str,
    session_id: str = "",
    segment_id: int = -1,
    text: str = "",
    message: str = "",
    audio_format: AudioConfig | None = None,
    meta: dict[str, str] | None = None,
) -> tts_pb2.SynthesizeResponse:
    kwargs = {
        "type": event_type,
        "session_id": session_id,
        "segment_id": segment_id,
        "text": text,
        "message": message,
        "meta": meta or {},
    }
    if audio_format is not None:
        kwargs["audio"] = tts_pb2.AudioFormat(
            encoding=_audio_encoding_to_proto(audio_format.encoding),
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
        )
    return tts_pb2.SynthesizeResponse(
        event=tts_pb2.StreamEvent(**kwargs)
    )


def _make_event_response_from_contract(event) -> tts_pb2.SynthesizeResponse:
    payload = serialize_stream_event(event)
    audio = payload.get("audio")
    return _make_event_response(
        event_type=payload["type"],
        session_id=payload["session_id"],
        segment_id=int(payload["segment_id"]),
        text=payload["text"],
        message=payload["message"],
        audio_format=(
            AudioConfig(
                encoding=_audio_encoding_from_name(str(audio.get("encoding", ""))),
                sample_rate=int(audio.get("sample_rate", ENGINE_SAMPLE_RATE)),
                channels=int(audio.get("channels", 1)),
            )
            if isinstance(audio, dict)
            else None
        ),
        meta={str(k): str(v) for k, v in (payload.get("meta", {}) or {}).items()},
    )


def _make_status_response(event: str, message: str = "") -> tts_pb2.SynthesizeResponse:
    return tts_pb2.SynthesizeResponse(
        status=tts_pb2.StatusUpdate(
            event=event,
            message=message,
        )
    )


def _make_capabilities_response(cap: dict) -> tts_pb2.GetCapabilitiesResponse:
    cap = normalize_capabilities(cap)
    return tts_pb2.GetCapabilitiesResponse(
        variant=str(cap.get("variant", "") or ""),
        loaded_model_type=str(cap.get("loaded_model_type", "") or ""),
        declared_supported_task_types=list(cap.get("declared_supported_task_types", ()) or ()),
        supported_input_modes=[
            _input_mode_to_proto(value) for value in cap.get("supported_input_modes", ()) or ()
        ],
        supported_group_policies=[
            _group_policy_to_proto(value) for value in cap.get("supported_group_policies", ()) or ()
        ],
        supported_audio_formats=[
            tts_pb2.AudioFormat(
                encoding=_audio_encoding_to_proto(_audio_encoding_from_name(fmt.get("encoding", ""))),
                sample_rate=int(fmt.get("sample_rate", ENGINE_SAMPLE_RATE)),
                channels=int(fmt.get("channels", 1)),
            )
            for fmt in (cap.get("supported_audio_formats", ()) or ())
        ],
        ref_audio_available=bool(cap.get("ref_audio_available", False)),
        ref_audio_reason=str(cap.get("ref_audio_reason", "") or ""),
        speaker_encoder_available=bool(cap.get("speaker_encoder_available", False)),
        ref_codec_available=bool(cap.get("ref_codec_available", False)),
        icl_available=bool(cap.get("icl_available", False)),
        ref_audio_max_duration_sec=float(cap.get("ref_audio_max_duration_sec", 0.0) or 0.0),
        ref_c2w_warm_state_available=bool(cap.get("ref_c2w_warm_state_available", False)),
        ref_codec_reason=str(cap.get("ref_codec_reason", "") or ""),
        protocol_version=str(cap.get("protocol_version", "") or ""),
        supported_output_policy_features=[
            str(value) for value in cap.get("supported_output_policy_features", ()) or ()
        ],
        supported_vad_strategies=[
            str(value) for value in cap.get("supported_vad_strategies", ()) or ()
        ],
        supported_timing_fields=[
            str(value) for value in cap.get("supported_timing_fields", ()) or ()
        ],
    )


def _queue_message_to_response(
    msg_type_q: str, payload,
) -> tts_pb2.SynthesizeResponse | None:
    if msg_type_q == "audio":
        return payload
    if msg_type_q == "event":
        return payload
    return None


def _is_done_response(response: tts_pb2.SynthesizeResponse) -> bool:
    if response is None:
        return False
    which = response.WhichOneof("response")
    if which == "event":
        return response.event.type in {"done", "error"}
    return which == "status" and response.status.event in {"done", "error"}


async def serve(engine: TTSEngine, port: int = 50051, *, stop_event: asyncio.Event) -> None:
    """Start gRPC aio server. Call from within an asyncio event loop.

    Waits on ``stop_event`` then calls ``server.stop()`` so SIGINT/SIGTERM can shut
    down cleanly. ``wait_for_termination()`` alone does not reliably react to
    asyncio task cancellation.
    """
    try:
        import grpc
        from grpc import aio as grpc_aio
    except ImportError:
        logger.error("grpcio not installed. Run: pip install grpcio grpcio-tools")
        return

    server = grpc_aio.server()

    servicer = TTSServicer(engine)
    tts_pb2_grpc.add_TTSServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info("gRPC server listening on port %d", port)
    await stop_event.wait()
    await server.stop(5.0)


def _session_config_from_stream_request(request) -> SessionConfig:
    return _start_request_from_stream_request(request).config


def _start_request_from_stream_request(request) -> SessionStartRequest:
    if request.WhichOneof("request") == "start":
        cfg, output_policy, timing = _session_contract_from_proto(
            request.start.config,
            default_mode=InputMode.LONG_SEGMENT,
        )
        return SessionStartRequest(
            session_id=request.start.session_id,
            config=cfg,
            output_policy=output_policy,
            timing=timing,
        )
    init = request.init
    cfg, output_policy, timing = _session_contract_from_legacy_fields(
        task_type=init.task_type,
        language=init.language,
        speaker=init.speaker,
        instruct=getattr(init, "instruct", ""),
        ref_audio=init.ref_audio,
        ref_text=init.ref_text,
        x_vector_only=getattr(init, "x_vector_only", False),
        input_mode=getattr(init, "input_mode", tts_pb2.INPUT_MODE_LONG_SEGMENT),
        group_policy=getattr(init, "group_policy", tts_pb2.GROUP_POLICY_AUTO),
        audio=getattr(init, "audio", None),
        default_mode=InputMode.LONG_SEGMENT,
    )
    return SessionStartRequest(session_id=init.session_id, config=cfg, output_policy=output_policy, timing=timing)


def _session_config_from_oneshot_request(request) -> SessionConfig:
    return _start_request_from_oneshot_request(request).config


def _start_request_from_oneshot_request(request) -> SessionStartRequest:
    if request.HasField("config"):
        cfg, output_policy, timing = _session_contract_from_proto(
            request.config,
            default_mode=InputMode.FULL_TEXT,
        )
        cfg.input_mode = InputMode.FULL_TEXT
        if cfg.group_policy == GroupPolicy.NONE:
            cfg.group_policy = GroupPolicy.AUTO
        return SessionStartRequest(
            session_id=request.session_id,
            config=cfg,
            output_policy=output_policy,
            timing=timing,
            initial_text=request.text,
        )
    cfg, output_policy, timing = _session_contract_from_legacy_fields(
        task_type=request.task_type,
        language=request.language,
        speaker=request.speaker,
        instruct="",
        ref_audio=request.ref_audio,
        ref_text=request.ref_text,
        x_vector_only=False,
        input_mode=tts_pb2.INPUT_MODE_FULL_TEXT,
        group_policy=tts_pb2.GROUP_POLICY_AUTO,
        audio=None,
        default_mode=InputMode.FULL_TEXT,
    )
    return SessionStartRequest(
        session_id=request.session_id,
        config=cfg,
        output_policy=output_policy,
        timing=timing,
        initial_text=request.text,
    )


def _session_contract_from_legacy_fields(
    *,
    task_type: str,
    language: str,
    speaker: str,
    instruct: str,
    ref_audio: bytes,
    ref_text: str,
    x_vector_only: bool,
    input_mode,
    group_policy,
    audio,
    default_mode: InputMode,
) -> tuple[SessionConfig, object, object]:
    output_policy = parse_output_policy({})
    timing = parse_timing_context({})
    cfg = SessionConfig(
        task_type=task_type or "",
        language=language or "auto",
        speaker=speaker or None,
        instruct=instruct or None,
        ref_audio=ref_audio or None,
        ref_text=ref_text or None,
        x_vector_only=bool(x_vector_only),
        input_mode=_input_mode_from_proto(input_mode, default_mode=default_mode),
        group_policy=_group_policy_from_proto(group_policy),
        audio=_audio_config_from_proto(audio),
        output_policy=to_core_output_policy(output_policy),
        timing=to_core_timing_context(timing),
    )
    _validate_audio_config(cfg.audio)
    return cfg, output_policy, timing


def _session_config_from_legacy_fields(**kwargs) -> SessionConfig:
    return _session_contract_from_legacy_fields(**kwargs)[0]


def _session_contract_from_proto(proto_cfg, *, default_mode: InputMode) -> tuple[SessionConfig, object, object]:
    output_policy = _output_policy_from_proto(proto_cfg.output_policy if proto_cfg.HasField("output_policy") else None)
    timing = _timing_context_from_proto(proto_cfg.timing if proto_cfg.HasField("timing") else None)
    protocol_version = str(getattr(proto_cfg, "protocol_version", "") or "").strip()
    if protocol_version and "client_protocol_version" not in timing.extra:
        timing.extra["client_protocol_version"] = protocol_version
    cfg = SessionConfig(
        task_type=proto_cfg.task_type or "",
        language=proto_cfg.language or "auto",
        speaker=proto_cfg.speaker or None,
        instruct=proto_cfg.instruct or None,
        ref_audio=proto_cfg.ref_audio or None,
        ref_text=proto_cfg.ref_text or None,
        x_vector_only=bool(proto_cfg.x_vector_only),
        input_mode=_input_mode_from_proto(proto_cfg.input_mode, default_mode=default_mode),
        group_policy=_group_policy_from_proto(proto_cfg.group_policy),
        audio=_audio_config_from_proto(proto_cfg.audio if proto_cfg.HasField("audio") else None),
        output_policy=to_core_output_policy(output_policy),
        timing=to_core_timing_context(timing),
    )
    _validate_audio_config(cfg.audio)
    return cfg, output_policy, timing


def _session_config_from_proto(proto_cfg, *, default_mode: InputMode) -> SessionConfig:
    return _session_contract_from_proto(proto_cfg, default_mode=default_mode)[0]


def _input_mode_from_proto(value, *, default_mode: InputMode) -> InputMode:
    mapping = {
        tts_pb2.INPUT_MODE_TOKEN: InputMode.TOKEN,
        tts_pb2.INPUT_MODE_CLAUSE: InputMode.CLAUSE,
        tts_pb2.INPUT_MODE_LONG_SEGMENT: InputMode.LONG_SEGMENT,
        tts_pb2.INPUT_MODE_FULL_TEXT: InputMode.FULL_TEXT,
    }
    return mapping.get(value, default_mode)


def _input_mode_to_proto(value) -> int:
    if isinstance(value, InputMode):
        value = value.value
    mapping = {
        "token": tts_pb2.INPUT_MODE_TOKEN,
        "clause": tts_pb2.INPUT_MODE_CLAUSE,
        "long_segment": tts_pb2.INPUT_MODE_LONG_SEGMENT,
        "full_text": tts_pb2.INPUT_MODE_FULL_TEXT,
    }
    return mapping.get(value, tts_pb2.INPUT_MODE_UNSPECIFIED)


def _group_policy_from_proto(value) -> GroupPolicy:
    mapping = {
        tts_pb2.GROUP_POLICY_NONE: GroupPolicy.NONE,
        tts_pb2.GROUP_POLICY_AUTO: GroupPolicy.AUTO,
    }
    return mapping.get(value, GroupPolicy.AUTO)


def _group_policy_to_proto(value) -> int:
    if isinstance(value, GroupPolicy):
        value = value.value
    mapping = {
        "none": tts_pb2.GROUP_POLICY_NONE,
        "auto": tts_pb2.GROUP_POLICY_AUTO,
    }
    return mapping.get(value, tts_pb2.GROUP_POLICY_UNSPECIFIED)


def _audio_config_from_proto(audio_msg) -> AudioConfig:
    if audio_msg is None:
        return AudioConfig()
    sample_rate = audio_msg.sample_rate or ENGINE_SAMPLE_RATE
    channels = audio_msg.channels or 1
    encoding = {
        tts_pb2.AUDIO_ENCODING_PCM_S16LE: AudioEncoding.PCM_S16LE,
        tts_pb2.AUDIO_ENCODING_PCM_F32: AudioEncoding.PCM_F32,
    }.get(audio_msg.encoding, AudioEncoding.PCM_F32)
    return AudioConfig(sample_rate=sample_rate, encoding=encoding, channels=channels)


def _proto_map_to_dict(raw_map) -> dict[str, str]:
    if raw_map is None:
        return {}
    return {str(key): str(value) for key, value in dict(raw_map).items()}


# VAD tuning params have no dedicated proto fields; the client carries them
# through the proto VADPolicy.config string-map. Lift them back to top-level so
# parse_output_policy reads the real values instead of falling back to defaults.
_VAD_TUNING_FIELDS = (
    "chunk_ms",
    "begin_threshold",
    "begin_count",
    "end_threshold",
    "end_count",
    "start_margin_ms",
)


def _output_policy_from_proto(proto_policy) -> object:
    if proto_policy is None:
        return parse_output_policy({})
    emit_text_events = True
    if proto_policy.HasField("emit_text_events"):
        emit_text_events = bool(proto_policy.emit_text_events)
    vad_config = _proto_map_to_dict(proto_policy.vad_policy.config)
    tuning = {f: vad_config.pop(f) for f in _VAD_TUNING_FIELDS if f in vad_config}
    return parse_output_policy(
        {
            "vad_policy": {
                "enabled": bool(proto_policy.vad_policy.enabled),
                "strategy": str(proto_policy.vad_policy.strategy or "disabled"),
                "implementation": str(proto_policy.vad_policy.implementation or ""),
                "config": vad_config,
                **tuning,
            },
            "chunk_ms": int(proto_policy.chunk_ms or 0),
            "packet_format": str(proto_policy.packet_format or "raw_pcm"),
            "emit_text_events": emit_text_events,
            "config": _proto_map_to_dict(proto_policy.config),
        }
    )


def _timing_context_from_proto(proto_timing) -> object:
    if proto_timing is None:
        return parse_timing_context({})
    return parse_timing_context(
        {
            "request_id": str(proto_timing.request_id or ""),
            "turn_id": str(proto_timing.turn_id or ""),
            "client_request_ts_ms": int(proto_timing.client_request_ts_ms or 0),
            "client_text_ts_ms": int(proto_timing.client_text_ts_ms or 0),
            "client_end_ts_ms": int(proto_timing.client_end_ts_ms or 0),
            "extra": _proto_map_to_dict(proto_timing.extra),
        }
    )


def _audio_encoding_to_proto(value: AudioEncoding) -> int:
    if value == AudioEncoding.PCM_S16LE:
        return tts_pb2.AUDIO_ENCODING_PCM_S16LE
    return tts_pb2.AUDIO_ENCODING_PCM_F32


def _audio_encoding_from_name(value: str) -> AudioEncoding:
    if value == "pcm_s16le":
        return AudioEncoding.PCM_S16LE
    return AudioEncoding.PCM_F32


def _validate_audio_config(audio: AudioConfig) -> None:
    if audio.channels != 1:
        raise ValueError(f"Unsupported channel count: {audio.channels} (mono only)")
    if audio.sample_rate not in (16000, 24000):
        raise ValueError(f"Unsupported sample_rate: {audio.sample_rate} (expected 16000 or 24000)")
    if audio.encoding not in (AudioEncoding.PCM_F32, AudioEncoding.PCM_S16LE):
        raise ValueError(f"Unsupported audio encoding: {audio.encoding}")


def _convert_audio_chunk(pcm_bytes: bytes, audio_config: AudioConfig) -> bytes:
    start_request = SessionStartRequest(session_id="", config=SessionConfig(audio=audio_config))
    return OutputPipeline(start_request).convert_audio_chunk(pcm_bytes).pcm_bytes


def _build_vad_config(session_config: SessionConfig) -> TTSVADConfig:
    """Build TTSVADConfig from SessionConfig's output_policy.vad."""
    vad = session_config.output_policy.vad
    if not vad.enabled or vad.strategy == "disabled":
        return TTSVADConfig(mode=VADMode.DISABLED)

    mode_str = vad.strategy.strip().lower()
    try:
        mode = VADMode(mode_str)
    except ValueError:
        logger.warning("Unsupported VAD strategy '%s', disabling VAD", mode_str)
        return TTSVADConfig(mode=VADMode.DISABLED)

    return TTSVADConfig(
        mode=mode,
        chunk_ms=vad.chunk_ms,
        begin_threshold=vad.begin_threshold,
        begin_count=vad.begin_count,
        end_threshold=vad.end_threshold,
        end_count=vad.end_count,
        start_margin_ms=vad.start_margin_ms,
        # Pass through any extra config from the config dict
        **{
            k: v for k, v in vad.config.items()
            if k in ("preemphasis", "tenvad_hop_size", "tenvad_threshold")
        },
    )


def _inject_vad_metrics(
    vad_processor: "TTSVADProcessor",
    pipeline: OutputPipeline,
    metrics: dict,
) -> None:
    """Inject VAD observability into done_meta metrics and OutputPipeline."""
    m = vad_processor.metrics
    sr = ENGINE_SAMPLE_RATE

    if not vad_processor.config.enabled:
        return

    prefix_trimmed_ms = m.prefix_trimmed_samples / sr * 1000.0
    tail_trimmed_ms = m.tail_trimmed_samples / sr * 1000.0
    original_audio_ms = m.original_audio_samples / sr * 1000.0
    effective_audio_ms = m.effective_audio_samples / sr * 1000.0

    metrics["vad_mode"] = vad_processor.config.mode.value
    metrics["vad_prefix_trimmed_ms"] = f"{prefix_trimmed_ms:.3f}"
    metrics["vad_tail_trimmed_ms"] = f"{tail_trimmed_ms:.3f}"
    metrics["vad_original_audio_ms"] = f"{original_audio_ms:.3f}"
    metrics["vad_effective_audio_ms"] = f"{effective_audio_ms:.3f}"
    metrics["vad_begin_count"] = str(m.begin_trigger_count)
    metrics["vad_end_count"] = str(m.end_trigger_count)

    # Record prefix trim in OutputPipeline for first-effective-audio tracking
    if m.first_effective_audio_found and m.prefix_trimmed_samples > 0:
        pipeline.record_prefix_trim(m.prefix_trimmed_samples, sr)
