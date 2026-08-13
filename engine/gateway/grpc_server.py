"""gRPC streaming gateway for the TTS engine.

Handles bidirectional streaming: clients send text chunks, while the gateway
normalizes that transport protocol into the engine's internal token contract.

Protocol:
  Client sends:  StartRequest → TextChunk* → EndRequest
  Server sends:  AudioChunk* → StatusUpdate(done)

Uses grpcio.aio for async compatibility with the engine's asyncio event loop.

The protobuf ``session_id`` is a client correlation ID.  The gateway always
uses a separate server-generated ID for engine state and routing.

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
from typing import TYPE_CHECKING

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
    StreamingOutputProcessor,
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
)
from ..frontend.hold_window import PrefixGateGuardBypass
from . import tts_pb2, tts_pb2_grpc
from .session_identity import GatewaySessionIdentity

if TYPE_CHECKING:
    from ..server import TTSEngine
    from ..interface.vad import TTSVADProcessor


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
        identity: GatewaySessionIdentity,
        *,
        start_request: SessionStartRequest,
        audio_queue: asyncio.Queue,
    ) -> None:
        client_session_id = identity.client_session_id
        internal_session_id = identity.internal_session_id
        start_request.session_id = client_session_id
        config = start_request.config
        identity.bind_engine_config(config)

        # Create server timing accumulator for cross-thread observability
        timing_acc = ServerTimingAccumulator()
        timing_acc.request_received_epoch_ms = int(round(time.time() * 1000.0))
        timing_acc.session_created_epoch_ms = timing_acc.request_received_epoch_ms
        timing_acc.vad_policy = config.output_policy.vad.strategy or "disabled"
        timing_acc.text_input_mode = config.input_mode.value

        LifecycleLogger.emit(
            session_id=internal_session_id,
            phase="request.accepted",
            request_id=config.timing.request_id or None,
            turn_id=config.timing.turn_id or None,
            transport="grpc",
            client_session_id=client_session_id,
            client_request_ts_ms=config.timing.client_request_ts_ms or None,
        )

        # Store accumulator reference in timing extra for engine thread access
        config.timing.extra["_server_timing_accumulator"] = timing_acc

        # Create per-session VAD processor from config
        vad_config = _build_vad_config(config)
        vad_processor = create_vad_processor(vad_config, sample_rate=ENGINE_SAMPLE_RATE)
        output_processor = StreamingOutputProcessor(
            start_request,
            vad_processor=vad_processor,
            native_sample_rate=ENGINE_SAMPLE_RATE,
            timing_accumulator=timing_acc,
        )
        pipeline = output_processor.pipeline

        def _drain_vad_transitions():
            # L2 vad_transition: emit per begin/end transition with session
            # context (answers "为什么裁了这段"). No-op unless recording enabled.
            for tr in vad_processor.drain_transitions():
                LifecycleLogger.emit(
                    session_id=internal_session_id,
                    phase="vad_transition",
                    request_id=config.timing.request_id or None,
                    client_session_id=client_session_id,
                    session_level=config.observability_level,
                    min_level=obs.ObsLevel.DEBUG,
                    **tr,
                )

        first_effective_logged = False
        vad_enabled = vad_processor.config.enabled
        prefix_gate_guard_bypass = (
            PrefixGateGuardBypass(vad_processor.discard_pending)
            if vad_enabled
            else None
        )
        if prefix_gate_guard_bypass is not None:
            # Internal shared state: the frontend bypasses playhead pacing only
            # while this gateway-owned VAD is still consuming the prefix.
            config.timing.extra["_prefix_gate_guard_bypass"] = prefix_gate_guard_bypass

        def snapshot_prefix_bypass() -> tuple[int, float]:
            if prefix_gate_guard_bypass is None:
                return 0, 0.0
            bypass_chunks = prefix_gate_guard_bypass.bypassed_chunks
            bypass_audio_ms = (
                prefix_gate_guard_bypass.bypassed_audio_bytes
                / (ENGINE_SAMPLE_RATE * 4)
                * 1000.0
            )
            timing_acc.guarded_delivery_prefix_bypass_chunks = bypass_chunks
            timing_acc.guarded_delivery_prefix_bypass_audio_ms = bypass_audio_ms
            return bypass_chunks, bypass_audio_ms

        def log_first_effective_audio(audio_bytes: int) -> None:
            """Emit the first audible-output boundary with VAD context once."""
            nonlocal first_effective_logged
            if first_effective_logged:
                return
            first_effective_logged = True
            if prefix_gate_guard_bypass is not None:
                prefix_gate_guard_bypass.mark_first_effective(audio_bytes)
            bypass_chunks, bypass_audio_ms = snapshot_prefix_bypass()
            vad_metrics = vad_processor.metrics
            prefix_trimmed_ms = (
                vad_metrics.prefix_trimmed_samples / ENGINE_SAMPLE_RATE * 1000.0
            )
            gating_ms = None
            if (
                timing_acc.first_raw_audio_monotonic is not None
                and timing_acc.first_effective_audio_monotonic is not None
            ):
                gating_ms = (
                    timing_acc.first_effective_audio_monotonic
                    - timing_acc.first_raw_audio_monotonic
                ) * 1000.0
            LifecycleLogger.emit(
                session_id=internal_session_id,
                phase="output.audio.first_effective",
                request_id=config.timing.request_id or None,
                client_session_id=client_session_id,
                session_level=config.observability_level,
                vad_policy=vad_processor.config.mode.value,
                prefix_trim_applied=prefix_trimmed_ms > 0.0,
                prefix_trimmed_ms=round(prefix_trimmed_ms, 3),
                first_raw_to_first_effective_audio_ms=(
                    round(gating_ms, 3) if gating_ms is not None else None
                ),
                guarded_delivery_prefix_bypass_chunks=bypass_chunks,
                guarded_delivery_prefix_bypass_audio_ms=round(bypass_audio_ms, 3),
            )

        async def on_audio(sid, data):
            batch = output_processor.process(data)
            _drain_vad_transitions()
            if batch.audio is not None and batch.audio.pcm_bytes:
                log_first_effective_audio(len(batch.audio.pcm_bytes))
                await audio_queue.put(
                    (
                        "audio",
                        _make_audio_response(
                            batch.audio.pcm_bytes,
                            batch.audio.audio,
                            meta=batch.audio.meta,
                        ),
                    )
                )
            for event in (*batch.anchors, *batch.events):
                await audio_queue.put(
                    (
                        "event",
                        _make_event_response_from_contract(
                            build_forward_event(client_session_id, event, start_request)
                        ),
                    )
                )

        async def on_event(sid, event: dict):
            batch = output_processor.process_event(event)
            for item in (*batch.anchors, *batch.events):
                await audio_queue.put(
                    (
                        "event",
                        _make_event_response_from_contract(
                            build_forward_event(client_session_id, item, start_request)
                        ),
                    )
                )

        async def on_done(sid, metrics):
            for batch in output_processor.finish(
                emit_final=not bool(metrics.get("error") or metrics.get("cancelled"))
            ):
                if batch.audio is not None and batch.audio.pcm_bytes:
                    log_first_effective_audio(len(batch.audio.pcm_bytes))
                    await audio_queue.put(
                        (
                            "audio",
                            _make_audio_response(
                                batch.audio.pcm_bytes,
                                batch.audio.audio,
                                meta=batch.audio.meta,
                            ),
                        )
                    )
                for event in (*batch.anchors, *batch.events):
                    await audio_queue.put(
                        (
                            "event",
                            _make_event_response_from_contract(
                                build_forward_event(
                                    client_session_id, event, start_request
                                )
                            ),
                        )
                    )

            _drain_vad_transitions()
            # All-silence sessions never cross the first-effective boundary,
            # so snapshot the bypass counters unconditionally at completion.
            snapshot_prefix_bypass()
            # Inject VAD observability into metrics
            _inject_vad_metrics(vad_processor, pipeline, metrics)
            await audio_queue.put(
                (
                    "event",
                    _make_event_response_from_contract(
                        build_done_event(client_session_id, metrics, pipeline)
                    ),
                )
            )

        await self._engine.start_session(
            internal_session_id,
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
        await audio_queue.put(
            (
                "event",
                _make_event_response_from_contract(
                    build_start_event(client_session_id, start_request)
                ),
            )
        )
        logger.info(
            "gRPC session started: client=%s internal=%s",
            client_session_id,
            internal_session_id,
        )

    async def _drain_available_audio(self, audio_queue: asyncio.Queue):
        pending: tuple | None = None
        while pending is not None or not audio_queue.empty():
            if pending is not None:
                msg_type_q, payload = pending
                pending = None
            else:
                msg_type_q, payload = audio_queue.get_nowait()
            response = _queue_message_to_response(msg_type_q, payload)
            if response is not None and msg_type_q == "audio":
                response, pending = _coalesce_queued_audio(response, audio_queue)
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
        pending: tuple | None = None
        while True:
            if pending is not None:
                msg_type_q, payload = pending
                pending = None
            else:
                try:
                    msg_type_q, payload = await asyncio.wait_for(
                        audio_queue.get(),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("gRPC session %s: audio wait timeout", session_id)
                    break
            response = _queue_message_to_response(msg_type_q, payload)
            if response is not None and msg_type_q == "audio":
                response, pending = _coalesce_queued_audio(response, audio_queue)
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
        identity = GatewaySessionIdentity.create(request.session_id)
        client_session_id = identity.client_session_id
        internal_session_id = identity.internal_session_id
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=_GRPC_AUDIO_QUEUE_MAXSIZE)

        try:
            start_request = _start_request_from_oneshot_request(request)
            await self._create_session(
                identity,
                start_request=start_request,
                audio_queue=audio_queue,
            )
            await self._engine.push_text_input(internal_session_id, request.text)
            await self._engine.mark_input_complete(internal_session_id)
            async for response in self._drain_until_done(
                internal_session_id, audio_queue
            ):
                yield response
            return
        except asyncio.CancelledError:
            logger.info(
                "gRPC oneshot cancelled: client=%s internal=%s",
                client_session_id,
                internal_session_id,
            )
        except Exception as e:
            logger.error(
                "gRPC oneshot error: client=%s internal=%s: %s",
                client_session_id,
                internal_session_id,
                e,
            )
            yield _make_event_response(
                event_type="error", session_id=client_session_id, message=str(e)
            )
        finally:
            await self._engine.cancel(internal_session_id)

        yield _make_event_response(
            event_type="done", session_id=client_session_id, message="Stream ended"
        )

    async def SynthesizeStream(self, request_iterator, context):
        """Handle one bidirectional stream.

        Protocol semantics:
        - ``StartRequest`` declares task config, input mode, and audio format.
        - ``TextChunk`` carries transport text only; the frontend owns
          normalization/tokenization before the backend sees it.
        - ``EndRequest`` / legacy ``TextComplete`` signals no more transport input.
        """
        client_session_id: str | None = None
        internal_session_id: str | None = None
        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=_GRPC_AUDIO_QUEUE_MAXSIZE)
        request_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        got_done = False
        got_cancel = False
        input_eof = False
        first_text_received = False
        start_request: SessionStartRequest | None = None
        request_task: asyncio.Task | None = None
        audio_task: asyncio.Task | None = None
        pump_task = asyncio.create_task(
            self._pump_requests(request_iterator, request_queue)
        )

        try:
            while True:
                if request_task is None and not input_eof:
                    request_task = asyncio.create_task(request_queue.get())
                if audio_task is None and internal_session_id and not got_cancel:
                    audio_task = asyncio.create_task(audio_queue.get())

                wait_set = {t for t in (request_task, audio_task) if t is not None}
                if not wait_set:
                    break

                done, _ = await asyncio.wait(
                    wait_set, return_when=asyncio.FIRST_COMPLETED
                )

                # Flush any ready audio BEFORE handling a control frame, so a
                # parked chunk is not reordered behind chunks the request branch
                # drains via get_nowait (full-duplex: the client keeps sending
                # text while receiving audio).
                if audio_task in done:
                    msg_type_q, payload = audio_task.result()
                    audio_task = None
                    while True:
                        response = _queue_message_to_response(msg_type_q, payload)
                        leftover = None
                        if response is not None and msg_type_q == "audio":
                            response, leftover = _coalesce_queued_audio(
                                response, audio_queue
                            )
                        if response is not None:
                            yield response
                            if _is_done_response(response):
                                return
                        if leftover is None:
                            break
                        msg_type_q, payload = leftover

                if request_task in done:
                    kind, payload = request_task.result()
                    request_task = None

                    if kind == "error":
                        raise payload

                    if kind == "eof":
                        input_eof = True
                        if internal_session_id and not got_done and not got_cancel:
                            await self._engine.mark_input_complete(internal_session_id)
                            got_done = True
                    else:
                        request = payload
                        msg_type = request.WhichOneof("request")

                        if msg_type in {"start", "init"}:
                            if internal_session_id is not None:
                                raise ValueError(
                                    "gRPC stream session has already been started"
                                )
                            start_request = _start_request_from_stream_request(request)
                            start_req = (
                                request.start if msg_type == "start" else request.init
                            )
                            identity = GatewaySessionIdentity.create(
                                start_req.session_id
                            )
                            client_session_id = identity.client_session_id
                            internal_session_id = identity.internal_session_id
                            await self._create_session(
                                identity,
                                start_request=start_request,
                                audio_queue=audio_queue,
                            )

                        elif msg_type == "text":
                            if internal_session_id:
                                if (
                                    start_request is not None
                                    and request.text.client_timestamp_ms > 0
                                ):
                                    start_request.timing.client_text_ts_ms = int(
                                        request.text.client_timestamp_ms
                                    )
                                if not first_text_received:
                                    first_text_received = True
                                    LifecycleLogger.emit(
                                        session_id=internal_session_id,
                                        phase="text.first_received",
                                        transport="grpc",
                                        client_session_id=client_session_id,
                                        request_id=(
                                            start_request.timing.request_id
                                            if start_request
                                            else None
                                        )
                                        or None,
                                        client_text_ts_ms=request.text.client_timestamp_ms
                                        or None,
                                    )
                                await self._engine.push_text_input(
                                    internal_session_id, request.text.text
                                )

                        elif msg_type in {"end", "done"}:
                            if internal_session_id:
                                if (
                                    start_request is not None
                                    and msg_type == "end"
                                    and request.end.client_timestamp_ms > 0
                                ):
                                    start_request.timing.client_end_ts_ms = int(
                                        request.end.client_timestamp_ms
                                    )
                                await self._engine.mark_input_complete(
                                    internal_session_id
                                )
                            got_done = True

                        elif msg_type == "cancel":
                            if internal_session_id:
                                await self._engine.cancel(internal_session_id)
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
            logger.info(
                "gRPC stream cancelled: client=%s internal=%s",
                client_session_id,
                internal_session_id,
            )
        except Exception as e:
            logger.error(
                "gRPC stream error: client=%s internal=%s: %s",
                client_session_id,
                internal_session_id,
                e,
            )
            yield _make_event_response(
                event_type="error", session_id=client_session_id or "", message=str(e)
            )
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
            if internal_session_id:
                await self._engine.cancel(internal_session_id)

        yield _make_event_response(
            event_type="done",
            session_id=client_session_id or "",
            message="Stream ended",
        )


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
    return tts_pb2.SynthesizeResponse(event=tts_pb2.StreamEvent(**kwargs))


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
        declared_supported_task_types=list(
            cap.get("declared_supported_task_types", ()) or ()
        ),
        supported_input_modes=[
            _input_mode_to_proto(value)
            for value in cap.get("supported_input_modes", ()) or ()
        ],
        supported_group_policies=[
            _group_policy_to_proto(value)
            for value in cap.get("supported_group_policies", ()) or ()
        ],
        supported_audio_formats=[
            tts_pb2.AudioFormat(
                encoding=_audio_encoding_to_proto(
                    _audio_encoding_from_name(fmt.get("encoding", ""))
                ),
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
        ref_audio_max_duration_sec=float(
            cap.get("ref_audio_max_duration_sec", 0.0) or 0.0
        ),
        ref_c2w_warm_state_available=bool(
            cap.get("ref_c2w_warm_state_available", False)
        ),
        ref_codec_reason=str(cap.get("ref_codec_reason", "") or ""),
        protocol_version=str(cap.get("protocol_version", "") or ""),
        engine_version=str(cap.get("engine_version", "") or ""),
        supported_output_policy_features=[
            str(value)
            for value in cap.get("supported_output_policy_features", ()) or ()
        ],
        supported_vad_strategies=[
            str(value) for value in cap.get("supported_vad_strategies", ()) or ()
        ],
        supported_timing_fields=[
            str(value) for value in cap.get("supported_timing_fields", ()) or ()
        ],
    )


def _queue_message_to_response(
    msg_type_q: str,
    payload,
) -> tts_pb2.SynthesizeResponse | None:
    if msg_type_q == "audio":
        return payload
    if msg_type_q == "event":
        return payload
    return None


# Meta keys every chunk carries (OutputPipeline.convert_audio_chunk stamps
# chunk_index + timing_contract on all of them).  A chunk whose meta is a
# subset of these carries no per-chunk diagnostics worth preserving and may
# be absorbed into a merge; anything extra (first_audio_chunk, first-chunk
# timing fields) blocks absorption so diagnostics survive as message heads.
_MERGEABLE_META_KEYS = frozenset(
    {
        "chunk_index",
        "timing_contract",
        "output_sample_start",
        "output_sample_end",
        "output_sample_rate",
    }
)
# Stop merging before the message approaches gRPC's default 4 MiB client
# receive limit (a slow reader can backlog an entire session's audio).
_COALESCE_MAX_BYTES = 256 * 1024


def _coalesce_queued_audio(
    response: tts_pb2.SynthesizeResponse,
    audio_queue: asyncio.Queue,
) -> tuple[tts_pb2.SynthesizeResponse, tuple | None]:
    """Merge audio chunks already backlogged in the queue into ``response``.

    Under a wide burst the sender loop is the per-message bottleneck
    (~2.5k messages/s of protobuf build + stream write on the shared event
    loop).  Merging only what is ALREADY queued adds zero latency — the
    backlog exists precisely when the loop is overloaded.  The head keeps
    its meta (e.g. first-chunk timing); absorbed chunks carry only the
    standard per-chunk keys.  PCM bytes are concatenated unchanged.

    Returns (possibly-merged response, leftover queue message or None).
    The leftover is the first non-mergeable message popped during merging
    and MUST be processed by the caller before waiting on the queue again.
    """
    if response.WhichOneof("response") != "audio":
        return response, None
    head = response.audio
    parts: list[bytes] | None = None
    leftover: tuple | None = None
    merged_sample_end = int(head.meta.get("output_sample_end", "0") or 0)
    total_bytes = len(head.pcm_data)
    while total_bytes < _COALESCE_MAX_BYTES and not audio_queue.empty():
        msg_type_q, payload = audio_queue.get_nowait()
        if (
            msg_type_q == "audio"
            and payload.WhichOneof("response") == "audio"
            and set(payload.audio.meta) <= _MERGEABLE_META_KEYS
            and payload.audio.sample_rate == head.sample_rate
            and payload.audio.encoding == head.encoding
            and payload.audio.channels == head.channels
        ):
            if parts is None:
                parts = [head.pcm_data]
            parts.append(payload.audio.pcm_data)
            total_bytes += len(payload.audio.pcm_data)
            merged_sample_end = max(
                merged_sample_end,
                int(payload.audio.meta.get("output_sample_end", "0") or 0),
            )
            continue
        leftover = (msg_type_q, payload)
        break
    if parts is None:
        return response, leftover
    merged = tts_pb2.SynthesizeResponse(
        audio=tts_pb2.AudioChunk(
            pcm_data=b"".join(parts),
            sample_rate=head.sample_rate,
            encoding=head.encoding,
            channels=head.channels,
            meta={**dict(head.meta), "output_sample_end": str(merged_sample_end)},
        )
    )
    return merged, leftover


def _is_done_response(response: tts_pb2.SynthesizeResponse) -> bool:
    if response is None:
        return False
    which = response.WhichOneof("response")
    if which == "event":
        return response.event.type in {"done", "error"}
    return which == "status" and response.status.event in {"done", "error"}


async def serve(
    engine: TTSEngine,
    port: int = 50051,
    *,
    stop_event: asyncio.Event,
    started: asyncio.Event | None = None,
) -> None:
    """Start gRPC aio server. Call from within an asyncio event loop.

    Waits on ``stop_event`` then calls ``server.stop()`` so SIGINT/SIGTERM can shut
    down cleanly. ``wait_for_termination()`` alone does not reliably react to
    asyncio task cancellation. ``started`` is set once the port is bound so
    readiness can cover "gateway actually listening".
    """
    try:
        from grpc import aio as grpc_aio
    except ImportError:
        logger.error("grpcio not installed. Run: pip install grpcio grpcio-tools")
        return

    server = grpc_aio.server()

    servicer = TTSServicer(engine)
    tts_pb2_grpc.add_TTSServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    if started is not None:
        started.set()
    logger.info("gRPC server listening on port %d", port)
    await stop_event.wait()
    await server.stop(5.0)


def _session_config_from_stream_request(request) -> SessionConfig:
    return _start_request_from_stream_request(request).config


def _start_request_from_stream_request(request) -> SessionStartRequest:
    if request.WhichOneof("request") == "start":
        cfg, output_policy, timing = _session_contract_from_proto(
            request.start.config,
            default_mode=InputMode.AUTO,
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
        input_mode=getattr(init, "input_mode", tts_pb2.INPUT_MODE_UNSPECIFIED),
        group_policy=getattr(init, "group_policy", tts_pb2.GROUP_POLICY_AUTO),
        audio=getattr(init, "audio", None),
        default_mode=InputMode.AUTO,
    )
    return SessionStartRequest(
        session_id=init.session_id,
        config=cfg,
        output_policy=output_policy,
        timing=timing,
    )


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


def _session_contract_from_proto(
    proto_cfg, *, default_mode: InputMode
) -> tuple[SessionConfig, object, object]:
    output_policy = _output_policy_from_proto(
        proto_cfg.output_policy if proto_cfg.HasField("output_policy") else None
    )
    timing = _timing_context_from_proto(
        proto_cfg.timing if proto_cfg.HasField("timing") else None
    )
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
        input_mode=_input_mode_from_proto(
            proto_cfg.input_mode, default_mode=default_mode
        ),
        group_policy=_group_policy_from_proto(proto_cfg.group_policy),
        audio=_audio_config_from_proto(
            proto_cfg.audio if proto_cfg.HasField("audio") else None
        ),
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
        raise ValueError(
            f"Unsupported sample_rate: {audio.sample_rate} (expected 16000 or 24000)"
        )
    if audio.encoding not in (AudioEncoding.PCM_F32, AudioEncoding.PCM_S16LE):
        raise ValueError(f"Unsupported audio encoding: {audio.encoding}")


def _convert_audio_chunk(pcm_bytes: bytes, audio_config: AudioConfig) -> bytes:
    start_request = SessionStartRequest(
        session_id="", config=SessionConfig(audio=audio_config)
    )
    return OutputPipeline(start_request).convert_audio_chunk(pcm_bytes).pcm_bytes


def _build_vad_config(session_config: SessionConfig) -> TTSVADConfig:
    """Build TTSVADConfig from SessionConfig's output_policy.vad."""
    vad = session_config.output_policy.vad
    if not vad.enabled or vad.strategy == "disabled":
        return TTSVADConfig(mode=VADMode.DISABLED)

    mode_str = str(vad.strategy or "disabled").strip().lower()
    if mode_str == "prefix_trim":
        # ``prefix_trim`` is a protocol-level legacy policy name rather than
        # a processor implementation. Honor its declared implementation when
        # possible and use the built-in energy gate as the compatible default.
        implementation = str(vad.implementation or "").strip().lower()
        mode_str = (
            implementation if implementation in {"energy", "tenvad"} else "energy"
        )
    try:
        mode = VADMode(mode_str)
    except ValueError:
        logger.warning("Unsupported VAD strategy '%s', disabling VAD", mode_str)
        return TTSVADConfig(mode=VADMode.DISABLED)

    extra_config = {}
    for key, cast in (
        ("preemphasis", float),
        ("tenvad_hop_size", int),
        ("tenvad_threshold", float),
    ):
        if key not in vad.config:
            continue
        try:
            # gRPC carries VADPolicy.config as a string map, while WebSocket
            # JSON commonly supplies native numbers. Normalize both here.
            extra_config[key] = cast(vad.config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid VAD config {key}={vad.config[key]!r}") from exc

    return TTSVADConfig(
        mode=mode,
        chunk_ms=vad.chunk_ms,
        begin_threshold=vad.begin_threshold,
        begin_count=vad.begin_count,
        end_threshold=vad.end_threshold,
        end_count=vad.end_count,
        start_margin_ms=vad.start_margin_ms,
        **extra_config,
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

    # Prefix removal is a result even when the entire stream is silent and no
    # effective chunk exists.  Do not hide an all-silence discard behind the
    # first-effective condition.
    if m.prefix_trimmed_samples > 0:
        pipeline.record_prefix_trim(m.prefix_trimmed_samples, sr)
