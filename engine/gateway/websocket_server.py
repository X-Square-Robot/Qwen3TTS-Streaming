"""WebSocket streaming gateway for the standalone TTS engine.

Protocol:
  Client text frames:
    {"type":"start","session_id":"...","config":{...}}
    {"type":"text","text":"...","seq_no":1}
    {"type":"end"}
    {"type":"cancel"}
    {"type":"oneshot","session_id":"...","text":"...","config":{...}}
    {"type":"get_capabilities"}

  Server text frames:
    {"type":"event","event":{...}}
    {"type":"capabilities","capabilities":{...}}

  Server binary frames:
    raw PCM audio bytes matching the audio format declared by the ``start`` event.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.types import (
    AudioConfig,
    AudioEncoding,
    GroupPolicy,
    InputMode,
    SessionConfig,
)
from ..core.timing import ServerTimingAccumulator
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
)
from .grpc_server import _build_vad_config, _inject_vad_metrics

if TYPE_CHECKING:
    from ..server import TTSEngine

try:
    from aiohttp import WSMsgType, web
except ImportError:  # pragma: no cover - exercised in environments without aiohttp
    WSMsgType = None
    web = None


_WEBSOCKET_AUDIO_QUEUE_MAXSIZE = int(
    os.environ.get("ENGINE_WEBSOCKET_AUDIO_QUEUE_MAXSIZE", "4096") or "4096"
)
_WEBSOCKET_REQUEST_QUEUE_MAXSIZE = int(
    os.environ.get("ENGINE_WEBSOCKET_REQUEST_QUEUE_MAXSIZE", "64") or "64"
)
_CAPABILITIES_PATH = "/v1/capabilities"
_WEBSOCKET_HEARTBEAT_SEC = float(
    os.environ.get("ENGINE_WEBSOCKET_HEARTBEAT_SEC", "30") or "30"
)

logger = logging.getLogger(__name__)


class WebSocketGateway:
    """Bridge a single websocket connection to one TTS engine session."""

    def __init__(self, engine: TTSEngine):
        self._engine = engine

    async def handle_capabilities(self, request):
        return web.json_response(normalize_capabilities(self._engine.describe_capabilities()))

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=_WEBSOCKET_HEARTBEAT_SEC)
        await ws.prepare(request)

        session_id = None
        outbound_queue: asyncio.Queue = asyncio.Queue(maxsize=_WEBSOCKET_AUDIO_QUEUE_MAXSIZE)
        request_queue: asyncio.Queue = asyncio.Queue(maxsize=_WEBSOCKET_REQUEST_QUEUE_MAXSIZE)
        got_cancel = False
        connection_closed = False
        start_request: SessionStartRequest | None = None
        request_task: asyncio.Task | None = None
        outbound_task: asyncio.Task | None = None
        pump_task = asyncio.create_task(self._pump_messages(ws, request_queue))

        try:
            while True:
                if request_task is None and not connection_closed:
                    request_task = asyncio.create_task(request_queue.get())
                if outbound_task is None and session_id and not got_cancel:
                    outbound_task = asyncio.create_task(outbound_queue.get())

                wait_set = {task for task in (request_task, outbound_task) if task is not None}
                if not wait_set:
                    break

                done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                # Flush any ready outbound frame BEFORE handling a control frame,
                # so a parked chunk is not reordered behind frames the request
                # branch drains via get_nowait (full-duplex: the client keeps
                # sending text while receiving audio).
                if outbound_task in done:
                    frame = outbound_task.result()
                    outbound_task = None
                    await _send_frame(ws, frame)
                    if _is_terminal_frame(frame):
                        return ws

                if request_task in done:
                    kind, payload = request_task.result()
                    request_task = None

                    if kind == "error":
                        raise payload

                    if kind == "closed":
                        connection_closed = True
                        got_cancel = True
                    else:
                        message = payload
                        msg_type = str(message.get("type", "") or "").strip().lower()

                        if msg_type == "get_capabilities":
                            await ws.send_json(
                                {
                                    "type": "capabilities",
                                    "capabilities": normalize_capabilities(self._engine.describe_capabilities()),
                                }
                            )
                            continue

                        if msg_type == "start":
                            if session_id is not None:
                                raise ValueError("websocket session has already been started")
                            start_request = _start_request_from_ws_message(
                                message,
                                default_mode=InputMode.AUTO,
                            )
                            session_id = await self._create_session(
                                message.get("session_id"),
                                start_request=start_request,
                                outbound_queue=outbound_queue,
                            )

                        elif msg_type == "oneshot":
                            if session_id is not None:
                                raise ValueError("websocket session has already been started")
                            start_request = _start_request_from_ws_message(
                                message,
                                default_mode=InputMode.FULL_TEXT,
                            )
                            start_request.config.input_mode = InputMode.FULL_TEXT
                            if start_request.config.group_policy == GroupPolicy.NONE:
                                start_request.config.group_policy = GroupPolicy.AUTO
                            session_id = await self._create_session(
                                message.get("session_id"),
                                start_request=start_request,
                                outbound_queue=outbound_queue,
                            )
                            text = str(message.get("text", "") or "")
                            if not text:
                                raise ValueError("oneshot request requires non-empty 'text'")
                            start_request.initial_text = text
                            await self._engine.push_text_input(session_id, text)
                            await self._engine.mark_input_complete(session_id)

                        elif msg_type == "text":
                            if not session_id:
                                raise ValueError("received 'text' before 'start'")
                            if start_request is not None:
                                client_ts_ms = _coerce_ws_int(message.get("client_timestamp_ms"), 0)
                                if client_ts_ms > 0:
                                    start_request.timing.client_text_ts_ms = client_ts_ms
                            await self._engine.push_text_input(
                                session_id,
                                str(message.get("text", "") or ""),
                            )

                        elif msg_type == "end":
                            if not session_id:
                                raise ValueError("received 'end' before 'start'")
                            if start_request is not None:
                                client_ts_ms = _coerce_ws_int(message.get("client_timestamp_ms"), 0)
                                if client_ts_ms > 0:
                                    start_request.timing.client_end_ts_ms = client_ts_ms
                            await self._engine.mark_input_complete(session_id)

                        elif msg_type == "cancel":
                            if session_id:
                                await self._engine.cancel(session_id)
                            got_cancel = True
                            connection_closed = True

                        else:
                            raise ValueError(f"unsupported websocket message type: '{msg_type or '<empty>'}'")

                        async for frame in self._drain_available_messages(outbound_queue):
                            await _send_frame(ws, frame)
                            if _is_terminal_frame(frame):
                                return ws

                if connection_closed and got_cancel:
                    break

        except asyncio.CancelledError:
            logger.info("WebSocket stream cancelled: %s", session_id)
        except Exception as exc:
            logger.error("WebSocket stream error: %s: %s", session_id, exc)
            if not ws.closed:
                await ws.send_json(
                    _make_event_frame(
                        event_type="error",
                        session_id=session_id or "",
                        message=str(exc),
                    )
                )
        finally:
            for task in (request_task, outbound_task, pump_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (request_task, outbound_task, pump_task):
                if task is not None:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            if session_id:
                await self._engine.cancel(session_id)
            if not ws.closed:
                await ws.close()

        return ws

    async def _create_session(
        self,
        session_id: str | None,
        *,
        start_request: SessionStartRequest,
        outbound_queue: asyncio.Queue,
    ) -> str:
        session_id = str(session_id or uuid.uuid4())
        config = start_request.config

        # Create server timing accumulator for cross-thread observability
        import time as _time
        timing_acc = ServerTimingAccumulator()
        timing_acc.request_received_epoch_ms = int(round(_time.time() * 1000.0))
        timing_acc.session_created_epoch_ms = timing_acc.request_received_epoch_ms
        timing_acc.vad_policy = config.output_policy.vad.strategy or "disabled"
        timing_acc.text_input_mode = config.input_mode.value

        # Store accumulator reference in timing extra for engine thread access
        config.timing.extra["_server_timing_accumulator"] = timing_acc

        pipeline = OutputPipeline(start_request, timing_accumulator=timing_acc)

        # Create per-session VAD processor from config
        from .grpc_server import ENGINE_SAMPLE_RATE
        vad_config = _build_vad_config(config)
        vad_processor = create_vad_processor(vad_config, sample_rate=ENGINE_SAMPLE_RATE)

        async def on_audio(sid: str, data: bytes) -> None:
            # Apply VAD filtering before output pipeline
            raw = np.frombuffer(data, dtype=np.float32)
            if raw.size == 0:
                return
            audio_int16 = np.clip(raw, -1.0, 1.0)
            audio_int16 = (audio_int16 * 32767.0).astype(np.int16)

            filtered_int16 = vad_processor.process_chunk(audio_int16)
            if filtered_int16.size == 0:
                return

            filtered_f32 = (filtered_int16.astype(np.float32) / 32767.0)
            filtered_bytes = filtered_f32.tobytes()

            frame = pipeline.convert_audio_chunk(filtered_bytes)
            await outbound_queue.put(
                _make_audio_frame(frame.pcm_bytes, frame.audio, meta=frame.meta)
            )

        async def on_event(sid: str, event: dict) -> None:
            await outbound_queue.put(
                _make_event_frame_from_contract(build_forward_event(sid, event, start_request))
            )

        async def on_done(sid: str, metrics: dict) -> None:
            # Flush any remaining audio from VAD
            final_int16 = vad_processor.flush()
            if final_int16.size > 0:
                final_f32 = (final_int16.astype(np.float32) / 32767.0)
                final_bytes = final_f32.tobytes()
                frame = pipeline.convert_audio_chunk(final_bytes)
                await outbound_queue.put(
                    _make_audio_frame(frame.pcm_bytes, frame.audio, meta=frame.meta)
                )

            # Inject VAD observability into metrics
            _inject_vad_metrics(vad_processor, pipeline, metrics)
            await outbound_queue.put(
                _make_event_frame_from_contract(build_done_event(sid, metrics, pipeline))
            )

        await self._engine.start_session(
            session_id,
            config=config,
            on_audio=on_audio,
            on_done=on_done,
            on_event=on_event,
        )
        await outbound_queue.put(
            _make_event_frame_from_contract(build_start_event(session_id, start_request))
        )
        logger.info("WebSocket session started: %s", session_id)
        return session_id

    async def _pump_messages(
        self,
        ws,
        request_queue: asyncio.Queue,
    ) -> None:
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid websocket JSON payload: {exc}") from exc
                    if not isinstance(payload, dict):
                        raise ValueError("websocket payload must be a JSON object")
                    await request_queue.put(("request", payload))
                    continue
                if msg.type == WSMsgType.BINARY:
                    raise ValueError("binary client frames are not supported; send JSON control messages only")
                if msg.type == WSMsgType.ERROR:
                    raise msg.data
        except Exception as exc:
            await request_queue.put(("error", exc))
        finally:
            await request_queue.put(("closed", None))

    async def _drain_available_messages(self, outbound_queue: asyncio.Queue):
        while not outbound_queue.empty():
            yield outbound_queue.get_nowait()


def _make_audio_frame(
    pcm_bytes: bytes,
    audio_config: AudioConfig,
    *,
    meta: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "audio",
        "audio": {
            "pcm_data": pcm_bytes,
            "sample_rate": audio_config.sample_rate,
            "encoding": audio_config.encoding.value,
            "channels": audio_config.channels,
            "meta": meta or {},
        },
    }


def _make_event_frame(
    *,
    event_type: str,
    session_id: str = "",
    segment_id: int = -1,
    text: str = "",
    message: str = "",
    audio_format: AudioConfig | None = None,
    meta: dict[str, str] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": event_type,
        "session_id": session_id,
        "segment_id": segment_id,
        "text": text,
        "message": message,
        "meta": meta or {},
    }
    if audio_format is not None:
        event["audio"] = {
            "encoding": audio_format.encoding.value,
            "sample_rate": audio_format.sample_rate,
            "channels": audio_format.channels,
        }
    return {
        "type": "event",
        "event": event,
    }


def _make_event_frame_from_contract(event) -> dict[str, Any]:
    payload = serialize_stream_event(event)
    audio = payload.get("audio")
    return _make_event_frame(
        event_type=str(payload.get("type", "") or ""),
        session_id=str(payload.get("session_id", "") or ""),
        segment_id=int(payload.get("segment_id", -1)),
        text=str(payload.get("text", "") or ""),
        message=str(payload.get("message", "") or ""),
        audio_format=(
            AudioConfig(
                encoding=_audio_encoding_from_ws_value(str(audio.get("encoding", "pcm_f32"))),
                sample_rate=int(audio.get("sample_rate", 24000)),
                channels=int(audio.get("channels", 1)),
            )
            if isinstance(audio, dict)
            else None
        ),
        meta={str(k): str(v) for k, v in (payload.get("meta", {}) or {}).items()},
    )


async def _send_frame(ws, frame: dict[str, Any]) -> None:
    if frame.get("type") == "audio":
        await ws.send_bytes(frame["audio"]["pcm_data"])
        return
    await ws.send_json(frame)


def _is_terminal_frame(frame: dict[str, Any]) -> bool:
    if frame.get("type") != "event":
        return False
    return frame.get("event", {}).get("type") in {"done", "error"}


async def serve(
    engine: TTSEngine,
    port: int,
    *,
    stop_event: asyncio.Event,
    path: str = "/v1/ws",
) -> None:
    """Start the websocket gateway using aiohttp."""
    if web is None:
        logger.error("aiohttp not installed. Run: pip install aiohttp")
        return

    ws_path = _normalize_ws_path(path)
    gateway = WebSocketGateway(engine)
    app = web.Application()
    app.router.add_get(_CAPABILITIES_PATH, gateway.handle_capabilities)
    app.router.add_get(ws_path, gateway.handle_websocket)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    try:
        await site.start()
        logger.info(
            "WebSocket server listening on port %d (ws path %s, capabilities %s)",
            port,
            ws_path,
            _CAPABILITIES_PATH,
        )
        await stop_event.wait()
    finally:
        await runner.cleanup()


def _session_config_from_ws_message(
    message: dict[str, Any],
    *,
    default_mode: InputMode,
) -> SessionConfig:
    return _start_request_from_ws_message(message, default_mode=default_mode).config


def _start_request_from_ws_message(
    message: dict[str, Any],
    *,
    default_mode: InputMode,
) -> SessionStartRequest:
    raw = _extract_ws_config_payload(message)
    output_policy = parse_output_policy(_extract_ws_output_policy(raw))
    timing = parse_timing_context(raw.get("timing") or raw.get("timing_context"))
    protocol_version = str(raw.get("protocol_version") or "").strip()
    if protocol_version and "client_protocol_version" not in timing.extra:
        timing.extra["client_protocol_version"] = protocol_version
    cfg = SessionConfig(
        task_type=str(raw.get("task_type", "") or ""),
        language=str(raw.get("language", "auto") or "auto"),
        speaker=_optional_str(raw.get("speaker")),
        instruct=_optional_str(raw.get("instruct")),
        ref_audio=_decode_optional_base64(raw.get("ref_audio")),
        ref_text=_optional_str(raw.get("ref_text")),
        x_vector_only=_coerce_ws_bool(raw.get("x_vector_only", False)),
        input_mode=_input_mode_from_ws_value(raw.get("input_mode"), default_mode=default_mode),
        group_policy=_group_policy_from_ws_value(raw.get("group_policy")),
        audio=_audio_config_from_ws_value(raw.get("audio")),
        output_policy=to_core_output_policy(output_policy),
        timing=to_core_timing_context(timing),
    )
    _validate_audio_config(cfg.audio)
    return SessionStartRequest(
        session_id=str(message.get("session_id", "") or ""),
        config=cfg,
        output_policy=output_policy,
        timing=timing,
        initial_text=str(message.get("text", "") or ""),
    )


def _extract_ws_output_policy(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("output_policy")
    policy = dict(value) if isinstance(value, dict) else {}
    if "vad_policy" not in policy:
        if isinstance(raw.get("vad_policy"), dict):
            policy["vad_policy"] = raw["vad_policy"]
        elif isinstance(raw.get("vad"), dict):
            policy["vad_policy"] = raw["vad"]
    return policy


def _extract_ws_config_payload(message: dict[str, Any]) -> dict[str, Any]:
    cfg = message.get("config")
    if cfg is None:
        return {
            key: value
            for key, value in message.items()
            if key not in {"type", "text", "seq_no", "session_id"}
        }
    if not isinstance(cfg, dict):
        raise ValueError("websocket 'config' must be an object")
    merged = dict(cfg)
    for field in (
        "task_type",
        "language",
        "speaker",
        "instruct",
        "ref_audio",
        "ref_text",
        "x_vector_only",
        "input_mode",
        "group_policy",
        "audio",
        "output_policy",
        "vad_policy",
        "timing",
        "timing_context",
        "protocol_version",
    ):
        if field not in merged and field in message:
            merged[field] = message[field]
    return merged


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _decode_optional_base64(value: Any) -> bytes | None:
    if value in (None, ""):
        return None
    if isinstance(value, bytes):
        return value
    try:
        return base64.b64decode(str(value), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("websocket 'ref_audio' must be valid base64") from exc


def _coerce_ws_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, "", 0):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no"}:
            return False
        if normalized in {"true", "1", "yes"}:
            return True
    return bool(value)


def _coerce_ws_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _input_mode_from_ws_value(value: Any, *, default_mode: InputMode) -> InputMode:
    if value in (None, "", 0):
        return default_mode
    if isinstance(value, int):
        mapping = {
            1: InputMode.TOKEN,
            2: InputMode.CLAUSE,
            3: InputMode.LONG_SEGMENT,
            4: InputMode.FULL_TEXT,
            5: InputMode.AUTO,
        }
        if value in mapping:
            return mapping[value]
    normalized = str(value).strip().lower()
    mapping = {
        "auto": InputMode.AUTO,
        "token": InputMode.TOKEN,
        "clause": InputMode.CLAUSE,
        "long_segment": InputMode.LONG_SEGMENT,
        "full_text": InputMode.FULL_TEXT,
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"unsupported input_mode: {value!r}")


def _group_policy_from_ws_value(value: Any) -> GroupPolicy:
    if value in (None, "", 0):
        return GroupPolicy.AUTO
    if isinstance(value, int):
        mapping = {
            1: GroupPolicy.NONE,
            2: GroupPolicy.AUTO,
        }
        if value in mapping:
            return mapping[value]
    normalized = str(value).strip().lower()
    mapping = {
        "none": GroupPolicy.NONE,
        "auto": GroupPolicy.AUTO,
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"unsupported group_policy: {value!r}")


def _audio_config_from_ws_value(value: Any) -> AudioConfig:
    if value is None:
        return AudioConfig()
    if not isinstance(value, dict):
        raise ValueError("websocket 'audio' must be an object")
    encoding = _audio_encoding_from_ws_value(value.get("encoding"))
    sample_rate = int(value.get("sample_rate", 24000) or 24000)
    channels = int(value.get("channels", 1) or 1)
    return AudioConfig(
        sample_rate=sample_rate,
        encoding=encoding,
        channels=channels,
    )


def _audio_encoding_from_ws_value(value: Any) -> AudioEncoding:
    if value in (None, "", 0, 1):
        return AudioEncoding.PCM_F32
    if value == 2:
        return AudioEncoding.PCM_S16LE
    normalized = str(value).strip().lower()
    if normalized == "pcm_f32":
        return AudioEncoding.PCM_F32
    if normalized == "pcm_s16le":
        return AudioEncoding.PCM_S16LE
    raise ValueError(f"unsupported audio encoding: {value!r}")


def _validate_audio_config(audio: AudioConfig) -> None:
    if audio.channels != 1:
        raise ValueError(f"Unsupported channel count: {audio.channels} (mono only)")
    if audio.sample_rate not in (16000, 24000):
        raise ValueError(f"Unsupported sample_rate: {audio.sample_rate} (expected 16000 or 24000)")
    if audio.encoding not in (AudioEncoding.PCM_F32, AudioEncoding.PCM_S16LE):
        raise ValueError(f"Unsupported audio encoding: {audio.encoding}")


def _normalize_ws_path(path: str) -> str:
    normalized = str(path or "/v1/ws").strip()
    if not normalized:
        normalized = "/v1/ws"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized
