from __future__ import annotations

import queue
import threading
import time
from typing import Any

from qwen3tts_protocol import AudioChunk, AudioFormat, BytesResult, Capabilities, SessionStartRequest, StreamEvent

from .._internal.utils import build_bytes_result, capabilities_from_payload, decode_stream_event
from .._proto import tts_pb2, tts_pb2_grpc
from ..constants import TRANSPORT_ENGINE_GRPC
from ..exceptions import DependencyMissingError
from .._session import BaseStreamSession


def _require_grpc():
    try:
        import grpc
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DependencyMissingError(
            "engine-grpc requires the 'grpc' extra. Install qwen3-tts-client[grpc]."
        ) from exc
    return grpc


class EngineGrpcAdapter:
    transport_name = TRANSPORT_ENGINE_GRPC

    def __init__(self, endpoint: str, *, timeout: float, metadata=None, headers=None) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.metadata = metadata
        self.headers = headers or {}

    def _channel(self):
        grpc = _require_grpc()
        channel = grpc.insecure_channel(self.endpoint)
        grpc.channel_ready_future(channel).result(timeout=self.timeout)
        return grpc, channel

    def get_capabilities(self) -> Capabilities:
        grpc, channel = self._channel()
        try:
            stub = tts_pb2_grpc.TTSServiceStub(channel)
            response = stub.GetCapabilities(tts_pb2.GetCapabilitiesRequest(), timeout=self.timeout)
            return capabilities_from_payload(_capabilities_message_to_dict(response))
        finally:
            channel.close()

    def synthesize_bytes(self, text: str, *, request) -> BytesResult:
        grpc, channel = self._channel()
        session_id = request.session_id or ""
        try:
            stub = tts_pb2_grpc.TTSServiceStub(channel)
            rpc_request = tts_pb2.SynthesizeOnceRequest(
                session_id=session_id,
                text=text,
                config=_session_config_to_proto(request),
            )
            audio_parts: list[bytes] = []
            audio_format = request.config.audio
            events: list[StreamEvent] = []
            warnings: list[str] = []
            for response in stub.SynthesizeOnce(rpc_request, timeout=self.timeout):
                which = response.WhichOneof("response")
                if which == "audio":
                    audio_parts.append(bytes(response.audio.pcm_data))
                    audio_format = AudioFormat(
                        encoding=_audio_encoding_from_proto(response.audio.encoding),
                        sample_rate=int(response.audio.sample_rate or audio_format.sample_rate),
                        channels=int(response.audio.channels or audio_format.channels),
                    )
                elif which == "event":
                    event = decode_stream_event(_stream_event_to_dict(response.event))
                    events.append(event)
                    if event.type == "warning" and event.message:
                        warnings.append(event.message)
                    if event.type in {"done", "error"}:
                        break
            return build_bytes_result(
                audio_bytes=b"".join(audio_parts),
                audio_format=audio_format,
                session_id=session_id,
                transport=self.transport_name,
                events=events,
                warnings=warnings,
                details={},
            )
        finally:
            channel.close()

    def open_stream(self, start_request: SessionStartRequest):
        grpc, channel = self._channel()
        return EngineGrpcStreamSession(self, grpc, channel, start_request)


class EngineGrpcStreamSession(BaseStreamSession):
    def __init__(self, adapter: EngineGrpcAdapter, grpc, channel, start_request: SessionStartRequest) -> None:
        super().__init__(session_id=start_request.session_id, transport=adapter.transport_name)
        self._adapter = adapter
        self._grpc = grpc
        self._channel = channel
        self._stub = tts_pb2_grpc.TTSServiceStub(channel)
        self._start_request = start_request
        self._request_queue: queue.Queue[object] = queue.Queue()
        self._stream = self._stub.SynthesizeStream(self._request_iter(), timeout=adapter.timeout)
        self._reader = threading.Thread(target=self._reader_loop, name=f"grpc-session-{self.session_id}", daemon=True)
        self._reader.start()

    def _request_iter(self):
        yield tts_pb2.SynthesizeRequest(
            start=tts_pb2.StartRequest(
                session_id=self.session_id,
                config=_session_config_to_proto(self._start_request),
            )
        )
        while True:
            item = self._request_queue.get()
            if item is None:
                return
            yield item

    def _reader_loop(self) -> None:
        try:
            for response in self._stream:
                which = response.WhichOneof("response")
                if which == "audio":
                    self._put_message(
                        AudioChunk(
                            pcm_bytes=bytes(response.audio.pcm_data),
                            audio=AudioFormat(
                                encoding=_audio_encoding_from_proto(response.audio.encoding),
                                sample_rate=int(response.audio.sample_rate or self._start_request.config.audio.sample_rate),
                                channels=int(response.audio.channels or self._start_request.config.audio.channels),
                            ),
                            meta={str(k): str(v) for k, v in dict(response.audio.meta or {}).items()},
                        )
                    )
                elif which == "event":
                    event = decode_stream_event(_stream_event_to_dict(response.event))
                    self._put_message(event)
                    if event.type in {"done", "error"}:
                        break
        except Exception as exc:
            self._put_message(
                StreamEvent(type="error", session_id=self.session_id, message=str(exc))
            )
        finally:
            self._channel.close()

    def send_text(self, text: str, *, seq_no: int | None = None, client_timestamp_ms: int | None = None) -> None:
        self._check_send_open()
        self._request_queue.put(
            tts_pb2.SynthesizeRequest(
                text=tts_pb2.TextChunk(
                    text=text,
                    seq_no=int(seq_no or 0),
                    client_timestamp_ms=int(client_timestamp_ms or 0),
                )
            )
        )

    def end(self, *, client_timestamp_ms: int | None = None) -> None:
        self._check_send_open()
        self._mark_send_closed()
        self._request_queue.put(
            tts_pb2.SynthesizeRequest(
                end=tts_pb2.EndRequest(client_timestamp_ms=int(client_timestamp_ms or 0))
            )
        )
        self._request_queue.put(None)

    def cancel(self, reason: str = "") -> None:
        if self._send_closed:
            return
        self._mark_send_closed()
        self._request_queue.put(
            tts_pb2.SynthesizeRequest(cancel=tts_pb2.CancelRequest(reason=reason))
        )
        self._request_queue.put(None)


def _audio_encoding_from_proto(value) -> str:
    return "pcm_s16le" if value == tts_pb2.AUDIO_ENCODING_PCM_S16LE else "pcm_f32"


def _audio_encoding_to_proto(value: str):
    normalized = str(value or "").strip().lower()
    if normalized == "pcm_s16le":
        return tts_pb2.AUDIO_ENCODING_PCM_S16LE
    return tts_pb2.AUDIO_ENCODING_PCM_F32


def _session_config_to_proto(start_request: SessionStartRequest):
    cfg = start_request.config
    return tts_pb2.SessionConfig(
        task_type=cfg.task_type,
        language=cfg.language,
        speaker=cfg.speaker or "",
        instruct=cfg.instruct or "",
        ref_audio=cfg.ref_audio or b"",
        ref_text=cfg.ref_text or "",
        x_vector_only=bool(cfg.x_vector_only),
        input_mode=_input_mode_to_proto(cfg.input_mode),
        group_policy=_group_policy_to_proto(cfg.group_policy),
        audio=tts_pb2.AudioFormat(
            encoding=_audio_encoding_to_proto(cfg.audio.encoding),
            sample_rate=int(cfg.audio.sample_rate),
            channels=int(cfg.audio.channels),
        ),
        output_policy=_output_policy_to_proto(start_request.output_policy),
        timing=_timing_context_to_proto(start_request.timing),
        protocol_version=cfg.protocol_version or "",
    )


def _input_mode_to_proto(value: str):
    mapping = {
        "token": tts_pb2.INPUT_MODE_TOKEN,
        "clause": tts_pb2.INPUT_MODE_CLAUSE,
        "long_segment": tts_pb2.INPUT_MODE_LONG_SEGMENT,
        "full_text": tts_pb2.INPUT_MODE_FULL_TEXT,
    }
    return mapping.get(str(value or "").strip().lower(), tts_pb2.INPUT_MODE_LONG_SEGMENT)


def _group_policy_to_proto(value: str):
    mapping = {
        "none": tts_pb2.GROUP_POLICY_NONE,
        "auto": tts_pb2.GROUP_POLICY_AUTO,
    }
    return mapping.get(str(value or "").strip().lower(), tts_pb2.GROUP_POLICY_AUTO)


# VAD tuning params live on the dataclass but have no dedicated proto fields,
# so they ride through the proto VADPolicy.config string-map. The engine side
# (_output_policy_from_proto) lifts them back out.
_VAD_TUNING_FIELDS = (
    "chunk_ms",
    "begin_threshold",
    "begin_count",
    "end_threshold",
    "end_count",
    "start_margin_ms",
)


def _output_policy_to_proto(policy):
    vad = policy.vad
    vad_config = {str(k): str(v) for k, v in dict(vad.config or {}).items()}
    # Carry the tuning params through config for non-disabled strategies;
    # otherwise a gRPC client tuning thresholds silently gets engine defaults
    # (the WebSocket path transmits them, so this keeps transports consistent).
    if str(vad.strategy or "disabled") not in ("disabled", ""):
        for field_name in _VAD_TUNING_FIELDS:
            vad_config[field_name] = str(getattr(vad, field_name))
    return tts_pb2.OutputPolicy(
        vad_policy=tts_pb2.VADPolicy(
            enabled=bool(vad.enabled),
            strategy=str(vad.strategy or "disabled"),
            implementation=str(vad.implementation or ""),
            config=vad_config,
        ),
        chunk_ms=int(policy.chunk_ms or 0),
        packet_format=str(policy.packet_format or "raw_pcm"),
        emit_text_events=bool(policy.emit_text_events),
        config={str(k): str(v) for k, v in dict(policy.config or {}).items()},
    )


def _timing_context_to_proto(timing):
    return tts_pb2.TimingContext(
        request_id=timing.request_id or "",
        turn_id=timing.turn_id or "",
        client_request_ts_ms=int(timing.client_request_ts_ms or 0),
        client_text_ts_ms=int(timing.client_text_ts_ms or 0),
        client_end_ts_ms=int(timing.client_end_ts_ms or 0),
        extra={str(k): str(v) for k, v in dict(timing.extra or {}).items()},
    )


def _stream_event_to_dict(event) -> dict[str, Any]:
    payload = {
        "type": event.type,
        "session_id": event.session_id,
        "segment_id": event.segment_id,
        "text": event.text,
        "message": event.message,
        "meta": {str(k): str(v) for k, v in dict(event.meta or {}).items()},
    }
    if event.audio is not None:
        payload["audio"] = {
            "encoding": _audio_encoding_from_proto(event.audio.encoding),
            "sample_rate": int(event.audio.sample_rate),
            "channels": int(event.audio.channels),
        }
    return payload


def _capabilities_message_to_dict(resp) -> dict[str, Any]:
    return {
        "variant": resp.variant,
        "loaded_model_type": resp.loaded_model_type,
        "declared_supported_task_types": list(resp.declared_supported_task_types),
        # proto3 repeated enum fields come across as ints — map them back to the
        # enum names before stripping the prefix (str(int) would yield "4", etc.).
        "supported_input_modes": [
            tts_pb2.InputMode.Name(int(value)).lower().replace("input_mode_", "")
            for value in resp.supported_input_modes
        ],
        "supported_group_policies": [
            tts_pb2.GroupPolicy.Name(int(value)).lower().replace("group_policy_", "")
            for value in resp.supported_group_policies
        ],
        "supported_audio_formats": [
            {
                "encoding": _audio_encoding_from_proto(fmt.encoding),
                "sample_rate": fmt.sample_rate,
                "channels": fmt.channels,
            }
            for fmt in resp.supported_audio_formats
        ],
        "ref_audio_available": bool(resp.ref_audio_available),
        "ref_audio_reason": resp.ref_audio_reason,
        "speaker_encoder_available": bool(resp.speaker_encoder_available),
        "ref_codec_available": bool(resp.ref_codec_available),
        "icl_available": bool(resp.icl_available),
        "ref_audio_max_duration_sec": float(resp.ref_audio_max_duration_sec or 0.0),
        "ref_c2w_warm_state_available": bool(resp.ref_c2w_warm_state_available),
        "ref_codec_reason": resp.ref_codec_reason,
        "protocol_version": resp.protocol_version,
        "supported_output_policy_features": list(resp.supported_output_policy_features),
        "supported_vad_strategies": list(resp.supported_vad_strategies),
        "supported_timing_fields": list(resp.supported_timing_fields),
    }
