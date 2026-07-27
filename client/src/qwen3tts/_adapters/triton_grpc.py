from __future__ import annotations

import base64
import json
import queue
import threading
from typing import Any

from qwen3tts_protocol import (
    AudioChunk,
    AudioFormat,
    BytesResult,
    Capabilities,
    SessionStartRequest,
    StreamEvent,
)

from .._internal.auth import grpc_metadata_as_headers, normalize_grpc_metadata
from .._internal.utils import (
    build_bytes_result,
    capabilities_from_payload,
    synthesis_config_to_mapping,
)
from .._session import BaseStreamSession
from ..constants import DEFAULT_TRITON_GRPC_MODEL, TRANSPORT_TRITON_GRPC
from ..exceptions import DependencyMissingError


def _require_triton():
    try:
        import numpy as np
        import tritonclient.grpc as grpcclient
    except ImportError as exc:  # pragma: no cover
        raise DependencyMissingError(
            "triton-grpc requires the 'triton' extra. Install qwen3-tts-client[triton]."
        ) from exc
    return np, grpcclient


class TritonGrpcAdapter:
    transport_name = TRANSPORT_TRITON_GRPC

    def __init__(
        self,
        endpoint: str,
        *,
        model_name: str = DEFAULT_TRITON_GRPC_MODEL,
        model_version: str = "1",
        timeout: float,
        metadata=None,
        headers=None,
    ) -> None:
        self.endpoint = endpoint
        self.model_name = model_name
        self.model_version = model_version
        self.timeout = timeout
        self.metadata = normalize_grpc_metadata(metadata, headers)
        self.headers = grpc_metadata_as_headers(self.metadata)

    def get_capabilities(self) -> Capabilities:
        np, grpcclient = _require_triton()
        client = grpcclient.InferenceServerClient(url=self.endpoint)
        if not client.is_server_live(headers=self.headers, client_timeout=self.timeout):
            raise RuntimeError(f"triton gRPC server is not live at {self.endpoint}")
        if not client.is_server_ready(
            headers=self.headers, client_timeout=self.timeout
        ):
            raise RuntimeError(f"triton gRPC server is not ready at {self.endpoint}")
        if self.model_name and not client.is_model_ready(
            self.model_name,
            headers=self.headers,
            client_timeout=self.timeout,
        ):
            raise RuntimeError(f"triton model {self.model_name!r} is not ready")
        payload = {"action": "capabilities"}
        req_input = grpcclient.InferInput("request", [1], "BYTES")
        req_input.set_data_from_numpy(
            np.array([json.dumps(payload, ensure_ascii=False)], dtype=object)
        )
        audio_out, event_type_out, event_json_out, final_out = _outputs(grpcclient)
        done = threading.Event()
        caps_holder: list[dict[str, Any]] = []
        errors: list[str] = []

        def callback(result=None, error=None, **kwargs):
            infer_result = kwargs.get("result", result)
            error = kwargs.get("error", error)
            if error is not None:
                raw = str(error)
                marker = "CAPABILITIES:"
                if marker in raw:
                    try:
                        caps_holder.append(json.loads(raw.split(marker, 1)[1]))
                    except json.JSONDecodeError:
                        errors.append(raw)
                else:
                    errors.append(raw)
                done.set()
                return
            if infer_result is None:
                return
            payload_json = _scalar_from_result(infer_result, "event_json")
            if payload_json:
                try:
                    parsed = json.loads(str(payload_json))
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and "loaded_model_type" in parsed:
                    caps_holder.append(parsed)
                    done.set()
            final = _scalar_from_result(infer_result, "is_final")
            if final:
                done.set()

        client.start_stream(callback=callback, headers=self.headers)
        try:
            client.async_stream_infer(
                model_name=self.model_name,
                inputs=[req_input],
                outputs=[audio_out, event_type_out, event_json_out, final_out],
            )
            done.wait(self.timeout)
        finally:
            client.stop_stream()
        if caps_holder:
            return capabilities_from_payload(caps_holder[-1])
        if errors:
            raise RuntimeError(errors[0])
        return capabilities_from_payload(
            {"variant": "triton", "loaded_model_type": self.model_name}
        )

    def synthesize_bytes(self, text: str, *, request) -> BytesResult:
        session = self.open_stream(request)
        session.send_text(text)
        session.end()
        audio_parts: list[bytes] = []
        audio_format = request.config.audio
        events: list[StreamEvent] = []
        warnings: list[str] = []
        for message in session.iter_messages():
            if isinstance(message, AudioChunk):
                audio_parts.append(message.pcm_bytes)
                audio_format = message.audio
                continue
            events.append(message)
            if message.type == "warning" and message.message:
                warnings.append(message.message)
        return build_bytes_result(
            audio_bytes=b"".join(audio_parts),
            audio_format=audio_format,
            session_id=request.session_id,
            transport=self.transport_name,
            events=events,
            warnings=warnings,
            details={},
        )

    def open_stream(self, start_request: SessionStartRequest):
        return TritonGrpcStreamSession(self, start_request)


class TritonGrpcStreamSession(BaseStreamSession):
    def __init__(
        self, adapter: TritonGrpcAdapter, start_request: SessionStartRequest
    ) -> None:
        super().__init__(
            session_id=start_request.session_id, transport=adapter.transport_name
        )
        self._adapter = adapter
        self._start_request = start_request
        self._send_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._worker = threading.Thread(
            target=self._run, name=f"triton-grpc-{self.session_id}", daemon=True
        )
        self._worker.start()
        self._send_queue.put(_build_stream_request("start", start_request))

    def send_text(
        self,
        text: str,
        *,
        seq_no: int | None = None,
        client_timestamp_ms: int | None = None,
    ) -> None:
        self._check_send_open()
        payload = {
            "action": "append_text",
            "session_id": self.session_id,
            "text": text,
        }
        if seq_no is not None:
            payload["seq_no"] = int(seq_no)
        if client_timestamp_ms is not None:
            payload["client_timestamp_ms"] = int(client_timestamp_ms)
        self._send_queue.put(payload)

    def end(self, *, client_timestamp_ms: int | None = None) -> None:
        self._check_send_open()
        self._mark_send_closed()
        payload = {
            "action": "text_complete",
            "session_id": self.session_id,
        }
        if client_timestamp_ms is not None:
            payload["client_timestamp_ms"] = int(client_timestamp_ms)
        self._send_queue.put(payload)
        self._send_queue.put(None)

    def cancel(self, reason: str = "") -> None:
        if self._send_closed:
            return
        self._mark_send_closed()
        self._send_queue.put(
            {
                "action": "cancel",
                "session_id": self.session_id,
                "reason": reason,
            }
        )
        self._send_queue.put(None)

    def _run(self) -> None:
        np, grpcclient = _require_triton()
        client = grpcclient.InferenceServerClient(url=self._adapter.endpoint)
        audio_out, event_type_out, event_json_out, final_out = _outputs(grpcclient)
        done = threading.Event()
        first_audio_format = self._start_request.config.audio
        current_audio_format = first_audio_format
        errors: list[str] = []

        def callback(result=None, error=None, **kwargs):
            nonlocal current_audio_format
            infer_result = kwargs.get("result", result)
            error = kwargs.get("error", error)
            if error is not None:
                errors.append(str(error))
                done.set()
                return
            if infer_result is None:
                return
            event_type = str(_scalar_from_result(infer_result, "event_type") or "")
            event_json = _scalar_from_result(infer_result, "event_json")
            audio_value = _scalar_from_result(infer_result, "audio_chunk")
            is_final = bool(_scalar_from_result(infer_result, "is_final"))
            payload: dict[str, Any] = {}
            if event_json:
                try:
                    parsed = json.loads(str(event_json))
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    payload = parsed
            if event_type == "start":
                audio_meta = payload.get("audio_format") or payload.get("audio") or {}
                current_audio_format = AudioFormat(
                    encoding=str(
                        audio_meta.get("encoding", current_audio_format.encoding)
                    ),
                    sample_rate=int(
                        audio_meta.get("sample_rate", current_audio_format.sample_rate)
                    ),
                    channels=int(
                        audio_meta.get("channels", current_audio_format.channels)
                    ),
                )
                self._put_message(
                    StreamEvent(
                        type="start",
                        session_id=str(
                            payload.get("session_id", self.session_id)
                            or self.session_id
                        ),
                        audio=current_audio_format,
                        meta={
                            str(k): str(v)
                            for k, v in dict(payload.get("meta") or {}).items()
                        },
                    )
                )
            elif event_type == "audio":
                raw_bytes = _decode_audio_bytes_field(audio_value, payload)
                self._put_message(
                    AudioChunk(
                        pcm_bytes=raw_bytes,
                        audio=current_audio_format,
                        meta={
                            str(k): str(v)
                            for k, v in dict(payload.get("meta") or {}).items()
                        },
                    )
                )
            elif event_type:
                self._put_message(
                    StreamEvent(
                        type=event_type,
                        session_id=str(
                            payload.get("session_id", self.session_id)
                            or self.session_id
                        ),
                        segment_id=int(payload.get("segment_id", -1)),
                        text=str(payload.get("text", "") or ""),
                        message=str(payload.get("message", "") or ""),
                        meta={
                            str(k): str(v)
                            for k, v in dict(payload.get("meta") or {}).items()
                        },
                    )
                )
            if is_final or event_type in {"done", "error"}:
                done.set()

        client.start_stream(callback=callback, headers=self._adapter.headers)
        try:
            while True:
                request_payload = self._send_queue.get()
                if request_payload is None:
                    break
                req_input = grpcclient.InferInput("request", [1], "BYTES")
                req_input.set_data_from_numpy(
                    np.array(
                        [json.dumps(request_payload, ensure_ascii=False)], dtype=object
                    )
                )
                client.async_stream_infer(
                    model_name=self._adapter.model_name,
                    inputs=[req_input],
                    outputs=[audio_out, event_type_out, event_json_out, final_out],
                )
            done.wait(self._adapter.timeout)
        except Exception as exc:
            errors.append(str(exc))
        finally:
            client.stop_stream()
        if errors:
            self._put_message(
                StreamEvent(type="error", session_id=self.session_id, message=errors[0])
            )
        elif not self._closed:
            # The stream ended without an explicit end/error event (e.g. the
            # engine set is_final on an audio chunk). Emit a terminal so the
            # queue sentinel is enqueued and iter_messages() doesn't hang.
            self._put_message(StreamEvent(type="done", session_id=self.session_id))


def _outputs(grpcclient):
    return (
        grpcclient.InferRequestedOutput("audio_chunk"),
        grpcclient.InferRequestedOutput("event_type"),
        grpcclient.InferRequestedOutput("event_json"),
        grpcclient.InferRequestedOutput("is_final"),
    )


def _build_stream_request(
    action: str, start_request: SessionStartRequest
) -> dict[str, Any]:
    config = synthesis_config_to_mapping(start_request.config)
    config["session_id"] = start_request.session_id
    config["action"] = "init" if action == "start" else action
    if start_request.output_policy:
        config["output_policy"] = {
            "vad_policy": {
                "enabled": bool(start_request.output_policy.vad.enabled),
                "strategy": str(start_request.output_policy.vad.strategy or "disabled"),
                "implementation": str(
                    start_request.output_policy.vad.implementation or ""
                ),
                "config": dict(start_request.output_policy.vad.config or {}),
            },
            "chunk_ms": int(start_request.output_policy.chunk_ms or 0),
            "packet_format": str(
                start_request.output_policy.packet_format or "raw_pcm"
            ),
            "emit_text_events": bool(start_request.output_policy.emit_text_events),
            "config": dict(start_request.output_policy.config or {}),
        }
    if start_request.timing:
        config["timing"] = {
            "request_id": start_request.timing.request_id,
            "turn_id": start_request.timing.turn_id,
            "client_request_ts_ms": int(start_request.timing.client_request_ts_ms or 0),
            "client_text_ts_ms": int(start_request.timing.client_text_ts_ms or 0),
            "client_end_ts_ms": int(start_request.timing.client_end_ts_ms or 0),
            "extra": dict(start_request.timing.extra or {}),
        }
    return config


def _scalar_from_result(result, name: str):
    tensor = result.as_numpy(name)
    if tensor is None or not getattr(tensor, "size", 0):
        return None
    value = tensor.reshape(-1)[0]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    return value


def _decode_audio_bytes_field(value: Any, payload: dict[str, Any]) -> bytes:
    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        if (
            str(payload.get("audio_chunk_encoding", "") or "").strip().lower()
            == "base64"
        ):
            return base64.b64decode(value)
        return value.encode("utf-8")
    return bytes(value)
