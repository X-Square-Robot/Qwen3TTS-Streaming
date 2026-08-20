from __future__ import annotations

import base64
import json
import threading
from typing import Any

import requests

from qwen3tts_protocol import (
    AudioChunk,
    AudioFormat,
    Capabilities,
    SessionStartRequest,
    StreamEvent,
    serialize_output_policy,
)

from .._internal.utils import (
    build_bytes_result,
    capabilities_from_payload,
    synthesis_config_to_mapping,
)
from .._internal.tls import TLSConfig, TLSVerify
from .._session import BaseStreamSession
from ..constants import DEFAULT_MODEL_VERSION, TRANSPORT_TRITON_HTTP


class TritonHttpAdapter:
    transport_name = TRANSPORT_TRITON_HTTP

    def __init__(
        self,
        base_url: str,
        *,
        model_name: str,
        model_version: str = DEFAULT_MODEL_VERSION,
        timeout: float,
        headers: dict[str, str] | None = None,
        tls_verify: TLSVerify | TLSConfig = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.model_version = model_version
        self.timeout = timeout
        self.headers = dict(headers or {})
        self._tls = TLSConfig.from_value(tls_verify)

    def get_capabilities(self, *, timeout: float | None = None) -> Capabilities:
        effective_timeout = (
            self.timeout if timeout is None else max(0.0, float(timeout))
        )
        infer_url = self._infer_url()
        payload = self._infer_payload({"action": "capabilities"})
        response = requests.post(
            infer_url,
            json=payload,
            timeout=effective_timeout,
            headers=self.headers,
            **self._tls.requests_kwargs(),
        )
        caps = _parse_capabilities_from_http_response(response)
        if caps is None:
            raise RuntimeError(
                f"triton-http capabilities probe failed: HTTP {response.status_code} {response.text[:300]}"
            )
        return capabilities_from_payload(caps)

    def synthesize_bytes(self, text: str, *, request):
        infer_url = self._infer_url()
        request_payload = self._request_payload_for_text(request, text)
        response = requests.post(
            infer_url,
            json=self._infer_payload(request_payload),
            timeout=self.timeout,
            headers=self.headers,
            **self._tls.requests_kwargs(),
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"triton-http infer failed: HTTP {response.status_code} {response.text[:300]}"
            )
        body = response.json()
        outputs = {item.get("name"): item for item in body.get("outputs", [])}
        audio_field = _first_output_scalar(outputs.get("audio_chunk"))
        event_type = str(_first_output_scalar(outputs.get("event_type")) or "")
        event_json = str(_first_output_scalar(outputs.get("event_json")) or "")
        meta = json.loads(event_json) if event_json else {}
        if not isinstance(meta, dict):
            meta = {}
        raw_bytes = b""
        if audio_field:
            if (
                str(meta.get("audio_chunk_encoding", "") or "").strip().lower()
                == "base64"
            ):
                raw_bytes = base64.b64decode(str(audio_field))
            elif isinstance(audio_field, str):
                raw_bytes = audio_field.encode("utf-8")
            elif isinstance(audio_field, bytes):
                raw_bytes = audio_field
        audio_meta = meta.get("audio_format", {}) or {}
        audio_format = AudioFormat(
            encoding=str(audio_meta.get("encoding", request.config.audio.encoding)),
            sample_rate=int(
                audio_meta.get("sample_rate", request.config.audio.sample_rate)
            ),
            channels=int(audio_meta.get("channels", request.config.audio.channels)),
        )
        events = []
        warnings = [str(item) for item in meta.get("warnings", []) or []]
        events.append(
            StreamEvent(
                type="error" if event_type == "error" else "done",
                session_id=request.session_id,
                message=str(meta.get("message", "") or ""),
                meta={str(k): str(v) for k, v in meta.items() if k not in {"warnings"}},
            )
        )
        details = {"degraded_to_oneshot": False}
        return build_bytes_result(
            audio_bytes=raw_bytes,
            audio_format=audio_format,
            session_id=request.session_id,
            transport=self.transport_name,
            events=events,
            warnings=warnings,
            details=details,
        )

    def open_stream(self, start_request: SessionStartRequest):
        return TritonHttpBufferedSession(self, start_request)

    def _request_payload_for_text(
        self, request: SessionStartRequest, text: str
    ) -> dict[str, Any]:
        payload = synthesis_config_to_mapping(request.config)
        payload["text"] = text
        payload["session_id"] = request.session_id
        if request.output_policy:
            payload["output_policy"] = serialize_output_policy(request.output_policy)
        if request.timing:
            payload["timing"] = {
                "request_id": request.timing.request_id,
                "turn_id": request.timing.turn_id,
                "client_request_ts_ms": int(request.timing.client_request_ts_ms or 0),
                "client_text_ts_ms": int(request.timing.client_text_ts_ms or 0),
                "client_end_ts_ms": int(request.timing.client_end_ts_ms or 0),
                "extra": dict(request.timing.extra or {}),
            }
        return payload

    def _infer_payload(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "inputs": [
                {
                    "name": "request",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [json.dumps(request_payload, ensure_ascii=False)],
                }
            ],
            "outputs": [
                {"name": "audio_chunk"},
                {"name": "event_type"},
                {"name": "event_json"},
                {"name": "is_final"},
            ],
        }

    def _infer_url(self) -> str:
        return f"{self.base_url}/v2/models/{self.model_name}/versions/{self.model_version}/infer"


def _first_output_scalar(output: dict[str, Any] | None) -> Any:
    if not output:
        return None
    data = output.get("data")
    if isinstance(data, list) and data:
        return data[0]
    contents = output.get("contents")
    if isinstance(contents, dict):
        for key in (
            "bytes_contents",
            "string_contents",
            "bool_contents",
            "int_contents",
            "int64_contents",
            "uint_contents",
            "uint64_contents",
            "fp32_contents",
            "fp64_contents",
        ):
            values = contents.get(key)
            if isinstance(values, list) and values:
                return values[0]
    return None


def _parse_capabilities_from_http_response(response) -> dict[str, Any] | None:
    text = response.text or ""
    marker = "CAPABILITIES:"
    if marker in text:
        payload = text.split(marker, 1)[1]
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    try:
        body = response.json()
    except Exception:
        return None
    if isinstance(body, dict):
        outputs = {item.get("name"): item for item in body.get("outputs", [])}
        event_json = _first_output_scalar(outputs.get("event_json"))
        if event_json:
            try:
                parsed = json.loads(str(event_json))
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and "loaded_model_type" in parsed:
                return parsed
    return None


class TritonHttpBufferedSession(BaseStreamSession):
    def __init__(
        self, adapter: TritonHttpAdapter, start_request: SessionStartRequest
    ) -> None:
        super().__init__(
            session_id=start_request.session_id, transport=adapter.transport_name
        )
        self._adapter = adapter
        self._start_request = start_request
        self._text_parts: list[str] = []
        self.degraded_to_oneshot = True

    def send_text(
        self,
        text: str,
        *,
        seq_no: int | None = None,
        client_timestamp_ms: int | None = None,
    ) -> None:
        self._check_send_open()
        self._text_parts.append(text)

    def end(self, *, client_timestamp_ms: int | None = None) -> None:
        self._check_send_open()
        self._mark_send_closed()
        if client_timestamp_ms is not None:
            self._start_request.timing.client_end_ts_ms = int(client_timestamp_ms)
        worker = threading.Thread(
            target=self._run_degraded_oneshot,
            name=f"triton-http-{self.session_id}",
            daemon=True,
        )
        worker.start()

    def cancel(self, reason: str = "") -> None:
        if self._send_closed:
            return
        self._mark_send_closed()
        self._put_message(
            StreamEvent(type="done", session_id=self.session_id, message=reason)
        )

    def _run_degraded_oneshot(self) -> None:
        try:
            text = "".join(self._text_parts)
            result = self._adapter.synthesize_bytes(text, request=self._start_request)
            start_event = StreamEvent(
                type="start",
                session_id=self.session_id,
                audio=result.audio_format,
                meta={
                    "degraded_to_oneshot": "true",
                    "transport_warning": "triton-http-buffered-stream",
                },
            )
            self._put_message(start_event)
            if result.audio_bytes:
                self._put_message(
                    AudioChunk(
                        pcm_bytes=result.audio_bytes,
                        audio=result.audio_format,
                        chunk_index=0,
                        first_chunk=True,
                        final_chunk=True,
                        meta={"degraded_to_oneshot": "true"},
                    )
                )
            for warning in result.warnings:
                self._put_message(
                    StreamEvent(
                        type="warning",
                        session_id=self.session_id,
                        message=warning,
                        meta={"degraded_to_oneshot": "true"},
                    )
                )
            final_event = (
                result.events[-1]
                if result.events
                else StreamEvent(type="done", session_id=self.session_id)
            )
            final_meta = dict(final_event.meta)
            final_meta["degraded_to_oneshot"] = "true"
            self._put_message(
                StreamEvent(
                    type=final_event.type,
                    session_id=self.session_id,
                    message=final_event.message,
                    meta=final_meta,
                )
            )
        except Exception as exc:
            self._put_message(
                StreamEvent(type="error", session_id=self.session_id, message=str(exc))
            )
