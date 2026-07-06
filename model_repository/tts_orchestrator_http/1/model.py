"""HTTP/offline Triton adapter that aggregates a decoupled orchestrator stream.

This model does not load TTSEngine. It forwards one-shot synthesis requests to
the existing ``tts_orchestrator`` decoupled model through Triton BLS, collects
all streamed audio/event responses, and returns a single non-streaming response.

Why this exists:
  - Triton HTTP ``/infer`` does not support decoupled transaction policy.
  - Service layers that only support HTTP still need an offline synthesis path.
  - Loading a second TTSEngine-backed model would duplicate GPU memory usage.

Contract:
  input  ``request``     : JSON string, same payload accepted by ``tts_orchestrator``
  output ``audio_chunk`` : full aggregated PCM bytes
  output ``event_type``  : ``end`` on success, ``error`` on failure
  output ``event_json``  : final metadata JSON
  output ``is_final``    : always ``true``
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import numpy as np
import triton_python_backend_utils as pb_utils

logger = logging.getLogger("tts_orchestrator_http")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _param_string(params: dict[str, Any], key: str, default: str) -> str:
    raw = params.get(key, {})
    if isinstance(raw, dict):
        value = raw.get("string_value")
        if value is not None:
            return str(value)
    return default


def _decode_obj(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _tensor_scalar_bytes(response, name: str) -> bytes:
    tensor = pb_utils.get_output_tensor_by_name(response, name)
    if tensor is None:
        return b""
    arr = tensor.as_numpy()
    if arr is None or arr.size == 0:
        return b""
    raw = arr.reshape(-1)[0]
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return str(raw).encode("utf-8")


def _tensor_scalar_str(response, name: str) -> str:
    tensor = pb_utils.get_output_tensor_by_name(response, name)
    if tensor is None:
        return ""
    arr = tensor.as_numpy()
    if arr is None or arr.size == 0:
        return ""
    return _decode_obj(arr.reshape(-1)[0])


def _tensor_scalar_bool(response, name: str) -> bool:
    tensor = pb_utils.get_output_tensor_by_name(response, name)
    if tensor is None:
        return False
    arr = tensor.as_numpy()
    if arr is None or arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0])


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})
        self._target_model = _param_string(params, "target_model", "tts_orchestrator")
        timeout_ms = int(_param_string(params, "bls_timeout_ms", "600000"))
        self._timeout_us = max(0, timeout_ms) * 1000
        logger.info(
            "Initialized HTTP/offline aggregator: target_model=%s timeout_ms=%d",
            self._target_model,
            timeout_ms,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._execute_one(request))
            except Exception as exc:
                logger.error("Offline aggregation request failed: %s", exc)
                meta = json.dumps(
                    {"type": "error", "message": str(exc)}, ensure_ascii=False
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "audio_chunk", np.array([""], dtype=object)
                            ),
                            pb_utils.Tensor(
                                "event_type", np.array(["error"], dtype=object)
                            ),
                            pb_utils.Tensor(
                                "event_json", np.array([meta], dtype=object)
                            ),
                            pb_utils.Tensor("is_final", np.array([True], dtype=bool)),
                        ],
                        error=pb_utils.TritonError(str(exc)),
                    )
                )
        return responses

    def _execute_one(self, request):
        tensor = pb_utils.get_input_tensor_by_name(request, "request")
        if tensor is None:
            raise ValueError("Missing required input tensor 'request'")
        payload = tensor.as_numpy().reshape(-1)[0]
        if isinstance(payload, bytes):
            req_json = payload.decode("utf-8")
        else:
            req_json = str(payload)

        bls_request = pb_utils.InferenceRequest(
            model_name=self._target_model,
            requested_output_names=[
                "audio_chunk",
                "event_type",
                "event_json",
                "is_final",
            ],
            inputs=[pb_utils.Tensor("request", np.array([req_json], dtype=object))],
            timeout=self._timeout_us,
        )
        responses = bls_request.exec(decoupled=True)

        audio_parts: list[bytes] = []
        warnings: list[str] = []
        events: list[str] = []
        audio_format: dict[str, Any] = {}
        final_event_type = "done"
        final_event: dict[str, Any] = {}

        for response in responses:
            if response.has_error():
                raise RuntimeError(response.error().message())

            event_type = _tensor_scalar_str(response, "event_type")
            event_json = _tensor_scalar_str(response, "event_json")
            audio_bytes = _tensor_scalar_bytes(response, "audio_chunk")
            is_final = _tensor_scalar_bool(response, "is_final")

            payload_obj: dict[str, Any] = {}
            if event_json:
                parsed = json.loads(event_json)
                if isinstance(parsed, dict):
                    payload_obj = parsed

            if event_type:
                events.append(event_type)
            if event_type == "start":
                audio_format = payload_obj.get("audio_format", {}) or {}
            elif event_type == "audio" and audio_bytes:
                audio_parts.append(audio_bytes)
            elif event_type == "warning":
                message = str(payload_obj.get("message") or "").strip()
                if message:
                    warnings.append(message)
            elif event_type in ("done", "error"):
                final_event_type = event_type
                final_event = payload_obj

            if is_final:
                break

        if final_event_type == "error":
            raise RuntimeError(
                str(final_event.get("message") or "tts_orchestrator error")
            )

        final_meta = dict(final_event) if isinstance(final_event, dict) else {}
        if audio_format and "audio_format" not in final_meta:
            final_meta["audio_format"] = audio_format
        if warnings:
            final_meta["warnings"] = warnings
        if events:
            final_meta["events"] = events
        final_meta.setdefault("type", "done")
        final_meta["audio_chunk_encoding"] = "base64"

        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "audio_chunk",
                    np.array(
                        [base64.b64encode(b"".join(audio_parts)).decode("ascii")],
                        dtype=object,
                    ),
                ),
                pb_utils.Tensor("event_type", np.array(["done"], dtype=object)),
                pb_utils.Tensor(
                    "event_json",
                    np.array(
                        [json.dumps(final_meta, ensure_ascii=False)], dtype=object
                    ),
                ),
                pb_utils.Tensor("is_final", np.array([True], dtype=bool)),
            ]
        )
