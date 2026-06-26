from __future__ import annotations

from dataclasses import replace

from qwen3tts_protocol import ArrayResult, BytesResult, SessionStartRequest, SynthesisConfig

from ._adapters.engine_grpc import EngineGrpcAdapter
from ._adapters.engine_websocket import EngineWebSocketAdapter
from ._adapters.triton_grpc import TritonGrpcAdapter
from ._adapters.triton_http import TritonHttpAdapter
from .audio import decode_audio_bytes_to_array
from .constants import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_TRITON_GRPC_MODEL,
    DEFAULT_TRITON_HTTP_MODEL,
    TRANSPORT_ENGINE_GRPC,
    TRANSPORT_ENGINE_WEBSOCKET,
    TRANSPORT_TRITON_GRPC,
    TRANSPORT_TRITON_HTTP,
)
from .detect import detect_transport


class TTSClient:
    def __init__(self, *, endpoint: str, adapter, detected) -> None:
        self.endpoint = endpoint
        self._adapter = adapter
        self.resolved_transport = detected.transport
        self.probe_report = list(detected.probe_report or [])
        self.detected_transport = detected

    @classmethod
    def connect(
        cls,
        endpoint,
        *,
        transport: str = "auto",
        model_name: str | None = None,
        model_version: str = DEFAULT_MODEL_VERSION,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        metadata=None,
    ):
        detected = detect_transport(
            endpoint,
            transport=transport,
            model_name=model_name,
            model_version=model_version,
            timeout=timeout,
            headers=headers,
            metadata=metadata,
        )
        adapter = _build_adapter(
            detected.transport,
            endpoint=detected.resolved_endpoint,
            model_name=detected.model_name or model_name,
            model_version=detected.model_version or model_version,
            timeout=timeout,
            headers=headers,
            metadata=metadata,
        )
        return cls(endpoint=endpoint, adapter=adapter, detected=detected)

    def get_capabilities(self):
        return self._adapter.get_capabilities()

    def synthesize_bytes(self, text: str, *, request: SynthesisConfig | None = None) -> BytesResult:
        config = request or SynthesisConfig()
        start = SessionStartRequest(session_id="", config=config, output_policy=config.output_policy, timing=config.timing_context)
        return self._adapter.synthesize_bytes(text, request=start)

    def synthesize_array(self, text: str, *, request: SynthesisConfig | None = None) -> ArrayResult:
        result = self.synthesize_bytes(text, request=request)
        audio_array = decode_audio_bytes_to_array(result.audio_bytes, encoding=result.audio_format.encoding)
        return ArrayResult(
            audio_format=result.audio_format,
            session_id=result.session_id,
            transport=result.transport,
            events=list(result.events),
            warnings=list(result.warnings),
            details=dict(result.details),
            audio_array=audio_array,
        )

    def open_stream(self, start_request: SessionStartRequest):
        # Convenience: default the per-session output policy / timing from the
        # config so callers can pass just
        # ``SessionStartRequest(session_id=..., config=SynthesisConfig(...))``.
        if start_request.output_policy is None or start_request.timing is None:
            start_request = replace(
                start_request,
                output_policy=start_request.output_policy or start_request.config.output_policy,
                timing=start_request.timing or start_request.config.timing_context,
            )
        return self._adapter.open_stream(start_request)


def _build_adapter(
    transport: str,
    *,
    endpoint: str,
    model_name: str | None,
    model_version: str,
    timeout: float,
    headers,
    metadata,
):
    if transport == TRANSPORT_ENGINE_WEBSOCKET:
        return EngineWebSocketAdapter(endpoint, timeout=timeout, headers=headers)
    if transport == TRANSPORT_ENGINE_GRPC:
        return EngineGrpcAdapter(endpoint, timeout=timeout, metadata=metadata, headers=headers)
    if transport == TRANSPORT_TRITON_GRPC:
        return TritonGrpcAdapter(
            endpoint,
            model_name=model_name or DEFAULT_TRITON_GRPC_MODEL,
            model_version=model_version,
            timeout=timeout,
            metadata=metadata,
        )
    if transport == TRANSPORT_TRITON_HTTP:
        return TritonHttpAdapter(
            endpoint,
            model_name=model_name or DEFAULT_TRITON_HTTP_MODEL,
            model_version=model_version,
            timeout=timeout,
            headers=headers,
        )
    raise ValueError(f"unsupported transport: {transport!r}")
