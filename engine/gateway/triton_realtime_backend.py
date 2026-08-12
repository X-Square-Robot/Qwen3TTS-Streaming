"""Triton gRPC execution backend for the OpenAI Realtime gateway.

One public Realtime response maps to one Triton client stream.  The initial
``init`` request owns the decoupled audio response, while ``append_text``,
``text_complete`` and ``cancel`` requests travel in the opposite direction on
the same gRPC stream.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..frontend.interface import _normalize_tts_text
from ..frontend.spliter.tokenizer import LightQwen3TTSTokenizer
from ..interface import serialize_output_policy
from .session_identity import GatewaySessionIdentity

if TYPE_CHECKING:
    from ..interface import SessionStartRequest


logger = logging.getLogger(__name__)


def _require_triton():
    try:
        import numpy as np
        import tritonclient.grpc as grpcclient
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "Triton Realtime gateway requires tritonclient[grpc] and numpy"
        ) from exc
    return np, grpcclient


class _TokenizerCounter:
    def __init__(self, tokenizer_dir: str) -> None:
        path = Path(tokenizer_dir)
        if not path.is_dir():
            raise ValueError(f"Tokenizer directory does not exist: {path}")
        self._tokenizer = LightQwen3TTSTokenizer(str(path))

    def __call__(self, text: str) -> int:
        normalized = _normalize_tts_text(text).strip()
        if not normalized:
            return 0
        return len(self._tokenizer.encode_ids(normalized, add_special_tokens=False))


@dataclass(frozen=True)
class _DecodedResult:
    event_type: str = ""
    payload: dict[str, Any] | None = None
    audio: bytes = b""
    is_final: bool = False
    error: str = ""


class TritonRealtimeBackend:
    """Implement ``RealtimeSessionBackend`` over Triton streaming gRPC."""

    def __init__(
        self,
        endpoint: str,
        *,
        model_name: str = "tts_orchestrator",
        model_version: str = "",
        tokenizer_dir: str = "",
        token_counter: Callable[[str], int] | None = None,
        headers: dict[str, str] | None = None,
        start_timeout: float = 30.0,
        cancel_timeout: float = 5.0,
    ) -> None:
        self.endpoint = str(endpoint)
        self.model_name = str(model_name)
        self.model_version = str(model_version or "")
        self.headers = {str(k): str(v) for k, v in (headers or {}).items()}
        self.start_timeout = max(0.1, float(start_timeout))
        self.cancel_timeout = max(0.1, float(cancel_timeout))
        if token_counter is None:
            if not tokenizer_dir:
                raise ValueError(
                    "Triton Realtime backend requires tokenizer_dir for usage accounting"
                )
            token_counter = _TokenizerCounter(tokenizer_dir)
        self._token_counter = token_counter
        self._sessions: dict[str, _TritonResponseStream] = {}
        self._sessions_lock = asyncio.Lock()

    async def start(
        self,
        identity: GatewaySessionIdentity,
        *,
        start_request: "SessionStartRequest",
        outbound_queue: asyncio.Queue,
    ) -> None:
        session_id = identity.internal_session_id
        identity.bind_engine_config(start_request.config)
        stream = _TritonResponseStream(
            backend=self,
            session_id=session_id,
            start_payload=_start_payload(session_id, start_request),
            outbound_queue=outbound_queue,
            requested_audio={
                "encoding": start_request.config.audio.encoding.value,
                "sample_rate": int(start_request.config.audio.sample_rate),
                "channels": int(start_request.config.audio.channels),
            },
        )
        async with self._sessions_lock:
            if session_id in self._sessions:
                raise RuntimeError(f"duplicate Triton Realtime session: {session_id}")
            self._sessions[session_id] = stream
        try:
            await stream.open()
        except asyncio.CancelledError:
            # response.cancel will call backend.cancel with the same private ID.
            raise
        except Exception:
            await self._remove_and_close(session_id, stream)
            raise

    async def push_text(self, session_id: str, text: str) -> None:
        stream = await self._require_session(session_id)
        await stream.submit(
            {"action": "append_text", "session_id": session_id, "text": text}
        )

    async def complete_input(self, session_id: str) -> None:
        stream = await self._require_session(session_id)
        await stream.submit({"action": "text_complete", "session_id": session_id})

    async def cancel(self, session_id: str) -> None:
        async with self._sessions_lock:
            stream = self._sessions.get(session_id)
        if stream is None:
            return
        try:
            await stream.submit({"action": "cancel", "session_id": session_id})
            try:
                await asyncio.wait_for(
                    stream.wait_terminal(), timeout=self.cancel_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out waiting for Triton cancel acknowledgement: %s",
                    session_id,
                )
        finally:
            await self._remove_and_close(session_id, stream)

    def count_text_tokens(self, text: str) -> int:
        return int(self._token_counter(text))

    async def is_ready(self, *, timeout: float = 2.0) -> bool:
        def check() -> bool:
            _np, grpcclient = _require_triton()
            client = grpcclient.InferenceServerClient(url=self.endpoint)
            try:
                kwargs = {
                    "headers": self.headers,
                    "client_timeout": max(0.1, float(timeout)),
                }
                return bool(
                    client.is_server_live(**kwargs)
                    and client.is_server_ready(**kwargs)
                    and client.is_model_ready(
                        self.model_name,
                        model_version=self.model_version,
                        **kwargs,
                    )
                )
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()

        try:
            return await asyncio.to_thread(check)
        except Exception:
            logger.exception("Triton readiness check failed: %s", self.endpoint)
            return False

    async def close(self) -> None:
        async with self._sessions_lock:
            sessions = list(self._sessions.items())
        for session_id, stream in sessions:
            try:
                await stream.submit({"action": "cancel", "session_id": session_id})
            except Exception:
                pass
            await self._remove_and_close(session_id, stream)

    async def _require_session(self, session_id: str) -> "_TritonResponseStream":
        async with self._sessions_lock:
            stream = self._sessions.get(session_id)
        if stream is None:
            raise RuntimeError(f"Triton Realtime session not found: {session_id}")
        return stream

    async def _retire(self, session_id: str, stream: "_TritonResponseStream") -> None:
        await self._remove_and_close(session_id, stream)

    async def _remove_and_close(
        self, session_id: str, stream: "_TritonResponseStream"
    ) -> None:
        async with self._sessions_lock:
            if self._sessions.get(session_id) is stream:
                self._sessions.pop(session_id, None)
        await stream.close()


class _TritonResponseStream:
    def __init__(
        self,
        *,
        backend: TritonRealtimeBackend,
        session_id: str,
        start_payload: dict[str, Any],
        outbound_queue: asyncio.Queue,
        requested_audio: dict[str, Any],
    ) -> None:
        self._backend = backend
        self.session_id = session_id
        self._start_payload = start_payload
        self._outbound = outbound_queue
        self._audio = dict(requested_audio)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = None
        self._grpcclient = None
        self._outputs = None
        self._incoming: asyncio.Queue[_DecodedResult] = asyncio.Queue()
        self._start_future: asyncio.Future | None = None
        self._terminal = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._drain_task: asyncio.Task | None = None
        self._closed = False

    async def open(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._start_future = self._loop.create_future()
        _np, grpcclient = _require_triton()
        self._grpcclient = grpcclient
        self._client = grpcclient.InferenceServerClient(url=self._backend.endpoint)
        self._outputs = _outputs(grpcclient)
        self._client.start_stream(
            callback=self._callback, headers=self._backend.headers
        )
        self._drain_task = asyncio.create_task(self._drain())
        await self.submit(self._start_payload)
        try:
            await asyncio.wait_for(
                asyncio.shield(self._start_future), timeout=self._backend.start_timeout
            )
        except BaseException:
            if not self._start_future.done():
                self._start_future.cancel()
            raise

    async def submit(self, payload: dict[str, Any]) -> None:
        if self._closed or self._client is None or self._grpcclient is None:
            raise RuntimeError(f"Triton stream is closed: {self.session_id}")
        np, _grpcclient = _require_triton()
        request_input = self._grpcclient.InferInput("request", [1], "BYTES")
        request_input.set_data_from_numpy(
            np.array([json.dumps(payload, ensure_ascii=False)], dtype=object)
        )
        kwargs: dict[str, Any] = {
            "model_name": self._backend.model_name,
            "inputs": [request_input],
            "outputs": self._outputs,
        }
        if self._backend.model_version:
            kwargs["model_version"] = self._backend.model_version
        async with self._send_lock:
            self._client.async_stream_infer(**kwargs)

    async def wait_terminal(self) -> None:
        await self._terminal.wait()

    def _callback(self, result=None, error=None, **kwargs) -> None:
        if self._closed or self._loop is None:
            return
        infer_result = kwargs.get("result", result)
        callback_error = kwargs.get("error", error)
        try:
            decoded = _decode_result(infer_result, callback_error)
        except Exception as exc:
            decoded = _DecodedResult(error=str(exc), is_final=True)
        if not decoded.error and not decoded.event_type and not decoded.is_final:
            return
        self._loop.call_soon_threadsafe(self._incoming.put_nowait, decoded)

    async def _drain(self) -> None:
        terminal = False
        try:
            while not terminal:
                decoded = await self._incoming.get()
                if decoded.error:
                    await self._outbound.put(
                        _event_frame("error", self.session_id, message=decoded.error)
                    )
                    if self._start_future is not None and not self._start_future.done():
                        self._start_future.set_exception(RuntimeError(decoded.error))
                    terminal = True
                    continue

                payload = decoded.payload or {}
                event_type = decoded.event_type
                if event_type == "start":
                    audio = payload.get("audio_format") or payload.get("audio") or {}
                    if isinstance(audio, dict):
                        self._audio.update(
                            {
                                "encoding": str(
                                    audio.get("encoding", self._audio["encoding"])
                                ),
                                "sample_rate": int(
                                    audio.get("sample_rate", self._audio["sample_rate"])
                                ),
                                "channels": int(
                                    audio.get("channels", self._audio["channels"])
                                ),
                            }
                        )
                    if self._audio["encoding"] != "pcm_s16le":
                        message = (
                            "Triton Realtime backend requires pcm_s16le output, got "
                            f"{self._audio['encoding']!r}"
                        )
                        await self._outbound.put(
                            _event_frame("error", self.session_id, message=message)
                        )
                        if (
                            self._start_future is not None
                            and not self._start_future.done()
                        ):
                            self._start_future.set_exception(RuntimeError(message))
                        terminal = True
                        continue
                    await self._outbound.put(
                        _event_frame("start", self.session_id, payload=payload)
                    )
                    if self._start_future is not None and not self._start_future.done():
                        self._start_future.set_result(None)
                elif event_type == "audio":
                    if decoded.audio:
                        await self._outbound.put(
                            {
                                "type": "audio",
                                "audio": {
                                    "pcm_data": decoded.audio,
                                    **self._audio,
                                    "meta": {
                                        str(k): str(v)
                                        for k, v in dict(
                                            payload.get("meta") or {}
                                        ).items()
                                    },
                                },
                            }
                        )
                elif event_type:
                    await self._outbound.put(
                        _event_frame(event_type, self.session_id, payload=payload)
                    )

                if event_type in {"done", "error"}:
                    terminal = True
                elif decoded.is_final:
                    await self._outbound.put(_event_frame("done", self.session_id))
                    terminal = True
        except asyncio.CancelledError:
            raise
        finally:
            self._terminal.set()
            if self._start_future is not None and not self._start_future.done():
                self._start_future.set_exception(
                    RuntimeError("Triton stream ended before start acknowledgement")
                )
            asyncio.create_task(self._backend._retire(self.session_id, self))

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            task = self._drain_task
            if (
                task is not None
                and task is not asyncio.current_task()
                and not task.done()
            ):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            client = self._client
            self._client = None
            if client is not None:
                await asyncio.to_thread(_stop_client, client)


def _start_payload(
    session_id: str, start_request: "SessionStartRequest"
) -> dict[str, Any]:
    config = start_request.config
    payload: dict[str, Any] = {
        "action": "init",
        "session_id": session_id,
        "task_type": config.task_type,
        "language": config.language,
        "speaker": config.speaker or "",
        "instruct": config.instruct or "",
        "ref_audio": (
            base64.b64encode(config.ref_audio).decode("ascii")
            if config.ref_audio
            else ""
        ),
        "ref_text": config.ref_text or "",
        "x_vector_only": bool(config.x_vector_only),
        "input_mode": config.input_mode.value,
        "group_policy": config.group_policy.value,
        "audio": {
            "encoding": config.audio.encoding.value,
            "sample_rate": int(config.audio.sample_rate),
            "channels": int(config.audio.channels),
        },
        "output_policy": serialize_output_policy(start_request.output_policy),
        "timing": {
            "request_id": config.timing.request_id,
            "turn_id": config.timing.turn_id,
            "client_request_ts_ms": int(config.timing.client_request_ts_ms or 0),
            "client_text_ts_ms": int(config.timing.client_text_ts_ms or 0),
            "client_end_ts_ms": int(config.timing.client_end_ts_ms or 0),
            "extra": dict(config.timing.extra or {}),
        },
    }
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _outputs(grpcclient) -> list[Any]:
    return [
        grpcclient.InferRequestedOutput("audio_chunk"),
        grpcclient.InferRequestedOutput("event_type"),
        grpcclient.InferRequestedOutput("event_json"),
        grpcclient.InferRequestedOutput("is_final"),
    ]


def _decode_result(result: Any, error: Any) -> _DecodedResult:
    if error is not None:
        return _DecodedResult(error=str(error), is_final=True)
    if result is None:
        return _DecodedResult()
    event_type = _decode_text(_result_scalar(result, "event_type"))
    event_json = _decode_text(_result_scalar(result, "event_json"))
    payload: dict[str, Any] = {}
    if event_json:
        parsed = json.loads(event_json)
        if isinstance(parsed, dict):
            payload = parsed
    raw_audio = _result_scalar(result, "audio_chunk")
    audio = _decode_audio(raw_audio, payload)
    is_final = bool(_result_scalar(result, "is_final"))
    return _DecodedResult(
        event_type=event_type,
        payload=payload,
        audio=audio,
        is_final=is_final,
    )


def _result_scalar(result: Any, name: str) -> Any:
    value = result.as_numpy(name)
    if value is None or not getattr(value, "size", 0):
        return None
    return value.reshape(-1)[0]


def _decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_audio(value: Any, payload: dict[str, Any]) -> bytes:
    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        if str(payload.get("audio_chunk_encoding") or "").lower() == "base64":
            return base64.b64decode(value)
        return value.encode("utf-8")
    return bytes(value)


def _event_frame(
    event_type: str,
    session_id: str,
    *,
    payload: dict[str, Any] | None = None,
    message: str = "",
) -> dict[str, Any]:
    payload = payload or {}
    return {
        "type": "event",
        "event": {
            "type": event_type,
            "session_id": str(payload.get("session_id") or session_id),
            "segment_id": int(payload.get("segment_id", -1)),
            "text": str(payload.get("text") or ""),
            "message": str(payload.get("message") or message),
            "meta": {
                str(k): str(v) for k, v in dict(payload.get("meta") or {}).items()
            },
        },
    }


def _stop_client(client: Any) -> None:
    try:
        client.stop_stream()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


__all__ = ["TritonRealtimeBackend"]
