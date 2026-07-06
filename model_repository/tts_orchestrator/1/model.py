"""Thin Triton adapter over the new TTSEngine.

This backend keeps the legacy Triton JSON contract used by existing BLS tests:

- ``synthesize``: one-shot request with ``text``
- ``init``: start a streaming session
- ``append_text``: append transport text to an existing session
- ``text_complete``: mark streaming text input complete
- ``cancel``: cancel an existing session
- ``capabilities``: return capability metadata via Triton error payload

Unlike the legacy orchestrator, this file does not own batching / decode logic.
Its only responsibility is protocol conversion between Triton's decoupled Python
backend API and ``engine.server.TTSEngine``.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import functools
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np

import triton_python_backend_utils as pb_utils

_MODEL_VERSION_DIR = Path(__file__).resolve().parent
_IMPORT_ROOTS = [_MODEL_VERSION_DIR]
_repo_root_candidate = _MODEL_VERSION_DIR.parents[2]
if (_repo_root_candidate / "engine").is_dir():
    _IMPORT_ROOTS.append(_repo_root_candidate)
for _root in _IMPORT_ROOTS:
    _path = str(_root)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Fallback when engine/ is baked into the Triton deploy image (Dockerfile.triton → /opt/qwen3-tts/engine).
_BAKED_REPO = Path("/opt/qwen3-tts")
if (_BAKED_REPO / "engine").is_dir():
    _br = str(_BAKED_REPO)
    if _br not in sys.path:
        sys.path.append(_br)

from engine.config import (
    EngineConfig,
    apply_model_package_paths,
    load_model_manifest,
    resolve_model_package_paths,
)
from engine.core.timing import ServerTimingAccumulator
from engine.core.types import (
    AudioConfig,
    AudioEncoding,
    GroupPolicy,
    InputMode,
    SessionConfig,
)
from engine.interface import OutputPipeline, SessionStartRequest
from engine.runtime.fingerprint import (
    FingerprintCheckError,
    enforce_engine_fingerprint,
)
from engine.server import TTSEngine

logger = logging.getLogger("tts_orchestrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_ENGINE_SAMPLE_RATE = 24000
_FINAL_FLAGS = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL

_INPUT_MODE_BY_NAME = {
    "token": InputMode.TOKEN,
    "clause": InputMode.CLAUSE,
    "long_segment": InputMode.LONG_SEGMENT,
    "full_text": InputMode.FULL_TEXT,
}
_GROUP_POLICY_BY_NAME = {
    "none": GroupPolicy.NONE,
    "auto": GroupPolicy.AUTO,
}


def _param_string(params: dict[str, Any], key: str, default: str) -> str:
    raw = params.get(key, {})
    if isinstance(raw, dict):
        value = raw.get("string_value")
        if value is not None:
            return str(value)
    return default


def _parse_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _env_string(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def _normalize_request_task_type(requested: str, loaded_model_type: str) -> str:
    requested = (requested or "").strip()
    loaded_model_type = (loaded_model_type or "").strip()
    if not requested:
        return requested

    if requested in ("voice_clone", "voice_clone_xvec", "voice_clone_icl"):
        if loaded_model_type in ("base", "icl"):
            return loaded_model_type
        return requested

    if requested == "instruct" and loaded_model_type == "voice_design":
        return "voice_design"

    return requested


def _decode_ref_audio(raw: Any) -> bytes | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if not isinstance(raw, str):
        raise ValueError("Request field 'ref_audio' must be base64 text")
    try:
        return base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError(
            "Request field 'ref_audio' must be valid base64 audio data"
        ) from exc


def _parse_input_mode(raw: Any, *, default_mode: InputMode) -> InputMode:
    if isinstance(raw, InputMode):
        return raw
    if isinstance(raw, str):
        return _INPUT_MODE_BY_NAME.get(raw.strip().lower(), default_mode)
    if isinstance(raw, int):
        mapping = {
            1: InputMode.TOKEN,
            2: InputMode.CLAUSE,
            3: InputMode.LONG_SEGMENT,
            4: InputMode.FULL_TEXT,
        }
        return mapping.get(raw, default_mode)
    return default_mode


def _parse_group_policy(raw: Any) -> GroupPolicy:
    if isinstance(raw, GroupPolicy):
        return raw
    if isinstance(raw, str):
        return _GROUP_POLICY_BY_NAME.get(raw.strip().lower(), GroupPolicy.AUTO)
    if isinstance(raw, int):
        return {1: GroupPolicy.NONE, 2: GroupPolicy.AUTO}.get(raw, GroupPolicy.AUTO)
    return GroupPolicy.AUTO


def _parse_audio_config(raw: Any) -> AudioConfig:
    if not isinstance(raw, dict):
        return AudioConfig()
    sample_rate = int(raw.get("sample_rate") or _ENGINE_SAMPLE_RATE)
    channels = int(raw.get("channels") or 1)
    encoding_name = str(raw.get("encoding") or "pcm_f32").strip().lower()
    encoding = AudioEncoding.PCM_F32
    if encoding_name == "pcm_s16le":
        encoding = AudioEncoding.PCM_S16LE
    return AudioConfig(sample_rate=sample_rate, encoding=encoding, channels=channels)


def _validate_triton_audio_config(audio: AudioConfig) -> None:
    if audio.channels != 1:
        raise ValueError(f"Unsupported channel count: {audio.channels} (mono only)")
    if audio.sample_rate not in (16000, 24000):
        raise ValueError(
            f"Unsupported sample_rate: {audio.sample_rate} (expected 16000 or 24000)"
        )
    if audio.encoding not in (AudioEncoding.PCM_F32, AudioEncoding.PCM_S16LE):
        raise ValueError(f"Unsupported audio encoding: {audio.encoding}")


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if audio.size == 0 or src_sr == dst_sr:
        return audio
    duration = audio.shape[0] / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _convert_audio_chunk_bytes(pcm_bytes: bytes, audio_config: AudioConfig) -> bytes:
    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
    if audio_config.sample_rate != _ENGINE_SAMPLE_RATE:
        audio = _resample_linear(audio, _ENGINE_SAMPLE_RATE, audio_config.sample_rate)
    if audio_config.encoding == AudioEncoding.PCM_S16LE:
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767.0).astype(np.int16).tobytes()
    return audio.astype(np.float32, copy=False).tobytes()


def _build_legacy_capabilities(engine: TTSEngine) -> dict[str, Any]:
    cap = dict(engine.describe_capabilities())
    loaded = str(cap.get("loaded_model_type", "") or "")
    legacy_tasks: list[str]
    if loaded in ("base", "icl"):
        legacy_tasks = ["voice_clone"]
    elif loaded == "voice_design":
        legacy_tasks = ["voice_design"]
    elif loaded == "custom_voice":
        legacy_tasks = ["custom_voice"]
    else:
        legacy_tasks = list(cap.get("declared_supported_task_types", ()) or ())

    cap["tts_model_type"] = loaded
    cap["supported_task_types"] = legacy_tasks
    return cap


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})

        self.variant = _param_string(params, "model_variant", "unknown")
        self.device_id = int(
            _param_string(params, "device_id", os.environ.get("CUDA_DEVICE", "0"))
        )

        self._model_package_dir = Path(
            _param_string(params, "model_package_dir", str(_MODEL_VERSION_DIR))
        )
        package_paths = resolve_model_package_paths(str(self._model_package_dir))
        self._engine_dir = Path(package_paths.engine_dir)
        self._weights_dir = Path(package_paths.weights_dir)
        self._tokenizer_dir = Path(package_paths.tokenizer_dir)

        # Strict runtime fingerprint guard.  Same check as standalone
        # engine/server.py; failures here surface in Triton's model_repository
        # load status as a TritonModelException with the full report attached.
        try:
            enforce_engine_fingerprint(
                str(self._model_package_dir),
                device_index=self.device_id,
                runtime_dir=str(self._engine_dir),
            )
        except FingerprintCheckError as exc:
            raise pb_utils.TritonModelException(
                f"Engine fingerprint check failed for variant '{self.variant}':\n{exc}"
            ) from exc

        cfg = EngineConfig()
        apply_model_package_paths(cfg, package_paths)
        cfg.scheduler.max_batch_size = int(
            os.environ.get("MAX_BATCH_SLOTS")
            or _param_string(params, "max_batch_slots", "48")
        )
        cfg.server.max_sessions = int(
            _param_string(params, "max_sessions", os.environ.get("MAX_SESSIONS", "128"))
        )
        cfg.scheduler.max_seq_len = int(
            os.environ.get("ENGINE_MAX_DECODE_LEN")
            or _param_string(params, "engine_max_decode_len", "512")
        )
        cfg.scheduler.session_timeout_sec = float(
            _param_string(
                params,
                "request_timeout_sec",
                os.environ.get("REQUEST_TIMEOUT_SEC", "300"),
            )
        )
        cfg.prefix_cache.enabled = _parse_bool(
            _param_string(
                params,
                "enable_prefix_kv_cache",
                os.environ.get("ENABLE_PREFIX_KV_CACHE", "1"),
            ),
            True,
        )
        cfg.prefix_cache.max_entries = int(
            _param_string(
                params,
                "prefix_kv_cache_max_entries",
                os.environ.get("PREFIX_KV_CACHE_MAX_ENTRIES", "16"),
            )
        )
        # Speakers to prewarm the prefix KV cache for at startup (comma-separated).
        # Empty = off. Must be valid speaker ids or they fall back to a default.
        _prewarm = os.environ.get("ENGINE_PREWARM_SPEAKERS") or _param_string(
            params, "prewarm_speakers", ""
        )
        cfg.server.prewarm_speakers = [
            s.strip() for s in _prewarm.split(",") if s.strip()
        ]
        cfg.sampling.do_sample = _parse_bool(
            _param_string(
                params,
                "do_sample",
                _env_string("ENGINE_SAMPLING_DO_SAMPLE", "DO_SAMPLE", default="false"),
            ),
            False,
        )
        cfg.sampling.temperature = float(
            _param_string(
                params,
                "temperature",
                _env_string(
                    "ENGINE_SAMPLING_TEMPERATURE", "TEMPERATURE", default="0.9"
                ),
            )
        )
        cfg.sampling.repetition_penalty = float(
            _param_string(
                params,
                "repetition_penalty",
                _env_string(
                    "ENGINE_SAMPLING_REPETITION_PENALTY",
                    "REPETITION_PENALTY",
                    default="1.05",
                ),
            )
        )
        cfg.spliter.ema_ratio_initial = float(
            _param_string(
                params, "ratio_initial", os.environ.get("RATIO_INITIAL", "5.5")
            )
        )
        cfg.spliter.ema_alpha = float(
            _param_string(params, "ratio_alpha", os.environ.get("RATIO_ALPHA", "0.1"))
        )
        cfg.spliter.ema_overflow_alpha = float(
            _param_string(
                params,
                "ratio_overflow_alpha",
                os.environ.get("RATIO_OVERFLOW_ALPHA", "0.5"),
            )
        )
        cfg.spliter.ema_min_ratio = float(
            _param_string(params, "ratio_min", os.environ.get("RATIO_MIN", "2.0"))
        )
        cfg.spliter.ema_max_ratio = float(
            _param_string(params, "ratio_max", os.environ.get("RATIO_MAX", "10.0"))
        )
        cfg.spliter.l1_split_cap_ratio = float(
            _param_string(
                params,
                "l1_split_cap_ratio",
                os.environ.get(
                    "L1_SPLIT_CAP_RATIO", str(cfg.spliter.l1_split_cap_ratio)
                ),
            )
        )
        cfg.spliter.l2_split_cap_ratio = float(
            _param_string(
                params,
                "l2_split_cap_ratio",
                os.environ.get(
                    "L2_SPLIT_CAP_RATIO", str(cfg.spliter.l2_split_cap_ratio)
                ),
            )
        )
        cfg.spliter.l3_split_cap_ratio = float(
            _param_string(
                params,
                "l3_split_cap_ratio",
                os.environ.get(
                    "L3_SPLIT_CAP_RATIO", str(cfg.spliter.l3_split_cap_ratio)
                ),
            )
        )
        cfg.prefill.default_speaker = _param_string(
            params,
            "default_speaker",
            os.environ.get("DEFAULT_SPEAKER", cfg.prefill.default_speaker),
        )
        cfg.prefill.fallback_speaker = _param_string(
            params,
            "fallback_speaker",
            os.environ.get("FALLBACK_SPEAKER", cfg.prefill.fallback_speaker),
        )

        model_arch = load_model_manifest(
            str(self._engine_dir), cfg, tokenizer_dir=str(self._tokenizer_dir)
        )
        self._loaded_model_type = (model_arch.tts_model_type or "").strip() or (
            model_arch.supported_task_types[0]
            if model_arch.supported_task_types
            else "unknown"
        )

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            name="tts_orchestrator_loop",
            daemon=True,
        )
        self._loop_thread.start()

        # Audio-chunk sends (pb_utils.Tensor/InferenceResponse construction +
        # response_sender.send()) run on this pool instead of inline on the
        # single asyncio loop thread. With N concurrent sessions, on_audio has
        # no internal await, so inline sends serialize per-session responses
        # onto one thread — measured ~5-8ms of added tail latency across a
        # 32-session decode-step batch. InferenceResponseSender.send() is
        # documented thread-safe for exactly this decoupled-streaming pattern.
        self._send_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("TTS_BLS_SEND_WORKERS", "8")),
            thread_name_prefix="tts-send",
        )

        self._engine = TTSEngine(
            config=cfg,
            model_arch=model_arch,
            tokenizer_dir=str(self._tokenizer_dir),
            weights_dir=str(self._weights_dir),
            engine_dir=str(self._engine_dir),
            device_id=self.device_id,
            max_batch_size=cfg.scheduler.max_batch_size,
            max_sessions=cfg.server.max_sessions,
            max_seq_len=cfg.scheduler.max_seq_len,
        )
        self._sessions_lock = threading.Lock()
        self._active_sessions: set[str] = set()

        try:
            self._run_coro(self._engine.start())
        except Exception:
            self._shutdown_loop()
            raise

        logger.info(
            "Initialized TTSEngine-backed orchestrator: variant=%s model_package=%s engine_dir=%s tokenizer_dir=%s "
            "weights_dir=%s loaded_model_type=%s max_batch=%d max_sessions=%d max_seq_len=%d",
            self.variant,
            self._model_package_dir,
            self._engine_dir,
            self._tokenizer_dir,
            self._weights_dir,
            self._loaded_model_type,
            cfg.scheduler.max_batch_size,
            cfg.server.max_sessions,
            cfg.scheduler.max_seq_len,
        )

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                req = self._parse_request_json(request)
                action = (
                    str(req.get("action", "synthesize") or "synthesize").strip().lower()
                )

                if action == "capabilities":
                    self._handle_capabilities(response_sender)
                elif action in ("append_text", "append"):
                    self._handle_append_text(req, response_sender)
                elif action in ("text_complete", "done", "end"):
                    self._handle_text_complete(req, response_sender)
                elif action == "cancel":
                    self._handle_cancel(req, response_sender)
                elif action in ("init", "start", "synthesize"):
                    self._handle_start(
                        req, response_sender, streaming=(action in ("init", "start"))
                    )
                else:
                    raise ValueError(f"Unsupported action: {action}")
            except Exception as exc:
                logger.error("Request failed in execute: %s", exc)
                logger.error(traceback.format_exc())
                self._send_error(response_sender, str(exc))
        return None

    def finalize(self):
        try:
            if getattr(self, "_engine", None) is not None:
                self._run_coro(self._engine.stop())
        except Exception as exc:
            logger.warning("Failed to stop TTSEngine cleanly: %s", exc)
        finally:
            self._shutdown_loop()
            if getattr(self, "_send_pool", None) is not None:
                self._send_pool.shutdown(wait=True, cancel_futures=True)
        logger.info("Finalized TTSEngine-backed orchestrator")

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _shutdown_loop(self) -> None:
        if getattr(self, "_loop", None) is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if (
            getattr(self, "_loop_thread", None) is not None
            and self._loop_thread.is_alive()
        ):
            self._loop_thread.join(timeout=5)

    def _parse_request_json(self, request) -> dict[str, Any]:
        tensor = pb_utils.get_input_tensor_by_name(request, "request")
        if tensor is None:
            raise ValueError("Missing required input tensor 'request'")
        raw = tensor.as_numpy()[0]
        if isinstance(raw, bytes):
            payload = raw.decode("utf-8")
        else:
            payload = str(raw)
        req = json.loads(payload)
        if not isinstance(req, dict):
            raise ValueError("Request payload must be a JSON object")
        return req

    def _handle_capabilities(self, response_sender) -> None:
        caps_json = json.dumps(
            _build_legacy_capabilities(self._engine), ensure_ascii=False
        )
        response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("audio_chunk", np.array([b""], dtype=object)),
                pb_utils.Tensor("event_type", np.array(["capabilities"], dtype=object)),
                pb_utils.Tensor("event_json", np.array([caps_json], dtype=object)),
                pb_utils.Tensor("is_final", np.array([True], dtype=bool)),
            ],
            error=pb_utils.TritonError(f"CAPABILITIES:{caps_json}"),
        )
        self._safe_send_response(response_sender, response)
        self._safe_send_final(response_sender)

    def _handle_append_text(self, req: dict[str, Any], response_sender) -> None:
        session_id = str(req.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("append_text requires session_id")
        self._require_active_session(session_id)
        text = str(req.get("text") or "")
        if text:
            self._run_coro(self._engine.push_text_input(session_id, text))
        self._safe_send_final(response_sender)

    def _handle_text_complete(self, req: dict[str, Any], response_sender) -> None:
        session_id = str(req.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("text_complete requires session_id")
        self._require_active_session(session_id)
        self._run_coro(self._engine.mark_input_complete(session_id))
        self._safe_send_final(response_sender)

    def _handle_cancel(self, req: dict[str, Any], response_sender) -> None:
        session_id = str(req.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("cancel requires session_id")
        with self._sessions_lock:
            self._active_sessions.discard(session_id)
        self._run_coro(self._engine.cancel(session_id))
        self._safe_send_final(response_sender)

    def _send_event(
        self,
        response_sender,
        *,
        event_type: str,
        session_id: str = "",
        segment_idx: int = -1,
        text: str = "",
        message: str = "",
        audio_config: AudioConfig | None = None,
        meta: dict[str, str] | None = None,
        is_final: bool = False,
        audio_bytes: bytes = b"",
    ) -> None:
        event = {
            "type": event_type,
            "session_id": session_id,
            "segment_id": segment_idx,
            "text": text,
            "message": message,
            "meta": meta or {},
        }
        if audio_config is not None:
            event["audio_format"] = {
                "encoding": audio_config.encoding.value,
                "sample_rate": int(audio_config.sample_rate),
                "channels": int(audio_config.channels),
            }
        response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("audio_chunk", np.array([audio_bytes], dtype=object)),
                pb_utils.Tensor("event_type", np.array([event_type], dtype=object)),
                pb_utils.Tensor(
                    "event_json",
                    np.array([json.dumps(event, ensure_ascii=False)], dtype=object),
                ),
                pb_utils.Tensor("is_final", np.array([is_final], dtype=bool)),
            ]
        )
        self._safe_send_response(response_sender, response)
        if is_final:
            self._safe_send_final(response_sender)

    def _handle_start(
        self, req: dict[str, Any], response_sender, *, streaming: bool
    ) -> None:
        request_received = time.perf_counter()
        first_audio_sent = False
        session_id = str(req.get("session_id") or uuid.uuid4().hex)
        text = str(req.get("text") or "")
        if not streaming and not text.strip():
            raise ValueError("Request field 'text' is required and must be non-empty")

        config = self._session_config_from_request(req, streaming=streaming)

        # Server-side timing accumulator, mirroring engine/gateway/grpc_server.py
        # and engine/gateway/websocket_server.py's _create_session. Without this,
        # queue_wait/prefill/cache_hit/total_latency are silently never computed
        # for Triton sessions -- the engine-core code (dispatcher.py,
        # engine_loop.py) only populates the accumulator when one is present on
        # config.timing.extra, and Triton never attached one.
        timing_acc = ServerTimingAccumulator()
        timing_acc.request_received_epoch_ms = int(round(time.time() * 1000.0))
        timing_acc.session_created_epoch_ms = timing_acc.request_received_epoch_ms
        config.timing.extra["_server_timing_accumulator"] = timing_acc
        pipeline = OutputPipeline(
            SessionStartRequest(session_id=session_id, config=config),
            request_received_monotonic=time.monotonic(),
            timing_accumulator=timing_acc,
        )

        async def on_audio(_sid: str, data: bytes) -> None:
            nonlocal first_audio_sent
            meta = None
            if not first_audio_sent:
                first_audio_sent = True
                meta = {
                    "triton_adapter_ttft_ms": f"{(time.perf_counter() - request_received) * 1000.0:.3f}",
                    "first_audio_chunk": "true",
                }
            audio_bytes = _convert_audio_chunk_bytes(data, config.audio)
            # Off the shared asyncio loop thread and onto the send pool: with
            # many concurrent sessions, on_audio has no other await point, so
            # calling _send_event() inline here serializes every session's
            # response construction + response_sender.send() onto one thread.
            # run_in_executor() yields control back to the loop immediately,
            # so other sessions' on_audio calls can be dispatched while this
            # one's pb_utils/IPC work runs on a worker thread. Awaited (not
            # fire-and-forget), so ordering within this session is preserved
            # — the coroutine won't fetch the next chunk until this send
            # actually completes.
            await self._loop.run_in_executor(
                self._send_pool,
                functools.partial(
                    self._send_event,
                    response_sender,
                    event_type="audio",
                    session_id=_sid,
                    meta=meta,
                    audio_bytes=audio_bytes,
                    is_final=False,
                ),
            )

        async def on_event(_sid: str, event: dict) -> None:
            self._send_event(
                response_sender,
                event_type=str(event.get("type", "") or ""),
                session_id=_sid,
                segment_idx=int(event.get("segment_idx", -1)),
                text=str(event.get("text", "") or ""),
                message=str(event.get("message", "") or ""),
                meta={str(k): str(v) for k, v in (event.get("meta", {}) or {}).items()},
            )

        async def on_done(_sid: str, metrics: dict) -> None:
            with self._sessions_lock:
                self._active_sessions.discard(_sid)
            error = metrics.get("error") if isinstance(metrics, dict) else None
            # pipeline.done_meta() merges in the ServerTimingAccumulator data
            # (queue_wait/prefill/cache_hit/total_latency) and already excludes
            # the "error" key itself, matching build_done_event()'s behavior
            # in engine/gateway/grpc_server.py and websocket_server.py.
            done_meta = pipeline.done_meta(metrics)
            if error:
                self._send_event(
                    response_sender,
                    event_type="error",
                    session_id=_sid,
                    message=str(error),
                    meta=done_meta,
                    is_final=True,
                )
                return
            self._send_event(
                response_sender,
                event_type="done",
                session_id=_sid,
                meta=done_meta,
                is_final=True,
            )

        try:
            self._run_coro(
                self._engine.start_session(
                    session_id,
                    config=config,
                    on_audio=on_audio,
                    on_done=on_done,
                    on_event=on_event,
                )
            )
            with self._sessions_lock:
                self._active_sessions.add(session_id)
            self._send_event(
                response_sender,
                event_type="start",
                session_id=session_id,
                audio_config=config.audio,
                meta={
                    "input_mode": config.input_mode.value,
                    "group_policy": config.group_policy.value,
                    "task_type": config.task_type or "",
                },
            )
            if text:
                self._run_coro(self._engine.push_text_input(session_id, text))
            if not streaming:
                self._run_coro(self._engine.mark_input_complete(session_id))
        except Exception:
            with self._sessions_lock:
                self._active_sessions.discard(session_id)
            try:
                self._run_coro(self._engine.cancel(session_id))
            except Exception:
                pass
            raise

    def _session_config_from_request(
        self, req: dict[str, Any], *, streaming: bool
    ) -> SessionConfig:
        default_mode = InputMode.LONG_SEGMENT if streaming else InputMode.FULL_TEXT
        task_type = _normalize_request_task_type(
            str(req.get("task_type") or ""),
            self._loaded_model_type,
        )
        audio = _parse_audio_config(req.get("audio"))
        _validate_triton_audio_config(audio)
        config = SessionConfig(
            task_type=task_type,
            language=str(req.get("language") or "auto"),
            speaker=(str(req.get("speaker")).strip() or None)
            if req.get("speaker") is not None
            else None,
            instruct=(str(req.get("instruct")).strip() or None)
            if req.get("instruct") is not None
            else None,
            ref_audio=_decode_ref_audio(req.get("ref_audio")),
            ref_text=(str(req.get("ref_text")).strip() or None)
            if req.get("ref_text") is not None
            else None,
            x_vector_only=_parse_bool(req.get("x_vector_only"), False),
            input_mode=_parse_input_mode(
                req.get("input_mode"), default_mode=default_mode
            ),
            group_policy=_parse_group_policy(req.get("group_policy")),
            audio=audio,
        )
        if not streaming:
            config.input_mode = InputMode.FULL_TEXT
            if config.group_policy == GroupPolicy.NONE:
                config.group_policy = GroupPolicy.AUTO
        return config

    def _require_active_session(self, session_id: str) -> None:
        with self._sessions_lock:
            if session_id not in self._active_sessions:
                raise ValueError(f"session {session_id} not found")

    def _send_error(self, response_sender, error_msg: str) -> None:
        self._send_event(
            response_sender,
            event_type="error",
            message=error_msg,
            is_final=True,
        )

    @staticmethod
    def _safe_send_response(response_sender, response) -> None:
        try:
            response_sender.send(response)
        except Exception as exc:
            logger.warning("Response send failed: %s", exc)

    @staticmethod
    def _safe_send_final(response_sender) -> None:
        try:
            response_sender.send(flags=_FINAL_FLAGS)
        except Exception as exc:
            logger.warning("Final response send failed: %s", exc)
