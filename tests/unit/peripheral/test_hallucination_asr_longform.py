from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from tools.validation.hallucination import asr
from tools.validation.hallucination.models import AsrStatus


class FakeFunASRClient:
    events: list[dict[str, Any]] = []
    constructor_calls: list[tuple[str, dict[str, Any]]] = []
    transcription_calls: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, uri: str, **options: Any):
        type(self).constructor_calls.append((uri, options))

    async def __aenter__(self) -> FakeFunASRClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def transcribe_file(
        self, path: str, **options: Any
    ) -> AsyncIterator[dict[str, Any]]:
        type(self).transcription_calls.append((path, options))
        for event in type(self).events:
            yield event


@pytest.fixture(autouse=True)
def _reset_fake_client() -> None:
    FakeFunASRClient.events = []
    FakeFunASRClient.constructor_calls = []
    FakeFunASRClient.transcription_calls = []


async def _transcribe(
    tmp_path: Path,
    client_class: Any = FakeFunASRClient,
    **options: Any,
) -> dict[str, Any]:
    duration_s = options.pop("duration_s", 31.0)
    return await asr.transcribe_wav(
        client_class,
        tmp_path / "long.wav",
        uri="wss://asr.example.test/infer/instance/v1/ws",
        language="中文",
        chunk_ms=960,
        duration_s=duration_s,
        **options,
    )


async def test_long_wav_is_transcribed_with_strict_offline_configuration(
    tmp_path: Path,
) -> None:
    FakeFunASRClient.events = [
        {"type": "partial", "text": "临时文本"},
        {"type": "segment_final", "segment": {"text": "最终"}},
        {"type": "speech_activity", "phase": "started"},
        {"type": "segment_final", "segment": {"text": "文本"}},
        {"type": "stream_done", "audio_duration_ms": 31_000},
    ]

    result = await _transcribe(tmp_path)

    assert result["status"] == AsrStatus.OK.value
    assert result["transcript"] == "最终文本"
    assert [segment["text"] for segment in result["segments"]] == ["最终", "文本"]
    assert result["stream_done"]["audio_duration_ms"] == 31_000
    assert FakeFunASRClient.constructor_calls == [
        (
            "wss://asr.example.test/infer/instance/v1/ws",
            {
                "hotwords": [],
                "language": "中文",
                "partial_mode": "off",
                "hotword_config": None,
                "version_check": True,
                "vad_params": {"type": "fsmn"},
            },
        )
    ]
    assert FakeFunASRClient.transcription_calls == [
        (
            str(tmp_path / "long.wav"),
            {
                "chunk_ms": 960,
                "delivery_mode": "offline",
                "pacing": "none",
            },
        )
    ]


async def test_each_wav_uses_a_fresh_connection(tmp_path: Path) -> None:
    FakeFunASRClient.events = [{"type": "stream_done"}]

    first = await _transcribe(tmp_path)
    second = await _transcribe(tmp_path)

    assert first["status"] == second["status"] == AsrStatus.OK.value
    assert len(FakeFunASRClient.constructor_calls) == 2


async def test_legacy_constructor_remains_supported(tmp_path: Path) -> None:
    class LegacyClient:
        received: dict[str, Any]

        def __init__(
            self,
            uri: str,
            *,
            hotwords: list[str],
            language: str,
            partial_mode: str,
            hotword_config: None,
        ):
            type(self).received = {
                "uri": uri,
                "hotwords": hotwords,
                "language": language,
                "partial_mode": partial_mode,
                "hotword_config": hotword_config,
            }

        async def __aenter__(self) -> LegacyClient:
            return self

        async def __aexit__(self, *_exc: Any) -> None:
            return None

        async def transcribe_file(
            self, _path: str, **_options: Any
        ) -> AsyncIterator[dict[str, Any]]:
            yield {"type": "stream_done"}

    result = await _transcribe(tmp_path, LegacyClient)

    assert result["status"] == AsrStatus.OK.value
    assert "version_check" not in LegacyClient.received
    assert "vad_params" not in LegacyClient.received


@pytest.mark.parametrize(
    ("events", "error_fragment"),
    [
        ([], "without stream_done"),
        (
            [{"type": "stream_done"}, {"type": "stream_done"}],
            "duplicate stream_done",
        ),
        (
            [{"type": "error", "code": "decode_failed", "message": "bad"}],
            "decode_failed: bad",
        ),
        (
            [{"type": "stream_done"}, {"type": "partial", "text": "late"}],
            "after stream_done",
        ),
    ],
)
async def test_invalid_or_error_terminal_is_not_counted_clean(
    tmp_path: Path,
    events: list[dict[str, Any]],
    error_fragment: str,
) -> None:
    FakeFunASRClient.events = events

    result = await _transcribe(tmp_path)

    assert result["status"] == AsrStatus.ERROR.value
    assert error_fragment in result["error"]


async def test_disconnect_is_reported_as_error(tmp_path: Path) -> None:
    class DisconnectingClient(FakeFunASRClient):
        async def transcribe_file(
            self, _path: str, **_options: Any
        ) -> AsyncIterator[dict[str, Any]]:
            yield {"type": "segment_final", "segment": {"text": "partial result"}}
            raise ConnectionError("peer closed")

    result = await _transcribe(tmp_path, DisconnectingClient)

    assert result["status"] == AsrStatus.ERROR.value
    assert "ConnectionError: peer closed" in result["error"]


async def test_outer_deadline_turns_hung_stream_into_error(tmp_path: Path) -> None:
    class HangingClient(FakeFunASRClient):
        async def transcribe_file(
            self, _path: str, **_options: Any
        ) -> AsyncIterator[dict[str, Any]]:
            await asyncio.Event().wait()
            yield {"type": "stream_done"}

    result = await _transcribe(tmp_path, HangingClient, timeout_s=0.01)

    assert result["status"] == AsrStatus.ERROR.value
    assert "ASR transcription exceeded 0.01s" in result["error"]


async def test_default_deadline_is_derived_from_audio_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeFunASRClient.events = [{"type": "stream_done"}]
    wav_path = tmp_path / "long.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(1)
        wav_file.writeframes(b"\x00\x00" * 40)
    real_wait_for = asyncio.wait_for
    observed: list[float] = []

    async def capture_wait_for(awaitable: Any, *, timeout: float) -> Any:
        observed.append(timeout)
        return await real_wait_for(awaitable, timeout=1.0)

    monkeypatch.setattr(asr.asyncio, "wait_for", capture_wait_for)

    result = await asr.transcribe_wav(
        FakeFunASRClient,
        wav_path,
        uri="wss://asr.example.test/infer/instance/v1/ws",
        language="中文",
        chunk_ms=960,
    )

    assert result["status"] == AsrStatus.OK.value
    assert observed == [140.0]


def test_sdk_version_preflight_requires_the_pinned_wheel() -> None:
    assert (
        asr.validate_funasr_sdk_version("0.2.0a6")
        == asr.REQUIRED_FUNASR_VERSION
    )
    with pytest.raises(RuntimeError, match="SDK version mismatch"):
        asr.validate_funasr_sdk_version("0.2.0a5")
    with pytest.raises(RuntimeError, match="version is missing"):
        asr.validate_funasr_sdk_version("")


def test_service_preflight_uses_caller_fetched_capabilities() -> None:
    capabilities = {
        "protocol_version": "funasr-nano-ws-v1.1",
        "version": "0.2.0a6",
    }

    assert asr.validate_funasr_service_capabilities(capabilities) == "0.2.0a6"
    with pytest.raises(RuntimeError, match="service version mismatch"):
        asr.validate_funasr_service_capabilities({"version": "0.1.3a6"})
    with pytest.raises(RuntimeError, match=r"capabilities\.version is missing"):
        asr.validate_funasr_service_capabilities({})
