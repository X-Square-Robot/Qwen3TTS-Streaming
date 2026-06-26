from __future__ import annotations

from .audio_store import AudioStore
from qwen3tts_protocol.schemas import RunResult


def attach_audio_to_result(result: RunResult, audio_store: AudioStore) -> None:
    if not result.raw_audio:
        result.audio = {}
        result.warnings.append("No raw audio captured; playback disabled instead of attaching synthetic audio.")
        return
    sample_rate = int(result.audio_format.get("sample_rate") or 24000)
    audio = audio_store.save_pcm_f32_wav(
        result.raw_audio,
        sample_rate=sample_rate,
        name_hint=f"{result.backend}-{result.run_id}",
    )
    audio["scheduled_start_ms"] = scheduled_start_ms(result.metrics.to_dict(), result.backend)
    audio["source"] = result.source
    result.audio = audio


def scheduled_start_ms(metrics: dict, backend: str) -> float:
    return float(
        metrics.get("first_playable_ms")
        or metrics.get("client_ttfb_ms")
        or metrics.get("server_ttft_ms")
        or 0.0
    )
