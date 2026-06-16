from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .audio_store import AudioStore
from qwen3_tts_protocol.schemas import summarize_ttft
from qwen3_tts_protocol.triton_types import TtsRequest
from .triton_client import TritonUnavailable, measure_once


@dataclass
class ConcurrencyJob:
    job_id: str
    request: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    done: bool = False
    summary: dict[str, Any] | None = None

    async def publish(self, message: dict[str, Any]) -> None:
        self.history.append(message)
        for subscriber in list(self.subscribers):
            await subscriber.put(message)


class ConcurrencyJobManager:
    def __init__(
        self,
        *,
        enable_live: bool = False,
        audio_store: AudioStore | None = None,
        live_slot_limit: int = 64,
    ) -> None:
        self.enable_live = enable_live
        self.audio_store = audio_store
        self.live_slot_limit = max(1, int(live_slot_limit))
        self._jobs: dict[str, ConcurrencyJob] = {}

    def get(self, job_id: str) -> ConcurrencyJob | None:
        return self._jobs.get(job_id)

    async def create_job(self, request: dict[str, Any]) -> ConcurrencyJob:
        job_id = f"bench-{uuid.uuid4().hex[:10]}"
        job = ConcurrencyJob(job_id=job_id, request=dict(request))
        self._jobs[job_id] = job
        asyncio.create_task(self._run(job))
        return job

    async def subscribe(self, job: ConcurrencyJob) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        for item in job.history:
            await queue.put(item)
        job.subscribers.append(queue)
        return queue

    def unsubscribe(self, job: ConcurrencyJob, queue: asyncio.Queue) -> None:
        if queue in job.subscribers:
            job.subscribers.remove(queue)

    async def _run(self, job: ConcurrencyJob) -> None:
        concurrency = int(job.request.get("concurrency") or 128)
        concurrency = max(1, min(128, concurrency))
        live_requested = bool(job.request.get("live"))
        live = bool(self.enable_live and live_requested)
        started = time.perf_counter()
        await job.publish(
            {
                "type": "job_started",
                "job_id": job.job_id,
                "concurrency": concurrency,
                "source": "live_triton" if live else "simulated",
            }
        )
        if live:
            await self._run_live(job, concurrency, started)
        else:
            await self._run_simulated(job, concurrency, started)
        job.done = True

    async def _run_simulated(self, job: ConcurrencyJob, concurrency: int, started: float) -> None:
        rng = random.Random(job.job_id)
        ttfts: list[float] = []
        failed = 0
        for stream_idx in range(concurrency):
            stream_id = f"{stream_idx + 1:03d}"
            base = rng.gauss(178.0, 38.0)
            if stream_idx % 31 == 0:
                base += rng.uniform(40.0, 90.0)
            ttft_ms = max(92.0, min(390.0, base))
            ttfts.append(ttft_ms)
            await asyncio.sleep(0.006 if concurrency <= 32 else 0.002)
            await job.publish(
                {
                    "type": "lane_update",
                    "job_id": job.job_id,
                    "stream_id": stream_id,
                    "status": "playing",
                    "ttft_ms": ttft_ms,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                }
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        summary = {
            "type": "summary",
            "job_id": job.job_id,
            "source": "simulated",
            "concurrency": concurrency,
            "failed_streams": failed,
            "elapsed_ms": elapsed_ms,
            "throughput_audio_sec_per_sec": round(concurrency * 4.2 / max(elapsed_ms / 1000.0, 0.001), 2),
            **summarize_ttft(ttfts),
        }
        job.summary = summary
        await job.publish(summary)

    async def _run_live(self, job: ConcurrencyJob, concurrency: int, started: float) -> None:
        semaphore = asyncio.Semaphore(min(concurrency, 128))
        ttfts: list[tuple[int, float]] = []
        failed = 0
        slot_limit = min(self.live_slot_limit, concurrency)
        text = str(job.request.get("text") or "你好，这是 Qwen3 TTS Triton 并发压测。")
        speaker = str(job.request.get("speaker") or "Serena")
        language = str(job.request.get("language") or "auto")
        cache_mode = str(job.request.get("cache_mode") or "hit")

        async def run_one(stream_idx: int) -> None:
            nonlocal failed
            stream_id = f"{stream_idx + 1:03d}"
            async with semaphore:
                try:
                    result = await measure_once(
                        TtsRequest(text=text, speaker=speaker, language=language, cache_mode=cache_mode),
                        timeout_sec=float(job.request.get("timeout_sec") or 60.0),
                    )
                    ttft = first_available_ms(
                        result.metrics.server_ttft_ms,
                        result.metrics.triton_adapter_ttft_ms,
                        result.metrics.engine_internal_ttft_ms,
                        result.metrics.client_ttfb_ms,
                        result.metrics.first_playable_ms,
                    )
                    ttfts.append((stream_idx, ttft))
                    audio = None
                    if self.audio_store and result.raw_audio:
                        sample_rate = int(result.audio_format.get("sample_rate") or 24000)
                        audio = self.audio_store.save_pcm_f32_wav(
                            result.raw_audio,
                            sample_rate=sample_rate,
                            name_hint=f"concurrency-{job.job_id}-{stream_id}",
                        )
                        audio["source"] = "live_triton"
                    await job.publish(
                        {
                            "type": "lane_update",
                            "job_id": job.job_id,
                            "stream_id": stream_id,
                            "status": "playing",
                            "ttft_ms": ttft,
                            "queued_by_slot_limit": stream_idx >= slot_limit,
                            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                            "audio": audio,
                        }
                    )
                except (TritonUnavailable, TimeoutError, RuntimeError) as exc:
                    failed += 1
                    await job.publish(
                        {
                            "type": "lane_update",
                            "job_id": job.job_id,
                            "stream_id": stream_id,
                            "status": "error",
                            "error": str(exc),
                            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                        }
                    )

        await asyncio.gather(*(run_one(idx) for idx in range(concurrency)))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        all_ttfts = [value for _, value in ttfts]
        active_ttfts = [value for idx, value in ttfts if idx < slot_limit]
        queued_ttfts = [value for idx, value in ttfts if idx >= slot_limit]
        summary = {
            "type": "summary",
            "job_id": job.job_id,
            "source": "live_triton",
            "concurrency": concurrency,
            "active_slot_limit": slot_limit,
            "queued_streams": max(0, concurrency - slot_limit),
            "failed_streams": failed,
            "elapsed_ms": elapsed_ms,
            "active_avg_ttft_ms": summarize_ttft(active_ttfts)["avg_ttft_ms"],
            "queued_avg_ttft_ms": summarize_ttft(queued_ttfts)["avg_ttft_ms"],
            **summarize_ttft(all_ttfts),
        }
        job.summary = summary
        await job.publish(summary)


def first_available_ms(*values: float | None) -> float:
    for value in values:
        if value is not None:
            return float(value)
    return 0.0
