from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soxr

from ..core.types import AudioEncoding
from .protocol import PROTOCOL_VERSION, output_policy_json, timing_context_json
from .types import AudioFrame, SessionStartRequest, StreamEvent
from .vad import AttributedSamples, SampleProvenanceSpan, TTSVADProcessor

ENGINE_SAMPLE_RATE = 24000
TIMING_CONTRACT = "server_monotonic_v1"


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if audio.size == 0 or src_sr == dst_sr:
        return audio
    duration = audio.shape[0] / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


@dataclass(frozen=True)
class TextProgressCandidate:
    """A text marker waiting for final output-sample coordinates."""

    candidate_id: int
    event: dict[str, Any]
    segment_idx: int


@dataclass
class OutputBatch:
    """Atomic output unit: audio, anchors, and ordinary events."""

    audio: AudioFrame | None = None
    anchors: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class StreamingOutputProcessor:
    """Shared native→VAD→resample output and text-anchor processor.

    The frontend emits candidate markers with the engine audio that produced
    them.  This class is the only component allowed to assign output sample
    coordinates.  VAD provenance is carried through retained samples; a
    marker attached to discarded audio therefore never becomes an anchor.
    """

    def __init__(
        self,
        start_request: SessionStartRequest,
        *,
        vad_processor: TTSVADProcessor,
        native_sample_rate: int = ENGINE_SAMPLE_RATE,
        timing_accumulator: Any = None,
    ) -> None:
        self.pipeline = OutputPipeline(
            start_request,
            native_sample_rate=native_sample_rate,
            timing_accumulator=timing_accumulator,
        )
        self._vad = vad_processor
        self._native_sample_rate = int(native_sample_rate)
        self._next_candidate_id = 1
        self._next_anchor_seq = 1
        self._candidates: dict[int, TextProgressCandidate] = {}
        self._retained_candidate_ids: set[int] = set()
        self._emitted_candidate_ids: set[int] = set()
        self._pending_final: list[tuple[dict[str, Any], int | None]] = []
        self._segment_output_end: dict[int, int] = {}
        self._last_retained_segment = -1
        self._retained_native_cursor = 0
        self._candidate_targets: dict[int, int] = {}
        self._last_anchor_output_end = 0

    @property
    def output_sample_cursor(self) -> int:
        return self.pipeline.output_sample_cursor

    def process(self, chunk: Any) -> OutputBatch:
        """Process one native float32 chunk and return one atomic batch."""
        progress_event = getattr(chunk, "progress_event", None)
        segment_idx = int(getattr(chunk, "segment_idx", -1))
        pcm_bytes = getattr(chunk, "pcm_bytes", chunk)
        pcm_bytes = bytes(pcm_bytes or b"")
        raw = np.frombuffer(pcm_bytes, dtype=np.float32)
        if raw.size == 0:
            return OutputBatch()

        provenance = [
            SampleProvenanceSpan(
                0,
                int(raw.size),
                segment_idx,
                int(getattr(chunk, "source_frame_start", 0)),
                int(getattr(chunk, "source_frame_end", 0)),
                progress_event,
            )
        ]
        if progress_event:
            # Candidate identity is internal. A VAD discard must not consume a
            # public anchor sequence number.
            candidate_id = self._next_candidate_id
            self._next_candidate_id += 1
            candidate = TextProgressCandidate(
                candidate_id=candidate_id,
                event=_without_anchor_seq(progress_event),
                segment_idx=segment_idx,
            )
            self._candidates[candidate_id] = candidate
            provenance = [
                SampleProvenanceSpan(
                    span.sample_start,
                    span.sample_end,
                    span.segment_idx,
                    span.source_frame_start,
                    span.source_frame_end,
                    candidate,
                )
                for span in provenance
            ]

        if self._vad.config.enabled:
            int16 = np.clip(raw, -1.0, 1.0)
            int16 = (int16 * 32767.0).astype(np.int16)
            retained = self._vad.process_attributed_chunk(int16, provenance)
            if retained.samples.size == 0:
                return OutputBatch()
            native_bytes = (
                retained.samples.astype(np.float32, copy=False) / 32767.0
            ).tobytes()
        else:
            retained = AttributedSamples(raw, provenance)
            native_bytes = pcm_bytes
        return self._make_batch(retained, native_bytes)

    def process_event(self, event: dict[str, Any]) -> OutputBatch:
        """Accept a lifecycle event without inventing audio coordinates."""
        if not isinstance(event, dict):
            return OutputBatch(events=[event])
        if event.get("type") != "text_progress":
            return OutputBatch(events=[event])
        meta = dict(event.get("meta") or {})
        if _is_true(meta.get("alignment_final")):
            segment_idx = int(event.get("segment_idx", -1))
            target = self._segment_output_end.get(segment_idx)
            self._pending_final.append((event, target))
            # Native output usually reaches the segment boundary before this
            # lifecycle event. A streaming resampler may need one more chunk;
            # leave the marker pending until its target is observable.
            return OutputBatch(anchors=self._resolve_ready_anchors(self.output_sample_cursor))
        if "output_sample_end" not in meta:
            # Abort/cancel progress is diagnostic only.  Preserve the last
            # trusted cursor and do not turn an event-time estimate into a
            # synthetic output anchor.
            return OutputBatch(events=[event])
        # Legacy backends may provide a progress event without an attached
        # audio chunk.  Only use the currently confirmed output cursor; this is
        # intentionally not a sent-byte estimate for new engine paths.
        return OutputBatch(anchors=[self._anchor_event(event, self.output_sample_cursor)])

    def finish(self, *, emit_final: bool = True) -> list[OutputBatch]:
        """Flush VAD then resampler, and only then emit final text anchors."""
        batches: list[OutputBatch] = []
        retained = self._vad.flush_attributed()
        if retained.samples.size:
            native_bytes = (
                retained.samples.astype(np.float32, copy=False) / 32767.0
            ).tobytes()
            batches.append(self._make_batch(retained, native_bytes))

        flushed = self.pipeline.flush_resampler()
        if flushed is not None and flushed.pcm_bytes:
            if self._last_retained_segment >= 0:
                self._segment_output_end[
                    self._last_retained_segment
                ] = int(
                    flushed.meta.get("output_sample_end", str(self.output_sample_cursor))
                )
            batches.append(OutputBatch(audio=flushed))

        if emit_final:
            # Resolve delayed candidates and lifecycle finals in one
            # sample-ordered pass. This keeps a segment's final marker from
            # being emitted after a later segment's anchor.
            anchors = self._resolve_ready_anchors(
                self.output_sample_cursor,
                force_all=True,
            )
            if anchors:
                if batches and batches[-1].audio is not None:
                    # Keep delayed resampler output and its anchors in one
                    # logical batch for resumable registration.
                    batches[-1].anchors.extend(anchors)
                else:
                    batches.append(OutputBatch(anchors=anchors))
            self._candidates.clear()
            self._pending_final = []
        else:
            # Retry/abort/cancel keeps the last trusted position; it must not
            # be upgraded to a terminal 100% alignment merely because the
            # output pipeline is being flushed.
            self._pending_final = []
            self._candidates.clear()
        return batches

    def _make_batch(
        self, retained: AttributedSamples, native_bytes: bytes
    ) -> OutputBatch:
        native_base = self._retained_native_cursor
        self._retained_native_cursor += int(retained.samples.size)
        frame = self.pipeline.convert_audio_chunk(native_bytes)
        for span in retained.provenance:
            candidate = span.progress_event
            candidate_id = _candidate_id(candidate)
            if candidate_id is not None:
                self._retained_candidate_ids.add(candidate_id)
                self._candidate_targets[candidate_id] = round(
                    (native_base + span.sample_end)
                    * self.pipeline.start_request.config.audio.sample_rate
                    / self._native_sample_rate
                )
        if not frame.pcm_bytes:
            return OutputBatch()
        anchors: list[dict[str, Any]] = []
        output_end = int(frame.meta.get("output_sample_end", "0"))
        for span in retained.provenance:
            candidate = span.progress_event
            candidate_id = _candidate_id(candidate)
            if candidate_id is None:
                continue
            if candidate_id in self._emitted_candidate_ids:
                continue
            target = self._candidate_targets.get(candidate_id, output_end)
            self._segment_output_end[span.segment_idx] = target
            self._last_retained_segment = span.segment_idx
        anchors.extend(self._resolve_ready_anchors(output_end))
        return OutputBatch(audio=frame, anchors=anchors)

    def _resolve_ready_anchors(
        self,
        output_end: int,
        *,
        force_all: bool = False,
    ) -> list[dict[str, Any]]:
        """Resolve candidates and final markers in output-sample order."""

        ready: list[tuple[int, int, str, Any]] = []
        for candidate_id, target in self._candidate_targets.items():
            if candidate_id in self._emitted_candidate_ids:
                continue
            if target <= output_end or force_all:
                ready.append((min(target, output_end), 0, "candidate", candidate_id))
        for index, (event, target) in enumerate(self._pending_final):
            if target is None:
                if not force_all:
                    continue
                target = output_end
            if target <= output_end or force_all:
                ready.append((min(target, output_end), 1, "final", index))
        ready.sort(key=lambda item: (item[0], item[1], item[3]))

        anchors: list[dict[str, Any]] = []
        final_indices: set[int] = set()
        for target, _kind, marker_type, marker_id in ready:
            if marker_type == "candidate":
                candidate_id = int(marker_id)
                candidate = self._candidates.get(candidate_id)
                if candidate is None:
                    continue
                anchors.append(
                    self._anchor_event(
                        candidate.event,
                        target,
                        start=self._last_anchor_output_end,
                    )
                )
                self._emitted_candidate_ids.add(candidate_id)
                self._candidates.pop(candidate_id, None)
            else:
                index = int(marker_id)
                event, pending_target = self._pending_final[index]
                final_target = output_end if pending_target is None else pending_target
                anchors.append(
                    self._anchor_event(
                        event,
                        final_target,
                        start=self._last_anchor_output_end,
                    )
                )
                final_indices.add(index)
        if final_indices:
            self._pending_final = [
                item
                for index, item in enumerate(self._pending_final)
                if index not in final_indices
            ]
        return anchors

    def _anchor_event(
        self,
        event: dict[str, Any],
        end: int,
        *,
        start: int | None = None,
    ) -> dict[str, Any]:
        meta = dict(event.get("meta") or {})
        if start is None:
            start = int(meta.get("output_sample_start", end) or end)
        supplied_seq = _optional_int(meta.get("anchor_seq"))
        if supplied_seq is None:
            supplied_seq = self._allocate_anchor_seq()
        else:
            self._next_anchor_seq = max(self._next_anchor_seq, supplied_seq + 1)
        meta["anchor_seq"] = str(supplied_seq)
        meta["output_sample_start"] = str(max(0, int(start)))
        meta["output_sample_end"] = str(max(int(start), int(end)))
        meta["output_sample_rate"] = str(self.pipeline.start_request.config.audio.sample_rate)
        self._last_anchor_output_end = max(
            self._last_anchor_output_end, int(meta["output_sample_end"])
        )
        return {**event, "meta": meta}

    def _allocate_anchor_seq(self) -> int:
        seq = self._next_anchor_seq
        self._next_anchor_seq += 1
        return seq


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_true(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _without_anchor_seq(event: dict[str, Any]) -> dict[str, Any]:
    meta = dict(event.get("meta") or {})
    meta.pop("anchor_seq", None)
    return {**event, "meta": meta}


def _candidate_id(value: Any) -> int | None:
    if not isinstance(value, TextProgressCandidate):
        return None
    return value.candidate_id


class OutputPipeline:
    """Canonical output/timing pipeline shared by transport adapters."""

    def __init__(
        self,
        start_request: SessionStartRequest,
        *,
        native_sample_rate: int = ENGINE_SAMPLE_RATE,
        request_received_monotonic: float | None = None,
        request_received_epoch_ms: int | None = None,
        timing_accumulator: Any = None,
    ) -> None:
        self._start_request = start_request
        self._audio = start_request.config.audio
        self._native_sample_rate = int(native_sample_rate or ENGINE_SAMPLE_RATE)
        self._request_received_monotonic = (
            float(request_received_monotonic)
            if request_received_monotonic is not None
            else time.monotonic()
        )
        self._request_received_epoch_ms = (
            int(request_received_epoch_ms)
            if request_received_epoch_ms is not None
            else int(round(time.time() * 1000.0))
        )
        self._first_raw_audio_epoch_ms: int | None = None
        self._first_effective_audio_epoch_ms: int | None = None
        self._first_raw_audio_monotonic: float | None = None
        self._first_effective_audio_monotonic: float | None = None
        self._chunk_index = 0
        self._output_sample_cursor = 0
        self._resampler = (
            soxr.ResampleStream(
                self._native_sample_rate,
                self._audio.sample_rate,
                1,
                dtype="float32",
                quality="HQ",
            )
            if self._audio.sample_rate != self._native_sample_rate
            else None
        )
        self._resampler_flushed = False

        # Prefix trim / output gating observability (set when gating lands)
        self._prefix_trim_applied: bool = False
        self._prefix_trimmed_samples: int = 0
        self._prefix_trimmed_ms: float = 0.0

        # Cross-thread timing accumulator (optional, set by gateway)
        self._timing_accumulator = timing_accumulator

    @property
    def start_request(self) -> SessionStartRequest:
        return self._start_request

    @property
    def request_received_epoch_ms(self) -> int:
        return self._request_received_epoch_ms

    @property
    def chunk_count(self) -> int:
        return self._chunk_index

    @property
    def output_sample_cursor(self) -> int:
        """Number of samples retained on the final output PCM path."""
        return self._output_sample_cursor

    def record_prefix_trim(self, trimmed_samples: int, sample_rate: int) -> None:
        """Record prefix trim observability when output gating removes samples."""
        self._prefix_trim_applied = True
        self._prefix_trimmed_samples = trimmed_samples
        self._prefix_trimmed_ms = (trimmed_samples / sample_rate) * 1000.0
        if self._timing_accumulator is not None:
            self._timing_accumulator.prefix_trim_applied = True
            self._timing_accumulator.prefix_trimmed_ms = self._prefix_trimmed_ms

    def convert_audio_chunk(self, pcm_bytes: bytes) -> AudioFrame:
        audio = np.frombuffer(pcm_bytes, dtype=np.float32)
        if self._resampler is not None:
            audio = self._resampler.resample_chunk(audio, last=False)
        return self._encode_audio(audio)

    def flush_resampler(self) -> AudioFrame | None:
        """Flush streaming resampler delay into the final output timeline."""
        if self._resampler is None or self._resampler_flushed:
            return None
        self._resampler_flushed = True
        audio = self._resampler.resample_chunk(
            np.empty((0,), dtype=np.float32), last=True
        )
        if audio.size == 0:
            return None
        return self._encode_audio(audio)

    def _encode_audio(self, audio: np.ndarray) -> AudioFrame:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return AudioFrame(
                pcm_bytes=b"",
                audio=self._audio,
                chunk_index=self._chunk_index,
                first_chunk=False,
                final_chunk=False,
                meta={
                    "output_sample_start": str(self._output_sample_cursor),
                    "output_sample_end": str(self._output_sample_cursor),
                    "output_sample_rate": str(self._audio.sample_rate),
                },
            )
        if self._audio.encoding == AudioEncoding.PCM_S16LE:
            audio = np.clip(audio, -1.0, 1.0)
            payload = (audio * 32767.0).astype(np.int16).tobytes()
        else:
            payload = audio.astype(np.float32, copy=False).tobytes()

        chunk_index = self._chunk_index
        self._chunk_index += 1
        first_chunk = chunk_index == 0
        now_epoch_ms = int(round(time.time() * 1000.0))
        now_monotonic = time.monotonic()
        meta = {
            "chunk_index": str(chunk_index),
            "timing_contract": TIMING_CONTRACT,
            "output_sample_start": str(self._output_sample_cursor),
        }
        output_samples = audio.shape[0]
        meta["output_sample_end"] = str(self._output_sample_cursor + output_samples)
        meta["output_sample_rate"] = str(self._audio.sample_rate)
        self._output_sample_cursor += output_samples
        if first_chunk:
            # Raw audio = audio arriving from engine, before any output gating.
            # The gateway calls this conversion only after output gating, so
            # recover the true raw timestamp from the shared accumulator when
            # available instead of incorrectly equating raw with effective.
            raw_monotonic = getattr(
                self._timing_accumulator, "first_raw_audio_monotonic", None
            )
            if raw_monotonic is None:
                raw_monotonic = now_monotonic
            self._first_raw_audio_monotonic = raw_monotonic
            if self._timing_accumulator is not None:
                self._first_raw_audio_epoch_ms = (
                    self._timing_accumulator.monotonic_to_epoch_ms(raw_monotonic)
                )
            else:
                self._first_raw_audio_epoch_ms = now_epoch_ms
            self._first_effective_audio_epoch_ms = now_epoch_ms
            self._first_effective_audio_monotonic = now_monotonic
            if self._timing_accumulator is not None:
                self._timing_accumulator.first_effective_audio_monotonic = now_monotonic

            meta["first_audio_chunk"] = "true"

            # Semantic metric names (canonical)
            raw_ttft = (raw_monotonic - self._request_received_monotonic) * 1000.0
            effective_ttft = (now_monotonic - self._request_received_monotonic) * 1000.0
            meta["server_ttft_raw_ms"] = f"{raw_ttft:.3f}"
            meta["server_ttft_effective_ms"] = f"{effective_ttft:.3f}"
            meta["server_session_create_to_first_raw_audio_ms"] = f"{raw_ttft:.3f}"
            meta["server_first_raw_audio_epoch_ms"] = str(
                self._first_raw_audio_epoch_ms
            )
            meta["server_first_effective_audio_epoch_ms"] = str(now_epoch_ms)

            # Deprecated aliases (same value, matches old behavior)
            meta["server_ttft_ms"] = f"{raw_ttft:.3f}"
            meta["server_first_audio_epoch_ms"] = str(now_epoch_ms)

        return AudioFrame(
            pcm_bytes=payload,
            audio=self._audio,
            chunk_index=chunk_index,
            first_chunk=first_chunk,
            final_chunk=False,
            meta=meta,
        )

    def done_meta(self, metrics: dict[str, Any] | None = None) -> dict[str, str]:
        meta = {
            "timing_contract": TIMING_CONTRACT,
            "server_request_received_epoch_ms": str(self._request_received_epoch_ms),
            "server_done_epoch_ms": str(int(round(time.time() * 1000.0))),
            "server_total_latency_ms": f"{(time.monotonic() - self._request_received_monotonic) * 1000.0:.3f}",
            "audio_chunk_count": str(self._chunk_index),
            "final_output_sample": str(self._output_sample_cursor),
            "output_sample_rate": str(self._audio.sample_rate),
        }

        # Raw / effective audio timestamps
        if self._first_raw_audio_epoch_ms is not None:
            meta["server_first_raw_audio_epoch_ms"] = str(
                self._first_raw_audio_epoch_ms
            )
        if self._first_raw_audio_monotonic is not None:
            raw_ttft_ms = (
                self._first_raw_audio_monotonic - self._request_received_monotonic
            ) * 1000.0
            meta["server_ttft_raw_ms"] = f"{raw_ttft_ms:.3f}"
        if self._first_effective_audio_epoch_ms is not None:
            meta["server_first_effective_audio_epoch_ms"] = str(
                self._first_effective_audio_epoch_ms
            )
        if self._first_effective_audio_monotonic is not None:
            effective_ttft_ms = (
                self._first_effective_audio_monotonic - self._request_received_monotonic
            ) * 1000.0
            meta["server_ttft_effective_ms"] = f"{effective_ttft_ms:.3f}"

        # Derived raw-to-effective latency (gating delay)
        if (
            self._first_raw_audio_monotonic is not None
            and self._first_effective_audio_monotonic is not None
        ):
            gating_ms = (
                self._first_effective_audio_monotonic - self._first_raw_audio_monotonic
            ) * 1000.0
            meta["server_first_raw_to_first_effective_audio_ms"] = f"{gating_ms:.3f}"

        # Prefix trim / output gating
        if self._prefix_trim_applied:
            meta["server_prefix_trim_applied"] = "true"
            meta["server_prefix_trimmed_ms"] = f"{self._prefix_trimmed_ms:.3f}"
        else:
            meta["server_prefix_trim_applied"] = "false"

        # ServerTimingAccumulator data (cross-thread lifecycle timestamps)
        if self._timing_accumulator is not None:
            acc_meta = self._timing_accumulator.to_meta_dict()
            # Merge accumulator data (don't overwrite existing pipeline data)
            for key, value in acc_meta.items():
                if key not in meta:
                    meta[key] = value

        # Deprecated aliases for backward compatibility
        if self._first_effective_audio_epoch_ms is not None:
            meta["server_first_audio_epoch_ms"] = str(
                self._first_effective_audio_epoch_ms
            )

        timing = self._start_request.timing
        if timing.request_id:
            meta["request_id"] = timing.request_id
        if timing.turn_id:
            meta["turn_id"] = timing.turn_id
        if timing.client_request_ts_ms > 0:
            meta["client_request_ts_ms"] = str(timing.client_request_ts_ms)
        if timing.client_text_ts_ms > 0:
            meta["client_text_ts_ms"] = str(timing.client_text_ts_ms)
        if timing.client_end_ts_ms > 0:
            meta["client_end_ts_ms"] = str(timing.client_end_ts_ms)
        for key, value in dict(timing.extra or {}).items():
            if key.startswith("_"):
                continue  # skip internal keys (e.g. _server_timing_accumulator)
            meta[str(key)] = str(value)
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key == "error":
                    continue
                meta[str(key)] = str(value)
        return meta


def stamp_output_anchor(event: dict[str, Any], pipeline: OutputPipeline) -> dict[str, Any]:
    """Ensure a text anchor has coordinates on the final output timeline."""
    if not isinstance(event, dict):
        return event
    meta = dict(event.get("meta") or {})
    if (
        meta.get("anchor_seq")
        and (
            "output_sample_start" not in meta or "output_sample_end" not in meta
        )
    ):
        sample = int(pipeline.output_sample_cursor)
        meta.setdefault("output_sample_start", str(sample))
        meta.setdefault("output_sample_end", str(sample))
        meta["output_sample_rate"] = str(pipeline.start_request.config.audio.sample_rate)
        return {**event, "meta": meta}
    return event


def _base_meta(start_request: SessionStartRequest) -> dict[str, str]:
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "timing_contract": TIMING_CONTRACT,
        "input_mode": start_request.config.input_mode.value,
        "group_policy": start_request.config.group_policy.value,
        "task_type": start_request.config.task_type or "",
        "output_policy_json": output_policy_json(start_request.output_policy),
        "timing_context_json": timing_context_json(start_request.timing),
        "vad_enabled": "true" if start_request.output_policy.vad.enabled else "false",
        "vad_strategy": str(start_request.output_policy.vad.strategy or "disabled"),
    }
    if start_request.timing.request_id:
        meta["request_id"] = start_request.timing.request_id
    if start_request.timing.turn_id:
        meta["turn_id"] = start_request.timing.turn_id
    if start_request.timing.client_request_ts_ms > 0:
        meta["client_request_ts_ms"] = str(start_request.timing.client_request_ts_ms)
    return meta


def build_start_event(
    session_id: str, start_request: SessionStartRequest
) -> StreamEvent:
    return StreamEvent(
        type="start",
        session_id=session_id,
        audio=start_request.config.audio,
        meta=_base_meta(start_request),
    )


def build_forward_event(
    session_id: str,
    event: dict[str, Any],
    start_request: SessionStartRequest,
) -> StreamEvent:
    meta = {str(k): str(v) for k, v in (event.get("meta", {}) or {}).items()}
    if start_request.timing.request_id and "request_id" not in meta:
        meta["request_id"] = start_request.timing.request_id
    if start_request.timing.turn_id and "turn_id" not in meta:
        meta["turn_id"] = start_request.timing.turn_id
    return StreamEvent(
        type=str(event.get("type", "") or ""),
        session_id=session_id,
        segment_id=int(event.get("segment_idx", -1)),
        text=str(event.get("text", "") or ""),
        message=str(event.get("message", "") or ""),
        audio=start_request.config.audio if event.get("type") == "start" else None,
        meta=meta,
    )


def build_done_event(
    session_id: str,
    metrics: dict[str, Any] | None,
    pipeline: OutputPipeline,
) -> StreamEvent:
    error = metrics.get("error") if isinstance(metrics, dict) else None
    return StreamEvent(
        type="error" if error else "done",
        session_id=session_id,
        message=str(error or ""),
        meta=pipeline.done_meta(metrics),
    )
