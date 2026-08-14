"""Shared data contracts for deterministic hallucination regression sweeps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_BODY_TEXT = "好啦,我已经把可乐从厨房的冰箱里递送到当前房间"
DEFAULT_LEADING_PREFIX = " "
DEFAULT_SAMPLE_RATE = 24000
SCREENING_LABEL = "ASR-supported suspect (requires blind human review)"


class ChunkPattern(str, Enum):
    WHOLE_LEADING = "whole-leading"
    SPLIT_LEADING = "split-leading"
    CLEAN = "clean"


class TrialStatus(str, Enum):
    OK = "ok"
    NO_AUDIO = "no_audio"
    ERROR = "error"


class AsrStatus(str, Enum):
    OK = "ok"
    DISABLED = "disabled"
    SKIPPED = "skipped"
    ERROR = "error"


class SuspectReason(str, Enum):
    DURATION = "duration_threshold"
    CER = "cer_threshold"
    INSERTIONS = "insertion_threshold"


@dataclass(frozen=True)
class TextPacket:
    text: str
    delay_after_s: float = 0.0


@dataclass(frozen=True)
class SuspectThresholds:
    duration_s: float = 30.0
    cer: float = 0.30
    insertions: int = 3


@dataclass
class SynthesisResult:
    status: TrialStatus
    samples: Any
    sample_rate: int
    duration_s: float
    ttft_ms: int | None
    total_ms: int
    chunks: int
    terminal_event: str | None
    eos_reason: str | None
    events: list[dict[str, Any]]
    error: str | None


@dataclass(frozen=True)
class SweepConfig:
    endpoint: str
    transport: str
    speaker: str
    input_mode: str
    group_policy: str
    pattern: ChunkPattern
    body_text: str
    leading_prefix: str
    split_delay_ms: float
    num_trials: int
    start: int
    sid_prefix: str
    arm: str
    timeout: float
    output_dir: Path | None
    metadata: dict[str, str]
    asr_url: str
    funasr_client_src: Path | None
    asr_language: str
    asr_chunk_ms: int
    asr_max_duration_s: float
    thresholds: SuspectThresholds
    confidence: float


def build_text_packets(
    body_text: str,
    *,
    leading_prefix: str,
    pattern: ChunkPattern | str,
    split_delay_ms: float,
) -> list[TextPacket]:
    """Build the exact ordered ``send_text`` calls for one arm."""

    selected = ChunkPattern(pattern)
    if not body_text:
        raise ValueError("body text must not be empty")
    if split_delay_ms < 0 or not math.isfinite(split_delay_ms):
        raise ValueError("split delay must be a finite non-negative number")
    if selected is ChunkPattern.CLEAN:
        return [TextPacket(body_text)]
    if not leading_prefix:
        raise ValueError("leading prefix must not be empty for a leading arm")
    if selected is ChunkPattern.WHOLE_LEADING:
        return [TextPacket(leading_prefix + body_text)]
    return [
        TextPacket(leading_prefix, delay_after_s=split_delay_ms / 1000.0),
        TextPacket(body_text),
    ]
