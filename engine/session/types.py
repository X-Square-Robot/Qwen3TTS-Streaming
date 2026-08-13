"""Closed, transport-neutral output types for one logical synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from ..core.types import AudioConfig


class TerminalStatus(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """The final format of an :class:`AudioOutput` payload."""

    encoding: str
    sample_rate: int
    channels: int

    @classmethod
    def from_config(cls, config: AudioConfig) -> "AudioFormat":
        return cls(
            encoding=config.encoding.value,
            sample_rate=int(config.sample_rate),
            channels=int(config.channels),
        )


@dataclass(frozen=True, slots=True)
class StartedOutput:
    session_id: str
    audio: AudioFormat
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AudioOutput:
    session_id: str
    pcm_bytes: bytes
    audio: AudioFormat
    output_sample_start: int
    output_sample_end: int
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EventOutput:
    session_id: str
    event_type: str
    segment_id: int = -1
    text: str = ""
    message: str = ""
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TerminalOutput:
    session_id: str
    status: TerminalStatus
    message: str = ""
    metrics: dict[str, object] = field(default_factory=dict)
    usage: dict[str, object] = field(default_factory=dict)


SessionOutput: TypeAlias = (
    StartedOutput | AudioOutput | EventOutput | TerminalOutput
)
