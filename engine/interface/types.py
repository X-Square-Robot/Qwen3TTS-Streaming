from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..core.types import AudioConfig, SessionConfig
from qwen3tts_protocol import (
    OutputPolicy,
    SessionEndRequest,
    StreamCancelRequest,
    StreamTextChunk,
    TimingContext,
    VADPolicy,
)


@dataclass
class SessionStartRequest:
    session_id: str
    config: SessionConfig
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    timing: TimingContext = field(default_factory=TimingContext)
    initial_text: str = ""


@dataclass
class StreamEvent:
    type: str
    session_id: str = ""
    segment_id: int = -1
    text: str = ""
    message: str = ""
    audio: Optional[AudioConfig] = None
    meta: dict[str, str] = field(default_factory=dict)


@dataclass
class AudioFrame:
    pcm_bytes: bytes
    audio: AudioConfig
    chunk_index: int = 0
    first_chunk: bool = False
    final_chunk: bool = False
    meta: dict[str, str] = field(default_factory=dict)
