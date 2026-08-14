"""Reusable components for leading-prefix TTS regression sweeps."""

from tools.validation._bootstrap import bootstrap_tool_imports

REPO_ROOT = bootstrap_tool_imports()

from .asr import import_funasr_client, transcribe_wav
from .cli import build_parser, main, parse_metadata
from .metrics import (
    character_error_metrics,
    classify_trial,
    normalize_transcript,
    wilson_interval,
)
from .models import (
    DEFAULT_BODY_TEXT,
    DEFAULT_LEADING_PREFIX,
    DEFAULT_SAMPLE_RATE,
    SCREENING_LABEL,
    AsrStatus,
    ChunkPattern,
    SuspectReason,
    SuspectThresholds,
    SweepConfig,
    SynthesisResult,
    TextPacket,
    TrialStatus,
    build_text_packets,
)
from .report import (
    json_write,
    persist_trial,
    rewrite_trial_sidecar,
    sha256_file,
    summarize_records,
)
from .runner import default_output_dir, run_sweep
from .synthesis import synthesize_once

__all__ = [
    "DEFAULT_BODY_TEXT",
    "DEFAULT_LEADING_PREFIX",
    "DEFAULT_SAMPLE_RATE",
    "REPO_ROOT",
    "SCREENING_LABEL",
    "AsrStatus",
    "ChunkPattern",
    "SuspectReason",
    "SuspectThresholds",
    "SweepConfig",
    "SynthesisResult",
    "TextPacket",
    "TrialStatus",
    "build_parser",
    "build_text_packets",
    "character_error_metrics",
    "classify_trial",
    "default_output_dir",
    "import_funasr_client",
    "json_write",
    "main",
    "normalize_transcript",
    "parse_metadata",
    "persist_trial",
    "rewrite_trial_sidecar",
    "run_sweep",
    "sha256_file",
    "summarize_records",
    "synthesize_once",
    "transcribe_wav",
    "wilson_interval",
]
