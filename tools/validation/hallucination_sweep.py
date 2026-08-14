#!/usr/bin/env python3
"""Compatibility CLI for deterministic leading-prefix TTS regression sweeps.

Implementation lives in :mod:`tools.validation.hallucination`.  This module
keeps the original command and import surface stable.
"""

from __future__ import annotations

try:
    from tools.validation._bootstrap import bootstrap_tool_imports
except ModuleNotFoundError:  # Direct execution: python tools/validation/....py
    from _bootstrap import bootstrap_tool_imports

REPO_ROOT = bootstrap_tool_imports()

from tools.validation.hallucination import (  # noqa: F401
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
    build_parser,
    build_text_packets,
    character_error_metrics,
    classify_trial,
    main,
    normalize_transcript,
    persist_trial,
    run_sweep,
    sha256_file,
    summarize_records,
    synthesize_once,
    transcribe_wav,
    wilson_interval,
)
from tools.validation.hallucination.asr import (  # noqa: F401
    import_funasr_client as _import_funasr_client,
)
from tools.validation.hallucination.cli import (  # noqa: F401
    parse_metadata as _metadata,
)
from tools.validation.hallucination.report import (  # noqa: F401
    json_write as _json_write,
)
from tools.validation.hallucination.runner import (  # noqa: F401
    default_output_dir as _default_output_dir,
)
from tools.validation.hallucination.synthesis import (  # noqa: F401
    _audio_array,
    _event_record,
)

if __name__ == "__main__":
    raise SystemExit(main())
