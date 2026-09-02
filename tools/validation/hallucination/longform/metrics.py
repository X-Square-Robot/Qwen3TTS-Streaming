"""Compatibility facade for long-form transcript and rate metrics.

New code may import the focused owner modules directly. Existing callers retain
this stable surface so the CLI, reports, and tests do not depend on module layout.
"""

from .alignment_metrics import (
    AlignmentOpcode,
    character_error_metrics,
    levenshtein_opcodes,
    repetition_spans,
)
from .rate_statistics import (
    evaluate_comparison_gate,
    evaluate_invalid_sample_rules,
    evaluate_invalidity,
    paired_bootstrap_comparison,
    paired_hierarchical_bootstrap,
    wilson_interval,
)
from .text_normalization import canonicalize_numbers, normalize_transcript

__all__ = [
    "AlignmentOpcode",
    "canonicalize_numbers",
    "character_error_metrics",
    "evaluate_comparison_gate",
    "evaluate_invalid_sample_rules",
    "evaluate_invalidity",
    "levenshtein_opcodes",
    "normalize_transcript",
    "paired_bootstrap_comparison",
    "paired_hierarchical_bootstrap",
    "repetition_spans",
    "wilson_interval",
]
