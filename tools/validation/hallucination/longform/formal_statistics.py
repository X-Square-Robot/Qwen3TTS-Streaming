"""Predeclared statistical protocol for the formal 0818 truth gate.

Exploratory reports may deliberately use a smaller bootstrap.  Matched replay,
however, is admitted only when every comparison was produced with this exact
protocol.  Keeping the declaration outside the CLI prevents command-line
overrides from silently redefining the formal experiment after labels are known.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FORMAL_BOOTSTRAP_ITERATIONS = 10_000
FORMAL_BOOTSTRAP_CONFIDENCE = 0.95
FORMAL_BOOTSTRAP_SEED = 8_182_028
FORMAL_RESAMPLING_SCHEME = "two_way_crossed_seed_sentence_cartesian"


def require_formal_bootstrap_protocol(result: Mapping[str, Any]) -> None:
    """Reject a bootstrap result that does not match the predeclared protocol."""

    expected = {
        "iterations": FORMAL_BOOTSTRAP_ITERATIONS,
        "confidence": FORMAL_BOOTSTRAP_CONFIDENCE,
        "random_seed": FORMAL_BOOTSTRAP_SEED,
        "resampling_scheme": FORMAL_RESAMPLING_SCHEME,
    }
    actual = {field: result.get(field) for field in expected}
    if actual != expected:
        raise ValueError(
            "formal bootstrap protocol mismatch: "
            f"expected {expected!r}, received {actual!r}"
        )
    if (
        not isinstance(result.get("iterations"), int)
        or isinstance(result.get("iterations"), bool)
        or not isinstance(result.get("random_seed"), int)
        or isinstance(result.get("random_seed"), bool)
        or not isinstance(result.get("confidence"), float)
    ):
        raise ValueError("formal bootstrap protocol fields have invalid types")


__all__ = [
    "FORMAL_BOOTSTRAP_CONFIDENCE",
    "FORMAL_BOOTSTRAP_ITERATIONS",
    "FORMAL_BOOTSTRAP_SEED",
    "FORMAL_RESAMPLING_SCHEME",
    "require_formal_bootstrap_protocol",
]
