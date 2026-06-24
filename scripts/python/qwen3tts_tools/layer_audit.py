"""ONNX layer prefix audit for mixed-precision build validation.

Scans an ONNX graph and classifies nodes by known submodel prefixes.
Used before building a TRT engine with per-submodule precision to detect
prefix drift that could cause mis-classification.

Public interface:

- :func:`audit_onnx_prefixes` — classify ONNX nodes by prefix
- :func:`classify_layer_name` — classify a single TRT/ONNX layer name
- :exc:`LayerAuditError` — raised when unclassified nodes exceed threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Prefix classification table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefixRule:
    """A single prefix-to-category mapping rule."""

    prefix: str
    category: str


# Canonical prefix rules for talker_code2wav_fused.onnx
DEFAULT_PREFIX_RULES: tuple[PrefixRule, ...] = (
    PrefixRule(prefix="/talker_fused/talker_unified/", category="backbone"),
    PrefixRule(prefix="/talker_fused/codec_sum/", category="backbone"),
    PrefixRule(prefix="/talker_fused/cp/", category="cp"),
    PrefixRule(prefix="/code2wav/", category="code2wav"),
)


# ---------------------------------------------------------------------------
#  Classification
# ---------------------------------------------------------------------------

def classify_layer_name(
    name: str,
    rules: tuple[PrefixRule, ...] = DEFAULT_PREFIX_RULES,
) -> str:
    """Classify a single TRT/ONNX layer name into a submodel category.

    Args:
        name: Layer name (e.g. ``/talker_fused/cp/Linear_0``).
        rules: Prefix rules to match against.

    Returns:
        Category string (``"backbone"``, ``"cp"``, ``"code2wav"``)
        or ``"unclassified"`` if no rule matches.
    """
    for rule in rules:
        if name.startswith(rule.prefix):
            return rule.category
    return "unclassified"


# ---------------------------------------------------------------------------
#  Audit result
# ---------------------------------------------------------------------------

@dataclass
class AuditReport:
    """Result of an ONNX prefix audit."""

    total_nodes: int = 0
    categories: dict[str, list[str]] = field(default_factory=dict)
    unclassified_ratio: float = 0.0
    rules_used: tuple[PrefixRule, ...] = DEFAULT_PREFIX_RULES

    @property
    def is_healthy(self) -> bool:
        """True if unclassified ratio is within acceptable bounds."""
        return self.unclassified_ratio <= 0.05


class LayerAuditError(Exception):
    """Raised when ONNX prefix audit finds too many unclassified nodes."""


# ---------------------------------------------------------------------------
#  ONNX audit
# ---------------------------------------------------------------------------

def audit_onnx_prefixes(
    onnx_path: Path,
    rules: tuple[PrefixRule, ...] = DEFAULT_PREFIX_RULES,
    threshold: float = 0.05,
) -> AuditReport:
    """Scan an ONNX graph and classify all nodes by prefix.

    Args:
        onnx_path: Path to the ONNX model file.
        rules: Prefix rules for classification.
        threshold: Maximum allowed ratio of unclassified nodes (0.0–1.0).

    Returns:
        AuditReport with classification statistics.

    Raises:
        LayerAuditError: If unclassified ratio exceeds threshold.
        FileNotFoundError: If onnx_path does not exist.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "onnx package required for prefix audit. "
            "Install: pip install onnx"
        )

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    model = onnx.load(str(onnx_path))
    graph = model.graph

    categories: dict[str, list[str]] = {
        rule.category: [] for rule in rules
    }
    categories["unclassified"] = []

    # Classify all nodes in the graph
    for node in graph.node:
        category = classify_layer_name(node.name, rules)
        categories.setdefault(category, []).append(node.name)

    total = sum(len(v) for v in categories.values())
    unclassified = len(categories.get("unclassified", []))
    ratio = unclassified / total if total > 0 else 0.0

    report = AuditReport(
        total_nodes=total,
        categories=categories,
        unclassified_ratio=ratio,
        rules_used=rules,
    )

    # Log summary
    logger.info("Layer classification summary for %s:", onnx_path.name)
    for cat, names in sorted(categories.items()):
        count = len(names)
        if cat == "unclassified" and count > 0:
            logger.warning(
                "  %s: %d nodes (%.1f%%) — check prefix table",
                cat, count, ratio * 100,
            )
            # Show up to 5 unclassified names as examples
            for name in names[:5]:
                logger.warning("    example: %s", name)
        else:
            logger.info("  %s: %d layers", cat, count)

    if ratio > threshold:
        raise LayerAuditError(
            f"Over {threshold * 100:.0f}% of nodes ({unclassified}/{total}) "
            f"cannot be classified — possible prefix drift. "
            f"Unclassified examples: {categories['unclassified'][:10]}"
        )

    return report


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Run ONNX prefix audit from command line."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Audit ONNX graph node prefixes for mixed-precision build validation.",
    )
    parser.add_argument(
        "onnx_path",
        type=Path,
        help="Path to talker_code2wav_fused.onnx",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Maximum allowed unclassified node ratio (default: 0.05)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )
    args = parser.parse_args()

    try:
        report = audit_onnx_prefixes(args.onnx_path, threshold=args.threshold)
    except LayerAuditError as e:
        logger.error("%s", e)
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    if args.json:
        import json
        output = {
            "total_nodes": report.total_nodes,
            "categories": {k: len(v) for k, v in report.categories.items()},
            "unclassified_ratio": report.unclassified_ratio,
            "is_healthy": report.is_healthy,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nAudit result: {'PASS' if report.is_healthy else 'FAIL'}")
        print(f"Total nodes: {report.total_nodes}")
        print(f"Unclassified ratio: {report.unclassified_ratio:.1%}")


if __name__ == "__main__":
    main()
