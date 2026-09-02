#!/usr/bin/env python3
"""Apply the default RoPE precision policy to an exported TRT manifest.

The fused Qwen3-TTS graph keeps ``position_ids`` as integer I/O, but TensorRT
can still execute the position-to-angle path in BF16 when the engine's global
precision is BF16.  That loses integer positions above 256.  Phase B enables a
narrow FP32 pin for those RoPE layers by default; ``QWEN3_DISABLE_ROPE_FP32``
is an explicit escape hatch for numerical-debug builds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, MutableMapping


DISABLE_ENV = "QWEN3_DISABLE_ROPE_FP32"
ROPE_PRECISION = "fp32"


def env_truthy(value: str | None) -> bool:
    """Return whether an environment value explicitly requests disabling."""

    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def apply_rope_precision_policy(
    manifest: MutableMapping[str, Any], *, enabled: bool
) -> MutableMapping[str, Any]:
    """Set or remove the RoPE precision marker in a manifest in place.

    The top-level field is consumed by ``trt_fused_io_formats.py`` while
    ``engine_profile`` records the policy used to produce the artifact.  When
    disabled, both fields are removed so a previous fixed build cannot leave a
    stale FP32 marker behind during a debug rebuild.
    """

    profile = manifest.get("engine_profile")
    if enabled:
        manifest["rope_precision"] = ROPE_PRECISION
        if isinstance(profile, MutableMapping):
            profile["rope_precision"] = ROPE_PRECISION
    else:
        manifest.pop("rope_precision", None)
        if isinstance(profile, MutableMapping):
            profile.pop("rope_precision", None)
    return manifest


def update_manifest_file(path: Path, *, disable: bool) -> None:
    """Apply the policy and write a JSON manifest atomically enough for Phase B."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, MutableMapping):
        raise ValueError(f"manifest root must be an object: {path}")
    apply_rope_precision_policy(manifest, enabled=not disable)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    disable = env_truthy(os.environ.get(DISABLE_ENV))
    update_manifest_file(args.manifest, disable=disable)
    state = "disabled" if disable else f"enabled ({ROPE_PRECISION})"
    print(f"RoPE precision policy: {state} [{args.manifest}]")


if __name__ == "__main__":
    main()
