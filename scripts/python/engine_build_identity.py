#!/usr/bin/env python3
"""Generate the canonical TensorRT engine build identity.

The model release is owned by research and is intentionally not part of this
identity.  This module is kept under ``scripts/python`` because it is copied
into cross-host build bundles that do not contain the runtime package.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping


DEFAULT_EXPORT_PROTOCOL_VERSION = "v1"


class EngineBuildIdentityError(ValueError):
    """The build metadata cannot produce a valid engine build identity."""


def _component(value: object, *, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise EngineBuildIdentityError(f"{name} must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", result):
        raise EngineBuildIdentityError(
            f"{name} contains unsupported characters: {result!r}"
        )
    return result


def _build_date(value: object) -> str:
    raw = str(value or "").strip()
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EngineBuildIdentityError(
            f"built_at_utc must be an ISO timestamp, got {raw!r}"
        ) from exc
    return parsed.strftime("%Y%m%d")


def _driver_major(value: object) -> str:
    match = re.match(r"^(\d+)", str(value or "").strip())
    if not match:
        raise EngineBuildIdentityError(
            f"driver_version must start with a numeric major, got {value!r}"
        )
    return match.group(1)


def _device(value: object, gpu_sm: object = "") -> str:
    """Return a compact target-device token such as ``5090`` or ``A100``."""

    name = str(value or "").upper().strip()
    patterns = (
        r"\b(?:RTX|GTX)\s*([0-9]{3,4}(?:D|TI|SUPER)?)\b",
        r"\b([A-Z][0-9]{2,4}[A-Z0-9]*)\b",
        r"\b([0-9]{3,4})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1)

    sm = str(gpu_sm or "").strip().lower()
    if sm.startswith("sm_") and sm[3:].isdigit():
        return f"sm{sm[3:]}"
    return _component(value, name="gpu_name")


def _export_protocol(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw.isdigit():
        raw = f"v{raw}"
    if not re.fullmatch(r"v[0-9]+", raw):
        raise EngineBuildIdentityError(
            f"export_protocol_version must look like v1, got {value!r}"
        )
    return raw


def build_engine_build_version(
    *,
    builder: object,
    built_at_utc: object,
    driver_version: object,
    gpu_name: object = "",
    gpu_sm: object = "",
    export_protocol_version: object = DEFAULT_EXPORT_PROTOCOL_VERSION,
) -> str:
    """Build ``builder@YYYYMMDD_driver_device_export_protocol``."""

    result = (
        f"{_component(builder, name='engine_builder')}@"
        f"{_build_date(built_at_utc)}_"
        f"{_driver_major(driver_version)}_"
        f"{_device(gpu_name, gpu_sm)}_"
        f"{_export_protocol(export_protocol_version)}"
    )
    if len(result) > 128:
        raise EngineBuildIdentityError("engine build version exceeds 128 characters")
    return result


def build_engine_build_version_from_artifact(
    artifact: Mapping[str, object],
    *,
    builder: object | None = None,
    export_protocol_version: object | None = None,
) -> str:
    """Derive the identity from an artifact manifest's actual build fields."""

    return build_engine_build_version(
        builder=builder or artifact.get("engine_builder"),
        built_at_utc=artifact.get("built_at_utc"),
        driver_version=artifact.get("driver_version"),
        gpu_name=artifact.get("gpu_name"),
        gpu_sm=artifact.get("gpu_sm"),
        export_protocol_version=(
            export_protocol_version
            or artifact.get("export_protocol_version")
            or DEFAULT_EXPORT_PROTOCOL_VERSION
        ),
    )


__all__ = [
    "DEFAULT_EXPORT_PROTOCOL_VERSION",
    "EngineBuildIdentityError",
    "build_engine_build_version",
    "build_engine_build_version_from_artifact",
]
