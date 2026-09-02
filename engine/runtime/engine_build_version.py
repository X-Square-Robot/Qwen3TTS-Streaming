"""Read the immutable engine build identity carried by a model package."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from engine.runtime.model_version import (
    ModelVersionError,
    load_read_only_version_file,
    validate_version_identity,
)


ENGINE_BUILD_VERSION_FILENAME = "ENGINE_BUILD_VERSION"
_ENGINE_BUILD_VERSION_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*@[0-9]{8}_[0-9]+_[A-Za-z0-9._-]+_v[0-9]+"
)


def validate_engine_build_version(
    value: str,
    *,
    source: str = ENGINE_BUILD_VERSION_FILENAME,
) -> str:
    """Validate ``builder@YYYYMMDD_driver_device_export-protocol`` metadata."""

    version = validate_version_identity(value, source=source)
    if not _ENGINE_BUILD_VERSION_PATTERN.fullmatch(version):
        raise ModelVersionError(
            f"{source} must use builder@YYYYMMDD_driver_device_protocol format"
        )
    try:
        dt.datetime.strptime(version.split("@", 1)[1][:8], "%Y%m%d")
    except ValueError as exc:
        raise ModelVersionError(f"{source} contains an invalid build date") from exc
    return version


def load_engine_build_version(model_package_dir: str | Path) -> str:
    """Load read-only ``ENGINE_BUILD_VERSION`` from a model-package root."""

    return load_read_only_version_file(
        model_package_dir,
        ENGINE_BUILD_VERSION_FILENAME,
        validator=validate_engine_build_version,
    )


__all__ = [
    "ENGINE_BUILD_VERSION_FILENAME",
    "load_engine_build_version",
    "validate_engine_build_version",
]
