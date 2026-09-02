"""Read the immutable model release identity carried by a model package."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from pathlib import Path


MODEL_VERSION_FILENAME = "MODEL_VERSION"
MODEL_VERSION_MAX_CHARS = 128
_MODEL_RELEASE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*@[0-9]{8}")


class ModelVersionError(RuntimeError):
    """The model package has no usable immutable release identity."""


def validate_model_version(value: str, *, source: str = MODEL_VERSION_FILENAME) -> str:
    """Validate ``researcher@YYYYMMDD`` model release metadata."""

    version = validate_version_identity(value, source=source)
    if not _MODEL_RELEASE_PATTERN.fullmatch(version):
        raise ModelVersionError(
            f"{source} must use researcher@YYYYMMDD format"
        )
    try:
        dt.datetime.strptime(version.rsplit("@", 1)[1], "%Y%m%d")
    except ValueError as exc:
        raise ModelVersionError(
            f"{source} contains an invalid release date"
        ) from exc
    return version


def validate_version_identity(value: str, *, source: str) -> str:
    """Validate an opaque, single-line sidecar identity."""

    version = str(value or "").strip()
    if not version:
        raise ModelVersionError(f"{source} must not be empty")
    if len(version) > MODEL_VERSION_MAX_CHARS:
        raise ModelVersionError(
            f"{source} exceeds {MODEL_VERSION_MAX_CHARS} characters"
        )
    if any(character in version for character in ("\n", "\r", "\x00")):
        raise ModelVersionError(f"{source} must contain exactly one text line")
    return version


def load_model_version(model_package_dir: str | Path) -> str:
    """Load ``MODEL_VERSION`` from a resolved model-package root."""

    return load_read_only_version_file(
        model_package_dir,
        MODEL_VERSION_FILENAME,
        validator=validate_model_version,
    )


def load_read_only_version_file(
    package_dir: str | Path,
    filename: str,
    *,
    validator: Callable[..., str] = validate_version_identity,
) -> str:
    """Load and validate one immutable version sidecar from a package root."""

    package = Path(package_dir)
    version_path = package / filename
    try:
        mode = version_path.stat().st_mode
        raw = version_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ModelVersionError(
            f"model package is missing required {version_path}"
        ) from exc
    except OSError as exc:
        raise ModelVersionError(
            f"cannot read model version from {version_path}: {exc}"
        ) from exc
    if mode & 0o222:
        raise ModelVersionError(f"{version_path} must be read-only (mode 0444)")
    return validator(raw, source=str(version_path))


__all__ = [
    "MODEL_VERSION_FILENAME",
    "MODEL_VERSION_MAX_CHARS",
    "ModelVersionError",
    "load_model_version",
    "load_read_only_version_file",
    "validate_version_identity",
    "validate_model_version",
]
