"""Read immutable packaging provenance carried by a model package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping


PACKAGE_INFO_FILENAME = "PACKAGE_INFO.json"
PACKAGE_INFO_SCHEMA_VERSION = 1
PACKAGER_MAX_CHARS = 128


class PackageInfoError(RuntimeError):
    """The model package has no usable immutable packaging provenance."""


@dataclass(frozen=True)
class PackageInfo:
    """Validated package creator and calendar date."""

    packager: str
    packaged_on: str
    schema_version: int = PACKAGE_INFO_SCHEMA_VERSION

    def to_dict(self) -> dict[str, str | int]:
        """Return the canonical JSON representation."""

        return {
            "package_info_schema_version": self.schema_version,
            "packager": self.packager,
            "packaged_on": self.packaged_on,
        }


def validate_packager(value: object, *, source: str = "packager") -> str:
    """Validate a non-empty, single-line package creator identity."""

    if not isinstance(value, str):
        raise PackageInfoError(f"{source} must be a string")
    packager = value.strip()
    if not packager:
        raise PackageInfoError(f"{source} must not be empty")
    if len(packager) > PACKAGER_MAX_CHARS:
        raise PackageInfoError(f"{source} exceeds {PACKAGER_MAX_CHARS} characters")
    if any(character in packager for character in ("\n", "\r", "\x00")):
        raise PackageInfoError(f"{source} must contain exactly one text line")
    return packager


def validate_packaged_on(value: object, *, source: str = "packaged_on") -> str:
    """Validate a canonical ISO calendar date with day precision."""

    if not isinstance(value, str):
        raise PackageInfoError(f"{source} must be a string in YYYY-MM-DD format")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PackageInfoError(f"{source} must be a valid date in YYYY-MM-DD format") from exc
    if parsed.isoformat() != value:
        raise PackageInfoError(f"{source} must use canonical YYYY-MM-DD format")
    return value


def validate_package_info(
    payload: Mapping[str, Any], *, source: str = PACKAGE_INFO_FILENAME
) -> PackageInfo:
    """Validate package-info JSON data and return its typed representation."""

    schema_version = payload.get("package_info_schema_version")
    if type(schema_version) is not int or schema_version != PACKAGE_INFO_SCHEMA_VERSION:
        raise PackageInfoError(
            f"{source} package_info_schema_version must be "
            f"{PACKAGE_INFO_SCHEMA_VERSION}"
        )
    return PackageInfo(
        packager=validate_packager(payload.get("packager"), source=f"{source}.packager"),
        packaged_on=validate_packaged_on(
            payload.get("packaged_on"), source=f"{source}.packaged_on"
        ),
        schema_version=schema_version,
    )


def load_package_info(model_package_dir: str | Path) -> PackageInfo:
    """Load read-only ``PACKAGE_INFO.json`` from a model-package root."""

    package = Path(model_package_dir)
    info_path = package / PACKAGE_INFO_FILENAME
    try:
        mode = info_path.stat().st_mode
        raw = info_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PackageInfoError(f"model package is missing required {info_path}") from exc
    except OSError as exc:
        raise PackageInfoError(f"cannot read package info from {info_path}: {exc}") from exc
    if mode & 0o222:
        raise PackageInfoError(f"{info_path} must be read-only (mode 0444)")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PackageInfoError(f"{info_path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise PackageInfoError(f"{info_path} must contain a JSON object")
    return validate_package_info(payload, source=str(info_path))


__all__ = [
    "PACKAGE_INFO_FILENAME",
    "PACKAGE_INFO_SCHEMA_VERSION",
    "PACKAGER_MAX_CHARS",
    "PackageInfo",
    "PackageInfoError",
    "load_package_info",
    "validate_package_info",
    "validate_packaged_on",
    "validate_packager",
]
