"""Read the immutable engine build identity carried by a model package."""

from __future__ import annotations

from pathlib import Path

from engine.runtime.model_version import load_read_only_version_file


ENGINE_BUILD_VERSION_FILENAME = "ENGINE_BUILD_VERSION"


def load_engine_build_version(model_package_dir: str | Path) -> str:
    """Load read-only ``ENGINE_BUILD_VERSION`` from a model-package root."""

    return load_read_only_version_file(
        model_package_dir,
        ENGINE_BUILD_VERSION_FILENAME,
    )


__all__ = [
    "ENGINE_BUILD_VERSION_FILENAME",
    "load_engine_build_version",
]
