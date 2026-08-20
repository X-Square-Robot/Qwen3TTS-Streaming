from __future__ import annotations

import pytest

from engine.runtime.engine_build_version import (
    ENGINE_BUILD_VERSION_FILENAME,
    load_engine_build_version,
)
from engine.runtime.model_version import (
    MODEL_VERSION_FILENAME,
    ModelVersionError,
    load_model_version,
    validate_model_version,
)
from scripts.python.write_version_sidecar import write_version_sidecar


def test_load_model_version_reads_package_root(tmp_path):
    version_path = tmp_path / MODEL_VERSION_FILENAME
    version_path.write_text(
        "zehan@20260601\n",
        encoding="utf-8",
    )
    version_path.chmod(0o444)

    assert load_model_version(tmp_path) == "zehan@20260601"


def test_load_model_version_rejects_missing_file(tmp_path):
    with pytest.raises(ModelVersionError, match="missing required"):
        load_model_version(tmp_path)


def test_load_model_version_rejects_writable_file(tmp_path):
    (tmp_path / MODEL_VERSION_FILENAME).write_text(
        "zehan@20260601\n",
        encoding="utf-8",
    )

    with pytest.raises(ModelVersionError, match="read-only"):
        load_model_version(tmp_path)


def test_write_model_version_creates_and_replaces_read_only_sidecar(tmp_path):
    version_path = tmp_path / MODEL_VERSION_FILENAME

    assert write_version_sidecar(version_path, version="zehan@20260818") == (
        "zehan@20260818"
    )
    assert version_path.read_text(encoding="utf-8") == "zehan@20260818\n"
    assert version_path.stat().st_mode & 0o222 == 0

    write_version_sidecar(version_path, version="zehan@20260820")
    assert version_path.read_text(encoding="utf-8") == "zehan@20260820\n"
    assert version_path.stat().st_mode & 0o222 == 0


def test_load_engine_build_version_uses_package_sidecar(tmp_path):
    version_path = tmp_path / ENGINE_BUILD_VERSION_FILENAME
    write_version_sidecar(version_path, version="rime@20260820_580_5090_v2")

    assert load_engine_build_version(tmp_path) == "rime@20260820_580_5090_v2"


@pytest.mark.parametrize("value", ["", " \n", "first\nsecond", "bad\x00value"])
def test_validate_model_version_rejects_invalid_content(value):
    with pytest.raises(ModelVersionError):
        validate_model_version(value)
