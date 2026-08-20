from __future__ import annotations

import json

import pytest

from engine.runtime.package_info import (
    PACKAGE_INFO_FILENAME,
    PACKAGE_INFO_SCHEMA_VERSION,
    PackageInfoError,
    load_package_info,
    validate_package_info,
    validate_packaged_on,
    validate_packager,
)


def _write_package_info(tmp_path, payload, *, mode: int = 0o444):
    info_path = tmp_path / PACKAGE_INFO_FILENAME
    info_path.write_text(json.dumps(payload), encoding="utf-8")
    info_path.chmod(mode)
    return info_path


def test_load_package_info_reads_package_root(tmp_path):
    _write_package_info(
        tmp_path,
        {
            "package_info_schema_version": PACKAGE_INFO_SCHEMA_VERSION,
            "packager": "rime",
            "packaged_on": "2026-08-20",
        },
    )

    info = load_package_info(tmp_path)

    assert info.packager == "rime"
    assert info.packaged_on == "2026-08-20"
    assert info.to_dict() == {
        "package_info_schema_version": 1,
        "packager": "rime",
        "packaged_on": "2026-08-20",
    }


def test_load_package_info_rejects_missing_file(tmp_path):
    with pytest.raises(PackageInfoError, match="missing required"):
        load_package_info(tmp_path)


def test_load_package_info_rejects_writable_file(tmp_path):
    _write_package_info(
        tmp_path,
        {
            "package_info_schema_version": 1,
            "packager": "rime",
            "packaged_on": "2026-08-20",
        },
        mode=0o644,
    )

    with pytest.raises(PackageInfoError, match="read-only"):
        load_package_info(tmp_path)


@pytest.mark.parametrize(
    "value",
    [
        "2026-08-20T16:20:02+08:00",
        "2026-8-20",
        "2026-02-29",
        "20260820",
        "",
        None,
    ],
)
def test_validate_packaged_on_rejects_noncanonical_or_invalid_dates(value):
    with pytest.raises(PackageInfoError, match="YYYY-MM-DD"):
        validate_packaged_on(value)


@pytest.mark.parametrize("value", ["", "  ", "first\nsecond", "bad\x00value", None])
def test_validate_packager_rejects_invalid_values(value):
    with pytest.raises(PackageInfoError):
        validate_packager(value)


@pytest.mark.parametrize("schema_version", [None, 0, 2, True, "1"])
def test_validate_package_info_rejects_unknown_schema(schema_version):
    with pytest.raises(PackageInfoError, match="schema_version"):
        validate_package_info(
            {
                "package_info_schema_version": schema_version,
                "packager": "rime",
                "packaged_on": "2026-08-20",
            }
        )


@pytest.mark.parametrize("raw", ["not-json", "[]"])
def test_load_package_info_rejects_invalid_json_shape(tmp_path, raw):
    info_path = tmp_path / PACKAGE_INFO_FILENAME
    info_path.write_text(raw, encoding="utf-8")
    info_path.chmod(0o444)

    with pytest.raises(PackageInfoError):
        load_package_info(tmp_path)
