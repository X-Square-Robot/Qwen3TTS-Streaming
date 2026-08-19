from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.distribution.sdk import SdkDistribution
from engine.distribution.site import (
    build_demo_config,
    demo_enabled,
    mount_demo_config_route,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
WHEEL_NAME = "qwen3_tts_client-0.2.0-py3-none-any.whl"


def test_demo_is_enabled_by_default_and_can_be_disabled():
    for environ in (
        {},
        {"DEMO_ENABLED": ""},
        {"DEMO_ENABLED": "1"},
        {"DEMO_ENABLED": "true"},
        {"DEMO_ENABLED": "YES"},
        {"DEMO_ENABLED": "on"},
    ):
        assert demo_enabled(environ) is True

    for value in ("0", "false", "FALSE", "no", "off"):
        assert demo_enabled({"DEMO_ENABLED": value}) is False


def test_demo_rejects_invalid_environment_value():
    with pytest.raises(ValueError, match="DEMO_ENABLED"):
        demo_enabled({"DEMO_ENABLED": "sometimes"})


def test_demo_config_contains_prefix_neutral_sdk_metadata(tmp_path):
    (tmp_path / WHEEL_NAME).write_bytes(b"wheel")
    payload = build_demo_config(
        runtime_type="standalone",
        capabilities={"engine_version": "v0.2.0"},
        sdk_distribution=SdkDistribution.discover(tmp_path),
    )

    assert payload["endpoints"]["openai_realtime_url"] == "../v1/realtime"
    assert payload["python_sdk"]["filename"] == WHEEL_NAME
    assert payload["python_sdk"]["download_url"] == f"../sdk/{WHEEL_NAME}"
    assert payload["python_sdk"]["version"] == "0.2.0"

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (REPO_ROOT / "protocol/contracts/demo-config.schema.json").read_text()
    )
    jsonschema.validate(payload, schema)


def test_demo_config_disables_ambiguous_sdk_download(tmp_path):
    (tmp_path / WHEEL_NAME).write_bytes(b"one")
    (tmp_path / "qwen3_tts_client-0.2.1-py3-none-any.whl").write_bytes(b"two")
    payload = build_demo_config(
        runtime_type="triton",
        capabilities={},
        sdk_distribution=SdkDistribution.discover(tmp_path),
    )
    assert payload["python_sdk"]["available"] is False
    assert "multiple" in payload["python_sdk"]["reason"]


def test_demo_config_rejects_unsafe_lab_url(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_LAB_URL", "javascript:alert(1)")
    payload = build_demo_config(
        runtime_type="standalone",
        capabilities={},
        sdk_distribution=SdkDistribution.discover(tmp_path),
    )
    assert payload["lab"] == {"available": False, "url": ""}


@pytest.mark.asyncio
async def test_demo_routes_are_prefix_safe_cached_and_hardened(tmp_path, monkeypatch):
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    site = tmp_path / "site"
    (site / "assets").mkdir(parents=True)
    (site / "index.html").write_text("<h1>demo</h1>", encoding="utf-8")
    (site / "assets/app-12345678.js").write_text("export {};", encoding="utf-8")
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / WHEEL_NAME).write_bytes(b"wheel")
    monkeypatch.setenv("DEMO_LAB_URL", "https://lab.example.test/tools")
    app = web.Application()
    mount_demo_config_route(
        app,
        runtime_type="standalone",
        capabilities_provider=lambda: {"engine_version": "v0.2.0"},
        sdk_distribution=SdkDistribution.discover(sdk),
        enabled=True,
        site_dir=site,
    )

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            redirect = await client.get("/demo", allow_redirects=False)
            assert redirect.status == 308
            assert redirect.headers["Location"] == "./demo/"
            index = await client.get("/demo/")
            assert index.status == 200
            assert index.headers["Cache-Control"] == "no-store"
            assert index.headers["X-Frame-Options"] == "DENY"
            assert index.headers["Permissions-Policy"].startswith("microphone=(self)")
            assert (
                "https://lab.example.test" in index.headers["Content-Security-Policy"]
            )
            asset = await client.get("/demo/assets/app-12345678.js")
            assert asset.status == 200
            assert "immutable" in asset.headers["Cache-Control"]
            assert (await client.get("/demo/../engine.yaml")).status == 404
