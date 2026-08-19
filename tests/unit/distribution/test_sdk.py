from __future__ import annotations

import base64
import hashlib

import pytest

from engine.distribution.sdk import SdkDistribution, mount_sdk_routes


WHEEL_NAME = "qwen3_tts_client-0.2.0-py3-none-any.whl"


def _write_wheel(tmp_path, content: bytes = b"wheel-bytes"):
    wheel = tmp_path / WHEEL_NAME
    wheel.write_bytes(content)
    return wheel


def test_discovery_ignores_unrelated_files_and_symlinks(tmp_path):
    wheel = _write_wheel(tmp_path)
    (tmp_path / "notes.txt").write_text("not an artifact", encoding="utf-8")
    (tmp_path / "qwen3_tts_client-linked.whl").symlink_to(wheel)

    distribution = SdkDistribution.discover(tmp_path)

    assert [artifact.filename for artifact in distribution.artifacts] == [WHEEL_NAME]
    assert distribution.artifact("../" + WHEEL_NAME) is None


@pytest.mark.asyncio
async def test_index_and_download_are_prefix_safe_and_verifiable(tmp_path):
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    content = b"matching-release-wheel"
    _write_wheel(tmp_path, content)
    app = web.Application()
    mount_sdk_routes(app, sdk_dir=tmp_path)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            redirect = await client.get("/sdk", allow_redirects=False)
            assert redirect.status == 308
            assert redirect.headers["Location"] == "./sdk/"

            index = await client.get("/sdk/?source=demo")
            assert index.status == 200
            body = await index.text()
            assert f'href="{WHEEL_NAME}"' in body
            assert f'href="/sdk/{WHEEL_NAME}"' not in body

            response = await client.get(f"/sdk/{WHEEL_NAME}?download=1")
            assert response.status == 200
            assert await response.read() == content
            sha256 = hashlib.sha256(content).hexdigest()
            digest = base64.b64encode(hashlib.sha256(content).digest()).decode("ascii")
            assert response.headers["Content-Disposition"] == (
                f'attachment; filename="{WHEEL_NAME}"'
            )
            assert response.headers["Content-Digest"] == f"sha-256=:{digest}:"
            assert response.headers["X-Checksum-SHA256"] == sha256
            assert response.headers["Content-Length"] == str(len(content))

            head = await client.head(f"/sdk/{WHEEL_NAME}")
            assert head.status == 200
            assert await head.read() == b""
            assert head.headers["X-Checksum-SHA256"] == sha256


@pytest.mark.asyncio
async def test_missing_or_untrusted_artifacts_return_404(tmp_path):
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    app = web.Application()
    mount_sdk_routes(app, sdk_dir=tmp_path / "missing")

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            assert (await client.get("/sdk/")).status == 404
            assert (await client.get(f"/sdk/{WHEEL_NAME}")).status == 404
            assert (await client.get("/sdk/../engine.yaml")).status == 404
