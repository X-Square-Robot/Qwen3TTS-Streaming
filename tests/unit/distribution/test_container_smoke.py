from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from engine.distribution.container_smoke import run_container_smoke


@pytest.mark.parametrize("runtime_type", ["standalone", "triton"])
def test_container_distribution_smoke_contract(tmp_path: Path, runtime_type: str):
    pytest.importorskip("aiohttp")
    site = tmp_path / "demo"
    assets = site / "assets"
    assets.mkdir(parents=True)
    (site / "index.html").write_text(
        '<script type="module" src="./assets/app-deadbeef.js"></script>',
        encoding="utf-8",
    )
    (assets / "app-deadbeef.js").write_text("export {};", encoding="utf-8")
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "qwen3_tts_client-1.2.3-py3-none-any.whl").write_bytes(b"wheel")

    asyncio.run(
        run_container_smoke(
            runtime_type=runtime_type,
            site_dir=site,
            sdk_dir=sdk,
        )
    )
