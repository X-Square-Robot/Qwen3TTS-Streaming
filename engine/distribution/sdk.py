"""Prefix-safe HTTP distribution of the Python client wheel.

The component deliberately owns only release artifact discovery and HTTP
responses.  Engine and gateway bootstraps mount it without duplicating route
or security behavior.
"""

from __future__ import annotations

import base64
import hashlib
import html
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


_WHEEL_GLOB = "qwen3_tts_client-*.whl"
_SDK_DIR_ENV = "ENGINE_SDK_DIR"
_DEFAULT_SDK_DIR = "/app/sdk"


@dataclass(frozen=True, slots=True)
class WheelArtifact:
    """An immutable wheel discovered directly inside the SDK directory."""

    path: Path
    filename: str
    size: int
    sha256: str
    content_digest: str


class SdkDistribution:
    """Snapshot and serve trusted wheel artifacts from one directory."""

    def __init__(self, directory: Path, artifacts: tuple[WheelArtifact, ...]) -> None:
        self.directory = directory
        self.artifacts = artifacts
        self._artifacts_by_name = {
            artifact.filename: artifact for artifact in artifacts
        }

    @classmethod
    def discover(cls, directory: str | os.PathLike[str]) -> "SdkDistribution":
        sdk_dir = Path(directory)
        artifacts: list[WheelArtifact] = []
        if sdk_dir.is_dir():
            for candidate in sorted(sdk_dir.glob(_WHEEL_GLOB)):
                # Release artifacts must be regular files owned by this
                # directory.  Symlinks could escape it after discovery.
                if not candidate.is_file() or candidate.is_symlink():
                    continue
                digest = hashlib.sha256(candidate.read_bytes()).digest()
                artifacts.append(
                    WheelArtifact(
                        path=candidate,
                        filename=candidate.name,
                        size=candidate.stat().st_size,
                        sha256=digest.hex(),
                        content_digest=f"sha-256=:{base64.b64encode(digest).decode('ascii')}:",
                    )
                )
        return cls(sdk_dir, tuple(artifacts))

    def artifact(self, filename: str) -> WheelArtifact | None:
        """Return a previously discovered direct child, never an arbitrary path."""

        return self._artifacts_by_name.get(filename)

    def python_sdk_metadata(self) -> dict[str, object]:
        """Return structured config data without making clients scrape HTML."""

        if len(self.artifacts) != 1:
            reason = (
                "Python SDK wheel is not bundled"
                if not self.artifacts
                else "Python SDK directory contains multiple matching wheels"
            )
            return {
                "available": False,
                "project": "qwen3-tts-client",
                "reason": reason,
                "index_url": "../sdk/",
            }
        from packaging.utils import InvalidWheelFilename, parse_wheel_filename

        artifact = self.artifacts[0]
        try:
            _name, version, _build, _tags = parse_wheel_filename(artifact.filename)
        except InvalidWheelFilename:
            return {
                "available": False,
                "project": "qwen3-tts-client",
                "reason": "Bundled Python SDK has an invalid wheel filename",
                "index_url": "../sdk/",
            }
        return {
            "available": True,
            "project": "qwen3-tts-client",
            "version": str(version),
            "filename": artifact.filename,
            "sha256": artifact.sha256,
            "index_url": "../sdk/",
            "download_url": f"../sdk/{quote(artifact.filename)}",
        }

    def index_html(self) -> str:
        links = "\n".join(
            (
                f'<a href="{quote(artifact.filename)}" '
                f'data-sha256="{artifact.sha256}">'
                f"{html.escape(artifact.filename)}</a><br>"
            )
            for artifact in self.artifacts
        )
        return (
            "<!doctype html>\n"
            '<html><head><meta charset="utf-8">'
            "<title>Qwen3-TTS Python SDK</title></head>"
            f"<body><h1>Qwen3-TTS Python SDK</h1>\n{links}\n</body></html>\n"
        )


def mount_sdk_routes(
    app, *, sdk_dir: str | os.PathLike[str] | None = None
) -> SdkDistribution:
    """Mount the pip-compatible ``/sdk/`` index and wheel download routes.

    ``Location: ./sdk/`` and relative index links preserve any deployment
    prefix added by a reverse proxy.  GET routes use aiohttp's implicit HEAD
    support, so both methods share exactly the same headers and status rules.
    """

    from aiohttp import web

    distribution = SdkDistribution.discover(
        sdk_dir or os.environ.get(_SDK_DIR_ENV, _DEFAULT_SDK_DIR)
    )

    async def redirect_to_index(_request):
        return web.Response(status=308, headers={"Location": "./sdk/"})

    async def index(_request):
        if not distribution.artifacts:
            raise web.HTTPNotFound(
                text="Python SDK wheel is not bundled in this runtime\n"
            )
        return web.Response(
            text=distribution.index_html(),
            content_type="text/html",
            charset="utf-8",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def download(request):
        artifact = distribution.artifact(request.match_info["filename"])
        if artifact is None:
            raise web.HTTPNotFound(text="SDK wheel not found\n")
        return web.FileResponse(
            artifact.path,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "Content-Digest": artifact.content_digest,
                "Content-Disposition": f'attachment; filename="{artifact.filename}"',
                "ETag": f'"{artifact.sha256}"',
                "X-Checksum-SHA256": artifact.sha256,
                "X-Content-Type-Options": "nosniff",
            },
        )

    app.router.add_get("/sdk", redirect_to_index)
    app.router.add_get("/sdk/", index)
    app.router.add_get("/sdk/{filename}", download)
    return distribution


__all__ = ["SdkDistribution", "WheelArtifact", "mount_sdk_routes"]
