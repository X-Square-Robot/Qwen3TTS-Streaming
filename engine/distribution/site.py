"""Instance-local Demo metadata and static site distribution."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .sdk import SdkDistribution


DEMO_CONFIG_SCHEMA_VERSION = "qwen.tts.demo-config.v1"
_DEFAULT_DEMO_DIR = "/app/demo"


def _security_headers() -> dict[str, str]:
    connect_sources = ["'self'", "ws:", "wss:"]
    lab_url = os.environ.get("DEMO_LAB_URL", "").strip()
    parsed = urlsplit(lab_url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        connect_sources.append(f"{parsed.scheme}://{parsed.netloc}")
    return {
        "Content-Security-Policy": (
            "default-src 'self'; script-src 'self' data:; style-src 'self'; "
            f"connect-src {' '.join(connect_sources)}; media-src 'self' blob:; "
            "worker-src 'self' blob: data:; img-src 'self' data:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        ),
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    }


def demo_enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return str(values.get("DEMO_ENABLED", "")).strip().lower() == "true"


def _lab_url() -> str:
    value = os.environ.get("DEMO_LAB_URL", "").strip()
    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value
    if not parsed.scheme and not parsed.netloc and value.startswith(("/", "./", "../")):
        return value
    return ""


def build_demo_config(
    *,
    runtime_type: str,
    capabilities: Mapping[str, Any],
    sdk_distribution: SdkDistribution,
) -> dict[str, Any]:
    """Build prefix-neutral instance metadata consumed by the browser Demo."""

    engine_version = str(capabilities.get("engine_version", "") or "")
    return {
        "schema_version": DEMO_CONFIG_SCHEMA_VERSION,
        "engine_version": engine_version,
        "runtime_type": runtime_type,
        "endpoints": {
            "capabilities_url": "../v1/capabilities",
            "openai_realtime_url": "../v1/realtime",
            "native_websocket_url": "../v1/ws",
        },
        "python_sdk": sdk_distribution.python_sdk_metadata(),
        "browser_sdk": {
            "available": bool(os.environ.get("BROWSER_SDK_VERSION", "").strip()),
            "package": "@xmultimodalinteraction/qwen3tts-browser",
            "version": os.environ.get("BROWSER_SDK_VERSION", "").strip(),
            "registry_url": os.environ.get("BROWSER_SDK_REGISTRY_URL", "").strip(),
            "tarball_url": os.environ.get("BROWSER_SDK_TARBALL_URL", "").strip(),
        },
        "docs": {
            "version": os.environ.get("DOCS_VERSION", engine_version).strip(),
            "route": "./#/docs/",
        },
        "lab": {"available": bool(_lab_url()), "url": _lab_url()},
    }


def mount_demo_config_route(
    app,
    *,
    runtime_type: str,
    capabilities_provider: Callable[[], Mapping[str, Any]],
    sdk_distribution: SdkDistribution,
    enabled: bool | None = None,
    site_dir: str | os.PathLike[str] | None = None,
) -> bool:
    """Mount the Demo and config only when explicitly enabled."""

    if enabled is None:
        enabled = demo_enabled()
    if not enabled:
        return False

    from aiohttp import web

    root = Path(site_dir or os.environ.get("DEMO_SITE_DIR", _DEFAULT_DEMO_DIR))
    resolved_root = root.resolve()

    def asset(relative: str) -> Path | None:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            return None
        if not candidate.is_file() or candidate.is_symlink():
            return None
        return candidate

    async def redirect(_request):
        return web.Response(status=308, headers={"Location": "./demo/"})

    async def index(_request):
        index_file = asset("index.html")
        if index_file is None:
            raise web.HTTPNotFound(text="Demo site artifact is not bundled\n")
        return web.FileResponse(
            index_file,
            headers={**_security_headers(), "Cache-Control": "no-store"},
        )

    async def config(_request):
        payload = build_demo_config(
            runtime_type=runtime_type,
            capabilities=capabilities_provider(),
            sdk_distribution=sdk_distribution,
        )
        return web.json_response(
            payload,
            headers={
                "Cache-Control": "no-store",
                **_security_headers(),
            },
        )

    async def static_file(request):
        relative = str(request.match_info.get("path", ""))
        target = asset(relative)
        if target is None:
            raise web.HTTPNotFound(text="Demo asset not found\n")
        immutable = relative.startswith("assets/") and "-" in target.stem
        cache = "public, max-age=31536000, immutable" if immutable else "no-cache"
        return web.FileResponse(
            target,
            headers={**_security_headers(), "Cache-Control": cache},
        )

    app.router.add_get("/demo", redirect)
    app.router.add_get("/demo/", index)
    app.router.add_get("/demo/config.json", config)
    app.router.add_get("/demo/{path:.*}", static_file)
    return True


__all__ = [
    "DEMO_CONFIG_SCHEMA_VERSION",
    "build_demo_config",
    "demo_enabled",
    "mount_demo_config_route",
]
