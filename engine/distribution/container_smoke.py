"""Self-contained distribution smoke used inside both release runtime images."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
from pathlib import Path

from .sdk import SdkDistribution, mount_sdk_routes
from .site import mount_demo_config_route


def build_smoke_app(*, runtime_type: str, enabled: bool, site_dir: Path, sdk_dir: Path):
    from aiohttp import web

    app = web.Application()
    sdk = mount_sdk_routes(app, sdk_dir=sdk_dir)
    capabilities = {
        "schema_version": "qwen.tts.capabilities.v1",
        "engine_version": os.environ.get("ENGINE_VERSION", "smoke"),
        "runtime": {"type": runtime_type, "backend": "container-smoke"},
        "model": "container-smoke",
        "tasks": ["custom_voice"],
        "task_status": [
            {"task": "custom_voice", "available": True, "stability": "stable"}
        ],
        "audio_formats": [
            {"encoding": "pcm_s16le", "sample_rate": 24000, "channels": 1}
        ],
        "limits": {
            "max_input_tokens": 128,
            "max_realtime_message_bytes": 8 * 1024 * 1024,
        },
        "output_policy": {
            "features": ["guarded_delivery"],
            "vad_strategies": ["disabled"],
        },
        "reference": {
            "available": False,
            "max_duration_sec": 0,
            "max_bytes": 4 * 1024 * 1024,
            "mime_types": ["audio/wav", "audio/x-wav"],
            "reason": "smoke",
        },
        "protocols": {
            "openai_realtime": {
                "path": "/v1/realtime",
                "base": "openai-realtime-v1",
                "extension_protocol": "qwen-realtime-v1",
                "supported_extensions": [
                    "qwen.input_text_buffer.v1",
                    "qwen.text_progress.v1",
                    "qwen.playback_ack.v1",
                    "qwen.response_resume.v1",
                ],
                "features": ["playback_ack"],
                "audio_formats": ["pcm_s16le"],
            }
        },
    }

    async def get_capabilities(_request):
        return web.json_response(capabilities)

    async def realtime(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "session": {"id": "smoke"}})
        async for message in ws:
            if message.type.name != "TEXT":
                continue
            event = json.loads(message.data)
            if event.get("type") == "session.update":
                await ws.send_json(
                    {"type": "session.updated", "session": {"id": "smoke"}}
                )
            elif event.get("type") == "conversation.item.create":
                await ws.send_json(
                    {"type": "conversation.item.added", "item": event.get("item")}
                )
                await ws.send_json(
                    {"type": "conversation.item.done", "item": event.get("item")}
                )
            elif event.get("type") == "response.create":
                await ws.send_json(
                    {"type": "response.created", "response": {"id": "resp_smoke"}}
                )
                await ws.send_json(
                    {
                        "type": "response.output_audio.delta",
                        "response_id": "resp_smoke",
                        "delta": base64.b64encode(b"\x00\x00\x01\x00").decode(),
                        "qwen_delivery_seq": 1,
                        "qwen_output_sample_start": 0,
                        "qwen_output_sample_end": 2,
                    }
                )
                await ws.send_json(
                    {
                        "type": "response.done",
                        "qwen_delivery_seq": 2,
                        "response": {
                            "id": "resp_smoke",
                            "status": "completed",
                            "usage": {"audio_tokens": 1},
                        },
                    }
                )
        return ws

    app.router.add_get("/v1/capabilities", get_capabilities)
    app.router.add_get("/v1/realtime", realtime)
    mount_demo_config_route(
        app,
        runtime_type=runtime_type,
        capabilities_provider=lambda: capabilities,
        sdk_distribution=sdk,
        enabled=enabled,
        site_dir=site_dir,
    )
    return app


async def run_container_smoke(
    *, runtime_type: str, site_dir: Path, sdk_dir: Path
) -> None:
    from aiohttp.test_utils import TestClient, TestServer

    distribution = SdkDistribution.discover(sdk_dir)
    if len(distribution.artifacts) != 1:
        raise RuntimeError("release image must contain exactly one Python SDK wheel")
    wheel = distribution.artifacts[0]

    disabled = TestServer(
        build_smoke_app(
            runtime_type=runtime_type,
            enabled=False,
            site_dir=site_dir,
            sdk_dir=sdk_dir,
        )
    )
    async with TestClient(disabled) as client:
        assert (await client.get("/demo")).status == 404
        assert (await client.get("/demo/config.json")).status == 404
        assert (await client.get("/sdk/")).status == 200

    enabled = TestServer(
        build_smoke_app(
            runtime_type=runtime_type,
            enabled=True,
            site_dir=site_dir,
            sdk_dir=sdk_dir,
        )
    )
    async with TestClient(enabled) as client:
        redirect = await client.get("/demo", allow_redirects=False)
        assert redirect.status == 308 and redirect.headers["Location"] == "./demo/"
        index = await client.get("/demo/")
        assert index.status == 200
        html = await index.text()
        asset_match = re.search(r'(?:src|href)="\./(assets/[^"]+)"', html)
        if asset_match:
            assert (await client.get(f"/demo/{asset_match.group(1)}")).status == 200
        config = await (await client.get("/demo/config.json")).json()
        assert config["runtime_type"] == runtime_type
        assert config["python_sdk"]["filename"] == wheel.filename
        download = await client.get(f"/sdk/{wheel.filename}?smoke=1")
        payload = await download.read()
        assert download.status == 200
        assert hashlib.sha256(payload).hexdigest() == wheel.sha256
        ws = await client.ws_connect("/v1/realtime")
        assert (await ws.receive_json())["type"] == "session.created"
        await ws.send_json({"type": "session.update", "session": {}})
        assert (await ws.receive_json())["type"] == "session.updated"
        await ws.send_json(
            {"type": "conversation.item.create", "item": {"type": "message"}}
        )
        assert (await ws.receive_json())["type"] == "conversation.item.added"
        assert (await ws.receive_json())["type"] == "conversation.item.done"
        await ws.send_json({"type": "response.create"})
        assert (await ws.receive_json())["type"] == "response.created"
        audio = await ws.receive_json()
        assert audio["type"] == "response.output_audio.delta"
        assert base64.b64decode(audio["delta"]) == b"\x00\x00\x01\x00"
        assert (await ws.receive_json())["type"] == "response.done"
        await ws.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("standalone", "triton"), required=True)
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=Path(os.environ.get("DEMO_SITE_DIR", "/app/demo")),
    )
    parser.add_argument(
        "--sdk-dir",
        type=Path,
        default=Path(os.environ.get("ENGINE_SDK_DIR", "/app/sdk")),
    )
    args = parser.parse_args()
    asyncio.run(
        run_container_smoke(
            runtime_type=args.runtime, site_dir=args.site_dir, sdk_dir=args.sdk_dir
        )
    )
    print(f"{args.runtime} distribution container smoke: OK")


if __name__ == "__main__":
    main()
