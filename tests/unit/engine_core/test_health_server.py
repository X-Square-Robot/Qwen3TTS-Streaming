"""Tests for engine.server.HealthServerThread — HTTP probe semantics.

The health server answers unified platform probes (liveness/readiness/startup
on one path) while the main loop is blocked by the model load, so both server
branches (aiohttp and the raw-asyncio fallback) must agree on status codes.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

pytest.importorskip("torch")

from engine.server import HealthServerThread, HealthState


class FakeEngine:
    """Minimal stand-in for TTSEngine's health surface."""

    def __init__(self):
        self.running = False
        self.alive = False

    def health_stats(self) -> dict:
        return {"running": self.running, "active_sessions": 0}

    def engine_thread_alive(self) -> bool:
        return self.alive


def _get(port: int, path: str) -> tuple[int, dict | None]:
    """GET 127.0.0.1:{port}{path} -> (status_code, json_body_or_None)."""
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return exc.code, None


@pytest.fixture(params=[False, True], ids=["aiohttp", "fallback"])
def health(request):
    if not request.param:
        pytest.importorskip("aiohttp")
    engine = FakeEngine()
    server = HealthServerThread(engine, 0, _force_fallback=request.param)
    server.start()
    yield server, engine
    server.stop()


class TestProbeSemantics:
    def test_health_503_while_loading(self, health):
        server, _ = health
        code, body = _get(server.bound_port, "/health")
        assert code == 503
        assert body["status"] == "loading"
        assert body["running"] is False

    def test_livez_and_metrics_always_200(self, health):
        server, _ = health
        assert _get(server.bound_port, "/livez")[0] == 200
        code, body = _get(server.bound_port, "/metrics")
        assert code == 200
        assert body["status"] == "loading"

    def test_readyz_flips_on_mark_ready(self, health):
        server, engine = health
        assert _get(server.bound_port, "/readyz")[0] == 503
        engine.running = True
        engine.alive = True
        server.mark_ready()
        code, body = _get(server.bound_port, "/readyz")
        assert code == 200
        assert body["status"] == "ok"

    def test_ready_health_200_keeps_running_true(self, health):
        server, engine = health
        engine.running = True
        engine.alive = True
        server.mark_ready()
        code, body = _get(server.bound_port, "/health")
        assert code == 200
        # compose.sh compose_wait_engine_http_health greps this exact key.
        assert body["running"] is True
        assert body["status"] == "ok"

    def test_engine_thread_death_drops_health_but_not_livez(self, health):
        server, engine = health
        engine.running = True
        engine.alive = True
        server.mark_ready()
        assert _get(server.bound_port, "/health")[0] == 200
        engine.alive = False
        code, body = _get(server.bound_port, "/health")
        assert code == 503
        assert body["status"] == "engine_loop_dead"
        assert _get(server.bound_port, "/livez")[0] == 200

    def test_unknown_path_404(self, health):
        server, _ = health
        assert _get(server.bound_port, "/nope")[0] == 404


class TestProbeMode:
    def test_alive_mode_health_200_while_loading(self):
        server = HealthServerThread(FakeEngine(), 0, probe_mode="alive")
        server.start()
        try:
            code, body = _get(server.bound_port, "/health")
            assert code == 200
            assert body["status"] == "loading"
            # /readyz keeps strict semantics regardless of the knob.
            assert _get(server.bound_port, "/readyz")[0] == 503
        finally:
            server.stop()

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="health_probe_mode"):
            HealthServerThread(FakeEngine(), 0, probe_mode="bogus")


class TestBindFailure:
    def test_port_conflict_raises(self):
        first = HealthServerThread(FakeEngine(), 0)
        first.start()
        try:
            second = HealthServerThread(FakeEngine(), first.bound_port)
            with pytest.raises(RuntimeError, match="failed to start"):
                second.start(bind_timeout_sec=5)
        finally:
            first.stop()

    @pytest.mark.parametrize(
        "force_fallback", [False, True], ids=["aiohttp", "fallback"]
    )
    def test_out_of_range_port_raises(self, force_fallback):
        # bind() raises OverflowError (not OSError) for ports > 65535; a
        # typo'd ENGINE_HEALTH_PORT must fail start(), not silently leave a
        # dead health thread while the model loads.
        server = HealthServerThread(
            FakeEngine(), 99999999, _force_fallback=force_fallback
        )
        with pytest.raises(RuntimeError, match="failed to start"):
            server.start(bind_timeout_sec=5)


@pytest.mark.asyncio
async def test_gateway_port_routes_share_health_state():
    """The WS-port probe routes must mirror the health-port semantics."""
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from engine.gateway.websocket_server import add_health_routes

    engine = FakeEngine()
    state = HealthState(engine)
    app = web.Application()
    add_health_routes(app, state)

    server = TestServer(app)
    async with server:
        client = TestClient(server)
        async with client:
            resp = await client.get("/health")
            assert resp.status == 503
            assert (await resp.json())["status"] == "loading"
            assert (await client.get("/livez")).status == 200
            assert (await client.get("/metrics")).status == 200
            assert (await client.get("/readyz")).status == 503

            engine.running = True
            engine.alive = True
            state.mark_ready()

            resp = await client.get("/health")
            assert resp.status == 200
            body = await resp.json()
            assert body["running"] is True
            assert body["status"] == "ok"
            assert (await client.get("/readyz")).status == 200


@pytest.mark.asyncio
async def test_gateway_and_thread_read_one_state():
    """mark_ready on the shared state flips both probe surfaces at once."""
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from engine.gateway.websocket_server import add_health_routes

    engine = FakeEngine()
    state = HealthState(engine)
    thread_server = HealthServerThread(engine, 0, state=state)
    thread_server.start()
    app = web.Application()
    add_health_routes(app, state)
    ws_server = TestServer(app)
    try:
        async with ws_server:
            client = TestClient(ws_server)
            async with client:
                assert _get(thread_server.bound_port, "/health")[0] == 503
                assert (await client.get("/health")).status == 503

                engine.running = True
                engine.alive = True
                state.mark_ready()

                assert _get(thread_server.bound_port, "/health")[0] == 200
                assert (await client.get("/health")).status == 200
    finally:
        thread_server.stop()
