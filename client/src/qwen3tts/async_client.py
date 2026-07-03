from __future__ import annotations

import asyncio

from ._session import AsyncStreamSession
from .client import TTSClient


class AsyncTTSClient:
    def __init__(self, sync_client: TTSClient) -> None:
        self._sync = sync_client
        self.endpoint = sync_client.endpoint
        self.resolved_transport = sync_client.resolved_transport
        self.probe_report = list(sync_client.probe_report)
        self.detected_transport = sync_client.detected_transport

    @classmethod
    async def connect(cls, endpoint, **kwargs):
        sync_client = await asyncio.to_thread(TTSClient.connect, endpoint, **kwargs)
        return cls(sync_client)

    async def get_capabilities(self):
        return await asyncio.to_thread(self._sync.get_capabilities)

    async def synthesize_bytes(self, text: str, *, request=None):
        return await asyncio.to_thread(
            self._sync.synthesize_bytes, text, request=request
        )

    async def synthesize_array(self, text: str, *, request=None):
        return await asyncio.to_thread(
            self._sync.synthesize_array, text, request=request
        )

    async def aopen_stream(self, start_request):
        sync_session = await asyncio.to_thread(self._sync.open_stream, start_request)
        return AsyncStreamSession(sync_session)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._sync.close)

    async def __aenter__(self) -> "AsyncTTSClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
