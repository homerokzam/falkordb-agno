"""FalkorDB connection: one async client per event loop plus a sync bridge."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

from falkordb.asyncio import FalkorDB


class FalkorConnection:
    """Holds connection settings and hands out graphs.

    redis-py async connections are bound to the loop that created them, so a client is kept
    per running loop. ``run_sync`` runs a coroutine on a private background loop; agno's
    ``Knowledge.__post_init__`` calls the sync VectorDb API even inside a running loop.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        self._kwargs: dict[str, Any] = {"host": host, "port": port, "password": password}
        self._kwargs.update(client_kwargs)
        self._clients: dict[asyncio.AbstractEventLoop, FalkorDB] = {}
        self._bridge: asyncio.AbstractEventLoop | None = None
        self._bridge_lock = threading.Lock()

    def _client(self) -> FalkorDB:
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            client = FalkorDB(**self._kwargs)
            self._clients[loop] = client
        return client

    async def query(
        self, graph: str, cypher: str, params: dict[str, Any] | None = None, *, ro: bool = False
    ) -> list[list[Any]]:
        g = self._client().select_graph(graph)
        res = await (g.ro_query(cypher, params) if ro else g.query(cypher, params))
        return list(res.result_set)

    async def graph_exists(self, graph: str) -> bool:
        return graph in await self._client().list_graphs()

    async def delete_graph(self, graph: str) -> None:
        if await self.graph_exists(graph):
            await self._client().select_graph(graph).delete()

    def run_sync[T](self, coro: Coroutine[Any, Any, T]) -> T:
        with self._bridge_lock:
            if self._bridge is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, daemon=True).start()
                self._bridge = loop
        return asyncio.run_coroutine_threadsafe(coro, self._bridge).result()

    async def aclose(self) -> None:
        for client in list(self._clients.values()):
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                pass
        self._clients.clear()

    def close(self) -> None:
        if self._bridge is not None:
            try:
                self.run_sync(self.aclose())
            finally:
                self._bridge.call_soon_threadsafe(self._bridge.stop)
                self._bridge = None
