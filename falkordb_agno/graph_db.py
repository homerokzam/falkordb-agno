"""FalkorGraphDb: an agno ``VectorDb`` that stores a GraphRAG knowledge graph in FalkorDB."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Coroutine
from typing import Any

from agno.knowledge.document import Document
from agno.vectordb.base import VectorDb

from .config import ExtractionConfig, RetrievalConfig
from .extraction.extractor import ExtractedGraph, ExtractionError, GraphExtractor
from .extraction.pruning import prune
from .extraction.resolution import resolve
from .llm import StructuredLLM
from .ontology.manager import OntologyManager
from .ontology.models import Ontology
from .retrieval.retriever import GraphRetriever
from .store import writer
from .store.connection import FalkorConnection
from .store.indexes import ensure_indexes

_log = logging.getLogger("falkordb_agno")


class FalkorGraphDb(VectorDb):
    def __init__(
        self,
        *,
        graph_name: str,
        model: Any,
        embedder: Any,
        connection: FalkorConnection | None = None,
        ontology: Ontology | None = None,
        require_approval: bool = False,
        extraction: ExtractionConfig | None = None,
        retrieval: RetrievalConfig | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(id=graph_name, name=name or graph_name, description=description)
        self.graph_name = graph_name
        self.conn = connection or FalkorConnection()
        self.model = model
        self.embedder = embedder
        self.llm = StructuredLLM(model)
        self.extraction = extraction or ExtractionConfig()
        self.retriever = GraphRetriever(self.conn, graph_name, embedder, retrieval)
        self.extractor = GraphExtractor(
            self.llm, self.extraction.max_retries, self.extraction.max_entities_per_chunk
        )
        self.manager = OntologyManager(self, declared=ontology, require_approval=require_approval)
        self._write_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}

    # ---- helpers ------------------------------------------------------------------------
    @property
    def ontology(self) -> Ontology:
        return self.manager.current

    def _sync[T](self, coro: Coroutine[Any, Any, T]) -> T:
        return self.conn.run_sync(coro)

    def evolution_lock(self) -> asyncio.Lock:
        """Serialises ingestion and ontology apply inside this process."""
        loop = asyncio.get_running_loop()
        return self._write_locks.setdefault(loop, asyncio.Lock())

    @staticmethod
    def _uid(user_id: str | None) -> str:
        return user_id or ""

    async def _embed_many(self, texts: list[str]) -> list[list[float]]:
        sem = asyncio.Semaphore(self.extraction.concurrency)

        async def one(t: str) -> list[float]:
            async with sem:
                return [float(x) for x in await self.embedder.async_get_embedding(t)]

        return list(await asyncio.gather(*(one(t) for t in texts)))

    # ---- lifecycle ----------------------------------------------------------------------
    async def async_create(self) -> None:
        dim = int(getattr(self.embedder, "dimensions", None) or 1536)
        await ensure_indexes(self.conn, self.graph_name, dim)
        await self.manager.ensure_initialized()

    def create(self) -> None:
        self._sync(self.async_create())

    async def async_exists(self) -> bool:
        return await self.conn.graph_exists(self.graph_name)

    def exists(self) -> bool:
        return self._sync(self.async_exists())

    async def async_drop(self) -> None:
        await self.conn.delete_graph(self.graph_name)
        await self.conn.delete_graph(self.manager.store.graph)
        self.manager.reset()

    def drop(self) -> None:
        self._sync(self.async_drop())

    def get_supported_search_types(self) -> list[str]:
        return ["hybrid"]

    # ---- existence checks ---------------------------------------------------------------
    async def _count(self, cypher: str, params: dict[str, Any]) -> int:
        if not await self.async_exists():
            return 0
        rows = await self.conn.query(self.graph_name, cypher, params, ro=True)
        return int(rows[0][0]) if rows else 0

    async def async_name_exists(self, name: str) -> bool:  # type: ignore[override]
        return await self._count("MATCH (d:Document {name:$n}) RETURN count(d)", {"n": name}) > 0

    def name_exists(self, name: str) -> bool:
        return self._sync(self.async_name_exists(name))

    def id_exists(self, id: str) -> bool:
        return self._sync(self._count("MATCH (c:Chunk {id:$i}) RETURN count(c)", {"i": id})) > 0

    def content_hash_exists(self, content_hash: str, user_id: str | None = None) -> bool:
        return (
            self._sync(
                self._count(
                    "MATCH (d:Document {content_hash:$h, user_id:$u}) RETURN count(d)",
                    {"h": content_hash, "u": self._uid(user_id)},
                )
            )
            > 0
        )

    # ---- ingestion ----------------------------------------------------------------------
    def upsert_available(self) -> bool:
        return True

    async def async_insert(
        self,
        content_hash: str,
        documents: list[Document],
        filters: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        if not documents:
            return
        await self.manager.ensure_initialized()
        async with self.evolution_lock():
            await self._ingest(content_hash, documents, filters or {}, self._uid(user_id))

    def insert(
        self,
        content_hash: str,
        documents: list[Document],
        filters: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        self._sync(self.async_insert(content_hash, documents, filters, user_id))

    async def async_upsert(
        self,
        content_hash: str,
        documents: list[Document],
        filters: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        for cid in {d.content_id for d in documents if d.content_id}:
            await self.async_delete_by_content_id(cid, user_id)
        await self.async_insert(content_hash, documents, filters, user_id)

    def upsert(
        self,
        content_hash: str,
        documents: list[Document],
        filters: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        self._sync(self.async_upsert(content_hash, documents, filters, user_id))

    async def _ingest(
        self, content_hash: str, documents: list[Document], filters: dict[str, Any], uid: str
    ) -> None:
        groups: dict[str, list[Document]] = {}
        for d in documents:
            groups.setdefault(d.content_id or content_hash, []).append(d)

        for content_id, docs in groups.items():
            name = docs[0].name or content_id
            vectors = await self._embed_many([d.content for d in docs])
            chunks = []
            for i, (d, vec) in enumerate(zip(docs, vectors, strict=True)):
                meta = {**(d.meta_data or {}), **filters}
                chunks.append(
                    {
                        "id": d.id or f"{content_id}:{i}:{uuid.uuid4().hex[:6]}",
                        "text": d.content,
                        "idx": i,
                        "embedding": vec,
                        "meta": writer.meta_props(meta),
                    }
                )
            # Extract before writing: a document whose every chunk fails leaves nothing behind.
            per_chunk = await self._extract(name, [(str(c["id"]), str(c["text"])) for c in chunks])
            await writer.write_document(
                self.conn,
                self.graph_name,
                content_id=content_id,
                name=name,
                content_hash=content_hash,
                user_id=uid,
                meta={**(docs[0].meta_data or {}), **filters},
                chunks=chunks,
            )
            await self._write_graph(per_chunk)

    async def _extract(
        self, document: str, chunks: list[tuple[str, str]]
    ) -> list[tuple[str, ExtractedGraph]]:
        """Pruned graph per chunk. A failed chunk is logged and skipped; if all of them fail,
        ``ExtractionError`` is raised instead of writing an empty graph."""
        onto = self.ontology
        sem = asyncio.Semaphore(self.extraction.concurrency)
        errors: dict[str, str] = {}

        async def one(cid: str, text: str) -> tuple[str, ExtractedGraph] | None:
            async with sem:
                try:
                    g = await self.extractor.extract(onto, text)
                except Exception as e:  # noqa: BLE001 - one bad chunk must not abort the document
                    errors[cid] = str(e)
                    _log.warning("extraction failed for chunk %s of %r: %s", cid, document, e)
                    return None
                return cid, prune(g, onto)

        done = await asyncio.gather(*(one(c, t) for c, t in chunks))
        if chunks and len(errors) == len(chunks):
            raise ExtractionError(document, errors)
        return [r for r in done if r is not None]

    async def _write_graph(self, per_chunk: list[tuple[str, ExtractedGraph]]) -> None:
        ents, rels = resolve(per_chunk)
        if ents:
            ids = list(ents)
            vecs = await self._embed_many(
                [f"{ents[i].name}: {ents[i].description}".strip(": ") for i in ids]
            )
            await writer.write_entities(
                self.conn, self.graph_name, ents, dict(zip(ids, vecs, strict=True))
            )
        if rels:
            rl = list(rels.values())
            rvecs = await self._embed_many([r.fact or f"{r.src} {r.rel_type} {r.tgt}" for r in rl])
            await writer.write_relations(self.conn, self.graph_name, rl, rvecs)

    # ---- search -------------------------------------------------------------------------
    async def async_search(
        self, query: str, limit: int = 5, filters: Any = None, user_id: str | None = None
    ) -> list[Document]:
        if not await self.async_exists():
            return []
        await self.manager.ensure_initialized()
        return await self.retriever.search(query, limit, filters, user_id)

    def search(
        self, query: str, limit: int = 5, filters: Any = None, user_id: str | None = None
    ) -> list[Document]:
        return self._sync(self.async_search(query, limit, filters, user_id))

    # ---- deletion -----------------------------------------------------------------------
    async def async_delete_by_content_id(self, content_id: str, user_id: str | None = None) -> bool:
        if not await self.async_exists():
            return False
        async with self.evolution_lock():
            return await writer.delete_content(self.conn, self.graph_name, content_id)

    def delete_by_content_id(self, content_id: str, user_id: str | None = None) -> bool:
        return self._sync(self.async_delete_by_content_id(content_id, user_id))

    async def _delete_where(self, match: str, params: dict[str, Any]) -> bool:
        if not await self.async_exists():
            return False
        rows = await self.conn.query(
            self.graph_name,
            f"MATCH (d:Document) WHERE {match} RETURN d.content_id",
            params,
            ro=True,
        )
        ok = False
        for (cid,) in rows:
            ok = await self.async_delete_by_content_id(cid) or ok
        return ok

    def delete_by_id(self, id: str) -> bool:
        return self._sync(self._delete_where("d.content_id = $v", {"v": id}))

    def delete_by_name(self, name: str) -> bool:
        return self._sync(self._delete_where("d.name = $v", {"v": name}))

    def delete_by_metadata(self, metadata: dict[str, Any]) -> bool:
        if not metadata:
            return False
        clauses, params = [], {}
        for i, (k, v) in enumerate(metadata.items()):
            clauses.append(f"d.meta_json CONTAINS $m{i}")
            params[f"m{i}"] = f'"{k}": {__import__("json").dumps(v)}'
        return self._sync(self._delete_where(" AND ".join(clauses), params))

    def delete(self) -> bool:
        self.drop()
        return True

    async def aclose(self) -> None:
        await self.conn.aclose()
