"""Multi-path GraphRAG retrieval -> agno Documents."""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any

from agno.knowledge.document import Document

from ..config import RetrievalConfig
from ..store.connection import FalkorConnection


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _ft_query(q: str) -> str:
    words = [w for w in re.findall(r"\w+", q) if len(w) > 2]
    return " ".join(words[:8])


def build_filter(filters: Any, user_id: str | None, alias: str = "c") -> tuple[str, dict[str, Any]]:
    """WHERE fragment (starting with ' AND ') over chunk props."""
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if user_id:
        clauses.append(f"{alias}.user_id = $f_user")
        params["f_user"] = user_id
    items: list[tuple[str, Any]] = []
    if isinstance(filters, dict):
        items = list(filters.items())
    elif isinstance(filters, list):  # agno FilterExpr list: only EQ is supported
        for f in filters:
            if hasattr(f, "key") and hasattr(f, "value"):
                items.append((f.key, f.value))
    for i, (k, v) in enumerate(items):
        if not re.fullmatch(r"\w+", str(k)):
            continue
        clauses.append(f"{alias}.meta_{k} = $f_{i}")
        params[f"f_{i}"] = v
    return ("".join(f" AND {c}" for c in clauses), params)


class GraphRetriever:
    def __init__(
        self,
        conn: FalkorConnection,
        graph: str,
        embedder: Any,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.conn, self.graph, self.embedder = conn, graph, embedder
        self.cfg = config or RetrievalConfig()

    async def _q(self, cypher: str, params: dict[str, Any]) -> list[list[Any]]:
        return await self.conn.query(self.graph, cypher, params, ro=True)

    async def search(
        self, query: str, limit: int = 5, filters: Any = None, user_id: str | None = None
    ) -> list[Document]:
        cfg = self.cfg
        qv: list[float] = await self.embedder.async_get_embedding(query)
        where, fparams = build_filter(filters, user_id)
        v = {"v": qv}

        chunk_hits, fact_hits, ent_hits, ft_hits = await asyncio.gather(
            self._q(
                "CALL db.idx.vector.queryNodes('Chunk','embedding',$k,vecf32($v)) "
                f"YIELD node AS c, score WHERE true{where} RETURN c.id, score",
                {**v, "k": cfg.chunk_k * 3, **fparams},
            ),
            self._q(
                "CALL db.idx.vector.queryRelationships('RELATES','embedding',$k,vecf32($v)) "
                "YIELD relationship AS r, score "
                "RETURN r.fact, r.source_chunk_ids, startNode(r).id, startNode(r).name, "
                "endNode(r).id, endNode(r).name, r.rel_type, score",
                {**v, "k": cfg.fact_k},
            ),
            self._q(
                "CALL db.idx.vector.queryNodes('__Entity__','embedding',$k,vecf32($v)) "
                "YIELD node AS e, score RETURN e.id, e.name, e.description, e.source_chunk_ids",
                {**v, "k": cfg.entity_k},
            ),
            self._ft(query),
        )

        entities: dict[str, tuple[str, str]] = {}
        chunk_support: dict[str, float] = {}
        coverage: dict[str, int] = {}
        for cid, dist in chunk_hits:
            chunk_support[cid] = max(chunk_support.get(cid, 0.0), 1.0 - float(dist))
        for eid, name, desc, cids in [*ent_hits, *ft_hits]:
            entities[eid] = (name, desc or "")
            for cid in cids or []:
                coverage[cid] = coverage.get(cid, 0) + 1
        facts: list[dict[str, Any]] = []
        for fact, cids, sid, sname, tid, tname, rt, _score in fact_hits:
            facts.append({"fact": fact, "src": sname, "tgt": tname, "rel": rt})
            entities.setdefault(sid, (sname, ""))
            entities.setdefault(tid, (tname, ""))
            for cid in cids or []:
                coverage[cid] = coverage.get(cid, 0) + 1

        if entities and cfg.expand_hops >= 1:
            rows = await self._q(
                "MATCH (e:__Entity__)-[r:RELATES]-(o:__Entity__) WHERE e.id IN $ids "
                "RETURN r.fact, r.source_chunk_ids, startNode(r).name, endNode(r).name, r.rel_type "
                "LIMIT 40",
                {"ids": list(entities)},
            )
            for fact, cids, sname, tname, rt in rows:
                facts.append({"fact": fact, "src": sname, "tgt": tname, "rel": rt})
                for cid in cids or []:
                    coverage[cid] = coverage.get(cid, 0) + 1

        ids = list({*chunk_support, *coverage})
        if not ids:
            return []
        rows = await self._q(
            f"MATCH (c:Chunk) WHERE c.id IN $ids{where} "
            "RETURN c.id, c.text, c.doc_name, c.content_id, c.embedding, c.idx, "
            "c.meta_page_start, c.meta_page_end",
            {"ids": ids, **fparams},
        )
        scored: list[tuple[float, list[Any]]] = []
        for row in rows:
            sim = cosine(qv, list(row[4] or []))
            scored.append((sim + 0.05 * min(coverage.get(row[0], 0), 5), row))
        scored.sort(key=lambda x: -x[0])

        uniq_facts = list({f["fact"]: f for f in facts}.values())
        docs: list[Document] = []
        # Facts are not scoped by chunk filters, so they'd leak across filter/user boundaries.
        if cfg.include_graph_context and uniq_facts and not fparams:
            docs.append(
                Document(
                    id="graph_context",
                    name="graph_context",
                    content="\n".join(f"- {f['fact']}" for f in uniq_facts[:15]),
                    meta_data={
                        "source": "graph",
                        "entities": sorted(n for n, _ in entities.values()),
                    },
                )
            )
        for score, (cid, text, dname, content_id, _emb, idx, p_start, p_end) in scored[:limit]:
            related = [f for f in uniq_facts if f["src"] in text or f["tgt"] in text]
            pages = {
                k: v for k, v in (("page_start", p_start), ("page_end", p_end)) if v is not None
            }
            docs.append(
                Document(
                    id=cid,
                    name=dname,
                    content=text,
                    content_id=content_id,
                    meta_data={
                        "document_name": dname,
                        "content_id": content_id,
                        "chunk_id": cid,
                        "chunk_index": idx,
                        **pages,
                        "score": round(score, 4),
                        "facts": [f["fact"] for f in related[:5]],
                    },
                )
            )
        return docs

    async def _ft(self, query: str) -> list[list[Any]]:
        q = _ft_query(query)
        if not q:
            return []
        try:
            return await self._q(
                "CALL db.idx.fulltext.queryNodes('__Entity__',$q) YIELD node AS e "
                "RETURN e.id, e.name, e.description, e.source_chunk_ids LIMIT $k",
                {"q": q, "k": self.cfg.entity_k},
            )
        except Exception:  # noqa: BLE001 - fulltext is a best-effort extra path
            return []
