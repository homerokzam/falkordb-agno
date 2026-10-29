"""Batched writes to the data graph."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from ..extraction.resolution import ResolvedEntity, ResolvedRelation
from .connection import FalkorConnection
from .indexes import safe_label

_log = logging.getLogger("falkordb_agno")


def meta_props(meta: dict[str, Any]) -> dict[str, Any]:
    """Scalar metadata becomes filterable ``meta_<key>`` properties."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            out[f"meta_{k}"] = v
    return out


async def write_document(
    conn: FalkorConnection,
    graph: str,
    *,
    content_id: str,
    name: str,
    content_hash: str,
    user_id: str,
    meta: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> None:
    await conn.query(
        graph,
        "MERGE (d:Document {content_id:$cid}) "
        "SET d.name=$name, d.content_hash=$h, d.user_id=$u, d.meta_json=$mj",
        {
            "cid": content_id,
            "name": name,
            "h": content_hash,
            "u": user_id,
            "mj": json.dumps(meta, default=str),
        },
    )
    await conn.query(
        graph,
        "UNWIND $rows AS row MATCH (d:Document {content_id:$cid}) "
        "MERGE (c:Chunk {id: row.id}) "
        "SET c.text=row.text, c.idx=row.idx, c.content_id=$cid, c.doc_name=$name, c.user_id=$u, "
        "c.embedding=vecf32(row.embedding), c.extracted_ops=[] "
        "SET c += row.meta "
        "MERGE (c)-[:PART_OF]->(d)",
        {"rows": chunks, "cid": content_id, "name": name, "u": user_id},
    )
    await conn.query(
        graph,
        "MATCH (a:Chunk {content_id:$cid}), (b:Chunk {content_id:$cid}) WHERE b.idx = a.idx + 1 "
        "MERGE (a)-[:NEXT_CHUNK]->(b)",
        {"cid": content_id},
    )


def _entity_row(e: ResolvedEntity, embedding: list[float]) -> dict[str, Any]:
    return {
        "id": e.id,
        "name": e.name,
        "description": e.description,
        "aliases": sorted(e.aliases),
        "attrs": e.attributes,
        "cids": sorted(e.chunk_ids),
        "embedding": embedding,
    }


async def write_entities(
    conn: FalkorConnection,
    graph: str,
    entities: dict[str, ResolvedEntity],
    embeddings: dict[str, list[float]],
) -> None:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for e in entities.values():
        by_label.setdefault(e.label, []).append(_entity_row(e, embeddings[e.id]))
    for label, rows in by_label.items():
        try:  # one bad label group must not abort the document
            await conn.query(
                graph,
                f"UNWIND $rows AS row MERGE (e:__Entity__:{safe_label(label)} {{id: row.id}}) "
                "ON CREATE SET e.name=row.name, e.aliases=row.aliases, e.source_chunk_ids=[] "
                "SET e.description = CASE "
                "WHEN size(row.description) > size(coalesce(e.description,'')) "
                "THEN row.description ELSE e.description END, "
                "e.source_chunk_ids = e.source_chunk_ids + row.cids, "
                "e.embedding = vecf32(row.embedding) "
                "SET e += row.attrs",
                {"rows": rows},
            )
        except Exception as err:  # noqa: BLE001
            _log.warning("writing %d %r entities failed: %s", len(rows), label, err)
    mentions = [{"id": e.id, "cid": c} for e in entities.values() for c in e.chunk_ids]
    if mentions:
        await conn.query(
            graph,
            "UNWIND $rows AS row MATCH (e:__Entity__ {id: row.id}), (c:Chunk {id: row.cid}) "
            "MERGE (e)-[:MENTIONED_IN]->(c)",
            {"rows": mentions},
        )


async def write_relations(
    conn: FalkorConnection,
    graph: str,
    relations: list[ResolvedRelation],
    embeddings: list[list[float]],
) -> None:
    rows: list[dict[str, Any]] = [
        {
            "s": r.src,
            "t": r.tgt,
            "rt": r.rel_type,
            "key": hashlib.sha1(f"{r.src}|{r.tgt}|{r.rel_type}|{r.fact}".encode()).hexdigest()[:16],
            "fact": r.fact or f"{r.src} {r.rel_type} {r.tgt}",
            "desc": r.description,
            "cids": sorted(r.chunk_ids),
            "embedding": emb,
        }
        for r, emb in zip(relations, embeddings, strict=True)
    ]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row["rt"], []).append(row)
    for rel_type, group in by_type.items():
        try:  # one bad relation group must not abort the document
            await conn.query(
                graph,
                "UNWIND $rows AS row MATCH (s:__Entity__ {id: row.s}), (t:__Entity__ {id: row.t}) "
                "MERGE (s)-[r:RELATES {rel_type: row.rt, key: row.key}]->(t) "
                "ON CREATE SET r.source_chunk_ids = [] "
                "SET r.fact=row.fact, r.description=row.desc, "
                "r.source_chunk_ids = r.source_chunk_ids + row.cids, "
                "r.embedding=vecf32(row.embedding)",
                {"rows": group},
            )
        except Exception as err:  # noqa: BLE001
            _log.warning("writing %d %r relations failed: %s", len(group), rel_type, err)


async def delete_content(conn: FalkorConnection, graph: str, content_id: str) -> bool:
    rows = await conn.query(
        graph, "MATCH (c:Chunk {content_id:$cid}) RETURN collect(c.id)", {"cid": content_id}
    )
    ids: list[str] = rows[0][0] if rows else []
    docs = await conn.query(
        graph, "MATCH (d:Document {content_id:$cid}) RETURN count(d)", {"cid": content_id}
    )
    if not ids and not docs[0][0]:
        return False
    p = {"ids": ids}
    await conn.query(
        graph,
        "MATCH ()-[r:RELATES]->() WHERE any(x IN r.source_chunk_ids WHERE x IN $ids) "
        "SET r.source_chunk_ids = [x IN r.source_chunk_ids WHERE NOT x IN $ids]",
        p,
    )
    await conn.query(graph, "MATCH ()-[r:RELATES]->() WHERE size(r.source_chunk_ids) = 0 DELETE r")
    await conn.query(
        graph,
        "MATCH (e:__Entity__) WHERE any(x IN e.source_chunk_ids WHERE x IN $ids) "
        "SET e.source_chunk_ids = [x IN e.source_chunk_ids WHERE NOT x IN $ids]",
        p,
    )
    await conn.query(
        graph, "MATCH (e:__Entity__) WHERE size(e.source_chunk_ids) = 0 DETACH DELETE e"
    )
    await conn.query(
        graph, "MATCH (c:Chunk {content_id:$cid}) DETACH DELETE c", {"cid": content_id}
    )
    await conn.query(
        graph, "MATCH (d:Document {content_id:$cid}) DETACH DELETE d", {"cid": content_id}
    )
    return True
