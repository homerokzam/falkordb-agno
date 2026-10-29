"""Idempotent LLM backfills. Each chunk is marked in ``c.extracted_ops`` once processed."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..extraction.extractor import NamedValues
from ..extraction.prompts import SYSTEM, backfill_attribute_prompt
from ..extraction.pruning import coerce, prune
from ..extraction.resolution import resolve
from ..store import writer
from ..store.indexes import safe_label
from .changes import AddAttribute, BackfillEntity, BackfillRelationPattern
from .errors import OntologyEvolutionError
from .models import Ontology

if TYPE_CHECKING:
    from ..graph_db import FalkorGraphDb


@dataclass
class BackfillResult:
    op_id: str
    chunks_in_scope: int = 0
    scanned: int = 0
    skipped: int = 0
    llm_calls: int = 0
    values_filled: int = 0
    failed_chunks: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _AttrValues(BaseModel):
    values: NamedValues = Field(default_factory=dict)  # entity name -> raw value


Scope = list[tuple[str, str, Any]]  # (chunk_id, text, extra)
Processor = Callable[[str, str, Any], Awaitable[int]]


async def run_backfill(
    db: FalkorGraphDb,
    op_id: str,
    scope: Scope,
    process: Processor,
    *,
    concurrency: int,
    dry_run: bool = False,
) -> BackfillResult:
    t0 = time.monotonic()
    done = {
        r[0]
        for r in await db.conn.query(
            db.graph_name,
            "MATCH (c:Chunk) WHERE $op IN c.extracted_ops RETURN c.id",
            {"op": op_id},
            ro=True,
        )
    }
    res = BackfillResult(op_id=op_id, chunks_in_scope=len(scope), dry_run=dry_run)
    todo = [s for s in scope if s[0] not in done]
    res.skipped = len(scope) - len(todo)
    if dry_run:
        res.llm_calls = len(todo)
        res.elapsed_s = time.monotonic() - t0
        return res
    sem = asyncio.Semaphore(concurrency)

    async def one(cid: str, text: str, extra: Any) -> None:
        async with sem:
            try:
                res.llm_calls += 1
                res.values_filled += await process(cid, text, extra)
                await db.conn.query(
                    db.graph_name,
                    "MATCH (c:Chunk {id:$id}) SET c.extracted_ops = c.extracted_ops + [$op]",
                    {"id": cid, "op": op_id},
                )
                res.scanned += 1
            except Exception:  # noqa: BLE001 - collected and reported below
                res.failed_chunks.append(cid)

    await asyncio.gather(*(one(*s) for s in todo))
    res.elapsed_s = time.monotonic() - t0
    if res.failed_chunks:
        raise OntologyEvolutionError(
            f"{op_id}: {len(res.failed_chunks)} chunk(s) failed", res.failed_chunks
        )
    return res


async def backfill_attribute(
    db: FalkorGraphDb, op: AddAttribute, *, concurrency: int, dry_run: bool = False
) -> BackfillResult:
    label, attr = safe_label(op.owner), safe_label(op.attribute.name)
    rows = await db.conn.query(
        db.graph_name,
        f"MATCH (e:__Entity__:{label})-[:MENTIONED_IN]->(c:Chunk) "
        "RETURN c.id, c.text, collect(e.name)",
        ro=True,
    )
    scope: Scope = [(cid, text, names) for cid, text, names in rows]

    async def process(cid: str, text: str, names: list[str]) -> int:
        out = await db.llm.extract(
            backfill_attribute_prompt(op.owner, op.attribute.name, op.attribute.type, names, text),
            _AttrValues,
            max_retries=db.extraction.max_retries,
            system=SYSTEM,
        )
        filled = 0
        for name, raw in out.values.items():
            val = coerce(raw, op.attribute.type)
            if val is None:
                continue
            await db.conn.query(
                db.graph_name,
                f"MATCH (e:__Entity__:{label} {{name:$n}}) SET e.{attr} = $v",
                {"n": name, "v": val},
            )
            filled += 1
        return filled

    op_id = f"add_attribute:{op.owner}:{op.attribute.name}:{op.attribute.type}"
    return await run_backfill(db, op_id, scope, process, concurrency=concurrency, dry_run=dry_run)


def _sub_ontology(ontology: Ontology, labels: list[str], relation: str | None) -> Ontology:
    ents = [e for e in ontology.entity_types if e.label in labels]
    rels = [r for r in ontology.relation_types if r.label == relation] if relation else []
    return Ontology(entity_types=ents, relation_types=rels)


async def _extract_into_graph(db: FalkorGraphDb, onto: Ontology, cid: str, text: str) -> int:
    g = prune(await db.extractor.extract(onto, text), onto)
    ents, rels = resolve([(cid, g)])
    n = len(ents) + len(rels)
    if ents:
        ids = list(ents)
        vecs = await db._embed_many(
            [f"{ents[i].name}: {ents[i].description}".strip(": ") for i in ids]
        )
        await writer.write_entities(db.conn, db.graph_name, ents, dict(zip(ids, vecs, strict=True)))
    if rels:
        rl = list(rels.values())
        rv = await db._embed_many([r.fact or f"{r.src} {r.rel_type} {r.tgt}" for r in rl])
        await writer.write_relations(db.conn, db.graph_name, rl, rv)
    return n


async def backfill_entity(
    db: FalkorGraphDb,
    op: BackfillEntity,
    ontology: Ontology,
    *,
    concurrency: int,
    dry_run: bool = False,
) -> BackfillResult:
    rows = await db.conn.query(db.graph_name, "MATCH (c:Chunk) RETURN c.id, c.text", ro=True)
    sub = _sub_ontology(ontology, [op.label], None)
    return await run_backfill(
        db,
        f"backfill_entity:{op.label}",
        [(c, t, None) for c, t in rows],
        lambda cid, text, _x: _extract_into_graph(db, sub, cid, text),
        concurrency=concurrency,
        dry_run=dry_run,
    )


async def backfill_relation_pattern(
    db: FalkorGraphDb,
    op: BackfillRelationPattern,
    ontology: Ontology,
    *,
    concurrency: int,
    dry_run: bool = False,
) -> BackfillResult:
    s, t = safe_label(op.source), safe_label(op.target)
    rows = await db.conn.query(
        db.graph_name,
        f"MATCH (a:__Entity__:{s})-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(b:__Entity__:{t}) "
        "RETURN DISTINCT c.id, c.text",
        ro=True,
    )
    sub = _sub_ontology(ontology, [op.source, op.target], op.relation)
    return await run_backfill(
        db,
        f"backfill_relation_pattern:{op.relation}:{op.source}:{op.target}",
        [(c, tx, None) for c, tx in rows],
        lambda cid, text, _x: _extract_into_graph(db, sub, cid, text),
        concurrency=concurrency,
        dry_run=dry_run,
    )
