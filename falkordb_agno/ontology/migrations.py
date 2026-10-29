"""Data-graph migrations for each ontology op. All are idempotent (safe to re-run)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..store.indexes import safe_label
from .backfill import backfill_attribute, backfill_entity, backfill_relation_pattern
from .changes import (
    AddAttribute,
    BackfillEntity,
    BackfillRelationPattern,
    DropAttribute,
    DropEntityType,
    DropRelationPattern,
    DropRelationType,
    Op,
    RenameAttribute,
    RenameEntityType,
    RenameRelationType,
)
from .models import Ontology

if TYPE_CHECKING:
    from ..graph_db import FalkorGraphDb


def _is_entity(o: Ontology, owner: str) -> bool:
    return o.entity(owner) is not None


async def migrate(
    db: FalkorGraphDb, op: Op, before: Ontology, after: Ontology, *, concurrency: int
) -> dict[str, Any]:
    q, g = db.conn.query, db.graph_name
    if isinstance(op, RenameEntityType):
        old, new = safe_label(op.old), safe_label(op.new)
        await q(
            g,
            f"MATCH (n:__Entity__:{old}) SET n:{new} "
            f"SET n.id = left(n.id, size(n.id) - size('__{old}')) + '__{new}' REMOVE n:{old}",
        )
        return {}
    if isinstance(op, RenameRelationType):
        await q(
            g,
            "MATCH ()-[r:RELATES {rel_type:$o}]->() SET r.rel_type=$n",
            {"o": op.old, "n": op.new},
        )
        return {}
    if isinstance(op, RenameAttribute):
        old_p, new_p = safe_label(op.old), safe_label(op.new)
        if _is_entity(before, op.owner):
            await q(
                g,
                f"MATCH (n:__Entity__:{safe_label(op.owner)}) WHERE n.{old_p} IS NOT NULL "
                f"SET n.{new_p} = n.{old_p} SET n.{old_p} = NULL",
            )
        else:
            await q(
                g,
                f"MATCH ()-[r:RELATES {{rel_type:$rt}}]->() WHERE r.{old_p} IS NOT NULL "
                f"SET r.{new_p} = r.{old_p} SET r.{old_p} = NULL",
                {"rt": op.owner},
            )
        return {}
    if isinstance(op, DropAttribute):
        p = safe_label(op.name)
        if _is_entity(before, op.owner):
            await q(g, f"MATCH (n:__Entity__:{safe_label(op.owner)}) SET n.{p} = NULL")
        else:
            await q(
                g, f"MATCH ()-[r:RELATES {{rel_type:$rt}}]->() SET r.{p} = NULL", {"rt": op.owner}
            )
        return {}
    if isinstance(op, DropEntityType):
        await q(g, f"MATCH (n:__Entity__:{safe_label(op.label)}) DETACH DELETE n")
        return {}
    if isinstance(op, DropRelationType):
        await q(g, "MATCH ()-[r:RELATES {rel_type:$rt}]->() DELETE r", {"rt": op.label})
        return {}
    if isinstance(op, DropRelationPattern):
        await q(
            g,
            f"MATCH (s:__Entity__:{safe_label(op.source)})-[r:RELATES {{rel_type:$rt}}]->"
            f"(t:__Entity__:{safe_label(op.target)}) DELETE r",
            {"rt": op.relation},
        )
        return {}
    if isinstance(op, AddAttribute):
        if op.backfill and op.owner_kind == "entity":
            res = await backfill_attribute(db, op, concurrency=concurrency)
            return {"backfill": res.as_dict()}
        return {}
    if isinstance(op, BackfillEntity):
        return {
            "backfill": (await backfill_entity(db, op, after, concurrency=concurrency)).as_dict()
        }
    if isinstance(op, BackfillRelationPattern):
        res = await backfill_relation_pattern(db, op, after, concurrency=concurrency)
        return {"backfill": res.as_dict()}
    return {}  # purely schema-level ops
