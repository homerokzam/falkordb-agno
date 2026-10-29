"""Persistence of the ontology in the ``<graph>__ontology`` FalkorDB graph."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from ..store.connection import FalkorConnection
from .changes import ChangeSet
from .errors import OntologyLockError
from .models import Attribute, EntityType, Ontology, RelationType

ONTOLOGY_SUFFIX = "__ontology"


@dataclass
class VersionInfo:
    version: int
    hash: str
    created_at: float
    author: str
    message: str
    changeset_id: str | None


class OntologyStore:
    def __init__(self, conn: FalkorConnection, graph_name: str) -> None:
        self.conn = conn
        self.graph = graph_name + ONTOLOGY_SUFFIX

    async def _q(self, cypher: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        return await self.conn.query(self.graph, cypher, params)

    # ---- current state ------------------------------------------------------------------
    async def load(self) -> Ontology:
        if not await self.conn.graph_exists(self.graph):
            return Ontology()
        ents = await self._q(
            "MATCH (e:EntityType) OPTIONAL MATCH (e)-[:HAS_PROPERTY]->(p:Property) "
            "RETURN e.label, e.description, collect(p {.name, .type, .description, .required})"
        )
        entity_types = [
            EntityType(
                label=lab, description=desc, attributes=[Attribute(**p) for p in props if p["name"]]
            )
            for lab, desc, props in ents
        ]
        rels = await self._q(
            "MATCH (r:RelationType) "
            "OPTIONAL MATCH (r)-[:SOURCE]->(s:EntityType) "
            "OPTIONAL MATCH (r)-[:TARGET]->(t:EntityType) "
            "OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(p:Property) "
            "RETURN r.label, r.description, r.key, s.label, t.label, "
            "collect(DISTINCT p {.name, .type, .description, .required})"
        )
        by_label: dict[str, RelationType] = {}
        for lab, desc, _key, src, tgt, props in rels:
            rt = by_label.setdefault(
                lab,
                RelationType(
                    label=lab,
                    description=desc,
                    attributes=[Attribute(**p) for p in props if p["name"]],
                ),
            )
            if src and tgt and (src, tgt) not in rt.patterns:
                rt.patterns.append((src, tgt))
        return Ontology(entity_types=entity_types, relation_types=list(by_label.values()))

    async def save_state(self, ontology: Ontology) -> None:
        """Replace the current-state nodes (versions/changesets/lock are untouched)."""
        await self._q(
            "MATCH (n) WHERE n:EntityType OR n:RelationType OR n:Property DETACH DELETE n"
        )
        for e in ontology.entity_types:
            await self._q(
                "CREATE (e:EntityType {label:$l, description:$d}) "
                "WITH e UNWIND $props AS p CREATE (e)-[:HAS_PROPERTY]->(:Property "
                "{name:p.name, type:p.type, description:p.description, required:p.required})",
                {
                    "l": e.label,
                    "d": e.description,
                    "props": [a.model_dump() for a in e.attributes] or [],
                },
            )
        for r in ontology.relation_types:
            props = [a.model_dump() for a in r.attributes]
            patterns: list[tuple[str | None, str | None]] = list(r.patterns) or [(None, None)]
            for i, (s, t) in enumerate(patterns):
                await self._q(
                    "CREATE (r:RelationType {label:$l, description:$d, key:$k}) "
                    "WITH r CALL { WITH r UNWIND $props AS p CREATE (r)-[:HAS_PROPERTY]->"
                    "(:Property {name:p.name, type:p.type, description:p.description, "
                    "required:p.required}) RETURN count(*) AS c } "
                    "WITH r OPTIONAL MATCH (s:EntityType {label:$s}), (t:EntityType {label:$t}) "
                    "FOREACH (_ IN CASE WHEN s IS NULL OR t IS NULL THEN [] ELSE [1] END | "
                    "CREATE (r)-[:SOURCE]->(s), (r)-[:TARGET]->(t))",
                    {"l": r.label, "d": r.description, "k": i, "props": props, "s": s, "t": t},
                )

    # ---- versions -----------------------------------------------------------------------
    async def current_version(self) -> int:
        if not await self.conn.graph_exists(self.graph):
            return 0
        rows = await self._q("MATCH (v:OntologyVersion) RETURN max(v.version)")
        return int(rows[0][0]) if rows and rows[0][0] is not None else 0

    async def add_version(
        self, ontology: Ontology, *, author: str, message: str, changeset_id: str | None
    ) -> int:
        n = await self.current_version() + 1
        await self._q(
            "CREATE (v:OntologyVersion {version:$n, hash:$h, created_at:$t, author:$a, "
            "message:$m, changeset_id:$c, snapshot_json:$s})",
            {
                "n": n,
                "h": ontology.digest(),
                "t": time.time(),
                "a": author,
                "m": message,
                "c": changeset_id,
                "s": ontology.model_dump_json(),
            },
        )
        return n

    async def history(self) -> list[VersionInfo]:
        if not await self.conn.graph_exists(self.graph):
            return []
        rows = await self._q(
            "MATCH (v:OntologyVersion) RETURN v.version, v.hash, v.created_at, v.author, "
            "v.message, v.changeset_id ORDER BY v.version"
        )
        return [VersionInfo(*r) for r in rows]

    async def get_version(self, n: int) -> Ontology | None:
        rows = await self._q(
            "MATCH (v:OntologyVersion {version:$n}) RETURN v.snapshot_json", {"n": n}
        )
        return Ontology.model_validate_json(rows[0][0]) if rows else None

    # ---- changesets ---------------------------------------------------------------------
    async def save_changeset(self, cs: ChangeSet) -> None:
        d = cs.model_dump(mode="json")
        await self._q(
            "MERGE (c:ChangeSet {id:$id}) SET c.status=$status, c.base_version=$bv, "
            "c.ops_json=$ops, c.message=$msg, c.created_by=$cb, c.source=$src, "
            "c.created_at=$ca, c.reviewed_by=$rb, c.reviewed_at=$ra, c.review_note=$rn, "
            "c.applied_version=$av, c.error=$err, c.results_json=$res",
            {
                "id": cs.id,
                "status": d["status"],
                "bv": cs.base_version,
                "ops": json.dumps(d["ops"]),
                "msg": cs.message,
                "cb": cs.created_by,
                "src": cs.source,
                "ca": cs.created_at,
                "rb": cs.reviewed_by,
                "ra": cs.reviewed_at,
                "rn": cs.review_note,
                "av": cs.applied_version,
                "err": cs.error,
                "res": json.dumps(d["results"]),
            },
        )

    @staticmethod
    def _cs_from_row(r: list[Any]) -> ChangeSet:
        return ChangeSet.model_validate(
            {
                "id": r[0],
                "status": r[1],
                "base_version": r[2],
                "ops": json.loads(r[3]),
                "message": r[4] or "",
                "created_by": r[5],
                "source": r[6],
                "created_at": r[7],
                "reviewed_by": r[8],
                "reviewed_at": r[9],
                "review_note": r[10],
                "applied_version": r[11],
                "error": r[12],
                "results": json.loads(r[13] or "{}"),
            }
        )

    _CS_COLS = (
        "c.id, c.status, c.base_version, c.ops_json, c.message, c.created_by, c.source, "
        "c.created_at, c.reviewed_by, c.reviewed_at, c.review_note, c.applied_version, "
        "c.error, c.results_json"
    )

    async def get_changeset(self, cs_id: str) -> ChangeSet | None:
        if not await self.conn.graph_exists(self.graph):
            return None
        rows = await self._q(
            f"MATCH (c:ChangeSet {{id:$id}}) RETURN {self._CS_COLS}", {"id": cs_id}
        )
        return self._cs_from_row(rows[0]) if rows else None

    async def list_changesets(self, status: str | None = None) -> list[ChangeSet]:
        if not await self.conn.graph_exists(self.graph):
            return []
        where = "WHERE c.status = $s" if status else ""
        rows = await self._q(
            f"MATCH (c:ChangeSet) {where} RETURN {self._CS_COLS} ORDER BY c.created_at",
            {"s": status} if status else None,
        )
        return [self._cs_from_row(r) for r in rows]

    # ---- lock ---------------------------------------------------------------------------
    async def acquire_lock(self, holder: str | None = None, ttl: float = 600.0) -> str:
        holder = holder or uuid.uuid4().hex
        now = time.time()
        await self._q(
            "MERGE (l:Lock {name:'ontology'}) "
            "ON CREATE SET l.holder = $h, l.expires_at = $exp "
            "ON MATCH SET l.holder = CASE WHEN l.holder IS NULL OR l.expires_at < $now "
            "THEN $h ELSE l.holder END, "
            "l.expires_at = CASE WHEN l.holder IS NULL OR l.expires_at < $now "
            "THEN $exp ELSE l.expires_at END",
            {"h": holder, "exp": now + ttl, "now": now},
        )
        rows = await self._q("MATCH (l:Lock {name:'ontology'}) RETURN l.holder")
        if not rows or rows[0][0] != holder:
            raise OntologyLockError("ontology lock is held by another process")
        return holder

    async def release_lock(self, holder: str) -> None:
        await self._q(
            "MATCH (l:Lock {name:'ontology', holder:$h}) SET l.holder = NULL, l.expires_at = 0",
            {"h": holder},
        )
