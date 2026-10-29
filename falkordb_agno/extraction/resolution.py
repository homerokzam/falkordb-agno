"""Deterministic entity resolution within one ingestion batch."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .extractor import ExtractedGraph


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def entity_id(name: str, label: str) -> str:
    return f"{normalize(name)}__{label}"


@dataclass
class ResolvedEntity:
    id: str
    name: str
    label: str
    description: str = ""
    aliases: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)
    chunk_ids: set[str] = field(default_factory=set)


@dataclass
class ResolvedRelation:
    src: str
    tgt: str
    rel_type: str
    fact: str
    description: str
    chunk_ids: set[str] = field(default_factory=set)


def resolve(
    per_chunk: list[tuple[str, ExtractedGraph]],
) -> tuple[dict[str, ResolvedEntity], dict[tuple[str, str, str, str], ResolvedRelation]]:
    ents: dict[str, ResolvedEntity] = {}
    rels: dict[tuple[str, str, str, str], ResolvedRelation] = {}
    for chunk_id, g in per_chunk:
        local: dict[str, str] = {}
        for e in g.entities:
            eid = entity_id(e.name, e.type)
            cur = ents.setdefault(eid, ResolvedEntity(eid, e.name.strip(), e.type))
            if len(e.description) > len(cur.description):
                cur.description = e.description
            if e.name.strip() != cur.name:
                cur.aliases.add(e.name.strip())
            cur.attributes.update(e.attributes)
            cur.chunk_ids.add(chunk_id)
            local[normalize(e.name)] = eid
        for r in g.relations:
            s, t = local.get(normalize(r.source)), local.get(normalize(r.target))
            if not s or not t:
                continue
            key = (s, t, r.type, r.fact.strip().lower())
            cur_r = rels.setdefault(key, ResolvedRelation(s, t, r.type, r.fact, r.description))
            cur_r.chunk_ids.add(chunk_id)
    return ents, rels
