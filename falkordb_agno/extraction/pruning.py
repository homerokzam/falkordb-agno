"""Ontology-based pruning and attribute coercion."""

from __future__ import annotations

import json
from typing import Any

from ..ontology.models import Attribute, Ontology
from ..store.indexes import entity_label, normalize_label
from .extractor import ExtractedGraph


def coerce(value: Any, attr_type: str) -> Any | None:
    """Convert ``value`` to the declared type; ``None`` if it can't be."""
    if value is None:
        return None
    try:
        if attr_type == "STRING":
            return str(value)
        if attr_type == "INTEGER":
            if isinstance(value, bool):
                return None
            return int(float(value))
        if attr_type == "FLOAT":
            return None if isinstance(value, bool) else float(value)
        if attr_type == "BOOLEAN":
            if isinstance(value, bool):
                return value
            s = str(value).strip().lower()
            return (
                True if s in {"true", "yes", "1"} else False if s in {"false", "no", "0"} else None
            )
        if attr_type == "DATE":
            return str(value)
        if attr_type == "LIST":
            if isinstance(value, str) and value.lstrip().startswith("["):
                value = json.loads(value)  # wire values are strings: '["a", "b"]'
            return list(value) if isinstance(value, (list, tuple)) else [value]
    except TypeError, ValueError:
        return None
    return None


def coerce_attributes(declared: list[Attribute], raw: dict[str, Any]) -> dict[str, Any]:
    by_name = {a.name: a for a in declared}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k in by_name:
            c = coerce(v, by_name[k].type)
            if c is not None:
                out[k] = c
    return out


def prune(graph: ExtractedGraph, ontology: Ontology) -> ExtractedGraph:
    """Drop what the ontology doesn't declare. Open ontologies keep everything."""
    if ontology.is_open:
        return graph
    # Compare normalised labels on both sides; keep the declared spelling.
    ent_types = {entity_label(et.label): et for et in ontology.entity_types}
    rel_types = {normalize_label(rt.label): rt for rt in ontology.relation_types}
    ents = []
    for e in graph.entities:
        et = ent_types.get(entity_label(e.type))
        if et is None:
            continue
        ents.append(
            e.model_copy(
                update={
                    "type": et.label,
                    "attributes": coerce_attributes(et.attributes, e.attributes),
                }
            )
        )
    kind = {e.name.strip().lower(): e.type for e in ents}
    rels = []
    for r in graph.relations:
        rt = rel_types.get(normalize_label(r.type))
        st, tt = kind.get(r.source.strip().lower()), kind.get(r.target.strip().lower())
        if rt is None or st is None or tt is None:
            continue
        if rt.patterns and (st, tt) not in rt.patterns:
            continue
        rels.append(r.model_copy(update={"type": rt.label}))
    return ExtractedGraph(entities=ents, relations=rels)
